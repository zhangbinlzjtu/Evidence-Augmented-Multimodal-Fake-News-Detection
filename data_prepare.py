import os
import json
import jieba
import jieba.analyse
import re
import numpy as np
from PIL import Image
import cv2
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
from tqdm import tqdm
from summa import keywords as summa_keywords
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pp_onnx.onnx_paddleocr import ONNXPaddleOcr,draw_ocr

# 代理配置
PROXY_CONFIG = {
    'http': 'http://brd-customer-hl_64cad720-zone-datacenter_proxy1-country-us:hy6p3mo4aso1@brd.superproxy.io:33335',
    'https': 'http://brd-customer-hl_64cad720-zone-datacenter_proxy1-country-us:hy6p3mo4aso1@brd.superproxy.io:33335'
}

# 下载nltk资源
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class DataProcessor:
    def __init__(self, input_file: str, output_file: str, img_base_dir: Optional[str] = None) -> None:
        """
        初始化数据处理器
        
        参数:
            input_file: JSON输入文件路径
            output_file: 输出文件路径
            img_base_dir: 图片基础目录，如果为None则使用相对路径
        """
        self.input_file: str = input_file
        self.output_file: str = output_file
        self.img_base_dir: Optional[str] = img_base_dir
        self.ocr_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=True)
        # 加载中文停用词
        stop_words_path = os.path.join(os.path.dirname(__file__), 'stop_words_baidu.txt')
        if not os.path.exists(stop_words_path):
            self.download_stopwords(stop_words_path)
        
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            self.cn_stopwords: Set[str] = set([line.strip() for line in f])
        
        # 英文停用词
        self.en_stopwords: Set[str] = set(stopwords.words('english'))
        
    
    def download_stopwords(self, path: str) -> None:
        """下载中文停用词表"""
        url = "https://github.com/baipengyan/Chinese-StopWords/raw/refs/heads/master/%E7%99%BE%E5%BA%A6%E5%81%9C%E7%94%A8%E8%AF%8D%E8%A1%A8.txt"
        try:
            r = requests.get(url)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(r.text)
        except:
            # 如果下载失败，创建一个空文件
            with open(path, 'w', encoding='utf-8') as f:
                pass
            print("警告: 无法下载停用词表，使用空停用词表")
    
    def load_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_text(self, text: str) -> Tuple[List[str], str]:
        """
        处理文本：分词、去停用词并提取关键词
        
        参数:
            text: 待处理的文本
            
        返回:
            keywords: 提取的关键词列表
            processed_text: 处理后的文本
        """
        # 检测语言
        is_chinese = self.is_chinese(text)
        
        if is_chinese:
            # 使用结巴分词处理中文
            words = jieba.cut(text)
            words = [w for w in words if w not in self.cn_stopwords and len(w.strip()) > 0]
            processed_text = " ".join(words)
            
            # 使用jieba的TextRank算法提取关键词
            keywords = jieba.analyse.textrank(
                processed_text, 
                topK=5, 
                withWeight=False, 
                allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'PER','LOC','ORG', 'TIME','nt')  # 只保留地名、名词、动名词和动词
            )
        else:
            # 使用NLTK处理英文
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in self.en_stopwords and len(w.strip()) > 0]
            processed_text = " ".join(words)
            
            # 使用TextRank算法提取关键词
            text_for_keywords = " ".join(words)
            extracted_keywords = summa_keywords.keywords(text_for_keywords, scores=False, ratio=0.2)
            keywords = extracted_keywords.split('\n')[:10] if extracted_keywords else words[:10]
        
        return keywords, processed_text
    
    def is_chinese(self, text: str) -> bool:
        """判断文本主要是否为中文"""
        # 统计中文字符数量
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) > len(text) * 0.5
    
    def perform_ocr(self, image_path: str) -> str:
        """
        对图片进行OCR识别
        
        参数:
            image_path: 图片路径
            
        返回:
            ocr_text: OCR识别的文本
        """
        try:
            # 如果提供了基础目录，则构建完整路径
            if self.img_base_dir:
                full_path = os.path.join(self.img_base_dir, image_path)
            else:
                full_path = image_path
                
            # 读取图片
            if not os.path.exists(full_path):
                print(f"图片 {full_path} 不存在")
                return ""
            image = cv2.imread(full_path)
            
            # 进行OCR识别
            result = self.ocr_model.ocr(image)
            result = result[0]
            txts = [line[1][0] for line in result]
            ocr_text = "\n".join(txts)
            
            return ocr_text.strip()
        except Exception as e:
            print(f"OCR处理图片 {image_path} 时出错: {str(e)}")
            return ""
    
    def search_web(self, keywords: List[str], num_results: int = 3) -> List[str]:
        """
        使用DuckDuckGo搜索关键词并抓取网页内容
        
        参数:
            keywords: 要搜索的关键词列表
            num_results: 要返回的结果数量
            
        返回:
            results: 包含网页内容的列表
        """
        # 将关键词组合成搜索词
        search_query = " ".join(keywords[:-1])  # 不包含caption的关键词
        caption_query = keywords[-1]
        results = []
        
        try:
            # 进行搜索
            with DDGS(proxies=PROXY_CONFIG, verify=False) as ddgs:
                search_results = list(ddgs.text(search_query, max_results=num_results))
                caption_results = list(ddgs.text(caption_query, max_results=num_results))
                final_results = list(set(search_results + caption_results))
            # 抓取每个搜索结果的网页内容
            for result in final_results:
                try:
                    url = result['href']
                    response = requests.get(url, timeout=10, proxies=PROXY_CONFIG, verify=False)
                    
                    # 使用BeautifulSoup解析HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 删除脚本标签
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # 获取文本
                    text = soup.get_text()
                    
                    # 清理文本
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # 截断长文本
                    text = text[:5000] if len(text) > 5000 else text
                    
                    results.append(result.get('title', '') + '[SEP]' + text)
                except Exception as e:
                    print(f"抓取网页 {result.get('href', 'unknown')} 时出错: {str(e)}")
            
            return results
        except Exception as e:
            print(f"搜索关键词时出错: {str(e)}")
            return []
    
    def save_results(self, results: Dict[str, Any], is_final: bool = False) -> None:
        """
        保存处理结果
        
        参数:
            results: 处理结果字典
            is_final: 是否为最终保存
        """
        # 创建临时文件路径
        temp_file = self.output_file + '.tmp'
        
        try:
            # 保存到临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 如果是最终保存，则将临时文件重命名为目标文件
            if is_final:
                if os.path.exists(self.output_file):
                    os.remove(self.output_file)
                os.rename(temp_file, self.output_file)
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def process_data(self) -> Dict[str, Any]:
        """处理数据并生成结果"""
        # 加载数据
        data = self.load_data()
        results = {}
        
        # 尝试加载已有的处理结果
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"已加载 {len(results)} 条已处理的数据")
            except Exception as e:
                print(f"加载已有结果时出错: {str(e)}")
        
        # 获取未处理的数据ID
        processed_ids = set(results.keys())
        remaining_ids = [id_str for id_str in data.keys() if id_str not in processed_ids]
        
        for id_str in tqdm(remaining_ids, desc="处理数据"):
            try:
                item = data[id_str]
                # 1. 处理caption
                caption = item.get('caption', '')
                caption_keywords, processed_caption = self.process_text(caption)
                
                # 2. 进行OCR识别
                image_path = item.get('image_path', '')
                ocr_text = self.perform_ocr(image_path)
                ocr_keywords, processed_ocr = self.process_text(ocr_text)
                
                # 3. 合并关键词
                all_keywords = list(set(caption_keywords + ocr_keywords))
                all_keywords.append(caption)
                
                # 4. 网页搜索
                web_results = self.search_web(all_keywords)
                
                # 5. 保存结果
                results[id_str] = {
                    'id': id_str,
                    'caption': caption,
                    'processed_caption': processed_caption,
                    'caption_keywords': caption_keywords,
                    'img_path': image_path,
                    'img_ocr': ocr_text,
                    'processed_ocr': processed_ocr,
                    'ocr_keywords': ocr_keywords,
                    'web_contents': web_results,
                    'label': item.get('label', 0)  # 添加标签字段，默认为0
                }
                
                # 每处理完一条数据就保存一次
                self.save_results(results)
                
            except Exception as e:
                print(f"处理ID {id_str} 时出错: {str(e)}")
                # 即使出错也保存当前结果
                self.save_results(results)
        
        # 最终保存
        self.save_results(results, is_final=True)
        print(f"处理完成，结果保存至: {self.output_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--input', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--img_dir', help='图片基础目录，默认为None')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.input, args.output, args.img_dir)
    processor.process_data()

if __name__ == "__main__":
    main()
