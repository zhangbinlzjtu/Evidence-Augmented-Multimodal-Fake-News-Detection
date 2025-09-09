import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import logging
from PIL import Image
import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.auto_padding import AutoPaddingUtil, MixupDataAugmentation


class FakeNewsDataset(Dataset):
    """假新闻检测数据集类"""
    
    def __init__(self, data_file, llm_model_name=None, clip_processor=None, 
                 ocr_model=None, include_evidence=False, include_image=False,
                 sort_by_length=False, use_web_contents=True, ablation_mode=False):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            llm_model_name: 文本模型名称，用于初始化tokenizer
            clip_processor: CLIP图像处理器
            ocr_model: OCR模型
            include_evidence: 是否包含证据
            include_image: 是否包含图像
            sort_by_length: 是否按序列长度排序
            use_web_contents: 是否使用网络抓取的内容（额外字段）
            ablation_mode: 是否为消融实验模式（不使用额外抓取字段）
        """
        self.data_file = data_file
        self.include_evidence = include_evidence
        self.include_image = include_image
        self.sort_by_length = sort_by_length
        self.use_web_contents = use_web_contents and not ablation_mode  # 消融实验时不使用网络内容
        self.ablation_mode = ablation_mode
        
        # 加载数据
        logging.info(f"正在加载数据文件: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 转换数据格式
        self.samples = []
        for key, item in self.data.items():
            self.samples.append(item)
        
        logging.info(f"共加载 {len(self.samples)} 条数据")
        
        # 初始化tokenizer
        if llm_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        else:
            self.tokenizer = None
            
        # 设置图像处理器
        self.clip_processor = clip_processor
        
        # 设置OCR模型
        self.ocr_model = ocr_model
        
        # 如果需要按序列长度排序，预计算长度并排序
        if self.sort_by_length and self.tokenizer:
            logging.info("按序列长度排序数据...")
            self._sort_by_sequence_length()
        
    def _sort_by_sequence_length(self):
        """按文本长度排序样本"""
        lengths = []
        for sample in self.samples:
            caption = sample.get('caption', '')
            ocr_text = sample.get('img_ocr', '')
            evidence = ''
            
            # 根据配置决定是否使用网络抓取内容
            if self.include_evidence and self.use_web_contents and 'web_contents' in sample:
                evidence = sample['web_contents']
                if isinstance(evidence, list):
                    evidence = ' '.join([str(e) for e in evidence if e])
            
            # 组合文本
            if self.include_evidence and evidence:
                combined_text = f"{caption} [SEP] {evidence}"
            else:
                combined_text = caption
                
            # 获取组合文本+OCR文本的总长度
            combined_length = len(combined_text) + len(ocr_text)
            lengths.append((combined_length, sample))
        
        # 按长度排序
        lengths.sort(key=lambda x: x[0], reverse=True)
        self.samples = [item[1] for item in lengths]
        logging.info(f"数据已按序列长度排序完成，消融模式：{self.ablation_mode}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 提取标签
        label = item.get('label', 0)
        
        # 提取文本
        caption = item.get('caption', '')
        
        # 处理图像路径
        img_path = item.get('img_path', '')
        image = None
        if self.include_image and img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logging.warning(f"无法加载图像 {img_path}: {e}")
                image = None
        
        # 处理OCR文本
        ocr_text = item.get('img_ocr', '')
        
        # 处理证据 - 根据配置决定是否使用网络抓取内容
        evidence = ''
        if self.include_evidence and self.use_web_contents and 'web_contents' in item:
            evidence = item['web_contents']
            if isinstance(evidence, list):
                evidence = ' '.join([str(e) for e in evidence if e])
        
        # 组合文本
        if self.include_evidence and evidence:
            combined_text = f"{caption} [SEP] {evidence}"
        else:
            combined_text = caption
        
        # 组装结果
        result = {
            'id': item.get('id', ''),
            'text': combined_text,
            'caption': caption,
            'ocr_text': ocr_text,
            'evidence': evidence,  # 添加证据字段
            'label': torch.tensor(label, dtype=torch.long),
            'use_web_contents': self.use_web_contents,  # 添加标志
        }
        
        if image is not None:
            result['image'] = image
            
        return result
    
    @staticmethod
    def collate_fn(batch, tokenizer=None, clip_processor=None, use_auto_padding=True, max_length=512):
        """
        数据批次整理函数，支持自动填充（Auto-Padding）和Mixup数据增强
        
        Args:
            batch: 批次数据
            tokenizer: 分词器
            clip_processor: CLIP处理器
            use_auto_padding: 是否使用自动填充
            max_length: 最大序列长度
            
        Returns:
            批次字典
        """
        ids = [item['id'] for item in batch]
        texts = [item['text'] for item in batch]
        captions = [item['caption'] for item in batch]
        ocr_texts = [item['ocr_text'] for item in batch]
        evidence_texts = [item['evidence'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        use_web_contents = batch[0].get('use_web_contents', True)
        
        # 使用Auto-Padding进行tokenization
        tokenized_data = {}
        if tokenizer and use_auto_padding:
            # 对主文本进行Auto-Padding
            tokenized_texts = AutoPaddingUtil.tokenize_with_auto_padding(texts, tokenizer, max_length)
            tokenized_data.update(tokenized_texts)
            
            # 如果有OCR文本，也进行tokenization
            if any(ocr_texts):
                tokenized_ocr = AutoPaddingUtil.tokenize_with_auto_padding(ocr_texts, tokenizer, max_length)
                tokenized_data['ocr_input_ids'] = tokenized_ocr['input_ids']
                tokenized_data['ocr_attention_mask'] = tokenized_ocr['attention_mask']
        
        # 处理图像
        images = None
        if 'image' in batch[0] and batch[0]['image'] is not None:
            images = [item['image'] for item in batch if 'image' in item and item['image'] is not None]
            
            # 如果提供了CLIP处理器，处理图像
            if clip_processor and images:
                try:
                    images = clip_processor(images=images, return_tensors="pt", padding=True)
                except Exception as e:
                    logging.error(f"处理图像时出错: {e}")
                    images = None
        
        result = {
            'ids': ids,
            'texts': texts,
            'captions': captions,
            'ocr_texts': ocr_texts,
            'evidence_texts': evidence_texts,
            'images': images,
            'labels': labels,
            'use_auto_padding': use_auto_padding,
            'use_web_contents': use_web_contents,
        }
        
        # 添加tokenized数据
        result.update(tokenized_data)
        
        return result


class SortedBatchSampler:
    """按序列长度分组的批次采样器"""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        """
        初始化
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否在每个epoch打乱数据
            drop_last: 是否丢弃最后不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        
    def __iter__(self):
        # 按batch_size分组
        batches = []
        for i in range(0, self.num_samples, self.batch_size):
            if i + self.batch_size > self.num_samples and self.drop_last:
                continue
            end_idx = min(i + self.batch_size, self.num_samples)
            batches.append(list(range(i, end_idx)))
        
        # 打乱批次顺序（但保持批次内部顺序不变）
        if self.shuffle:
            np.random.shuffle(batches)
            
        # 返回索引
        for batch in batches:
            yield batch
            
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

