#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理脚本：将原始数据中的标签信息合并到处理后的数据中
用于解决data_prepare.py没有保存label字段的问题
"""

import json
import argparse
import logging
from typing import Dict, Any


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"成功加载文件 {file_path}，包含 {len(data)} 条记录")
        return data
    except Exception as e:
        logging.error(f"加载文件 {file_path} 失败: {str(e)}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str):
    """保存JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"成功保存文件 {file_path}，包含 {len(data)} 条记录")
    except Exception as e:
        logging.error(f"保存文件 {file_path} 失败: {str(e)}")
        raise


def merge_labels(original_data: Dict[str, Any], processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始数据中的标签信息合并到处理后的数据中
    
    Args:
        original_data: 原始数据，包含label字段
        processed_data: 处理后的数据，缺少label字段
        
    Returns:
        合并后的数据
    """
    merged_data = {}
    matched_count = 0
    missing_in_processed = []
    missing_in_original = []
    
    # 遍历处理后的数据，添加标签
    for item_id, processed_item in processed_data.items():
        if item_id in original_data:
            # 复制处理后的数据
            merged_item = processed_item.copy()
            
            # 添加原始数据中的label字段
            original_item = original_data[item_id]
            merged_item['label'] = original_item.get('label', 0)  # 默认标签为0
            
            # 可选：也保存其他可能有用的原始字段
            if 'caption' not in merged_item or not merged_item['caption']:
                merged_item['caption'] = original_item.get('caption', '')
            if 'image_path' not in merged_item or not merged_item['image_path']:
                merged_item['img_path'] = original_item.get('image_path', '')
                
            merged_data[item_id] = merged_item
            matched_count += 1
        else:
            logging.warning(f"处理后的数据中ID {item_id} 在原始数据中未找到")
            missing_in_original.append(item_id)
            # 仍然保留该条数据，但设置默认标签
            merged_item = processed_item.copy()
            merged_item['label'] = 0  # 默认标签
            merged_data[item_id] = merged_item
    
    # 检查原始数据中是否有遗漏的ID
    for item_id in original_data:
        if item_id not in processed_data:
            missing_in_processed.append(item_id)
    
    # 输出统计信息
    logging.info(f"合并完成统计:")
    logging.info(f"  - 成功匹配: {matched_count} 条")
    logging.info(f"  - 处理后数据中缺失的原始ID: {len(missing_in_original)} 条")
    logging.info(f"  - 原始数据中未被处理的ID: {len(missing_in_processed)} 条")
    
    if missing_in_original:
        logging.warning(f"缺失的原始ID（前10个）: {missing_in_original[:10]}")
    
    if missing_in_processed:
        logging.warning(f"未处理的原始ID（前10个）: {missing_in_processed[:10]}")
    
    return merged_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='后处理脚本：添加标签到处理后的数据')
    parser.add_argument('--original_file', required=True, help='原始数据文件路径（包含标签）')
    parser.add_argument('--processed_file', required=True, help='处理后的数据文件路径（缺少标签）')
    parser.add_argument('--output_file', required=True, help='输出文件路径（合并后的数据）')
    parser.add_argument('--backup', action='store_true', default=True, 
                       help='是否备份原处理文件（默认：True）')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    logging.info("开始标签合并后处理...")
    logging.info(f"原始文件: {args.original_file}")
    logging.info(f"处理后文件: {args.processed_file}")
    logging.info(f"输出文件: {args.output_file}")
    
    try:
        # 加载数据
        original_data = load_json_file(args.original_file)
        processed_data = load_json_file(args.processed_file)
        
        # 备份原文件
        if args.backup:
            backup_file = args.processed_file + '.backup'
            logging.info(f"备份原处理文件到: {backup_file}")
            with open(args.processed_file, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        
        # 合并标签
        merged_data = merge_labels(original_data, processed_data)
        
        # 保存结果
        save_json_file(merged_data, args.output_file)
        
        # 验证结果
        logging.info("验证合并结果...")
        sample_id = list(merged_data.keys())[0]
        sample_item = merged_data[sample_id]
        
        logging.info(f"样本数据字段: {list(sample_item.keys())}")
        if 'label' in sample_item:
            logging.info(f"标签字段存在，样本标签值: {sample_item['label']}")
        else:
            logging.error("合并后的数据中仍然缺少label字段！")
            return False
        
        logging.info("标签合并后处理完成！")
        return True
        
    except Exception as e:
        logging.error(f"后处理失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)