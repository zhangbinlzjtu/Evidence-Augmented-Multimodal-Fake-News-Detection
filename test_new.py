import argparse
import torch
import json
import csv
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.text_dataset import FakeNewsDataset
from src.models.llm_clip_model import LLMCLIPModel  
from src.models.llm_ocr_model import LLMOCRModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="假新闻检测模型测试")
    
    # 数据和模型参数
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--model_type", type=str, choices=["llm_clip", "llm_ocr"], 
                       required=True, help="模型类型")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="结果输出目录")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="推理设备")
    
    # 消融实验参数
    parser.add_argument("--ablation_mode", action="store_true", default=False,
                       help="消融实验模式（不使用额外抓取字段）")
    
    return parser.parse_args()


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    
    # 根据模型类型创建模型
    if args.model_type == "llm_clip":
        model = LLMCLIPModel(
            llm_model_name=args.llm_model_name,
            clip_model_name=args.clip_model_name,
            num_classes=args.num_classes,
            device=device,
            use_mixup=False  # 测试时不使用mixup
        )
    elif args.model_type == "llm_ocr":
        model = LLMOCRModel(
            llm_model_name=args.llm_model_name,
            num_classes=args.num_classes,
            device=device,
            use_mixup=False
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"成功加载模型，最佳F1: {checkpoint.get('best_f1', 'Unknown')}")
    
    return model, args


def create_test_loader(test_file, model, train_args, ablation_mode, batch_size, max_length):
    """创建测试数据加载器"""
    tokenizer = model.tokenizer
    clip_processor = getattr(model, 'clip_processor', None)
    
    # 创建测试数据集
    test_dataset = FakeNewsDataset(
        data_file=test_file,
        llm_model_name=train_args.llm_model_name,
        clip_processor=clip_processor,
        include_evidence=train_args.include_evidence,
        include_image=train_args.include_image and train_args.model_type == "llm_clip",
        sort_by_length=False,  # 测试时不排序
        use_web_contents=not ablation_mode,
        ablation_mode=ablation_mode
    )
    
    # 定义collate_fn
    def collate_fn(batch):
        return FakeNewsDataset.collate_fn(
            batch, 
            tokenizer=tokenizer, 
            clip_processor=clip_processor, 
            use_auto_padding=train_args.use_auto_padding,
            max_length=max_length
        )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return test_loader


def test_model(model, test_loader, device, model_type):
    """测试模型并返回详细结果"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_ids = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            # 准备输入
            labels = batch['labels'].to(device)
            ids = batch['ids']
            
            # 根据模型类型准备不同的输入
            if model_type == "llm_clip":
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    images=batch.get('images'),
                    labels=labels,
                    apply_mixup=False
                )
            elif model_type == "llm_ocr":
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    ocr_input_ids=batch.get('ocr_input_ids', torch.empty(0)).to(device),
                    ocr_attention_mask=batch.get('ocr_attention_mask', torch.empty(0)).to(device),
                    labels=labels,
                    apply_mixup=False
                )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # 计算概率和预测
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'labels': np.array(all_labels),
        'ids': all_ids,
        'loss': avg_loss
    }


def calculate_metrics(labels, predictions, probabilities):
    """计算各种评估指标"""
    # 基本指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro'
    )
    
    # 分类报告
    class_report = classification_report(
        labels, predictions, 
        target_names=['真实', '虚假'], 
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    # AUC相关指标（二分类）
    if probabilities.shape[1] == 2:
        auc_score = roc_auc_score(labels, probabilities[:, 1])
    else:
        auc_score = None
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'auc_score': auc_score
    }


def save_results(results, metrics, output_dir, model_type, ablation_mode):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细预测结果
    result_suffix = "_ablation" if ablation_mode else ""
    predictions_file = os.path.join(output_dir, f'{model_type}_predictions{result_suffix}.csv')
    
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'true_label', 'predicted_label', 'prob_fake', 'prob_real'])
        
        for i, id_val in enumerate(results['ids']):
            writer.writerow([
                id_val,
                results['labels'][i],
                results['predictions'][i],
                results['probabilities'][i, 1] if results['probabilities'].shape[1] > 1 else 0,
                results['probabilities'][i, 0]
            ])
    
    # 保存指标摘要
    metrics_file = os.path.join(output_dir, f'{model_type}_metrics{result_suffix}.json')
    
    # 转换numpy类型为Python原生类型以支持JSON序列化
    metrics_for_json = {}
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            metrics_for_json[key] = value.tolist()
        elif key == 'classification_report':
            metrics_for_json[key] = value
        elif isinstance(value, np.ndarray):
            metrics_for_json[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            metrics_for_json[key] = float(value)
        else:
            metrics_for_json[key] = value
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_for_json, f, ensure_ascii=False, indent=2)
    
    # 保存混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, fmt='d', 
                cmap='Blues',
                xticklabels=['真实', '虚假'],
                yticklabels=['真实', '虚假'])
    plt.title(f'{model_type.upper()} 混淆矩阵{"（消融实验）" if ablation_mode else ""}')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_confusion_matrix{result_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 如果是二分类，绘制ROC和PR曲线
    if results['probabilities'].shape[1] == 2:
        # ROC曲线
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'][:, 1])
        
        plt.figure(figsize=(12, 5))
        
        # ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_score"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend()
        
        # PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(
            results['labels'], results['probabilities'][:, 1]
        )
        
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确度')
        plt.title('精确度-召回率曲线')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_roc_pr_curves{result_suffix}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"结果已保存到 {output_dir}")


def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 加载模型
    model, train_args = load_model(args.model_path, args.device)
    
    # 创建测试数据加载器
    test_loader = create_test_loader(
        args.test_file, model, train_args, args.ablation_mode,
        args.batch_size, args.max_length
    )
    
    logging.info(f"测试集大小: {len(test_loader)}")
    
    # 测试模型
    results = test_model(model, test_loader, args.device, args.model_type)
    
    # 计算指标
    metrics = calculate_metrics(
        results['labels'], 
        results['predictions'], 
        results['probabilities']
    )
    
    # 打印结果
    logging.info("=" * 50)
    logging.info("测试结果:")
    logging.info(f"准确率: {metrics['accuracy']:.4f}")
    logging.info(f"精确度 (Macro): {metrics['precision_macro']:.4f}")
    logging.info(f"召回率 (Macro): {metrics['recall_macro']:.4f}")
    logging.info(f"F1分数 (Macro): {metrics['f1_macro']:.4f}")
    if metrics['auc_score'] is not None:
        logging.info(f"AUC: {metrics['auc_score']:.4f}")
    logging.info(f"测试损失: {results['loss']:.4f}")
    
    # 详细分类报告
    logging.info("\n详细分类报告:")
    for class_name, class_metrics in metrics['classification_report'].items():
        if isinstance(class_metrics, dict):
            logging.info(f"{class_name}: Precision={class_metrics['precision']:.3f}, "
                        f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
    
    # 保存结果
    save_results(results, metrics, args.output_dir, args.model_type, args.ablation_mode)
    
    return metrics


if __name__ == "__main__":
    main()