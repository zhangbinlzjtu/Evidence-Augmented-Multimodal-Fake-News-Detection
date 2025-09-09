import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, CLIPProcessor
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
import logging
import os
import csv
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.text_dataset import FakeNewsDataset, SortedBatchSampler
from src.models.llm_clip_model import LLMCLIPModel
from src.models.llm_ocr_model import LLMOCRModel
from src.utils.auto_padding import AutoPaddingUtil


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="假新闻检测模型训练")

    # 数据相关参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--val_file", type=str, required=True, help="验证数据文件路径")
    parser.add_argument("--test_file", type=str, help="测试数据文件路径")
    
    # 模型相关参数
    parser.add_argument("--model_type", type=str, choices=["llm_clip", "llm_ocr"], 
                       required=True, help="模型类型")
    parser.add_argument("--llm_model_name", type=str, default="ernie-3.0-xbase-zh", 
                       required=True, help="LLM预训练模型名称")
    parser.add_argument("--clip_model_name", type=str, help="CLIP预训练模型名称")
    parser.add_argument("--num_classes", type=int, default=2, help="分类类别数")
    
    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="早停步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    
    # 数据增强和技术参数
    parser.add_argument("--use_auto_padding", action="store_true", default=True, 
                       help="是否使用Auto-Padding")
    parser.add_argument("--use_mixup", action="store_true", default=True, 
                       help="是否使用Mixup数据增强")
    parser.add_argument("--sort_by_length", action="store_true", default=True, 
                       help="是否按序列长度排序")
    parser.add_argument("--include_evidence", action="store_true", default=True,
                       help="是否包含证据（网络抓取内容）")
    parser.add_argument("--include_image", action="store_true", default=True,
                       help="是否包含图像")
    
    # 消融实验参数
    parser.add_argument("--ablation_mode", action="store_true", default=False,
                       help="消融实验模式（不使用额外抓取字段）")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./output", help="模型输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="训练设备")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="模型保存步数")

    return parser.parse_args()


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args):
    """创建模型"""
    if args.model_type == "llm_clip":
        if not args.clip_model_name:
            raise ValueError("LLM+CLIP模型需要指定clip_model_name")
        model = LLMCLIPModel(
            llm_model_name=args.llm_model_name,
            clip_model_name=args.clip_model_name,
            num_classes=args.num_classes,
            device=args.device,
            use_mixup=args.use_mixup
        )
    elif args.model_type == "llm_ocr":
        model = LLMOCRModel(
            llm_model_name=args.llm_model_name,
            num_classes=args.num_classes,
            device=args.device,
            use_mixup=args.use_mixup
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    return model


def create_data_loaders(args, tokenizer, clip_processor=None):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = FakeNewsDataset(
        data_file=args.train_file,
        llm_model_name=args.llm_model_name,
        clip_processor=clip_processor,
        include_evidence=args.include_evidence,
        include_image=args.include_image and args.model_type == "llm_clip",
        sort_by_length=args.sort_by_length,
        use_web_contents=not args.ablation_mode,
        ablation_mode=args.ablation_mode
    )
    
    val_dataset = FakeNewsDataset(
        data_file=args.val_file,
        llm_model_name=args.llm_model_name,
        clip_processor=clip_processor,
        include_evidence=args.include_evidence,
        include_image=args.include_image and args.model_type == "llm_clip",
        sort_by_length=False,  # 验证时不排序
        use_web_contents=not args.ablation_mode,
        ablation_mode=args.ablation_mode
    )
    
    # 定义collate_fn
    def collate_fn(batch):
        return FakeNewsDataset.collate_fn(
            batch, 
            tokenizer=tokenizer, 
            clip_processor=clip_processor, 
            use_auto_padding=args.use_auto_padding,
            max_length=args.max_length
        )
    
    # 创建数据加载器
    if args.sort_by_length:
        train_sampler = SortedBatchSampler(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader


def evaluate_model(model, data_loader, device, model_type):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            # 准备输入
            labels = batch['labels'].to(device)
            
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
            
            # 收集预测和标签
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro
    }


def save_csv_log(log_file, epoch, train_metrics, val_metrics, lr):
    """保存CSV日志"""
    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # 写入表头
            writer.writerow([
                'epoch', 'train_loss', 'train_f1_macro', 'val_loss', 
                'val_f1_macro', 'val_accuracy', 'learning_rate'
            ])
        
        # 写入数据
        writer.writerow([
            epoch,
            round(train_metrics['loss'], 4),
            round(train_metrics['f1_macro'], 4),
            round(val_metrics['loss'], 4),
            round(val_metrics['f1_macro'], 4),
            round(val_metrics['accuracy'], 4),
            f"{lr:.2e}"
        ])


def train_model(args):
    """训练模型"""
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置日志
    setup_logging(args.log_dir)
    logging.info(f"训练参数: {args}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型
    model = create_model(args)
    logging.info(f"创建{args.model_type}模型完成")
    
    # 创建tokenizer和处理器
    tokenizer = model.tokenizer
    clip_processor = getattr(model, 'clip_processor', None)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(args, tokenizer, clip_processor)
    logging.info(f"训练集大小: {len(train_loader)}, 验证集大小: {len(val_loader)}")
    
    # 创建优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_f1 = 0
    early_stopping_counter = 0
    csv_log_file = os.path.join(args.log_dir, f'{args.model_type}_training_log.csv')
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # 准备输入
            labels = batch['labels'].to(args.device)
            
            # 根据模型类型准备不同的输入
            if args.model_type == "llm_clip":
                outputs = model(
                    input_ids=batch['input_ids'].to(args.device),
                    attention_mask=batch['attention_mask'].to(args.device),
                    images=batch.get('images'),
                    labels=labels,
                    apply_mixup=args.use_mixup
                )
            elif args.model_type == "llm_ocr":
                outputs = model(
                    input_ids=batch['input_ids'].to(args.device),
                    attention_mask=batch['attention_mask'].to(args.device),
                    ocr_input_ids=batch.get('ocr_input_ids', torch.empty(0)).to(args.device),
                    ocr_attention_mask=batch.get('ocr_attention_mask', torch.empty(0)).to(args.device),
                    labels=labels,
                    apply_mixup=args.use_mixup
                )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 收集预测用于计算指标（仅在非Mixup情况下）
            if not args.use_mixup or not model.training:
                predictions = torch.argmax(logits, dim=-1)
                train_predictions.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 记录步骤日志
            if (step + 1) % args.logging_steps == 0:
                logging.info(f"Step {step + 1}, Loss: {loss.item():.4f}")
        
        # 计算训练指标
        avg_train_loss = total_train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        if train_predictions and train_labels:  # 只有非Mixup时才计算
            train_f1_macro = precision_recall_fscore_support(
                train_labels, train_predictions, average='macro'
            )[2]
        else:
            train_f1_macro = 0.0  # Mixup时设为0
        
        train_metrics = {
            'loss': avg_train_loss,
            'f1_macro': train_f1_macro
        }
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, args.device, args.model_type)
        
        # 记录日志
        logging.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1_macro:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, LR: {current_lr:.2e}"
        )
        
        # 保存CSV日志
        save_csv_log(csv_log_file, epoch + 1, train_metrics, val_metrics, current_lr)
        
        # 早停和模型保存
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            early_stopping_counter = 0
            
            # 保存最佳模型
            model_save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_f1': best_f1,
                'args': args
            }, model_save_path)
            
            logging.info(f"保存最佳模型，F1: {best_f1:.4f}")
        else:
            early_stopping_counter += 1
            
        # 早停检查
        if early_stopping_counter >= args.early_stopping_patience:
            logging.info(f"早停触发，最佳F1: {best_f1:.4f}")
            break
    
    logging.info("训练完成")
    return model


def main():
    args = parse_args()
    
    # 参数校验
    if args.model_type == "llm_clip" and not args.clip_model_name:
        raise ValueError("LLM+CLIP模型需要指定clip_model_name参数")
    
    # 训练模型
    model = train_model(args)
    
    return model


if __name__ == "__main__":
    main()