from transformers import BertTokenizer, ErnieModel, BertModel
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.auto_padding import MeanPoolingAttention, MixupDataAugmentation


class LLMOCRModel(torch.nn.Module):
    """
    LLM+OCR模型类
    支持文本+OCR的多模态分类，使用MeanPooling-Attention机制
    OCR数据在预处理阶段完成，加快模型训练速度
    """
    
    def __init__(self, llm_model_name, num_classes=2, device=None, use_mixup=True):
        super(LLMOCRModel, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.use_mixup = use_mixup

        # 初始化文本模型
        if "ernie" in llm_model_name:
            self.llm_model = ErnieModel.from_pretrained(llm_model_name)
        else:
            try:
                self.llm_model = BertModel.from_pretrained(llm_model_name)
            except Exception as e:
                raise ValueError(f"加载文本模型失败: {e}")
            
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(llm_model_name)
        
        # 模型配置
        self.hidden_size = self.llm_model.config.hidden_size
        
        # MeanPooling-Attention机制
        self.text_attention = MeanPoolingAttention(self.hidden_size)
        self.ocr_attention = MeanPoolingAttention(self.hidden_size)
        
        # 多模态融合层
        self.text_projection = nn.Linear(self.hidden_size, 512)
        self.ocr_projection = nn.Linear(self.hidden_size, 512)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),  # 文本+OCR特征融合
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Mixup数据增强
        if self.use_mixup:
            self.mixup = MixupDataAugmentation(alpha=0.2)
        
        # 将模型移至指定设备
        self.to(self.device)

    def to(self, device):
        """将模型移至指定设备"""
        self.device = device
        super(LLMOCRModel, self).to(device)
        return self

    def forward(self, input_ids=None, attention_mask=None, ocr_input_ids=None, 
                ocr_attention_mask=None, labels=None, apply_mixup=False, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: 文本token ids [batch_size, seq_len]
            attention_mask: 文本注意力掩码 [batch_size, seq_len]
            ocr_input_ids: OCR文本token ids [batch_size, ocr_seq_len]
            ocr_attention_mask: OCR注意力掩码 [batch_size, ocr_seq_len]
            labels: 标签（训练时需要）
            apply_mixup: 是否应用Mixup数据增强
            
        Returns:
            如果有labels: 返回损失和logits
            否则: 返回logits
        """
        batch_size = input_ids.size(0)
        
        # 1. 处理主文本 - 使用LLM提取文本特征
        text_outputs = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 2. 使用MeanPooling-Attention机制提取文本特征
        text_features = self.text_attention(
            text_outputs.last_hidden_state, 
            attention_mask
        )  # [batch_size, hidden_size]
        
        # 3. 处理OCR文本
        if ocr_input_ids is not None and ocr_input_ids.numel() > 0:
            # OCR文本存在且不为空
            ocr_outputs = self.llm_model(
                input_ids=ocr_input_ids,
                attention_mask=ocr_attention_mask,
                return_dict=True
            )
            
            # 使用MeanPooling-Attention机制提取OCR特征
            ocr_features = self.ocr_attention(
                ocr_outputs.last_hidden_state,
                ocr_attention_mask
            )  # [batch_size, hidden_size]
        else:
            # 如果没有OCR文本，创建零特征
            ocr_features = torch.zeros(
                batch_size, self.hidden_size, 
                device=self.device
            )
        
        # 4. 投影到统一维度
        text_proj = self.text_projection(text_features)  # [batch_size, 512]
        ocr_proj = self.ocr_projection(ocr_features)  # [batch_size, 512]
        
        # 5. 多模态特征融合
        fused_features = torch.cat([text_proj, ocr_proj], dim=1)  # [batch_size, 1024]
        
        # 6. 应用Mixup数据增强（仅在训练时）
        mixed_features = fused_features
        mixed_labels = labels
        mixup_lambda = 1.0
        
        if apply_mixup and self.training and self.use_mixup and labels is not None:
            mixed_features, labels_a, labels_b, mixup_lambda = self.mixup.mixup_data(
                fused_features, labels
            )
            # 保存原始标签用于损失计算
            mixed_labels = (labels_a, labels_b)
        
        # 7. 分类
        logits = self.classifier(mixed_features)  # [batch_size, num_classes]
        
        # 8. 计算损失（如果提供了标签）
        if labels is not None:
            if apply_mixup and self.training and self.use_mixup and isinstance(mixed_labels, tuple):
                # Mixup损失
                labels_a, labels_b = mixed_labels
                loss = self.mixup.mixup_loss(
                    F.cross_entropy, logits, labels_a, labels_b, mixup_lambda
                )
            else:
                # 标准损失
                loss = F.cross_entropy(logits, labels)
            
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

    def compute_loss(self, logits, labels):
        """计算损失"""
        return F.cross_entropy(logits, labels)
        
    def predict(self, texts, ocr_texts=None, max_length=512):
        """
        预测函数
        
        Args:
            texts: 文本列表
            ocr_texts: OCR文本列表（可选）
            max_length: 最大序列长度
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            # 处理文本
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize主文本
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 处理OCR文本
            ocr_inputs = None
            if ocr_texts is not None:
                if isinstance(ocr_texts, str):
                    ocr_texts = [ocr_texts]
                    
                # 确保OCR文本列表长度与主文本一致
                while len(ocr_texts) < len(texts):
                    ocr_texts.append("")
                
                ocr_inputs = self.tokenizer(
                    ocr_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
            
            # 前向传播
            forward_kwargs = {
                'input_ids': text_inputs['input_ids'].to(self.device),
                'attention_mask': text_inputs['attention_mask'].to(self.device),
                'apply_mixup': False
            }
            
            if ocr_inputs is not None:
                forward_kwargs.update({
                    'ocr_input_ids': ocr_inputs['input_ids'].to(self.device),
                    'ocr_attention_mask': ocr_inputs['attention_mask'].to(self.device)
                })
            
            outputs = self.forward(**forward_kwargs)
            
            logits = outputs['logits']
            predictions = torch.softmax(logits, dim=-1)
            
            return predictions.cpu().numpy()