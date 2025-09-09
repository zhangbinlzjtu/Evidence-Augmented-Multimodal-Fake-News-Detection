from transformers import BertTokenizer, ErnieModel, BertModel, CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.auto_padding import MeanPoolingAttention, MixupDataAugmentation


class LLMCLIPModel(torch.nn.Module):
    """
    LLM+CLIP模型类
    支持文本+图片的多模态分类，使用MeanPooling-Attention机制
    """

    def __init__(self, llm_model_name, clip_model_name, num_classes=2, device=None, use_mixup=True):
        super(LLMCLIPModel, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.use_mixup = use_mixup
        
        # 初始化CLIP模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # 初始化文本模型
        if "ernie" in llm_model_name:
            self.llm_model = ErnieModel.from_pretrained(llm_model_name)
        else:
            self.llm_model = BertModel.from_pretrained(llm_model_name)

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(llm_model_name)

        # 模型配置
        self.hidden_size = self.llm_model.config.hidden_size
        self.clip_hidden_size = self.clip_model.config.projection_dim
        
        # MeanPooling-Attention机制
        self.text_attention = MeanPoolingAttention(self.hidden_size)
        
        # 多模态融合层
        self.text_projection = nn.Linear(self.hidden_size, 512)
        self.image_projection = nn.Linear(self.clip_hidden_size, 512)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),  # 文本+图像特征融合
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
        super(LLMCLIPModel, self).to(device)
        return self

    def forward(self, input_ids=None, attention_mask=None, images=None, labels=None, 
                apply_mixup=False, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: 文本token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            images: 图像数据（已经过CLIP处理器处理）
            labels: 标签（训练时需要）
            apply_mixup: 是否应用Mixup数据增强
            
        Returns:
            如果有labels: 返回损失和logits
            否则: 返回logits
        """
        batch_size = input_ids.size(0)
        
        # 1. 处理文本 - 使用LLM提取文本特征
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
        
        # 3. 处理图像 - 提取CLIP图像特征
        if images is not None:
            # images应该是已经过CLIP处理器处理的字典
            if isinstance(images, dict) and 'pixel_values' in images:
                image_features = self.clip_model.get_image_features(
                    pixel_values=images['pixel_values'].to(self.device)
                )  # [batch_size, clip_hidden_size]
            else:
                # 如果图像未处理，使用处理器处理
                processed_images = self.clip_processor(
                    images=images, 
                    return_tensors="pt", 
                    padding=True
                )
                image_features = self.clip_model.get_image_features(
                    pixel_values=processed_images['pixel_values'].to(self.device)
                )
        else:
            # 如果没有图像，创建零特征
            image_features = torch.zeros(
                batch_size, self.clip_hidden_size, 
                device=self.device
            )
        
        # 4. 投影到统一维度
        text_proj = self.text_projection(text_features)  # [batch_size, 512]
        image_proj = self.image_projection(image_features)  # [batch_size, 512]
        
        # 5. 多模态特征融合
        fused_features = torch.cat([text_proj, image_proj], dim=1)  # [batch_size, 1024]
        
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
        
    def predict(self, texts, images=None, max_length=512):
        """
        预测函数
        
        Args:
            texts: 文本列表
            images: 图像列表（可选）
            max_length: 最大序列长度
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            # 处理文本
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize文本
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 处理图像
            processed_images = None
            if images is not None:
                if not isinstance(images, list):
                    images = [images]
                processed_images = self.clip_processor(
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
            
            # 前向传播
            outputs = self.forward(
                input_ids=text_inputs['input_ids'].to(self.device),
                attention_mask=text_inputs['attention_mask'].to(self.device),
                images=processed_images,
                apply_mixup=False
            )
            
            logits = outputs['logits']
            predictions = torch.softmax(logits, dim=-1)
            
            return predictions.cpu().numpy()