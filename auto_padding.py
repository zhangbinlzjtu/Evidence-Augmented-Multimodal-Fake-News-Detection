import torch
from typing import List, Dict, Any, Tuple
import torch.nn.functional as F


class AutoPaddingUtil:
    """Auto-Padding工具类，实现动态批次填充"""
    
    @staticmethod
    def pad_sequences(sequences: List[torch.Tensor], pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对序列进行Auto-Padding填充
        
        Args:
            sequences: 待填充的序列列表，每个元素为tensor
            pad_token_id: 填充token的ID
            
        Returns:
            padded_sequences: 填充后的tensor，shape为[batch_size, max_len]
            attention_masks: 注意力掩码，shape为[batch_size, max_len]
        """
        # 步骤1：测量序列长度
        seq_lengths = [len(seq) for seq in sequences]
        
        # 步骤2：设置填充长度为批次内最长序列的长度
        max_len_in_batch = max(seq_lengths)
        
        # 步骤3：将所有序列填充到统一长度
        batch_size = len(sequences)
        padded_sequences = torch.full((batch_size, max_len_in_batch), pad_token_id, dtype=sequences[0].dtype)
        attention_masks = torch.zeros(batch_size, max_len_in_batch, dtype=torch.bool)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            padded_sequences[i, :seq_len] = seq
            attention_masks[i, :seq_len] = True
            
        return padded_sequences, attention_masks
    
    @staticmethod
    def tokenize_with_auto_padding(texts: List[str], tokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        使用Auto-Padding对文本进行tokenize
        
        Args:
            texts: 文本列表
            tokenizer: tokenizer对象
            max_length: 最大序列长度限制
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        # 先对每个文本进行tokenize（不进行填充）
        tokenized_texts = []
        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,  # 不进行填充
                return_tensors="pt"
            )
            tokenized_texts.append(tokens['input_ids'].squeeze(0))
        
        # 使用Auto-Padding进行填充
        padded_input_ids, attention_masks = AutoPaddingUtil.pad_sequences(
            tokenized_texts, 
            pad_token_id=tokenizer.pad_token_id
        )
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_masks
        }


class MixupDataAugmentation:
    """Mixup数据增强类"""
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化Mixup增强
        
        Args:
            alpha: Beta分布的参数，控制混合强度
        """
        self.alpha = alpha
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        执行Mixup数据增强
        
        Args:
            x: 输入数据 [batch_size, ...]
            y: 标签数据 [batch_size]
            alpha: Beta分布参数，如果为None则使用初始化时的值
            
        Returns:
            mixed_x: 混合后的输入数据
            y_a: 原始标签
            y_b: 混合标签
            lam: 混合比例
        """
        if alpha is None:
            alpha = self.alpha
            
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample()
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_loss(self, criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        计算Mixup损失
        
        Args:
            criterion: 损失函数
            pred: 模型预测结果
            y_a: 原始标签
            y_b: 混合标签
            lam: 混合比例
            
        Returns:
            混合后的损失值
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MeanPoolingAttention(torch.nn.Module):
    """
    MeanPooling-Attention机制实现
    结合了mean pooling和attention mechanism
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8):
        """
        初始化MeanPooling-Attention
        
        Args:
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
        """
        super(MeanPoolingAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 注意力计算层
        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        
        # 输出层
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        
        # Mean pooling层
        self.mean_pooling = torch.nn.AdaptiveAvgPool1d(1)
        
        # 最终融合层
        self.fusion_layer = torch.nn.Linear(hidden_size * 2, hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重新排列tensor以便计算attention scores"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            融合后的表示 [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 步骤1: 编码、表示和分类序列 - 计算attention
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 步骤2: 计算attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        
        # 应用attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -10000.0)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 步骤3: attention-weighted pooling
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        # 对attention output进行mean pooling
        attention_pooled = attention_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Mean pooling分支
        # 转换为 [batch_size, hidden_size, seq_len] 以便应用AdaptiveAvgPool1d
        hidden_transposed = hidden_states.transpose(1, 2)
        mean_pooled = self.mean_pooling(hidden_transposed).squeeze(-1)  # [batch_size, hidden_size]
        
        # 步骤4: 融合attention pooling和mean pooling的结果
        combined = torch.cat([attention_pooled, mean_pooled], dim=1)  # [batch_size, hidden_size * 2]
        fused_output = self.fusion_layer(combined)  # [batch_size, hidden_size]
        
        return fused_output