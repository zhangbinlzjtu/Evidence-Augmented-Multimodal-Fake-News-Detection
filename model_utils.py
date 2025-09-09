import os
import torch
import logging
from models.llm_clip_model import LLMCLIPModel
from models.llm_ocr_model import LLMOCRModel
from transformers import CLIPProcessor


def save_model(model, output_dir, model_name="model.pt", save_processor=True):
    """
    保存模型和相关处理器
    
    Args:
        model: 模型对象
        output_dir: 输出目录
        model_name: 模型文件名
        save_processor: 是否保存处理器
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型状态
    model_path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存到: {model_path}")
    
    # 保存模型配置
    if hasattr(model, 'config'):
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(model.config.to_json_string())
        logging.info(f"模型配置已保存到: {config_path}")
    
    # 保存处理器（如果有）
    if save_processor:
        if isinstance(model, LLMCLIPModel) and model.clip_processor:
            processor_dir = os.path.join(output_dir, "processor")
            model.clip_processor.save_pretrained(processor_dir)
            logging.info(f"CLIP处理器已保存到: {processor_dir}")


def load_model(model_path, model_type, llm_model_name=None, clip_model_name=None, 
               ocr_model_name=None, device=None, num_classes=2):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，'llm_clip' 或 'llm_ocr'
        llm_model_name: LLM模型名称
        clip_model_name: CLIP模型名称
        ocr_model_name: OCR模型名称
        device: 设备
        num_classes: 分类类别数
        
    Returns:
        加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 确定设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据类型创建模型
    if model_type == "llm_clip":
        if not clip_model_name:
            raise ValueError("加载LLM+CLIP模型需要指定clip_model_name")
        
        # 加载CLIP处理器
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 创建模型
        model = LLMCLIPModel(
            llm_model_name, 
            clip_model_name,
            num_classes=num_classes,
            device=device
        )
    elif model_type == "llm_ocr":
        
        # 创建模型
        model = LLMOCRModel(
            llm_model_name,
            num_classes=num_classes,
            device=device
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型状态
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"模型已从 {model_path} 加载")
    
    # 将模型设置为评估模式
    model.eval()
    return model


def get_model_info(model):
    """
    获取模型信息
    
    Args:
        model: 模型对象
        
    Returns:
        模型信息字典
    """
    model_info = {
        "model_type": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # 添加LLM模型信息
    if hasattr(model, 'llm_model') and hasattr(model.llm_model, 'config'):
        model_info["llm_model_name"] = model.llm_model.config.name_or_path
        model_info["llm_hidden_size"] = model.llm_model.config.hidden_size
        
    # 添加CLIP模型信息（如果有）
    if hasattr(model, 'clip_model') and hasattr(model.clip_model, 'config'):
        model_info["clip_model_name"] = model.clip_model.config.name_or_path
        model_info["clip_projection_dim"] = model.clip_model.config.projection_dim
    
    return model_info 