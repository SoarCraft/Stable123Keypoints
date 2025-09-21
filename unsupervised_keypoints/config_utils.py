"""
配置管理工具模块
提供基于OmegaConf的配置加载和管理功能
"""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig
import torch


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    从YAML文件加载配置，并可选择性地进行覆盖
    
    Args:
        config_path: 配置文件路径
        overrides: 要覆盖的配置字典
        
    Returns:
        OmegaConf配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        raise ValueError(f"配置文件格式错误: {e}")
    
    # 应用覆盖
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config


def validate_config(config: DictConfig) -> None:
    """
    验证配置的必要字段和合法性
    
    Args:
        config: 要验证的配置对象
        
    Raises:
        ValueError: 配置验证失败
    """
    # 验证必需的字段
    required_fields = [
        "model.my_token",
        "dataset.name", 
        "training.device",
        "training.lr",
        "training.num_steps"
    ]
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            if field == "model.my_token":
                raise ValueError(f"必须提供Hugging Face token: {field}")
            else:
                raise ValueError(f"配置中缺少必需字段: {field}")
    
    # 验证数据集名称
    valid_datasets = [
        "celeba_aligned", "celeba_wild", "cub_aligned", "cub_001", 
        "cub_002", "cub_003", "cub_all", "deepfashion", "taichi", 
        "human3.6m", "unaligned_human3.6m", "custom"
    ]
    if config.dataset.name not in valid_datasets:
        raise ValueError(f"不支持的数据集: {config.dataset.name}. 支持的数据集: {valid_datasets}")
    
    # 验证策略选择
    valid_top_k_strategies = ["entropy", "gaussian", "consistent"]
    if config.keypoints.top_k_strategy not in valid_top_k_strategies:
        raise ValueError(f"不支持的top_k策略: {config.keypoints.top_k_strategy}")
    
    valid_max_loc_strategies = ["argmax", "weighted_avg"]
    if config.keypoints.max_loc_strategy not in valid_max_loc_strategies:
        raise ValueError(f"不支持的max_loc策略: {config.keypoints.max_loc_strategy}")
    
    valid_evaluation_methods = [
        "inter_eye_distance", "visible", "mean_average_error", "pck", "orientation_invariant"
    ]
    if config.evaluation.method not in valid_evaluation_methods:
        raise ValueError(f"不支持的评估方法: {config.evaluation.method}")
    
    # 验证设备可用性
    if config.training.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA设备不可用，但配置要求使用: {config.training.device}")


def convert_config_to_args(config: DictConfig) -> argparse.Namespace:
    """
    将OmegaConf配置转换为与原始argparse兼容的Namespace对象
    这是为了保持与现有代码的兼容性
    
    Args:
        config: OmegaConf配置对象
        
    Returns:
        argparse.Namespace对象，包含所有配置值
    """
    args = argparse.Namespace()
    
    # 模型配置
    args.model_type = config.model.type
    args.my_token = config.model.my_token
    
    # 数据集配置
    args.dataset_name = config.dataset.name
    args.dataset_loc = config.dataset.location
    args.max_len = config.dataset.max_len
    args.validation = config.dataset.validation
    
    # 训练配置
    args.device = config.training.device
    args.lr = config.training.lr
    args.num_steps = config.training.num_steps
    args.num_tokens = config.training.num_tokens
    args.batch_size = config.training.batch_size
    
    # 特征配置
    args.feature_upsample_res = config.features.upsample_res
    args.layers = config.features.layers
    args.noise_level = config.features.noise_level
    
    # 关键点配置
    args.top_k = config.keypoints.top_k
    args.top_k_strategy = config.keypoints.top_k_strategy
    args.max_loc_strategy = config.keypoints.max_loc_strategy
    args.min_dist = config.keypoints.min_dist
    args.max_num_points = config.keypoints.max_num_points
    args.num_indices = config.keypoints.num_indices
    args.num_subjects = config.keypoints.num_subjects
    args.sigma = config.keypoints.sigma
    
    # 损失权重
    args.sharpening_loss_weight = config.loss.sharpening_weight
    args.equivariance_attn_loss_weight = config.loss.equivariance_attn_weight
    
    # 数据增强
    args.augment_degrees = config.augmentation.degrees
    args.augment_scale = config.augmentation.scale
    args.augment_translate = config.augmentation.translate
    args.augmentation_iterations = config.augmentation.iterations
    
    # 评估
    args.evaluation_method = config.evaluation.method
    args.furthest_point_num_samples = config.evaluation.furthest_point_num_samples
    
    # 可视化
    args.visualize = config.visualization.visualize
    args.save_folder = config.visualization.save_folder
    
    # WandB
    args.wandb = config.wandb.enabled
    args.wandb_name = config.wandb.name
    
    return args


def create_config_parser() -> argparse.ArgumentParser:
    """
    创建简化的命令行解析器，只接受配置文件路径
    
    Returns:
        配置的ArgumentParser对象
    """
    parser = argparse.ArgumentParser(
        description="StableImageKeypoints训练 - 基于OmegaConf配置管理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="覆盖配置项，格式: key=value (例如: training.lr=0.01 dataset.name=cub_aligned)"
    )
    
    return parser


def parse_overrides(override_list: Optional[list]) -> Dict[str, Any]:
    """
    解析命令行覆盖参数
    
    Args:
        override_list: 覆盖参数列表，格式为 ["key=value", ...]
        
    Returns:
        解析后的覆盖字典
    """
    overrides = {}
    if not override_list:
        return overrides
    
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"覆盖参数格式错误: {item}. 应为 key=value 格式")
        
        key, value = item.split("=", 1)
        
        # 尝试转换类型
        try:
            # 尝试转换为数字
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # 尝试转换为布尔值
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            # 否则保持为字符串
        
        # 处理嵌套键
        keys = key.split(".")
        temp_dict = overrides
        for k in keys[:-1]:
            if k not in temp_dict:
                temp_dict[k] = {}
            temp_dict = temp_dict[k]
        temp_dict[keys[-1]] = value
    
    return overrides


def get_default_config_path() -> str:
    """
    获取默认配置文件路径
    
    Returns:
        默认配置文件的绝对路径
    """
    current_dir = Path(__file__).parent.parent
    return str(current_dir / "configs" / "default.yaml")


def setup_config() -> DictConfig:
    """
    设置配置的主入口函数
    解析命令行参数，加载配置文件，应用覆盖，并进行验证
    
    Returns:
        验证后的配置对象
    """
    parser = create_config_parser()
    cmd_args = parser.parse_args()
    
    # 解析覆盖参数
    overrides = parse_overrides(cmd_args.override)
    
    # 加载配置
    config = load_config(cmd_args.config, overrides)
    
    # 验证配置
    validate_config(config)
    
    return config