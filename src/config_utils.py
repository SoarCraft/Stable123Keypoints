"""
配置管理工具模块
提供基于强类型配置类的配置加载和管理功能
"""

import os
import argparse
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import torch


@dataclass
class Config:
    """主配置类，包含所有配置项"""
    # Model configuration
    model_type: str = "sd-legacy/stable-diffusion-v1-5"
    my_token: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "celeba_aligned"
    dataset_loc: str = "~"
    max_len: int = -1
    validation: bool = False
    
    # Training configuration
    device: str = "cuda:0"
    lr: float = 0.005
    num_steps: int = 500
    num_tokens: int = 500
    batch_size: int = 4
    
    # Features configuration
    feature_upsample_res: int = 128
    layers: List[int] = None
    noise_level: int = -1
    
    # Keypoints configuration
    top_k: int = 10
    top_k_strategy: str = "gaussian"
    max_loc_strategy: str = "argmax"
    min_dist: float = 0.1
    max_num_points: int = 50000
    num_indices: int = 100
    num_subjects: int = 1
    sigma: float = 2.0
    
    # Loss configuration
    sharpening_loss_weight: float = 100.0
    equivariance_attn_loss_weight: float = 1000.0
    
    # Augmentation configuration
    augment_degrees: float = 15.0
    augment_scale: List[float] = None
    augment_translate: List[float] = None
    augmentation_iterations: int = 10
    
    # Evaluation configuration
    evaluation_method: str = "inter_eye_distance"
    furthest_point_num_samples: int = 25
    
    # Visualization configuration
    visualize: bool = False
    save_folder: str = "outputs"
    
    # Wandb configuration
    wandb_enabled: bool = False
    wandb_name: str = "temp"
    wandb_project: str = "stable-image-keypoints"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [0, 1, 2, 3]
        if self.augment_scale is None:
            self.augment_scale = [0.8, 1.0]
        if self.augment_translate is None:
            self.augment_translate = [0.25, 0.25]
    
    @classmethod
    def from_dict_config(cls, dict_config: DictConfig) -> 'Config':
        """从OmegaConf DictConfig创建Config实例"""
        return cls(
            # Model configuration
            model_type=dict_config.model.type,
            my_token=dict_config.model.get('my_token', None),
            
            # Dataset configuration
            dataset_name=dict_config.dataset.name,
            dataset_loc=dict_config.dataset.location,
            max_len=dict_config.dataset.max_len,
            validation=dict_config.dataset.validation,
            
            # Training configuration
            device=dict_config.training.device,
            lr=dict_config.training.lr,
            num_steps=dict_config.training.num_steps,
            num_tokens=dict_config.training.num_tokens,
            batch_size=dict_config.training.batch_size,
            
            # Features configuration
            feature_upsample_res=dict_config.features.upsample_res,
            layers=list(dict_config.features.layers),
            noise_level=dict_config.features.noise_level,
            
            # Keypoints configuration
            top_k=dict_config.keypoints.top_k,
            top_k_strategy=dict_config.keypoints.top_k_strategy,
            max_loc_strategy=dict_config.keypoints.max_loc_strategy,
            min_dist=dict_config.keypoints.min_dist,
            max_num_points=dict_config.keypoints.max_num_points,
            num_indices=dict_config.keypoints.num_indices,
            num_subjects=dict_config.keypoints.num_subjects,
            sigma=dict_config.keypoints.sigma,
            
            # Loss configuration
            sharpening_loss_weight=dict_config.loss.sharpening_weight,
            equivariance_attn_loss_weight=dict_config.loss.equivariance_attn_weight,
            
            # Augmentation configuration
            augment_degrees=dict_config.augmentation.degrees,
            augment_scale=list(dict_config.augmentation.scale),
            augment_translate=list(dict_config.augmentation.translate),
            augmentation_iterations=dict_config.augmentation.iterations,
            
            # Evaluation configuration
            evaluation_method=dict_config.evaluation.method,
            furthest_point_num_samples=dict_config.evaluation.furthest_point_num_samples,
            
            # Visualization configuration
            visualize=dict_config.visualization.visualize,
            save_folder=dict_config.visualization.save_folder,
            
            # Wandb configuration
            wandb_enabled=dict_config.wandb.enabled,
            wandb_name=dict_config.wandb.name,
            wandb_project=dict_config.wandb.project,
            wandb_entity=dict_config.wandb.entity
        )


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


def validate_config(config: Config) -> None:
    """
    验证配置的必要字段和合法性
    
    Args:
        config: 要验证的配置对象
        
    Raises:
        ValueError: 配置验证失败
    """
    # 验证必需的字段
    if not config.dataset_name:
        raise ValueError("配置中缺少必需字段: dataset_name")
    if not config.device:
        raise ValueError("配置中缺少必需字段: device")
    if config.lr <= 0:
        raise ValueError("学习率必须大于0")
    if config.num_steps <= 0:
        raise ValueError("训练步数必须大于0")
    
    # 验证数据集名称
    valid_datasets = [
        "celeba_aligned", "celeba_wild", "cub_aligned", "cub_001", 
        "cub_002", "cub_003", "cub_all", "deepfashion", "taichi", 
        "human3.6m", "unaligned_human3.6m", "custom"
    ]
    if config.dataset_name not in valid_datasets:
        raise ValueError(f"不支持的数据集: {config.dataset_name}. 支持的数据集: {valid_datasets}")
    
    # 验证策略选择
    valid_top_k_strategies = ["entropy", "gaussian", "consistent"]
    if config.top_k_strategy not in valid_top_k_strategies:
        raise ValueError(f"不支持的top_k策略: {config.top_k_strategy}")
    
    valid_max_loc_strategies = ["argmax", "weighted_avg"]
    if config.max_loc_strategy not in valid_max_loc_strategies:
        raise ValueError(f"不支持的max_loc策略: {config.max_loc_strategy}")
    
    valid_evaluation_methods = [
        "inter_eye_distance", "visible", "mean_average_error", "pck", "orientation_invariant"
    ]
    if config.evaluation_method not in valid_evaluation_methods:
        raise ValueError(f"不支持的评估方法: {config.evaluation_method}")
    
    # 验证设备可用性
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA设备不可用，但配置要求使用: {config.device}")


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


def setup_config() -> Config:
    """
    设置配置的主入口函数
    解析命令行参数，加载配置文件，应用覆盖，并进行验证
    
    Returns:
        验证后的强类型配置对象
    """
    parser = create_config_parser()
    cmd_args = parser.parse_args()
    
    # 解析覆盖参数
    overrides = parse_overrides(cmd_args.override)
    
    # 加载配置
    dict_config = load_config(cmd_args.config, overrides)
    
    # 转换为强类型配置对象
    config = Config.from_dict_config(dict_config)
    
    # 验证配置
    validate_config(config)
    
    return config
