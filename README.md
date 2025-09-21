# StableImageKeypoints

**中文版本 | [English](README_EN.md)**

> 这是对原始 StableKeypoints 项目的改进版本。如需了解原始项目的详细信息，请访问[原始仓库](https://github.com/ubc-vision/StableKeypoints)。

## 改进内容

我对原始代码进行了以下关键改进：

1. **配置文件重构**: 将命令行参数改为基于 YAML 的配置文件系统
2. **Diffusers 版本升级**: 从 0.8 版本升级到 0.33 版本，适配最新的 API 变化
3. **多 GPU 训练优化**: 修复了多 GPU 训练时设备不同步的问题
4. **现代化兼容**: 更新代码以适配最新的库和语言要求
5. **代码整理优化**: 清理和重构代码结构

## 快速开始

### 环境配置

确保你已经安装了必要的依赖包（包括 PyTorch、diffusers、transformers 等）。

### Weights & Biases 登录

首先登录 wandb 用于训练监控：

```bash
wandb login
```

### 数据集准备

以 CUB 数据集为例：

1. 从 [Caltech 官方网站](https://data.caltech.edu/records/65de6-vp158) 下载 CUB-200-2011 数据集
2. 将下载的数据集解压到 `data/` 目录下
3. 将项目中的 `datasets/cub_cachedir.zip` 解压到 `data/CUB_200_2011/` 目录中

最终的目录结构应该如下：

```
data/
└── CUB_200_2011/
    ├── attributes/
    ├── cachedir/
    │   └── cub # 来自解压的 cub_cachedir.zip
    ├── images/
    │   ├── 001.Black_footed_Albatross/
    │   └── ...
    ├── parts/
    └── ...
```

> 如需使用其他数据集（CelebA、Taichi、Human3.6M、DeepFashion 等），请参考原始仓库的数据集下载指南。

### 配置文件

编辑 `configs/default.yaml` 文件以适应你的需求：

```yaml
# 更多配置项...

# 数据集配置
dataset:
  name: "cub_001" # 或其他数据集名称
  location: "data" # 数据集路径

# 模型配置
model:
  type: "sd-legacy/stable-diffusion-v1-5" # 或 "stabilityai/stable-diffusion-2-1-base"
  my_token: null # 如需要，填入你的 Hugging Face token

# 训练配置
training:
  num_steps: 100 # 根据需要调整
  batch_size: 8
  device: "cuda"
```

### 开始训练

配置完成后，运行以下命令开始训练：

```bash
python3 -m src.main
```

训练完成后，你将在 `outputs/` 目录（或配置中指定的目录）下找到检查点，包括可视化图像。

## 模型建议

为了获得更好的结果，建议使用 `stabilityai/stable-diffusion-2-1-base` 模型。这是最后一个使用 UNet 架构且 Attention 模块相同的 Stable Diffusion 模型。

**注意**: 本项目不支持 Stable Diffusion 3 及以上版本，因为它不再使用 UNet 架构。

## 结果展示

![示例结果](assets/res.png)

![收敛性](assets/heat.png)

![一致性](assets/augmentation.png)

此项目并非是一个困难任务，它主要是对现有代码的改进。希望这个改进版本能够帮助到你的研究。
