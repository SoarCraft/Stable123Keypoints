# Stable123Keypoints PoC

一个基于 Zero123Plus 模型进行关键点提取的概念验证项目。

## 项目概述

Stable123Keypoints 是一个目前处于概念验证阶段的项目，旨在使用 `sudo-ai/zero123plus-v1.2` 模型进行图像关键点提取。本项目探索了 `Zero123Plus` 模型权重在关键点检测任务中的应用潜力。

## 技术背景

### 核心发现

研究发现，`Zero123Plus` 模型的预训练权重能够成功复现 [StableImageKeypoints](https://github.com/Aloento/StableImageKeypoints/tree/v1.5) 项目的关键点提取效果。

### 实现策略

- **完整加载** `Zero123Plus Pipeline` 并且进行针对性适配
- **选择性禁用** 包括但不限于以下 `Zero123Plus` 特有功能:
  - 视觉全局嵌入
  - 分类器自由引导
  - 参考图注意力机制

### 架构说明

虽然理论上可以在 `StableImageKeypoints` 的基础上通过少量修改获得相同效果，但考虑到项目的长期发展目标，我们选择加载完整的 Pipeline:

- **未来目标**: 引入多视角一致性等高级功能

## 快速开始

请参考 [StableImageKeypoints v1.5](https://github.com/Aloento/StableImageKeypoints/tree/v1.5#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B) 的环境配置要求。

1. **克隆项目**

   ```bash
   git clone https://github.com/SoarCraft/Stable123Keypoints.git
   cd Stable123Keypoints
   ```

2. **安装依赖**

   参照 `StableImageKeypoints v1.5` 的依赖安装流程。

3. **预处理数据**

   运行以下命令生成 `Zero123Plus` 的图像抠像结果:

   ```bash
   python -m datasets.cub_preprocess
   ```

4. **训练/推理**

   其余操作步骤与 `StableImageKeypoints` 项目保持一致。

## 结果展示

![示例结果](assets/res.png)

![收敛性](assets/heat.png)

![一致性](assets/augmentation.png)

我们可以观察到生成结果与 `StableImageKeypoints v1.5` 项目相似。

> [!CAUTION]  
> **请勿使用 FP16 精度**  
> 使用半精度浮点数会导致显著的精度损失，进而造成模型无法收敛。
