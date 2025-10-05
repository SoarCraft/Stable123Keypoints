# Stable123Keypoints Stage1

[English](README_EN.md) | 简体中文

基于 Zero123Plus 模型的关键点提取探索项目 - 第一阶段研究报告。

## 项目概述

Stable123Keypoints 旨在探索 `sudo-ai/zero123plus-v1.2` 模型在关键点检测任务中的应用可能性。本阶段研究聚焦于在 [StableImageKeypoints v1.5](https://github.com/Aloento/StableImageKeypoints/tree/v1.5) 相同架构下，评估 Zero123Plus 预训练权重的直接可用性。

## 实验设计

### 测试方案

- **基准模型**: `sd-legacy/stable-diffusion-v1-5`
- **测试模型**: `sudo-ai/zero123plus-v1.2`
- **网络架构**: 保持与 `StableImageKeypoints v1.5` 基本一致
- **对比维度**:
  - 损失函数收敛情况
  - 注意力机制激活模式
  - 关键点提取效果

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

## 实验结果

### 训练收敛性分析

![损失曲线收敛](assets/sk123.png)

如图所示，在使用 `Zero123Plus` 模型权重进行训练时，损失函数能够正常收敛，这初步表明模型具备学习能力。

### 注意力机制分析

![注意力激活模式](assets/keypoint.png)

然而，通过对模型被 `context` 激活后的注意力图进行可视化分析，我们发现了**关键性问题**：注意力分布呈现**发散状态**，未能在关键点位置形成预期的集中响应模式。

### 对比实验验证

为排除加载方式的影响，我们进行了以下对比测试：

1. **完整加载 Zero123Plus Pipeline**: 注意力发散 ❌
2. **仅加载 Zero123Plus 权重（不加载 Pipeline）**: 注意力发散 ❌
3. **使用 stable-diffusion-v1-5 权重（相同代码和配置）**: 关键点提取正常 ✅

## 阶段性结论

### 核心发现

**在不针对性修改代码的前提下，Zero123Plus 预训练权重无法直接用于关键点提取任务。**

尽管模型训练过程中损失函数能够正常收敛，但模型并不对单纯的 context 做出期望的响应。具体表现为：

- ✅ **训练可行性**: 损失函数收敛正常
- ❌ **功能有效性**: 注意力机制未在关键点位置激活
- ✅ **代码正确性**: 相同代码下 `SD-1.5` 权重工作正常

### 问题归因分析

考虑到 `Zero123Plus` 与 `Stable Diffusion v1.5` 的模型结构差异较小，我们推断：

**Zero123Plus 在预训练过程中引入的特殊操作（如多视角条件注入、参考图注意力等），已经从根本上改变了模型内部权重对 `encoder_hidden_states` 的处理方式。**

这种改变并非简单的特征提取差异，而是涉及到注意力机制的深层重构，使得模型难以像原始 SD 模型那样对纯文本 `context` 产生空间局部化的响应。

> [!CAUTION]  
> **请勿使用 FP16 精度**  
> 使用半精度浮点数会导致显著的精度损失，进而造成模型无法收敛。
