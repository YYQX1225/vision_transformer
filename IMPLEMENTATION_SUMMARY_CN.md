# 砂浆流变性能预测模型 - 实现总结

## 项目概述

本项目实现了一个基于Vision Transformer的深度学习模型，用于从9帧图像序列预测搅拌过程中砂浆的流变性能（屈服应力YS和塑性粘度PV）。

## 实现的功能

### ✅ 核心功能

1. **9帧图像序列处理**
   - 支持从Sequence5目录加载9帧图像序列
   - 自动调整图像大小至224x224
   - 支持多种图像格式（JPG等）

2. **数据集划分**
   - 训练集：70%
   - 验证集：15%
   - 测试集：15%
   - 支持随机打乱和固定种子

3. **循环位移增强**
   - 在训练阶段自动应用循环位移增强
   - 随机循环移位图像序列
   - 提高模型的时间不变性

4. **Vision Transformer架构**
   - 基于标准ViT架构
   - 支持多种patch大小（默认16x16）
   - 多层Transformer编码器
   - 时间聚合层（mean/max/attention）

5. **回归预测**
   - 输出两个连续值：YS和PV
   - 使用MSE损失函数
   - 支持分别计算YS和PV的指标

6. **训练和评估**
   - 完整的训练流程
   - 支持学习率warmup和cosine衰减
   - 定期评估和检查点保存
   - 多GPU/TPU支持

## 文件结构

```
vision_transformer/
├── MORTAR_README.md              # 英文详细文档
├── MORTAR_QUICKSTART_CN.md      # 中文快速开始指南
├── demo_mortar.py                # 演示脚本（创建示例数据）
├── examples_mortar.py            # 代码示例
├── vit_jax/
│   ├── mortar_data_loader.py    # 数据加载器
│   ├── models_mortar.py         # 模型架构
│   ├── train_mortar.py          # 训练逻辑
│   ├── main_mortar.py           # 训练入口
│   ├── inference_mortar.py      # 推理和评估
│   ├── test_mortar.py           # 单元测试
│   └── configs/
│       └── mortar_config.py     # 配置文件
└── .gitignore                    # 已更新（排除数据文件）
```

## 使用方法

### 1. 准备数据

数据格式要求：

**图像序列目录结构（Sequence5/）：**
```
Sequence5/
├── seq_0001/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   ├── ...
│   └── frame_8.jpg
├── seq_0002/
│   └── ...
└── ...
```

**标签CSV文件（List5.csv）：**
```csv
sequence_id,YS,PV
seq_0001,45.2,2.3
seq_0002,52.1,3.1
...
```

### 2. 创建演示数据（可选）

```bash
python demo_mortar.py
```

### 3. 训练模型

**快速测试（小模型）：**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_small \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

**标准训练：**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

**高性能训练（大模型）：**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_large \
    --workdir=/tmp/mortar_vit_large \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

### 4. 评估和预测

```bash
python -m vit_jax.inference_mortar \
    --checkpoint_dir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv \
    --split=test \
    --output_file=predictions.csv
```

## 模型配置

### 预设配置对比

| 配置 | 隐藏层维度 | 层数 | 注意力头 | 训练步数 | 批大小 | 适用场景 |
|------|-----------|------|---------|----------|--------|---------|
| small | 384 | 6 | 6 | 1,000 | 4 | 快速测试/调试 |
| base | 768 | 12 | 12 | 10,000 | 8 | 标准训练 |
| large | 1024 | 24 | 16 | 20,000 | 8 | 最佳性能 |

### 主要参数

```python
# 数据参数
num_frames = 9              # 帧数（固定）
image_size = 224            # 图像大小
train_ratio = 0.7           # 训练集比例
val_ratio = 0.15            # 验证集比例
test_ratio = 0.15           # 测试集比例

# 模型参数
patch_size = [16, 16]       # 补丁大小
hidden_size = 768           # 隐藏层维度
num_layers = 12             # Transformer层数
num_heads = 12              # 注意力头数
temporal_aggregation = 'mean'  # 时间聚合方法

# 训练参数
batch = 8                   # 批大小
total_steps = 10000         # 总训练步数
base_lr = 0.001             # 基础学习率
weight_decay = 0.0001       # 权重衰减
```

## 技术特点

### 1. Vision Transformer架构

- **Patch Embedding**: 将每帧图像分割成16x16的patches
- **Position Encoding**: 为每个patch添加位置编码
- **Transformer Encoder**: 多层自注意力机制
- **Temporal Aggregation**: 聚合9帧的时间信息
- **Regression Head**: 输出YS和PV预测值

### 2. 数据增强

- **循环位移**: 随机循环移位9帧序列
- **仅训练集**: 验证集和测试集不使用增强

### 3. 训练策略

- **Warmup + Cosine Decay**: 学习率调度
- **AdamW优化器**: 带权重衰减
- **Gradient Clipping**: 梯度裁剪防止爆炸
- **Multi-device Support**: 自动使用所有GPU/TPU

### 4. 评估指标

- **MSE**: 均方误差（整体和分指标）
- **MAE**: 平均绝对误差（整体和分指标）
- **R²**: 决定系数（分指标）

## 代码示例

详见 `examples_mortar.py` 文件，包含：

1. 基础训练示例
2. 自定义配置示例
3. 模型评估示例
4. 预测新数据示例
5. 数据增强示例

## 测试

运行单元测试：

```bash
python -m vit_jax.test_mortar
```

测试内容：
- 数据加载和划分
- 配置文件创建
- 模型初始化和前向传播

## 依赖项

主要依赖（来自 `vit_jax/requirements.txt`）：

- JAX >= 0.4.2
- Flax >= 0.6.4
- TensorFlow >= 2.4.0
- NumPy >= 1.19.5
- Pandas >= 1.1.0
- ml-collections >= 0.1.0

## 扩展和定制

### 使用预训练模型

```python
config.model_or_filename = 'path/to/pretrained/checkpoint.npz'
```

### 修改时间聚合方法

```python
config.model.temporal_aggregation = 'attention'  # 或 'mean', 'max'
```

### 调整模型大小

```python
config.model.hidden_size = 512
config.model.transformer.num_layers = 8
config.model.transformer.num_heads = 8
```

### 修改训练参数

```python
config.batch = 16
config.total_steps = 20000
config.base_lr = 0.0005
```

## 性能优化建议

1. **内存不足**：
   - 减小批大小
   - 使用梯度累积（增加 `accum_steps`）
   - 使用小模型配置

2. **训练速度**：
   - 使用GPU/TPU
   - 增加批大小
   - 减少评估频率

3. **模型性能**：
   - 使用大模型配置
   - 增加训练步数
   - 调整学习率和权重衰减
   - 尝试不同的时间聚合方法

## 常见问题

### Q: 如果数据格式不同怎么办？

A: 修改 `mortar_data_loader.py` 中的 `load_image_sequence` 方法以适应你的数据格式。

### Q: 如何使用自己的模型架构？

A: 在 `models_mortar.py` 中修改 `MortarVisionTransformer` 类，或创建新的模型类。

### Q: 如何添加更多的输出变量？

A: 修改 `num_outputs` 参数，并相应调整CSV文件格式和评估指标。

### Q: 支持其他数量的帧吗？

A: 是的，修改 `config.num_frames` 参数即可。模型会自动适应。

## 未来改进方向

1. 支持视频文件输入（而非单独图像）
2. 添加更多的数据增强方法
3. 支持多任务学习（同时预测其他性能指标）
4. 实现3D卷积或时序卷积作为baseline对比
5. 添加注意力可视化工具
6. 支持在线学习和增量训练

## 贡献和反馈

如有问题或建议，请通过GitHub Issues反馈。

## 许可证

Copyright 2024 Google LLC. Licensed under the Apache License, Version 2.0.

## 参考文献

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

---

**实现完成日期**: 2024
**版本**: 1.0.0
