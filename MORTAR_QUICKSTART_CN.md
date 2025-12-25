# 搅拌砂浆流变性能预测模型 - 快速开始指南

## 概述

本项目实现了一个基于Vision Transformer的深度学习模型，用于从9帧图像序列预测搅拌过程中砂浆的流变性能（屈服应力YS和塑性粘度PV）。

## 主要特点

✅ **9帧序列处理**：处理时间序列信息  
✅ **Vision Transformer架构**：利用自注意力机制捕获空间和时间模式  
✅ **循环位移增强**：训练数据采用循环位移增强提高鲁棒性  
✅ **70/15/15数据集划分**：训练集70%，验证集15%，测试集15%  
✅ **回归输出**：直接预测YS和PV的连续值  

## 快速开始

### 1. 准备数据

数据应按以下结构组织：

```
Sequence5/                  # 图像序列目录
├── seq_0001/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   ├── ...
│   └── frame_8.jpg        # 共9帧
├── seq_0002/
│   └── ...
└── ...

List5.csv                   # 标签文件
```

`List5.csv` 格式：
```csv
sequence_id,YS,PV
seq_0001,45.2,2.3
seq_0002,52.1,3.1
...
```

### 2. 创建演示数据（可选）

如果还没有真实数据，可以运行演示脚本创建示例数据：

```bash
python demo_mortar.py
```

这将创建100个示例序列用于测试。

### 3. 训练模型

#### 基础训练（推荐用于快速测试）

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_small \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

#### 标准训练（推荐用于生产）

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

#### 大模型训练（最佳性能）

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_large \
    --workdir=/tmp/mortar_vit_large \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

### 4. 评估和预测

训练完成后，在测试集上评估模型：

```bash
python -m vit_jax.inference_mortar \
    --checkpoint_dir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv \
    --split=test \
    --output_file=predictions.csv
```

这将输出：
- MSE（均方误差）
- MAE（平均绝对误差）
- R²（决定系数）

预测结果保存在 `predictions.csv` 中。

## 模型配置

### 预设配置

| 配置 | 隐藏层维度 | 层数 | 注意力头 | 适用场景 |
|------|-----------|------|---------|---------|
| small | 384 | 6 | 6 | 快速测试/调试 |
| base | 768 | 12 | 12 | 标准训练 |
| large | 1024 | 24 | 16 | 最佳性能 |

### 关键参数

可以在 `vit_jax/configs/mortar_config.py` 中修改：

```python
# 数据配置
config.num_frames = 9              # 帧数（固定为9）
config.image_size = 224            # 图像大小
config.train_ratio = 0.7           # 训练集比例
config.val_ratio = 0.15            # 验证集比例
config.test_ratio = 0.15           # 测试集比例

# 训练配置
config.batch = 8                   # 批大小
config.total_steps = 10000         # 总训练步数
config.base_lr = 0.001             # 学习率
config.weight_decay = 0.0001       # 权重衰减

# 模型配置
config.model.temporal_aggregation = 'mean'  # 时间聚合方法
```

## 数据增强

训练过程自动应用循环位移增强：
- 随机循环移位图像序列
- 示例：[f0, f1, f2, ..., f8] → [f3, f4, f5, ..., f8, f0, f1, f2]
- 增强模型的时间不变性
- 仅应用于训练集

## 评估指标

模型报告以下指标：

1. **整体指标**
   - MSE（均方误差）
   - MAE（平均绝对误差）

2. **分指标**（YS和PV分别计算）
   - MSE_YS / MSE_PV
   - MAE_YS / MAE_PV
   - R²_YS / R²_PV

## 预期输出示例

```
==================================================
评估结果
==================================================
数据集: test

整体指标:
  MSE: 12.345
  MAE: 2.789

屈服应力 (YS) 指标:
  MSE: 15.234
  MAE: 3.012
  R²:  0.892

塑性粘度 (PV) 指标:
  MSE: 0.456
  MAE: 0.512
  R²:  0.876
==================================================
```

## 故障排除

### 内存不足（OOM）

如果遇到内存错误：
1. 减小 `config.batch` 批大小
2. 增加 `config.accum_steps` 进行梯度累积
3. 使用更小的模型配置（small）

### 数据加载问题

- 确保图像序列目录结构正确
- 确保CSV文件格式正确
- 检查图像文件命名（frame_0.jpg 到 frame_8.jpg）

### 训练速度慢

- 使用GPU/TPU加速
- 增加批大小（如果内存允许）
- 使用smaller配置进行快速迭代

## 高级用法

### 迁移学习

使用预训练的Vision Transformer作为起点：

```python
config.model_or_filename = 'path/to/pretrained/checkpoint.npz'
```

### 自定义时间聚合

实验不同的时间聚合方法：

```python
config.model.temporal_aggregation = 'attention'  # 或 'mean', 'max'
```

### 多GPU训练

代码自动使用所有可用的GPU/TPU。相应调整批大小：

```python
config.batch = 16  # 每个设备的批大小
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `vit_jax/mortar_data_loader.py` | 数据加载器，处理9帧序列 |
| `vit_jax/models_mortar.py` | 模型架构定义 |
| `vit_jax/train_mortar.py` | 训练逻辑 |
| `vit_jax/main_mortar.py` | 训练入口脚本 |
| `vit_jax/inference_mortar.py` | 推理和评估脚本 |
| `vit_jax/configs/mortar_config.py` | 配置文件 |
| `demo_mortar.py` | 演示脚本 |
| `MORTAR_README.md` | 详细英文文档 |

## 更多信息

详细的技术文档请参阅 [MORTAR_README.md](MORTAR_README.md)

## 许可证

Copyright 2024 Google LLC. Licensed under the Apache License, Version 2.0.
