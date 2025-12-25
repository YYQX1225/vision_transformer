# Mortar Rheology Prediction Model - Implementation Summary

## Project Overview

This project implements a Vision Transformer-based deep learning model for predicting rheological properties (Yield Stress and Plastic Viscosity) of mortar during the mixing process from 9-frame image sequences.

## Completed Features

### ✅ Core Functionality

1. **9-Frame Image Sequence Processing**
   - Load 9-frame sequences from Sequence5 directory
   - Automatic image resizing to 224x224
   - Support for multiple image formats (JPG, etc.)

2. **Dataset Splitting**
   - Training set: 70%
   - Validation set: 15%
   - Test set: 15%
   - Random shuffling with fixed seed support

3. **Cyclic Shift Augmentation**
   - Automatic application during training
   - Random cyclic shift of image sequences
   - Improves temporal invariance

4. **Vision Transformer Architecture**
   - Based on standard ViT architecture
   - Configurable patch size (default 16x16)
   - Multi-layer Transformer encoder
   - Temporal aggregation layer (mean/max/attention)

5. **Regression Prediction**
   - Output two continuous values: YS and PV
   - MSE loss function
   - Separate metrics for YS and PV

6. **Training and Evaluation**
   - Complete training pipeline
   - Learning rate warmup and cosine decay
   - Regular evaluation and checkpoint saving
   - Multi-GPU/TPU support

## File Structure

```
vision_transformer/
├── MORTAR_README.md                    # Detailed English documentation
├── MORTAR_QUICKSTART_CN.md           # Chinese quick start guide
├── IMPLEMENTATION_SUMMARY_CN.md       # Chinese implementation summary
├── IMPLEMENTATION_SUMMARY.md          # English implementation summary
├── demo_mortar.py                     # Demo script (creates sample data)
├── examples_mortar.py                 # Code examples
├── vit_jax/
│   ├── mortar_data_loader.py         # Data loader
│   ├── models_mortar.py              # Model architecture
│   ├── train_mortar.py               # Training logic
│   ├── main_mortar.py                # Training entry point
│   ├── inference_mortar.py           # Inference and evaluation
│   ├── test_mortar.py                # Unit tests
│   └── configs/
│       └── mortar_config.py          # Configuration file
└── .gitignore                         # Updated (excludes data files)
```

## Usage

### 1. Prepare Data

**Image sequence directory structure (Sequence5/):**
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

**Label CSV file (List5.csv):**
```csv
sequence_id,YS,PV
seq_0001,45.2,2.3
seq_0002,52.1,3.1
...
```

### 2. Create Demo Data (Optional)

```bash
python demo_mortar.py
```

### 3. Train Model

**Quick test (small model):**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_small \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

**Standard training:**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

**High-performance training (large model):**
```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_large \
    --workdir=/tmp/mortar_vit_large \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

### 4. Evaluate and Predict

```bash
python -m vit_jax.inference_mortar \
    --checkpoint_dir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv \
    --split=test \
    --output_file=predictions.csv
```

## Model Configurations

### Preset Configuration Comparison

| Config | Hidden Dim | Layers | Heads | Steps | Batch | Use Case |
|--------|-----------|--------|-------|-------|-------|----------|
| small | 384 | 6 | 6 | 1,000 | 4 | Quick test/debug |
| base | 768 | 12 | 12 | 10,000 | 8 | Standard training |
| large | 1024 | 24 | 16 | 20,000 | 8 | Best performance |

## Key Components

### 1. Data Loader (`mortar_data_loader.py`)
- Loads 9-frame image sequences
- Implements train/val/test splitting (70/15/15)
- Applies cyclic shift augmentation to training data
- Creates TensorFlow datasets for efficient batching

### 2. Model Architecture (`models_mortar.py`)
- `MortarVisionTransformer`: Main model class
- `TemporalAggregation`: Aggregates information across frames
- Patch embedding, position encoding, and Transformer layers
- Regression head for YS and PV prediction

### 3. Training Script (`train_mortar.py`)
- Training loop with MSE loss
- Learning rate scheduling (warmup + cosine decay)
- Regular evaluation on validation set
- Checkpoint saving

### 4. Configuration (`configs/mortar_config.py`)
- Three preset configurations (small, base, large)
- Easy customization of all hyperparameters
- Data paths and split ratios

### 5. Inference Script (`inference_mortar.py`)
- Load trained model from checkpoint
- Evaluate on any split (train/val/test)
- Save predictions to CSV
- Compute and display metrics (MSE, MAE, R²)

## Technical Features

### Vision Transformer Architecture
- **Patch Embedding**: Divides each frame into 16x16 patches
- **Position Encoding**: Adds positional information to patches
- **Transformer Encoder**: Multi-layer self-attention mechanism
- **Temporal Aggregation**: Combines information across 9 frames
- **Regression Head**: Outputs YS and PV predictions

### Data Augmentation
- **Cyclic Shift**: Random cyclic shift of 9-frame sequences
- **Training Only**: Validation and test sets use no augmentation

### Training Strategy
- **Warmup + Cosine Decay**: Learning rate scheduling
- **AdamW Optimizer**: With weight decay
- **Gradient Clipping**: Prevents gradient explosion
- **Multi-device Support**: Automatically uses all GPUs/TPUs

### Evaluation Metrics
- **MSE**: Mean Squared Error (overall and per-output)
- **MAE**: Mean Absolute Error (overall and per-output)
- **R²**: Coefficient of Determination (per-output)

## Code Examples

See `examples_mortar.py` for:
1. Basic training example
2. Custom configuration example
3. Model evaluation example
4. Prediction on new data example
5. Data augmentation example

## Testing

Run unit tests:
```bash
python -m vit_jax.test_mortar
```

Tests cover:
- Data loading and splitting
- Configuration creation
- Model initialization and forward pass

## Dependencies

Main dependencies (from `vit_jax/requirements.txt`):
- JAX >= 0.4.2
- Flax >= 0.6.4
- TensorFlow >= 2.4.0
- NumPy >= 1.19.5
- Pandas >= 1.1.0
- ml-collections >= 0.1.0

## Customization

### Use Pretrained Model
```python
config.model_or_filename = 'path/to/pretrained/checkpoint.npz'
```

### Change Temporal Aggregation
```python
config.model.temporal_aggregation = 'attention'  # or 'mean', 'max'
```

### Adjust Model Size
```python
config.model.hidden_size = 512
config.model.transformer.num_layers = 8
config.model.transformer.num_heads = 8
```

### Modify Training Parameters
```python
config.batch = 16
config.total_steps = 20000
config.base_lr = 0.0005
```

## Performance Optimization

1. **Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation (increase `accum_steps`)
   - Use smaller model configuration

2. **Training Speed**:
   - Use GPU/TPU
   - Increase batch size
   - Reduce evaluation frequency

3. **Model Performance**:
   - Use larger model configuration
   - Increase training steps
   - Tune learning rate and weight decay
   - Try different temporal aggregation methods

## Documentation

- **MORTAR_README.md**: Comprehensive English documentation
- **MORTAR_QUICKSTART_CN.md**: Chinese quick start guide
- **IMPLEMENTATION_SUMMARY_CN.md**: Chinese implementation summary
- **examples_mortar.py**: Runnable code examples
- **demo_mortar.py**: Demo script with sample data generation

## Future Improvements

1. Support video file input (instead of separate images)
2. Add more data augmentation methods
3. Support multi-task learning (predict other properties)
4. Implement 3D CNN or temporal CNN as baseline
5. Add attention visualization tools
6. Support online learning and incremental training

## License

Copyright 2024 Google LLC. Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

---

**Implementation Date**: 2024
**Version**: 1.0.0
