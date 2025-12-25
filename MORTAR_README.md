# Mortar Rheology Prediction Model

This module implements a Vision Transformer-based deep learning model for predicting rheological properties (Yield Stress and Plastic Viscosity) of mortar during the mixing process from 9-frame image sequences.

## Overview

The model takes a sequence of 9 images as input and predicts two continuous values:
- **YS (Yield Stress)**: The stress required to initiate flow
- **PV (Plastic Viscosity)**: The resistance to flow once started

### Key Features

1. **9-Frame Sequence Processing**: The model processes temporal information from 9 consecutive frames
2. **Vision Transformer Architecture**: Uses self-attention mechanisms to capture spatial and temporal patterns
3. **Cyclic Shift Augmentation**: Training data is augmented with cyclic shifts to increase robustness
4. **Dataset Split**: 70% training, 15% validation, 15% test split
5. **Regression Output**: Direct prediction of continuous YS and PV values

## Model Architecture

### Components

1. **Patch Embedding**: Divides each frame into patches and embeds them
2. **Positional Encoding**: Adds spatial position information
3. **Transformer Encoder**: Multiple layers of self-attention and feed-forward networks
4. **Temporal Aggregation**: Combines information across the 9 frames (mean, max, or attention-based)
5. **Regression Head**: Outputs YS and PV predictions

### Architecture Details

- **Input**: (batch_size, 9, 224, 224, 3) - 9 RGB images of 224x224 pixels
- **Patch Size**: 16x16 pixels
- **Hidden Dimension**: 768 (configurable)
- **Transformer Layers**: 12 layers (configurable)
- **Attention Heads**: 12 heads (configurable)
- **Output**: (batch_size, 2) - YS and PV predictions

## Data Format

### Image Sequences

Images should be organized in the following structure:
```
Sequence5/
├── seq_0001/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   ├── ...
│   └── frame_8.jpg
├── seq_0002/
│   ├── frame_0.jpg
│   └── ...
└── ...
```

### Labels CSV

The CSV file (`List5.csv`) should have the following format:
```csv
sequence_id,YS,PV
seq_0001,45.2,2.3
seq_0002,52.1,3.1
...
```

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r vit_jax/requirements.txt
```

## Usage

### Training

To train the model with default configuration:

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py \
    --workdir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv
```

For a smaller model (faster training, debugging):

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_small \
    --workdir=/tmp/mortar_vit_small
```

For a larger model (better performance):

```bash
python -m vit_jax.main_mortar \
    --config=vit_jax/configs/mortar_config.py:get_config_large \
    --workdir=/tmp/mortar_vit_large
```

### Inference

To evaluate a trained model and generate predictions:

```bash
python -m vit_jax.inference_mortar \
    --checkpoint_dir=/tmp/mortar_vit \
    --sequence_dir=Sequence5 \
    --csv_file=List5.csv \
    --split=test \
    --output_file=predictions.csv
```

This will:
1. Load the trained model from the checkpoint directory
2. Evaluate on the specified split (train/val/test)
3. Save predictions to `predictions.csv`
4. Display evaluation metrics (MSE, MAE, R²)

## Configuration

The model can be configured using the `mortar_config.py` file. Key parameters:

### Model Configuration
- `hidden_size`: Transformer hidden dimension (default: 768)
- `num_layers`: Number of transformer layers (default: 12)
- `num_heads`: Number of attention heads (default: 12)
- `patch_size`: Size of image patches (default: 16x16)
- `temporal_aggregation`: Method to combine frames ('mean', 'max', 'attention')

### Training Configuration
- `batch`: Batch size per device (default: 8)
- `total_steps`: Total training steps (default: 10000)
- `base_lr`: Base learning rate (default: 0.001)
- `weight_decay`: Weight decay for regularization (default: 0.0001)

### Data Configuration
- `image_size`: Input image size (default: 224)
- `num_frames`: Number of frames in sequence (default: 9)
- `train_ratio`: Training data ratio (default: 0.7)
- `val_ratio`: Validation data ratio (default: 0.15)
- `test_ratio`: Test data ratio (default: 0.15)

## Data Augmentation

The training pipeline includes cyclic shift augmentation:
- Randomly shifts the frame sequence in time
- Example: [f0, f1, f2, f3, f4, f5, f6, f7, f8] → [f3, f4, f5, f6, f7, f8, f0, f1, f2]
- Helps the model learn temporal invariance
- Only applied to training data

## Evaluation Metrics

The model reports the following metrics:

- **MSE (Mean Squared Error)**: Overall and per-output (YS, PV)
- **MAE (Mean Absolute Error)**: Overall and per-output (YS, PV)
- **R² (Coefficient of Determination)**: Per-output (YS, PV)

## Example Results

After training, you can expect evaluation outputs like:

```
EVALUATION RESULTS
==================================================
Dataset: test

Overall Metrics:
  MSE: 12.345
  MAE: 2.789

Yield Stress (YS) Metrics:
  MSE: 15.234
  MAE: 3.012
  R²:  0.892

Plastic Viscosity (PV) Metrics:
  MSE: 0.456
  MAE: 0.512
  R²:  0.876
==================================================
```

## Advanced Usage

### Transfer Learning

To use a pretrained Vision Transformer as a starting point:

```python
config.model_or_filename = 'path/to/pretrained/checkpoint.npz'
```

### Custom Temporal Aggregation

You can experiment with different temporal aggregation methods:

```python
config.model.temporal_aggregation = 'attention'  # or 'mean', 'max'
```

### Multi-GPU Training

The code automatically uses all available GPUs/TPUs. Adjust batch size accordingly:

```python
config.batch = 16  # Per-device batch size
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce `config.batch` size
2. Increase `config.accum_steps` for gradient accumulation
3. Use a smaller model configuration

### Missing Data

If image sequences are missing, the data loader will:
1. Log a warning
2. Use dummy data (for testing)
3. Continue training

Make sure your data is properly formatted before production training.

## Citation

If you use this model, please cite the Vision Transformer paper:

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

## License

Copyright 2024 Google LLC. Licensed under the Apache License, Version 2.0.
