# Akida-Compatible Skin cancer Classification

This repository implements a deep learning model for classifying skin lesions, designed to be compatible with BrainChip's Akida neuromorphic hardware. The model combines GhostNet-style separable convolutions, Squeeze-and-Excitation (SE), and Efficient Channel Attention (ECA), allowing for efficient quantization and spike-based conversion.

## Model Overview

### The model follows a lightweight CNN structure composed of:

- **Ghost Blocks:**
  - SeparableConv2D (1x1 and 3x3)
  - Concatenation
  - BatchNormalization + ReLU6

- **ECA Block (Efficient Channel Attention):**
  - DepthwiseConv2D + 1x1 Conv with sigmoid gating

- **SE Block (Squeeze-and-Excitation):**
  - AveragePooling2D
  - Two 1x1 Conv layers (ReLU + Sigmoid)

- **Residual Shortcuts** with projection if needed

- **Spike Generator:**
  - SeparableConv2D + ReLU1 + SE

- **Dense Block:**
  - From `akida_models.layer_blocks.dense_block`
  - Final reshape to 7 logits for classification

**Total:** 4 Ghost-SE-ECA blocks → spike generator → dense head

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python working_CNN.py
```

This will:
- Load and preprocess the HAM10000 dataset
- Augment data and balance with SMOTE
- Train the Ghost-SE-ECA-CNN model
- Quantize and export to Akida `.fbz` format

## Results

### Output Files
- `initial_model83.h5`: Full-precision trained model
- `model_quantized.h5`: Quantized 8-bit model
- `model_akida.fbz`: Akida-compatible SNN model

### Performance
- **>90% test accuracy** on HAM10000
- Maintains accuracy post-quantization
- Visualizations include:
  - Confusion matrix
  - Top-5 prediction bar chart
  - Accuracy/loss curves