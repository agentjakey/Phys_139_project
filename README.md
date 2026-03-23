# Efficient CNN Architectures for Medical Image Classification
### Reproducing and Evaluating Eff-PCNet — UCSD PHYS 139 Final Project

**Authors:** Jacob Ortiz · Garren McKinley · David Culver · Andres Flores

[![Open in Colab for Fully Ran Code](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tSDXoeicGtcOS97Lz0Jh_QbIYUjpacNh?usp=sharing)

---

## Overview

This project implements and evaluates two CNN architectures for medical image classification, motivated by the **Eff-PCNet** paper (Yue et al., 2023). The core question: can a lightweight CNN inspired by Eff-PCNet's multi-scale and multi-branch design principles match or beat a strong transfer learning baseline at a fraction of the parameter cost?

**We implement:**
- An **EfficientNetV2-S transfer learning baseline** (ImageNet pretrained, frozen backbone, ~20.3M parameters)
- A custom **EffPCNet-Small** model inspired by Eff-PCNet's M2C and Rep-C modules (~4.39M parameters)

**Evaluated on three medical imaging datasets:** HAM10000, SkinCancer MNIST, and Chest X-ray Pneumonia.

**Key result:** EffPCNet-Small outperforms the EfficientNetV2 baseline on SkinCancer and HAM10000 across all three metrics (Accuracy, F1, AUC) using only ~1/5 the parameters.

---

## Results

### Accuracy Comparison

| Dataset | Eff-PCNet (paper) | EfficientNetV2 (ours) | EffPCNet-Small (ours) |
|---|---|---|---|
| SkinCancer | 0.91 | 0.6449 | **0.8498** |
| HAM10000 | 0.87 | 0.6755 | **0.7668** |
| ChestXray | 0.97 | **0.8275** | 0.6362 |

### AUC Comparison

| Dataset | Eff-PCNet (paper) | EfficientNetV2 (ours) | EffPCNet-Small (ours) |
|---|---|---|---|
| SkinCancer | 0.97 | 0.7445 | **0.9266** |
| HAM10000 | 0.95 | 0.5583 | **0.9226** |
| ChestXray | 0.99 | 0.8700 | **0.9721** |

### Parameter Efficiency

| Model | Parameters |
|---|---|
| EfficientNetV2-S (baseline) | ~20.3M |
| EffPCNet-Small (ours) | ~4.39M |

EffPCNet-Small achieves better performance on 2/3 datasets with ~1/5 the parameters.

---

## Architecture

### EffPCNet-Small

Our lightweight model is inspired by — but not an exact replica of — Eff-PCNet. It captures the core modular design using three key block types:

```
Input (224x224x3)
    |
Stem: Conv-BN-Swish blocks (24 -> 32 -> 48 -> 64 channels, with downsampling)
    |
Stage 4: M2C-style block
    - Split channels into 4 branches:
        Branch 1: 3x3 DepthwiseConv
        Branch 2: 1xK DepthwiseConv (K=11)
        Branch 3: Kx1 DepthwiseConv (K=11)
        Branch 4: Identity pass-through
    - Concatenate all branches -> 1x1 Conv + BN + ReLU
    |
Stage 5: MBC-style block (128 channels, stride=2, x2 repeats)
    |
Stage 6: Rep-C-style block (128 channels)
    - Three parallel branches: 3x3 Conv | 1x1 Conv | DWConv + pointwise projection
    - Element-wise Add -> SE attention -> ReLU
    |
Stages 7-8: MBC-style blocks (192 -> 256 channels, stride=2, x2 repeats each)
    |
Head: Global Average Pool -> Dropout(0.3) -> Dense(softmax)
```

**Differences from full Eff-PCNet:**
- Fewer repeated blocks and smaller channel widths (to fit in Colab RAM)
- No inference-time structural re-parameterization (Rep-C branches remain separate at inference)
- Trained for 20 epochs vs. 300 in the paper

### Block Descriptions

**M2C (Multi-branch Multi-scale Convolution):** Splits feature channels into 4 branches — three depthwise convolutions (3x3, 1xK, Kx1 with K=11) and one identity pass-through. Approximates large receptive fields cheaply by decomposing 2D convolutions into asymmetric 1D pairs. Branches are concatenated and compressed with a 1x1 convolution.

**Rep-C (Re-parameterization Convolution):** Uses multiple parallel convolutional branches (3x3, 1x1, depthwise) during training, whose outputs are summed element-wise. In the full Eff-PCNet, these are fused into a single equivalent kernel at inference time. Our implementation keeps branches separate at inference but retains the multi-branch representational advantage during training.

**MBC (Multi-Branch Convolution):** Residual block with two Conv-BN-Swish layers followed by Squeeze-and-Excitation (SE) channel recalibration and a skip connection.

---

## Datasets

All datasets are publicly available on Kaggle and downloaded automatically via `kagglehub`.

| Dataset | Task | Classes | Images |
|---|---|---|---|
| [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Multi-class skin lesion | 7 (akiec, bcc, bkl, df, mel, nv, vasc) | 10,015 |
| [SkinCancer MNIST](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) | Binary (malignant vs. benign) | 2 | 3,297 |
| [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Binary (normal vs. pneumonia) | 2 | 5,856 |

Datasets are organized automatically into `data/<DatasetName>/<class>/` and split 80/20 train/val (fixed seed, non-overlapping).

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs | 20 |
| Early stopping | None |
| Image size | 224x224 |

**Preprocessing (train):** Random horizontal flip, rotation (+-5 deg), zoom (+-10%), contrast (+-10%), rescale to [0, 1].

**Preprocessing (val):** Rescale to [0, 1] only.

**Evaluation metrics:** Accuracy, macro F1, macro AUC (one-vs-rest), confusion matrix.

---

## Getting Started

### Option 1: Google Colab (Recommended)

Click the badge at the top of this README, or go directly to:
https://colab.research.google.com/drive/1tSDXoeicGtcOS97Lz0Jh_QbIYUjpacNh

1. Set runtime: **Runtime > Change runtime type > T4 GPU**
2. Run all cells in order: **Runtime > Run all**
3. When prompted by `kagglehub`, authenticate with your Kaggle API credentials

### Option 2: Local Setup

```bash
git clone https://github.com/agentjakey/Phys_139_project.git
cd Phys_139_project
pip install -r requirements.txt
jupyter notebook Full_Pipeline.ipynb
```

Run cells in order. A GPU is strongly recommended — training all 6 model/dataset combinations on CPU will be very slow.

### Kaggle Authentication

The notebook uses `kagglehub` to download datasets automatically. You need a Kaggle account and API token:

1. Go to kaggle.com > Account > API > Create New Token — this downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)
3. Or set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`

---

## Repository Structure

```
Phys_139_project/
├── Full_Pipeline.ipynb        # Main notebook: data prep, training, evaluation
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── results_baseline.csv       # Auto-generated: baseline model results
├── results_all_models.csv     # Auto-generated: all model results
└── data/                      # Auto-created by notebook (gitignored)
    ├── SkinCancer/
    │   ├── benign/
    │   └── malignant/
    ├── HAM10000/
    │   └── akiec/ bcc/ bkl/ df/ mel/ nv/ vasc/
    └── ChestXray/
        ├── NORMAL/
        └── PNEUMONIA/
```

> The `data/` directory is created automatically. Do not commit it to Git (it is in `.gitignore`).

---

## Bug Fixes Applied

The following bugs were identified and fixed in `Full_Pipeline.ipynb` relative to the original Colab version:

| # | Bug | Fix Applied |
|---|---|---|
| 1 | **`rep_c_block` shape mismatch** — the depthwise branch retains input channel count, causing `layers.Add()` to crash when `x.channels != filters` | Added a pointwise 1x1 Conv projection after depthwise to match `filters` before the Add |
| 2 | **Epoch count inconsistency** — all training cells used `epochs=10` but the report and abstract describe 20 epochs | Corrected all 6 training cells to `epochs=20` |
| 3 | **Duplicate pandas import** — `import pandas as pd` appeared in both the Cell 15 summary and the final summary cell | Removed redundant import from final cell |
| 4 | **Missing dependency note** — `get_datasets()` and `get_preprocessed_datasets()` are defined in the EDA cell but used in later training cells with no indication they must be run first | Added a clear note comment at the top of the EDA cell |

---

## Discussion

**Why EffPCNet-Small beats the baseline on skin datasets:**
HAM10000 and SkinCancer require fine-grained texture discrimination across visually similar lesion types. The M2C block's asymmetric depthwise branches (1xK, Kx1) capture subtle local patterns at multiple effective receptive field sizes. EfficientNetV2's frozen ImageNet weights are not optimized for this type of fine-grained medical texture.

**Why the baseline wins on ChestXray accuracy:**
Pneumonia detection relies on global structural patterns (opacification, consolidation across lung fields) rather than local texture. EfficientNetV2's ImageNet pretraining provides strong global feature detectors well-suited to this. EffPCNet-Small, with its architecture biased toward local multi-scale texture, likely underfits ChestXray in 20 epochs. Notably, EffPCNet-Small still achieves higher AUC (0.9721 vs. 0.8700), indicating good class separability but poor calibration at a fixed 0.5 decision threshold.

**Gap from paper results:**
The full Eff-PCNet was trained for 300 epochs with a learning rate schedule, uses full channel widths, full block repetition counts, and inference-time structural re-parameterization. Our constrained version underperforms on absolute metrics but reproduces the qualitative trend: the Eff-PCNet style design provides a better performance-per-parameter tradeoff than standard transfer learning.

---

## Future Work

- Implement true structural re-parameterization (fuse Rep-C branches into a single conv at inference)
- Train longer with cosine annealing schedule (as in the original paper)
- Apply class-balanced sampling or weighted loss for HAM10000 (heavily imbalanced)
- Implement full Eff-PCNet depth and channel schedule
- Threshold calibration for ChestXray to recover accuracy

---

## Reference

Yue, W., Liu, S., & Li, Y. (2023). Eff-PCNet: An Efficient Pure CNN Network for Medical Image Classification. *Applied Sciences*, 13(16), 9226. https://doi.org/10.3390/app13169226

```bibtex
@article{yue2023effpcnet,
  title={Eff-PCNet: An Efficient Pure CNN Network for Medical Image Classification},
  author={Yue, W. and Liu, S. and Li, Y.},
  journal={Applied Sciences},
  volume={13},
  number={16},
  pages={9226},
  year={2023},
  doi={10.3390/app13169226}
}
```

---

## License

This project is for academic purposes (UCSD PHYS 139). Datasets are subject to their respective Kaggle licenses.
