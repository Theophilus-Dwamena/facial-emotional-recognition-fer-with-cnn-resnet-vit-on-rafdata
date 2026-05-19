# Facial Emotion Recognition: CNN vs ResNet-50 vs ViT-B/16

A comparative study of three deep learning architectures for 7-class facial 
emotion recognition on the Real-world Affective Faces Database (RAF-DB).

## Overview

This project implements and evaluates three architectures under identical 
training conditions so that performance differences are attributable to 
architectural choice alone:

| Model       | Accuracy | Macro-F1 | Macro-AUC |
|-------------|----------|----------|-----------|
| Custom CNN  | 61.54%   | 0.5217   | 0.872     |
| ResNet-50   | **77.18%**   | **0.6939**   | **0.944**     |
| ViT-B/16    | 70.96%   | 0.6041   | 0.928     |

ResNet-50 achieves the strongest performance across all metrics, demonstrating 
that pretrained convolutional networks outperform both lightweight baselines 
and vision transformers on moderately sized facial expression datasets.

## Dataset

**RAF-DB** (Real-world Affective Faces Database) — Li et al., CVPR 2017  
- 15,339 crowd-annotated RGB images  
- 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise  
- 17× class imbalance (Happy: 4,772 vs Fear: 281)  
- Split: 11,044 train / 1,227 validation / 3,068 test  

> RAF-DB requires a license request. Download from the 
> [official site](http://www.whdeng.cn/RAF/model1.html) and place under 
> `data/RAF-DB/`.

## Models

### Custom CNN (Baseline)
- 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- Conv3×3 → BatchNorm → ReLU → MaxPool at each block
- Global Average Pooling + FC(256) + Dropout(0.5)
- ~456K parameters, trained from scratch (He initialization)

### ResNet-50 (Transfer Learning)
- ImageNet-pretrained weights (IMAGENET1K_V2)
- Frozen: Stem + Layers 1–3 (~8.5M params)
- Trainable: Layer 4 + 7-class head (~15M params)
- Dropout(0.4) + hidden layer in classification head

### ViT-B/16 (Transfer Learning)
- ImageNet-21k pretrained weights via HuggingFace
- Frozen: Blocks 1–9 (~64M params)
- Trainable: Blocks 10–12 + 7-class head (~21M params)
- Differential learning rates: 1e-5 (blocks) / 1e-4 (head)

## Training Details

- **Loss**: Class-weighted cross-entropy (wc = N / C·nc)
- **Optimizer**: Adam
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: patience=10, monitor val macro-F1
- **Batch size**: 32
- **Hardware**: NVIDIA A100 40GB (Google Colab)
- **Seed**: 42 (fully reproducible)

## Results

### Per-Class F1

| Emotion  | Custom CNN | ResNet-50 | ViT-B/16 |
|----------|-----------|-----------|----------|
| Angry    | 0.496     | **0.710** | 0.585    |
| Disgust  | 0.252     | **0.471** | 0.369    |
| Fear     | 0.493     | **0.559** | 0.373    |
| Happy    | 0.774     | **0.884** | 0.848    |
| Neutral  | 0.612     | **0.722** | 0.685    |
| Sad      | 0.386     | **0.735** | 0.643    |
| Surprise | 0.640     | **0.776** | 0.726    |

ResNet-50 achieves the best F1 in every single class. Disgust is the hardest 
class across all models — not due to sample scarcity (717 samples) but 
intrinsic visual ambiguity with happy and neutral.

## Project Structure
