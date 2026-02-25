# Medical Image AI: Chest X-ray & Abdominal CT Interpretation

A comprehensive Deep Learning project for automated medical image analysis, focusing on **Abdominal Trauma Detection (CT)** and **Chest Disease Classification (X-ray)**. This repository demonstrates end-to-end AI development from advanced preprocessing to sophisticated model architecture design.

## 🚀 Key Technical Highlights

### 1. 2.5D Abdominal Trauma Detection (CT)
- **Architecture**: ConvNeXt-Tiny Backbone + Transformer Encoder + Gated Heads.
- **Innovation**:
    - **Contextual Analysis**: Utilizes a Transformer Encoder to capture spatial relationships across 64 consecutive CT slices.
    - **Gated Multi-Head Strategy**: Implements a global "Suspicion Head" that gates organ-specific classifiers, significantly reducing false positives by conditioning organ damage detection on the overall injury probability.
    - **Optimization**: Layer-wise Learning Rate Decay (LLRD) for stable fine-tuning of deep vision backbones.
- **Tech Stack**: PyTorch, MONAI, TIMM, TorchMetrics.

### 2. Multi-Dataset Benchmarking (X-ray)
- **Model**: DenseNet-121 with custom multi-label heads.
- **Research**: Compared performance across two major datasets (**NIH Chest X-ray** and **CheXpert**) using standardized labels and balanced log-scale weighting.
- **Optimization**: Strategic unfreezing and scheduled learning rate reductions for transfer learning.

## 📂 Repository Structure

```text
├── src/
│   ├── data/       # MONAI-based datasets & advanced medical transforms
│   ├── models/     # CT (ConvNeXt+Transformer) & X-ray (DenseNet) architectures
│   ├── engine/     # Modular trainer with LLRD and gradient accumulation
│   └── utils/      # Visualization & medical imaging metrics
├── experiments/    # Historical experimental notebooks (Archives)
├── assets/         # Model weights, results history, and visualizations
├── README_v2.md    # Portfolio showcase (Current)
└── requirements.txt # Reproducibility guide
```

## 📊 Performance Visuals

### CT Training Progress
![CT History](assets/ct_result_history_9.png)
*Figure: Training loss and organ-specific AUC improvement over 25 epochs using LLRD.*

### X-ray Dataset Comparison
![X-ray Results](assets/x-ray.png)
*Figure: Comparative analysis between NIH and CheXpert datasets for common thoracic pathologies.*

## 🛠️ Installation & Usage

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Modular Components**:
   - Model: `from src.models.model_ct import CTConvNeXtModel`
   - Data: `from src.data.dataset import CTDataset, get_ct_transforms`
   - Engine: `from src.engine.trainer import CTTrainer`

## 👨‍💻 Author
**Healthcare AI Enthusiast**
- Deep learning research in Medical Imaging
- Specializing in Transformers for Volumetric Data
