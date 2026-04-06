# Deep Learning Module — 04_deep_learning.py

## Overview

This module implements the **Deep Learning (DL)** phase of Project 5: Robust Medical Vision. It uses a ResNet-18 convolutional neural network with **MC-Dropout** for uncertainty estimation.

---

## Model Architecture

### ResNet-18 with MC-Dropout

**Why ResNet-18?**
- **Transfer Learning:** Pre-trained on ImageNet, providing strong initial features
- **Optimal Depth:** 18 layers provide sufficient capacity without overfitting on ~5600 images
- **Skip Connections:** Residual blocks enable stable gradient flow — critical for medical imaging
- **Efficient:** Smaller than ResNet-50/101, faster training on CPU

**Architecture:**
```
ResNet-18 (pretrained)
├── conv1 (7x7, 64)
├── layer1 (residual block)
├── layer2 (residual block) ← Frozen (feature extraction)
├── layer3 (residual block) ← Frozen
├── layer4 (residual block) ← Fine-tuned
├── avgpool
└── fc (512 → 4 classes) + Dropout(0.3)
```

---

## Key Components

### 1. Transfer Learning
- Loads ImageNet pretrained weights
- Freezes first 2 residual blocks (layer1, layer2)
- Fine-tunes layer3, layer4, and final FC layer
- Reduces training time and prevents overfitting

### 2. Data Augmentation
```python
train_transform = transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15),
    RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])
```
Simulates clinical variation: patient positioning, scanner differences, brightness variation.

### 3. MC-Dropout (Monte Carlo Dropout)
**What is it?**
- Keeps dropout enabled during inference
- Runs 20 forward passes per image
- Each pass produces slightly different predictions due to random dropout

**Why use it for uncertainty?**
- Standard neural networks are overconfident — they output deterministic probabilities
- MC-Dropout approximates Bayesian inference by treating dropout as Bayesian approximation
- The **variance** across 20 passes indicates model uncertainty

**How it works:**
```python
def mc_predict(model, images, n_passes=20):
    # Enable dropout at inference
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.training = True
    
    all_probs = []
    for _ in range(n_passes):
        probs = F.softmax(model(images), dim=1)
        all_probs.append(probs.cpu().numpy())
    
    mean_probs = np.mean(all_probs, axis=0)      # Expected probability
    std_probs = np.std(all_probs, axis=0).mean(axis=1)  # Uncertainty (std)
    entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)  # Shannon entropy
    
    return mean_probs, std_probs, entropy
```

### 4. Uncertainty Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Confidence** | max(P(y\|x)) | Highest class probability |
| **Entropy** | -Σ p(y) log p(y) | Information uncertainty (0 = certain, log(4) = uncertain) |
| **MC-Std** | std(P(y\|x) across passes) | Variance in predictions |

**Key Finding:** Wrong predictions consistently show **higher entropy** than correct ones. This enables the model to "know when it doesn't know" — critical for clinical safety.

### 5. Training Configuration
```python
criterion = CrossEntropyLoss(label_smoothing=0.1)  # Prevents overconfidence
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=5)
```

- **Label Smoothing:** Reduces overconfidence, improves calibration
- **AdamW:** Weight decay decoupled from gradient update
- **Cosine Annealing:** Smooth learning rate decay for stable convergence

---

## Output Plots (PNG)

All plots saved to `outputs/dl/`:

| File | Description |
|---|---|
| `dl_training_curves.png` | Loss and accuracy over epochs |
| `dl_confusion_matrix.png` | Test set confusion matrix |
| `dl_per_class_metrics.png` | Precision/Recall/F1 per class |
| `dl_confidence_distribution.png` | Confidence & entropy for correct vs wrong |
| `dl_entropy_analysis.png` | Entropy scatter by class |
| `dl_calibration_curve.png` | Reliability diagram |
| `dl_uncertainty_threshold.png` | Accuracy vs coverage trade-off |
| `dl_mc_dropout_uncertainty.png` | Std vs entropy (uncertainty correlation) |
| `dl_ood_detection.png` | High entropy = potential OOD |
| `dl_bml_aml_dl_comparison.png` | BML vs AML vs DL comparison |
| `dl_learning_rate_schedule.png` | Cosine annealing curve |
| `dl_feature_maps.png` | First conv layer activations |

---

## Results (Actual)

| Model | Accuracy | Macro F1 | Brier Score |
|---|---|---|---|
| ResNet-18 (DL) | **97.98%** | **0.9797** | **0.0100** |

**Improvement over AML:** +10% accuracy, significantly better calibration (Brier: 0.01 vs 0.09)

---

## Clinical Relevance

1. **Uncertainty Flagging:** Low-confidence predictions (entropy > threshold) can be flagged for radiologist review
2. **OOD Detection:** High-entropy samples may indicate out-of-distribution cases (e.g., rare tumor types)
3. **Calibrated Probabilities:** Brier score of 0.01 means confidence matches actual accuracy — critical for medical decision-making

---

## References

1. He et al. (2016) — ResNet. *IEEE CVPR*
2. Gal & Ghahramani (2016) — MC-Dropout. *ICML*
3. Guo et al. (2017) — Calibration. *ICML*
4. Lakshminarayanan et al. (2017) — Deep Ensembles. *NeurIPS*
5. Kendall & Gal (2017) — What uncertainties do we need? *CVPR*