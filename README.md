# Project 5: Robust Medical Vision
## Brain Tumor MRI Classification with Uncertainty Estimation

**Student:** Kalash Kumari Thakur | **Enrollment:** 230136

---

## Problem Statement

Clinical AI systems must know when they do not know. Standard classifiers produce a prediction regardless of internal confidence — in medical imaging this is unsafe. This project builds a brain tumour MRI classifier that **explicitly measures uncertainty** and flags low-confidence predictions for radiologist review.

---

## Dataset

**Brain Tumor MRI Dataset** — Kaggle  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

| Class | Count | Description |
|-------|-------|-------------|
| Glioma | ~1,621 | Malignant; irregular ragged boundaries |
| Meningioma | ~1,645 | Benign; smooth well-defined edges |
| No Tumor | ~2,000 | Healthy brain tissue |
| Pituitary | ~1,757 | Pituitary gland tumour |

---

## Project Structure

```
medical_vision_project/
├── README.md                     # This file
├── DL_DOCUMENTATION.md           # Deep Learning technical details
├── requirements.txt              # Python dependencies
├── RUN_INSTRUCTIONS.md           # Quick start guide
├── index.html                    # Interactive dashboard
├── styles.css                    # Dashboard styling
├── app.js                        # Dashboard logic
├── server.py                    # Flask inference server
├── data/
│   └── Training/
│       ├── glioma/               # ~1400 images
│       ├── meningioma/           # ~1400 images
│       ├── notumor/              # ~1400 images
│       └── pituitary/            # ~1400 images
├── src/
│   ├── 01_eda.py              ← EDA & data quality analysis
│   ├── 02_baseline_ml.py      ← BML: Random Forest + uncertainty
│   ├── 03_advanced_ml.py      ← AML: SVM + PCA + Platt calibration
│   ├── 04_deep_learning.py     ← DL: ResNet-18 + MC-Dropout
│   └── 05_hybrid.py           ← Hybrid: Neuro-symbolic (ResNet → SVM)
├── outputs/
│   ├── eda/                   ← 7 EDA PNGs
│   ├── bml/                   ← 10 BML PNGs
│   ├── aml/                   ← 10 AML PNGs
│   ├── dl/                    ← 12 DL PNGs
│   └── hybrid/                 ← 4 Hybrid PNGs + architecture diagram
└── index.html                 ← Interactive dashboard
```
--- 
```
medical_vision_project/
├── README.md
├── server.py                    # Flask inference server
├── data/
│   └── Training/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── src/
│   ├── 01_eda.py              ← EDA & data quality analysis
│   ├── 02_baseline_ml.py      ← BML: Random Forest + uncertainty
│   └── 03_advanced_ml.py      ← AML: SVM + PCA + Platt calibration
└── outputs/
    ├── eda/                      # 7 EDA visualizations (PNG)
    ├── bml/                      # 10 BML visualizations (PNG)
    ├── aml/                      # 10 AML visualizations (PNG)
    └── dl/                       # 12 DL visualizations (PNG)
```
---

## Models Overview

### Phase 1: Traditional ML

#### Baseline ML (BML) — `02_baseline_ml.py`

| Aspect | Details |
|--------|---------|
| **Model** | Random Forest (n=200 trees, class_weight='balanced') |
| **Features** | HOG (~324d) + LBP (26d) + GLCM (10d) + Intensity Stats (10d) = ~370d |
| **Uncertainty** | Native `predict_proba()` from tree vote distribution + Shannon entropy |
| **Expected Accuracy** | ~85% |

#### Advanced ML (AML) — `03_advanced_ml.py`

| Aspect | Details |
|--------|---------|
| **Model** | SVM (RBF kernel, C=10) with Platt Scaling + PCA |
| **Key Advance** | PCA reduces ~370d → ~N dimensions retaining 95% variance |
| **Uncertainty** | Platt-calibrated probabilities + Shannon entropy + reliability diagrams |
| **Why SVM** | Maximum-margin classifier handles non-linear Glioma/Meningioma boundary |
| **Expected Accuracy** | ~87% |

### Phase 2: Deep Learning

#### Deep Learning (DL) — `04_deep_learning.py`

| Aspect | Details |
|--------|---------|
| **Model** | ResNet-18 (pretrained on ImageNet) with fine-tuning |
| **Key Advance** | Transfer learning + MC-Dropout (20 passes) for uncertainty |
| **Uncertainty** | Shannon entropy over MC-Dropout predictions + prediction variance |
| **Why ResNet** | Skip connections preserve texture; optimal depth; efficient |
| **Expected Accuracy** | ~92% |

### Phase 3: Hybrid (Neuro-Symbolic)

#### Hybrid Model (Hybrid) — `05_hybrid.py`

| Aspect | Details |
|--------|---------|
| **Architecture** | ResNet-18 (frozen) → 512-dim features → PCA → SVM (RBF + Platt) |
| **Innovation** | Neuro-symbolic: DL learns features, ML classifies |
| **Why Hybrid** | Best calibration (Brier: 0.043) — trustworthy probabilities |
| **Expected Accuracy** | ~90% (slightly lower than DL) |
| **Trade-off** | Lower accuracy but best calibration for clinical safety |

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image pillow scipy torch torchvision
```

### 2. Download Dataset

Place the Kaggle dataset so that:

```
data/Training/glioma/      (contains .jpg files)
data/Training/meningioma/
data/Training/notumor/
data/Training/pituitary/
```

### 3. Run Scripts in Order

```bash
# Step 1: EDA (Exploratory Data Analysis)
python3 src/01_eda.py

# Step 2: Baseline ML (Random Forest)
python3 src/02_baseline_ml.py

# Step 3: Advanced ML (SVM + PCA)
python3 src/03_advanced_ml.py

# Step 4: Deep Learning (ResNet-18 + MC-Dropout)
python3 src/04_deep_learning.py

# Step 5: Hybrid (Neuro-symbolic: ResNet → SVM)
python3 src/05_hybrid.py
```

### 4. Run Inference Server

For live predictions, start the Flask server:

```bash
python3 server.py
```

The server runs on `http://localhost:5000` (or port 5001 if 5000 is in use).

### 5. View Dashboard

Open `index.html` in a web browser to explore:

-  EDA visualizations (7 plots)
-  BML model results (10 plots)
-  AML model results (10 plots)
-  DL model results (12 plots)
-  Live prediction with uncertainty estimation

---

## Output Files

### EDA (`outputs/eda/`)

| File | Description |
|------|-------------|
| `class_distribution.png` | Class counts and proportions |
| `image_resolution.png` | Width, height, file size distributions |
| `pixel_intensity_stats.png` | Per-class mean, std, skewness, kurtosis |
| `intensity_distribution.png` | Overlaid intensity histograms |
| `sample_grid.png` | Example MRI from each class |
| `correlation_heatmap.png` | Metadata correlation matrix |
| `eda_quality_report.png` | Data quality issues and fixes |

### BML (`outputs/bml/`)

| File | Description |
|------|-------------|
| `bml_hog_visualization.png` | HOG edge maps per class |
| `bml_lbp_visualization.png` | LBP patterns and histograms |
| `bml_feature_summary.png` | All features visualized per class |
| `bml_confusion_matrix.png` | Random Forest confusion matrix |
| `bml_per_class_report.png` | Precision, Recall, F1 per class |
| `bml_confidence_distribution.png` | Confidence for correct vs wrong predictions |
| `bml_entropy_analysis.png` | Shannon entropy analysis per class |
| `bml_calibration_curve.png` | Reliability diagram |
| `bml_uncertainty_threshold.png` | Threshold analysis |
| `bml_feature_importance.png` | Top 30 Gini importances |

### AML (`outputs/aml/`)

| File | Description |
|------|-------------|
| `aml_pca_variance.png` | Cumulative variance and scree plot |
| `aml_pca_2d.png` | 2D PCA projection colored by class |
| `aml_svm_confusion_matrix.png` | SVM confusion matrix |
| `aml_per_class_report.png` | Per-class metrics |
| `aml_confidence_distribution.png` | Confidence and entropy distributions |
| `aml_entropy_analysis.png` | Entropy scatter plot by class |
| `aml_calibration_curve.png` | Reliability diagram with Platt calibration |
| `aml_uncertainty_threshold.png` | Threshold analysis |
| `aml_decision_boundary_pca.png` | SVM decision boundary in PCA-2D |
| `aml_bml_comparison.png` | BML vs AML comparison |

### DL (`outputs/dl/`)

| File | Description |
|------|-------------|
| `dl_training_curves.png` | Loss and accuracy over epochs |
| `dl_confusion_matrix.png` | ResNet-18 confusion matrix |
| `dl_per_class_metrics.png` | Per-class Precision, Recall, F1 |
| `dl_confidence_distribution.png` | Confidence and entropy (correct vs wrong) |
| `dl_entropy_analysis.png` | Entropy scatter plot by class |
| `dl_calibration_curve.png` | Reliability diagram |
| `dl_uncertainty_threshold.png` | Accuracy vs coverage trade-off |
| `dl_mc_dropout_uncertainty.png` | MC-Dropout std vs entropy |
| `dl_ood_detection.png` | High entropy for OOD detection |
| `dl_bml_aml_dl_comparison.png` | BML vs AML vs DL comparison |
| `dl_learning_rate_schedule.png` | Cosine annealing schedule |
| `dl_feature_maps.png` | First conv layer activation maps |

### Hybrid (`outputs/hybrid/`)

| File | Description |
|------|-------------|
| `architecture_diagram.png` | Publication-ready hybrid architecture |
| `hybrid_comparison.png` | Model comparison (BML vs AML vs DL vs Hybrid) |
| `hybrid_confusion_matrix.png` | Hybrid model confusion matrix |
| `hybrid_per_class.png` | Per-class Precision, Recall, F1 |

---

## Key Results

### Performance Comparison

| Model | Accuracy | Macro F1 | Brier Score |
|-------|-----------|----------|-------------|
| Random Forest (BML) | ~0.85 | ~0.83 | ~0.08 |
| SVM + PCA (AML) | ~0.87 | ~0.85 | ~0.09 |
| ResNet-18 (DL) | ~0.92 | ~0.91 | ~0.06 |
| **ResNet → SVM (Hybrid)** | **0.90** | **0.90** | **0.04** |

### Key Findings

1. **Uncertainty is Informative:** Wrong predictions have significantly higher Shannon entropy than correct ones — validates the clinical safety flagging mechanism.

2. **Hybrid Achieves Best Calibration:** The Neuro-symbolic hybrid achieves the lowest Brier score (0.043), meaning its predicted probabilities most closely match actual outcomes.

3. **Synergistic Combination:** Hybrid outperforms both traditional ML (by +5-7% F1) by using ResNet's learned features instead of hand-crafted HOG/LBP.

4. **Know When You Don't Know:** Uncertainty metrics (entropy, MC-std) reliably identify incorrect predictions — enabling the model to flag uncertain cases for radiologist review.

---

## Technical Details

### Uncertainty Estimation Methods

| Phase | Method | Description |
|-------|--------|-------------|
| BML | Tree Vote Distribution | Native RF `predict_proba()` + Shannon entropy |
| AML | Platt Scaling | Calibrated probabilities + entropy |
| DL | MC-Dropout | 20 forward passes with dropout → entropy and variance |

### Why Uncertainty Matters in Medical Imaging

- **Patient Safety:** Model can flag low-confidence predictions for expert review
- **Out-of-Distribution Detection:** High entropy may indicate rare or unknown cases
- **Calibrated Probabilities:** Brier score ensures confidence matches actual accuracy

---

## References

1. Dalal & Triggs (2005) — HOG. *IEEE CVPR*
2. Ojala et al. (2002) — LBP. *IEEE TPAMI*
3. Haralick et al. (1973) — GLCM. *IEEE SMC*
4. Guo et al. (2017) — Calibration. *ICML*
5. Platt (1999) — Platt Scaling. *MIT Press*
6. Lakshminarayanan et al. (2017) — Deep Ensembles. *NeurIPS*
7. Kompa et al. (2021) — Second Opinion Protocol. *NPJ Digital Medicine*
8. Nickparvar (2021) — Brain Tumor MRI Dataset. *Kaggle*
