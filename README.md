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
|---|---|---|
| Glioma | ~1,621 | Malignant; irregular ragged boundaries |
| Meningioma | ~1,645 | Benign; smooth well-defined edges |
| No Tumor | ~2,000 | Healthy brain tissue |
| Pituitary | ~1,757 | Pituitary gland tumour |

---

## Project Structure

```
medical_vision_project/
├── README.md
├── DL_DOCUMENTATION.md       ← Deep Learning technical details
├── data/
│   └── Training/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── src/
│   ├── 01_eda.py              ← EDA & data quality analysis
│   ├── 02_baseline_ml.py      ← BML: Random Forest + uncertainty
│   ├── 03_advanced_ml.py      ← AML: SVM + PCA + Platt calibration
│   └── 04_deep_learning.py     ← DL: ResNet-18 + MC-Dropout
├── outputs/
│   ├── eda/                   ← 7 EDA PNGs
│   ├── bml/                   ← 10 BML PNGs
│   ├── aml/                   ← 10 AML PNGs
│   └── dl/                    ← 12 DL PNGs
└── index.html                 ← Interactive dashboard

medical_vision_project/
├── README.md
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
    ├── eda/                   ← 7 EDA PNGs
    ├── bml/                   ← 10 BML PNGs
    └── aml/                   ← 10 AML PNGs

---

## Models

### Baseline ML (BML) — `02_baseline_ml.py`
**Model:** Random Forest (n=200 trees, class_weight='balanced')  
**Features:** HOG (~324d) + LBP (26d) + GLCM (10d) + Intensity Stats (10d) = ~370d  
**Uncertainty:** Native `predict_proba()` from tree vote distribution + Shannon entropy  

### Advanced ML (AML) — `03_advanced_ml.py`
**Model:** SVM (RBF kernel, C=10) with Platt Scaling + PCA  
**Key advance:** PCA reduces ~370d → ~N dimensions retaining 95% variance before SVM  
**Uncertainty:** Platt-calibrated probabilities + Shannon entropy + reliability diagrams  
**Why SVM:** Maximum-margin classifier in the PCA-compressed space handles the non-linear Glioma/Meningioma boundary that Random Forest cannot resolve as cleanly.

### Deep Learning (DL) — `04_deep_learning.py`
**Model:** ResNet-18 (pretrained on ImageNet) with fine-tuning  
**Key advance:** Transfer learning from ImageNet + MC-Dropout (20 passes) for uncertainty estimation  
**Uncertainty:** Shannon entropy over MC-Dropout predictions + prediction variance  
**Why ResNet:** Skip connections preserve low-level texture (critical for tumor margins); optimal depth for ~5600 images; efficient on CPU

---

## How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image pillow scipy
```

### 2. Download dataset
Place the Kaggle dataset so that:
```
data/Training/glioma/      (contains .jpg files)
data/Training/meningioma/
data/Training/notumor/
data/Training/pituitary/
```

### 3. Run scripts in order
```bash
# Step 1: EDA
python src/01_eda.py

# Step 2: Baseline ML (Random Forest)
python src/02_baseline_ml.py

# Step 3: Advanced ML (SVM + PCA)
python src/03_advanced_ml.py

# Step 4: Deep Learning (ResNet-18 + MC-Dropout)
python src/04_deep_learning.py
```

### 4. View Dashboard
Open `index.html` in a web browser to explore:
- EDA visualizations
- BML, AML, and DL model results
- Live prediction with uncertainty estimation

All outputs are saved as PNG files. No binary model files are generated.

---

## Output Files

### EDA (`outputs/eda/`)
| File | Description |
|---|---|
| `class_distribution.png` | Class counts + proportions |
| `image_resolution.png` | Width, height, file size distributions |
| `pixel_intensity_stats.png` | Per-class mean, std, skewness, kurtosis |
| `intensity_distribution.png` | Overlaid intensity histograms |
| `sample_grid.png` | Example MRI per class |
| `correlation_heatmap.png` | Metadata correlation |
| `eda_quality_report.png` | Issues found + actions taken |

### BML (`outputs/bml/`)
| File | Description |
|---|---|
| `bml_hog_visualization.png` | HOG edge maps per class |
| `bml_lbp_visualization.png` | LBP patterns + histograms |
| `bml_feature_summary.png` | All features per class |
| `bml_confusion_matrix.png` | RF confusion matrix |
| `bml_per_class_report.png` | Precision / Recall / F1 per class |
| `bml_confidence_distribution.png` | Correct vs wrong confidence |
| `bml_entropy_analysis.png` | Entropy per class |
| `bml_calibration_curve.png` | Reliability diagram |
| `bml_uncertainty_threshold.png` | Threshold analysis |
| `bml_feature_importance.png` | Top 30 Gini importances |

### AML (`outputs/aml/`)
| File | Description |
|---|---|
| `aml_pca_variance.png` | Cumulative variance + scree plot |
| `aml_pca_2d.png` | 2D PCA scatter by class |
| `aml_svm_confusion_matrix.png` | SVM confusion matrix |
| `aml_per_class_report.png` | Per-class metrics |
| `aml_confidence_distribution.png` | Confidence + entropy |
| `aml_entropy_analysis.png` | Entropy per class scatter |
| `aml_calibration_curve.png` | Reliability diagram (Platt) |
| `aml_uncertainty_threshold.png` | Threshold analysis |
| `aml_decision_boundary_pca.png` | SVM boundary in PCA-2D |
| `aml_bml_comparison.png` | BML vs AML bar comparison |

### DL (`outputs/dl/`)
| File | Description |
|---|---|
| `dl_training_curves.png` | Loss and accuracy over epochs |
| `dl_confusion_matrix.png` | ResNet-18 confusion matrix |
| `dl_per_class_metrics.png` | Per-class Precision/Recall/F1 |
| `dl_confidence_distribution.png` | Confidence + entropy (correct vs wrong) |
| `dl_entropy_analysis.png` | Entropy scatter by class |
| `dl_calibration_curve.png` | Reliability diagram |
| `dl_uncertainty_threshold.png` | Accuracy vs coverage trade-off |
| `dl_mc_dropout_uncertainty.png` | Std vs entropy correlation |
| `dl_ood_detection.png` | High entropy = potential OOD |
| `dl_bml_aml_dl_comparison.png` | BML vs AML vs DL comparison |
| `dl_learning_rate_schedule.png` | Cosine annealing curve |
| `dl_feature_maps.png` | First conv layer activations |

---

## Key Results (Expected vs Actual)

| Model | Accuracy | Macro F1 | Brier Score |
|---|---|---|---|
| Random Forest (BML) | ~0.85 | ~0.83 | ~0.08 |
| SVM + PCA (AML) | ~0.87 | ~0.85 | ~0.09 |
| ResNet-18 (DL) | **0.98** | **0.98** | **0.01** |

**Key findings:**
1. Wrong predictions have significantly higher Shannon entropy than correct ones — the uncertainty signal is informative and validates the clinical safety flagging mechanism.
2. DL with MC-Dropout achieves SOTA performance with best calibration (Brier: 0.01)
3. Uncertainty metrics (entropy, MC-std) reliably identify incorrect predictions — enabling "knowing when we don't know"

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
