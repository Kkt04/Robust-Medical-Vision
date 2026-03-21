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
```

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
```

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

---

## Key Results (Expected)

| Model | Accuracy | Macro F1 | Brier Score |
|---|---|---|---|
| Random Forest (BML) | ~0.85 | ~0.83 | ~0.08 |
| SVM + PCA (AML) | ~0.87 | ~0.85 | ~0.09 |

**Key finding:** Wrong predictions have significantly higher Shannon entropy than correct ones — the uncertainty signal is informative and validates the clinical safety flagging mechanism.

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