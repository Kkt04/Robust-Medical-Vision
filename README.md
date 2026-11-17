<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🧠 Robust Medical Vision</h1>

<p align="center">
  <strong>Brain Tumor MRI Classification with Uncertainty-Aware Neuro-Symbolic AI</strong>
</p>

<p align="center">
  <em>A multi-phase ML/DL pipeline that classifies brain tumors from MRI scans and <b>knows when it doesn't know</b> — flagging uncertain predictions for radiologist review.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-results">Results</a> •
  <a href="#-models">Models</a> •
  <a href="#-interactive-dashboard">Dashboard</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## 🎯 Problem

Clinical AI systems must know when they don't know. Standard classifiers produce a prediction regardless of internal confidence — in medical imaging, this is **unsafe**. A model that confidently misclassifies a glioma as benign could delay critical treatment.

**Robust Medical Vision** solves this by building a brain tumor MRI classifier that:

- ✅ Achieves **99% classification accuracy** across 4 tumor types
- ✅ **Estimates uncertainty** via calibrated probabilities and entropy
- ✅ **Flags low-confidence cases** for radiologist review (second-opinion protocol)
- ✅ Provides an **interactive dashboard** for real-time inference with uncertainty visualization

---

## 🏆 Key Results

<table>
<tr>
<td>

| Model | Accuracy | Macro F1 | Brier Score |
|:------|:--------:|:--------:|:-----------:|
| Random Forest (BML) | 85.0% | 0.83 | 0.080 |
| SVM + PCA (AML) | 87.0% | 0.85 | 0.090 |
| ResNet-18 + MC-Dropout (DL) | 92.0% | 0.91 | 0.060 |
| **Hybrid Ensemble (Ours)** | **99.0%** | **0.99** | **0.008** |

</td>
<td>

### Why the Hybrid Wins
- 🥇 **+14%** accuracy over traditional ML
- 🥇 **+7%** accuracy over standalone deep learning
- 🥇 **7.5×** better calibration than BML (Brier: 0.008)
- 🥇 Every component proven necessary via ablation study

</td>
</tr>
</table>

> **Ablation Study** — Removing TTA: −1.5% · Removing fine-tuning: −1.8% · Using ResNet-18 instead of 50: −7.0%

---

## 📂 Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** — Kaggle (Nickparvar, 2021)

| Class | Count | Description |
|:------|:-----:|:------------|
| Glioma | ~1,621 | Malignant; irregular ragged boundaries |
| Meningioma | ~1,645 | Benign; smooth well-defined edges |
| No Tumor | ~2,000 | Healthy brain tissue |
| Pituitary | ~1,757 | Pituitary gland tumor |

**Total:** ~7,023 MRI images across 4 classes

---

## 🔬 Models

### Pipeline Overview

```
                    ┌──────────────────────────────────────────────────────┐
                    │              Robust Medical Vision Pipeline          │
                    └──────────────────────────────────────────────────────┘

   Phase 1: Traditional ML              Phase 2: Deep Learning       Phase 3: Neuro-Symbolic Hybrid
  ┌─────────────────────────┐          ┌────────────────────┐       ┌──────────────────────────────┐
  │  BML: Random Forest     │          │  ResNet-18         │       │  ResNet-50 (fine-tuned)       │
  │  ├─ HOG + LBP + GLCM   │          │  ├─ Transfer Learn │       │        ↓                     │
  │  ├─ 370-dim features    │          │  ├─ MC-Dropout     │       │  2048-dim features           │
  │  └─ Tree vote entropy   │          │  └─ 20 MC passes   │       │        ↓                     │
  │                         │          │                    │       │  PCA (98% variance)           │
  │  AML: SVM + PCA         │          │  Accuracy: 92%     │       │        ↓                     │
  │  ├─ RBF kernel, C=10    │          │  Uncertainty: MC    │       │  TTA (5 views)               │
  │  ├─ Platt calibration   │          │  Dropout entropy   │       │        ↓                     │
  │  └─ 95% variance PCA    │          └────────────────────┘       │  Calibrated SVM Ensemble     │
  │                         │                                       │  ├─ SVM (w≈0.62)             │
  │  Accuracy: 85-87%       │                                       │  ├─ RF  (w≈0.22)             │
  │  Uncertainty: Shannon   │                                       │  └─ GB  (w≈0.16)             │
  │  entropy + calibration  │                                       │                              │
  └─────────────────────────┘                                       │  Accuracy: 99% 🏆            │
                                                                    │  Brier Score: 0.008          │
                                                                    └──────────────────────────────┘
```

### Phase 1 — Baseline ML (`02_baseline_ml.py`)

| Aspect | Details |
|:-------|:--------|
| **Model** | Random Forest (200 trees, balanced class weights) |
| **Features** | HOG (~324d) + LBP (26d) + GLCM (10d) + Intensity Stats (10d) ≈ 370d |
| **Uncertainty** | Native `predict_proba()` from tree vote distribution + Shannon entropy |
| **Accuracy** | ~85% |

### Phase 1 — Advanced ML (`03_advanced_ml.py`)

| Aspect | Details |
|:-------|:--------|
| **Model** | SVM (RBF kernel, C=10) with Platt Scaling |
| **Key Advance** | PCA reduces ~370d → N dimensions (95% variance retained) |
| **Uncertainty** | Platt-calibrated probabilities + Shannon entropy + reliability diagrams |
| **Accuracy** | ~87% |

### Phase 2 — Deep Learning (`04_deep_learning.py`)

| Aspect | Details |
|:-------|:--------|
| **Model** | ResNet-18 (ImageNet pretrained, layer3-4 fine-tuned) |
| **Key Advance** | Transfer learning + MC-Dropout (20 stochastic forward passes) |
| **Uncertainty** | Shannon entropy over MC-Dropout predictions + prediction variance |
| **Accuracy** | ~92% |

### Phase 3 — Hybrid Neuro-Symbolic (`05_hybrid.py`) 🏆

| Aspect | Details |
|:-------|:--------|
| **Architecture** | ResNet-50 → 2048-dim features → PCA (98% var) → Calibrated SVM Ensemble + TTA |
| **Innovation** | DL learns rich features; ML ensemble classifies with validation-optimized weights |
| **TTA** | 5 views: original, center-crop, h-flip, +10° rotation, −10° rotation |
| **Ensemble** | SVM (w≈0.62) + Random Forest (w≈0.22) + Gradient Boosting (w≈0.16) |
| **Accuracy** | **99.0%** — exceeds all individual models |
| **Calibration** | Brier Score 0.008 — best across all phases |

---

## 🖥️ Interactive Dashboard

The project includes a **live inference dashboard** (`index.html`) powered by a Flask backend:

- 📊 **EDA Explorer** — class distributions, intensity stats, resolution analysis
- 📈 **Model Results** — confusion matrices, calibration curves, feature visualizations for all 4 models
- 🔮 **Live Prediction** — upload any brain MRI → get classification + uncertainty + confidence scores
- ⚠️ **Uncertainty Alerts** — low-confidence predictions are flagged for radiologist review

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- ~2 GB disk space for dataset

### 1. Clone & Install

```bash
git clone https://github.com/Kkt04/Robust-Medical-Vision.git
cd Robust-Medical-Vision
pip install -r requirements.txt
```

### 2. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place so the structure is:

```
data/Training/
├── glioma/        # ~1,621 .jpg files
├── meningioma/    # ~1,645 .jpg files
├── notumor/       # ~2,000 .jpg files
└── pituitary/     # ~1,757 .jpg files
```

### 3. Run the Pipeline

```bash
# Step 1: Exploratory Data Analysis
python3 src/01_eda.py

# Step 2: Baseline ML (Random Forest + HOG/LBP/GLCM)
python3 src/02_baseline_ml.py

# Step 3: Advanced ML (SVM + PCA + Platt Calibration)
python3 src/03_advanced_ml.py

# Step 4: Deep Learning (ResNet-18 + MC-Dropout)
python3 src/04_deep_learning.py

# Step 5: Hybrid Neuro-Symbolic (ResNet-50 → SVM Ensemble + TTA)
python3 src/05_hybrid.py
```

### 4. Launch the Dashboard

```bash
# Start the Flask inference server
python3 server.py
# Server runs on http://localhost:5001

# Then open index.html in your browser
open index.html
```

---

## 🔑 Uncertainty Estimation

A core design principle — **every model phase implements uncertainty estimation:**

| Phase | Method | How It Works |
|:------|:-------|:-------------|
| **BML** | Tree Vote Distribution | 200 decision trees vote; Shannon entropy over vote proportions |
| **AML** | Platt Scaling | SVM decision values → sigmoid-calibrated probabilities → entropy |
| **DL** | MC-Dropout | 20 stochastic forward passes; variance + entropy over predictions |
| **Hybrid** | Calibrated Ensemble | Weighted average of 3 calibrated classifiers → entropy |

### Why This Matters for Clinical AI

```
  High Confidence (entropy < 0.1)          Low Confidence (entropy > 0.5)
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │   ✅ AUTO-REPORT         │              │   ⚠️  FLAG FOR REVIEW    │
  │   "Pituitary — 98.7%"   │              │   "Glioma? — 54.2%"     │
  │                         │              │   → Radiologist review   │
  │   ~92% of predictions   │              │   ~8% of predictions     │
  │   Accuracy: 99.5%       │              │   Avoids misdiagnosis    │
  └─────────────────────────┘              └─────────────────────────┘
```

---

## 📁 Project Structure

```
Robust-Medical-Vision/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── server.py                          # Flask inference server (MC-Dropout)
├── index.html                         # Interactive dashboard
├── styles.css                         # Dashboard styling
├── app.js                             # Dashboard logic
│
├── src/
│   ├── 01_eda.py                      # EDA & data quality analysis
│   ├── 02_baseline_ml.py              # BML: Random Forest + uncertainty
│   ├── 03_advanced_ml.py              # AML: SVM + PCA + Platt calibration
│   ├── 04_deep_learning.py            # DL: ResNet-18 + MC-Dropout
│   ├── 05_hybrid.py                   # Hybrid: ResNet-50 → SVM Ensemble
│   ├── 05_hybrid_diagram.py           # Architecture diagram generator
│   └── 06_hybrid_report.py            # Report generator
│
├── data/
│   └── Training/
│       ├── glioma/                    # ~1,621 MRI images
│       ├── meningioma/                # ~1,645 MRI images
│       ├── notumor/                   # ~2,000 MRI images
│       └── pituitary/                 # ~1,757 MRI images
│
├── outputs/                           # All generated visualizations
│   ├── eda/                           # 7 EDA plots
│   ├── bml/                           # 10 BML analysis plots
│   ├── aml/                           # 10 AML analysis plots
│   ├── dl/                            # 12 DL analysis plots + model weights
│   └── hybrid/                        # 4 Hybrid plots + architecture diagram
│
├── Documentations/
│   ├── BML_DOCUMENTATION.md           # Baseline ML technical details
│   ├── AML_DOCUMENTATION.md           # Advanced ML technical details
│   ├── DL_DOCUMENTATION.md            # Deep Learning technical details
│   ├── HYBRID_DOCUMENTATION.md        # Hybrid model technical details
│   ├── MODEL_COMPARISON_DOCUMENTATION.md
│   └── RUN_INSTRUCTIONS.md            # Detailed run guide
│
└── Reports/
    ├── Phase1_Report_BML_AML.pdf      # Traditional ML report
    ├── Phase2_Report_DLModel.pdf      # Deep Learning report
    └── Phase3_Report_HybridModel.pdf  # Hybrid model report
```

---

## 📊 Output Visualizations

<details>
<summary><b>EDA Outputs</b> (7 plots)</summary>

| File | Description |
|:-----|:------------|
| `class_distribution.png` | Class counts and proportions |
| `image_resolution.png` | Width, height, file size distributions |
| `pixel_intensity_stats.png` | Per-class mean, std, skewness, kurtosis |
| `intensity_distribution.png` | Overlaid intensity histograms |
| `sample_grid.png` | Example MRI from each class |
| `correlation_heatmap.png` | Metadata correlation matrix |
| `eda_quality_report.png` | Data quality issues and fixes |

</details>

<details>
<summary><b>Baseline ML Outputs</b> (10 plots)</summary>

| File | Description |
|:-----|:------------|
| `bml_hog_visualization.png` | HOG edge maps per class |
| `bml_lbp_visualization.png` | LBP patterns and histograms |
| `bml_feature_summary.png` | All features visualized per class |
| `bml_confusion_matrix.png` | Random Forest confusion matrix |
| `bml_per_class_report.png` | Precision, Recall, F1 per class |
| `bml_confidence_distribution.png` | Confidence: correct vs wrong |
| `bml_entropy_analysis.png` | Shannon entropy per class |
| `bml_calibration_curve.png` | Reliability diagram |
| `bml_uncertainty_threshold.png` | Accuracy vs coverage trade-off |
| `bml_feature_importance.png` | Top 30 Gini importances |

</details>

<details>
<summary><b>Advanced ML Outputs</b> (10 plots)</summary>

| File | Description |
|:-----|:------------|
| `aml_pca_variance.png` | Cumulative variance and scree plot |
| `aml_pca_2d.png` | 2D PCA projection colored by class |
| `aml_svm_confusion_matrix.png` | SVM confusion matrix |
| `aml_per_class_report.png` | Per-class metrics |
| `aml_confidence_distribution.png` | Confidence and entropy distributions |
| `aml_entropy_analysis.png` | Entropy scatter plot by class |
| `aml_calibration_curve.png` | Reliability diagram (Platt calibration) |
| `aml_uncertainty_threshold.png` | Threshold analysis |
| `aml_decision_boundary_pca.png` | SVM decision boundary in PCA-2D |
| `aml_bml_comparison.png` | BML vs AML comparison |

</details>

<details>
<summary><b>Deep Learning Outputs</b> (12 plots)</summary>

| File | Description |
|:-----|:------------|
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

</details>

<details>
<summary><b>Hybrid Outputs</b> (4 plots)</summary>

| File | Description |
|:-----|:------------|
| `architecture_diagram.png` | Publication-ready hybrid architecture |
| `hybrid_comparison.png` | Full model comparison (BML vs AML vs DL vs Hybrid) |
| `hybrid_confusion_matrix.png` | Hybrid model confusion matrix |
| `hybrid_per_class.png` | Per-class Precision, Recall, F1 |

</details>

---

## 🧪 Technical Details

### Inference Server API

The Flask server (`server.py`) exposes a REST API for real-time inference:

```bash
# Health check
curl http://localhost:5001/health

# Predict (base64-encoded MRI image)
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

**Response:**
```json
{
  "predictedClass": "Pituitary",
  "confidence": 0.987,
  "entropy": 0.042,
  "std": 0.008,
  "probabilities": [0.003, 0.005, 0.005, 0.987]
}
```

### Dependencies

```
torch>=2.0.0        torchvision>=0.15.0     numpy>=1.21.0
pandas>=1.3.0       matplotlib>=3.4.0       seaborn>=0.11.0
scikit-learn>=1.0.0  Pillow>=8.0.0          scipy>=1.7.0
scikit-image>=0.19.0 Flask>=2.0.0           Flask-CORS>=3.0.0
```

---

## 📚 References

1. Dalal & Triggs (2005) — *Histograms of Oriented Gradients for Human Detection.* IEEE CVPR
2. Ojala et al. (2002) — *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP.* IEEE TPAMI
3. Haralick et al. (1973) — *Textural Features for Image Classification.* IEEE SMC
4. Guo et al. (2017) — *On Calibration of Modern Neural Networks.* ICML
5. Platt (1999) — *Probabilistic Outputs for Support Vector Machines.* MIT Press
6. Lakshminarayanan et al. (2017) — *Simple and Scalable Predictive Uncertainty Estimation.* NeurIPS
7. Kompa et al. (2021) — *Second Opinion Needed: Communicating Uncertainty in Medical ML.* NPJ Digital Medicine
8. Nickparvar (2021) — *Brain Tumor MRI Dataset.* Kaggle

---

<p align="center">
  Built by <strong>Kalash Kumari Thakur</strong>
</p>
