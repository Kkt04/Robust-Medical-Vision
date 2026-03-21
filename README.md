# Robust Medical Vision
## Brain Tumor MRI Classification with Uncertainty Estimation

### Problem Statement
Clinical AI must know when it does not know. This project builds a brain
tumor classifier that flags ambiguous scans for radiologist review
instead of making unsafe overconfident predictions.

### Dataset
Brain Tumor MRI Dataset — Kaggle
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 4 Classes: glioma · meningioma · notumor · pituitary
- ~7,000 Training images (JPEG)

### How to Run

#### 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image pillow scipy joblib

#### 2. Place dataset
data/Training/glioma/
data/Training/meningioma/
data/Training/notumor/
data/Training/pituitary/

#### 3. Run pipeline
python src/01_eda_preprocessing.py
python src/02_feature_engineering.py
python src/03_baseline_model.py

### Phase 1 Summary

| Component         | Implementation                               |
|-------------------|----------------------------------------------|
| EDA               | Class dist, pixel stats, skewness, QA report |
| Features          | HOG + LBP + GLCM + Intensity (~370 features) |
| Models            | Random Forest, Calibrated SVM, Logistic Reg  |
| Uncertainty       | Entropy, confidence threshold, Brier score   |
| Calibration check | Reliability diagrams per model               |

### References
1. Nickparvar (2021) — Brain Tumor MRI Dataset, Kaggle
2. Guo et al. (2017) — On Calibration of Modern Neural Networks, ICML
3. Lakshminarayanan et al. (2017) — Deep Ensembles, NeurIPS
4. Gal & Ghahramani (2016) — MC Dropout, ICML
5. Dalal & Triggs (2005) — HOG features, CVPR
