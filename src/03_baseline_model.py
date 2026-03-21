"""
Phase 1 - Baseline ML Model with Uncertainty Estimation
Project 5: Robust Medical Vision - Brain Tumor Classification

Models: Random Forest · Calibrated SVM · Logistic Regression
Uncertainty: Prediction entropy · Max confidence · Brier score
Outputs: PNG plots only (no pkl, no npy)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    brier_score_loss, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = Path("data/Training")
OUTPUT_DIR = Path("outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (128, 128)
SEED     = 42

COLORS = {
    "glioma":     "#E63946",
    "meningioma": "#457B9D",
    "notumor":    "#2A9D8F",
    "pituitary":  "#E9C46A",
}

# ─────────────────────────────────────────────
# STEP 1: FEATURE EXTRACTION (inline, no npy)
# ─────────────────────────────────────────────

def load_image_gray(filepath, size=IMG_SIZE):
    img = Image.open(filepath).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def extract_hog_features(img):
    features, _ = hog(
        img, orientations=9,
        pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualize=True, feature_vector=True
    )
    return features

def extract_lbp_features(img, radius=3, n_points=24):
    lbp = local_binary_pattern(
        (img * 255).astype(np.uint8), n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)
    return hist

def extract_intensity_stats(img):
    flat = img.flatten()
    return np.array([
        flat.mean(), flat.std(), flat.min(), flat.max(),
        skew(flat), kurtosis(flat),
        np.percentile(flat, 25), np.percentile(flat, 50),
        np.percentile(flat, 75), np.percentile(flat, 90),
    ])

def extract_glcm_features(img):
    img_uint8 = (img * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        values = graycoprops(glcm, prop)
        features.extend([values.mean(), values.std()])
    return np.array(features)

def extract_all_features(filepath):
    img = load_image_gray(filepath)
    return np.concatenate([
        extract_hog_features(img),
        extract_lbp_features(img),
        extract_intensity_stats(img),
        extract_glcm_features(img),
    ])

print("=" * 55)
print("STEP 1: Extracting features from images...")
print("=" * 55)

label_map = {cls: i for i, cls in enumerate(CLASSES)}
X_all, y_all = [], []

for cls in CLASSES:
    cls_path = DATA_DIR / cls
    if not cls_path.exists():
        print(f"  [SKIP] {cls_path} not found")
        continue
    img_files = list(cls_path.glob("*.jpg"))
    print(f"  [{cls}] {len(img_files)} images...")
    for path in img_files:
        try:
            X_all.append(extract_all_features(path))
            y_all.append(label_map[cls])
        except Exception as e:
            print(f"    [ERROR] {path.name}: {e}")

X = np.array(X_all)
y = np.array(y_all)

# Fix NaN / Inf
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
print(f"\n  Feature matrix : {X.shape}")
print(f"  Label vector   : {y.shape}")

# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Preprocessing...")
print("=" * 55)

# Remove near-zero variance features
selector = VarianceThreshold(threshold=1e-6)
X_sel    = selector.fit_transform(X)
print(f"  Variance threshold : {X.shape[1]} → {X_sel.shape[1]} features")

# Z-score normalization
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val   = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

print(f"  Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ─────────────────────────────────────────────
# STEP 3: UNCERTAINTY FUNCTIONS
# ─────────────────────────────────────────────

def prediction_entropy(proba):
    """
    Shannon entropy H(p) = -sum(p * log(p))
    High H → model uncertain | Low H → model confident
    """
    proba = np.clip(proba, 1e-10, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)

def max_confidence(proba):
    """Max class probability — low max means uncertain."""
    return proba.max(axis=1)

def uncertainty_report(proba, y_true, threshold=0.7):
    """
    Split into high-confidence (safe) vs low-confidence (flag for review).
    Key clinical safety mechanism.
    """
    conf   = max_confidence(proba)
    y_pred = proba.argmax(axis=1)
    hi     = conf >= threshold
    lo     = ~hi
    return {
        "high_conf_count":    int(hi.sum()),
        "low_conf_count":     int(lo.sum()),
        "high_conf_accuracy": round(accuracy_score(y_true[hi], y_pred[hi]), 4) if hi.sum() else 0,
        "low_conf_accuracy":  round(accuracy_score(y_true[lo], y_pred[lo]), 4) if lo.sum() else 0,
        "mean_confidence":    round(conf.mean(), 4),
        "mean_entropy":       round(prediction_entropy(proba).mean(), 4),
    }

# ─────────────────────────────────────────────
# STEP 4: DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ),
    "SVM (Calibrated)": CalibratedClassifierCV(
        SVC(kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', random_state=SEED),
        method='sigmoid',    # Platt scaling → calibrated probabilities
        cv=3
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=SEED
    ),
}

# ─────────────────────────────────────────────
# STEP 5: TRAIN & EVALUATE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Training models...")
print("=" * 55)

results = {}

for name, model in models.items():
    print(f"\n── Training: {name} ──")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report  = classification_report(
        y_test, y_pred, target_names=CLASSES, output_dict=True)
    brier   = np.mean([
        brier_score_loss((y_test == i).astype(int), y_proba[:, i])
        for i in range(len(CLASSES))
    ])

    results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "brier":    round(brier, 4),
        "unc":      uncertainty_report(y_proba, y_test),
        "y_proba":  y_proba,
        "y_pred":   y_pred,
    }

    print(f"  Accuracy : {results[name]['accuracy']}")
    print(f"  Macro F1 : {results[name]['macro_f1']}")
    print(f"  Brier    : {results[name]['brier']}")
    print(f"  Uncertainty: {results[name]['unc']}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

# ─────────────────────────────────────────────
# PNG 1: CONFUSION MATRICES
# ─────────────────────────────────────────────
print("\nSaving PNG outputs...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues', linewidths=0.5, linecolor='gray')
    ax.set_title(f"{name}\nAcc={res['accuracy']}  F1={res['macro_f1']}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] confusion_matrices.png")

# ─────────────────────────────────────────────
# PNG 2: CONFIDENCE DISTRIBUTIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Prediction Confidence Distributions\n"
             "Correct vs Wrong predictions — threshold at 0.70",
             fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    conf    = max_confidence(res["y_proba"])
    correct = (res["y_pred"] == y_test)

    ax.hist(conf[correct],  bins=20, alpha=0.7, color='steelblue',
            label=f'Correct (n={correct.sum()})', density=True)
    ax.hist(conf[~correct], bins=20, alpha=0.7, color='coral',
            label=f'Wrong (n={(~correct).sum()})', density=True)
    ax.axvline(0.7, color='black', linestyle='--',
               linewidth=1.5, label='Threshold=0.70')
    ax.set_title(f"{name}\nMean conf={res['unc']['mean_confidence']}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Max Class Probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] confidence_distributions.png")

# ─────────────────────────────────────────────
# PNG 3: RELIABILITY DIAGRAMS (Calibration)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Reliability Diagrams — Calibration Check\n"
             "Closer to diagonal = better calibrated (Meningioma class shown)",
             fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    true_b = (y_test == 1).astype(int)    # meningioma = hardest class
    frac_pos, mean_pred = calibration_curve(
        true_b, res["y_proba"][:, 1], n_bins=10)

    ax.plot(mean_pred, frac_pos, marker='o', linewidth=2,
            color='steelblue', markersize=6, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.12, color='steelblue', label='Calibration gap')
    ax.set_title(f"{name}\nBrier Score = {res['brier']}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Mean Predicted Probability", fontsize=9)
    ax.set_ylabel("Fraction of Positives", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] calibration_curves.png")

# ─────────────────────────────────────────────
# PNG 4: ENTROPY ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Prediction Entropy (Uncertainty) Analysis\n"
             "Wrong predictions should have higher entropy — validates uncertainty signal",
             fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    entropy = prediction_entropy(res["y_proba"])
    correct = (res["y_pred"] == y_test)

    ax.hist(entropy[correct],  bins=20, alpha=0.7, color='mediumseagreen',
            label=f'Correct (n={correct.sum()})', density=True)
    ax.hist(entropy[~correct], bins=20, alpha=0.7, color='tomato',
            label=f'Wrong (n={(~correct).sum()})', density=True)
    ax.axvline(entropy[correct].mean(),  color='green',
               linestyle='--', linewidth=1.2,
               label=f'Mean correct H={entropy[correct].mean():.3f}')
    ax.axvline(entropy[~correct].mean(), color='red',
               linestyle='--', linewidth=1.2,
               label=f'Mean wrong H={entropy[~correct].mean():.3f}')
    ax.set_title(f"{name}\nOverall Mean H={entropy.mean():.3f}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Shannon Entropy H(p)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "entropy_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] entropy_analysis.png")

# ─────────────────────────────────────────────
# PNG 5: MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────
model_names = list(results.keys())
metrics     = {
    "Accuracy":       [r["accuracy"]            for r in results.values()],
    "Macro F1":       [r["macro_f1"]             for r in results.values()],
    "1 - Brier":      [round(1 - r["brier"], 4)  for r in results.values()],
    "Mean Confidence":[r["unc"]["mean_confidence"] for r in results.values()],
}

x     = np.arange(len(model_names))
width = 0.2
bar_colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#E63946"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight='bold')

for i, (metric, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + i * width, vals, width,
                  label=metric, color=bar_colors[i],
                  edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] model_comparison_chart.png")

# ─────────────────────────────────────────────
# PNG 6: PER-CLASS F1 HEATMAP
# ─────────────────────────────────────────────
f1_data = []
for name, res in results.items():
    report = classification_report(
        y_test, res["y_pred"], target_names=CLASSES, output_dict=True)
    row = [round(report[cls]["f1-score"], 3) for cls in CLASSES]
    f1_data.append(row)

f1_df = pd.DataFrame(f1_data, index=model_names, columns=CLASSES)

fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle("Per-Class F1 Score Heatmap\n"
             "Meningioma consistently hardest — expected due to visual overlap with Glioma",
             fontsize=12, fontweight='bold')

sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax, vmin=0.5, vmax=1.0, linewidths=0.5,
            linecolor='white', annot_kws={"size": 11, "weight": "bold"})
ax.set_xlabel("Tumor Class", fontsize=10)
ax.set_ylabel("Model", fontsize=10)
ax.tick_params(axis='x', rotation=20, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_f1_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] per_class_f1_heatmap.png")

# ─────────────────────────────────────────────
# PNG 7: UNCERTAINTY SUMMARY DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Uncertainty Summary Dashboard\n"
             "High-confidence predictions are significantly more accurate across all models",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: High-conf vs Low-conf accuracy per model
ax_top = fig.add_subplot(gs[0, :])
bar_w  = 0.25
x_pos  = np.arange(len(model_names))

hi_accs = [r["unc"]["high_conf_accuracy"] for r in results.values()]
lo_accs = [r["unc"]["low_conf_accuracy"]  for r in results.values()]
hi_cnts = [r["unc"]["high_conf_count"]    for r in results.values()]
lo_cnts = [r["unc"]["low_conf_count"]     for r in results.values()]

b1 = ax_top.bar(x_pos - bar_w/2, hi_accs, bar_w,
                color='steelblue', label='High Confidence (≥0.70)',
                edgecolor='black', linewidth=0.5)
b2 = ax_top.bar(x_pos + bar_w/2, lo_accs, bar_w,
                color='coral', label='Low Confidence (<0.70)',
                edgecolor='black', linewidth=0.5)

for bars, cnts in [(b1, hi_cnts), (b2, lo_cnts)]:
    for bar, acc, cnt in zip(bars, [hi_accs, lo_accs][bars == b1], cnts):
        ax_top.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}\n(n={cnt})",
                    ha='center', fontsize=8, fontweight='bold')

ax_top.set_xticks(x_pos)
ax_top.set_xticklabels(model_names, fontsize=10)
ax_top.set_ylim(0, 1.18)
ax_top.set_ylabel("Accuracy", fontsize=10)
ax_top.set_title("High-Confidence vs Low-Confidence Prediction Accuracy\n"
                 "(Low-confidence = flagged for radiologist review)", fontsize=11)
ax_top.legend(fontsize=9)
ax_top.grid(axis='y', alpha=0.3)

# Row 1: Mean entropy per model
ax1 = fig.add_subplot(gs[1, 0])
entropies = [r["unc"]["mean_entropy"] for r in results.values()]
ax1.bar(model_names, entropies,
        color=["#2A9D8F", "#457B9D", "#E9C46A"],
        edgecolor='black', linewidth=0.6)
ax1.set_title("Mean Prediction Entropy", fontsize=10, fontweight='bold')
ax1.set_ylabel("H(p)")
ax1.tick_params(axis='x', rotation=15, labelsize=8)
for i, v in enumerate(entropies):
    ax1.text(i, v + 0.003, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')

# Row 1: Mean confidence per model
ax2 = fig.add_subplot(gs[1, 1])
confs = [r["unc"]["mean_confidence"] for r in results.values()]
ax2.bar(model_names, confs,
        color=["#2A9D8F", "#457B9D", "#E9C46A"],
        edgecolor='black', linewidth=0.6)
ax2.set_title("Mean Prediction Confidence", fontsize=10, fontweight='bold')
ax2.set_ylabel("Max P(class)")
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='x', rotation=15, labelsize=8)
for i, v in enumerate(confs):
    ax2.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')

# Row 1: Brier scores
ax3 = fig.add_subplot(gs[1, 2])
briers = [r["brier"] for r in results.values()]
ax3.bar(model_names, briers,
        color=["#2A9D8F", "#457B9D", "#E9C46A"],
        edgecolor='black', linewidth=0.6)
ax3.set_title("Brier Score (lower = better calibrated)", fontsize=10, fontweight='bold')
ax3.set_ylabel("Brier Score")
ax3.tick_params(axis='x', rotation=15, labelsize=8)
for i, v in enumerate(briers):
    ax3.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')

plt.savefig(OUTPUT_DIR / "uncertainty_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] uncertainty_dashboard.png")

# ─────────────────────────────────────────────
# SUMMARY CSV (still useful as text record)
# ─────────────────────────────────────────────
summary = pd.DataFrame([{
    "Model":            name,
    "Accuracy":         r["accuracy"],
    "Macro F1":         r["macro_f1"],
    "Brier Score":      r["brier"],
    "Mean Confidence":  r["unc"]["mean_confidence"],
    "Mean Entropy":     r["unc"]["mean_entropy"],
    "High-Conf Count":  r["unc"]["high_conf_count"],
    "Low-Conf Count":   r["unc"]["low_conf_count"],
    "High-Conf Acc":    r["unc"]["high_conf_accuracy"],
    "Low-Conf Acc":     r["unc"]["low_conf_accuracy"],
} for name, r in results.items()])

summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# ─────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("ALL OUTPUTS SAVED")
print("=" * 55)
print(f"Directory: {OUTPUT_DIR}/")
print()
print("  confusion_matrices.png       — 3 confusion matrices")
print("  confidence_distributions.png — correct vs wrong confidence")
print("  calibration_curves.png       — reliability diagrams")
print("  entropy_analysis.png         — entropy correct vs wrong")
print("  model_comparison_chart.png   — all metrics bar chart")
print("  per_class_f1_heatmap.png     — F1 per class per model")
print("  uncertainty_dashboard.png    — full uncertainty summary")
print("  model_comparison.csv         — numeric results table")
print("=" * 55)
print("\n── Model Comparison ──")
print(summary.to_string(index=False))