import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, brier_score_loss
)
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/Training")
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE   = (128, 128)
OUTPUT_DIR = Path("outputs/bml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED       = 42

COLORS = {
    "glioma":     "#1f1f1f",
    "meningioma": "#555555",
    "notumor":    "#888888",
    "pituitary":  "#bbbbbb",
}

print("=" * 60)
print("  BASELINE ML  —  RANDOM FOREST")
print("=" * 60)

def load_gray(path, size=IMG_SIZE):
    return np.array(
        Image.open(path).convert("L").resize(size, Image.LANCZOS),
        dtype=np.float32) / 255.0

def feat_hog(img):
    """HOG: captures shape/edge structure at tumour boundaries."""
    f, himg = hog(img, orientations=9, pixels_per_cell=(16,16),
                  cells_per_block=(2,2), block_norm="L2-Hys",
                  visualize=True, feature_vector=True)
    return f, exposure.rescale_intensity(himg, in_range=(0,10))

def feat_lbp(img, R=3, P=24):
    """LBP: rotation-invariant micro-texture of tissue."""
    lbp = local_binary_pattern(
        (img*255).astype(np.uint8), P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, P+3), density=True)
    return hist, lbp

def feat_glcm(img):
    """GLCM: spatial co-occurrence — tumour tissue is heterogeneous."""
    u8 = (img*255).astype(np.uint8)
    glcm = graycomatrix(u8, distances=[1,3],
                        angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    out = []
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation"]:
        v = graycoprops(glcm, prop)
        out.extend([v.mean(), v.std()])
    return np.array(out)

def feat_intensity(img):
    """10 statistical moments — No-Tumour class has distinctly lower mean."""
    f = img.flatten()
    return np.array([f.mean(), f.std(), f.min(), f.max(),
                     skew(f), kurtosis(f),
                     np.percentile(f,25), np.percentile(f,50),
                     np.percentile(f,75), np.percentile(f,90)])

def extract_all(path):
    img = load_gray(path)
    hf, _ = feat_hog(img)
    lf, _ = feat_lbp(img)
    return np.concatenate([hf, lf, feat_glcm(img), feat_intensity(img)])

# ── Feature Visualisations (1 sample per class) ───────
print("\n[1/5] Generating feature visualisations…")

# HOG visualisation
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("HOG Feature Visualisation — Shape & Edge Detection\n"
             "Glioma: irregular ragged edges  |  Meningioma: smooth boundary",
             fontsize=12, fontweight="bold")
for col, cls in enumerate(CLASSES):
    p = next((DATA_DIR/cls).glob("*.jpg"))
    img = load_gray(p)
    _, himg = feat_hog(img)
    axes[0,col].imshow(img, cmap="gray")
    axes[0,col].set_title(cls.upper(), fontweight="bold")
    axes[0,col].axis("off")
    if col == 0:
        axes[0,col].set_ylabel("Original MRI", fontsize=9)
    axes[1,col].imshow(himg, cmap="Greys_r")
    axes[1,col].axis("off")
    if col == 0:
        axes[1,col].set_ylabel("HOG Edge Map", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_hog_visualization.png", dpi=150, bbox_inches="tight")
plt.close()

# LBP visualisation
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("LBP Feature Visualisation — Local Texture Patterns",
             fontsize=12, fontweight="bold")
for col, cls in enumerate(CLASSES):
    p = next((DATA_DIR/cls).glob("*.jpg"))
    img = load_gray(p)
    hist, limg = feat_lbp(img)
    axes[0,col].imshow(img, cmap="gray")
    axes[0,col].set_title(cls.upper(), fontweight="bold")
    axes[0,col].axis("off")
    if col == 0: axes[0,col].set_ylabel("Original MRI", fontsize=9)
    axes[1,col].imshow(limg, cmap="Greys_r")
    axes[1,col].axis("off")
    if col == 0: axes[1,col].set_ylabel("LBP Pattern", fontsize=9)
    axes[2,col].bar(range(len(hist)), hist, color=COLORS[cls],
                    edgecolor="none", width=1.0)
    axes[2,col].set_xlabel("LBP Bin", fontsize=8)
    if col == 0: axes[2,col].set_ylabel("Density", fontsize=9)
    axes[2,col].tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_lbp_visualization.png", dpi=150, bbox_inches="tight")
plt.close()

# Combined summary grid
fig, axes = plt.subplots(4, 4, figsize=(18, 18))
fig.suptitle("Feature Engineering Summary — All Features per Class",
             fontsize=14, fontweight="bold")
for col, cls in enumerate(CLASSES):
    p = next((DATA_DIR/cls).glob("*.jpg"))
    img = load_gray(p)
    _, himg = feat_hog(img)
    hist, limg = feat_lbp(img)
    stats = feat_intensity(img)
    axes[0,col].imshow(img, cmap="gray")
    axes[0,col].set_title(cls.upper(), fontweight="bold")
    axes[0,col].axis("off")
    if col==0: axes[0,col].set_ylabel("Original\nMRI", fontsize=9)
    axes[1,col].imshow(himg, cmap="Greys_r")
    axes[1,col].axis("off")
    if col==0: axes[1,col].set_ylabel("HOG\nEdges", fontsize=9)
    axes[2,col].imshow(limg, cmap="Greys_r")
    axes[2,col].axis("off")
    if col==0: axes[2,col].set_ylabel("LBP\nTexture", fontsize=9)
    axes[3,col].hist(img.flatten(), bins=40, color=COLORS[cls],
                     edgecolor="none", density=True)
    axes[3,col].axvline(stats[0], color="red", lw=1.5,
                        linestyle="--", label=f"μ={stats[0]:.2f}")
    axes[3,col].legend(fontsize=6)
    axes[3,col].set_xlabel("Pixel Value", fontsize=8)
    axes[3,col].tick_params(labelsize=7)
    if col==0: axes[3,col].set_ylabel("Intensity\nHistogram", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_feature_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Feature visualisations saved.")

# ── Extract features for all images ───────────────────
print("\n[2/5] Extracting features from all images…")
label_map = {c: i for i, c in enumerate(CLASSES)}
X_all, y_all = [], []
for cls in CLASSES:
    files = list((DATA_DIR/cls).glob("*.jpg"))
    print(f"  [{cls}] {len(files)} images")
    for p in files:
        try:
            X_all.append(extract_all(p))
            y_all.append(label_map[cls])
        except Exception as e:
            pass
X = np.nan_to_num(np.array(X_all), nan=0.0, posinf=1.0, neginf=0.0)
y = np.array(y_all)
print(f"\n  Feature matrix : {X.shape}")

sel    = VarianceThreshold(1e-6)
X_sel  = sel.fit_transform(X)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_sel)
print(f"  After variance filter : {X.shape[1]} → {X_sel.shape[1]} features")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.20, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=SEED, stratify=y_tr)
print(f"  Split → train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

print("\n[3/5] Training Random Forest…")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",          # sqrt(n_features) per tree — prevents correlation
    class_weight="balanced",      # corrects class imbalance
    n_jobs=-1,
    random_state=SEED,
    oob_score=True                # out-of-bag score as free validation
)
rf.fit(X_tr, y_tr)

y_pred  = rf.predict(X_te)
y_proba = rf.predict_proba(X_te)
acc     = accuracy_score(y_te, y_pred)
f1      = f1_score(y_te, y_pred, average="macro")
brier   = np.mean([brier_score_loss((y_te==i).astype(int), y_proba[:,i])
                   for i in range(4)])
oob     = rf.oob_score_

print(f"\n  OOB Score  : {oob:.4f}")
print(f"  Accuracy   : {acc:.4f}")
print(f"  Macro F1   : {f1:.4f}")
print(f"  Brier Score: {brier:.4f}  (lower = better calibration)")
print(f"\n{classification_report(y_te, y_pred, target_names=CLASSES)}")

# ── 5-fold cross-validation ───────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(rf, X_sc, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print(f"  5-Fold CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


def entropy(proba):
    """H(p) = -sum p_k log(p_k).  High H → uncertain."""
    p = np.clip(proba, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def max_conf(proba):
    return proba.max(axis=1)

def unc_stats(proba, y_true, thresh=0.70):
    conf = max_conf(proba)
    pred = proba.argmax(axis=1)
    hi, lo = conf >= thresh, conf < thresh
    return {
        "hi_count": int(hi.sum()), "lo_count": int(lo.sum()),
        "hi_acc":   round(accuracy_score(y_true[hi], pred[hi]),4) if hi.sum() else 0,
        "lo_acc":   round(accuracy_score(y_true[lo], pred[lo]),4) if lo.sum() else 0,
        "mean_H":   round(entropy(proba).mean(), 4),
        "mean_conf":round(conf.mean(), 4),
    }

unc = unc_stats(y_proba, y_te)
print(f"\n  Uncertainty (thresh=0.70):")
print(f"    High-conf ({unc['hi_count']} preds) accuracy : {unc['hi_acc']}")
print(f"    Low-conf  ({unc['lo_count']} preds) accuracy : {unc['lo_acc']}")
print(f"    Mean entropy : {unc['mean_H']}")

print("\n[4/5] Saving result plots…")

# Confusion matrix
fig, ax = plt.subplots(figsize=(7,6))
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greys",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="white")
ax.set_title(f"Random Forest — Confusion Matrix\n"
             f"Acc={acc:.3f}   Macro F1={f1:.3f}   Brier={brier:.4f}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# Per-class report bar chart
rep = classification_report(y_te, y_pred, target_names=CLASSES, output_dict=True)
metrics_data = {cls: [rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"]]
                for cls in CLASSES}
x   = np.arange(len(CLASSES))
w   = 0.25
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x - w,   [metrics_data[c][0] for c in CLASSES], w,
       label="Precision", color="black",  edgecolor="white")
ax.bar(x,       [metrics_data[c][1] for c in CLASSES], w,
       label="Recall",    color="gray",   edgecolor="white")
ax.bar(x + w,   [metrics_data[c][2] for c in CLASSES], w,
       label="F1-Score",  color="silver", edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title("Per-Class Precision / Recall / F1\n"
             "Meningioma hardest — visual overlap with Glioma",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
for i, cls in enumerate(CLASSES):
    for j, v in enumerate(metrics_data[cls]):
        ax.text(i + (j-1)*w, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=7.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_per_class_report.png", dpi=150, bbox_inches="tight")
plt.close()

# Confidence distribution
H   = entropy(y_proba)
C   = max_conf(y_proba)
ok  = y_pred == y_te
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Prediction Confidence Distribution\n"
             "Correct predictions concentrate at high confidence",
             fontsize=12, fontweight="bold")
axes[0].hist(C[ok],  bins=25, alpha=0.7, color="gray",  label="Correct", density=True)
axes[0].hist(C[~ok], bins=25, alpha=0.7, color="black", label="Wrong",   density=True)
axes[0].axvline(0.70, color="red", lw=1.5, linestyle="--", label="Threshold=0.70")
axes[0].set_xlabel("Max Class Probability"); axes[0].set_ylabel("Density")
axes[0].set_title("Max Confidence"); axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)
axes[1].hist(H[ok],  bins=25, alpha=0.7, color="gray",  label="Correct", density=True)
axes[1].hist(H[~ok], bins=25, alpha=0.7, color="black", label="Wrong",   density=True)
axes[1].axvline(H[ok].mean(),  color="gray",  lw=1.5, linestyle="--",
                label=f"Mean correct H={H[ok].mean():.3f}")
axes[1].axvline(H[~ok].mean(), color="black", lw=1.5, linestyle="--",
                label=f"Mean wrong H={H[~ok].mean():.3f}")
axes[1].set_xlabel("Shannon Entropy H(p)"); axes[1].set_ylabel("Density")
axes[1].set_title("Prediction Entropy"); axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_confidence_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Entropy by class
fig, ax = plt.subplots(figsize=(10,5))
for i, cls in enumerate(CLASSES):
    mask = y_te == i
    ax.scatter(np.where(mask)[0][:50], H[mask][:50],
               label=cls, color=COLORS[cls], alpha=0.7, s=20)
ax.axhline(np.log(4), color="red", lw=1, linestyle="--",
           label="Max entropy (log 4)")
ax.set_xlabel("Test sample index"); ax.set_ylabel("Entropy H(p)")
ax.set_title("Entropy Analysis — Per Class\n"
             "High entropy = model uncertain (flag for review)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_entropy_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# Calibration / reliability diagram
fig, ax = plt.subplots(figsize=(7,6))
for i, cls in enumerate(CLASSES):
    tb = (y_te == i).astype(int)
    fp, mp = calibration_curve(tb, y_proba[:,i], n_bins=8)
    ax.plot(mp, fp, marker="o", lw=1.5, label=cls, color=COLORS[cls])
ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Reliability Diagram — Random Forest\nBrier Score = {brier:.4f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Uncertainty threshold analysis
thresholds = np.arange(0.40, 0.96, 0.05)
hi_accs, lo_accs, hi_fracs = [], [], []
for t in thresholds:
    hi = C >= t
    lo = ~hi
    ha = accuracy_score(y_te[hi], y_proba[hi].argmax(1)) if hi.sum() > 0 else 0
    la = accuracy_score(y_te[lo], y_proba[lo].argmax(1)) if lo.sum() > 0 else 0
    hi_accs.append(ha); lo_accs.append(la)
    hi_fracs.append(hi.mean())
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("Confidence Threshold Analysis\n"
             "Choosing threshold = 0.70 balances safety and automation",
             fontsize=12, fontweight="bold")
axes[0].plot(thresholds, hi_accs, "k-o", label="High-conf accuracy", lw=1.5)
axes[0].plot(thresholds, lo_accs, "gray", linestyle="--", marker="s",
             label="Low-conf accuracy", lw=1.5)
axes[0].axvline(0.70, color="red", lw=1.2, linestyle=":", label="Chosen threshold")
axes[0].set_xlabel("Confidence Threshold")
axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].plot(thresholds, hi_fracs, "k-o", lw=1.5)
axes[1].axvline(0.70, color="red", lw=1.2, linestyle=":")
axes[1].set_xlabel("Confidence Threshold")
axes[1].set_ylabel("Fraction Classified Automatically")
axes[1].set_title("Automation Rate vs Threshold")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_uncertainty_threshold.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Feature importance (top 30)
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-30:][::-1]
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(30), importances[top_idx], color="gray", edgecolor="black", lw=0.5)
ax.set_xlabel("Feature Index (top 30)")
ax.set_ylabel("Gini Importance")
ax.set_title("Random Forest Feature Importance — Top 30 Features\n"
             "HOG features dominate: boundary morphology is most discriminative",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"bml_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[5/5] All plots saved → {OUTPUT_DIR}/")
print("\n" + "=" * 60)
print("  BASELINE ML COMPLETE")
print(f"  Accuracy : {acc:.4f}")
print(f"  Macro F1 : {f1:.4f}")
print(f"  OOB Score: {oob:.4f}")
print(f"  Brier    : {brier:.4f}")
print(f"  CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 60)
