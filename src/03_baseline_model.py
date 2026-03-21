"""
Models: Random Forest · Calibrated SVM · Logistic Regression
Uncertainty: Prediction entropy · Max confidence · Brier score
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
import joblib
import warnings
warnings.filterwarnings("ignore")

FEATURE_DIR = Path("outputs/features")
OUTPUT_DIR  = Path("outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED = 42

X = np.load(FEATURE_DIR / "X_features.npy")
y = np.load(FEATURE_DIR / "y_labels.npy")
print(f"Loaded: X={X.shape}, y={y.shape}")

# Remove near-zero variance features (they carry no signal)
selector = VarianceThreshold(threshold=1e-6)
X_sel = selector.fit_transform(X)
print(f"After variance threshold: {X.shape[1]} → {X_sel.shape[1]} features")

# Z-score normalization (required for SVM and LR; harmless for RF)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Stratified split preserves class proportions in every subset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

joblib.dump(selector, OUTPUT_DIR / "variance_selector.pkl")
joblib.dump(scaler,   OUTPUT_DIR / "scaler.pkl")

def prediction_entropy(proba):
    """
    Shannon entropy: H(p) = -sum(p * log(p))
    High H → model is uncertain (probabilities spread across classes)
    Low H  → model is confident (probability mass on one class)
    """
    proba = np.clip(proba, 1e-10, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)

def max_confidence(proba):
    """Max class probability. Low max → uncertain prediction."""
    return proba.max(axis=1)

def uncertainty_report(proba, y_true, threshold=0.7):
    """
    Split predictions into high-confidence and low-confidence groups.
    Low-confidence predictions should be flagged for radiologist review.
    """
    conf   = max_confidence(proba)
    y_pred = proba.argmax(axis=1)
    hi = conf >= threshold
    lo = ~hi
    return {
        "high_conf_count":    int(hi.sum()),
        "low_conf_count":     int(lo.sum()),
        "high_conf_accuracy": round(accuracy_score(y_true[hi], y_pred[hi]), 4) if hi.sum() else 0,
        "low_conf_accuracy":  round(accuracy_score(y_true[lo], y_pred[lo]), 4) if lo.sum() else 0,
        "mean_confidence":    round(conf.mean(), 4),
        "mean_entropy":       round(prediction_entropy(proba).mean(), 4),
    }

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        class_weight='balanced',   # handles class imbalance
        random_state=SEED, n_jobs=-1
    ),
    "SVM (Calibrated)": CalibratedClassifierCV(
        SVC(kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', random_state=SEED),
        method='sigmoid',   # Platt scaling → calibrated probabilities
        cv=3
    ),
    "Logistic Regression": LogisticRegression(
    C=1.0, max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=SEED
    ),
}

results = {}

for name, model in models.items():
    print(f"\n── Training: {name} ──")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report  = classification_report(y_test, y_pred,
                                    target_names=CLASSES, output_dict=True)
    brier   = np.mean([
        brier_score_loss((y_test == i).astype(int), y_proba[:, i])
        for i in range(len(CLASSES))
    ])

    results[name] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "macro_f1":  round(report["macro avg"]["f1-score"], 4),
        "brier":     round(brier, 4),
        "unc":       uncertainty_report(y_proba, y_test),
        "model":     model,
        "y_proba":   y_proba,
        "y_pred":    y_pred,
    }

    print(f"  Accuracy : {results[name]['accuracy']}")
    print(f"  Macro F1 : {results[name]['macro_f1']}")
    print(f"  Brier    : {results[name]['brier']}  (calibration quality)")
    print(f"  Uncertainty: {results[name]['unc']}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    ax.set_title(f"{name}\nAcc={res['accuracy']}, F1={res['macro_f1']}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Prediction Confidence Distributions", fontsize=14, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    conf    = max_confidence(res["y_proba"])
    correct = (res["y_pred"] == y_test)
    ax.hist(conf[correct],  bins=20, alpha=0.7, color='steelblue',
            label='Correct', density=True)
    ax.hist(conf[~correct], bins=20, alpha=0.7, color='coral',
            label='Wrong', density=True)
    ax.axvline(0.7, color='black', linestyle='--', label='Threshold=0.7')
    ax.set_title(name); ax.set_xlabel("Max Confidence"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Reliability Diagrams — Close to diagonal = well calibrated",
             fontsize=13, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    true_b = (y_test == 1).astype(int)           # meningioma (hardest class)
    frac_pos, mean_pred = calibration_curve(true_b, res["y_proba"][:, 1], n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', color='steelblue', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_title(f"{name}  (Brier={res['brier']})")
    ax.set_xlabel("Mean Predicted Prob"); ax.set_ylabel("Fraction Positive")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_curves.png", dpi=150, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Prediction Entropy (Uncertainty) Analysis", fontsize=14, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    entropy = prediction_entropy(res["y_proba"])
    correct = (res["y_pred"] == y_test)
    ax.hist(entropy[correct],  bins=20, alpha=0.7, color='mediumseagreen',
            label='Correct', density=True)
    ax.hist(entropy[~correct], bins=20, alpha=0.7, color='tomato',
            label='Wrong', density=True)
    ax.set_title(f"{name}  H̄={entropy.mean():.3f}")
    ax.set_xlabel("Entropy H(p)"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "entropy_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

summary = pd.DataFrame([{
    "Model":             name,
    "Accuracy":          r["accuracy"],
    "Macro F1":          r["macro_f1"],
    "Brier Score":       r["brier"],
    "Mean Confidence":   r["unc"]["mean_confidence"],
    "Mean Entropy":      r["unc"]["mean_entropy"],
    "High-Conf Acc":     r["unc"]["high_conf_accuracy"],
    "Low-Conf Acc":      r["unc"]["low_conf_accuracy"],
} for name, r in results.items()])

summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\n── Model Comparison ──")
print(summary.to_string(index=False))

best = max(results, key=lambda k: results[k]["macro_f1"])
joblib.dump(results[best]["model"], OUTPUT_DIR / "best_model.pkl")
print(f"\n[SAVED] Best model: {best} → outputs/models/best_model.pkl")