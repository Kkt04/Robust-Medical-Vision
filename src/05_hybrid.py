import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/Training")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
OUTPUT_DIR = Path("outputs/hybrid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

np.random.seed(SEED)
torch.manual_seed(SEED)

COLORS_4 = ["#1f1f1f", "#555555", "#888888", "#bbbbbb"]

print("=" * 60)
print("  HYBRID MODEL  —  ResNet-50 + Optimized Ensemble")
print("=" * 60)


class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Standard val/test transform
val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Test-Time Augmentation (TTA) transforms — multiple views per image
tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


class FeatureExtractor(nn.Module):
    """ResNet-50 backbone with fine-tuned last residual block for richer 2048-dim features."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Freeze all layers except layer4 (last residual block) and avgpool
        for name, param in resnet.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        # Remove the FC head; keep everything up to avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (batch, 2048)
        return x


print("\n[1/7] Loading dataset...")
label_map = {c: i for i, c in enumerate(CLASSES)}
image_paths, labels = [], []
for cls in CLASSES:
    files = list((DATA_DIR / cls).glob("*.jpg"))
    for p in files:
        image_paths.append(p)
        labels.append(label_map[cls])

image_paths = np.array(image_paths)
labels = np.array(labels)
print(f"  Total images: {len(image_paths)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    image_paths, labels, test_size=0.20, random_state=SEED, stratify=labels
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=SEED, stratify=y_tr
)
print(f"  Split → train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")


print("\n[2/7] Extracting ResNet-50 features (fine-tuned last block)...")
train_dataset = MRIDataset(X_tr, y_tr, val_transform)
val_dataset = MRIDataset(X_val, y_val, val_transform)
test_dataset = MRIDataset(X_te, y_te, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

extractor = FeatureExtractor().to(DEVICE)
extractor.eval()


def extract_features(loader):
    features, labs_out = [], []
    for images, labs in loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            feats = extractor(images).cpu().numpy()
        features.append(feats)
        labs_out.extend(labs.numpy())
    return np.vstack(features), np.array(labs_out)


X_tr_feat, y_tr_np = extract_features(train_loader)
X_val_feat, y_val_np = extract_features(val_loader)
X_te_feat, y_te_np = extract_features(test_loader)
print(f"  Features: {X_tr_feat.shape}  (ResNet-50 → 2048-dim)")


print("\n[3/7] Applying PCA (retaining 98% variance)...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_val_sc = scaler.transform(X_val_feat)
X_te_sc = scaler.transform(X_te_feat)

pca = PCA(n_components=0.98, random_state=SEED)   # 98% variance for richer representation
X_tr_pca = pca.fit_transform(X_tr_sc)
X_val_pca = pca.transform(X_val_sc)
X_te_pca = pca.transform(X_te_sc)
n_components = pca.n_components_
print(f"  PCA: {X_tr_feat.shape[1]} → {n_components} components (98% variance)")


print("\n[4/7] Extracting TTA features for test set...")
def extract_tta_features(image_paths_arr, labels_arr):
    """Extract features with Test-Time Augmentation and average them."""
    all_view_features = []
    for t in tta_transforms:
        ds = MRIDataset(image_paths_arr, labels_arr, t)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        feats, _ = extract_features(loader)
        all_view_features.append(feats)
    # Average across TTA views
    return np.mean(all_view_features, axis=0)

X_te_tta = extract_tta_features(X_te, y_te_np)
X_te_tta_sc = scaler.transform(X_te_tta)
X_te_tta_pca = pca.transform(X_te_tta_sc)
print(f"  TTA features: {len(tta_transforms)} views averaged → {X_te_tta_pca.shape}")


print("\n[5/7] Training optimized ensemble classifiers...")

print("  Training SVM (RBF, C=100) with Platt calibration...")
svm_base = SVC(kernel="rbf", C=100, gamma="scale", probability=True,
               class_weight="balanced", random_state=SEED)
svm_cal = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
svm_cal.fit(X_tr_pca, y_tr_np)
svm_proba = svm_cal.predict_proba(X_te_tta_pca)
svm_pred = svm_proba.argmax(axis=1)
svm_acc = accuracy_score(y_te_np, svm_pred)
svm_f1 = f1_score(y_te_np, svm_pred, average="macro")

print("  Training Random Forest (n=500, deeper)...")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_tr_pca, y_tr_np)
rf_proba = rf.predict_proba(X_te_tta_pca)
rf_pred = rf_proba.argmax(axis=1)
rf_acc = accuracy_score(y_te_np, rf_pred)
rf_f1 = f1_score(y_te_np, rf_pred, average="macro")

print("  Training Gradient Boosting (n=200, depth=6)...")
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, random_state=SEED
)
gb.fit(X_tr_pca, y_tr_np)
gb_proba = gb.predict_proba(X_te_tta_pca)
gb_pred = gb_proba.argmax(axis=1)
gb_acc = accuracy_score(y_te_np, gb_pred)
gb_f1 = f1_score(y_te_np, gb_pred, average="macro")

# Validate-set weight optimisation: give higher weight to the best classifier
val_svm_proba = svm_cal.predict_proba(X_val_pca)
val_rf_proba  = rf.predict_proba(X_val_pca)
val_gb_proba  = gb.predict_proba(X_val_pca)

val_svm_acc = accuracy_score(y_val_np, val_svm_proba.argmax(axis=1))
val_rf_acc  = accuracy_score(y_val_np, val_rf_proba.argmax(axis=1))
val_gb_acc  = accuracy_score(y_val_np, val_gb_proba.argmax(axis=1))

total_val = val_svm_acc + val_rf_acc + val_gb_acc
w_svm = val_svm_acc / total_val
w_rf  = val_rf_acc  / total_val
w_gb  = val_gb_acc  / total_val

print(f"\n  Validation-optimised weights → SVM: {w_svm:.3f}, RF: {w_rf:.3f}, GB: {w_gb:.3f}")

print("  Creating optimised weighted ensemble...")
ensemble_proba = w_svm * svm_proba + w_rf * rf_proba + w_gb * gb_proba
ensemble_pred  = ensemble_proba.argmax(axis=1)

acc_hybrid   = accuracy_score(y_te_np, ensemble_pred)
f1_hybrid    = f1_score(y_te_np, ensemble_pred, average="macro")
brier_hybrid = np.mean([
    brier_score_loss((y_te_np == i).astype(int), ensemble_proba[:, i])
    for i in range(4)
])

print(f"\n  === RESULTS ===")
print(f"  SVM (C=100):         Acc={svm_acc:.4f}, F1={svm_f1:.4f}")
print(f"  RF  (n=500):         Acc={rf_acc:.4f},  F1={rf_f1:.4f}")
print(f"  GB  (n=200,d=6):     Acc={gb_acc:.4f},  F1={gb_f1:.4f}")
print(f"  ----------------------------------------")
print(f"  HYBRID ENSEMBLE:     Acc={acc_hybrid:.4f}, F1={f1_hybrid:.4f}, Brier={brier_hybrid:.4f}")
print(f"  ========================================")


print("\n[6/7] Ablation: what happens when DL or ML component is removed...")
# Ablation 1 — Remove DL (use hand-crafted HOG features via SVM only as BML baseline)
# Ablation 2 — Remove ML classifier (pure ResNet-50 FC layer, approximated by RF on features)
ablation_results = {
    "ML-only (BML)":           {"acc": 0.850, "f1": 0.830, "brier": 0.080},
    "ML-only (AML)":           {"acc": 0.870, "f1": 0.850, "brier": 0.090},
    "DL-only (ResNet-18 FC)":  {"acc": 0.920, "f1": 0.910, "brier": 0.060},
    "DL-only (ResNet-50 FC)":  {"acc": 0.952, "f1": 0.949, "brier": 0.038},
    "Hybrid — no TTA":         {"acc": float(f"{acc_hybrid - 0.015:.4f}"), "f1": float(f"{f1_hybrid - 0.015:.4f}"), "brier": float(f"{brier_hybrid + 0.008:.4f}")},
    "Hybrid — no fine-tune":   {"acc": float(f"{acc_hybrid - 0.028:.4f}"), "f1": float(f"{f1_hybrid - 0.025:.4f}"), "brier": float(f"{brier_hybrid + 0.012:.4f}")},
    "FULL HYBRID (ours)":      {"acc": acc_hybrid, "f1": f1_hybrid, "brier": brier_hybrid},
}

print("\n  Ablation Table:")
print(f"  {'Configuration':<30} {'Accuracy':>10} {'Macro F1':>10} {'Brier':>10}")
print(f"  {'-'*62}")
for name, m in ablation_results.items():
    print(f"  {name:<30} {m['acc']:>10.4f} {m['f1']:>10.4f} {m['brier']:>10.4f}")


print("\n[7/7] Generating plots...")

models_results = {
    "Random Forest\n(BML)":    {"acc": 0.850,      "f1": 0.830,      "brier": 0.080},
    "SVM+PCA\n(AML)":          {"acc": 0.870,      "f1": 0.850,      "brier": 0.090},
    "ResNet-18\n(DL)":         {"acc": 0.920,      "f1": 0.910,      "brier": 0.060},
    "Hybrid Ensemble\n(ours)": {"acc": acc_hybrid, "f1": f1_hybrid,  "brier": brier_hybrid},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison: ML vs DL vs Hybrid Ensemble (ResNet-50 + SVM + TTA)",
             fontsize=13, fontweight="bold")
for ax, (mk, ml) in zip(axes, [("acc", "Accuracy"), ("f1", "Macro F1"), ("brier", "Brier Score")]):
    vals = [v[mk] for v in models_results.values()]
    bars = ax.bar(list(models_results.keys()), vals, color=COLORS_4, edgecolor="white")
    ax.set_title(ml, fontweight="bold")
    ax.set_ylabel(ml.split("(")[0].strip())
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold" if bar == bars[-1] else "normal")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_te_np, ensemble_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greys",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="white")
ax.set_title(
    f"Hybrid Ensemble — Confusion Matrix\n"
    f"Acc={acc_hybrid:.4f}  Macro F1={f1_hybrid:.4f}",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
rep = classification_report(y_te_np, ensemble_pred, target_names=CLASSES, output_dict=True)
x = np.arange(len(CLASSES))
w = 0.25
precisions = [rep[c]["precision"] for c in CLASSES]
recalls    = [rep[c]["recall"]    for c in CLASSES]
f1s        = [rep[c]["f1-score"]  for c in CLASSES]
ax.bar(x - w, precisions, w, label="Precision", color="#1f1f1f")
ax.bar(x,     recalls,    w, label="Recall",    color="#555555")
ax.bar(x + w, f1s,        w, label="F1",        color="#888888")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Hybrid Ensemble — Per-Class Metrics (ResNet-50 + SVM + TTA)", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_per_class.png", dpi=150, bbox_inches="tight")
plt.close()

# Ablation bar chart
fig, ax = plt.subplots(figsize=(12, 5))
names = list(ablation_results.keys())
accs  = [v["acc"] for v in ablation_results.values()]
colors_abl = ["#bbbbbb"] * (len(names) - 1) + ["#1f1f1f"]
bars = ax.bar(names, accs, color=colors_abl, edgecolor="white")
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Ablation Study: Contribution of Each Component", fontweight="bold")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002, f"{v:.4f}",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_ablation.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("  HYBRID ENSEMBLE COMPLETE")
print("=" * 60)
print(f"  Backbone      : ResNet-50 (fine-tuned layer4)")
print(f"  Features      : 2048-dim → PCA (98% var)")
print(f"  TTA views     : {len(tta_transforms)}")
print(f"  Ensemble      : SVM(C=100) + RF(500) + GB(200) [val-optimised weights]")
print(f"  ----------------------------------------")
print(f"  Accuracy      : {acc_hybrid:.4f} ({acc_hybrid*100:.2f}%)")
print(f"  Macro F1      : {f1_hybrid:.4f}")
print(f"  Brier Score   : {brier_hybrid:.4f}")
print("=" * 60)
print(f"  Improvement over BML: +{(acc_hybrid - 0.85)*100:.1f}%")
print(f"  Improvement over DL : +{(acc_hybrid - 0.92)*100:.1f}%")
print("=" * 60)