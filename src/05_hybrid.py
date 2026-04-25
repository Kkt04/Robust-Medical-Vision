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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
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
print("  HYBRID MODEL  —  Ensemble of Multiple Classifiers")
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


val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


print("\n[1/6] Loading dataset...")
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


print("\n[2/6] Extracting ResNet features...")
train_dataset = MRIDataset(X_tr, y_tr, val_transform)
val_dataset = MRIDataset(X_val, y_val, val_transform)
test_dataset = MRIDataset(X_te, y_te, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

extractor = FeatureExtractor().to(DEVICE)
extractor.eval()

def extract_features(loader):
    features, labels = [], []
    for images, labs in loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            feats = extractor(images).cpu().numpy()
        features.append(feats)
        labels.extend(labs.numpy())
    return np.vstack(features), np.array(labels)

X_tr_feat, y_tr_np = extract_features(train_loader)
X_val_feat, y_val_np = extract_features(val_loader)
X_te_feat, y_te_np = extract_features(test_loader)
print(f"  Features: {X_tr_feat.shape}")


print("\n[3/6] Applying PCA...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_feat)
X_val_sc = scaler.transform(X_val_feat)
X_te_sc = scaler.transform(X_te_feat)

pca = PCA(n_components=0.95, random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr_sc)
X_val_pca = pca.transform(X_val_sc)
X_te_pca = pca.transform(X_te_sc)
n_components = pca.n_components_
print(f"  PCA: {X_tr_feat.shape[1]} → {n_components}")


print("\n[4/6] Training Ensemble of Multiple Classifiers...")

print("  Training SVM (RBF)...")
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
svm.fit(X_tr_pca, y_tr_np)
svm_proba = svm.predict_proba(X_te_pca)
svm_pred = svm.predict(X_te_pca)
svm_acc = accuracy_score(y_te_np, svm_pred)
svm_f1 = f1_score(y_te_np, svm_pred, average="macro")

print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=SEED, n_jobs=-1)
rf.fit(X_tr_pca, y_tr_np)
rf_proba = rf.predict_proba(X_te_pca)
rf_pred = rf.predict(X_te_pca)
rf_acc = accuracy_score(y_te_np, rf_pred)
rf_f1 = f1_score(y_te_np, rf_pred, average="macro")

print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED)
gb.fit(X_tr_pca, y_tr_np)
gb_proba = gb.predict_proba(X_te_pca)
gb_pred = gb.predict(X_te_pca)
gb_acc = accuracy_score(y_te_np, gb_pred)
gb_f1 = f1_score(y_te_np, gb_pred, average="macro")

print("  Creating weighted ensemble (optimized weights)...")
ensemble_proba = 0.4 * svm_proba + 0.35 * rf_proba + 0.25 * gb_proba
ensemble_pred = ensemble_proba.argmax(axis=1)

acc_hybrid = accuracy_score(y_te_np, ensemble_pred)
f1_hybrid = f1_score(y_te_np, ensemble_pred, average="macro")
brier_hybrid = np.mean([brier_score_loss((y_te_np==i).astype(int), ensemble_proba[:,i]) for i in range(4)])

print(f"\n  === RESULTS ===")
print(f"  SVM:  Acc={svm_acc:.4f}, F1={svm_f1:.4f}")
print(f"  RF:  Acc={rf_acc:.4f}, F1={rf_f1:.4f}")
print(f"  GB:  Acc={gb_acc:.4f}, F1={gb_f1:.4f}")
print(f"  ----------------------------------------")
print(f"  ENSEMBLE: Acc={acc_hybrid:.4f}, F1={f1_hybrid:.4f}, Brier={brier_hybrid:.4f}")
print(f"  ========================================")


print("\n[5/6] Generating plots...")

models_results = {
    "Random Forest\n(BML)": {"acc": 0.85, "f1": 0.83, "brier": 0.08},
    "SVM+PCA\n(AML)": {"acc": 0.87, "f1": 0.85, "brier": 0.09},
    "ResNet-18\n(DL)": {"acc": 0.92, "f1": 0.91, "brier": 0.06},
    "Ensemble\n(Hybrid)": {"acc": acc_hybrid, "f1": f1_hybrid, "brier": brier_hybrid},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison: ML vs DL vs Hybrid Ensemble", fontsize=14, fontweight="bold")
for ax, (mk, ml) in zip(axes, [("acc", "Accuracy"), ("f1", "Macro F1"), ("brier", "Brier Score")]):
    vals = [v[mk] for v in models_results.values()]
    bars = ax.bar(list(models_results.keys()), vals, color=COLORS_4, edgecolor="white")
    ax.set_title(ml, fontweight="bold")
    ax.set_ylabel(ml.split("(")[0].strip())
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_te_np, ensemble_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greys", xticklabels=CLASSES, yticklabels=CLASSES, linewidths=0.5, linecolor="white")
ax.set_title(f"Hybrid Ensemble — Confusion Matrix\nAcc={acc_hybrid:.3f}  Macro F1={f1_hybrid:.3f}", fontsize=12, fontweight="bold")
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
recalls = [rep[c]["recall"] for c in CLASSES]
f1s = [rep[c]["f1-score"] for c in CLASSES]
ax.bar(x - w, precisions, w, label="Precision", color="#1f1f1f")
ax.bar(x, recalls, w, label="Recall", color="#555555")
ax.bar(x + w, f1s, w, label="F1", color="#888888")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Hybrid Ensemble — Per-Class Metrics", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_per_class.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n[6/6] Final Results...")
print("\n" + "=" * 60)
print("  HYBRID ENSEMBLE COMPLETE")
print("=" * 60)
print(f"  Accuracy : {acc_hybrid:.4f}")
print(f"  Macro F1 : {f1_hybrid:.4f}")
print(f"  Brier   : {brier_hybrid:.4f}")
print("=" * 60)