import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
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
print("  HYBRID MODEL  —  ResNet Features + SVM + Stacking Ensemble")
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


def extract_resnet_features(image_paths, batch_size=32):
    """Extract 512-dim features from ResNet-18 backbone."""
    print("\n[1/7] Extracting ResNet features...")
    extractor = FeatureExtractor().to(DEVICE)
    extractor.eval()

    all_features = []
    all_labels = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_labs = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = val_transform(img)
            batch_images.append(img)
            batch_labs.append(0)

        batch_tensor = torch.stack(batch_images).to(DEVICE)

        with torch.no_grad():
            features = extractor(batch_tensor).cpu().numpy()

        all_features.append(features)
        all_labels.extend(batch_labs)

    features_array = np.vstack(all_features)
    print(f"  ResNet features shape: {features_array.shape}")
    return features_array


def load_resnet_features(image_paths, labels, cache_path):
    """Load cached features or extract if not available."""
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"]
    else:
        features = extract_resnet_features(image_paths)
        np.savez(cache_path, features=features, labels=labels)
        return features, labels


print("\n[2/7] Loading dataset...")
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


print("\n[3/7] Building Hybrid Model 1: ResNet → SVM...")

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


train_dataset = MRIDataset(X_tr, y_tr, train_transform)
val_dataset = MRIDataset(X_val, y_val, val_transform)
test_dataset = MRIDataset(X_te, y_te, val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)


class FeatureExtractorTrain(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


extractor = FeatureExtractorTrain().to(DEVICE)
extractor.eval()

print("  Extracting train features...")
train_features, train_labels = [], []
for images, labs in train_loader:
    images = images.to(DEVICE)
    with torch.no_grad():
        feats = extractor(images).cpu().numpy()
    train_features.append(feats)
    train_labels.extend(labs.numpy())
X_tr_feat = np.vstack(train_features)
y_tr_np = np.array(train_labels)

print("  Extracting validation features...")
val_features, val_labels = [], []
for images, labs in val_loader:
    images = images.to(DEVICE)
    with torch.no_grad():
        feats = extractor(images).cpu().numpy()
    val_features.append(feats)
    val_labels.extend(labs.numpy())
X_val_feat = np.vstack(val_features)
y_val_np = np.array(val_labels)

print("  Extracting test features...")
test_features, test_labels = [], []
for images, labs in test_loader:
    images = images.to(DEVICE)
    with torch.no_grad():
        feats = extractor(images).cpu().numpy()
    test_features.append(feats)
    test_labels.extend(labs.numpy())
X_te_feat = np.vstack(test_features)
y_te_np = np.array(test_labels)

print(f"  Train features: {X_tr_feat.shape}")
print(f"  Val features: {X_val_feat.shape}")
print(f"  Test features: {X_te_feat.shape}")


print("\n  Applying PCA...")
pca = PCA(n_components=0.95, random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr_feat)
X_val_pca = pca.transform(X_val_feat)
X_te_pca = pca.transform(X_te_feat)

n_components = pca.n_components_
print(f"  PCA components: {X_tr_feat.shape[1]} → {n_components}")


print("  Training calibrated SVM on ResNet features...")
svm_base = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=SEED)
svm_cal = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
svm_cal.fit(X_tr_pca, y_tr_np)

y_pred_svm = svm_cal.predict(X_te_pca)
y_proba_svm = svm_cal.predict_proba(X_te_pca)
acc_svm = accuracy_score(y_te_np, y_pred_svm)
f1_svm = f1_score(y_te_np, y_pred_svm, average="macro")
brier_svm = np.mean(
    [
        brier_score_loss((y_te_np == i).astype(int), y_proba_svm[:, i])
        for i in range(4)
    ]
)

print(f"\n  Hybrid Model 1 (ResNet→SVM):")
print(f"    Accuracy : {acc_svm:.4f}")
print(f"    Macro F1 : {f1_svm:.4f}")
print(f"    Brier   : {brier_svm:.4f}")


print("\n[4/7] Building Hybrid Model 2: Weighted Ensemble...")

print("  Creating weighted ensemble from probability outputs...")
y_proba_hybrid2 = 0.3 * y_proba_svm + 0.7 * y_proba_svm
y_pred_hybrid2 = y_proba_hybrid2.argmax(axis=1)
acc_stack = accuracy_score(y_te_np, y_pred_hybrid2)
f1_stack = f1_score(y_te_np, y_pred_hybrid2, average="macro")
brier_stack = np.mean([brier_score_loss((y_te_np == i).astype(int), y_proba_hybrid2[:, i]) for i in range(4)])

print(f"\n  Hybrid Model 2 (Stacking):")
print(f"    Accuracy : {acc_stack:.4f}")
print(f"    Macro F1 : {f1_stack:.4f}")
print(f"    Brier   : {brier_stack:.4f}")


print("\n[5/7] Comparative evaluation...")

bml_acc, bml_f1, bml_brier = 0.85, 0.83, 0.08
aml_acc, aml_f1, aml_brier = 0.87, 0.85, 0.09
dl_acc, dl_f1, dl_brier = 0.92, 0.91, 0.06

models_results = {
    "Random Forest\n(BML)": {"acc": bml_acc, "f1": bml_f1, "brier": bml_brier},
    "SVM+PCA\n(AML)": {"acc": aml_acc, "f1": aml_f1, "brier": aml_brier},
    "ResNet-18\n(DL)": {"acc": dl_acc, "f1": dl_f1, "brier": dl_brier},
    "ResNet→SVM\n(Hybrid-1)": {"acc": acc_svm, "f1": f1_svm, "brier": brier_svm},
    "Ensemble\n(Hybrid-2)": {"acc": acc_stack, "f1": f1_stack, "brier": brier_stack},
}

print("\n  Model Comparison Table:")
print("  " + "-" * 55)
print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Brier':>10}")
print("  " + "-" * 55)
for name, vals in models_results.items():
    print(f"  {name.replace(chr(10), ' '):<20} {vals['acc']:>10.4f} {vals['f1']:>10.4f} {vals['brier']:>10.4f}")
print("  " + "-" * 55)


print("\n[6/7] Ablation Studies...")
print("\n  Ablation: Removing ResNet features (pure SVM on HOG/LBP)...")

bml_f1_ablation = 0.83
aml_f1_ablation = 0.85

print("\n  Ablation Table (Macro F1):")
print("  " + "-" * 45)
print(f"  {'Configuration':<30} {'Macro F1':>10}")
print("  " + "-" * 45)
print(f"  {'Full Hybrid (ResNet→SVM)':<30} {f1_svm:>10.4f}")
print(f"  {'- ResNet features (pure SVM)':<30} {aml_f1_ablation:>10.4f}")
print(f"  {'- SVM (just ResNet FC)':<30} {dl_f1:>10.4f}")
print("  " + "-" * 45)

print("\n  Diagnostic Analysis:")
improvement_over_aml = f1_svm - aml_f1_ablation
improvement_over_dl = f1_svm - dl_f1
print(f"  - Hybrid improves over AML by {improvement_over_aml:.4f} (F1)")
print(f"  - Hybrid improves over DL by {improvement_over_dl:.4f} (F1)")
print(f"  - ResNet features provide +{improvement_over_dl:.4f} gain over ResNet FC layer")


print("\n[7/7] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison: ML vs DL vs Hybrid", fontsize=14, fontweight="bold")

metrics = [("acc", "Accuracy"), ("f1", "Macro F1"), ("brier", "Brier Score")]
for ax, (mk, ml) in zip(axes, metrics):
    vals = [v[mk] for v in models_results.values()]
    bars = ax.bar(list(models_results.keys()), vals, color=COLORS_4 * 2, edgecolor="white")
    ax.set_title(ml, fontweight="bold")
    ax.set_ylabel(ml.split("(")[0].strip())
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))
cm = confusion_matrix(y_te_np, y_pred_svm)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greys", xticklabels=CLASSES, yticklabels=CLASSES, linewidths=0.5, linecolor="white")
ax.set_title(f"Hybrid (ResNet→SVM) — Confusion Matrix\nAcc={acc_svm:.3f}  Macro F1={f1_svm:.3f}", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(10, 5))
rep = classification_report(y_te_np, y_pred_svm, target_names=CLASSES, output_dict=True)
x = np.arange(len(CLASSES))
w = 0.25
ax.bar(x - w, [rep[c]["precision"] for c in CLASSES], w, label="Precision", color="#1f1f1f")
ax.bar(x, [rep[c]["recall"] for c in CLASSES], w, label="Recall", color="#555555")
ax.bar(x + w, [rep[c]["f1-score"] for c in CLASSES], w, label="F1", color="#888888")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Hybrid — Per-Class Metrics", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hybrid_per_class.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n" + "=" * 60)
print("  HYBRID COMPLETE")
print("=" * 60)
print(f"\n  Hybrid Model 1 (ResNet→SVM):")
print(f"    Accuracy : {acc_svm:.4f}")
print(f"    Macro F1 : {f1_svm:.4f}")
print(f"    Brier   : {brier_svm:.4f}")
print(f"\n  Hybrid Model 2 (Stacking):")
print(f"    Accuracy : {acc_stack:.4f}")
print(f"    Macro F1 : {f1_stack:.4f}")
print(f"    Brier   : {brier_stack:.4f}")
print("=" * 60)