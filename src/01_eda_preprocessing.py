import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


DATA_DIR = Path("data/Training")  # Adjust if dataset path differs
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_info(data_dir: Path):
    records = []
    for cls in CLASSES:
        cls_path = data_dir / cls
        if not cls_path.exists():
            print(f"[WARNING] Class folder not found: {cls_path}")
            continue
        for img_path in cls_path.glob("*.jpg"):
            try:
                img = Image.open(img_path)
                w, h = img.size
                records.append({
                    "filepath": str(img_path),
                    "class": cls,
                    "width": w,
                    "height": h,
                    "mode": img.mode,
                    "file_size_kb": img_path.stat().st_size / 1024,
                })
            except Exception as e:
                print(f"[ERROR] Could not read {img_path}: {e}")
    return pd.DataFrame(records)

df = load_dataset_info(DATA_DIR)
print(f"\nTotal images found: {len(df)}")
print(df.head())

class_counts = df["class"].value_counts()
print("\n── Class Distribution ──")
print(class_counts)
print(f"\nClass imbalance ratio (max/min): {class_counts.max()/class_counts.min():.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Distribution Analysis", fontsize=14, fontweight='bold')

bars = axes[0].bar(class_counts.index, class_counts.values,
                   color=['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'],
                   edgecolor='black', linewidth=0.8)
axes[0].set_title("Samples per Class")
axes[0].set_xlabel("Tumor Type")
axes[0].set_ylabel("Count")
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(count), ha='center', fontweight='bold')

axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'])
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n── Image Size Statistics ──")
print(df[["width", "height"]].describe())

non_square = df[df["width"] != df["height"]]
print(f"\nNon-square images: {len(non_square)} ({100*len(non_square)/len(df):.1f}%)")
print("\n── Image Mode Distribution ──")
print(df["mode"].value_counts())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Image Property Analysis", fontsize=14, fontweight='bold')

axes[0].hist(df["width"], bins=30, color='steelblue', edgecolor='black')
axes[0].set_title("Width Distribution")
axes[0].set_xlabel("Width (px)")
axes[0].axvline(df["width"].mean(), color='red', linestyle='--',
                label=f'Mean={df["width"].mean():.0f}')
axes[0].legend()

axes[1].hist(df["height"], bins=30, color='coral', edgecolor='black')
axes[1].set_title("Height Distribution")
axes[1].set_xlabel("Height (px)")
axes[1].axvline(df["height"].mean(), color='blue', linestyle='--',
                label=f'Mean={df["height"].mean():.0f}')
axes[1].legend()

axes[2].hist(df["file_size_kb"], bins=30, color='seagreen', edgecolor='black')
axes[2].set_title("File Size Distribution (KB)")
axes[2].set_xlabel("File Size (KB)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "image_properties.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 5. PIXEL INTENSITY ANALYSIS
# ─────────────────────────────────────────────
def compute_pixel_stats(df, n_samples=50):
    stats = []
    for cls in CLASSES:
        cls_df = df[df["class"] == cls].sample(
            min(n_samples, len(df[df["class"] == cls])), random_state=42)
        intensities = []
        for path in cls_df["filepath"]:
            try:
                img = Image.open(path).convert("L").resize(IMG_SIZE)
                intensities.extend(np.array(img).flatten().tolist())
            except:
                continue
        intensities = np.array(intensities)
        stats.append({
            "class": cls,
            "mean_intensity": intensities.mean(),
            "std_intensity": intensities.std(),
            "min_intensity": intensities.min(),
            "max_intensity": intensities.max(),
            "skewness": pd.Series(intensities).skew(),
            "kurtosis": pd.Series(intensities).kurtosis(),
        })
    return pd.DataFrame(stats)

print("\n[INFO] Computing pixel intensity statistics...")
pixel_stats = compute_pixel_stats(df)
print("\n── Pixel Intensity Statistics per Class ──")
print(pixel_stats.to_string(index=False))

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Sample Brain MRI Images per Class", fontsize=16, fontweight='bold')
cols = 5

for row_idx, cls in enumerate(CLASSES):
    cls_df = df[df["class"] == cls].sample(
        min(cols, len(df[df["class"] == cls])), random_state=row_idx)
    for col_idx, (_, record) in enumerate(cls_df.iterrows()):
        ax = fig.add_subplot(len(CLASSES), cols, row_idx * cols + col_idx + 1)
        img = Image.open(record["filepath"]).convert("L")
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if col_idx == 0:
            ax.set_ylabel(cls.upper(), fontsize=11, fontweight='bold',
                          rotation=90, labelpad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150, bbox_inches='tight')
plt.show()


df["class_id"] = df["class"].map({c: i for i, c in enumerate(CLASSES)})
numeric_df = df[["width", "height", "file_size_kb", "class_id"]].dropna()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)
print(f"Total images            : {len(df)}")
print(f"Class balance ratio     : {class_counts.max()/class_counts.min():.2f}x")
print(f"Non-square images       : {len(non_square)} — MUST resize")
print(f"Null values             : {df.isnull().sum().sum()} — Dataset is clean")
print()
print("── Preprocessing Actions Required ──")
print("  1. Resize all images to 224x224")
print("  2. Convert all to grayscale (L mode)")
print("  3. Normalize pixel values to [0, 1]")
print("  4. Handle class imbalance via class weights")
print("  5. Stratified Train/Val/Test split: 70/15/15")
print("="*60)

df.describe().to_csv(OUTPUT_DIR / "eda_summary.csv")
pixel_stats.to_csv(OUTPUT_DIR / "pixel_stats.csv", index=False)
print("\n[SAVED] EDA outputs → outputs/eda/")