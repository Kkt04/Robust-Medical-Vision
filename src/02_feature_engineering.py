"""

Features engineered:
  - HOG (Histogram of Oriented Gradients)  → shape/edge information
  - LBP (Local Binary Pattern)             → texture information
  - Intensity histogram statistics          → domain features
  - GLCM (Gray-Level Co-occurrence Matrix) → texture energy/contrast
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/Training")
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE   = (128, 128)
OUTPUT_DIR = Path("outputs/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image_gray(filepath, size=IMG_SIZE):
    """Load image as normalized grayscale numpy array [0,1]."""
    img = Image.open(filepath).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def extract_hog_features(img):
    """
    HOG captures shape & edge structure of the tumor.
    Gliomas have ragged edges (high gradient variance).
    Meningiomas have smooth edges (low gradient variance).
    Returns ~324 features for a 128x128 image.
    """
    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    return features

def extract_lbp_features(img, radius=3, n_points=24):
    """
    LBP captures local texture patterns.
    Tumor tissue has different micro-texture from healthy tissue.
    Returns 26 histogram bins (uniform patterns).
    """
    lbp = local_binary_pattern(
        (img * 255).astype(np.uint8), n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)
    return hist

def extract_intensity_stats(img):
    """
    10 statistical features of pixel intensity distribution.
    No Tumor class has darker mean than tumor classes — informative signal.
    """
    flat = img.flatten()
    return np.array([
        flat.mean(),
        flat.std(),
        flat.min(),
        flat.max(),
        skew(flat),
        kurtosis(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 50),
        np.percentile(flat, 75),
        np.percentile(flat, 90),
    ])

def extract_glcm_features(img):
    """
    GLCM captures spatial relationships between pixel intensities.
    Distinguishes tumor tissue from healthy tissue via contrast,
    energy, homogeneity, dissimilarity, and correlation.
    Returns 10 features (5 properties × 2 stats: mean + std).
    """
    img_uint8 = (img * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_uint8, distances=[1, 3],
        angles=[0, np.pi/4, np.pi/2],
        levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in props:
        values = graycoprops(glcm, prop)
        features.extend([values.mean(), values.std()])
    return np.array(features)

def extract_all_features(filepath):
    """Full feature pipeline for one image. Total ~370 features."""
    img = load_image_gray(filepath)
    return np.concatenate([
        extract_hog_features(img),
        extract_lbp_features(img),
        extract_intensity_stats(img),
        extract_glcm_features(img),
    ])

print("Starting feature extraction...")
label_map = {cls: i for i, cls in enumerate(CLASSES)}
X_all, y_all = [], []

for cls in CLASSES:
    cls_path = DATA_DIR / cls
    if not cls_path.exists():
        print(f"[SKIP] {cls_path} not found")
        continue
    img_files = list(cls_path.glob("*.jpg"))
    print(f"  [{cls}] Processing {len(img_files)} images...")
    for path in img_files:
        try:
            X_all.append(extract_all_features(path))
            y_all.append(label_map[cls])
        except Exception as e:
            print(f"    [ERROR] {path.name}: {e}")

X = np.array(X_all)
y = np.array(y_all)

print(f"\nFeature matrix shape : {X.shape}")
print(f"Label vector shape   : {y.shape}")
print(f"\nBreakdown:")
print(f"  HOG:             ~324 features")
print(f"  LBP:              26 features")
print(f"  Intensity Stats:  10 features")
print(f"  GLCM:             10 features")
print(f"  TOTAL:           {X.shape[1]} features")

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
print(f"\nNaN values: {nan_count}, Inf values: {inf_count}")
if nan_count > 0 or inf_count > 0:
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    print("[FIXED] Replaced NaN/Inf")

low_var = (X.var(axis=0) < 1e-6).sum()
print(f"Near-zero variance features: {low_var} (will be removed in modeling step)")

np.save(OUTPUT_DIR / "X_features.npy", X)
np.save(OUTPUT_DIR / "y_labels.npy", y)

meta = {
    "classes": CLASSES, "label_map": label_map,
    "feature_dim": X.shape[1], "n_samples": X.shape[0],
}
with open(OUTPUT_DIR / "feature_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[SAVED] → outputs/features/")