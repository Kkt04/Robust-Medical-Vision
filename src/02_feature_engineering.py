import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/Training")
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE   = (128, 128)
OUTPUT_DIR = Path("outputs/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "glioma":     "#E63946",
    "meningioma": "#457B9D",
    "notumor":    "#2A9D8F",
    "pituitary":  "#E9C46A",
}


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
    Returns ~324 features + HOG visualization image.
    """
    features, hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    # Rescale HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_image_rescaled

def extract_lbp_features(img, radius=3, n_points=24):
    """
    LBP captures local texture patterns.
    Tumor tissue has different micro-texture from healthy tissue.
    Returns 26 histogram bins + LBP image.
    """
    lbp = local_binary_pattern(
        (img * 255).astype(np.uint8), n_points, radius, method='uniform')
    hist, bins = np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)
    return hist, lbp

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


def get_sample_image(cls):
    """Return path to first available image in a class folder."""
    cls_path = DATA_DIR / cls
    images = sorted(cls_path.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {cls_path}")
    return images[0]

print("Generating HOG visualization...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("HOG Feature Visualization — Shape & Edge Detection per Class",
             fontsize=14, fontweight='bold', y=1.01)

for col, cls in enumerate(CLASSES):
    try:
        path = get_sample_image(cls)
        img  = load_image_gray(path)
        _, hog_img = extract_hog_features(img)

        # Row 0: original MRI
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(cls.upper(), fontsize=11, fontweight='bold',
                               color=COLORS[cls])
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original MRI", fontsize=9, labelpad=8)

        # Row 1: HOG image
        axes[1, col].imshow(hog_img, cmap='inferno')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel("HOG Edges", fontsize=9, labelpad=8)

    except Exception as e:
        print(f"  [SKIP] {cls}: {e}")
        axes[0, col].axis('off')
        axes[1, col].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hog_visualization.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] hog_visualization.png")


print("Generating LBP visualization...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("LBP Feature Visualization — Local Texture Patterns per Class",
             fontsize=14, fontweight='bold', y=1.01)

for col, cls in enumerate(CLASSES):
    try:
        path = get_sample_image(cls)
        img  = load_image_gray(path)
        hist, lbp_img = extract_lbp_features(img)

        # Row 0: original
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(cls.upper(), fontsize=11, fontweight='bold',
                               color=COLORS[cls])
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original MRI", fontsize=9)

        # Row 1: LBP image
        axes[1, col].imshow(lbp_img, cmap='viridis')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel("LBP Pattern", fontsize=9)

        # Row 2: LBP histogram
        axes[2, col].bar(range(len(hist)), hist, color=COLORS[cls],
                         edgecolor='black', linewidth=0.4)
        axes[2, col].set_xlabel("LBP Bin", fontsize=8)
        axes[2, col].set_ylabel("Density", fontsize=8)
        axes[2, col].tick_params(labelsize=7)
        if col == 0:
            axes[2, col].set_ylabel("LBP Histogram\nDensity", fontsize=9)

    except Exception as e:
        print(f"  [SKIP] {cls}: {e}")
        for row in range(3):
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lbp_visualization.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] lbp_visualization.png")

print("Generating intensity statistics visualization...")

# Collect stats for all 4 classes
stat_names = ["Mean", "Std", "Min", "Max", "Skewness",
              "Kurtosis", "P25", "P50", "P75", "P90"]
all_stats   = {}
all_hists   = {}

for cls in CLASSES:
    try:
        path = get_sample_image(cls)
        img  = load_image_gray(path)
        all_stats[cls] = extract_intensity_stats(img)
        all_hists[cls] = img.flatten()
    except Exception as e:
        print(f"  [SKIP] {cls}: {e}")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Intensity Statistics — Pixel Distribution per Class",
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Subplot 1: pixel intensity histograms overlay
ax1 = fig.add_subplot(gs[0, :])
for cls, flat in all_hists.items():
    ax1.hist(flat, bins=50, alpha=0.55, color=COLORS[cls],
             label=cls.upper(), density=True, edgecolor='none')
ax1.set_title("Pixel Intensity Distribution per Class (overlaid)", fontsize=11)
ax1.set_xlabel("Normalized Pixel Value [0–1]")
ax1.set_ylabel("Density")
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: mean intensity bar chart
ax2 = fig.add_subplot(gs[1, 0])
means = [all_stats[cls][0] for cls in CLASSES if cls in all_stats]
ax2.bar(CLASSES, means,
        color=[COLORS[c] for c in CLASSES], edgecolor='black', linewidth=0.6)
ax2.set_title("Mean Pixel Intensity per Class", fontsize=11)
ax2.set_ylabel("Mean Intensity [0–1]")
ax2.set_xlabel("Class")
for i, v in enumerate(means):
    ax2.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')

# Subplot 3: std intensity bar chart
ax3 = fig.add_subplot(gs[1, 1])
stds = [all_stats[cls][1] for cls in CLASSES if cls in all_stats]
ax3.bar(CLASSES, stds,
        color=[COLORS[c] for c in CLASSES], edgecolor='black', linewidth=0.6)
ax3.set_title("Std Dev of Pixel Intensity per Class", fontsize=11)
ax3.set_ylabel("Std Dev [0–1]")
ax3.set_xlabel("Class")
for i, v in enumerate(stds):
    ax3.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')

plt.savefig(OUTPUT_DIR / "intensity_stats.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] intensity_stats.png")

print("Generating GLCM visualization...")
glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
glcm_data  = {cls: {} for cls in CLASSES}

for cls in CLASSES:
    try:
        path     = get_sample_image(cls)
        img      = load_image_gray(path)
        img_uint8 = (img * 255).astype(np.uint8)
        glcm     = graycomatrix(img_uint8, distances=[1, 3],
                                angles=[0, np.pi/4, np.pi/2],
                                levels=256, symmetric=True, normed=True)
        for prop in glcm_props:
            glcm_data[cls][prop] = graycoprops(glcm, prop).mean()
    except Exception as e:
        print(f"  [SKIP] {cls}: {e}")

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("GLCM Properties per Class — Texture Co-occurrence Analysis",
             fontsize=13, fontweight='bold')

for ax, prop in zip(axes, glcm_props):
    vals = [glcm_data[cls].get(prop, 0) for cls in CLASSES]
    bars = ax.bar(CLASSES, vals,
                  color=[COLORS[c] for c in CLASSES],
                  edgecolor='black', linewidth=0.6)
    ax.set_title(prop.capitalize(), fontsize=11, fontweight='bold')
    ax.set_xlabel("Class", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f"{v:.3f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "glcm_properties.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] glcm_properties.png")

print("Generating combined feature summary...")
fig, axes = plt.subplots(4, 4, figsize=(18, 18))
fig.suptitle("Complete Feature Engineering Summary — One Sample per Class",
             fontsize=15, fontweight='bold', y=1.01)

for col, cls in enumerate(CLASSES):
    try:
        path             = get_sample_image(cls)
        img              = load_image_gray(path)
        _, hog_img       = extract_hog_features(img)
        lbp_hist, lbp_img = extract_lbp_features(img)
        stats            = extract_intensity_stats(img)

        # Row 0: original MRI
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(cls.upper(), fontsize=12, fontweight='bold',
                               color=COLORS[cls], pad=6)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Original\nMRI", fontsize=9, labelpad=10)

        # Row 1: HOG
        axes[1, col].imshow(hog_img, cmap='inferno')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel("HOG\nEdges", fontsize=9, labelpad=10)

        # Row 2: LBP
        axes[2, col].imshow(lbp_img, cmap='viridis')
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel("LBP\nTexture", fontsize=9, labelpad=10)

        # Row 3: Intensity histogram
        axes[3, col].hist(img.flatten(), bins=40, color=COLORS[cls],
                          edgecolor='black', linewidth=0.3, density=True)
        axes[3, col].axvline(stats[0], color='red', linestyle='--',
                             linewidth=1.2, label=f"μ={stats[0]:.2f}")
        axes[3, col].axvline(stats[6], color='orange', linestyle=':',
                             linewidth=1.0, label=f"P25={stats[6]:.2f}")
        axes[3, col].axvline(stats[8], color='blue', linestyle=':',
                             linewidth=1.0, label=f"P75={stats[8]:.2f}")
        axes[3, col].legend(fontsize=6, loc='upper right')
        axes[3, col].set_xlabel("Pixel Value", fontsize=8)
        axes[3, col].tick_params(labelsize=7)
        if col == 0:
            axes[3, col].set_ylabel("Intensity\nHistogram", fontsize=9, labelpad=10)

    except Exception as e:
        print(f"  [SKIP] {cls}: {e}")
        for row in range(4):
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_summary_grid.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] feature_summary_grid.png")

meta = {
    "classes": CLASSES,
    "label_map": {cls: i for i, cls in enumerate(CLASSES)},
    "img_size": list(IMG_SIZE),
    "features": {
        "HOG":  {"dims": "~324", "params": "orientations=9, cells=16x16, blocks=2x2"},
        "LBP":  {"dims": "26",   "params": "radius=3, n_points=24, method=uniform"},
        "GLCM": {"dims": "10",   "params": "distances=[1,3], angles=[0,45,90]"},
        "IntensityStats": {"dims": "10", "params": "mean,std,min,max,skew,kurtosis,P25,P50,P75,P90"},
    },
    "total_feature_dim": "~370",
    "outputs": [
        "hog_visualization.png",
        "lbp_visualization.png",
        "intensity_stats.png",
        "glcm_properties.png",
        "feature_summary_grid.png",
    ]
}
with open(OUTPUT_DIR / "feature_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("  [SAVED] feature_meta.json")

print("\n" + "="*55)
print("FEATURE ENGINEERING COMPLETE")
print("="*55)
print(f"Output directory : {OUTPUT_DIR}/")
print(f"Files saved:")
print(f"  hog_visualization.png    — HOG edge maps per class")
print(f"  lbp_visualization.png    — LBP texture + histograms")
print(f"  intensity_stats.png      — Pixel intensity analysis")
print(f"  glcm_properties.png      — GLCM texture properties")
print(f"  feature_summary_grid.png — Combined 4-row grid")
print(f"  feature_meta.json        — Feature configuration")
print("="*55)