import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("outputs/hybrid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

ax.set_title("Hybrid Architecture: Ensemble of Multiple Classifiers on Deep Features", fontsize=16, fontweight="bold", pad=20)

ax.add_patch(plt.Rectangle((0.8, 6.3), 2.5, 1.9, fill=True, facecolor="#2d2d2d", edgecolor="black", linewidth=2))
ax.text(2.05, 7.4, "MRI Input", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(2.05, 6.9, "(128×128)", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((4.5, 5.8), 3.2, 2.4, fill=True, facecolor="#404040", edgecolor="black", linewidth=2))
ax.text(6.1, 7.6, "ResNet-18", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(6.1, 7.1, "(frozen)", ha="center", va="center", fontsize=9, color="#cccccc")
ax.text(6.1, 6.4, "Backbone", ha="center", va="center", fontsize=9, color="#aaaaaa")

ax.add_patch(plt.Rectangle((8.8, 6.3), 2.2, 1.9, fill=True, facecolor="#606060", edgecolor="black", linewidth=2))
ax.text(9.9, 7.3, "512-dim", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
ax.text(9.9, 6.8, "Features", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((11.2, 6.3), 2.2, 1.9, fill=True, facecolor="#808080", edgecolor="black", linewidth=2))
ax.text(12.3, 7.3, "PCA", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(12.3, 6.8, "(227 dim)", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((7.0, 3.5), 2.2, 1.5, fill=True, facecolor="#a0a0a0", edgecolor="black", linewidth=2))
ax.text(8.1, 4.4, "SVM", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax.text(8.1, 3.9, "(RBF)", ha="center", va="center", fontsize=9, color="#404040")

ax.add_patch(plt.Rectangle((5.0, 1.8), 2.2, 1.5, fill=True, facecolor="#b0b0b0", edgecolor="black", linewidth=2))
ax.text(6.1, 2.7, "Random", ha="center", va="center", fontsize=10, fontweight="bold", color="black")
ax.text(6.1, 2.2, "Forest", ha="center", va="center", fontsize=9, color="#404040")

ax.add_patch(plt.Rectangle((9.0, 1.8), 2.2, 1.5, fill=True, facecolor="#c0c0c0", edgecolor="black", linewidth=2))
ax.text(10.1, 2.7, "Gradient", ha="center", va="center", fontsize=10, fontweight="bold", color="black")
ax.text(10.1, 2.2, "Boosting", ha="center", va="center", fontsize=9, color="#404040")

ax.add_patch(plt.Rectangle((11.0, 3.5), 2.4, 2.4, fill=True, facecolor="#d0d0d0", edgecolor="black", linewidth=2))
ax.text(12.2, 4.9, "Ensemble", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax.text(12.2, 4.4, "Weighted", ha="center", va="center", fontsize=9, color="#404040")
ax.text(12.2, 3.9, "Average", ha="center", va="center", fontsize=9, color="#404040")

ax.add_patch(plt.Rectangle((11.0, 0.8), 2.4, 1.4, fill=True, facecolor="#e0e0e0", edgecolor="black", linewidth=2))
ax.text(12.2, 1.6, "Output", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax.text(12.2, 1.1, "Prediction", ha="center", va="center", fontsize=9, color="#404040")

ax.annotate("", xy=(4.5, 7.25), xytext=(3.3, 7.25), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(8.8, 7.25), xytext=(8.2, 7.25), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(11.2, 7.25), xytext=(10.0, 6.3), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(8.1, 5.0), xytext=(8.1, 5.85), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(6.1, 3.35), xytext=(6.1, 3.95), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(10.1, 3.35), xytext=(10.1, 3.95), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(11.5, 4.75), xytext=(11.9, 4.75), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(12.2, 2.25), xytext=(12.2, 3.5), arrowprops=dict(arrowstyle="->", color="black", lw=2))

ax.text(5.8, 5.2, "Transfer", ha="center", fontsize=8, style="italic", color="#666666")
ax.text(5.8, 4.8, "Learning", ha="center", fontsize=8, style="italic", color="#666666")

ax.text(10.8, 5.5, "0.40×", ha="center", fontsize=9, fontweight="bold", color="#1f1f1f")
ax.text(8.5, 5.5, "0.35×", ha="center", fontsize=9, fontweight="bold", color="#555555")
ax.text(10.8, 2.3, "0.25×", ha="center", fontsize=9, fontweight="bold", color="#888888")

props = dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9, edgecolor="gray")
ax.text(3, 0.5, "🏆 BEST ACCURACY: 94.1%\n• SVM on DL features beats pure DL\n• Ensemble combines strengths",
        fontsize=10, bbox=props, ha="center", va="center")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Architecture diagram saved → outputs/hybrid/architecture_diagram.png")