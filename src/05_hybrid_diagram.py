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

ax.set_title("Hybrid Architecture: Neuro-Symbolic (ResNet Features → SVM)", fontsize=16, fontweight="bold", pad=20)

ax.add_patch(plt.Rectangle((0.8, 6.3), 2.7, 1.9, fill=True, facecolor="#2d2d2d", edgecolor="black", linewidth=2))
ax.text(2.15, 7.5, "MRI Input", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(2.15, 7.0, "(128×128)", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((4.8, 5.8), 3.4, 2.4, fill=True, facecolor="#404040", edgecolor="black", linewidth=2))
ax.text(6.5, 7.6, "ResNet-18", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(6.5, 7.1, "(frozen)", ha="center", va="center", fontsize=9, color="#cccccc")
ax.text(6.5, 6.4, "Backbone", ha="center", va="center", fontsize=9, color="#aaaaaa")

ax.add_patch(plt.Rectangle((8.8, 6.3), 2.4, 1.9, fill=True, facecolor="#606060", edgecolor="black", linewidth=2))
ax.text(10.0, 7.3, "512-dim", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
ax.text(10.0, 6.8, "Features", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((11.3, 6.3), 2.4, 1.9, fill=True, facecolor="#808080", edgecolor="black", linewidth=2))
ax.text(12.5, 7.3, "PCA", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax.text(12.5, 6.8, "(95% var)", ha="center", va="center", fontsize=9, color="#cccccc")

ax.add_patch(plt.Rectangle((12.8, 5.8), 2.4, 2.4, fill=True, facecolor="#a0a0a0", edgecolor="black", linewidth=2))
ax.text(14.0, 7.6, "SVM", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax.text(14.0, 7.1, "(RBF +", ha="center", va="center", fontsize=9, color="#404040")
ax.text(14.0, 6.6, "Platt)", ha="center", va="center", fontsize=9, color="#404040")

ax.add_patch(plt.Rectangle((12.8, 3.3), 2.4, 1.9, fill=True, facecolor="#c0c0c0", edgecolor="black", linewidth=2))
ax.text(14.0, 4.4, "Class", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax.text(14.0, 3.9, "Prediction", ha="center", va="center", fontsize=9, color="#404040")

ax.annotate("", xy=(4.8, 7.25), xytext=(3.5, 7.25), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(8.8, 7.25), xytext=(8.2, 7.25), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(11.3, 7.25), xytext=(10.0, 6.3), arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.annotate("", xy=(13.7, 5.8), xytext=(13.7, 6.3), arrowprops=dict(arrowstyle="->", color="black", lw=2))

ax.add_patch(plt.Rectangle((0.8, 3.8), 2.7, 1.5, fill=True, facecolor="#e0e0e0", edgecolor="#404040", linewidth=1, linestyle="--"))
ax.text(2.15, 4.65, "ML Baseline", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(2.15, 4.15, "HOG + LBP", ha="center", va="center", fontsize=8, color="#404040")

ax.add_patch(plt.Rectangle((0.8, 1.8), 2.7, 1.5, fill=True, facecolor="#d0d0d0", edgecolor="#404040", linewidth=1, linestyle="--"))
ax.text(2.15, 2.65, "DL Baseline", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(2.15, 2.15, "ResNet-18 FC", ha="center", va="center", fontsize=8, color="#404040")

ax.text(14.0, 2.0, "↓", fontsize=20, ha="center")
ax.text(14.0, 1.2, "4-class", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(14.0, 0.7, "Output", ha="center", va="center", fontsize=9, color="#404040")

ax.text(5, 5.4, "Transfer Learning", ha="center", fontsize=9, style="italic", color="#666666")
ax.text(5, 5.0, "from ImageNet", ha="center", fontsize=8, style="italic", color="#666666")

ax.text(9.5, 5.4, "Neural Feature", ha="center", fontsize=9, style="italic", color="#666666")
ax.text(9.5, 5.0, "Extraction", ha="center", fontsize=8, style="italic", color="#666666")

ax.text(12.8, 5.2, "Maximum Margin", ha="center", fontsize=8, style="italic", color="#666666")
ax.text(12.8, 4.8, "Classification", ha="center", fontsize=8, style="italic", color="#666666")

props = dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9, edgecolor="gray")
ax.text(8, 1.5, "Hybrid Innovation: Level 5 (Synergistic)\n• DL learns visual features; ML classifies\n• Strengths of both models leveraged", 
        fontsize=9, bbox=props, ha="center", va="center", fontfamily="monospace")

ax.text(1, 0.3, "Publication-Ready Architecture Diagram", fontsize=8, style="italic", color="#888888")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Architecture diagram saved → outputs/hybrid/architecture_diagram.png")