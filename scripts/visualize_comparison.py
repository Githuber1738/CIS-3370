"""Visualization #4: grouped bar comparison of Precision/Recall/FPR across
the paper's target, our static baseline, and both IMP-09 adaptive-threshold
operating points. Numbers are hardcoded from the verified results already
logged in Kitsune_Improvement.md's Round 4 section -- no live computation,
ready for slide/Wiki use as-is.

Usage: python -m scripts.visualize_comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_timeline"
FIG_PATH = ARTIFACT_DIR / "comparison.png"

# Verified numbers, Kitsune_Improvement.md Round 4 table.
METRICS = ["Precision", "Recall", "FPR"]
VARIANTS = [
    ("xNIDS paper target", (0.9568, 0.9972, 0.001)),
    ("Static baseline", (0.5001, 1.0000, 0.1039)),
    ("IMP-09 adaptive, k=2", (0.0568, 0.0245, 0.0423)),
    ("IMP-09 adaptive, best-k=4.5", (0.2841, 0.0041, 0.0011)),
]

# Reference palette (validated, see dataviz skill): 4 series, fixed
# categorical order, adjacent-pair safe for a grouped bar chart.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"


def main():
    n_metrics = len(METRICS)
    n_variants = len(VARIANTS)
    x = np.arange(n_metrics)
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    for i, (label, values) in enumerate(VARIANTS):
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=COLORS[i],
                       edgecolor=COLOR_SURFACE, linewidth=1)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color=COLOR_TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("Value", color=COLOR_TEXT_PRIMARY)
    ax.set_title(
        "Target vs. static baseline vs. adaptive local threshold (IMP-09)\n"
        "The adaptive threshold trades recall for FPR but never approaches the paper's target",
        color=COLOR_TEXT_PRIMARY, fontsize=11,
    )
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_TEXT_MUTED)
    ax.tick_params(colors=COLOR_TEXT_MUTED)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)

    fig.tight_layout()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor=COLOR_SURFACE)
    print(f"Saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
