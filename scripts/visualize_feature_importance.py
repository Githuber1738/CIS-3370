"""T-06.3 -- visualize per-feature importance for the TP/FP sample detections
already explained by src/explain/run_explanation.py. Reuses that module's
exact explanation pipeline (same KitsuneExplanation calls, same seed=42
weighted sampling) and additionally keeps the raw per-feature coefficients
(not just the per-group mean used for the printed summary), so the chart
shows which individual features within each channel actually drove the
surrogate model's score -- a finer-grained view than group_importance().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.explain.kitsune_explanation import KITSUNE_GROUP_NAMES, KITSUNE_GROUP_SIZES, KitsuneExplanation
from src.explain.run_explanation import SAMPLES_DIR, MODEL_PATH, load_selection_summary
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_feature_importance"
TOP_N = 15

# dataviz reference palette, categorical slots 1-5 (fixed order, never cycled)
GROUP_COLORS = {
    "MAC-IP": "#2a78d6",   # slot 1 blue
    "IP": "#eb6834",       # slot 2 orange
    "Jitter": "#1baf7a",   # slot 3 aqua
    "IPtoIP": "#eda100",   # slot 4 yellow
    "Socket": "#e87ba4",   # slot 5 magenta
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def feature_group_names():
    names = []
    for group, size in zip(KITSUNE_GROUP_NAMES, KITSUNE_GROUP_SIZES):
        names.extend([group] * size)
    return names


def explain_with_coef(name, original_score):
    df = pd.read_csv(SAMPLES_DIR / f"kitsune_selected_{name}_rows.csv")
    current_sample = df.iloc[-1].to_numpy()
    history_samples = df.iloc[:-1].to_numpy()
    feature_names = list(df.columns)

    exp = KitsuneExplanation(current_sample, history_samples, MODEL_PATH,
                              original_score, feature_names)
    exp.capture_relevant_history()
    exp.weighted_sampling(num_samples=100, seed=42)
    exp.sparse_group_lasso()
    return exp.coef, feature_names


def top_features(coef, feature_names, group_names, top_n=TOP_N):
    order = np.argsort(-np.abs(coef))[:top_n]
    labels = [feature_names[i] for i in order]
    values = [abs(coef[i]) for i in order]
    colors = [GROUP_COLORS[group_names[i]] for i in order]
    # ascending for barh so the biggest bar ends up on top
    return labels[::-1], values[::-1], colors[::-1]


def plot_panel(ax, labels, values, colors, title):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("|surrogate coefficient|", color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    group_names = feature_group_names()
    summary = load_selection_summary()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=SURFACE,
                              constrained_layout=True)

    for ax, name in zip(axes, ["tp", "fp"]):
        info = summary[name]
        print(f"Explaining {name.upper()} (test row {info['test_index']}, "
              f"reconstruction_error={info['reconstruction_error']:.6f})...")
        coef, feature_names = explain_with_coef(name, info["reconstruction_error"])
        labels, values, colors = top_features(coef, feature_names, group_names)
        title = f"{name.upper()} -- test row {info['test_index']}"
        plot_panel(ax, labels, values, colors, title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, GROUP_COLORS.keys(), loc="lower center", ncol=5,
               frameon=False, labelcolor=INK, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Top {TOP_N} Features by Surrogate Importance -- TP vs. FP Detections",
                 color=INK, fontsize=13, fontweight="bold")

    out_path = OUT_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
