"""Visualization #3: feature-space overlap plot. Reduces the existing
5-channel RMSE artifacts (artifacts/kitsune_test_group_rmse.npy, from the
per-channel ensemble model, Replication Attempt #4 -- no retraining) to
2D via PCA, colored by three categories: benign outside the attack span,
benign inside it, and attack. This is the most direct visual evidence for
*why* the detector can't separate the classes: if the categories overlap
in this reduced space, they overlap in the original 5-channel error space
the detector actually scores on.

Usage: python -m scripts.visualize_latent_overlap
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_timeline"
RMSE_PATH = PROJECT_ROOT / "artifacts" / "kitsune_test_group_rmse.npy"
FIG_PATH = ARTIFACT_DIR / "latent_overlap.png"
SPLIT_INDEX = 1_000_000
ATTACK_SPAN = (1_314_658, 1_576_578)  # absolute dataset row indices

# Reference palette (validated, see dataviz skill): a 3-category scatter
# needs all-pairs CVD safety, which only the first three categorical slots
# (blue/orange/aqua) clear -- used here even though "attack" isn't red.
COLOR_BENIGN_OUT = "#2a78d6"
COLOR_BENIGN_IN = "#eb6834"
COLOR_ATTACK = "#1baf7a"
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"

N_PER_CATEGORY = 4000


def main():
    if not RMSE_PATH.exists():
        raise FileNotFoundError(
            f"{RMSE_PATH} not found -- run src/models/kitsune_ensemble.py once first "
            "(Replication Attempt #4) to generate it."
        )

    from src.ingestion.kitsune import load_kitsune

    print(f"Loading cached per-channel RMSE from {RMSE_PATH} (no model load needed)...")
    group_rmse = np.load(RMSE_PATH)  # (697851, 5)

    df = load_kitsune()
    y = df["label"].to_numpy()
    y_test = y[SPLIT_INDEX:]
    n = len(y_test)
    abs_row = np.arange(SPLIT_INDEX, SPLIT_INDEX + n)
    in_window = (abs_row >= ATTACK_SPAN[0]) & (abs_row <= ATTACK_SPAN[1])

    benign_out_idx = np.where((y_test == 0) & ~in_window)[0]
    benign_in_idx = np.where((y_test == 0) & in_window)[0]
    attack_idx = np.where(y_test == 1)[0]

    rng = np.random.default_rng(0)

    def subsample(idx_array):
        if len(idx_array) <= N_PER_CATEGORY:
            return idx_array
        return rng.choice(idx_array, size=N_PER_CATEGORY, replace=False)

    benign_out_s = subsample(benign_out_idx)
    benign_in_s = subsample(benign_in_idx)
    attack_s = subsample(attack_idx)

    print(f"plotting {len(benign_out_s)} benign-out-of-window, {len(benign_in_s)} benign-in-window, "
          f"{len(attack_s)} attack points")

    all_idx = np.concatenate([benign_out_s, benign_in_s, attack_s])
    X = group_rmse[all_idx]
    # Standardize each channel first -- the 5 channels have different
    # native RMSE scales, and an unstandardized PCA is dominated almost
    # entirely by whichever channel has the largest raw magnitude (verified:
    # PC1 explained 98.7% of variance unstandardized, which is a scale
    # artifact, not genuine structure).
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_std)
    var_explained = pca.explained_variance_ratio_

    n1, n2, n3 = len(benign_out_s), len(benign_in_s), len(attack_s)
    coords_benign_out = coords[:n1]
    coords_benign_in = coords[n1:n1 + n2]
    coords_attack = coords[n1 + n2:]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.scatter(*coords_benign_out.T, s=6, alpha=0.35, color=COLOR_BENIGN_OUT,
               label=f"Benign, outside attack span (n={n1})", linewidths=0)
    ax.scatter(*coords_benign_in.T, s=6, alpha=0.45, color=COLOR_BENIGN_IN,
               label=f"Benign, inside attack span (n={n2})", linewidths=0)
    ax.scatter(*coords_attack.T, s=6, alpha=0.45, color=COLOR_ATTACK,
               label=f"Attack, OS Scan (n={n3})", linewidths=0)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", color=COLOR_TEXT_PRIMARY)
    ax.set_title(
        "5-channel reconstruction error, reduced to 2D (PCA)\n"
        "Benign-in-window and attack overlap -- this is the overlap the detector can't resolve",
        color=COLOR_TEXT_PRIMARY, fontsize=11,
    )
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_TEXT_MUTED)
    ax.tick_params(colors=COLOR_TEXT_MUTED)
    legend = ax.legend(loc="best", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)

    fig.tight_layout()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor=COLOR_SURFACE)
    print(f"Saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
