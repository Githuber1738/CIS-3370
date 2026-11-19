"""Visualization #1: reconstruction-error timeline across the OS Scan test
set -- the single clearest visual of this project's root-cause finding.

First run needs the trained baseline model (no training, inference only)
to compute and cache test_error/y_test as small .npy artifacts. Every
run after that is pure matplotlib -- no TensorFlow import, no model load --
so this re-renders in well under a second and is safe to iterate on or
re-run live.

Usage: python -m scripts.visualize_timeline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_timeline"
ERROR_PATH = ARTIFACT_DIR / "baseline_test_error.npy"
LABEL_PATH = ARTIFACT_DIR / "baseline_test_labels.npy"
THRESHOLD_PATH = ARTIFACT_DIR / "baseline_threshold.npy"
FIG_PATH = ARTIFACT_DIR / "timeline.png"
SPLIT_INDEX = 1_000_000

# Reference palette (validated, see dataviz skill): categorical slot 1
# (blue) and slot 8 (red), adjacent-pair safe for a 2-series chart.
COLOR_BENIGN = "#2a78d6"
COLOR_ATTACK = "#e34948"
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"


def compute_and_cache():
    print("No cached scores found -- computing once from the trained model (inference only, no retraining)...")
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    from tensorflow import keras

    from src.ingestion.kitsune import load_kitsune

    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]

    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    model = keras.models.load_model(PROJECT_ROOT / "models" / "kitsune_os_scan_autoencoder.keras")
    train_recon = model.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = float(np.mean(train_error) + 2 * np.std(train_error))

    recon = model.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - recon), axis=1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ERROR_PATH, test_error)
    np.save(LABEL_PATH, y_test)
    np.save(THRESHOLD_PATH, np.array([threshold]))
    print(f"Cached to {ERROR_PATH}, {LABEL_PATH}, {THRESHOLD_PATH} (threshold={threshold:.6f})")
    return test_error, y_test, threshold


def main():
    if ERROR_PATH.exists() and LABEL_PATH.exists() and THRESHOLD_PATH.exists():
        print(f"Loading cached scores from {ERROR_PATH} (no TensorFlow needed)...")
        test_error = np.load(ERROR_PATH)
        y_test = np.load(LABEL_PATH)
        threshold = float(np.load(THRESHOLD_PATH)[0])
    else:
        test_error, y_test, threshold = compute_and_cache()
    n = len(test_error)
    idx = np.arange(n)
    benign_mask = y_test == 0
    attack_mask = y_test == 1

    fig, (ax_main, ax_rug) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06},
        facecolor=COLOR_SURFACE, constrained_layout=True,
    )

    # Subsample benign points for render speed/clarity; keep every attack point.
    rng = np.random.default_rng(0)
    benign_idx_all = idx[benign_mask]
    benign_show = rng.choice(benign_idx_all, size=min(40_000, len(benign_idx_all)), replace=False)

    ax_main.set_facecolor(COLOR_SURFACE)
    ax_main.scatter(benign_show, test_error[benign_show], s=2, alpha=0.25,
                     color=COLOR_BENIGN, label="Benign (correctly labeled)", linewidths=0)
    ax_main.scatter(idx[attack_mask], test_error[attack_mask], s=3, alpha=0.5,
                     color=COLOR_ATTACK, label="Attack (OS Scan)", linewidths=0)
    ax_main.axhline(threshold, color=COLOR_TEXT_PRIMARY, linewidth=1, linestyle="--",
                     label=f"Static threshold ({threshold:.3f})")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Reconstruction error (log scale)", color=COLOR_TEXT_PRIMARY)
    ax_main.set_title(
        "Kitsune OS Scan: reconstruction error vs. test row\n"
        "Benign error rises inside the attack span -- the model never trained on this network condition",
        color=COLOR_TEXT_PRIMARY, fontsize=12,
    )
    ax_main.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.spines[["left", "bottom"]].set_color(COLOR_BASELINE_AXIS)
    ax_main.tick_params(colors=COLOR_TEXT_MUTED)
    legend = ax_main.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)

    # Rug plot: every individual attack-labeled row as a tick, showing the
    # 34,069-run scattered structure directly (see Kitsune_Replication.md,
    # "Root-Cause Diagnostics") rather than implying a single solid block.
    ax_rug.set_facecolor(COLOR_SURFACE)
    ax_rug.eventplot(idx[attack_mask], colors=COLOR_ATTACK, linewidths=0.4, lineoffsets=0, linelengths=0.9)
    ax_rug.set_yticks([])
    ax_rug.set_xlabel("Test row index", color=COLOR_TEXT_PRIMARY)
    ax_rug.set_ylabel("Attack\nrows", color=COLOR_TEXT_MUTED, fontsize=8, rotation=0, labelpad=25, va="center")
    ax_rug.spines[["top", "right", "left"]].set_visible(False)
    ax_rug.spines["bottom"].set_color(COLOR_BASELINE_AXIS)
    ax_rug.tick_params(colors=COLOR_TEXT_MUTED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor=COLOR_SURFACE)
    print(f"Saved figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
