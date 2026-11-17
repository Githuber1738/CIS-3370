"""ROUND 2, EXPERIMENT 2.1 -- ratio-based redesign of the explanation-
informed post-filter (IMP-03).

IMP-03's threshold-crossing rule ("only MAC-IP crosses its own threshold,
all 4 others don't") never fired: flagged rows have multiple channels
elevated together, not MAC-IP in isolation. This redesign asks a
different question, directly motivated by that result: is MAC-IP
elevated by a much larger MARGIN than the next-highest channel, even if
the others are also somewhat elevated? The dominance ratio is calibrated
leak-free from the TRAINING set's own channel RMSEs (never test labels).
"""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate
from src.models.kitsune_ensemble import GROUP_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_INDEX = 1_000_000
RATIO_PERCENTILE = 99  # "unusual dominance" cutoff, calibrated from train data


def main():
    print("=== Round 2, Experiment 2.1: Ratio-Based Post-Filter ===")
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

    baseline = keras.models.load_model(MODEL_DIR / "kitsune_os_scan_autoencoder.keras")
    train_recon = baseline.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    test_recon = baseline.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred_baseline = (test_error > threshold).astype(int)
    print("baseline flags:", int(y_pred_baseline.sum()))

    train_group_rmse = np.load(PROJECT_ROOT / "artifacts" / "kitsune_train_group_rmse.npy")
    test_group_rmse = np.load(PROJECT_ROOT / "artifacts" / "kitsune_test_group_rmse.npy")
    mac_ip_idx = GROUP_NAMES.index("MAC-IP")
    other_idx = [i for i in range(5) if i != mac_ip_idx]

    # Calibrate the dominance ratio leak-free: MAC-IP RMSE / max(other 4) on TRAIN rows.
    eps = 1e-12
    train_ratio = train_group_rmse[:, mac_ip_idx] / (train_group_rmse[:, other_idx].max(axis=1) + eps)
    ratio_cutoff = np.percentile(train_ratio, RATIO_PERCENTILE)
    print(f"train ratio distribution: mean={train_ratio.mean():.3f}, "
          f"p50={np.percentile(train_ratio,50):.3f}, p99={ratio_cutoff:.3f}, max={train_ratio.max():.3f}")

    test_ratio = test_group_rmse[:, mac_ip_idx] / (test_group_rmse[:, other_idx].max(axis=1) + eps)
    dominant = test_ratio > ratio_cutoff
    n_suppressed = int(((y_pred_baseline == 1) & dominant).sum())
    print(f"flags suppressed: {n_suppressed} / {int(y_pred_baseline.sum())} "
          f"(ratio_cutoff={ratio_cutoff:.3f}, percentile={RATIO_PERCENTILE})")

    y_pred_filtered = y_pred_baseline.copy()
    y_pred_filtered[dominant] = 0

    metrics = evaluate(y_test, y_pred_filtered)
    print("\n--- Results (Experiment 2.1: Ratio-Based Post-Filter) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nBaseline: Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
