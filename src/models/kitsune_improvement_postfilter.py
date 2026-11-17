"""IMPROVEMENT EXPERIMENT 3 -- explanation-informed rule-based post-filter.

Directly reuses the src/explain/ investment: the explanation method found
MAC-IP dominates importance in both true-positive and false-positive
samples (a near-miss signature). This experiment builds a secondary
heuristic layer on top of the replicated baseline detector: when the
baseline flags a row, check the per-channel reconstruction error (from
replication Attempt #4's 5-channel ensemble: MAC-IP/IP/Jitter/IPtoIP/
Socket). If ONLY the MAC-IP channel is elevated and the other four look
normal, suppress the flag (reclassify as benign) -- the hypothesis is
that an "only MAC-IP is weird" row is more likely the kind of incidental
near-miss found in the explanation work, not a real attack signature
(which should show elevated error across multiple channels).

Per-channel thresholds are calibrated on TRAIN-only channel RMSE (leak-
free, same convention as everywhere else in this project). Reuses the
per-channel RMSE arrays already computed and saved by
src/models/kitsune_ensemble.py (artifacts/kitsune_{train,test}_group_rmse.npy).
"""

import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate
from src.models.kitsune_ensemble import GROUP_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_INDEX = 1_000_000


def main():
    t_start = time.time()
    print("=== Experiment 3: Explanation-Informed Rule-Based Post-Filter ===")
    print("Loading Kitsune OS Scan data and baseline model...")
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
    print(f"baseline threshold: {threshold}")
    baseline_metrics = evaluate(y_test, y_pred_baseline)
    print("baseline metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in baseline_metrics.items()})

    print("\nLoading per-channel RMSE from replication Attempt #4...")
    train_group_rmse = np.load(PROJECT_ROOT / "artifacts" / "kitsune_train_group_rmse.npy")
    test_group_rmse = np.load(PROJECT_ROOT / "artifacts" / "kitsune_test_group_rmse.npy")
    assert len(train_group_rmse) == SPLIT_INDEX
    assert len(test_group_rmse) == len(y_test)

    channel_thresholds = np.mean(train_group_rmse, axis=0) + 2 * np.std(train_group_rmse, axis=0)
    print("per-channel thresholds:", dict(zip(GROUP_NAMES, channel_thresholds.round(6))))

    elevated = test_group_rmse > channel_thresholds  # shape (n_test, 5)
    mac_ip_idx = GROUP_NAMES.index("MAC-IP")
    only_mac_ip_elevated = elevated[:, mac_ip_idx] & ~elevated[:, [i for i in range(5) if i != mac_ip_idx]].any(axis=1)

    n_suppressed = int(((y_pred_baseline == 1) & only_mac_ip_elevated).sum())
    print(f"\nflags suppressed by post-filter: {n_suppressed} / {int(y_pred_baseline.sum())} total baseline flags")

    y_pred_filtered = y_pred_baseline.copy()
    y_pred_filtered[only_mac_ip_elevated] = 0

    metrics = evaluate(y_test, y_pred_filtered)
    total_time = time.time() - t_start
    print("\n--- Results (Experiment 3: Explanation-Informed Post-Filter) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\ntotal_runtime: {total_time:.1f}s (no training -- reuses existing models/arrays)")
    print("\nBaseline (no post-filter): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
