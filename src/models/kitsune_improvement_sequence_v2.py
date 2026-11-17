"""ROUND 2, EXPERIMENT 2.3 -- sequence/temporal modeling at a larger window.

Round 1's IMP-02 (window=5) was measurably WORSE than the per-row
baseline, not merely unchanged -- raising the question of whether 5 rows
simply isn't enough context, versus temporal context not helping at all
regardless of scale. This experiment tests a 3x larger window (15 rows)
with the same architecture and training regime as IMP-02, isolating
window size as the one variable.
"""

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate
from src.models.kitsune_improvement_sequence import build_lstm_autoencoder

SPLIT_INDEX = 1_000_000
WINDOW = 15
TRAIN_STRIDE = 5
LSTM_UNITS = 32


def main():
    t_start = time.time()
    print("=== Round 2, Experiment 2.3: Sequence Modeling, window=15 ===")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]

    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train).astype(np.float32)
    X_test_s = robust.transform(X_test).astype(np.float32)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s).astype(np.float32)
    X_test_s = minmax.transform(X_test_s).astype(np.float32)

    X_full_s = np.vstack([X_train_s, X_test_s])
    n_features = X_full_s.shape[1]

    print(f"Building sliding windows (window={WINDOW})...")
    full_windows = sliding_window_view(X_full_s, (WINDOW, n_features)).reshape(-1, WINDOW, n_features)

    train_windows_all = full_windows[: SPLIT_INDEX - WINDOW + 1]
    train_windows = np.ascontiguousarray(train_windows_all[::TRAIN_STRIDE])
    print(f"train windows: {len(train_windows_all)} available, using {len(train_windows)} (stride={TRAIN_STRIDE})")

    test_start = SPLIT_INDEX - WINDOW + 1
    test_windows = np.ascontiguousarray(full_windows[test_start: test_start + len(y_test)])
    print(f"test windows: {len(test_windows)} (matches y_test: {len(test_windows) == len(y_test)})")

    print("Building and training LSTM autoencoder...")
    t_train_start = time.time()
    model = build_lstm_autoencoder(WINDOW, n_features, units=LSTM_UNITS)
    history = model.fit(train_windows, train_windows, epochs=10, batch_size=2000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "kitsune_improvement_sequence_v2.keras")

    train_recon = model.predict(train_windows, batch_size=5000, verbose=0)
    train_error = np.mean(np.square(train_windows[:, -1, :] - train_recon[:, -1, :]), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    print(f"threshold: {threshold}")

    test_recon = model.predict(test_windows, batch_size=5000, verbose=0)
    test_error = np.mean(np.square(test_windows[:, -1, :] - test_recon[:, -1, :]), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print("\n--- Results (Experiment 2.3: Sequence Modeling, window=15) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nRound 1 window=5 result: Precision 0.4161 / Recall 1.0000 / FPR 0.1458")
    print("Baseline (per-row): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
