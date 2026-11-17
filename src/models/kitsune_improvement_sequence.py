"""IMPROVEMENT EXPERIMENT 2 -- sequence/temporal modeling.

Every detector tried in the replication phase (monolithic AE, real
KitNET, Isolation Forest) scores each row independently. This experiment
tests a genuinely different signal source: an LSTM autoencoder over a
sliding window of W consecutive rows, reconstructing the window and
scoring anomaly by reconstruction error on the window's LAST timestep
(i.e. "how anomalous is this row, given its preceding W-1 rows of
context"). The hypothesis: a true attack row is surrounded by other
attack rows, while an incidental attack-adjacent benign row is mostly
surrounded by benign rows -- a signal no per-row detector can see.

Uses numpy's sliding_window_view (zero-copy) to build windows without
materializing a full copy. Training windows are subsampled (stride) to
keep LSTM training time reasonable; test windows use every row (stride
1) since full per-row metrics are needed for a fair comparison.
"""

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate

SPLIT_INDEX = 1_000_000
WINDOW = 5
TRAIN_STRIDE = 5  # subsample training windows to keep LSTM training tractable
LSTM_UNITS = 32


def build_lstm_autoencoder(window, n_features, units=LSTM_UNITS):
    inp = keras.Input(shape=(window, n_features))
    encoded = keras.layers.LSTM(units)(inp)
    x = keras.layers.RepeatVector(window)(encoded)
    x = keras.layers.LSTM(units, return_sequences=True)(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(n_features, activation="sigmoid"))(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def main():
    t_start = time.time()
    print("=== Experiment 2: Sequence/Temporal Modeling (LSTM autoencoder) ===")
    print("Loading Kitsune OS Scan data...")
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
    # full_windows[i] covers rows [i, i+WINDOW-1]; window i ends at original row i+WINDOW-1.

    train_windows_all = full_windows[: SPLIT_INDEX - WINDOW + 1]
    train_windows = np.ascontiguousarray(train_windows_all[::TRAIN_STRIDE])
    print(f"train windows: {len(train_windows_all)} available, using {len(train_windows)} "
          f"(stride={TRAIN_STRIDE})")

    # Test windows: window i ends at row SPLIT_INDEX+i for i in range(len(y_test)).
    test_start = SPLIT_INDEX - WINDOW + 1
    test_windows = np.ascontiguousarray(full_windows[test_start: test_start + len(y_test)])
    print(f"test windows: {len(test_windows)} (should equal {len(y_test)}: {len(test_windows) == len(y_test)})")

    print("Building and training LSTM autoencoder...")
    t_train_start = time.time()
    model = build_lstm_autoencoder(WINDOW, n_features)
    history = model.fit(train_windows, train_windows, epochs=10, batch_size=2000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "kitsune_improvement_sequence.keras")

    # Threshold from training windows' last-timestep reconstruction error (leak-free).
    train_recon = model.predict(train_windows, batch_size=5000, verbose=0)
    train_error = np.mean(np.square(train_windows[:, -1, :] - train_recon[:, -1, :]), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    print(f"threshold: {threshold}")

    test_recon = model.predict(test_windows, batch_size=5000, verbose=0)
    test_error = np.mean(np.square(test_windows[:, -1, :] - test_recon[:, -1, :]), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print("\n--- Results (Experiment 2: Sequence/Temporal Modeling) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nBaseline (per-row monolithic AE): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
