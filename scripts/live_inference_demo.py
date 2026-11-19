"""Visualization #2: live inference replay -- the one actually-live moment
for the presentation. Loads the already-trained baseline model (inference
only, no training) and streams a fixed ~10k-row slice spanning pre-attack
normal traffic into the start of the attack window, printing a running
Precision/Recall/FPR counter as it goes. Pure inference: fast, bounded,
deterministic -- nothing here can run long or fail unpredictably on stage.

Usage:
    python -m scripts.live_inference_demo              # paced, ~0.05s/row-batch
    python -m scripts.live_inference_demo --fast        # no pacing, for rehearsal
    python -m scripts.live_inference_demo --pace 0.1    # custom pacing
"""

import argparse
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_INDEX = 1_000_000
# Test-row slice: pre-attack normal (310,000-314,657) into the start of the
# attack window (314,658 onward) -- see Kitsune_Replication.md for the span.
SLICE_START, SLICE_END = 309_000, 319_000
BATCH_SIZE = 250

COLOR_RESET = "\033[0m"
COLOR_ATTACK = "\033[91m"
COLOR_BENIGN = "\033[94m"
COLOR_DIM = "\033[90m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="disable pacing delay")
    parser.add_argument("--pace", type=float, default=0.05, help="seconds of delay per batch")
    args = parser.parse_args()
    pace = 0.0 if args.fast else args.pace

    print("Loading data and the existing trained model (inference only, no training)...")
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
    print(f"Threshold: {threshold:.6f}\n")

    X_slice = X_test_s[SLICE_START:SLICE_END]
    y_slice = y_test[SLICE_START:SLICE_END]
    n = len(X_slice)
    print(f"Streaming test rows {SLICE_START}-{SLICE_END} ({n} rows, "
          f"{int(y_slice.sum())} attack) in batches of {BATCH_SIZE}...\n")

    tp = tn = fp = fn = 0
    entered_window = False
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x_batch = X_slice[start:end]
        y_batch = y_slice[start:end]
        recon = model.predict(x_batch, verbose=0)
        err = np.mean(np.square(x_batch - recon), axis=1)
        pred = (err > threshold).astype(int)

        tp += int(np.sum((pred == 1) & (y_batch == 1)))
        tn += int(np.sum((pred == 0) & (y_batch == 0)))
        fp += int(np.sum((pred == 1) & (y_batch == 0)))
        fn += int(np.sum((pred == 0) & (y_batch == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        if not entered_window and y_batch.sum() > 0:
            entered_window = True
            print(f"{COLOR_ATTACK}>>> entering attack window at test row {SLICE_START + start}{COLOR_RESET}")

        bar_color = COLOR_ATTACK if y_batch.sum() > 0 else COLOR_BENIGN
        print(f"{bar_color}rows {SLICE_START + start:>7}-{SLICE_START + end:>7}{COLOR_RESET}  "
              f"{COLOR_DIM}batch attacks={int(y_batch.sum()):>3} flags={int(pred.sum()):>3}{COLOR_RESET}  "
              f"running: precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f}")
        time.sleep(pace)

    print(f"\nFinal over this slice: precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f} "
          f"(tp={tp} tn={tn} fp={fp} fn={fn})")


if __name__ == "__main__":
    main()
