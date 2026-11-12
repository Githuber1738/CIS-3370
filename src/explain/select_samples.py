"""Select real TP/FN/FP Kitsune test samples from our trained autoencoder,
for xNIDS-style explanation (direct analog to xNIDS's own
kitsune_selected_{tp,fn,fp}_rows.csv, used by their Scripts/explanation.py).

Scalers are not persisted by src/models/kitsune_autoencoder.py, but
RobustScaler/MinMaxScaler have no randomness -- refitting them on the same
X_train reproduces identical scaling every time. Only the autoencoder's
weights are non-deterministic across retrains, so we load the already
-saved model rather than retraining it here.

For each of TP / FN / FP, exports an 11-row CSV: the 10 rows immediately
preceding the selected event ("history") plus the event row itself
("current", the last row) -- unlike xNIDS's own export (X_test[k-10:k],
which excludes row k entirely, then their explanation.py treats the last
*history* row as "current", a mismatch in their own code). Ours keeps the
actual flagged row as the explained sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "kitsune_os_scan_autoencoder.keras"
OUT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_explain_samples"
SPLIT_INDEX = 1_000_000
HISTORY_LEN = 10


def main():
    print("Loading data and rebuilding the deterministic scaler fit...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]

    robust = RobustScaler().fit(X_train)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(X_train_s)
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    print("Loading trained model...")
    autoencoder = keras.models.load_model(MODEL_PATH)

    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    print(f"threshold (from train reconstruction error): {threshold:.6f}")

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    print(f"available: {len(tp_idx)} TP, {len(fn_idx)} FN, {len(fp_idx)} FP "
          f"(out of {len(y_test)} test rows)")

    # Pick a representative example of each: one with reconstruction error
    # closest to the median within its own category, so it's a typical
    # case rather than a rare extreme.
    def pick_representative(idx_array):
        errs = test_error[idx_array]
        median_err = np.median(errs)
        closest = idx_array[np.argmin(np.abs(errs - median_err))]
        return int(closest), float(test_error[closest])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = {}
    for name, idx_array in [("tp", tp_idx), ("fn", fn_idx), ("fp", fp_idx)]:
        if len(idx_array) == 0:
            print(f"no {name.upper()} samples found, skipping")
            continue
        pos, err = pick_representative(idx_array)
        start = max(0, pos - HISTORY_LEN)
        window = X_test_s[start:pos + 1]  # history rows + the event row itself
        out_df = pd.DataFrame(window, columns=feature_cols)
        out_path = OUT_DIR / f"kitsune_selected_{name}_rows.csv"
        out_df.to_csv(out_path, index=False)
        picks[name] = {"test_index": pos, "reconstruction_error": err, "rows": len(out_df)}
        print(f"{name.upper()}: test row {pos}, reconstruction_error={err:.6f}, "
              f"exported {len(out_df)} rows -> {out_path}")

    summary_path = OUT_DIR / "selection_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"threshold: {threshold}\n")
        for name, info in picks.items():
            f.write(f"{name}: {info}\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
