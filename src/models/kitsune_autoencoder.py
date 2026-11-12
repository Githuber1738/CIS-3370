"""Train and evaluate an autoencoder-based anomaly detector on Kitsune's
OS Scan scenario, reproducing the xNIDS paper's Kitsune row (Table 12).

Architecture and training hyperparameters (256-128-64-128-256 dense
autoencoder, dropout 0.1, Adam lr=1e-3, MSE loss, 10 epochs, batch 5000,
1M-row train/test split) are taken directly from the xNIDS authors'
Scripts/kitsune.py (github.com/CactiLab/code-xNIDS) to reproduce their
result as faithfully as possible.

One deliberate correction vs. that reference script, which is genuine
train/test leakage: the anomaly threshold (mean + 2*std of reconstruction
error) is computed from the *training* set's reconstruction error, then
applied to the test set -- the reference script computes it from the
test set's own reconstruction error, using label-correlated information
(the error distribution being thresholded) to threshold itself.

RobustScaler/MinMaxScaler ARE fit on train+test combined (matching the
reference script), not train-only. This was tried both ways: fitting on
train-only caused unclipped extrapolation on out-of-range test values,
producing reconstruction-error outliers as large as 1e26 (verified) --
a numerical artifact, not signal. Fitting on train+test (features only,
never labels) keeps every value in-range and removes that artifact, and,
when compared directly, does not meaningfully change precision/recall/FPR
either way -- so there's no leakage trade-off being made here, just a
numerically safer choice. This is NOT the same kind of leakage as #1
above: a scaler's fit statistics don't depend on which rows are attacks.

Even with both fixes, our result still diverges sharply from the paper's
Table 12 (Precision ~0.50 / Recall ~1.00 / FPR ~0.10 vs. 0.9568/0.9972/
0.001). Root cause, confirmed empirically (see docs/TASKS.md and the
Wiki): roughly 65,000 benign-labeled test rows produce elevated
reconstruction error almost independent of threshold choice (FPR barely
moves across threshold=0.001 to threshold=2.0). Those rows are
concentrated in the exact time window the attack occurs in (test rows
~279k-488k, matching the attack's own 315k-577k range) -- the model
never saw this network condition in its training split (the first
1M rows, which predate the attack entirely), so no amount of
hyperparameter tuning closes this; it's a training-data coverage gap
inherent to the paper's own prescribed chronological split.
"""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
SPLIT_INDEX = 1_000_000
RANDOM_SEED = 42


def build_autoencoder(input_dim: int):
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam

    input_data = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation="relu")(input_data)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    decoded = keras.layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return autoencoder


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    print("Loading Kitsune OS Scan data...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]

    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]
    assert y[:SPLIT_INDEX].sum() == 0, "training split must be attack-free"

    print(f"train: {len(X_train)} rows (all benign) | test: {len(X_test)} rows "
          f"({int(y_test.sum())} attack, {len(y_test) - int(y_test.sum())} benign)")

    # Fit on train+test features combined (never labels) -- see module
    # docstring for why train-only fitting caused extrapolation blowups.
    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)

    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    print("Building and training autoencoder...")
    autoencoder = build_autoencoder(X_train_s.shape[1])
    autoencoder.fit(X_train_s, X_train_s, epochs=10, batch_size=5000, verbose=2)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "kitsune_os_scan_autoencoder.keras"
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")

    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    print(f"Anomaly threshold (from train reconstruction error): {threshold:.6f}")

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    print("\n--- Results (OS Scan test split) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nxNIDS paper's reported Kitsune row (Table 12): "
          "Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
