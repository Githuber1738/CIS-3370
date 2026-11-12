"""Train and evaluate an autoencoder-based anomaly detector on Kitsune's
OS Scan scenario, reproducing the xNIDS paper's Kitsune row (Table 12).

Architecture and training hyperparameters (256-128-64-128-256 dense
autoencoder, dropout 0.1, Adam lr=1e-3, MSE loss, 10 epochs, batch 5000,
1M-row train/test split) are taken directly from the xNIDS authors'
Scripts/kitsune.py (github.com/CactiLab/code-xNIDS) to reproduce their
result as faithfully as possible.

Two deliberate corrections vs. that reference script, both train/test
leakage:
  1. RobustScaler and MinMaxScaler are fit on X_train only, then applied to
     X_test -- the reference script fits both scalers on the full dataset
     (train+test combined), which leaks test-set statistics into
     preprocessing.
  2. The anomaly threshold (mean + 2*std of reconstruction error) is
     computed from the *training* set's reconstruction error, then applied
     to the test set -- the reference script computes it from the test
     set's own reconstruction error, which uses label-correlated
     information (the error distribution being thresholded) to threshold
     itself.
These corrections may shift results slightly from the paper's reported
numbers; that's expected and worth noting in the Wiki report rather than
silently matching the leaky version.
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

    robust = RobustScaler().fit(X_train)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)

    minmax = MinMaxScaler().fit(X_train_s)
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
