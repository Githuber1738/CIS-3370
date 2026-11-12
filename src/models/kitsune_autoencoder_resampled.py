"""IMPROVEMENT ATTEMPT -- not a reproduction of the paper's methodology.

src/models/kitsune_autoencoder.py faithfully reproduces xNIDS's Kitsune
setup, including its chronological train/test split (first 1M rows =
train, rest = test). We diagnosed that split as the actual cause of our
precision/FPR gap vs. Table 12: the attack happens entirely after row 1M,
so the model's training data contains zero examples of benign traffic
occurring near/during a scan, and ~65,000 such rows in the test set get
flagged as anomalous purely because the model has never seen that
network condition, not because they're genuinely hard to tell apart.

This script tests one fix for that specific gap: train on a RANDOM
sample of benign rows drawn from across the *entire* dataset (not just
the pre-attack period), so the model sees some attack-adjacent benign
traffic during training. Still trains only on benign rows (same
unsupervised anomaly-detection design as the original), still derives
the threshold from training reconstruction error only (no leakage) --
the only thing that changes is which benign rows end up in the training
set. Everything else (architecture, hyperparameters, scaler approach) is
held constant so this isolates exactly one variable against the
faithful-reproduction baseline.

This is a deliberate departure from the paper's own experimental design,
not a "more correct" reproduction -- document it as such.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, build_autoencoder, evaluate

RANDOM_SEED = 42
N_TRAIN_BENIGN = 1_000_000


def main():
    print("Loading Kitsune OS Scan data...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    print(f"total benign rows: {len(benign_idx)} | total attack rows: {len(attack_idx)}")

    rng = np.random.default_rng(RANDOM_SEED)
    train_idx = rng.choice(benign_idx, size=N_TRAIN_BENIGN, replace=False)
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[train_idx] = True
    test_idx = np.where(~train_mask)[0]  # remaining benign rows + ALL attack rows

    X_train, X_test = X[train_idx], X[test_idx]
    y_test = y[test_idx]
    print(f"train: {len(X_train)} rows (all benign, sampled across the full timeline) | "
          f"test: {len(X_test)} rows ({int(y_test.sum())} attack, {len(y_test)-int(y_test.sum())} benign)")

    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    print("Building and training autoencoder (same architecture as the baseline)...")
    autoencoder = build_autoencoder(X_train_s.shape[1])
    autoencoder.fit(X_train_s, X_train_s, epochs=10, batch_size=5000, verbose=2)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "kitsune_os_scan_autoencoder_resampled.keras"
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
    print("\n--- Results (random-benign-sample training) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nBaseline (chronological split, faithful reproduction): "
          "Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): "
          "Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
