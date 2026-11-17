"""IMPROVEMENT EXPERIMENT 1 -- surgical data augmentation.

Unlike replication Attempt #1 (blind resampling of REAL benign rows from
across the full timeline, which fixed FPR but collapsed recall because it
exposed the model to real attack-adjacent rows), this experiment never
touches real attack-window data at all. Instead it synthesizes "benign-
but-unusual" examples by adding controlled Gaussian noise to a subset of
the real training-only benign rows (all from the pre-attack period), then
trains on original + synthetic rows together.

Same chronological split, scaling, and architecture as the replicated
baseline -- only the training SET composition changes (more rows, same
source distribution plus noise), isolating this one variable.
"""

import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, build_autoencoder, evaluate

SPLIT_INDEX = 1_000_000
AUGMENT_FRACTION = 0.2  # fraction of training rows to augment
NOISE_STD = 0.02  # in the [0,1]-scaled feature space
RANDOM_SEED = 42


def main():
    t_start = time.time()
    print("=== Experiment 1: Surgical Data Augmentation ===")
    print("Loading Kitsune OS Scan data...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]
    assert y[:SPLIT_INDEX].sum() == 0, "training split must be attack-free"

    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    rng = np.random.default_rng(RANDOM_SEED)
    n_augment = int(len(X_train_s) * AUGMENT_FRACTION)
    augment_idx = rng.choice(len(X_train_s), size=n_augment, replace=False)
    noise = rng.normal(loc=0.0, scale=NOISE_STD, size=(n_augment, X_train_s.shape[1]))
    X_augmented = np.clip(X_train_s[augment_idx] + noise, 0.0, 1.0)

    X_train_combined = np.vstack([X_train_s, X_augmented])
    print(f"training set: {len(X_train_s)} original + {len(X_augmented)} synthetic "
          f"= {len(X_train_combined)} total (noise_std={NOISE_STD}, fraction={AUGMENT_FRACTION})")

    print("Building and training autoencoder...")
    t_train_start = time.time()
    autoencoder = build_autoencoder(X_train_combined.shape[1])
    history = autoencoder.fit(X_train_combined, X_train_combined, epochs=10, batch_size=5000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "kitsune_improvement_augmentation.keras"
    autoencoder.save(model_path)

    # Threshold calibrated on ORIGINAL (non-augmented) train rows only, for a fair
    # comparison with every other experiment's threshold convention.
    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print(f"\nthreshold: {threshold}")
    print("\n--- Results (Experiment 1: Surgical Data Augmentation) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nBaseline (replicated, no augmentation): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
