"""ROUND 3, EXPERIMENT 3.1 -- differential (delta) feature transformation.

Every prior experiment fed the autoencoder absolute feature values. If the
attack-window disturbance shows up as a roughly uniform scale shift across
features (ambient chatter raising packet rates/sizes broadly), first-
differencing (feature_t - feature_{t-1}) could cancel that shift out while
leaving genuine per-row anomalies (which change the RELATIONSHIP between
consecutive rows, not just their absolute level) intact.

Deltas are computed on RAW (unscaled) features, row i minus row i-1, which
loses row 0 (no predecessor) -- negligible out of 1,697,851 rows. Scaled
with a fresh RobustScaler+MinMaxScaler fit on the deltas themselves (delta
values are centered near zero and have a different distribution than the
absolute features), same train+test-features-only-never-labels convention
as everywhere else in this project. Same architecture, hyperparameters,
and chronological split as the baseline -- isolates the feature
transformation as the one variable.
"""

import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, build_autoencoder, evaluate

SPLIT_INDEX = 1_000_000
RANDOM_SEED = 42


def main():
    t_start = time.time()
    print("=== Round 3, Experiment 3.1: Differential (Delta) Features ===")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    # First differences: row i -> X[i] - X[i-1]. Drop row 0 (no predecessor).
    X_delta = np.diff(X, axis=0)
    y_delta = y[1:]  # label of the CURRENT row (the later of the pair)
    # SPLIT_INDEX shifts by 1 since we dropped the first row.
    split = SPLIT_INDEX - 1

    X_train, X_test = X_delta[:split], X_delta[split:]
    y_test = y_delta[split:]
    assert y_delta[:split].sum() == 0, "training split must be attack-free"
    print(f"train: {len(X_train)} delta-rows | test: {len(X_test)} delta-rows "
          f"({int(y_test.sum())} attack)")

    robust = RobustScaler().fit(X_delta)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    print("Building and training autoencoder on delta features...")
    t_train_start = time.time()
    import tensorflow as _tf
    from tensorflow import keras as _keras
    _keras.utils.set_random_seed(RANDOM_SEED)  # reproducibility fix: see docs/TASKS.md 2026-11-19
    _tf.config.experimental.enable_op_determinism()  # seed alone wasn't bit-exact (CPU float-reduction order); see Kitsune_Replication.md
    autoencoder = build_autoencoder(X_train_s.shape[1])
    history = autoencoder.fit(X_train_s, X_train_s, epochs=10, batch_size=5000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    autoencoder.save(MODEL_DIR / "kitsune_improvement_delta_features.keras")

    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print(f"\nthreshold: {threshold}")
    print("\n--- Results (Experiment 3.1: Delta Features) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Also report recall at the paper's exact FPR=0.001 operating point, for
    # direct comparability with the Addendum finding in Kitsune_Replication.md.
    benign_error = test_error[y_test == 0]
    forced_threshold = np.percentile(benign_error, 100 * (1 - 0.001))
    y_pred_forced = (test_error > forced_threshold).astype(int)
    forced_metrics = evaluate(y_test, y_pred_forced)
    print(f"\n--- Recall at forced FPR=0.001 (threshold={forced_threshold:.6f}) ---")
    print(f"recall: {forced_metrics['recall']:.4f} | fpr: {forced_metrics['fpr']:.4f}")

    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nBaseline (absolute features): Precision 0.500 / Recall 1.000 / FPR 0.104, recall@FPR=0.001: 0.0191")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
