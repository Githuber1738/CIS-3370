"""ROUND 3, EXPERIMENT 3.2 -- conditional autoencoder (global network context).

Rather than asking the autoencoder to reconstruct benign-but-disturbed
traffic as if it were ordinary (what the targeted-augmentation experiment
tried, with a 4-row effect out of 71,848), this gives the model the
recent-context signal directly as an input, so it can learn "given that
recent traffic has looked like THIS, normal right now looks like X" --
conditioning the reconstruction on ambient state instead of asking the
model to generalize past what training ever showed it.

We don't have raw subnet-wide telemetry beyond Kitsune's own 115
extracted features, so the "global context" vector is a rolling mean of
those same 115 (scaled) features over the preceding W rows (W=50),
computed from real historical rows only (never the current or future
row). Input to the network is [current_row (115) ++ context (115)] = 230
dims; output is only the current row's 115 features (asymmetric
autoencoder) -- the context conditions the reconstruction, it isn't
itself reconstructed. Same hyperparameters and chronological split as
the baseline otherwise.
"""

import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate

SPLIT_INDEX = 1_000_000
RANDOM_SEED = 42
CONTEXT_WINDOW = 50


def build_conditional_autoencoder(current_dim, context_dim):
    total_dim = current_dim + context_dim
    inp = keras.Input(shape=(total_dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    out = keras.layers.Dense(current_dim, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def rolling_mean_context(X_scaled, window):
    """context[i] = mean of X_scaled[i-window:i] (strictly historical, real
    rows only). Rows before `window` get a partial-window average; row 0
    gets zeros (no history)."""
    n, d = X_scaled.shape
    cumsum = np.vstack([np.zeros((1, d)), np.cumsum(X_scaled, axis=0)])  # cumsum[k] = sum of rows 0..k-1
    context = np.zeros((n, d), dtype=X_scaled.dtype)
    # Rows >= window: fixed-size trailing window, fully vectorized.
    if n > window:
        context[window:] = (cumsum[window:n] - cumsum[0: n - window]) / window
    # Rows 1..window-1: partial window (row 0 stays zero, no history).
    for i in range(1, min(window, n)):
        context[i] = cumsum[i] / i
    return context


def main():
    t_start = time.time()
    print("=== Round 3, Experiment 3.2: Conditional Autoencoder (Global Context) ===")
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

    print(f"Computing rolling context (window={CONTEXT_WINDOW})...")
    X_full_s = np.vstack([X_train_s, X_test_s])
    context_full = rolling_mean_context(X_full_s, CONTEXT_WINDOW)
    context_train = context_full[:SPLIT_INDEX]
    context_test = context_full[SPLIT_INDEX:]

    X_train_cond = np.hstack([X_train_s, context_train])
    X_test_cond = np.hstack([X_test_s, context_test])
    print(f"conditioned input shape: train {X_train_cond.shape}, test {X_test_cond.shape}")

    print("Building and training conditional autoencoder...")
    t_train_start = time.time()
    import tensorflow as _tf
    keras.utils.set_random_seed(RANDOM_SEED)  # reproducibility fix: see docs/TASKS.md 2026-11-19
    _tf.config.experimental.enable_op_determinism()  # seed alone wasn't bit-exact (CPU float-reduction order); see Kitsune_Replication.md
    model = build_conditional_autoencoder(X_train_s.shape[1], context_train.shape[1])
    history = model.fit(X_train_cond, X_train_s, epochs=10, batch_size=5000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "kitsune_improvement_conditional.keras")

    train_recon = model.predict(X_train_cond, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = model.predict(X_test_cond, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print(f"\nthreshold: {threshold}")
    print("\n--- Results (Experiment 3.2: Conditional Autoencoder) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    benign_error = test_error[y_test == 0]
    forced_threshold = np.percentile(benign_error, 100 * (1 - 0.001))
    y_pred_forced = (test_error > forced_threshold).astype(int)
    forced_metrics = evaluate(y_test, y_pred_forced)
    print(f"\n--- Recall at forced FPR=0.001 (threshold={forced_threshold:.6f}) ---")
    print(f"recall: {forced_metrics['recall']:.4f} | fpr: {forced_metrics['fpr']:.4f}")

    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nBaseline (no conditioning): Precision 0.500 / Recall 1.000 / FPR 0.104, recall@FPR=0.001: 0.0191")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
