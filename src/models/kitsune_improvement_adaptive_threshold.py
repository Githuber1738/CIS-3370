"""IMP-09 -- adaptive local threshold (final experiment before locking).

Every prior experiment changed the MODEL (architecture, algorithm,
features, training data). This changes nothing about the model -- it
reuses the already-trained baseline (`models/kitsune_os_scan_autoencoder
.keras`, no retraining) and replaces the single global threshold
(mean + 2*std of TRAIN reconstruction error, fixed for the whole test
set) with a rolling LOCAL threshold: local_mean(error) + k*local_std
(error), computed causally over a trailing window of the error stream
itself (train tail feeding the warm-up of test, never future rows).

Rationale: our root cause is that benign rows inside the attack window
have persistently elevated reconstruction error relative to the
GLOBAL (pre-attack) baseline. A local threshold that adapts to each
region's own recent error level could, in principle, stop flagging
that persistent elevation as anomalous -- at the cost of also being
less sensitive to an attack that itself causes a slow, sustained
elevation (which IMP-07's delta-feature result already showed is
exactly how this attack behaves). That tension is the whole point of
running this: either it helps despite the tension, or it confirms the
same sustained-elevation shape defeats this approach too.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate

SPLIT_INDEX = 1_000_000
WINDOW = 5000


def main():
    print("=== IMP-09: Adaptive Local Threshold (no retraining) ===")
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

    model_path = MODEL_DIR / "kitsune_os_scan_autoencoder.keras"
    print(f"Loading existing baseline model from {model_path} (no retraining)...")
    autoencoder = keras.models.load_model(model_path)

    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    global_threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)

    # Static global-threshold baseline, recomputed here for a same-run sanity check.
    y_pred_static = (test_error > global_threshold).astype(int)
    static_metrics = evaluate(y_test, y_pred_static)
    print("\n--- Sanity check: static global threshold (should match original baseline) ---")
    for k in ("precision", "recall", "fpr"):
        print(f"{k}: {static_metrics[k]:.4f}")

    # Rolling local threshold: causal trailing window over the FULL
    # chronological error stream (train tail warms up test's early rows,
    # never future rows).
    full_error = pd.Series(np.concatenate([train_error, test_error]))
    local_mean = full_error.rolling(window=WINDOW, min_periods=1).mean()
    local_std = full_error.rolling(window=WINDOW, min_periods=1).std().fillna(0.0)
    local_mean_test = local_mean.to_numpy()[SPLIT_INDEX:]
    local_std_test = local_std.to_numpy()[SPLIT_INDEX:]

    def adaptive_metrics(k):
        local_threshold = local_mean_test + k * local_std_test
        y_pred = (test_error > local_threshold).astype(int)
        return evaluate(y_test, y_pred)

    print(f"\n--- Adaptive local threshold (window={WINDOW}, k=2, matching the static convention) ---")
    m_k2 = adaptive_metrics(2.0)
    for key, v in m_k2.items():
        print(f"{key}: {v:.4f}" if isinstance(v, float) else f"{key}: {v}")

    # Sweep k to find the operating point closest to the paper's FPR=0.001,
    # for direct comparability with every other experiment's "recall @
    # FPR=0.001" column.
    print("\n--- Sweeping k to find FPR closest to 0.001 ---")
    best_k, best_gap, best_metrics = None, None, None
    for k in np.arange(0.5, 15.01, 0.25):
        m = adaptive_metrics(k)
        gap = abs(m["fpr"] - 0.001)
        if best_gap is None or gap < best_gap:
            best_k, best_gap, best_metrics = k, gap, m
    print(f"best k: {best_k} | fpr: {best_metrics['fpr']:.4f} | recall: {best_metrics['recall']:.4f} "
          f"| precision: {best_metrics['precision']:.4f}")

    print("\nBaseline (static global threshold): Precision 0.500 / Recall 1.000 / FPR 0.104, recall@FPR=0.001: 0.0191")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
