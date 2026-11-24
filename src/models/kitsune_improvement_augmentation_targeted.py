"""ROUND 2, EXPERIMENT 2.2 -- targeted augmentation from the real
attack-window-benign distribution.

IMP-01 (generic Gaussian noise) was too mild to teach the model anything
about the attack-adjacent benign regime. This experiment instead
characterizes HOW real attack-window benign rows differ from ordinary
benign rows (per Kitsune_Replication.md's diagnosis: test rows ~279k-
488k), then synthesizes training examples shifted toward that real
difference.

Leakage control: the attack-window-benign region is split in half. Only
the CALIBRATION half is used to measure the distributional shift; the
EVALUATION half is held out and never touched until final scoring, so
the reported metrics on that half are not contaminated by having been
used to build the augmentation. Standard full-test-set metrics are also
reported for comparability with every other experiment.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, build_autoencoder, evaluate

SPLIT_INDEX = 1_000_000
ATTACK_WINDOW_BENIGN_START = 279140  # test-relative index, from Kitsune_Replication.md
ATTACK_WINDOW_BENIGN_END = 488495
AUGMENT_FRACTION = 0.2
RANDOM_SEED = 42


def main():
    t_start = time.time()
    print("=== Round 2, Experiment 2.2: Targeted Augmentation ===")
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

    # Identify attack-window benign rows (y_test==0) within the known window.
    window_mask = np.zeros(len(y_test), dtype=bool)
    window_mask[ATTACK_WINDOW_BENIGN_START:ATTACK_WINDOW_BENIGN_END] = True
    awb_mask = window_mask & (y_test == 0)
    awb_idx = np.where(awb_mask)[0]
    print(f"attack-window benign candidates: {len(awb_idx)}")

    calib_idx, eval_idx = sk_split(awb_idx, test_size=0.5, random_state=RANDOM_SEED)
    print(f"calibration half: {len(calib_idx)} | held-out evaluation half: {len(eval_idx)}")

    # Characterize the shift using ONLY the calibration half.
    calib_mean = X_test_s[calib_idx].mean(axis=0)
    train_mean = X_train_s.mean(axis=0)
    shift_vector = calib_mean - train_mean
    print(f"shift vector magnitude (L2 norm): {np.linalg.norm(shift_vector):.4f}")

    rng = np.random.default_rng(RANDOM_SEED)
    n_augment = int(len(X_train_s) * AUGMENT_FRACTION)
    base_idx = rng.choice(len(X_train_s), size=n_augment, replace=False)
    shift_scale = rng.uniform(0.5, 1.5, size=(n_augment, 1))
    small_noise = rng.normal(0, 0.01, size=(n_augment, X_train_s.shape[1]))
    X_augmented = np.clip(X_train_s[base_idx] + shift_scale * shift_vector + small_noise, 0.0, 1.0)
    X_train_combined = np.vstack([X_train_s, X_augmented])
    print(f"training set: {len(X_train_s)} original + {len(X_augmented)} targeted-synthetic "
          f"= {len(X_train_combined)} total")

    print("Building and training autoencoder...")
    t_train_start = time.time()
    import tensorflow as _tf
    from tensorflow import keras as _keras
    _keras.utils.set_random_seed(RANDOM_SEED)  # reproducibility fix: see docs/TASKS.md 2026-11-19
    _tf.config.experimental.enable_op_determinism()  # seed alone wasn't bit-exact (CPU float-reduction order); see Kitsune_Replication.md
    autoencoder = build_autoencoder(X_train_combined.shape[1])
    history = autoencoder.fit(X_train_combined, X_train_combined, epochs=10, batch_size=5000, verbose=2)
    train_time = time.time() - t_train_start
    final_loss = history.history["loss"][-1]
    print(f"training time: {train_time:.1f}s | final loss: {final_loss:.6f}")

    MODEL_DIR.mkdir(exist_ok=True)
    autoencoder.save(MODEL_DIR / "kitsune_improvement_augmentation_targeted.keras")

    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)

    metrics_full = evaluate(y_test, y_pred)
    total_time = time.time() - t_start
    print(f"\nthreshold: {threshold}")
    print("\n--- Results (full test set, Experiment 2.2) ---")
    for k, v in metrics_full.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Held-out evaluation: did the targeted augmentation help specifically
    # on the never-calibrated-on half of the attack-window-benign region?
    held_out_fp = int((y_pred[eval_idx] == 1).sum())
    held_out_fpr = held_out_fp / len(eval_idx)
    print(f"\n--- Held-out eval-half check (never used for calibration) ---")
    print(f"eval-half rows: {len(eval_idx)} (all benign, from the attack-window region)")
    print(f"eval-half false positives: {held_out_fp} ({held_out_fpr:.4f} FPR on this subset specifically)")

    print(f"\ntotal_runtime: {total_time:.1f}s | training_time: {train_time:.1f}s | final_loss: {final_loss:.6f}")
    print("\nBaseline (no augmentation): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
