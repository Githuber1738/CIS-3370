"""US-07: Average Descriptive Accuracy (ADA) and Mass Around Zero (MAZ),
reproducing xNIDS Section 6.1.1/6.1.2's exact methodology against our own
Kitsune autoencoder + sparse-group-lasso explanation pipeline.

ADA (fidelity, Fig. 3 / Table 3 in the paper):
    DA_k(x, f) = f(x | modify(k)). For a correctly-detected ANOMALY sample,
    modify(k) zeroes the k features the explanation ranked most important
    (by |coefficient|); ADA(k) is the percentage of such samples still
    classified anomalous after that, averaged over the sample population,
    for k = 0..10. A steep decline means the explanation found features
    that actually drive the detection.

Benign-sample side check (Table 4 in the paper, included here as a bonus,
not one of the two headline curves): for a correctly-classified BENIGN
sample, modify(k) replaces its top-k important features with the
corresponding features from its nearest anomaly sample (zeroing benign
features barely moves the score -- this is the paper's own finding, not
an assumption). We report the resulting "anomalous rate" (fraction newly
misclassified) at k=10, split by whether the benign sample sits inside or
outside the OS Scan attack span, since that's the specific question this
project cares about.

MAZ (sparsity, Fig. 4 / Table 5 in the paper):
    Per-sample explanation coefficients, |coef| scaled to [0,1] within
    each sample (the sample's own top feature = 1.0). Pooled across every
    feature of every explained sample, a half-normal distribution is
    fit to the resulting histogram; MAZ(c) = that half-normal's CDF at
    interval size c in [0,1]. A curve that rises steeply near c=0 (most
    mass near zero) means a sparse, more usable explanation.

Both reuse the existing trained model (models/kitsune_os_scan_autoencoder
.keras, no retraining) and the existing KitsuneExplanation class
(src/explain/kitsune_explanation.py) exactly as already used for the
Troubleshooting Showcase (US-08) -- this just runs it over a larger,
stratified sample population instead of the single hand-picked TP/FP pair,
so "average" in ADA is actually an average over something.

Scope, stated plainly: this evaluates OUR xNIDS-style explanation against
OUR own detector. It does not include a baseline-explainer comparison
(LIME/SHAP/etc, the paper's Tables 3/5 columns) -- that is a separate,
larger undertaking explicitly deferred (see Kitsune_Explanation_Fidelity.md
for the reasoning) given the Dec 1/3 timeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.explain.kitsune_explanation import KitsuneExplanation
from src.ingestion.kitsune import load_kitsune

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "kitsune_os_scan_autoencoder.keras"
OUT_DIR = PROJECT_ROOT / "artifacts" / "kitsune_fidelity_sparsity"
SPLIT_INDEX = 1_000_000
HISTORY_LEN = 10
MAX_K = 10

# OS Scan attack span, absolute dataset row indices (see Kitsune_Replication.md,
# "Root-Cause Diagnostics"): first/last attack-labeled row.
ATTACK_SPAN = (1_314_658, 1_576_578)

N_ANOMALY = 20
N_BENIGN_IN_WINDOW = 15
N_BENIGN_OUT_WINDOW = 15


def load_scored_test_set():
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

    autoencoder = keras.models.load_model(MODEL_PATH)
    train_recon = autoencoder.predict(X_train_s, verbose=0)
    train_error = np.mean(np.square(X_train_s - train_recon), axis=1)
    threshold = np.mean(train_error) + 2 * np.std(train_error)

    test_recon = autoencoder.predict(X_test_s, verbose=0)
    test_error = np.mean(np.square(X_test_s - test_recon), axis=1)
    y_pred = (test_error > threshold).astype(int)
    return X_test_s, y_test, y_pred, test_error, threshold, feature_cols


def select_population(X_test_s, y_test, y_pred, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    abs_row = np.arange(SPLIT_INDEX, SPLIT_INDEX + n)
    in_window = (abs_row >= ATTACK_SPAN[0]) & (abs_row <= ATTACK_SPAN[1])

    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0]
    tn_in_idx = np.where((y_pred == 0) & (y_test == 0) & in_window)[0]
    tn_out_idx = np.where((y_pred == 0) & (y_test == 0) & ~in_window)[0]

    anomaly_sample = rng.choice(tp_idx, size=min(N_ANOMALY, len(tp_idx)), replace=False)
    benign_in_sample = rng.choice(tn_in_idx, size=min(N_BENIGN_IN_WINDOW, len(tn_in_idx)), replace=False)
    benign_out_sample = rng.choice(tn_out_idx, size=min(N_BENIGN_OUT_WINDOW, len(tn_out_idx)), replace=False)

    print(f"population pools: {len(tp_idx)} TP, {len(tn_in_idx)} TN-in-window, "
          f"{len(tn_out_idx)} TN-out-of-window")
    print(f"selected: {len(anomaly_sample)} anomaly, {len(benign_in_sample)} benign-in-window, "
          f"{len(benign_out_sample)} benign-out-of-window")
    return anomaly_sample, benign_in_sample, benign_out_sample


def explain_sample(idx, X_test_s, test_error, feature_cols):
    pos = int(idx)
    start = max(0, pos - HISTORY_LEN)
    window = X_test_s[start:pos + 1]
    current_sample = window[-1]
    history_samples = window[:-1]
    exp = KitsuneExplanation(current_sample, history_samples, MODEL_PATH,
                              float(test_error[pos]), feature_cols)
    exp.capture_relevant_history()
    exp.weighted_sampling(num_samples=100, seed=42)
    exp.sparse_group_lasso()
    return exp.coef  # 115-dim, signed


def score_batch(model, X):
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def ada_curve_anomaly(anomaly_idx, X_test_s, coefs, threshold, model):
    """Zero out top-k |coef| features per anomaly sample; ADA(k) = % still
    classified anomalous."""
    ada = []
    for k in range(MAX_K + 1):
        still_anomalous = []
        for idx, coef in zip(anomaly_idx, coefs):
            x = X_test_s[idx].copy()
            if k > 0:
                top_k = np.argsort(-np.abs(coef))[:k]
                x[top_k] = 0.0
            err = score_batch(model, x.reshape(1, -1))[0]
            still_anomalous.append(err > threshold)
        ada.append(100.0 * np.mean(still_anomalous))
    return np.array(ada)


def benign_anomalous_rate(benign_idx, X_test_s, coefs, threshold, model, attack_pool):
    """Replace top-10 |coef| features with the nearest attack sample's
    values; return the fraction newly classified anomalous (the paper's
    Table 4 'anomalous rate', at k=MAX_K)."""
    if len(benign_idx) == 0:
        return float("nan")
    flips = []
    for idx, coef in zip(benign_idx, coefs):
        x = X_test_s[idx].copy()
        dists = np.sum((attack_pool - x) ** 2, axis=1)
        nearest = attack_pool[np.argmin(dists)]
        top_k = np.argsort(-np.abs(coef))[:MAX_K]
        x[top_k] = nearest[top_k]
        err = score_batch(model, x.reshape(1, -1))[0]
        flips.append(err > threshold)
    return float(np.mean(flips))


def maz_curve(all_coefs):
    """Pool |coef| scaled to [0,1] per-sample (own max = 1.0) across every
    explained sample+feature; fit a half-normal, return its CDF curve."""
    scaled = []
    for coef in all_coefs:
        abs_coef = np.abs(coef)
        m = abs_coef.max()
        if m > 0:
            scaled.append(abs_coef / m)
    pooled = np.concatenate(scaled)
    params = stats.halfnorm.fit(pooled, floc=0)
    xs = np.linspace(0, 1, 41)
    cdf = stats.halfnorm.cdf(xs, *params)
    auc = float(np.trapezoid(cdf, xs))
    return xs, cdf, auc, params


def main():
    print("=== US-07: ADA (fidelity) and MAZ (sparsity) ===")
    print("Loading model + scoring full test set (no retraining)...")
    X_test_s, y_test, y_pred, test_error, threshold, feature_cols = load_scored_test_set()
    model = keras.models.load_model(MODEL_PATH)

    anomaly_idx, benign_in_idx, benign_out_idx = select_population(X_test_s, y_test, y_pred)
    attack_idx_all = np.where(y_test == 1)[0]
    attack_pool = X_test_s[attack_idx_all]

    print("\nRunning explanations for anomaly samples...")
    anomaly_coefs = [explain_sample(i, X_test_s, test_error, feature_cols) for i in anomaly_idx]
    print("Running explanations for benign-in-window samples...")
    benign_in_coefs = [explain_sample(i, X_test_s, test_error, feature_cols) for i in benign_in_idx]
    print("Running explanations for benign-out-of-window samples...")
    benign_out_coefs = [explain_sample(i, X_test_s, test_error, feature_cols) for i in benign_out_idx]

    print("\nComputing ADA curve (anomaly samples, zero-out)...")
    ada = ada_curve_anomaly(anomaly_idx, X_test_s, anomaly_coefs, threshold, model)
    ada_auc = float(np.trapezoid(ada / 100.0, np.linspace(0, 1, MAX_K + 1)))
    print(f"ADA(k): {np.round(ada, 1).tolist()}")
    print(f"ADA AUC (normalized k-axis): {ada_auc:.3f}  (paper's xNIDS/Kitsune: 0.316)")

    print("\nComputing benign anomalous-rate at k=10 (replace-with-nearest-anomaly)...")
    rate_in = benign_anomalous_rate(benign_in_idx, X_test_s, benign_in_coefs, threshold, model, attack_pool)
    rate_out = benign_anomalous_rate(benign_out_idx, X_test_s, benign_out_coefs, threshold, model, attack_pool)
    print(f"anomalous rate, in-window benign: {rate_in:.3f}")
    print(f"anomalous rate, out-of-window benign: {rate_out:.3f}")
    print("(paper's Table 4, Kitsune, replacement operation: 0.239)")

    print("\nComputing MAZ curve (pooled over all explained samples)...")
    all_coefs = anomaly_coefs + benign_in_coefs + benign_out_coefs
    xs, cdf, maz_auc, params = maz_curve(all_coefs)
    print(f"MAZ AUC: {maz_auc:.3f}  (paper's xNIDS/Kitsune: 0.774)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": range(MAX_K + 1), "ada_pct": ada}).to_csv(OUT_DIR / "ada_curve.csv", index=False)
    pd.DataFrame({"interval_size": xs, "maz_cdf": cdf}).to_csv(OUT_DIR / "maz_curve.csv", index=False)
    results = {
        "n_anomaly": len(anomaly_idx), "n_benign_in_window": len(benign_in_idx),
        "n_benign_out_window": len(benign_out_idx),
        "ada_curve_pct": ada.tolist(), "ada_auc": ada_auc,
        "benign_anomalous_rate_in_window_k10": rate_in,
        "benign_anomalous_rate_out_window_k10": rate_out,
        "maz_auc": maz_auc, "maz_halfnorm_scale": float(params[1]),
        "paper_reference": {"ada_auc_kitsune_xnids": 0.316, "maz_auc_kitsune_xnids": 0.774,
                             "table4_replacement_kitsune": 0.239},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'ada_curve.csv'}, {OUT_DIR / 'maz_curve.csv'}, {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
