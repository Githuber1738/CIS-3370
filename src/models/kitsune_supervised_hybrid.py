"""IMPROVEMENT ATTEMPT #5 -- a supervised hybrid using per-channel
reconstruction error as features, motivated directly by the explanation
findings in src/explain/ (MAC-IP dominates importance in both true and
false positives, suggesting the 5 channels carry different separating
power that a single global threshold can't use).

This is an explicit, flagged departure from the paper's fully
unsupervised design: it uses labels at the scoring stage, not just for
evaluation. The underlying detectors are still trained unsupervised,
benign-only (the per-group autoencoders from kitsune_ensemble.py) --
only the final decision boundary becomes supervised.

To avoid the obvious leakage of "tune and evaluate on the same test
rows," the test set itself is split in half: a labeled "tune" half fits
a small classifier on the 5 per-channel RMSE features, and the other,
disjoint "report" half is the only one the final metrics are computed
on. Requires kitsune_train_group_rmse.npy / kitsune_test_group_rmse.npy
from kitsune_ensemble.py to have already been generated.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_INDEX = 1_000_000


def main():
    test_rmse = np.load(PROJECT_ROOT / "artifacts" / "kitsune_test_group_rmse.npy")

    df = load_kitsune()
    y = df["label"].to_numpy()
    y_test = y[SPLIT_INDEX:]
    assert len(y_test) == len(test_rmse)

    tune_idx, report_idx = train_test_split(
        np.arange(len(y_test)), test_size=0.5, stratify=y_test, random_state=42
    )

    X_tune, y_tune = test_rmse[tune_idx], y_test[tune_idx]
    X_report, y_report = test_rmse[report_idx], y_test[report_idx]

    print(f"tune: {len(X_tune)} rows ({int(y_tune.sum())} attack) | "
          f"report: {len(X_report)} rows ({int(y_report.sum())} attack)")

    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(X_tune, y_tune)

    y_pred = clf.predict(X_report)
    metrics = evaluate(y_report, y_pred)
    print("\n--- Results (supervised hybrid, held-out report half) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nchannel coefficients (MAC-IP, IP, Jitter, IPtoIP, Socket):", clf.coef_[0])

    print("\nBaseline (unsupervised ensemble): see kitsune_ensemble.py output")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
