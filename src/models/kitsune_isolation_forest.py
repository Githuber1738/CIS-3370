"""IMPROVEMENT ATTEMPT #3 -- tests whether the overlap found in Attempts 1-2
is specific to autoencoder reconstruction error as a scoring mechanism, or
baked into the feature space regardless of algorithm.

Same chronological split and scaling as the faithful reproduction
(src/models/kitsune_autoencoder.py); only the detector changes, from a
dense autoencoder to an Isolation Forest. Threshold derived the same
leak-free way (mean + 2*std of anomaly score on the training set only).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import evaluate

SPLIT_INDEX = 1_000_000


def main():
    print("Loading Kitsune OS Scan data...")
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

    print(f"Training Isolation Forest on {len(X_train_s)} benign rows...")
    clf = IsolationForest(n_estimators=100, max_samples=min(256, len(X_train_s)),
                           contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X_train_s)

    train_scores = -clf.score_samples(X_train_s)
    threshold = np.mean(train_scores) + 2 * np.std(train_scores)
    test_scores = -clf.score_samples(X_test_s)
    y_pred = (test_scores > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    print(f"threshold: {threshold}")
    print("\n--- Results (Isolation Forest, chronological split) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nBaseline (autoencoder, same split): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
