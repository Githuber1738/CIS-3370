"""DIAGNOSTIC: rules out overfitting-by-epoch-count as the cause of the
precision/FPR gap, rather than just inferring it.

Trains the same architecture on the same chronological split as
src/models/kitsune_autoencoder.py, but evaluates on the test set after
EVERY epoch (not just at the end), each time deriving its own threshold
from that epoch's train reconstruction error (leak-free, same method as
the baseline). If overfitting across epochs were the cause, we'd expect
an early epoch to generalize better (lower FPR without losing recall)
than epoch 10. If every epoch lands in the same place, that rules out
epoch count/overfitting as the explanation.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import build_autoencoder, evaluate, MODEL_DIR

SPLIT_INDEX = 1_000_000
N_EPOCHS = 15


class PerEpochEval(keras.callbacks.Callback):
    def __init__(self, X_train_s, X_test_s, y_test):
        super().__init__()
        self.X_train_s = X_train_s
        self.X_test_s = X_test_s
        self.y_test = y_test
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        train_recon = self.model.predict(self.X_train_s, verbose=0)
        train_error = np.mean(np.square(self.X_train_s - train_recon), axis=1)
        threshold = np.mean(train_error) + 2 * np.std(train_error)

        test_recon = self.model.predict(self.X_test_s, verbose=0)
        test_error = np.mean(np.square(self.X_test_s - test_recon), axis=1)
        y_pred = (test_error > threshold).astype(int)

        m = evaluate(self.y_test, y_pred)
        m["epoch"] = epoch + 1
        m["train_loss"] = logs.get("loss")
        m["threshold"] = threshold
        self.history.append(m)
        print(f"  [epoch {epoch+1:>2}] loss={logs.get('loss'):.6f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  fpr={m['fpr']:.4f}")


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

    print(f"Training for {N_EPOCHS} epochs, evaluating on the test set after each one...\n")
    autoencoder = build_autoencoder(X_train_s.shape[1])
    cb = PerEpochEval(X_train_s, X_test_s, y_test)
    autoencoder.fit(X_train_s, X_train_s, epochs=N_EPOCHS, batch_size=5000, verbose=0, callbacks=[cb])

    print("\n--- Full per-epoch results ---")
    print(f"{'epoch':>6}{'loss':>12}{'precision':>12}{'recall':>10}{'fpr':>10}{'f1':>8}")
    for m in cb.history:
        print(f"{m['epoch']:>6}{m['train_loss']:>12.6f}{m['precision']:>12.4f}"
              f"{m['recall']:>10.4f}{m['fpr']:>10.4f}{m['f1']:>8.4f}")

    best = max(cb.history, key=lambda m: m["f1"])
    print(f"\nBest F1: epoch {best['epoch']} (precision={best['precision']:.4f}, "
          f"recall={best['recall']:.4f}, fpr={best['fpr']:.4f})")
    print("For reference -- paper's Table 12: Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
