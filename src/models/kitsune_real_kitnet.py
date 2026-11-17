"""IMPROVEMENT ATTEMPT #6 -- the REAL KitNET algorithm (Mirsky et al.'s own
implementation, vendored in src/third_party/kitnet/), not xNIDS's
substitute monolithic autoencoder.

Motivated by Appendix C.1 of the xNIDS paper, which states Kitsune's one
reported hyperparameter is m=10 (max autoencoder size) -- a detail that
only makes sense for the real KitNET ensemble (online, incremental,
correlation-clustered), not the single 115-wide autoencoder in xNIDS's
own Scripts/kitsune.py. This is the first attempt that tests the paper's
actual documented setup rather than a substitute we designed ourselves.

Mechanics (see src/third_party/kitnet/README.md for the vendoring note):
- Feeds RAW (unscaled) features -- KitNET does its own online 0-1
  normalization internally, so pre-scaling would double-normalize.
- FM_grace_period=5000, AD_grace_period=50000 (the original repo's own
  documented example defaults) -- KitNET trains (feature-mapper, then
  ensemble) for the first 55,000 rows, then switches to execute-only.
- Processes ALL 1,697,851 rows in order via a single process() call per
  row (trains during grace, executes after) -- this is simpler and more
  faithful to how an online algorithm is actually used than manually
  splitting train/execute calls.
- Threshold: mean + 2*std of RMSE on rows [55000:1000000) -- benign,
  past the grace period, before the test region -- same leak-free
  convention used throughout this project. Rows [1000000:) are "test",
  same split as every other attempt, for direct comparability.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "third_party" / "kitnet"))
import KitNET as kit  # noqa: E402

from src.ingestion.kitsune import load_kitsune  # noqa: E402
from src.models.kitsune_autoencoder import evaluate  # noqa: E402

FM_GRACE = 5000
AD_GRACE = 50000
SPLIT_INDEX = 1_000_000
MAX_AE_SIZE = 10


def main():
    print("Loading Kitsune OS Scan data (raw, unscaled)...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    n = X.shape[1]

    print(f"Building KitNET: n={n}, m={MAX_AE_SIZE}, FM_grace={FM_GRACE}, AD_grace={AD_GRACE}")
    K = kit.KitNET(n=n, max_autoencoder_size=MAX_AE_SIZE,
                   FM_grace_period=FM_GRACE, AD_grace_period=AD_GRACE)

    rmse = np.zeros(len(X))
    print(f"Processing all {len(X)} rows (streaming, in order)...")
    t0 = time.time()
    for i in range(len(X)):
        rmse[i] = K.process(X[i])
        if i % 200000 == 0:
            print(f"  row {i}/{len(X)}  elapsed={time.time()-t0:.0f}s")
    print(f"Done. Total elapsed: {time.time()-t0:.0f}s")

    np.save(PROJECT_ROOT / "artifacts" / "kitsune_real_kitnet_rmse.npy", rmse)

    calib_start = FM_GRACE + AD_GRACE
    calibration = rmse[calib_start:SPLIT_INDEX]
    assert y[calib_start:SPLIT_INDEX].sum() == 0, "calibration window must be attack-free"
    threshold = np.mean(calibration) + 2 * np.std(calibration)
    print(f"\ncalibration window: rows {calib_start}:{SPLIT_INDEX} ({len(calibration)} rows, all benign)")
    print(f"threshold: {threshold}")

    test_rmse = rmse[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]
    y_pred = (test_rmse > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    print("\n--- Results (real KitNET, m=10) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nBaseline (xNIDS's monolithic autoencoder substitute): "
          "Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): "
          "Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
