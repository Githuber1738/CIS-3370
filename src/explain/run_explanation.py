"""Run the xNIDS-style explanation on our selected TP/FN/FP Kitsune samples
and report per-channel-group feature importance for each.
"""

from pathlib import Path

import pandas as pd

from src.explain.kitsune_explanation import KitsuneExplanation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "artifacts" / "kitsune_explain_samples"
MODEL_PATH = PROJECT_ROOT / "models" / "kitsune_os_scan_autoencoder.keras"


def load_selection_summary():
    summary = {}
    with open(SAMPLES_DIR / "selection_summary.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("threshold"):
                continue
            name, info = line.split(": ", 1)
            summary[name] = eval(info)
    return summary


def explain_one(name, original_score):
    df = pd.read_csv(SAMPLES_DIR / f"kitsune_selected_{name}_rows.csv")
    current_sample = df.iloc[-1].to_numpy()
    history_samples = df.iloc[:-1].to_numpy()
    feature_names = list(df.columns)

    exp = KitsuneExplanation(current_sample, history_samples, MODEL_PATH,
                              original_score, feature_names)
    exp.capture_relevant_history()
    exp.weighted_sampling(num_samples=100, seed=42)
    exp.sparse_group_lasso()
    return exp.group_importance()


def main():
    summary = load_selection_summary()
    for name, info in summary.items():
        print(f"\n=== {name.upper()} (test row {info['test_index']}, "
              f"reconstruction_error={info['reconstruction_error']:.6f}) ===")
        importances = explain_one(name, info["reconstruction_error"])
        for group, score in importances.items():
            print(f"  {group:<10} {score:.6f}")


if __name__ == "__main__":
    main()
