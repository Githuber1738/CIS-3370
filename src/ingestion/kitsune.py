"""Load the Kitsune Network Attack Dataset (OS Scan scenario) into a DataFrame.

Source: the `ymirsky/network-attack-dataset-kitsune` Kaggle mirror (the
official UCI archive only ships one 17.7GB zip covering all 9 scenarios; the
Kaggle mirror lets individual scenario files be pulled instead -- see
scripts/download_kitsune.sh header for how OS Scan was chosen).

Scenario choice: OS Scan, confirmed (not guessed) by reading the xNIDS
authors' own Scripts/kitsune.py in github.com/CactiLab/code-xNIDS, which
loads OS_Scan_dataset.csv / OS_Scan_labels.csv to reproduce their paper's
Kitsune row.

Row alignment note: the features file (`OS_Scan_dataset.csv`) is headerless
-- 115 Kitsune statistical features per row, no column names. The xNIDS
reference script reads it with pandas' default header, which silently
discards row 0 as a bogus "header" and then also drops the labels file's
last row, producing a one-row misalignment (and a NaN label on the final
test row) in their own code. Reading the features file with `header=None`
instead avoids that bug entirely: both files have exactly the same row
count (1,697,851, verified against the downloaded files) and align 1:1 by
position with no trimming needed.
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_kitsune(scenario: str = "os_scan") -> pd.DataFrame:
    """Load Kitsune feature vectors + binary labels for the given scenario.

    Only "os_scan" is wired up right now (see config/config.yaml); other
    scenarios would need their own features/labels paths added there first.
    """
    config = _load_config()
    kitsune_cfg = config["data"]["kitsune"]
    if kitsune_cfg.get("scenario") != scenario:
        raise ValueError(
            f"config/config.yaml has scenario={kitsune_cfg.get('scenario')!r}, "
            f"not {scenario!r}. Update the config (and download that "
            "scenario's files) before loading it."
        )

    features_path = PROJECT_ROOT / kitsune_cfg["features"]
    labels_path = PROJECT_ROOT / kitsune_cfg["labels"]

    features = pd.read_csv(features_path, header=None)
    features.columns = [f"feature_{i}" for i in range(features.shape[1])]

    labels = pd.read_csv(labels_path)
    label_col = labels.columns[-1]

    if len(features) != len(labels):
        raise ValueError(
            f"Row count mismatch: {features_path.name} has {len(features)} "
            f"rows, {labels_path.name} has {len(labels)} rows. Expected "
            "them to match 1:1 by position -- see this module's docstring."
        )

    return pd.concat([features, labels[label_col].rename("label")], axis=1)


if __name__ == "__main__":
    df = load_kitsune()
    print(f"loaded {len(df)} rows, {df.shape[1]} columns")
    print("label counts:", df["label"].value_counts().to_dict())
