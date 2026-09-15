"""Load the CIC-DoS2017 dataset into a labeled pandas DataFrame.

Source: the `dhoogla/cicdos2017` Kaggle mirror (the official UNB CIC host is
down as of this project's setup -- see scripts/download_cic_dos2017.sh).
Unlike NSL-KDD's raw connection records, this mirror ships pre-extracted
CICFlowMeter flow features (78 columns) in Parquet, already deduplicated of
the identifying columns (source/destination IP, timestamp) and, as verified
against this specific file, free of the inf/NaN values that raw CICFlowMeter
output is notorious for (division-by-zero on single-packet flows produces
`Flow Bytes/s` / `Flow Packets/s` = inf). This loader re-checks for those
defensively rather than assuming the mirror stays clean if it's ever
re-downloaded or swapped for the official source later.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_cic_dos2017(drop_bad_rows: bool = True) -> pd.DataFrame:
    """Load CIC-DoS2017 flow records with binary + multiclass labels attached.

    Adds:
      - "is_attack": 0 for Benign, 1 for any DoS variant (ddossim, slowloris,
        hulk, goldeneye, rudy, slowread, slowheaders, slowbody2, ...).
      - "attack_type": the raw multiclass Label, kept for finer-grained
        analysis/reporting.

    drop_bad_rows: if True (default), rows containing inf/NaN in any numeric
    feature column are dropped and a count is printed. CICFlowMeter output
    can contain these from single-packet flows; set to False to inspect them
    yourself instead.
    """
    config = _load_config()
    cic_cfg = config["data"]["cic_dos2017"]
    path = PROJECT_ROOT / cic_cfg["dir"] / cic_cfg["file"]

    df = pd.read_parquet(path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a {LABEL_COLUMN!r} column in {path.name}; "
            f"found columns: {list(df.columns)}"
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    bad_mask = (
        df[numeric_cols].isna().any(axis=1)
        | np.isinf(df[numeric_cols]).any(axis=1)
    )
    n_bad = int(bad_mask.sum())
    if n_bad:
        print(f"load_cic_dos2017: {n_bad} row(s) with inf/NaN feature values"
              f"{' -- dropping' if drop_bad_rows else ' -- kept, inspect via the bad_mask logic above'}")
        if drop_bad_rows:
            df = df.loc[~bad_mask].reset_index(drop=True)

    df["attack_type"] = df[LABEL_COLUMN].astype(str)
    df["is_attack"] = (df["attack_type"] != BENIGN_LABEL).astype(int)

    return df


if __name__ == "__main__":
    df = load_cic_dos2017()
    print(f"loaded {len(df)} rows, {df.shape[1]} columns")
    print("is_attack counts:", df["is_attack"].value_counts().to_dict())
    print("attack_type counts:", df["attack_type"].value_counts().to_dict())
