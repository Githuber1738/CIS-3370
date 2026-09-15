"""Load NSL-KDD's raw text files into labeled pandas DataFrames.

NSL-KDD ships as headerless comma-separated text: 41 KDD features, then the
fine-grained attack (or "normal") label, then a "difficulty" score (how hard
KDD's own scoring rig found the record to classify) tacked on as an
undocumented 43rd column. Downstream preprocessing/model code should import
from here rather than re-parsing the files, so column names and the
5-class attack mapping stay consistent everywhere.
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

COLUMN_NAMES = FEATURE_COLUMNS + ["label", "difficulty"]

# Maps every fine-grained NSL-KDD label to its 5-class attack category.
# KDDTest+ contains several attack types absent from KDDTrain+ (e.g.
# "mailbomb", "processtable", "sqlattack") -- that held-out gap is the whole
# point of NSL-KDD (generalization to unseen attacks), so it's captured here
# rather than left for each model script to rediscover independently.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (remote-to-local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R (user-to-root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_nsl_kdd(split: str = "train") -> pd.DataFrame:
    """Load one NSL-KDD split into a labeled DataFrame.

    split: one of "train", "train_20pct", "test", "test_21" -- matching the
    keys under data.nsl_kdd in config/config.yaml.
    """
    config = _load_config()
    nsl_kdd_cfg = config["data"]["nsl_kdd"]

    filename_key = {
        "train": "train",
        "train_20pct": "train_20pct",
        "test": "test",
        "test_21": "test_21",
    }.get(split)
    if filename_key is None:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            "'train', 'train_20pct', 'test', 'test_21'"
        )

    path = PROJECT_ROOT / nsl_kdd_cfg["dir"] / nsl_kdd_cfg[filename_key]
    df = pd.read_csv(path, names=COLUMN_NAMES, header=None)

    unmapped = set(df["label"]) - set(ATTACK_CATEGORY_MAP)
    if unmapped:
        raise ValueError(
            f"Unmapped NSL-KDD label(s) in {path.name}: {sorted(unmapped)}. "
            "Add them to ATTACK_CATEGORY_MAP in src/ingestion/nsl_kdd.py."
        )
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP)

    return df


if __name__ == "__main__":
    for split in ("train", "train_20pct", "test", "test_21"):
        df = load_nsl_kdd(split)
        print(f"{split}: {len(df)} rows, categories={df['attack_category'].value_counts().to_dict()}")
