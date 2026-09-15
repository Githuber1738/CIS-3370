# IDS Aegis — Network Security Data Analytics Project

Team project for the Information Decision Sciences (IDS) course. Three teammates,
three datasets, one reproducible pipeline.

**Current scope:** CIC-DoS2017 (primary target, per professor's guidance —
NSL-KDD is already covered in class, so the team is reproducing against a
dataset the course hasn't handed them). NSL-KDD ingestion code stays in the
repo as reference/baseline. Kitsune remains an open stretch goal.

## Datasets

| Dataset      | Folder                | Source status (checked at setup) |
|--------------|------------------------|------------------------------|
| NSL-KDD      | `data/nsl_kdd/`        | Official UNB page is down ("no longer available"); using the `defcom17/NSL_KDD` GitHub mirror |
| Kitsune      | `data/kitsune/`        | Official UCI archive is live — 17.7 GB full set, or per-attack files 1.9–3 GB each |
| CIC-DoS2017  | `data/cic_dos2017/`    | Official UNB page is down ("not available at this time"); needs a Kaggle mirror + your own Kaggle API key, or a direct request to the CIC team |

## Project layout

```
IDS_Aegis/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml          # dataset paths, shared constants
├── data/
│   ├── nsl_kdd/
│   ├── kitsune/
│   └── cic_dos2017/
├── scripts/
│   ├── download_nsl_kdd.sh
│   ├── download_kitsune.sh
│   ├── download_cic_dos2017.sh
│   └── download_all.sh
├── src/
│   ├── ingestion/           # loading raw files into DataFrames
│   ├── preprocessing/       # cleaning, encoding, scaling
│   └── models/              # baseline + target models
├── notebooks/                # exploratory analysis
└── docs/
    └── TASKS.md              # role breakdown and task log
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
bash scripts/download_nsl_kdd.sh          # small, safe to run immediately
bash scripts/download_kitsune.sh          # large — read the script header first
bash scripts/download_cic_dos2017.sh      # requires a Kaggle API key — read the script header first
```

## Team roles

See `docs/TASKS.md`.
