# IDS Aegis — xNIDS Reproduction Capstone

CIS 3370 (Intrusion Detection Systems) capstone project, University of the Incarnate Word. Reproduces *xNIDS: Explaining Deep Learning-based Network Intrusion Detection Systems for Active Intrusion Responses* (Wei et al., USENIX Security 2023), scoped to its **Kitsune** target detector on the **OS Scan** attack scenario.

## Current scope

**Kitsune only.** The project moved through NSL-KDD-only → CIC-DoS2017/ODDS → Kitsune-only over the semester (see `docs/TASKS.md`'s status log for the full history); Kitsune has been the sole active target since 2026-10-08. NSL-KDD and CIC-DoS2017 ingestion code remains in the repo as historical/reference material but is not part of the active reproduction.

## Where things stand

| Phase | Status | Record |
|---|---|---|
| Ingestion + detector reproduction | ✅ Done | `src/ingestion/kitsune.py`, `src/models/kitsune_autoencoder.py` |
| xNIDS explanation method (history-sampling + sparse-group-lasso) | ✅ Done | `src/explain/` |
| Replication phase (match the paper's exact numbers) | ✅ **Closed** — exhaustively investigated, documented divergence | [`Kitsune_Replication.md`](Kitsune_Replication.md) |
| Improvement phase (close the gap via new approaches) | ✅ **Closed** — 16 experiments, diminishing returns confirmed | [`Kitsune_Improvement.md`](Kitsune_Improvement.md) |
| GitHub Wiki reproduction report | 🔶 In progress (team) — now the main open item | repo Wiki tab |

**The short version of the finding:** our detector reproduces the paper's architecture and training setup faithfully, but lands at Precision ≈0.50 / Recall ≈1.00 / FPR ≈0.10 against the paper's claimed 0.9568 / 0.9972 / 0.001. Sixteen independent experiments (different architectures, algorithms, training regimes, feature transformations, context conditioning, adaptive thresholding, and a direct test of the paper's own evaluation protocol), plus two root-cause diagnostics ruling out a labeling artifact, all converge on the same cause: benign traffic *during* the attack window statistically resembles the attack itself, and the standard chronological train/test split means no model ever sees that condition during training. **The experimental phase is now locked** — full reasoning, evidence, and every experiment's results are in `Kitsune_Replication.md` and `Kitsune_Improvement.md`; read those before re-deriving any of this from scratch. Remaining work is the Wiki report and Dec 1/3 presentation prep (see `docs/TASKS.md`).

## Project layout

```
IDS_Aegis/
├── README.md
├── Kitsune_Replication.md      # closed: why we can't match the paper's numbers, in full
├── Kitsune_Improvement.md      # in progress: experiments attempting to close the gap anyway
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml              # dataset paths, shared constants
├── data/                         # gitignored; see Setup below
│   ├── nsl_kdd/                  # reference only, not active
│   ├── kitsune/os_scan/          # active dataset
│   └── cic_dos2017/              # reference only, not active
├── artifacts/                    # small derived evidence (explanation samples); large .npy gitignored
├── models/                       # trained model files, gitignored (regenerate via src/models/*.py)
├── scripts/
│   ├── download_kitsune.sh       # the one actively used
│   ├── download_nsl_kdd.sh       # reference
│   ├── download_cic_dos2017.sh   # reference
│   └── download_all.sh
├── src/
│   ├── ingestion/                # dataset loaders (kitsune.py is the active one)
│   ├── preprocessing/            # empty — scaling/encoding lives inline in src/models/ scripts
│   ├── models/                   # detector + every replication/improvement experiment script
│   ├── explain/                  # xNIDS explanation method (history-sampling + sparse-group-lasso)
│   └── third_party/kitnet/       # vendored original KitNET algorithm (MIT, Mirsky et al.)
└── docs/
    ├── TASKS.md                  # role breakdown + dated status log (the detailed history)
    ├── CIS_3370_Course_Hub.pdf
    ├── xNIDS_sec23summer_77-wei-prepub.pdf          # our working copy of the paper
    ├── xNIDS_authors-site-corrected-copy.pdf        # corrected copy, one typo fixed
    └── Kitsune_NDSS2018_original_paper.pdf          # the original Kitsune (2018) paper
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Get the Kitsune OS Scan data** (the only dataset actually needed):
1. Get a free Kaggle API token (kaggle.com → Settings → API → Create New Token), save it to `~/.kaggle/kaggle.json`.
2. Pull the two needed files directly (skips the full 17.7GB archive and the 333MB pcap, which aren't needed):
   ```bash
   kaggle datasets download -d ymirsky/network-attack-dataset-kitsune -f "OS Scan/OS_Scan_dataset.csv" --unzip -p data/kitsune/os_scan
   kaggle datasets download -d ymirsky/network-attack-dataset-kitsune -f "OS Scan/OS_Scan_labels.csv" --unzip -p data/kitsune/os_scan
   ```
   (See `scripts/download_kitsune.sh` for background on why this scenario specifically — confirmed from xNIDS's own reference code, not guessed.)

## Running the pipeline

```bash
python -m src.ingestion.kitsune                      # verify the loader
python -m src.models.kitsune_autoencoder              # train + evaluate the baseline detector
python -m src.explain.select_samples                  # pick real TP/FP samples for explanation
python -m src.explain.run_explanation                 # run the xNIDS explanation method on them
```

Every replication and improvement experiment is its own script under `src/models/kitsune_*.py` or `src/models/kitsune_improvement_*.py` — run any of them the same way (`python -m src.models.<name>`). Each prints its own results; see `Kitsune_Replication.md` / `Kitsune_Improvement.md` for what each one tested and found.

## Team roles & project history

See `docs/TASKS.md` for role assignments and the complete dated decision log.
