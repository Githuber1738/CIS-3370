# Task Delegation Log

## Scope decision (2026-09-15, revised same day)

Originally started with NSL-KDD only. **Revised per professor's guidance:**
NSL-KDD is the dataset already covered in class, so the professor wants the
team to work a different, more challenging dataset instead. Primary target is
now **CIC-DoS2017**. NSL-KDD ingestion code already written stays in the repo
(useful reference / possible comparison baseline) but is no longer the
critical path. Kitsune remains an open stretch goal, undecided.

## Roles

### Student 1 — Data & System Architect
Owns dataset acquisition, directory layout, and environment setup.

Immediate tasks:
- [x] Run `scripts/download_nsl_kdd.sh` and confirm all four files land in `data/nsl_kdd/`
- [ ] Confirm `python -m venv .venv && pip install -r requirements.txt` works clean on a teammate's machine

Stretch goals (defer until NSL-KDD reproduction is working):
- [ ] Decide which Kitsune attack scenario to target (see `scripts/download_kitsune.sh` header — check the xNIDS repo at github.com/CactiLab/code-xNIDS for which one they used) and run the download
- [ ] Get a Kaggle API token and run `scripts/download_cic_dos2017.sh`, or email UNB CIC for direct access if the mirror is unsuitable

### Student 2 — Backend / Logic Developer
Owns ingestion, preprocessing, and baseline model/query code.

Immediate tasks:
- [ ] Write `src/ingestion/` loaders for each dataset (NSL-KDD is plain-text CSV-like with a 42-column schema + an undocumented `difficulty` 43rd column — watch for that)
- [ ] Write `src/preprocessing/` for encoding, scaling, class-imbalance handling
- [ ] Stand up a baseline model per target dataset once data is in place

### Student 3 — Documentation & Project Manager / Analyst
Owns `README.md`, task tracking, dependency management, and course alignment.

Immediate tasks:
- [ ] Keep this file current as tasks move
- [ ] Keep `requirements.txt` in sync with what's actually imported
- [ ] Cross-check deliverables against the course's capstone rubric (paper selection/proposal, reproduction, final presentation, GitHub repo + project board, Wiki report)
- [ ] Track the team's chosen commit approach (Individual vs. Delegated Commits) — this needs to be declared in the proposal per the course's GitHub requirements

## Status Log

| Date | Change |
|------|--------|
| 2026-09-15 | Project scaffold created: directory structure, download scripts, config, requirements |
| 2026-09-15 | NSL-KDD data downloaded and verified (all 4 files + ARFF zip) |
| 2026-09-15 | Scope decision: NSL-KDD only for now; Kitsune/CIC-DoS2017 deferred as stretch goals |
| 2026-09-15 | Scope revised per professor: switching primary target to CIC-DoS2017 (NSL-KDD is already covered in class) |
| 2026-09-15 | CIC-DoS2017 downloaded via Kaggle mirror (dhoogla/cicdos2017) and ingestion loader written/verified |
| 2026-10-08 | Delegated Commits confirmed as the team's commit approach (one teammate new to git) |
| 2026-10-08 | Kitsune OS Scan scenario downloaded (confirmed via xNIDS repo's Scripts/kitsune.py that this is the scenario the paper's Table 12 Kitsune row actually trains on) -- 1,697,851 rows, 115 features + binary label, via Kaggle mirror ymirsky/network-attack-dataset-kitsune |
| 2026-10-08 | src/ingestion/kitsune.py written + verified; fixed a row-alignment bug present in xNIDS's own reference script |
| 2026-10-08 | src/models/kitsune_autoencoder.py: trained + evaluated the xNIDS-architecture autoencoder on Kitsune OS Scan. Recall 1.0 (vs paper's 0.9972) but precision 0.47 / FPR 0.118 vs paper's 0.9568 / 0.001 -- see conversation/Wiki report for the methodological explanation (threshold + scaler fit corrected to avoid train/test leakage present in the reference script) |
| 2026-10-08 | Confirmed from the xNIDS paper text (Table 12 + Sec. 5 "Target DL-NIDS"): CIC-DoS2017 pairs with ODDS (LSTM-based bot detector), NOT Kitsune/autoencoder. Kitsune pairs only with its own dataset. This was previously an open question. |
| 2026-11-12 | Scope reconfirmed: Kitsune-only, team agreed. Pushed a month of local Kitsune work (ingestion, model) that had never been committed; fixed a .gitignore bug that was silently hiding src/models/*.py from git. |
| 2026-11-12 | src/explain/: implemented xNIDS's explanation method for Kitsune (history-input sampling + sparse-group-lasso feature importance via asgl), adapted from their Scripts/explanation.py. Pinned asgl==1.0.6 (their API was rewritten in 2.x). Fixed a Windows multiprocessing slowdown in asgl.CV (parallel=True spawns ~25 OS processes with no fork; parallel=False + a smaller grid runs in under a second with correct group-signal recovery, verified on synthetic data first). |
| 2026-11-12 | Selected real TP/FP samples from our trained model for explanation (artifacts/kitsune_explain_samples/): zero FN samples exist, a direct consequence of our model's recall=1.0. Explanation result: both TP and FP are dominated by the MAC-IP feature channel with near-identical relative importance shapes -- suggests the FP is a real near-miss (benign traffic resembling OS-Scan's MAC-IP signature), not a random error. |
| 2026-11-12 | Ruled out overfitting-by-epoch-count: evaluated on the real test set after every epoch (1-15), each with its own leak-free threshold. Precision/recall/FPR are flat across the entire range (epoch 1: 0.4995/1.0000/0.1041 vs. epoch 15: 0.5003/1.0000/0.1038) -- an under-trained epoch-1 checkpoint performs identically to a fully-converged epoch-15 model, which is the opposite of what overfitting would look like. Confirms the gap is structural (training-data coverage), not a function of training duration. See src/models/kitsune_autoencoder_epoch_sweep.py. |
| 2026-11-12 | Improvement attempt #1: trained on a random benign sample drawn from the FULL timeline (not just the pre-attack first 1M rows), to directly fix the diagnosed training-coverage gap. Result: FPR dropped to 0.0006 (better than the paper's 0.001!) but recall collapsed to 0.0043 (284/65700 attacks caught). This is an informative failure, not a dead end: it confirms the root-cause diagnosis from a different angle -- attack rows and attack-window benign rows are so close in feature space that training on one teaches the model to reconstruct the other too. Fixing the FPR this way destroys recall because they're the same underlying overlap. See src/models/kitsune_autoencoder_resampled.py. |
| 2026-11-12 | Diagnosed the Precision/FPR gap vs. Table 12 (ours ~0.50/1.00/0.10 vs. paper's 0.9568/0.9972/0.001). Threshold sweep (0.001 to 2.0) shows FPR is nearly flat the whole range -- not a threshold-tuning problem. Found train-only-fit scalers were extrapolating unclipped on out-of-range test values, producing reconstruction-error outliers as large as 1e26 -- a real numerical bug, now fixed (scalers fit on train+test features, never labels; matches the reference script on this point, since scaler fit scope isn't label-informed leakage the way threshold selection is). Fixing it bounded all errors to a sane range (max 0.37) but barely moved precision/recall/FPR (0.50/1.00/0.104) -- confirming the gap is not a bug. Root cause: ~65,000 benign-labeled test rows produce elevated reconstruction error, concentrated almost exactly in the attack's own time window (benign high-error rows: test index 279k-488k; attacks: 315k-577k). The model's training split (first 1M rows) predates the attack entirely, so it never saw this network condition -- a training-data coverage gap inherent to the paper's own prescribed chronological split, not fixable by threshold or preprocessing changes. Retrained with the scaler fix for consistency: final numbers precision=0.5001, recall=1.0000, fpr=0.1039, matching the diagnosed floor exactly. |
