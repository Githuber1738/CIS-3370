# Kitsune Replication

Replication log for reproducing xNIDS's Kitsune autoencoder detector and explanation method on the Kitsune Network Attack Dataset (OS Scan scenario), as part of the CIS 3370 capstone reproduction of *xNIDS: Explaining Deep Learning-based Network Intrusion Detection Systems for Active Intrusion Responses* (Wei et al., USENIX Security 2023).

Target paper: https://www.usenix.org/conference/usenixsecurity23/presentation/wei-feng
Original authors' reference code: https://github.com/CactiLab/code-xNIDS

---

## Phase 1: Workspace Audit & Baseline Replication

### 1.1 Data acquisition & preprocessing

**Dataset identification.** The Kitsune Network Attack Dataset ships as nine separate attack scenarios (Mirai, ARP MitM, Active Wiretap, SSDP Flood, OS Scan, and others), distributed as one 17.7GB archive on the UCI ML Repository. Downloading all nine was neither necessary nor practical. We determined the correct scenario not by guessing but by reading the xNIDS authors' own reference script, `Scripts/kitsune.py`, which loads exactly `OS_Scan_dataset.csv` and `OS_Scan_labels.csv` to reproduce their paper's Table 12 Kitsune row. That made OS Scan the only scenario in scope.

**Source.** The official UCI archive only offers the full 9-scenario zip; the Kaggle mirror `ymirsky/network-attack-dataset-kitsune` exposes individual scenario files. We pulled only the two needed files (skipping the 333MB pcap, since raw packets aren't needed for feature-based training):
- `OS_Scan_dataset.csv` — 1,697,851 rows × 115 headerless statistical features, ~4.9GB
- `OS_Scan_labels.csv` — 1,697,851 rows, binary label (`0`=benign, `1`=attack)

**Feature structure.** Kitsune's 115 features are not raw packet fields — no `src_ip`/`dst_ip`/`protocol` columns exist. They are pre-computed damped time-window statistics (via Kitsune's own "AfterImage" extractor) across 5 traffic-channel groupings, confirmed against xNIDS's own `kitsune_group_sizes`/`kitsune_group_names`:

| Channel | Features | Size |
|---|---|---|
| MAC-IP | 3 stats × 5 time windows | 15 |
| IP | 3 stats × 5 time windows | 15 |
| Jitter | 3 stats × 5 time windows | 15 |
| IP-to-IP | 7 stats × 5 time windows | 35 |
| Socket-to-Socket | 7 stats × 5 time windows | 35 |

**A bug in the reference implementation, found and fixed.** xNIDS's own `Scripts/kitsune.py` reads the headerless feature file with pandas' default header behavior, which silently discards row 0 as a bogus "header," then separately drops the labels file's last row — producing a one-row misalignment (and a NaN label on the final row of their own pipeline). Our loader (`src/ingestion/kitsune.py`) reads the feature file with `header=None` instead, which avoids the bug entirely: both files have exactly 1,697,851 rows and align 1:1 by position with no trimming needed. Verified directly against the downloaded files, not assumed.

### 1.2 Model architecture & setup

Architecture, hyperparameters, and the 1M-row chronological train/test split are taken directly from `Scripts/kitsune.py` to match the paper's own setup as closely as possible:

- **Architecture:** Dense autoencoder, 115 → 256 → 128 → 64 → 128 → 256 → 115, ReLU hidden activations, sigmoid output, dropout 0.1 after each hidden layer. 141,747 trainable parameters.
- **Training:** Adam optimizer (lr=1e-3), MSE loss, 10 epochs, batch size 5000.
- **Split:** First 1,000,000 rows (chronological) = train, all attack-free; remaining 697,851 rows = test (65,700 attack / 632,151 benign).
- **Preprocessing:** RobustScaler then MinMaxScaler.
- **Anomaly scoring:** mean squared reconstruction error; threshold = mean + 2σ.

**Deliberate deviation from the reference script (label leakage):** xNIDS's script computes the anomaly threshold from the *test* set's own reconstruction error — using the very distribution being thresholded, which is a real form of leakage. We compute the threshold from the *train* set's reconstruction error only, and apply it to test. This is the single most consequential methodological difference and is flagged everywhere it affects results.

**Scaler fit scope — tried both ways.** Initially fit RobustScaler/MinMaxScaler on train-only (avoiding any leakage), but this let out-of-range test values extrapolate unclipped, producing reconstruction-error outliers as large as 1×10²⁶ — a real numerical bug, not signal. Fixed by fitting both scalers on train+test *features* combined (never labels) — this matches the reference script on this one point and is not the same kind of leakage as threshold selection, since scaler statistics don't depend on which rows are attacks. Verified this fix alone does not materially change detection metrics (see 1.4) — it only bounds the numbers to a sane range.

### 1.3 Troubleshooting & outcomes

| Issue | What happened | Resolution |
|---|---|---|
| Kitsune scenario unknown | 9 scenarios, no obvious default | Read xNIDS's own code to confirm OS Scan |
| Full dataset is 17.7GB | Impractical to download blind | Pulled only OS Scan via Kaggle's per-file download |
| Row-alignment bug in reference script | Headerless CSV read with default header, losing row 0 | Read with `header=None`; verified row counts match exactly |
| `asgl` library API mismatch | `pip install asgl` (2.2.0) has no `CV`/`ASGL`/`train_test_split` — the public API was rewritten after xNIDS was published | Pinned `asgl==1.0.6`, the last version matching their calls |
| `asgl.CV(parallel=True)` extremely slow | ~25 OS processes spawned on Windows (no `fork`, each boots a fresh interpreter) — minutes on toy 100×10 data | `parallel=False` + smaller CV grid: 0.68s at real scale (100×115), verified correct signal recovery on synthetic data first |
| `.gitignore` bug | Unanchored `models/` pattern matched `src/models/` too, silently hiding real source code (`kitsune_autoencoder.py`) from git for ~a month | Anchored to `/models/` |
| Scaler extrapolation blowup | Train-only-fit scalers let test outliers extrapolate to reconstruction errors of 1×10²⁶ | Fit scalers on train+test features (not labels); bounded to max 0.37 |

### 1.4 Results comparison — Fallback Trigger reached

| Metric | Our reproduction | xNIDS paper (Table 12) |
|---|---|---|
| Precision | 0.5001 | 0.9568 |
| Recall | 1.0000 | 0.9972 |
| FPR | 0.1039 | 0.001 |

**Exact replication was not achieved.** Not because of missing dependencies or corrupted code — the pipeline runs end-to-end and produces stable, reproducible numbers — but because of an irreconcilable gap in the paper's own prescribed experimental design, diagnosed (not assumed) through direct experiment:

1. **Threshold sweep** (0.001 → 2.0): FPR moves by less than 2 points across three orders of magnitude. Rules out threshold tuning as the fix.
2. **Time-window correlation:** ~65,000 benign-labeled test rows produce elevated reconstruction error, concentrated almost exactly inside the attack's own time window (high-error benign rows span test index ~279k–488k; the attack spans ~315k–577k).
3. **Root cause:** the training split (first 1M rows) ends *before* the attack occurs, so the model has zero training examples of benign traffic coexisting with a scan. It cannot learn that condition is normal, because it never saw it.

This satisfies the fallback trigger ("exact replication proves impossible due to... irreconcilable environment constraints") — the constraint here is the dataset's single chronological attack window combined with the paper's own first-N-rows training split, not a code or dependency problem. Proceeding to Phase 3.

---

## Phase 2: Documentation

This document. Kept current as Phase 3 work continues — see `docs/TASKS.md` for the dated, append-only decision/status log this document is synthesized from.

---

## Phase 3: Pipeline & Model Optimization

**Target:** match or exceed Precision 0.9568 / Recall 0.9972 / FPR 0.001.

### Attempt 1 — Resample training data across the full timeline

**Hypothesis:** if the model never saw attack-adjacent benign traffic in training, training on a random sample of benign rows drawn from the *entire* dataset (not just the pre-attack period) should fix it.

**Change:** train on 1,000,000 benign rows sampled uniformly at random from all 1,632,151 benign rows (still never training on attack rows — same unsupervised design). Test on the remaining benign rows + all attack rows. Everything else (architecture, hyperparameters, scaler approach) held constant.

**Result:**

| Metric | Value |
|---|---|
| Precision | 0.4431 |
| Recall | 0.0043 |
| FPR | 0.0006 |

**Verdict: informative failure, not a dead end.** FPR actually beats the paper (0.0006 vs. 0.001) — but recall collapses to 0.4% (284/65,700 attacks caught). This is direct, independent confirmation of the root cause from Phase 1: attack rows and attack-window benign rows occupy nearly the same region of feature space. Training on one teaches the model to reconstruct the other too. Fixing the FPR this way destroys recall because both problems are the same underlying overlap — not two separate issues to solve independently. Kept as `src/models/kitsune_autoencoder_resampled.py`, deliberately not merged into the faithful-reproduction script.

### Attempt 2 — Rule out overfitting-by-training-duration

**Hypothesis (user-proposed):** maybe the model overfits the training set with more epochs, and an earlier checkpoint would generalize better.

**Change:** evaluated on the real test set after every epoch (1–15), each epoch deriving its own leak-free threshold from that epoch's train reconstruction error.

**Result:** flat across the entire range.

| Epoch | Loss | Precision | Recall | FPR |
|---|---|---|---|---|
| 1 | 0.015243 | 0.4995 | 1.0000 | 0.1041 |
| 5 | 0.000225 | 0.5000 | 1.0000 | 0.1039 |
| 10 | 0.000130 | 0.5002 | 1.0000 | 0.1038 |
| 15 | 0.000098 | 0.5003 | 1.0000 | 0.1038 |

**Verdict: ruled out.** A barely-trained epoch-1 checkpoint (loss 0.015) performs identically to the fully-converged epoch-15 model (loss 0.0001). If overfitting were the cause, the early checkpoint would generalize *better* than the late one — it doesn't, at all. Training duration is not the lever. Kept as `src/models/kitsune_autoencoder_epoch_sweep.py`.

### Attempt 3 — Alternative anomaly-detection algorithm (Isolation Forest)

**Hypothesis:** the overlap found in Attempts 1–2 might be specific to *autoencoder reconstruction error* as a scoring mechanism, not necessarily baked into the feature space itself regardless of algorithm.

**Change:** same chronological split and scaling, but score anomalies with an Isolation Forest instead of an autoencoder.

**Result:**

| Metric | Value |
|---|---|
| Precision | 0.4311 |
| Recall | 1.0000 |
| FPR | 0.1372 |

**Verdict: rules out "autoencoder-specific."** A completely different algorithm family (tree-based isolation, not reconstruction error at all) hits the same wall — slightly *worse*, in fact, than the autoencoder baseline (FPR 0.1372 vs. 0.1039). This confirms the overlap between attack and attack-window-benign traffic is a property of the feature space and the chronological split itself, not an artifact of how the autoencoder scores anomalies. Kept as `src/models/kitsune_isolation_forest.py`.

### Not yet attempted (next steps, see below)
- The real KitNET algorithm (ensemble of small per-channel autoencoders with incremental correlation clustering) in place of xNIDS's substitute single monolithic autoencoder.
- A supervised hybrid using per-channel reconstruction error (MAC-IP/IP/Jitter/IPtoIP/Socket) as features for a small classifier, built directly on the explanation-method findings (`src/explain/`) — MAC-IP dominates importance in both true and false positives, suggesting a channel-aware decision boundary might outperform one global threshold.

---

## Final Reporting

### What worked vs. what didn't

**Worked:**
- Full ingestion + training + evaluation pipeline, end-to-end, reproducible
- Identifying the correct scenario and fixing two real bugs in the reference implementation (row alignment, and our own scaler extrapolation)
- Diagnosing the root cause of the metric gap with direct, falsifiable experiments rather than assumption
- Implementing xNIDS's actual explanation method (history-input sampling + sparse-group-lasso) on top of the detector

**Didn't work (informatively):**
- Resampling training data across the timeline: fixes FPR, destroys recall — proves the two failure modes share one cause
- More/less training: ruled out as a factor entirely
- A different algorithm family (Isolation Forest instead of autoencoder reconstruction error): no better, slightly worse — rules out the scoring mechanism as the cause

### Final metrics

| | Precision | Recall | FPR |
|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 |
| **Replicated baseline** (faithful reproduction) | 0.5001 | 1.0000 | 0.1039 |
| **Improved — Attempt 1** (full-timeline resampling) | 0.4431 | 0.0043 | 0.0006 |
| **Improved — Attempt 2** (epoch sweep, best epoch) | 0.5004 | 1.0000 | 0.1038 |
| **Improved — Attempt 3** (Isolation Forest) | 0.4311 | 1.0000 | 0.1372 |

### Next steps
1. All three attempted levers (training-data composition, training duration, algorithm family) have now been tried and ruled out individually. The strongest remaining options are the real KitNET ensemble architecture (isolating per-channel anomalies instead of one global score) or a supervised hybrid using the per-channel reconstruction errors from `src/explain/` (MAC-IP dominates importance in both TP and FP, suggesting a channel-aware decision boundary) — both bigger lifts, but each targets the root cause from a genuinely different angle than the three attempts above.
2. Regardless of further attempts, this gap is now a well-evidenced reproduction finding (four independent experiments pointing at the same cause), strong enough to stand as the capstone's documented result per the course's own grading language: *"A failed reproduction is a valid result. An undocumented one is not."*
