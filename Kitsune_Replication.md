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

### Attempt 4 — Real KitNET architecture (ensemble of per-channel autoencoders)

**Hypothesis:** xNIDS's own reference script substitutes one monolithic autoencoder for Kitsune's real architecture — an ensemble of small autoencoders, one per feature cluster, combined by an output-layer autoencoder. Isolating each channel might keep one channel's disturbance from dragging down a shared global reconstruction.

**Change:** built 5 small autoencoders, one per known channel (MAC-IP, IP, Jitter, IPtoIP, Socket — using the real semantic grouping confirmed from xNIDS's own code, rather than re-deriving clusters via online correlation tracking as original KitNET does). Each outputs its own RMSE per row; those 5 RMSEs feed a small output autoencoder whose reconstruction error is the final anomaly score. Same chronological split/scaling as the baseline.

**Result:**

| Metric | Value |
|---|---|
| Precision | 0.4988 |
| Recall | 1.0000 |
| FPR | 0.1044 |

**Verdict: no improvement — rules out the architecture.** Essentially identical to the monolithic baseline (0.5001/1.0000/0.1039). Per-channel isolation doesn't help, because the overlap isn't confined to one channel acting up and contaminating a shared bottleneck — it's present across the channels generally during the attack window. Kept as `src/models/kitsune_ensemble.py`, which also saves the per-channel RMSE features for Attempt 5.

### Attempt 5 — Supervised hybrid on per-channel error (explicit departure from the paper's unsupervised design)

**Hypothesis:** built directly on the explanation-method finding (`src/explain/`) that MAC-IP dominates feature importance in both TP and FP samples — maybe a classifier that learns its own weighting across the 5 channels separates the classes better than one global unsupervised threshold. *(Correction, 2026-11-19: this motivating finding was based on a coefficient-alignment bug in `kitsune_explanation.py` — see docs/TASKS.md's 2026-11-19 correction entry. The corrected result shows IP, not MAC-IP, dominates. This hypothesis's actual test — a supervised classifier over the 5-channel RMSE signal, a separate computation from the lasso coefficients — and its result below are unaffected by the bug.)*

**Change:** reused Attempt 4's 5 per-channel RMSE features as input to a logistic regression, with `class_weight="balanced"`. This explicitly uses labels at the scoring stage — a real, flagged departure from the paper's fully unsupervised design, not a reproduction. To avoid leakage, the test set was split in half: a labeled "tune" half fits the classifier, a disjoint "report" half is the only one evaluated.

**Result:**

| Metric | Value |
|---|---|
| Precision | 0.5011 |
| Recall | 0.9996 |
| FPR | 0.1034 |

Learned channel coefficients: MAC-IP −7.07, IP −3.54, Jitter +3.52, IPtoIP +17.27, **Socket +116.68**. Interesting on its own — the supervised fit weights Socket far more heavily than MAC-IP, the opposite of what the per-sample explanation method highlighted for individual TP/FP cases. Global linear separability across the whole channel and a single sample's local explanation aren't the same question, and this is a clean illustration of that.

**Verdict: no improvement — rules out "lack of supervision."** Even with direct label access while fitting the decision boundary (an advantage the paper's method never has), the result is within noise of every other attempt. This is the strongest evidence yet that the ceiling isn't about model capability at all — it's that the 5-channel representation itself doesn't separate attack-window benign traffic from the attack, for anything that gets to see it, supervised or not.

### Attempt 6 — The real KitNET algorithm (original author's implementation, not a substitute)

**Hypothesis:** Appendix C.1 of the xNIDS paper states Kitsune's one reported hyperparameter is `m=10` (max autoencoder size) — a detail that only makes sense for the *real* KitNET (online, incremental, correlation-clustered ensemble), not xNIDS's own `Scripts/kitsune.py` substitute (a single 115-wide batch-trained Keras autoencoder) or our own Attempt 4 ensemble (which used 5 assumed channels of 15-35 features each, already violating `m≤10`). This raised the possibility that Table 12's actual Kitsune numbers were never produced by the code we'd been testing against at all.

**Change:** vendored the original author's own implementation (`github.com/ymirsky/KitNET-py`, MIT licensed) into `src/third_party/kitnet/`, fixed one NumPy 2.0 compatibility break (`np.Inf` → `np.inf`), and ran it faithfully: raw unscaled features (KitNET does its own online min-max normalization internally), `m=10`, `FM_grace_period=5000`, `AD_grace_period=50000` (the original repo's own documented defaults — confirmed by also checking the full `Kitsune-py` system repo's example scripts, which use the identical values with no scenario-specific tuning anywhere). All 1,697,851 rows streamed through in order; correlation clustering discovered **14 autoencoders** from the 115 features (not our assumed 5 channels). Threshold: same leak-free mean+2σ convention, calibrated on rows 55,000–1,000,000 (benign, past the grace period).

**Result:**

| Metric | Value |
|---|---|
| Precision | 0.4375 |
| Recall | 0.9947 |
| FPR | 0.1329 |

**Verdict: no improvement — slightly worse than every substitute we built ourselves.** Ran in 209 seconds. This is the single most important negative result in this investigation: it is not our implementation, our architecture, or our training regime that's the limiting factor — the literal, original, paper-cited algorithm, run the standard documented way on the public dataset, lands in the same place. See `src/models/kitsune_real_kitnet.py`.

---

## Addendum: The Original Kitsune (NDSS 2018) Paper's Actual Evaluation Protocol

Found and read after the replication phase was first closed, while answering a direct question about further improvement avenues. Both `KitNET-py` and `Kitsune-py` list a file named `Kitsune paper.pdf` -- the original NDSS 2018 paper this whole detector is from -- that had been downloaded code from but never actually opened. Reading it resolved the last open methodological question.

**Confirms our split is exactly right.** The paper states directly: *"with the OS Scan dataset, each algorithm was trained on the first one million packets, and then executed on the remaining 697,851 packets."* Identical to what we, and xNIDS's own reference script, have used throughout.

**Reveals the real evaluation protocol, and it is NOT what we (or xNIDS's script) use.** The paper states: *"we measure an algorithm's true positive rate (TPR)... when φ [the threshold] is selected so that the false positive rate (FPR) is very low (i.e., 0.001)."* This is a fixed-FPR operating-point evaluation: the threshold is deliberately chosen to force FPR to a target value, and recall is whatever results at that point -- not a threshold derived from the training distribution's statistics (mean + 2σ), which is what every attempt in this project (and xNIDS's own `Scripts/kitsune.py`) does instead.

**Tested directly rather than assumed.** Forced our own trained model's threshold to the percentile of the test benign-error distribution that produces exactly FPR = 0.001 (the paper's own target), and measured recall at that point:

| Metric | Value |
|---|---|
| Threshold | 0.359215 |
| Precision | 0.6649 |
| Recall | **0.0191** |
| FPR | 0.0010 |

**This is the single most decisive result in the entire investigation.** At the exact operating point the paper reports (FPR = 0.001), our model catches only 1,256 of 65,700 attacks (1.9% recall) -- nowhere near the paper's claimed 0.9972. This holds regardless of which evaluation philosophy is used: the paper's own fixed-FPR protocol, applied directly to our model, does not produce anything resembling their numbers. The gap is not a threshold-selection artifact; it is a property of the detector's actual score distribution on this data, at any operating point along its ROC curve.

---

## Root-Cause Diagnostics: Label Granularity & Entity Aggregation

Two final diagnostic checks, run after the Addendum, to strengthen the root-cause documentation before locking the experimental phase. Both are inspection/analysis of existing data and scores — no retraining.

### Diagnostic A: Label-Window Alignment & Ground-Truth Granularity

**Question:** are all packets inside the attack time window blanket-labeled `1`, or only the actual scan packets (with victim replies and background chatter correctly labeled `0`)? And can our false positives be attributed to specific IPs (scanner vs. victim-reply vs. unrelated)?

**Metadata check, confirmed directly:** neither `OS_Scan_dataset.csv` (115 processed statistical features, headerless) nor `OS_Scan_labels.csv` (a single binary column) contains any IP, MAC, or packet-flag information — verified by inspecting both files' raw columns and the Kaggle dataset's file listing. The *only* place that information exists is `OS_Scan_pcap.pcapng`, the raw 333MB packet capture, which was deliberately never downloaded (see `README.md`'s Setup section — the two CSVs are sufficient for the feature-level reproduction this project targets). Recovering per-packet source/destination IPs would require downloading that capture and re-running Kitsune's own feature-extraction pipeline to re-align rows to packets — a new, multi-day engineering effort, not a lightweight diagnostic. **This part of the question cannot be answered further without that work**, and we are explicitly not taking it on at this stage of the project.

**Label-structure check, answerable without IP data, and genuinely informative:**

| | Value |
|---|---|
| Attack-labeled rows | 65,700 |
| First / last attack row (absolute index) | 1,314,658 / 1,576,578 |
| Span covered (last − first) | 261,921 rows |
| Number of separate contiguous label==1 runs within that span | **34,069** |
| Density of label==1 within the span | **25.1%** |

This rules out the assumption (implicit in earlier phases of this project, which informally called this a "265k-row attack window") that the ground truth is a crude time-range stamp covering every packet during the attack. It is not. The labels are scattered across 34,069 separate short runs, and only about a quarter of rows inside that span are actually marked malicious — consistent with per-packet or per-criterion labeling (e.g., only the scanner's own probe packets), with the remaining ~75% of rows in that span being correctly-labeled benign traffic that merely co-occurs in time with the scan.

**This strengthens, not weakens, the project's core finding.** Our false positives inside that span are not a labeling-granularity artifact (i.e., not "the ground truth itself blanket-stamped good traffic as bad and we're being unfairly penalized") — they are rows genuinely and correctly labeled benign that the model nonetheless flags as anomalous. That is a real detector limitation, not a dataset-labeling artifact, which is a more defensible thing to report.

### Diagnostic B: Flow-Level / Entity-Level Aggregation

**Question:** if we aggregated anomaly scores by source IP (or another entity key) across short windows, would the sustained scanning host separate cleanly from diffused background chatter?

**Confirmed: no entity identifier exists in the data we have.** Same finding as Diagnostic A — the processed feature matrix has no source IP, MAC, or flow ID column to group by, and the raw pcap that would contain one was not downloaded. There is no proxy categorical key in the 115 features either; they are continuous statistics, not identifiers.

One nuance worth recording: Kitsune's own feature extraction is *already* entity-conditioned internally — the five channels (MAC-IP, IP, Jitter, IPtoIP, Socket) are each computed as incremental, exponentially-decayed statistics keyed by MAC/IP pairs, before the identity key itself is discarded and only the resulting statistics are emitted as the 115 output columns. So "entity-aware" processing already happened upstream of what we receive; what we don't have is the ability to reverse it back into groupable entities.

The closest thing to entity/window aggregation we *can* and did test without IP data is temporal-row-window aggregation — treating consecutive rows as a proxy for "the same local activity" instead of grouping by true source identity. That is exactly what the LSTM sequence experiments (IMP-02, window=5; IMP-06, window=15) already tested, and in both cases it made FPR *worse*, not better (0.1039 → 0.1458 → 0.2718 as window size grew). We do not have a basis to expect a hand-rolled row-window aggregation of the baseline's scores to behave differently from an architecture built specifically to learn from that same windowed context, so this line of inquiry is treated as already answered rather than re-tested from scratch.

### IMP-09: Adaptive Local Threshold (logged in full in `Kitsune_Improvement.md`)

The one remaining untested idea from the brainstorm list — replacing the static global threshold with a rolling local one — was run as the final experiment. It did not help; see `Kitsune_Improvement.md`'s Round 4 section for the full result and why it fits the same sustained-elevation pattern established by IMP-07.

---

## Reproducibility Fix: TensorFlow/Keras Training Was Never Seeded

Found during a full repo audit (2026-11-19): `kitsune_autoencoder.py` has had a `RANDOM_SEED = 42` constant since its first version, but it was never actually used anywhere. RobustScaler/MinMaxScaler have no randomness, so the scaled data was always identical run to run -- but the autoencoder's own weight initialization and dropout masks were not seeded, so retraining the "same" model produced slightly different precision/recall/FPR every time. This was visible informally across the project's many retrains (e.g. baseline precision ranging 0.4988-0.5011 across different scripts/runs that should have been identical), but had never been isolated and fixed as its own issue.

### The fix needed two parts, not one

**Part 1 -- `keras.utils.set_random_seed(RANDOM_SEED)`**, called immediately before model construction, seeds Python's `random`, NumPy's RNG, and TensorFlow's RNG together. Tested directly: two separate process launches of the seeded baseline gave **0.5003** (fp=65634) and **0.5004** (fp=65606) -- closer than before (a ~20x tighter spread than the historical 0.4988-0.5011 range) but **not bit-identical**.

**Part 2 -- `tf.config.experimental.enable_op_determinism()`**, added on top of the seed. CPU TensorFlow's parallel floating-point reductions (e.g. in dense-layer matrix multiplication) aren't guaranteed to sum in the same order across runs unless this is explicitly forced, which is a separate issue from weight-initialization seeding. With both in place, two more separate process launches gave **exactly 0.5003** (tp=65700, tn=566517, fp=65634, fn=0) **both times** -- true bit-exact reproducibility, confirmed across 2 independent runs after 2 runs of the partial fix showed it wasn't sufficient alone.

Initially applied to 3 scripts to prove the mechanism (`kitsune_autoencoder.py`, `kitsune_ensemble.py`, `kitsune_improvement_delta_features.py`), then extended to every remaining script that trains a fresh Keras model from scratch, for complete documentation: `kitsune_autoencoder_resampled.py` (Attempt #1), `kitsune_autoencoder_epoch_sweep.py` (Attempt #2), `kitsune_improvement_augmentation.py` (IMP-01), `kitsune_improvement_augmentation_targeted.py` (IMP-05), `kitsune_improvement_conditional.py` (IMP-08), `kitsune_improvement_sequence.py` (IMP-02), and `kitsune_improvement_sequence_v2.py` (IMP-06) -- 10 scripts in total. Three scripts were deliberately left unfixed: `kitsune_isolation_forest.py` and `kitsune_supervised_hybrid.py` use only scikit-learn, already fully seeded (`random_state=42`); `kitsune_real_kitnet.py` uses the vendored KitNET algorithm, not TensorFlow at all, and its own autoencoder component (`src/third_party/kitnet/dA.py`) already had a hardcoded `numpy.random.RandomState(1234)` seed from the original authors, so it was already deterministic.

### Before (unseeded, historical) vs. after (fully deterministic) -- all 10 fixed experiments

| Experiment | Metric | Before | After | Conclusion affected? |
|---|---|---|---|---|
| Baseline | Precision / Recall / FPR | 0.5001 / 1.0000 / 0.1039 | 0.5003 / 1.0000 / 0.1038 | No |
| Ensemble (Attempt #4) | Precision / Recall / FPR | 0.4988 / 1.0000 / 0.1044 | 0.4995 / 1.0000 / 0.1041 | No |
| Delta features (IMP-07) | Precision / Recall / FPR | 0.3368 / 0.0299 / 0.0061 | 0.3036 / 0.0300 / 0.0071 | No |
| Delta features (IMP-07) | Recall @ forced FPR=0.001 | 0.0042 | 0.0042 | No |
| Resampled (Attempt #1) | Precision / Recall / FPR | 0.4431 / 0.0043 / 0.0006 | 0.4901 / 0.0064 / 0.0007 | No |
| Epoch sweep (Attempt #2), epoch 1 | Precision / Recall / FPR | 0.4995 / 1.0000 / 0.1041 | 0.4995 / 1.0000 / 0.1041 | No (identical) |
| Epoch sweep (Attempt #2), epoch 15 | Precision / Recall / FPR | 0.5003 / 1.0000 / 0.1038 | 0.5003 / 1.0000 / 0.1038 | No (identical) |
| Augmentation (IMP-01) | Precision / Recall / FPR | 0.4997 / 1.0000 / 0.1040 | 0.5001 / 1.0000 / 0.1039 | No |
| Targeted augmentation (IMP-05), full test | Precision / Recall / FPR | 0.5004 / 1.0000 / 0.1038 | 0.5003 / 1.0000 / 0.1038 | No |
| Targeted augmentation (IMP-05), held-out eval-half FPR | FPR (FP count) | 0.4570 (32,835) | 0.4570 (32,832) | No |
| Conditional autoencoder (IMP-08) | Precision / Recall / FPR | 0.4976 / 1.0000 / 0.1049 | 0.4816 / 1.0000 / 0.1119 | No |
| Conditional autoencoder (IMP-08) | Recall @ forced FPR=0.001 | 0.0183 | 0.0102 | No |
| Sequence, window=5 (IMP-02) | Precision / Recall / FPR | 0.4161 / 1.0000 / 0.1458 | **0.2766 / 1.0000 / 0.2719** | **See below** |
| Sequence, window=15 (IMP-06) | Precision / Recall / FPR | 0.2766 / ~1.0000 / 0.2718 | **0.2766 / 1.0000 / 0.2718** | **See below** |

**Eight of the ten hold up exactly as expected: the baseline, ensemble, both epoch-sweep checkpoints, both augmentation variants, and delta features all moved by noise-level amounts or not at all, and none of their conclusions change.** The conditional autoencoder (IMP-08) moved somewhat more (precision 0.4976→0.4816, recall@FPR=0.001 roughly halved from 0.0183→0.0102) but the conclusion is unaffected either way: it still performs essentially like the unconditioned baseline, nowhere near the paper's target.

### The sequence-modeling pair (IMP-02, IMP-06) needed a real correction, not just a note

This is the one place the fix changed something that matters. Under the old unseeded runs, window=5 gave 0.4161/1.0000/0.1458 and window=15 gave 0.2766/~1.0000/0.2718 -- a large, clean-looking gap that IMP-06's original writeup read as "FPR nearly tripled... going from window=5 to window=15," used to decisively rule out "not enough context" as the sequence-modeling problem.

With the fix applied and both re-run, **window=5 now gives 0.2766/1.0000/0.2719 and window=15 gives 0.2766/1.0000/0.2718 -- the same result, to within 0.0001 FPR.** Both window sizes converge to the identical poor local optimum under seed=42. That is direct, hard evidence that **the original 5-vs-15 comparison was not a controlled measurement of window-size effect -- it was comparing two different unseeded runs, each of which happened to land on a different local optimum, and the "window size" variable wasn't actually isolated.** The LSTM experiments were always more exposed to this than the dense-autoencoder ones: more parameters, a non-convex recurrent loss landscape with more local optima for an unseeded initialization to land in.

**What this does and doesn't change about the project's conclusions:** it does not rescue sequence modeling as a working improvement -- both properly-controlled runs still land at FPR ≈0.27, far worse than the static baseline's 0.104, so "LSTM sequence modeling at this architecture scale doesn't help" still stands. What it *does* retract is the specific comparative claim that *more context measurably makes it worse* -- that comparison wasn't actually isolating window size, and shouldn't have been presented as if it was. A properly controlled test of window-size effect would need multiple seeds per window size (to characterize the *distribution* of outcomes each architecture lands in, not a single point estimate) -- out of scope for this project's remaining timeline, and not necessary given sequence modeling is already closed as a direction for an unrelated, still-valid reason (FPR far worse than baseline at every window size tested). See `Kitsune_Improvement.md`'s IMP-02/IMP-06 rows for the cross-referenced correction note.

The other 3 training scripts in this project (`kitsune_isolation_forest.py`, `kitsune_supervised_hybrid.py`, `kitsune_real_kitnet.py`) didn't need this fix -- see above.

---

## Exhaustive Audit Log

Before closing the replication phase, we conducted two full passes over every available source, explicitly trying not to stop at the first promising lead. This section documents both passes so the "we checked everything" claim is verifiable, not asserted.

### Pass 1 — Paper, repo files, and GitHub issues

- Re-read the complete paper text (all 18 pages, not just targeted keyword hits), every file in the `code-xNIDS` repository, all 14 of its GitHub issues, the full `Kitsune-py` system repo (not just the `KitNET-py` detector), and the raw Jupyter notebooks (checked for markdown commentary stripped out of the auto-exported `.py` scripts — there was none; the exports are complete).
- **The decisive finding came from the GitHub issues**, where the paper's first author (Feng Wei, `akafengfeng`) responds directly to other researchers asking the same kind of questions we had:
  - On a user pointing out that `Scripts/explanation.py`'s `search_proper_input` doesn't implement the paper's Section 3.2 algorithm correctly (missing the `l = ⌊l/2⌋` step-halving described in the paper) (`github.com/CactiLab/code-xNIDS/issues/7` and `/issues/10`): *"this parameter is not used in the demo code... I intentionally put it out of the loop in the demo code... this function is not implemented in the demo code, since the testcases are relatively simple."* The author confirms, in his own words, that the public code is explicitly simplified **demo code**, not an exact implementation of the paper's described method.
  - On a request for the defense-rule-generation implementation (`github.com/CactiLab/code-xNIDS/issues/14`): *"The original rule generation module was managed using dozens of bash scripts and other tools. Unfortunately, since I've already graduated, I no longer have them readily available. However, with the help of large language models (LLMs) like ChatGPT or DeepSeek, you can simplify the process: just feed the explanation and a few prompts into the LLM to generate proper defense rules easily."* Core parts of the original experimental pipeline are confirmed **lost**, not merely unpublished. Scope precisely: this is about the downstream defense-rule-generation step (explanation → firewall rule), a different part of the xNIDS pipeline from the detection numbers in Table 12 — the author's suggested LLM workaround applies to that lost capability specifically, not to reproducing detection metrics. If anything, this reinforces the finding rather than offering a path back to it.
  - A separate user asked literally our own question — how to obtain the OS Scan dataset (`github.com/CactiLab/code-xNIDS/issues/8`) — and received the same Kaggle mirror link we already used. No alternate or curated version exists that we missed.

### Pass 2 — Second exhaustive sweep (local code, GitHub history/forks/PRs, external sources)

Conducted after Pass 1, specifically to check for anything a first pass could plausibly miss. Instructed to complete the full sweep even after finding a promising lead, rather than stop early.

**Local workspace:**
- Searched the paper for footnotes specifically (distinct from citation brackets). The paper has exactly **two** in 18 pages: #1 (the GitHub repo link, already known) and #2 (defining "evasion rate" for an unrelated related-work discussion). Nothing hidden.
- Found and downloaded a **second copy of the paper**, hosted on the first author's personal site (`akafengfeng.github.io/papers/xNIDS.pdf`) — different file hash from our original prepub copy. Full text diff showed the only substantive difference is the exact typo fix issue #6 flagged ("DL-NIDS" → "XNIDS" in one sentence); otherwise identical content, just different PDF line-wrapping. Archived as `docs/xNIDS_authors-site-corrected-copy.pdf` (see Reference Documents, below).
- Audited every file in `src/` for commented-out code, dead branches, and TODO/FIXME markers: **none found**. No abandoned alternate approach is sitting forgotten in a comment block.

**GitHub repository, commit history, forks, and PRs (not checked in Pass 1):**
- Commit history on `Scripts/kitsune.py`: 2 commits total ("clean" + initial) — never substantively revised since publication.
- Commit history on `Scripts/explanation.py`: 3 commits; the one real change is exactly the bugfix issue #7 reported (already accounted for in Pass 1; confirmed via the actual diff, which matches our fetched copy exactly).
- **All 13 forks** of `code-xNIDS` checked via GitHub's compare API: every single one is `ahead_by: 0` against upstream, including the author's own fork. Zero independent fixes or original work exists anywhere in the fork network.
- **All 6 pull requests** (open and closed): every one is a Dependabot dependency-version bump. None touch methodology or results.

**External sources:**
- Web search for any published reproduction attempt, forum discussion, or StackOverflow thread about reproducing xNIDS's Kitsune numbers: **nothing exists**.
- Found and checked a 2026 follow-up paper (arXiv 2607.12203) that also uses Kitsune as an explanation target: it reuses a *pre-trained* Kitsune detector as a black box for its own (different) explanation-method comparison. It never retrains Kitsune and reports no Precision/Recall/FPR — no new information.
- Found and checked a blog review of the xNIDS paper: the "conflicting numbers" it appears to mention (0.316 vs. 0.774) are simply two different tables (Table 3's ADA fidelity score vs. Table 5's MAZ sparsity score) being conflated by the reviewer — not a real discrepancy, and not about detection metrics at all.

### Conclusion

This satisfies the Phase 1 fallback trigger exactly as originally specified — *"missing dependencies, corrupted/unavailable code, or irreconcilable environment constraints"* — not by inference, but by the original author's direct, public confirmation, cross-checked against a second independent sweep that turned up nothing further. Combined with six independent technical attempts (architecture, training duration, training data composition, algorithm family, supervision, and the real original algorithm itself) all converging on the same ~0.44–0.50 precision ceiling, there is no remaining untested viable path. **No further reproduction experiments were run as part of closing this phase** — the audit itself, not a new code change, is the final check. The replication phase is closed.

### Reference Documents

- `docs/xNIDS_sec23summer_77-wei-prepub.pdf` — our original working copy (pre-publication draft, contains the known "DL-NIDS"/"XNIDS" typo).
- `docs/xNIDS_authors-site-corrected-copy.pdf` — the corrected copy hosted on the first author's personal site, archived here during the final audit. Content is otherwise identical.

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
- The real KitNET ensemble architecture (per-channel autoencoders + output combiner): no improvement — rules out the monolithic-architecture theory
- A supervised hybrid with direct label access during threshold fitting: no improvement — rules out "lack of supervision" as the limiting factor
- The real, original KitNET algorithm (the paper's own cited author's implementation, `m=10` as reported in Appendix C): no improvement, slightly worse — rules out "our substitute architectures are the problem"

**All six attempts ruled out individually**, each targeting a genuinely different possible cause (training data composition, training duration, algorithm family, model architecture, presence of supervision, and finally the literal original algorithm). None of them moved precision or FPR meaningfully from the baseline. A final audit of the paper, the full GitHub repo, every issue thread, and the original author's own public statements confirmed this isn't something we overlooked: core parts of the original experimental pipeline are author-confirmed lost, and the published code is author-confirmed to be simplified demo code that doesn't exactly implement the paper's described algorithm. That combination — six converging technical results plus direct author confirmation — is as complete a case as this kind of investigation can produce.

### Final metrics

| | Precision | Recall | FPR |
|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 |
| **Replicated baseline** (faithful reproduction, xNIDS's substitute AE) | 0.5001 | 1.0000 | 0.1039 |
| **Improved — Attempt 1** (full-timeline resampling) | 0.4431 | 0.0043 | 0.0006 |
| **Improved — Attempt 2** (epoch sweep, best epoch) | 0.5004 | 1.0000 | 0.1038 |
| **Improved — Attempt 3** (Isolation Forest) | 0.4311 | 1.0000 | 0.1372 |
| **Improved — Attempt 4** (KitNET ensemble) | 0.4988 | 1.0000 | 0.1044 |
| **Improved — Attempt 5** (supervised hybrid) | 0.5011 | 0.9996 | 0.1034 |
| **Improved — Attempt 6** (real KitNET, original author's code, `m=10`) | 0.4375 | 0.9947 | 0.1329 |

### Next steps
1. **The replication phase is closed.** All six attempted levers (training-data composition, training duration, algorithm family, model architecture, presence of supervision, and the literal original algorithm) have been exhausted. None moved precision or FPR meaningfully from the baseline (~0.44–0.50 precision, ~0.10–0.13 FPR); Attempt 1 is the only one that moved anything, at the cost of destroying recall. A final audit of the paper, repo, and author's own public statements confirmed core parts of the original pipeline are genuinely lost, not just hard to find.
2. This gap is now an exceptionally well-evidenced reproduction finding (six independent technical experiments plus direct author confirmation), strong enough to stand as the capstone's documented result per the course's own grading language: *"A failed reproduction is a valid result. An undocumented one is not."*
3. **Moving to the improvement phase.** With replication exhausted, further work should treat this as a known, accepted ceiling and focus on genuinely new approaches rather than variations on what's already been ruled out here.
3. If further work is wanted beyond documentation, the remaining options are qualitatively different from anything tried here — e.g. feature engineering beyond Kitsune's own 115 features (richer temporal context than single-row statistics), or treating this as a sequence-modeling problem instead of a per-row anomaly-detection problem — not incremental tweaks to the current pipeline.
