# Kitsune Improvement

Optimization roadmap for the Kitsune detector, kept strictly separate from
`Kitsune_Replication.md` (the closed, baseline reproduction record). Every
experiment logged here is a deliberate departure from xNIDS's methodology,
not an attempt to faithfully reproduce it — that phase is done; see the
replication document for why six independent technical attempts and a full
author/repo audit concluded the paper's exact Table 12 numbers aren't
achievable with the public dataset and pipeline.

---

## Target Benchmarks

| Metric | xNIDS paper, Table 12 (Kitsune row) |
|---|---|
| Precision | 0.9568 |
| Recall | 0.9972 |
| FPR | 0.001 |

**Current best (replicated baseline, closed):** Precision 0.5001 / Recall 1.0000 / FPR 0.1039 — see `Kitsune_Replication.md` for the full six-attempt history. Every experiment below is measured against this baseline, not against a blank slate — the bar to clear is "better than 0.50 / 1.00 / 0.104," and the stretch goal is closing meaningfully more of the gap to the paper's numbers.

---

## Proposed Improvement Strategies

Ranked by expected impact vs. implementation effort. Each is explicitly a new angle, not a variation on anything already ruled out in replication (training data composition, training duration, algorithm family, architecture, and supervision were all tested there — see that document before proposing a variant of those).

### 1. Surgical data augmentation (High impact, Medium effort)
**Idea:** Replication Attempt #1 (blind resampling of benign rows across the full timeline) fixed FPR but destroyed recall, because it exposed the model to real attack-adjacent benign rows that are too similar to actual attacks. A more surgical version: synthetically perturb *existing* training-only benign rows (noise injection, mild feature jitter simulating "network under load") to create synthetic "benign-but-unusual" examples, without ever touching real attack-window data. This directly extends the paper's own stated fix for "Unseen Benign" error (*"we can reduce the FPR after retraining... with the augmented data"* — Section 6.3.2) using a mechanism we control and can document precisely, unlike the paper's own under-specified version.
**Risk:** could reproduce Attempt #1's recall collapse if the synthetic examples end up too close to real attack signatures. Needs a tunable "how aggressive" knob and should be evaluated across several settings, not just once.

### 2. Sequence/temporal modeling instead of per-row scoring (High impact, High effort)
**Idea:** Every detector tried so far (the monolithic AE, the real KitNET, Isolation Forest) scores each row independently. The actual failure mode is temporal: attack-adjacent benign rows are hard to distinguish from attack rows *in isolation*, but a model that sees a short window of consecutive rows might pick up on sequence-level patterns (e.g., a true attack row is surrounded by other attack rows; an incidental benign row near the attack window is surrounded mostly by benign rows). An LSTM or small transformer over a sliding window of rows is a genuinely different signal source, not a tuning variant of anything tried.
**Risk:** biggest engineering lift here; also a bigger departure from Kitsune's own per-packet design philosophy, worth being explicit about when documenting results.

### 3. Explanation-informed rule-based post-filter (Medium impact, Medium effort)
**Idea:** Directly reuses the `src/explain/` investment. We know MAC-IP dominates importance in both TP and FP explanations (a near-miss signature). Build a secondary heuristic layer: when the primary detector flags a row, check whether the *other* four channels (IP/Jitter/IPtoIP/Socket) also show elevated error — if only MAC-IP is elevated and the rest look normal, downweight the flag. This is a concrete, explainability-driven idea rather than a blind architecture swap, and ties the improvement phase back to work already done rather than starting over.
**Risk:** needs careful leak-free calibration (the "other channels normal" threshold itself needs to be fit on benign-only data, same discipline as everything else in this project).

### 4. Feature engineering: log-transform before scaling (Low impact, Low effort)
**Idea:** Kitsune's 115 features span large dynamic ranges (packet counts/rates are typically heavy-tailed). A `log1p` transform before RobustScaler/MinMaxScaler could improve numerical conditioning and give the autoencoder an easier reconstruction target. Cheap to test — worth doing early precisely because it's cheap, not because it's expected to fully close the gap.
**Expectation, stated honestly:** unlikely to fix the structural overlap diagnosed in replication (that's about the *information* in the features, not their scale), but worth ruling in or out quickly before investing in the bigger items above.

### 5. Threshold/decision calibration (isotonic regression, Platt scaling) (Low impact, Low effort)
**Idea:** Replace the flat mean+2σ threshold with a calibrated probability mapping.
**Expectation, stated honestly:** the replication phase's threshold sweep (0.001 → 2.0) already showed FPR is nearly flat across three orders of magnitude — strong evidence that *no* threshold choice, however it's calibrated, moves the needle much. Included for completeness and because it's nearly free to test, not because it's expected to help. Rank last; don't spend real time here before 1–3.

---

## Experiment Tracking Template

Copy a new row per experiment run. Keep hypotheses falsifiable and record the actual result even when it's a null result — a ruled-out idea is still useful information (see how much of `Kitsune_Replication.md` is exactly that).

| ID | Date | Strategy | Hypothesis | Configuration | Precision | Recall | FPR | F1 | Verdict | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| IMP-01 | 2026-11-17 | #1 Surgical data augmentation | Noise-perturbed synthetic benign rows (never touching real attack-window data) will teach the model to tolerate mild variation without the recall collapse blind resampling caused | `src/models/kitsune_improvement_augmentation.py`: +200,000 synthetic rows (20% of train, Gaussian noise std=0.02 in [0,1]-scaled space, clipped), same architecture/epochs as baseline | 0.4997 | 1.0000 | 0.1040 | 0.6664 | No change | Within noise of baseline (0.5001/1.000/0.1039). The noise level tested was mild enough to not collapse recall (unlike replication Attempt #1) but also too mild to change what gets flagged at all — synthetic "jittered benign" doesn't resemble real attack-window benign closely enough to teach the model anything new about that regime. A stronger noise level might move this, but risks drifting back toward Attempt #1's failure mode; not retried this round. |
| IMP-02 | 2026-11-17 | #2 Sequence/temporal modeling | A model that sees a 5-row window will distinguish true attack sequences (surrounded by other attacks) from incidental attack-adjacent benign rows (surrounded by benign) | `src/models/kitsune_improvement_sequence.py`: LSTM(32)-autoencoder, window=5, 200k subsampled training windows (stride 5), scored on last-timestep reconstruction error, 10 epochs, runtime 73.1s | 0.4161 | 1.0000 | 0.1458 | 0.5877 | Worse | FPR got worse, not better. A 5-row window apparently isn't enough context to separate the classes, or the attack-adjacent benign rows are themselves temporally clustered with OTHER attack-adjacent benign rows (i.e. the local neighborhood looks uniformly "unusual" either way), which would defeat the whole hypothesis. Worth knowing before investing in a bigger architecture here. |
| IMP-03 | 2026-11-17 | #3 Explanation-informed post-filter | Rows where ONLY MAC-IP is elevated (and the other 4 channels look normal) are more likely incidental near-misses than real attacks, and can be safely un-flagged | `src/models/kitsune_improvement_postfilter.py`: baseline detector's 131,361 flagged test rows checked against replication Attempt #4's per-channel RMSE, suppress if MAC-IP elevated AND all 4 others below their train-calibrated threshold | 0.5001 | 1.0000 | 0.1039 | 0.6668 | No change | **The suppression rule never fired — 0 of 131,361 flags matched the condition.** When the detector flags a row during the attack window, multiple channels are elevated together, not MAC-IP alone. The underlying explanation finding (MAC-IP dominates *relative* importance in TP/FP samples) doesn't mean MAC-IP is elevated *in isolation* — it's elevated *most*, among others that are also elevated. The rule's premise doesn't hold; a magnitude-ratio rule (MAC-IP much higher than the rest, rather than "only MAC-IP crosses its own threshold") would be a different, more defensible version of this idea. |
| IMP-04 | 2026-11-17 | Round 2, Exp 2.1: ratio-based post-filter | A row where MAC-IP's error dominates the next-highest channel by a large margin (ratio, not absolute threshold) is more likely a near-miss than a real attack | `src/models/kitsune_improvement_postfilter_ratio.py`: suppress flags where MAC-IP/max(other 4) exceeds the 99th-percentile ratio seen on train data (cutoff=1.057) | 0.5002 | 0.9620 | 0.0999 | 0.6581 | Worse (F1 down) | The rule fired this time (4,991/131,361 suppressed) but it's a straight trade, not a win: FP dropped by ~2,495 while FN rose by 2,496 -- almost exactly 1-for-1. The dominance ratio doesn't cleanly separate real attacks from near-misses among the rows it catches; both are mixed roughly evenly in the top 1% most MAC-IP-dominant flagged rows. Confirms MAC-IP dominance alone (ratio or threshold) isn't a usable discriminator here. |
| IMP-05 | 2026-11-17 | Round 2, Exp 2.2: targeted augmentation from the real attack-window-benign distribution | Synthetic training rows shifted toward the REAL measured difference of attack-window benign traffic (not generic noise) will teach the model to treat that condition as normal, specifically on the held-out half never used to measure the shift | `src/models/kitsune_improvement_augmentation_targeted.py`: attack-window-benign region (test rows 279140-488495, y=0) split 50/50 into calibration/eval halves; shift vector = calibration-half mean minus train mean; +200k synthetic rows = random train rows + scaled shift vector + small noise | 0.5004 (full test) | 1.0000 | 0.1038 | 0.6670 | No change | **Decisive null result, not just "no effect."** On the held-out eval half (71,848 rows, never used to calibrate the shift, specifically the hardest subset) the baseline gets 32,835/71,848 FP (FPR 0.4570) and this targeted model gets 32,831/71,848 (FPR 0.4570) -- a difference of 4 rows. Using the exact real distributional shift, properly leak-controlled, produced zero detectable improvement on the exact population it targeted. This suggests the problem isn't a crude/mismatched augmentation (IMP-01's issue) -- a mean-shift characterization is too coarse a description of what actually differs; the real difference is likely in higher-order structure (variance/covariance), or is heterogeneous across rows rather than one consistent direction a single shift vector can capture. |
| IMP-06 | 2026-11-17 | Round 2, Exp 2.3: sequence modeling, window=15 | A larger window (3x Round 1's window=5) will give the LSTM enough context to distinguish real attack sequences from incidental attack-adjacent benign rows, where window=5 wasn't enough | `src/models/kitsune_improvement_sequence_v2.py`: same LSTM(32) autoencoder architecture and training regime as IMP-02, window increased from 5 to 15, 199,998 subsampled training windows (stride 5) | 0.2766 | ~1.0000 (65697/65700) | 0.2718 | 0.4333 | Worse (substantially) | **Decisively rules out "not enough context."** FPR nearly tripled (0.1458 -> 0.2718) and precision collapsed (0.4161 -> 0.2766) going from window=5 to window=15 -- the opposite of the hypothesis. Final training loss was also notably worse (0.0361 vs. window=5's 0.0247), suggesting the larger window is harder for this small architecture to reconstruct well at all, not just harder to use for anomaly discrimination. Combined with IMP-02, this closes the sequence-modeling direction at this architecture scale: more temporal context doesn't help, and actively hurts as window size grows. |

**Column guide:**
- **Strategy**: which numbered idea above (or a new one — add it to the list first with impact/effort estimates before running it).
- **Configuration**: exact parameters/code path, enough detail that a teammate could rerun it from this row alone.
- **Verdict**: one of `Improved`, `No change`, `Worse`, `Inconclusive — needs rerun`.
- **Notes**: anything that would change how the next experiment should be designed — this is the field that turns a log into a roadmap instead of a list.

---

## Round 1 Results: Benchmark Comparison

| | Precision | Recall | FPR |
|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 |
| Replicated baseline | 0.5001 | 1.0000 | 0.1039 |
| IMP-01 Surgical augmentation | 0.4997 | 1.0000 | 0.1040 |
| IMP-02 Sequence/temporal (LSTM) | 0.4161 | 1.0000 | 0.1458 |
| IMP-03 Explanation post-filter | 0.5001 | 1.0000 | 0.1039 |

**Verdict: none of the three matched or surpassed the baseline, let alone the paper's numbers.** IMP-01 and IMP-03 landed within noise of the baseline (no real effect either way); IMP-02 was measurably worse. None closed any of the gap to Table 12.

**Which modification drove the largest effect:** IMP-02 (sequence modeling) — but in the wrong direction. It's also the most informative failure of the three: it directly tested whether temporal context helps, and the answer was no, at least at this window size and architecture scale. IMP-01 and IMP-03 weren't so much "tested and failed" as "tested and never got the chance to matter" — IMP-01's noise level was too mild to teach anything new, and IMP-03's trigger condition never fired. Both are inconclusive about the *idea*, conclusive about *this specific implementation* of it.

---

## Round 2: Next-Step Recommendations

Based on what Round 1 actually showed (not a restart — each proposal below is a direct, targeted fix to why its predecessor didn't move anything):

1. **Redesign IMP-03 as a ratio-based rule, not a threshold-crossing rule.** We now know flagged rows have *multiple* channels elevated together, not MAC-IP alone — so "is MAC-IP the only elevated channel" was the wrong question. The better question, directly motivated by this round's data: "is MAC-IP elevated by a much larger margin than the next-highest channel" (e.g. MAC-IP's z-score is 3x the second-highest channel's z-score). This is a small code change, cheap to test, and directly informed by IMP-03's own result rather than a guess.
2. **Retry IMP-01 with noise drawn from the real attack-window-benign distribution, not generic Gaussian noise.** The replication phase already characterized exactly how attack-window benign rows differ from ordinary benign rows (concentrated in test rows ~279k-488k, see `Kitsune_Replication.md`). Sampling synthetic perturbations from *that* empirical difference, rather than an arbitrary noise scale, is a more principled version of the same idea and more likely to actually teach the model something, without reintroducing real attack-adjacent rows (which is what collapsed recall in replication Attempt #1).
3. **One more sequence-modeling attempt at a larger window (10-20 rows) before abandoning that direction**, specifically because IMP-02 got *worse*, not merely unchanged — worth distinguishing "5 rows isn't enough context" from "temporal context doesn't help at all" before writing off the whole approach. Cheap to test (73s runtime at window=5); a 2-4x larger window is still fast.
4. **De-prioritized for now:** alternative loss functions, ensemble/voting across IMP-01/02/03's predictions, and further threshold calibration. All three IMP- experiments landed in the same narrow band (~0.10-0.15 FPR, ~0.42-0.50 precision) as every replication-phase attempt — consistent with a shared information ceiling, not a model-fitting problem. Averaging or re-weighting predictions that all sit in the same place is unlikely to produce something meaningfully different; the replication phase's own threshold sweep already showed calibration specifically doesn't move this. Revisit only if 1-3 produce a genuinely different result worth combining with something else.
