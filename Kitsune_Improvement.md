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
| IMP-07 | 2026-11-19 | Round 3, Exp 3.1: differential (delta) feature transformation | If the attack-window disturbance is a roughly uniform scale shift across features, first-differencing (feature_t - feature_t-1) should cancel it while leaving genuine attack signatures intact | `src/models/kitsune_improvement_delta_features.py`: same architecture/split as baseline, but trained on X[i]-X[i-1] instead of absolute features (fresh scaler fit on the deltas) | 0.3368 | 0.0299 | 0.0061 | 0.0550 | Worse (recall collapse) | **FPR dropped sharply (0.1039 -> 0.0061) but recall collapsed even harder (1.0000 -> 0.0299).** Differencing cancels the ambient-disturbance signal for benign rows, but it ALSO erases whatever distinguished real attacks -- meaning the attack itself shows up as a *sustained* elevation over many rows, not a sharp one-row spike, so taking a derivative throws out the useful signal along with the noise. At forced FPR=0.001, recall is 0.0042 -- worse than the baseline's 0.0191. Rules out "the problem is scale drift, not shape" as the explanation. |
| IMP-08 | 2026-11-19 | Round 3, Exp 3.2: conditional autoencoder (global network context) | Giving the model a rolling-context vector (mean of the last 50 rows) as an additional input should let it learn "given recent traffic looked like X, normal right now looks like Y," rather than asking it to generalize past what training showed it | `src/models/kitsune_improvement_conditional.py`: input = [current 115 features ++ 115-dim rolling-mean context], output = reconstruct only current 115 features; same architecture depth/hyperparameters otherwise | 0.4976 | 1.0000 | 0.1049 | 0.6645 | No change | Essentially identical to the unconditioned baseline (recall@FPR=0.001: 0.0183 vs. baseline's 0.0191). The context vector is built from the same 115 features being conditioned on, so during a sustained disturbance both the current row AND its recent history are elevated together -- the context carries little information independent of what the current row already shows, so the network had nothing new to condition on and effectively learned to ignore it. A genuinely independent context source (not derived from the same features) would be needed to test this idea properly; we don't have one without reprocessing raw packets. |
| IMP-09 | 2026-11-19 | Round 4: adaptive local threshold (no retraining) | Replacing the static global threshold (mean+2std of train error, fixed) with a rolling LOCAL threshold computed from the error stream's own recent history should stop flagging a persistently-elevated region as anomalous relative to a now-local baseline | `src/models/kitsune_improvement_adaptive_threshold.py`: loads the existing baseline model (no retraining), computes a causal rolling mean+k*std over a trailing window=5000 of the chronological error stream (train tail warms up test's early rows) | 0.0568 (k=2) / 0.2841 (best-k=4.5) | 0.0245 (k=2) / 0.0041 (best-k=4.5) | 0.0423 (k=2) / 0.0011 (best-k=4.5) | 0.0343 (k=2) | Worse (substantially) | **At the standard k=2, dramatically worse than the static baseline across every metric.** At the k that forces FPR closest to the paper's 0.001 target (k=4.5), recall is 0.0041 -- worse than the static threshold's 0.0191 at the same FPR. This fits the mechanism IMP-07 already established: because the attack causes a *sustained* elevation (not a spike), a rolling window that includes the elevated attack-period errors has its own local mean/std rise right along with them -- the local threshold chases the attack's own elevated baseline upward and becomes progressively less sensitive to it, the opposite of the intended effect. This was the last untested idea from the brainstorm list; it closes the same way the others did. |

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

---

## Round 2 Results: Updated Cumulative Comparison

Executed sequentially (2.1 → 2.2 → 2.3), reviewing and logging each before starting the next, per the Round 2 execution constraint.

| | Precision | Recall | FPR |
|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 |
| Replicated baseline | 0.5001 | 1.0000 | 0.1039 |
| Round 1 best (IMP-01/03, tied, no real effect) | 0.5001–0.4997 | 1.0000 | 0.1039–0.1040 |
| Round 1 worst (IMP-02, window=5 LSTM) | 0.4161 | 1.0000 | 0.1458 |
| IMP-04 (Round 2, Exp 2.1: ratio post-filter) | 0.5002 | 0.9620 | 0.0999 |
| IMP-05 (Round 2, Exp 2.2: targeted augmentation) | 0.5004 | 1.0000 | 0.1038 |
| IMP-06 (Round 2, Exp 2.3: window=15 LSTM) | 0.2766 | ~1.0000 | 0.2718 |

**Performance verdict: Round 2 did not bridge the gap — if anything, it sharpened the picture of why.** Nothing in either round came close to the paper's numbers. More importantly, Round 2 specifically tested the *best, most targeted version* of each Round 1 idea (a ratio instead of a threshold, the real measured shift instead of generic noise, 3x more context instead of too little) and every one of those improved versions either made no real difference or made things measurably worse. IMP-05 in particular is the cleanest negative result in the whole project: using the exact real distributional shift, properly leak-controlled, produced a 4-row difference out of 71,848 on the exact population it targeted. We have plateaued, and the plateau is well-characterized rather than merely assumed.

---

## Phase 3: Beyond Standard Tuning

### Brainstormed unconventional strategies, ranked

1. **Adaptive/local threshold instead of a single global one (most promising, cheapest to test).** Rather than training a new model, compute a rolling average of the EXISTING baseline detector's per-row reconstruction error over a short trailing window (e.g. 50-100 rows) as a live "local disturbance level," and raise the flagging threshold when that local level is already elevated — the logic being that during any genuinely disturbed network period, some baseline elevation in benign traffic is expected, so the bar for flagging an individual row should rise with it, reserved for rows far above even the locally-elevated norm. This is conceptually different from everything tried (no new model, context via a simple statistic rather than a learned sequence model) and cheap enough to test in under an hour.
2. **Richer features from the raw packets.** Every experiment so far, in both replication and improvement, used Kitsune's same 115 pre-extracted features. The OS Scan scenario's raw pcap (333MB, deliberately skipped when downloading to save bandwidth) could be reprocessed for additional payload- or protocol-level features not in Kitsune's original 115. This is a genuinely different information source, not a reweighting of what's already been tried — but it's a real time investment, not a quick experiment.
3. **Pseudo-labeling / self-training.** Use the model's own highest-confidence predictions to iteratively retrain. Flagged here mainly for completeness: given recall is already 1.0 (nothing confidently-attack is being missed) and the entire problem is benign-side confusion, self-training is unlikely to manufacture information the feature representation doesn't already contain — low expected value.
4. **Ensemble/voting across all detectors built so far** (monolithic AE, real KitNET, Isolation Forest, LSTM). De-prioritized: these different algorithms land in the same ~0.10-0.27 FPR band and, from the replication phase's own diagnosis, are likely flagging substantially the same rows (the same attack-window population), since the underlying cause is shared. Voting across detectors that agree isn't expected to change the outcome.

### Project Closure Recommendation

**Further quick experimentation has clearly diminishing returns.** Twelve experiments now (six in replication, six in improvement — IMP-01 through IMP-06), each targeting a genuinely different possible cause, plus a two-pass exhaustive document/code/community audit, have all converged on the same conclusion: this is an information ceiling in Kitsune's 115-feature representation during this specific attack window, not a model-fitting problem any of the standard levers can fix.

**Recommendation at the time: close this round of experimentation and move to packaging the findings.** If there's remaining appetite and time, strategy #1 above (adaptive local threshold) is cheap enough to be worth one more shot since it's genuinely untested territory — but it should not block finishing the Wiki report, repo hygiene, and presentation prep, all of which are time-boxed by the Dec 1/3 deadline in a way that further modeling experiments are not. The empirical record built across both documents (`Kitsune_Replication.md` and this one) is already a strong, well-evidenced capstone result on its own terms.

---

## Round 3: External Ideas (Delta Features & Conditional Autoencoder)

Run after getting a second opinion on remaining improvement ideas. Two of five externally-suggested strategies were judged worth testing (differential features and conditional/context-aware autoencoding); the other three (disentangled representation learning, masked transformer autoencoders, semi-supervised contrastive fine-tuning) were judged too research-heavy or too close to an already-tested idea for the remaining timeline — see the conversation record for the full reasoning.

| | Precision | Recall | FPR | Recall @ forced FPR=0.001 |
|---|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 | 0.9972 |
| Replicated baseline | 0.5001 | 1.0000 | 0.1039 | 0.0191 |
| IMP-07 Delta features | 0.3368 | 0.0299 | 0.0061 | 0.0042 |
| IMP-08 Conditional autoencoder | 0.4976 | 1.0000 | 0.1049 | 0.0183 |

**Verdict: Round 3 did not close the gap either, and for a genuinely informative reason this time.** IMP-07 revealed something new: the attack signature itself is a *sustained* elevation, not a sharp spike, so any transformation that cancels slow drift (differencing) cancels the attack signal along with the noise — this rules out an entire class of "normalize away the drift" approaches, not just this one implementation. IMP-08 revealed that conditioning the model on its own recent history doesn't help when the context source isn't informationally independent from what's being conditioned — a real, final line in the sand for any future context-injection idea that forgets this.

### Final Closure Recommendation (Round 3)

**Fifteen experiments at that point** (six replication + eight improvement + the direct forced-FPR protocol test), spanning every category of fix — architecture, algorithm, training data, temporal context (twice, at two scales), supervision, post-hoc filtering (twice), feature transformation, and context conditioning — had all converged on the same wall, each for its own specific, now-documented reason. No further modeling experiments were recommended before the Dec 1/3 presentations at that point, with one exception: the adaptive local threshold idea, left open as the one remaining untested item.

---

## Round 4: Final Diagnostics & Adaptive Threshold — Experimental Phase Locked

Two root-cause diagnostics (label-granularity and entity-aggregation checks — full writeups in `Kitsune_Replication.md`'s "Root-Cause Diagnostics" section, no retraining involved) plus the one remaining untested idea, run as the final experiment before locking this phase entirely.

| | Precision | Recall | FPR |
|---|---|---|---|
| **Target (xNIDS paper, Table 12)** | 0.9568 | 0.9972 | 0.001 |
| Static global threshold (baseline) | 0.5001 | 1.0000 | 0.1039 |
| IMP-09 Adaptive local threshold, k=2 | 0.0568 | 0.0245 | 0.0423 |
| IMP-09 Adaptive local threshold, best-k=4.5 (FPR≈0.001) | 0.2841 | 0.0041 | 0.0011 |

**Diagnostic A finding:** confirmed no IP/MAC/packet metadata exists anywhere in the processed dataset or labels file (only the undownloaded raw pcap has it) — but the label-structure check revealed something new: the "attack window" is not a blanket time-range stamp, it's 34,069 scattered short runs within a 261,921-row span at 25% density. This means our false positives are on rows *correctly* labeled benign, not mislabeling artifacts — a real detector limitation, not a dataset quirk.

**Diagnostic B finding:** no entity/flow identifier exists to aggregate by; the closest proxy (temporal-window aggregation) was already tested via IMP-02/IMP-06 and made things worse, not better.

**IMP-09 finding:** adaptive local thresholding is worse than the static baseline at every comparable operating point, for a mechanistically clear reason — a rolling window that includes the attack's own sustained elevation has its local statistics rise right along with it, so the threshold chases the attack upward instead of catching it.

### Final Closure — Experimental Phase Locked

**Sixteen total experiments** (six replication + nine improvement + the forced-FPR protocol test), plus two root-cause diagnostics that confirmed the finding isn't a labeling or metadata artifact, now span every category of fix that doesn't require re-engineering the feature-extraction pipeline from raw packets. The experimental phase is closed. Remaining project time goes to the GitHub Wiki report, repo hygiene, and Dec 1/3 presentation preparation — see `README.md` and `docs/TASKS.md` for the current checklist.
