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
| IMP-01 | | | | | | | | | | |
| IMP-02 | | | | | | | | | | |
| IMP-03 | | | | | | | | | | |

**Column guide:**
- **Strategy**: which numbered idea above (or a new one — add it to the list first with impact/effort estimates before running it).
- **Configuration**: exact parameters/code path, enough detail that a teammate could rerun it from this row alone.
- **Verdict**: one of `Improved`, `No change`, `Worse`, `Inconclusive — needs rerun`.
- **Notes**: anything that would change how the next experiment should be designed — this is the field that turns a log into a roadmap instead of a list.
