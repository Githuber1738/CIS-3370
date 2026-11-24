# US-07: Explanation Fidelity (ADA) and Sparsity (MAZ)

Evaluates our xNIDS-style sparse-group-lasso explanation (`src/explain/`) against our own trained Kitsune autoencoder, reproducing the exact methodology of xNIDS Section 6.1.1/6.1.2 (Figs. 3-4, Tables 3-5). All computed from the already-trained baseline model and the already-built explanation pipeline — no retraining.

## A bug found and fixed first

Building this surfaced a real bug in `src/explain/kitsune_explanation.py`: `asgl`'s SGL solver prepends the intercept as its own group when `intercept=True` (confirmed from the pinned `1.0.6` source). The code was storing that 116-length vector directly as the per-feature coefficient array, so every channel-group slice was off by one, and the true last feature (`feat_114`) was silently dropped from every group. **Fixed** — see `docs/TASKS.md`'s 2026-11-19 correction entry for the full account. The corrected result reverses the 2026-11-12 Troubleshooting Showcase finding: **IP, not MAC-IP, dominates feature importance** for both the original hand-picked TP and FP samples.

## Methodology and scope

The paper's own methodology (quoted from Section 6.1.1):
- **ADA** (Average Descriptive Accuracy): for correctly-detected ANOMALY samples, zero out the top-*k* most important features (by \|coefficient\|) and re-score through the detector; ADA(*k*) = % still classified anomalous, averaged over the population, for *k* = 0..10. A steep decline means the explanation found features that actually drive the detection.
- **Benign-sample side check** (the paper's Table 4): zeroing benign features barely moves the score (the paper's own finding), so instead replace the top-*k* important features of a correctly-classified BENIGN sample with the corresponding values from its nearest anomaly sample, and report the resulting "anomalous rate" (fraction newly misclassified) at *k*=10.
- **MAZ** (Mass Around Zero): per-sample \|coefficient\| scaled to [0,1] (own max = 1.0), pooled across every explained sample+feature, fit to a half-normal distribution; MAZ(*c*) = that distribution's CDF at interval size *c*. AUC over *c* ∈ [0,1] measures sparsity — more mass near zero is better.

**Sample population** (stratified to directly test whether fidelity/sparsity shifts inside vs. outside the attack span, per this project's own investigative interest): 20 correctly-detected anomaly (TP) samples, 15 correctly-classified benign samples from inside the attack span, 15 from outside it — all drawn randomly from much larger pools (65,700 / 130,639 / 435,851 respectively). This is a representative-sample estimate, not the paper's full-dataset average; see Scope Deferrals below.

## Results

| Metric | Our result | Paper (xNIDS/Kitsune) |
|---|---|---|
| ADA AUC (k=0..10, zero-out, anomaly samples) | **1.000** | 0.316 |
| Benign anomalous rate at k=10 (replace-with-nearest-anomaly), in-window | **1.000** | 0.239 (combined) |
| Benign anomalous rate at k=10, out-of-window | **1.000** | 0.239 (combined) |
| MAZ AUC | **0.817** | 0.774 |

ADA(*k*) itself is flat at 100% for every *k* from 0 to 10 — not a gradual decline, a hard ceiling.

## Why ADA and the benign rate both saturate (verified, not assumed)

Both saturated results are mechanistically explainable, confirmed by inspecting actual pre/post-modification error values rather than just the binary classification flags:

- **Anomaly samples**: original reconstruction error ≈0.173, static threshold ≈0.00076 — the error sits **~226x above threshold**. Zeroing the top 10 (of 115) most important features drops it only to ≈0.160, still **~210x above threshold**. Removing 8.7% of the input can't meaningfully move an error that's two orders of magnitude past the decision boundary.
- **Benign samples**: original error ≈0.00011 (safely below threshold). Replacing the top 10 features with a nearest attack sample's values means a jump of roughly 0.77-0.78 (on a [0,1] scaled feature) in each of those 10 dimensions — large enough that the resulting error (≈0.053, **~69x above threshold**) trivially clears the tiny threshold regardless of which 10 features were chosen.

**This is the same finding as the rest of this project, from a new angle.** Every one of the 16 prior experiments converged on the same cause: elevated error during the attack window is diffuse across nearly all 115 features, not concentrated in a handful. ADA stuck at 100% is that same diffuseness seen through the explanation-fidelity lens — no small, targeted feature subset drives the detection enough for its removal to matter, because nothing is concentrated enough to remove. MAZ (0.817) lands close to the paper's own number because it measures something different — the *shape* of the coefficient distribution itself, not whether modifying the ranked features changes the outcome — and that shape question isn't affected by the diffuseness problem the same way.

**Important scope caveat:** this evaluates explanation fidelity against *our* detector, whose own precision/recall/FPR already diverge sharply from the paper's (see `Kitsune_Replication.md`). A direct numeric comparison to the paper's ADA/MAZ columns is informative but not apples-to-apples — we are not just scoring our explanation method against theirs, we are also implicitly scoring it against a detector with a fundamentally different (much larger, much more diffuse) error-to-threshold margin than whatever model produced their Table 3/5 numbers.

## Scope deferrals (US-07 subtasks)

- **T-07.1, T-07.2, T-07.3** (implement ADA, implement MAZ, run both against our explanations): done, above.
- **T-07.4** (baseline-explainer comparison, e.g. SHAP): **explicitly deferred, not attempted.** Implementing a model-agnostic baseline explainer (SHAP/LIME/etc.) and running it against our reconstruction-error-based detector is a separate, non-trivial build — and given our own detector already diverges from the paper's, a baseline-explainer comparison would mainly tell us how explanation methods differ on a non-representative detector, which is of secondary value against the Dec 1/3 deadline. Recommend closing this subtask as explicitly out of scope rather than leaving it open indefinitely.
- **T-07.5** (document divergence from the paper's Table 3/5 Kitsune columns): done, in the table above.

## Per-feature importance visualization (T-06.3)

`group_importance()` reports the mean \|coefficient\| per channel — useful for a
one-line summary ("IP dominates"), but it dilutes signal that's concentrated in
a few individual features within a large group. `scripts/visualize_feature_importance.py`
re-runs the exact same explanation pipeline (`KitsuneExplanation`, same seed=42
weighted sampling) on the same TP/FP samples, but keeps the raw per-feature
coefficients and plots the top 15 by \|coefficient\| for each, colored by channel.

**Finding**: at the individual-feature level, the single largest bars for both
the TP and the FP sample are two **IPtoIP** features (`feature_58`, `feature_77`),
not MAC-IP or IP. This doesn't contradict the group-level finding — IPtoIP is
the largest channel (35 of 115 features), so a couple of strongly-weighted
features inside it can dominate the per-feature ranking while still averaging
out to a lower *mean* than IP's more uniformly-weighted 15 features. The two
summaries answer different questions (which single features matter most vs.
which channel matters most on average) and both are legitimate; this is a
clarification, not a correction of the 2026-11-19 IP-dominance finding above.
The TP and FP samples' top-15 lists are nearly identical (same top 2 features,
heavy overlap throughout) — further direct evidence, at this finer grain, that
the FP is a genuine near-miss in feature space rather than a random error.

Saved to `artifacts/kitsune_feature_importance/feature_importance.png`.

## Files

- `src/explain/fidelity_sparsity.py` — the full computation (stratified sampling, ADA curve, benign anomalous-rate check, MAZ curve), no retraining.
- `artifacts/kitsune_fidelity_sparsity/results.json`, `ada_curve.csv`, `maz_curve.csv` — saved outputs.
- `src/explain/kitsune_explanation.py` — the intercept-alignment bug fix.
- `scripts/visualize_feature_importance.py` — T-06.3's per-feature importance chart.
- `artifacts/kitsune_feature_importance/feature_importance.png` — saved output.
