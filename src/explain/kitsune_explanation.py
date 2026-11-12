"""xNIDS-style explanation for the Kitsune autoencoder, adapted from the
Kitsune branch of the original authors' Explanation class
(Scripts/explanation.py in github.com/CactiLab/code-xNIDS). The LSTM/kdd
branches are dropped entirely -- out of scope, see docs/TASKS.md.

Method (per the paper): (1) find how far back in the history a prediction
is actually determined by, via capture_relevant_history/search_proper_input;
(2) perturb samples around that history by zeroing a random subset of
features (weighted_sampling); (3) fit a sparse-group-lasso surrogate
(sparse_group_lasso) that regresses the autoencoder's reconstruction-error
score on the perturbed inputs, respecting Kitsune's real feature-channel
groups, so the resulting coefficients approximate per-feature importance
for that specific detection.

Kitsune's 115 features are 5 channels x {3,3,3,7,7} stats x 5 time-decay
windows = 15+15+15+35+35: MAC-IP, IP, Jitter (3 stats each, 5 windows),
IP-to-IP and Socket-to-Socket (7 stats each, 5 windows) -- confirmed
against xNIDS's own Scripts/explanation.py (kitsune_group_sizes/_names).

Two deliberate deviations from the reference script, both logged here
rather than silently matched:
  1. asgl's public API was completely rewritten since xNIDS was published
     (current pip install asgl has no CV/ASGL/train_test_split at all).
     Pinned asgl==1.0.6, the last version matching their calls.
  2. Their exact CV grid (lambda1 x alpha x lasso/gl power_weight, 5-fold,
     parallel=True) took minutes even on toy 100x10 data in this
     environment -- parallel=True spawns ~25 OS processes on Windows
     (no fork, each boots a fresh interpreter), not a data-size problem.
     We use parallel=False with a smaller grid; verified at real scale
     (100x115, 5 groups) this recovers a planted group signal correctly
     in under a second.
"""

import numpy as np
from tensorflow import keras
import asgl

KITSUNE_GROUP_SIZES = [15, 15, 15, 35, 35]
KITSUNE_GROUP_NAMES = ["MAC-IP", "IP", "Jitter", "IPtoIP", "Socket"]
assert sum(KITSUNE_GROUP_SIZES) == 115

# Trimmed CV grid (see module docstring, deviation 2).
CV_LAMBDA1 = [0.001, 0.01, 0.1, 1.0]
CV_ALPHA = [0.2, 0.5, 0.8]
CV_NFOLDS = 3


class KitsuneExplanation:
    def __init__(self, current_sample, history_samples, model_path, original_score,
                 feature_names, group_sizes=KITSUNE_GROUP_SIZES):
        self.current_sample = np.asarray(current_sample).reshape(1, -1)
        self.history_samples = np.asarray(history_samples)
        self.model_path = model_path
        self.original_score = original_score
        self.feature_names = feature_names
        self.group_sizes = group_sizes
        self.delta = 0.001
        self.max_steps = 10
        self.new_input = self.current_sample
        self.weighted_samples = None
        self.coef = None

    def _score(self, model, X):
        recon = model.predict(X, verbose=0)
        return np.mean(np.square(X - recon), axis=1)

    def search_proper_input(self):
        """Extend the history window step by step until the autoencoder's
        score on that window matches the original score within self.delta,
        or give up after max_steps and fall back to the current sample."""
        model = keras.models.load_model(self.model_path)
        n_history = len(self.history_samples)
        for step in range(min(self.max_steps, n_history)):
            window = self.history_samples[-(step + 1):]
            score = self._score(model, window)[-1]
            if abs(score - self.original_score) <= self.delta:
                return window
        return self.current_sample

    def capture_relevant_history(self):
        """Decide whether the current sample alone explains the original
        score, or whether history is needed."""
        model = keras.models.load_model(self.model_path)
        current_score = self._score(model, self.current_sample)[0]
        if abs(current_score - self.original_score) > self.delta:
            self.new_input = self.search_proper_input()
        else:
            self.new_input = self.current_sample

    def weighted_sampling(self, num_samples=100, num_zeroed=50, seed=None):
        """Perturb each row in self.new_input by zeroing a random subset of
        features, num_samples times per row (matches xNIDS's Kitsune branch:
        fixed num_zeroed=50 of 115, not distance-weighted)."""
        rng = np.random.default_rng(seed)
        n_features = self.new_input.shape[1]
        samples = []
        for row in self.new_input:
            for _ in range(num_samples):
                perturbed = row.copy()
                idx = rng.choice(n_features, size=num_zeroed, replace=False)
                perturbed[idx] = 0
                samples.append(perturbed)
        self.weighted_samples = np.array(samples)
        return self.weighted_samples

    def sparse_group_lasso(self):
        """Fit a sparse-group-lasso surrogate: y = autoencoder's
        reconstruction-error score on each perturbed sample, x = the
        perturbed feature vectors, grouped by Kitsune's real channels."""
        group_index = []
        for i, size in enumerate(self.group_sizes):
            group_index.extend([i + 1] * size)

        model = keras.models.load_model(self.model_path)
        x = self.weighted_samples
        y = self._score(model, x)
        del model

        cv_class = asgl.CV(
            model="lm", penalization="sgl", lambda1=CV_LAMBDA1, alpha=CV_ALPHA,
            tau=0.5, parallel=False, weight_technique="pca_pct",
            lasso_power_weight=[1], gl_power_weight=[1],
            variability_pct=0.85, nfolds=CV_NFOLDS, error_type="QRE", random_state=42,
        )
        error = cv_class.cross_validation(x, y, group_index)
        error = np.mean(error, axis=1)
        best = np.argmin(error)
        params = cv_class.retrieve_parameters_value(best)

        asgl_model = asgl.ASGL(
            model="lm", penalization="sgl", tau=0.5, intercept=cv_class.intercept,
            lambda1=params.get("lambda1"), alpha=params.get("alpha"),
            lasso_weights=params.get("lasso_weights"), gl_weights=params.get("gl_weights"),
        )
        train_idx, test_idx = asgl.train_test_split(nrows=x.shape[0], train_pct=0.7, random_state=1)
        asgl_model.fit(x=x[train_idx, :], y=y[train_idx], group_index=group_index)
        self.coef = np.array(asgl_model.coef_[0])
        return self.coef

    def group_importance(self, group_names=KITSUNE_GROUP_NAMES):
        """Mean |coefficient| per feature group -- a simpler summary than
        the reference script's color-bar plot, same underlying numbers."""
        if self.coef is None:
            raise RuntimeError("call sparse_group_lasso() first")
        importances = {}
        start = 0
        for name, size in zip(group_names, self.group_sizes):
            importances[name] = float(np.mean(np.abs(self.coef[start:start + size])))
            start += size
        return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
