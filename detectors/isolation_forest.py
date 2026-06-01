"""
IsolationForest anomaly detector, non-linear tree-based.

Fitted on the calm pool of one patient. Raw scores from `decision_function`
are higher for inliers and lower for outliers; we flip the sign so that
"higher = more anomalous" and remap to [0, 1] via the empirical CDF of the
calm-pool scores.

Attribution: sklearn `IsolationForest` does not expose per-feature
contributions natively. We approximate the contribution of feature `j` to
the score of a window `x` with the absolute robust z-score of `x_j` against
the calm pool. This stays consistent with the quantile detector and gives
the user a comparable explanation.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

import config
from detectors.base import (
    CalmStats,
    Contribution,
    fit_calm_stats,
    impute_nan_per_column,
    normalise_quantile,
    robust_z,
    topk_indices,
)


class IsolationForestDetector:
    def __init__(self, feature_names: tuple[str, ...],
                 n_estimators: int | None = None,
                 max_samples: int | None = None,
                 random_state: int | None = None) -> None:
        self.feature_names = feature_names
        self.n_estimators = n_estimators or config.IFOREST_N_ESTIMATORS
        self.max_samples = max_samples or config.IFOREST_MAX_SAMPLES
        self.random_state = random_state if random_state is not None else config.RANDOM_SEED
        self._model: IsolationForest | None = None
        self._calm_scores: np.ndarray | None = None
        self._stats: CalmStats | None = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, calm_features: np.ndarray) -> None:
        if calm_features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"calm_features has {calm_features.shape[1]} columns but "
                f"detector expects {len(self.feature_names)}"
            )
        self._stats = fit_calm_stats(calm_features)
        calm_clean = impute_nan_per_column(calm_features, self._stats.median)
        self._column_medians = self._stats.median

        max_samples = min(self.max_samples, len(calm_clean))
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(calm_clean)
        # Higher = more anomalous. Save the calm-pool distribution for the
        # quantile-based normalisation in `score`.
        raw_calm = -self._model.decision_function(calm_clean)
        self._calm_scores = np.asarray(raw_calm, dtype=float)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._model is None or self._calm_scores is None or self._stats is None:
            raise RuntimeError("IsolationForestDetector not fitted")
        clean = impute_nan_per_column(features, self._stats.median)
        raw = -self._model.decision_function(clean)
        return normalise_quantile(np.asarray(raw, dtype=float), self._calm_scores)

    def attribution(self, features: np.ndarray,
                    top_k: int = 3) -> list[list[Contribution]]:
        if self._stats is None:
            raise RuntimeError("IsolationForestDetector not fitted")
        # Approximation: rank features by the absolute robust z-score of the
        # input. This matches the explainer used elsewhere and avoids the
        # expensive (and unstable on small N) SHAP computation.
        z_signed = robust_z(features, self._stats)
        out: list[list[Contribution]] = []
        for i in range(len(features)):
            idx = topk_indices(z_signed[i], top_k)
            out.append([
                (self.feature_names[j], float(z_signed[i, j]))
                for j in idx
            ])
        return out
