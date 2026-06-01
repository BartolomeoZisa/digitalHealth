"""
PCA reconstruction-error anomaly detector.

We robust-standardise the features against the calm pool, fit a low-rank
PCA model on the standardised calm pool, and use the per-window
reconstruction sum-of-squared-residuals as the raw anomaly score. The score
is then normalised to [0, 1] via the empirical CDF of the calm-pool scores.

Attribution: the per-feature squared residual `(x_i - x_hat_i) ** 2` tells
us which features cannot be explained by the calm subspace; the top-k of
those, signed by the residual sign, are the explanation.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

import config
from detectors.base import (
    CalmStats,
    Contribution,
    EPS,
    fit_calm_stats,
    impute_nan_per_column,
    normalise_quantile,
    robust_z,
    topk_indices,
)


class PCAReconstructionDetector:
    def __init__(self, feature_names: tuple[str, ...],
                 n_components_max: int | None = None,
                 random_state: int | None = None) -> None:
        self.feature_names = feature_names
        self.n_components_max = (
            n_components_max if n_components_max is not None
            else config.PCA_N_COMPONENTS_MAX
        )
        self.random_state = (
            random_state if random_state is not None else config.RANDOM_SEED
        )
        self._stats: CalmStats | None = None
        self._pca: PCA | None = None
        self._calm_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, calm_features: np.ndarray) -> None:
        n_features = calm_features.shape[1]
        if n_features != len(self.feature_names):
            raise ValueError(
                f"calm_features has {n_features} columns but detector "
                f"expects {len(self.feature_names)}"
            )

        self._stats = fit_calm_stats(calm_features)
        clean = impute_nan_per_column(calm_features, self._stats.median)
        z = robust_z(clean, self._stats)

        # n_components capped by min(n_samples, n_features, max).
        n_components = min(self.n_components_max, n_features, len(calm_features))
        n_components = max(n_components, 1)
        self._pca = PCA(n_components=n_components,
                        random_state=self.random_state)
        self._pca.fit(z)

        z_recon = self._pca.inverse_transform(self._pca.transform(z))
        residuals = z - z_recon
        self._calm_scores = (residuals ** 2).sum(axis=1)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None or self._stats is None or self._calm_scores is None:
            raise RuntimeError("PCAReconstructionDetector not fitted")
        clean = impute_nan_per_column(features, self._stats.median)
        z = robust_z(clean, self._stats)
        z_recon = self._pca.inverse_transform(self._pca.transform(z))
        residuals = z - z_recon
        sse = (residuals ** 2).sum(axis=1)
        return normalise_quantile(sse, self._calm_scores)

    def attribution(self, features: np.ndarray,
                    top_k: int = 3) -> list[list[Contribution]]:
        if self._pca is None or self._stats is None:
            raise RuntimeError("PCAReconstructionDetector not fitted")
        clean = impute_nan_per_column(features, self._stats.median)
        z = robust_z(clean, self._stats)
        z_recon = self._pca.inverse_transform(self._pca.transform(z))
        residuals = z - z_recon
        out: list[list[Contribution]] = []
        for i in range(len(features)):
            idx = topk_indices(residuals[i], top_k)
            out.append([
                (self.feature_names[j], float(residuals[i, j]))
                for j in idx
            ])
        return out
