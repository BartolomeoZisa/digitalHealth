"""
Robust-quantile detector, the interpretable baseline.

The detector estimates the per-feature `median` and scaled-MAD on the calm
pool of one patient. The anomaly score of a window is the maximum robust
z-score across its features, mapped onto [0, 1] via `1 - exp(-z / k)` so
that `z = k` produces score ≈ 0.63 and `z >> k` saturates near 1.

Attribution: feature whose absolute robust z-score is the largest are the
ones responsible for the score; we return them ordered by |z|.
"""
from __future__ import annotations

import numpy as np

import config
from detectors.base import (
    CalmStats,
    Contribution,
    fit_calm_stats,
    robust_z,
    topk_indices,
)


def _saturate_z(z: np.ndarray, scale: float) -> np.ndarray:
    """`1 - exp(-z / scale)` clipped to [0, 1]; only the positive tail matters."""
    z = np.maximum(z, 0.0)
    return 1.0 - np.exp(-z / max(scale, 1e-9))


class RobustQuantileDetector:
    """Per-feature median + MAD anomaly detector."""

    def __init__(self, feature_names: tuple[str, ...],
                 k: float | None = None) -> None:
        self.feature_names = feature_names
        self.k = float(k) if k is not None else config.DETECTOR_KMAD
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

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("RobustQuantileDetector not fitted")
        # Take absolute z so that both tails of each feature contribute.
        z = np.abs(robust_z(features, self._stats))
        # Per-window score: the most-anomalous feature drives the score.
        max_z = z.max(axis=1)
        return _saturate_z(max_z, scale=self.k)

    def attribution(self, features: np.ndarray,
                    top_k: int = 3) -> list[list[Contribution]]:
        if self._stats is None:
            raise RuntimeError("RobustQuantileDetector not fitted")
        z_signed = robust_z(features, self._stats)
        out: list[list[Contribution]] = []
        for i in range(len(features)):
            idx = topk_indices(z_signed[i], top_k)
            out.append([
                (self.feature_names[j], float(z_signed[i, j]))
                for j in idx
            ])
        return out

    # ------------------------------------------------------------------
    # Inspectors used by the explainer module.
    # ------------------------------------------------------------------
    @property
    def stats(self) -> CalmStats:
        if self._stats is None:
            raise RuntimeError("RobustQuantileDetector not fitted")
        return self._stats
