"""
Detector protocol + shared helpers for the intra-patient anomaly detectors.

All detectors share the same surface so the ensemble orchestrator can treat
them interchangeably:

    detector = Foo(...)
    detector.fit(calm_features)          # (n_calm, n_features)
    scores = detector.score(features)    # (n_windows,) in [0, 1]
    contribs = detector.attribution(features, top_k=3)
        # list[list[(feature_name, contribution_value)]] per window

`contribution_value` is detector-specific (z-score, residual, ...). Larger
absolute values mean "more responsible for the score". The ensemble layer
only needs the relative ranking, not the absolute scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


# A tuple kept compact: (feature_name, raw_contribution).
Contribution = tuple[str, float]


class Detector(Protocol):
    """Common interface implemented by every Sprint-4 detector."""

    feature_names: tuple[str, ...]

    def fit(self, calm_features: np.ndarray) -> None: ...

    def score(self, features: np.ndarray) -> np.ndarray: ...

    def attribution(self, features: np.ndarray,
                    top_k: int = 3) -> list[list[Contribution]]: ...


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
MAD_SCALE: float = 1.4826
EPS: float = 1e-12


@dataclass(frozen=True)
class CalmStats:
    """Per-feature median and scaled-MAD computed on the calm pool."""
    median: np.ndarray   # (n_features,)
    mad: np.ndarray      # (n_features,)


def impute_nan_per_column(features: np.ndarray,
                          column_medians: np.ndarray | None = None) -> np.ndarray:
    """Replace NaN with the per-column median (or 0 if all-NaN)."""
    features = np.asarray(features, dtype=float).copy()
    if column_medians is None:
        column_medians = np.nanmedian(features, axis=0)
        column_medians = np.where(np.isnan(column_medians), 0.0, column_medians)
    nan_mask = np.isnan(features)
    if nan_mask.any():
        idx_rows, idx_cols = np.where(nan_mask)
        features[idx_rows, idx_cols] = column_medians[idx_cols]
    return features


def fit_calm_stats(calm_features: np.ndarray) -> CalmStats:
    """Robust per-feature location and scale on the calm pool.

    NaN handling: per-column median uses `nanmedian` so isolated NaNs do not
    poison the location estimate; the MAD then ignores those positions via
    the same trick. Downstream detectors should still call `impute_nan_per_column`
    before fitting sklearn estimators.
    """
    med = np.nanmedian(calm_features, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    raw_mad = np.nanmedian(np.abs(calm_features - med[None, :]), axis=0)
    raw_mad = np.where(np.isnan(raw_mad), 0.0, raw_mad)
    mad = np.maximum(MAD_SCALE * raw_mad, EPS)
    return CalmStats(median=med.astype(float), mad=mad.astype(float))


def robust_z(features: np.ndarray, stats: CalmStats) -> np.ndarray:
    """(x - median) / scaled-MAD, broadcast per feature; NaN -> 0."""
    z = (features - stats.median[None, :]) / stats.mad[None, :]
    z = np.where(np.isnan(z), 0.0, z)
    z = np.where(np.isinf(z), 0.0, z)
    return z


def normalise_quantile(scores: np.ndarray, calm_scores: np.ndarray) -> np.ndarray:
    """Map every score onto the empirical CDF of the calm-pool scores.

    Returns values in [0, 1]: 0 = at-or-below the calm minimum, 1 = at-or-above
    the calm maximum. Equivalent to an empirical p-value reflected to make
    high = anomalous.
    """
    if len(calm_scores) == 0:
        return np.zeros_like(scores)
    sorted_calm = np.sort(calm_scores)
    ranks = np.searchsorted(sorted_calm, scores, side="right")
    return np.clip(ranks / max(len(sorted_calm), 1), 0.0, 1.0)


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k absolute values (descending)."""
    if k <= 0 or len(values) == 0:
        return np.zeros(0, dtype=int)
    k = min(k, len(values))
    return np.argsort(-np.abs(values))[:k]
