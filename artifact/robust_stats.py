"""
Robust statistics used by the artifact filter.

The acceleration windows of one BBT session are not i.i.d. Gaussian samples:
they include occasional gross outliers (arm raises) interleaved with normal
stillness. Mean and standard deviation are therefore unreliable estimators of
the central tendency. We use the median and the Median Absolute Deviation
(MAD), which have a 50% breakdown point and are scale-equivariant.

The iterative thresholding routine recomputes the robust statistics after
removing the previously detected outliers, until either the outlier set
stabilises or `max_iter` is reached.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scale factor so that for Gaussian data MAD * MAD_SCALE -> sigma.
MAD_SCALE: float = 1.4826


def mad(x: np.ndarray, *, scale: bool = True) -> float:
    """Median absolute deviation.

    Returns the (optionally scaled) MAD as a float. A small floor (1e-12) is
    applied so that division never explodes when all samples are identical.
    """
    med = float(np.median(x))
    raw = float(np.median(np.abs(x - med)))
    out = MAD_SCALE * raw if scale else raw
    return max(out, 1e-12)


def robust_z(x: np.ndarray, med: float, mad_value: float) -> np.ndarray:
    """Robust z-score: (x - median) / scaled-MAD."""
    return (x - med) / mad_value


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of one iterative-threshold run.

    `mask_outlier` is True for samples flagged as outliers at convergence.
    `threshold` is the final upper threshold `median + k * mad`. We do not
    apply a lower threshold here because all four artifact indicators are
    non-negative magnitudes; large values are always suspicious, small ones
    are stillness.
    """
    mask_outlier: np.ndarray   # bool (n,)
    threshold: float
    median: float
    mad_value: float
    n_iter: int


def iterative_threshold(values: np.ndarray, k: float,
                        max_iter: int = 5) -> ThresholdResult:
    """One-sided robust threshold with iterative refinement.

    At every iteration we compute the median and MAD on the currently
    "inlier" subset (initially the whole array) and define the threshold
    as `median + k * mad`. Samples above the threshold are flagged outlier
    for the next iteration. The loop terminates when the outlier mask is
    stable or after `max_iter` iterations.
    """
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {values.shape}")

    n = len(values)
    if n == 0:
        return ThresholdResult(
            mask_outlier=np.zeros(0, dtype=bool),
            threshold=float("nan"),
            median=float("nan"),
            mad_value=float("nan"),
            n_iter=0,
        )

    mask_outlier = np.zeros(n, dtype=bool)
    for it in range(1, max_iter + 1):
        inliers = values[~mask_outlier]
        if len(inliers) < 3:
            # Not enough inliers to estimate stats; bail out.
            break
        med = float(np.median(inliers))
        m = mad(inliers, scale=True)
        thr = med + k * m
        new_mask = values > thr
        if np.array_equal(new_mask, mask_outlier):
            break
        mask_outlier = new_mask

    return ThresholdResult(
        mask_outlier=mask_outlier,
        threshold=float(thr),
        median=med,
        mad_value=m,
        n_iter=it,
    )


def quantiles(values: np.ndarray,
              q: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)
              ) -> dict[float, float]:
    """Sample quantiles, returned as a `{q: value}` dict."""
    if len(values) == 0:
        return {float(qi): float("nan") for qi in q}
    qs = np.quantile(values, q)
    return {float(qi): float(v) for qi, v in zip(q, qs)}
