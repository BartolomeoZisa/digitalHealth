"""
Family 4, Temporal-distribution features (per session).

The Sprint 2 evidence showed that `pct_outlier` alone does not separate UCP
from TD: the *distribution in time* of outlier events does. This module
quantifies that distribution.

A "burst" is a contiguous run of outlier windows; two bursts are considered
separate when the gap between them exceeds `BURST_GAP_S` seconds.

Inputs:
    outlier_mask_window : (n_windows,) bool, outlier flag per window
    starts_idx          : (n_windows,)     , sample-index of each window start
    win_len_samples     : int              , samples per window
    fs                  : float
    duration_s          : float            , session length in seconds

Outputs: a flat dict of scalar features.
"""
from __future__ import annotations

import numpy as np

import config

EPS: float = 1e-12


# -----------------------------------------------------------------------------
# Conversion from window-level mask to sample-level mask and to burst spans.
# -----------------------------------------------------------------------------
def window_mask_to_sample_mask(outlier_mask: np.ndarray,
                               starts_idx: np.ndarray,
                               win_len_samples: int,
                               n_samples: int) -> np.ndarray:
    """Project window-level outlier flags onto a sample-level boolean array."""
    sm = np.zeros(n_samples, dtype=bool)
    for ok, start in zip(outlier_mask, starts_idx):
        if not ok:
            continue
        end = min(start + win_len_samples, n_samples)
        sm[start:end] = True
    return sm


def _bursts_from_sample_mask(sample_mask: np.ndarray, fs: float,
                             gap_s: float) -> list[tuple[float, float]]:
    """Group True-runs of a sample-level mask into bursts of (t_start, t_end).

    Two adjacent True-runs are merged into a single burst when the silent gap
    between them is shorter than `gap_s`. Output spans are in seconds.
    """
    if not sample_mask.any():
        return []
    # Find True-run boundaries.
    diff = np.diff(sample_mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if sample_mask[0]:
        starts = np.r_[0, starts]
    if sample_mask[-1]:
        ends = np.r_[ends, len(sample_mask)]
    spans = [(s / fs, e / fs) for s, e in zip(starts, ends)]

    # Merge adjacent spans separated by less than `gap_s`.
    merged: list[tuple[float, float]] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        prev_s, prev_e = merged[-1]
        s, e = span
        if s - prev_e < gap_s:
            merged[-1] = (prev_s, e)
        else:
            merged.append(span)
    return merged


# -----------------------------------------------------------------------------
# Distribution-shape scalars
# -----------------------------------------------------------------------------
def _gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values. Returns 0 for constant input."""
    if len(values) == 0:
        return float("nan")
    x = np.sort(np.asarray(values, dtype=float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _shannon_entropy(counts: np.ndarray, normalise: bool = True) -> float:
    """Shannon entropy (nats) of a histogram; optionally divided by log(K)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    if normalise and len(p) > 1:
        h = h / np.log(len(p))
    return h


def _burstiness_B(intervals: np.ndarray) -> float:
    """Goh-Barabasi burstiness coefficient B = (σ - μ) / (σ + μ).

    -1 → perfectly regular, 0 → Poisson, +1 → very bursty.
    Returns nan for fewer than two intervals.
    """
    if len(intervals) < 2:
        return float("nan")
    mu = float(np.mean(intervals))
    sigma = float(np.std(intervals))
    if mu + sigma < EPS:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def temporal_features_from_sample_mask(sample_mask: np.ndarray,
                                       fs: float,
                                       duration_s: float | None = None,
                                       gap_s: float | None = None,
                                       n_bins: int | None = None
                                       ) -> dict[str, float]:
    """Compute every Family-4 feature from a sample-level outlier mask."""
    if gap_s is None:
        gap_s = config.BURST_GAP_S
    if n_bins is None:
        n_bins = config.TEMPORAL_ENTROPY_BINS
    if duration_s is None:
        duration_s = len(sample_mask) / fs

    bursts = _bursts_from_sample_mask(sample_mask, fs, gap_s)
    burst_durations = np.array([e - s for s, e in bursts], dtype=float)
    burst_centres = np.array([(s + e) / 2.0 for s, e in bursts], dtype=float)

    coverage_pct = 100.0 * float(sample_mask.mean()) if len(sample_mask) else 0.0
    n_bursts = len(bursts)

    if n_bursts >= 2:
        inter_gaps = np.diff(burst_centres)
        mean_burst_dur = float(burst_durations.mean())
        max_burst_dur = float(burst_durations.max())
        cv_burst_dur = (
            float(burst_durations.std() / (burst_durations.mean() + EPS))
            if burst_durations.mean() > EPS else 0.0
        )
        mean_inter_gap = float(inter_gaps.mean())
        cv_inter_gap = (
            float(inter_gaps.std() / (inter_gaps.mean() + EPS))
            if inter_gaps.mean() > EPS else 0.0
        )
        burstiness_B = _burstiness_B(inter_gaps)
    elif n_bursts == 1:
        mean_burst_dur = float(burst_durations[0])
        max_burst_dur = float(burst_durations[0])
        cv_burst_dur = 0.0
        mean_inter_gap = float("nan")
        cv_inter_gap = float("nan")
        burstiness_B = float("nan")
    else:
        mean_burst_dur = max_burst_dur = cv_burst_dur = 0.0
        mean_inter_gap = cv_inter_gap = burstiness_B = float("nan")

    # Gini and entropy of burst centres along the session duration.
    if n_bursts >= 2:
        bins = np.linspace(0.0, duration_s, n_bins + 1)
        hist, _ = np.histogram(burst_centres, bins=bins)
        gini_temporal = _gini(hist.astype(float))
        temporal_entropy = _shannon_entropy(hist, normalise=True)
    elif n_bursts == 1:
        gini_temporal = 1.0   # all mass in one bin -> maximum concentration
        temporal_entropy = 0.0
    else:
        gini_temporal = float("nan")
        temporal_entropy = float("nan")

    # Autocorrelation lag-1 of the sample-level indicator function.
    if len(sample_mask) > 2:
        x = sample_mask.astype(float)
        x0 = x[:-1] - x[:-1].mean()
        x1 = x[1:] - x[1:].mean()
        denom = float(np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum()))
        autocorr_lag1 = float((x0 * x1).sum() / denom) if denom > 0 else 0.0
    else:
        autocorr_lag1 = float("nan")

    return {
        "outlier_coverage_pct": coverage_pct,
        "n_bursts": float(n_bursts),
        "mean_burst_duration_s": mean_burst_dur,
        "max_burst_duration_s": max_burst_dur,
        "cv_burst_duration": cv_burst_dur,
        "mean_inter_burst_gap_s": mean_inter_gap,
        "cv_inter_burst_gap": cv_inter_gap,
        "gini_temporal": gini_temporal,
        "temporal_entropy": temporal_entropy,
        "burstiness_B": burstiness_B,
        "autocorr_lag1_outlier": autocorr_lag1,
    }


TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "outlier_coverage_pct",
    "n_bursts",
    "mean_burst_duration_s",
    "max_burst_duration_s",
    "cv_burst_duration",
    "mean_inter_burst_gap_s",
    "cv_inter_burst_gap",
    "gini_temporal",
    "temporal_entropy",
    "burstiness_B",
    "autocorr_lag1_outlier",
)
