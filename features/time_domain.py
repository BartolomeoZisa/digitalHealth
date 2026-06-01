"""
Family 1, Time-domain (activity) features per window.

These are the most directly interpretable quantities for clinicians: they
answer "how much is the still hand moving in this window?". All functions are
pure and operate on a single window of the preprocessed still-hand signal
(numpy arrays from sprint 1's `ProcessedHand`).

Inputs:
    - 1-D windows for scalar signals (e.g. `enmo`, `jerk_mag`, `vec_mag`)
    - 2-D (T, k) windows for multi-axis signals (e.g. `yz_bp`)

Outputs: a single scalar per feature per window.
"""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    """Root mean square of a 1-D or 2-D window (g)."""
    return float(np.sqrt(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    """Max absolute value (g). Robust to sign for multi-axis arrays."""
    return float(np.max(np.abs(x)))


def jerk_rms(jerk_mag_window: np.ndarray) -> float:
    """RMS of pre-computed jerk magnitude (g/s)."""
    return float(np.sqrt(np.mean(jerk_mag_window ** 2)))


def vec_mag_integral(vec_mag_window: np.ndarray, fs: float) -> float:
    """Time integral of vector magnitude across the window (g·s).

    Approximation: trapezoidal rule with `dt = 1/fs`. Captures "total movement"
    over the window irrespective of frequency content.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    return float(np.trapezoid(vec_mag_window, dx=1.0 / fs))


def zero_crossing_rate(x_window: np.ndarray) -> float:
    """Zero-crossing rate of a 1-D (preferably zero-mean / band-passed) window.

    Returned as crossings per sample (∈ [0, 1]). For multi-axis input the rate
    is averaged over channels.
    """
    if x_window.ndim == 1:
        x = x_window
        signs = np.sign(x)
        # Drop exact zeros so they don't double-count.
        signs = signs[signs != 0]
        if len(signs) < 2:
            return 0.0
        return float(np.mean(np.diff(signs) != 0))
    # Multi-axis: mean ZCR over channels.
    rates = [zero_crossing_rate(x_window[:, j]) for j in range(x_window.shape[1])]
    return float(np.mean(rates))


def enmo_mean(enmo_window: np.ndarray) -> float:
    return float(np.mean(enmo_window))


def enmo_peak(enmo_window: np.ndarray) -> float:
    return float(np.max(enmo_window))


# Names exported as the canonical Family 1 set. Used by the registry.
TIME_DOMAIN_FEATURE_NAMES: tuple[str, ...] = (
    "enmo_mean",
    "enmo_peak",
    "yz_bp_rms",
    "yz_bp_peak",
    "jerk_mag_rms",
    "vec_mag_integral",
    "yz_bp_zcr",
)
