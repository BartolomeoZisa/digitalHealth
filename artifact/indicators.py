"""
Per-window indicators for gross artifact detection.

Four scalar quantities are computed per window on the STILL hand of a BBT
session. Each one has a direct clinical interpretation; together they should
flag windows that contain non-MM gross movements (e.g. the patient briefly
raising the resting arm) so they can be excluded from the intra-patient
"stillness" baseline of Sprint 3 onwards.

All functions are pure and operate on a single window (numpy array). Vectorised
batch evaluation is offered for convenience.
"""
from __future__ import annotations

import numpy as np

from preprocessing.derivatives import jerk_magnitude


# -----------------------------------------------------------------------------
# Per-window primitives
# -----------------------------------------------------------------------------
def enmo_peak(enmo_window: np.ndarray) -> float:
    """Maximum ENMO sample within the window (g, clipped already)."""
    return float(np.max(enmo_window))


def jerk_peak(yz_bp_window: np.ndarray, fs: float) -> float:
    """Maximum |jerk| magnitude (g / s) on the band-passed YZ window.

    Parameters
    ----------
    yz_bp_window : np.ndarray, shape (T, 2)
        Band-passed Y, Z samples in g.
    fs : float
        Sample rate.
    """
    j = jerk_magnitude(yz_bp_window, fs)
    return float(np.max(j))


def yz_bp_rms(yz_bp_window: np.ndarray) -> float:
    """RMS amplitude of the band-passed YZ window (g)."""
    return float(np.sqrt(np.mean(yz_bp_window ** 2)))


def energy_ratio(still_window: np.ndarray, active_window: np.ndarray,
                 eps: float = 1e-6) -> float:
    """Energy ratio still / active over the same time window.

    Interpretation: in a non-artifact moment the active hand carries more
    energy than the still one, so this ratio is well below 1. Values >> 1
    indicate that the "still" hand is moving more than the "active" one ,
    a strong artifact signature (e.g. patient raises the resting arm while
    pausing the BBT execution).

    Energy is the sum of squared samples; on (T, k) windows it is summed
    across channels.
    """
    e_s = float(np.sum(still_window ** 2))
    e_a = float(np.sum(active_window ** 2))
    return e_s / (e_a + eps)


# -----------------------------------------------------------------------------
# Vectorised batch APIs
# -----------------------------------------------------------------------------
def batch_enmo_peak(enmo_windows: np.ndarray) -> np.ndarray:
    """`enmo_windows`: (n, T) -> (n,)."""
    return enmo_windows.max(axis=1)


def batch_yz_bp_rms(yz_bp_windows: np.ndarray) -> np.ndarray:
    """`yz_bp_windows`: (n, T, 2) -> (n,)."""
    return np.sqrt(np.mean(yz_bp_windows ** 2, axis=(1, 2)))


def batch_jerk_peak(yz_bp_windows: np.ndarray, fs: float) -> np.ndarray:
    """`yz_bp_windows`: (n, T, 2) -> (n,). Computes peak |jerk| per window."""
    # Central differences along time axis of each window.
    dt = 1.0 / fs
    j = np.gradient(yz_bp_windows, dt, axis=1)  # (n, T, 2)
    mag = np.linalg.norm(j, axis=2)             # (n, T)
    return mag.max(axis=1)


def batch_energy_ratio(still_w: np.ndarray, active_w: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """Per-window energy ratio for batched windows of matching shape."""
    e_s = (still_w ** 2).sum(axis=tuple(range(1, still_w.ndim)))
    e_a = (active_w ** 2).sum(axis=tuple(range(1, active_w.ndim)))
    return e_s / (e_a + eps)


INDICATOR_NAMES: tuple[str, ...] = (
    "enmo_peak", "jerk_peak", "yz_bp_rms", "energy_ratio",
)
