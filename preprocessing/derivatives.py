"""
Time derivatives of the acceleration signal.

Jerk is the time derivative of acceleration; its magnitude is a sensitive
indicator of abrupt movements (peaks during sudden onset, very low during
smooth stillness). We use central differences via `np.gradient` so the output
has the same length as the input, important for time-aligned plotting.
"""
from __future__ import annotations

import numpy as np


def jerk(accel: np.ndarray, fs: float) -> np.ndarray:
    """Central-difference time derivative of an acceleration signal.

    Parameters
    ----------
    accel : np.ndarray, shape (T,) or (T, k)
        Acceleration samples in g.
    fs : float
        Sampling rate, Hz.

    Returns
    -------
    np.ndarray, same shape as `accel`
        Jerk in g / s.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    dt = 1.0 / fs
    return np.gradient(accel, dt, axis=0)


def jerk_magnitude(accel: np.ndarray, fs: float) -> np.ndarray:
    """Scalar jerk magnitude per sample.

    For a 1-D input returns `|d(accel)/dt|`. For a 2-D (T, k) input returns
    the Euclidean norm across channels of the per-channel jerk.
    """
    j = jerk(accel, fs)
    if j.ndim == 1:
        return np.abs(j)
    return np.linalg.norm(j, axis=1)
