"""
Windowing utilities.

Provides project-wide defaults from `config`, a multi-axis variant for (T, k)
signals, and a multi-scale helper that returns windows at several durations in
one call. This is the entry point used downstream by the feature pipeline
(Sprint 3) and the artifact filter (Sprint 2).
"""
from __future__ import annotations

import numpy as np

import config


def sliding_windows_1d(signal: np.ndarray, fs: float,
                       window_s: float | None = None,
                       overlap: float | None = None) -> np.ndarray:
    """Reusable 1-D windowing with project-wide defaults."""
    if signal.ndim != 1:
        raise ValueError(f"expected 1-D signal, got shape {signal.shape}")
    if window_s is None:
        window_s = config.DEFAULT_WINDOW_S
    if overlap is None:
        overlap = config.DEFAULT_OVERLAP
    win_len = int(window_s * fs)
    step = max(int(win_len * (1.0 - overlap)), 1)
    starts = range(0, len(signal) - win_len + 1, step)
    return np.stack([signal[s:s + win_len] for s in starts])


def sliding_windows_multi(signal: np.ndarray, fs: float,
                          window_s: float | None = None,
                          overlap: float | None = None) -> np.ndarray:
    """Windowing for (T, k) signals.

    Returns
    -------
    np.ndarray, shape (n_windows, win_len, k)
    """
    if signal.ndim != 2:
        raise ValueError(f"expected 2-D (T, k), got shape {signal.shape}")
    if window_s is None:
        window_s = config.DEFAULT_WINDOW_S
    if overlap is None:
        overlap = config.DEFAULT_OVERLAP
    win_len = int(window_s * fs)
    step = max(int(win_len * (1.0 - overlap)), 1)
    starts = range(0, len(signal) - win_len + 1, step)
    return np.stack([signal[s:s + win_len] for s in starts])


def window_starts(n_samples: int, fs: float,
                  window_s: float, overlap: float) -> np.ndarray:
    """Sample indices at which each window starts (useful for plotting)."""
    win_len = int(window_s * fs)
    step = max(int(win_len * (1.0 - overlap)), 1)
    return np.arange(0, n_samples - win_len + 1, step)


def multi_scale_windows(signal: np.ndarray, fs: float,
                        scales_s: tuple[float, ...] | None = None,
                        overlap: float | None = None
                        ) -> dict[float, np.ndarray]:
    """Compute windows at several scales in one pass.

    Returns
    -------
    dict[float, np.ndarray]
        Maps window duration (seconds) -> windowed array. The array is 2-D
        (n, win_len) for 1-D inputs and 3-D (n, win_len, k) for 2-D inputs.
    """
    if scales_s is None:
        scales_s = config.WINDOW_SIZES_S
    if overlap is None:
        overlap = config.DEFAULT_OVERLAP
    out: dict[float, np.ndarray] = {}
    for s in scales_s:
        if signal.ndim == 1:
            out[float(s)] = sliding_windows_1d(signal, fs, window_s=s,
                                               overlap=overlap)
        else:
            out[float(s)] = sliding_windows_multi(signal, fs, window_s=s,
                                                  overlap=overlap)
    return out
