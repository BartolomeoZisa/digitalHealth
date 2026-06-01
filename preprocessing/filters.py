"""
Zero-phase Butterworth filtering via second-order sections.

We use `scipy.signal.butter` to build the SOS representation (numerically
stable for the orders we need) and `scipy.signal.sosfiltfilt` to apply it
forward-backward, zero phase distortion is critical because downstream
analyses look at lag/timing of signals (cross-correlation between hands).

The bandpass operates on the per-axis Y/Z signals after gravity is dealt with
via ENMO. Cutoffs and order come from `config`.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

import config


def butter_bandpass_sos(low_hz: float, high_hz: float, fs: float,
                        order: int | None = None) -> np.ndarray:
    """Return SOS coefficients for a Butterworth bandpass."""
    if order is None:
        order = config.FILTER_ORDER
    nyq = 0.5 * fs
    if not 0.0 < low_hz < high_hz < nyq:
        raise ValueError(
            f"invalid cutoffs: low={low_hz}, high={high_hz}, nyq={nyq}"
        )
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band",
                  output="sos")


def bandpass(signal: np.ndarray, fs: float,
             low_hz: float | None = None,
             high_hz: float | None = None,
             order: int | None = None) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    Accepts 1-D or 2-D (T, k) signals; filters along the time axis (axis 0).
    """
    if low_hz is None:
        low_hz = config.BANDPASS_LOW_HZ
    if high_hz is None:
        high_hz = config.BANDPASS_HIGH_HZ
    sos = butter_bandpass_sos(low_hz, high_hz, fs, order=order)
    # sosfiltfilt requires len(signal) > padlen; for short windows we fall back
    # to less aggressive padding.
    return sosfiltfilt(sos, signal, axis=0)


def butter_highpass_sos(low_hz: float, fs: float,
                        order: int | None = None) -> np.ndarray:
    if order is None:
        order = config.FILTER_ORDER
    nyq = 0.5 * fs
    if not 0.0 < low_hz < nyq:
        raise ValueError(f"invalid cutoff: low={low_hz}, nyq={nyq}")
    return butter(order, low_hz / nyq, btype="high", output="sos")


def highpass(signal: np.ndarray, fs: float, low_hz: float,
             order: int | None = None) -> np.ndarray:
    """Zero-phase Butterworth high-pass (used for diagnostics)."""
    sos = butter_highpass_sos(low_hz, fs, order=order)
    return sosfiltfilt(sos, signal, axis=0)
