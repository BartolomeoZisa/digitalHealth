"""
Family 2, Frequency-domain features per window.

We use a one-sided FFT on the (already detrended / band-passed) signal so the
spectrum is well defined within [0, fs/2]. The band-power features sum the
PSD over clinically-motivated frequency bands defined in `config.FREQ_BANDS_HZ`.

Inputs: 1-D windows (typically `enmo_bp` or vector magnitude of `yz_bp`).
Outputs: scalar per feature per window.
"""
from __future__ import annotations

import numpy as np

import config


def _psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd) of a real signal via one-sided FFT.

    PSD here is `|X(f)|^2 / (fs * N)`, proportional to a periodogram. We do
    not need absolute calibration because all features are either ratios or
    summaries that are scale-equivariant.
    """
    n = len(x)
    if n < 4:
        return np.zeros(0), np.zeros(0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x - x.mean())
    psd = (np.abs(spec) ** 2) / (fs * n)
    return freqs, psd


def spectral_centroid(x_window: np.ndarray, fs: float) -> float:
    """Centre of mass of the PSD (Hz)."""
    freqs, psd = _psd(x_window, fs)
    if psd.sum() <= 0:
        return float("nan")
    return float(np.sum(freqs * psd) / np.sum(psd))


def dominant_frequency(x_window: np.ndarray, fs: float,
                       f_min: float | None = None,
                       f_max: float | None = None) -> float:
    """Frequency of the largest PSD bin within [f_min, f_max].

    Defaults to the full band-pass range from `config`.
    """
    if f_min is None:
        f_min = config.BANDPASS_LOW_HZ
    if f_max is None:
        f_max = config.BANDPASS_HIGH_HZ
    freqs, psd = _psd(x_window, fs)
    if len(freqs) == 0:
        return float("nan")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return float("nan")
    sub_f = freqs[mask]
    sub_p = psd[mask]
    return float(sub_f[int(np.argmax(sub_p))])


def band_power(x_window: np.ndarray, fs: float,
               f_low: float, f_high: float) -> float:
    """Integral of the PSD between `f_low` and `f_high`."""
    freqs, psd = _psd(x_window, fs)
    if len(freqs) == 0:
        return 0.0
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def spectral_entropy(x_window: np.ndarray, fs: float,
                     f_min: float | None = None,
                     f_max: float | None = None,
                     normalise: bool = True) -> float:
    """Shannon entropy of the normalised PSD inside the analysis band.

    Returned in nats; with `normalise=True` divided by `log(N_bins)` so the
    feature lives in `[0, 1]` (1 = flat spectrum, 0 = pure tone).
    """
    if f_min is None:
        f_min = config.BANDPASS_LOW_HZ
    if f_max is None:
        f_max = config.BANDPASS_HIGH_HZ
    freqs, psd = _psd(x_window, fs)
    if len(freqs) == 0:
        return float("nan")
    mask = (freqs >= f_min) & (freqs <= f_max)
    psd_sub = psd[mask]
    total = psd_sub.sum()
    if total <= 0:
        return 0.0
    p = psd_sub / total
    # Drop zeros to avoid log(0).
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    if normalise and len(p) > 1:
        h = h / np.log(len(p))
    return h


# Names exported as the canonical Family 2 set.
FREQ_FEATURE_NAMES: tuple[str, ...] = (
    "spectral_centroid",
    "dominant_freq",
    "band_power_slow",     # 0.5–2 Hz
    "band_power_medium",   # 2–6 Hz
    "band_power_fast",     # 6–15 Hz
    "spectral_entropy",
)
