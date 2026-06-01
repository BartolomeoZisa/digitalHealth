"""
Feature registry, single source of truth for per-window feature definitions.

Each entry maps a feature name to a callable that takes a `WindowInputs`
dataclass and returns a scalar. The orchestrator (`extractor.py`) iterates
the registry to build a (n_windows × n_features) DataFrame.

Family 4 (temporal distribution) is **per session**, not per window, so it
lives outside this registry and is invoked separately by the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import config
from features import bilateral, frequency, time_domain


@dataclass(frozen=True)
class WindowInputs:
    """Bundle of aligned signals for a single window pair (still + active).

    Every signal has the same length on the time axis.

    enmo_still / enmo_active : (T,) ENMO clipped, gravity-free
    yz_bp_still / yz_bp_active : (T, 2) band-passed Y, Z
    jerk_mag_still / jerk_mag_active : (T,) |jerk| on band-passed YZ
    vec_mag_still : (T,) raw vector magnitude (includes gravity)
    fs : sampling rate
    """
    enmo_still: np.ndarray
    enmo_active: np.ndarray
    yz_bp_still: np.ndarray
    yz_bp_active: np.ndarray
    jerk_mag_still: np.ndarray
    jerk_mag_active: np.ndarray
    vec_mag_still: np.ndarray
    fs: float


# Adapter helpers, wrap pure feature functions to the WindowInputs signature.
def _enmo_mean(w: WindowInputs) -> float:
    return time_domain.enmo_mean(w.enmo_still)


def _enmo_peak(w: WindowInputs) -> float:
    return time_domain.enmo_peak(w.enmo_still)


def _yz_bp_rms(w: WindowInputs) -> float:
    return time_domain.rms(w.yz_bp_still)


def _yz_bp_peak(w: WindowInputs) -> float:
    return time_domain.peak(w.yz_bp_still)


def _jerk_mag_rms(w: WindowInputs) -> float:
    return time_domain.jerk_rms(w.jerk_mag_still)


def _vec_mag_integral(w: WindowInputs) -> float:
    return time_domain.vec_mag_integral(w.vec_mag_still, w.fs)


def _yz_bp_zcr(w: WindowInputs) -> float:
    return time_domain.zero_crossing_rate(w.yz_bp_still)


def _spectral_centroid(w: WindowInputs) -> float:
    return frequency.spectral_centroid(w.enmo_still, w.fs)


def _dominant_freq(w: WindowInputs) -> float:
    return frequency.dominant_frequency(w.enmo_still, w.fs)


def _band_power_slow(w: WindowInputs) -> float:
    f_lo, f_hi = config.FREQ_BANDS_HZ["slow"]
    return frequency.band_power(w.enmo_still, w.fs, f_lo, f_hi)


def _band_power_medium(w: WindowInputs) -> float:
    f_lo, f_hi = config.FREQ_BANDS_HZ["medium"]
    return frequency.band_power(w.enmo_still, w.fs, f_lo, f_hi)


def _band_power_fast(w: WindowInputs) -> float:
    f_lo, f_hi = config.FREQ_BANDS_HZ["fast"]
    return frequency.band_power(w.enmo_still, w.fs, f_lo, f_hi)


def _spectral_entropy(w: WindowInputs) -> float:
    return frequency.spectral_entropy(w.enmo_still, w.fs)


# Bilateral features return (corr, lag); split them into two registry entries.
def _xcorr_max(w: WindowInputs) -> float:
    corr, _ = bilateral.xcorr_max_dom_ndom(w.enmo_active, w.enmo_still, w.fs)
    return corr


def _xcorr_lag_ms(w: WindowInputs) -> float:
    _, lag_ms = bilateral.xcorr_max_dom_ndom(w.enmo_active, w.enmo_still, w.fs)
    return lag_ms


def _asymmetry_index(w: WindowInputs) -> float:
    return bilateral.signed_asymmetry_index(w.enmo_active, w.enmo_still)


def _bilateral_jerk_corr(w: WindowInputs) -> float:
    return bilateral.bilateral_jerk_corr(w.jerk_mag_active, w.jerk_mag_still)


# Master registry: feature name -> (family, callable).
REGISTRY: dict[str, tuple[str, Callable[[WindowInputs], float]]] = {
    # Family 1, Activity (time domain)
    "enmo_mean":          ("activity", _enmo_mean),
    "enmo_peak":          ("activity", _enmo_peak),
    "yz_bp_rms":          ("activity", _yz_bp_rms),
    "yz_bp_peak":         ("activity", _yz_bp_peak),
    "jerk_mag_rms":       ("activity", _jerk_mag_rms),
    "vec_mag_integral":   ("activity", _vec_mag_integral),
    "yz_bp_zcr":          ("activity", _yz_bp_zcr),
    # Family 2, Spectral
    "spectral_centroid":  ("spectral", _spectral_centroid),
    "dominant_freq":      ("spectral", _dominant_freq),
    "band_power_slow":    ("spectral", _band_power_slow),
    "band_power_medium":  ("spectral", _band_power_medium),
    "band_power_fast":    ("spectral", _band_power_fast),
    "spectral_entropy":   ("spectral", _spectral_entropy),
    # Family 3, Bilateral mirror score
    "xcorr_max":          ("bilateral", _xcorr_max),
    "xcorr_lag_ms":       ("bilateral", _xcorr_lag_ms),
    "asymmetry_index":    ("bilateral", _asymmetry_index),
    "bilateral_jerk_corr": ("bilateral", _bilateral_jerk_corr),
}


def enabled_features() -> list[str]:
    """Return the active feature names, honouring `FEATURE_REGISTRY_ENABLED`."""
    if not config.FEATURE_REGISTRY_ENABLED:
        return list(REGISTRY.keys())
    return [name for name in REGISTRY if name in config.FEATURE_REGISTRY_ENABLED]
