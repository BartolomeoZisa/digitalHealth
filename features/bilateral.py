"""
Family 3, Bilateral mirror-score features per window.

Mirror movements manifest as the still hand replicating the active hand's
gesture with a small temporal lag. The features in this module therefore
score the *relationship* between dom and ndom rather than each hand
individually.

The cross-correlation primitive z-scores both inputs, removing amplitude bias,
and scans an integer-sample lag window.

Inputs: paired 1-D windows `active_window`, `still_window` (typically the
vector magnitude of `yz_bp`, but any aligned signals work).

Outputs: scalar per feature per window.
"""
from __future__ import annotations

import numpy as np

import config

EPS: float = 1e-12


def _max_lag_samples(fs: float) -> int:
    """Translate `config.XCORR_MAX_LAG_MS` to integer samples."""
    return max(1, int(round(config.XCORR_MAX_LAG_MS * 1e-3 * fs)))


def _max_abs_xcorr(a: np.ndarray, m: np.ndarray,
                   max_lag: int) -> tuple[float, int]:
    """Maximum absolute normalized cross-correlation over a lag range."""
    n = min(len(a), len(m))
    if n < 1:
        return 0.0, 0
    a_z = (a[:n] - a[:n].mean()) / (a[:n].std() + EPS)
    m_z = (m[:n] - m[:n].mean()) / (m[:n].std() + EPS)

    best_c, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(a_z[lag:], m_z[:n - lag]) / n) if (n - lag) > 0 else 0.0
        else:
            c = float(np.dot(a_z[:n + lag], m_z[-lag:]) / n) if (n + lag) > 0 else 0.0
        if abs(c) > abs(best_c):
            best_c, best_lag = c, lag
    return abs(best_c), best_lag


def xcorr_max_dom_ndom(active: np.ndarray, still: np.ndarray,
                       fs: float) -> tuple[float, float]:
    """Maximum |normalised cross-correlation| between active and still hand.

    Returns
    -------
    (max_corr, lag_ms):
        `max_corr` ∈ [0, 1] from the z-scored cross-correlation.
        `lag_ms` is the lag (in milliseconds) at which the max occurs;
        positive means `still` lags behind `active`.
    """
    n = min(len(active), len(still))
    if n < 4:
        return 0.0, float("nan")
    max_lag = _max_lag_samples(fs)
    corr, lag = _max_abs_xcorr(active[:n], still[:n], max_lag=max_lag)
    return float(corr), float(lag * 1000.0 / fs)


def signed_asymmetry_index(active_window: np.ndarray,
                           still_window: np.ndarray) -> float:
    """Signed RMS asymmetry index based on signal energy.

    `(RMS_active - RMS_still) / (RMS_active + RMS_still + eps)`
    in [-1, +1]. Positive ⇒ active hand dominates (normal stillness).
    Around zero ⇒ both hands carry similar energy (suspect MM).
    Negative ⇒ still hand dominates (likely artifact such as arm raise).
    """
    a = float(np.sqrt(np.mean(active_window ** 2)))
    s = float(np.sqrt(np.mean(still_window ** 2)))
    return float((a - s) / (a + s + EPS))


def bilateral_jerk_corr(jerk_mag_active: np.ndarray,
                        jerk_mag_still: np.ndarray) -> float:
    """Pearson correlation between active and still jerk-magnitude traces.

    Captures onset-time alignment: when both hands accelerate together the
    correlation is high regardless of amplitude. Returns 0.0 if either trace
    has zero variance.
    """
    n = min(len(jerk_mag_active), len(jerk_mag_still))
    if n < 4:
        return 0.0
    a = jerk_mag_active[:n]
    s = jerk_mag_still[:n]
    if a.std() < EPS or s.std() < EPS:
        return 0.0
    c = float(np.corrcoef(a, s)[0, 1])
    return 0.0 if np.isnan(c) else c


# Names exported as the canonical Family 3 set.
BILATERAL_FEATURE_NAMES: tuple[str, ...] = (
    "xcorr_max",          # max |cross-corr|
    "xcorr_lag_ms",       # lag at max (ms)
    "asymmetry_index",    # signed
    "bilateral_jerk_corr",
)
