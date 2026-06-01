"""
Median-of-three ensemble over the per-window anomaly scores.

The detectors disagree most on the tails: combining them via the **median**
of the three normalised scores yields a value that is high only when at
least two of the three find the window suspicious, and low whenever two of
the three call it calm. This is the most conservative ensemble for our
intra-patient setting where false positives are visually expensive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from detectors.base import Contribution


SCORE_COLS: tuple[str, ...] = ("score_quantile", "score_iforest", "score_pca")
ENSEMBLE_COL: str = "score_median"


def median_ensemble(score_quantile: np.ndarray,
                    score_iforest: np.ndarray,
                    score_pca: np.ndarray) -> np.ndarray:
    """Median across the 3 per-window scores."""
    stacked = np.vstack([score_quantile, score_iforest, score_pca])
    return np.median(stacked, axis=0)


def attribution_to_str(contribs: list[Contribution]) -> str:
    """Render a (name, value) list as 'feature(value); feature(value); ...'."""
    return "; ".join(f"{name}({val:+.2f})" for name, val in contribs)


def compute_is_mm_candidate(
    *,
    score_median: np.ndarray,
    asymmetry_index: np.ndarray | None,
    xcorr_max: np.ndarray | None,
    is_artifact: np.ndarray,
    is_boundary: np.ndarray,
    score_cutoff: float | None = None,
    asym_cutoff: float | None = None,
    xcorr_cutoff: float | None = None,
) -> np.ndarray:
    """Composite MM rule: high score AND bilateral signature AND not noise.

    Returns a boolean array. Falls back to score-only when asymmetry / xcorr
    columns are absent (e.g. a feature set that does not include them).
    """
    if score_cutoff is None:
        score_cutoff = config.ENSEMBLE_THRESHOLD
    if asym_cutoff is None:
        asym_cutoff = config.ASYM_MM_CUTOFF
    if xcorr_cutoff is None:
        xcorr_cutoff = config.XCORR_MM_CUTOFF

    mask = (score_median >= score_cutoff) & (~is_artifact) & (~is_boundary)
    if asymmetry_index is not None:
        mask &= asymmetry_index <= asym_cutoff
    if xcorr_max is not None:
        mask &= xcorr_max >= xcorr_cutoff
    return mask


def build_window_dataframe(
    *,
    t_start_s: np.ndarray,
    is_outlier: np.ndarray,
    is_boundary: np.ndarray,
    features_df: pd.DataFrame,
    score_quantile: np.ndarray,
    score_iforest: np.ndarray,
    score_pca: np.ndarray,
    attribution_quantile: list[list[Contribution]],
) -> pd.DataFrame:
    """Pack everything a downstream UI/CSV needs into one DataFrame."""
    score_median = median_ensemble(score_quantile, score_iforest, score_pca)

    asym = (features_df["asymmetry_index"].to_numpy()
            if "asymmetry_index" in features_df.columns else None)
    xcorr = (features_df["xcorr_max"].to_numpy()
             if "xcorr_max" in features_df.columns else None)
    is_mm = compute_is_mm_candidate(
        score_median=score_median,
        asymmetry_index=asym,
        xcorr_max=xcorr,
        is_artifact=is_outlier.astype(bool),
        is_boundary=is_boundary.astype(bool),
    )

    df = pd.DataFrame({
        "t_start_s": t_start_s,
        "is_artifact": is_outlier.astype(bool),
        "is_boundary": is_boundary.astype(bool),
        "score_quantile": score_quantile,
        "score_iforest": score_iforest,
        "score_pca": score_pca,
        "score_median": score_median,
        "is_mm_candidate": is_mm,
        "attribution_top3": [attribution_to_str(c) for c in attribution_quantile],
    })
    # Append all raw features so the row is self-explanatory in a CSV.
    for col in features_df.columns:
        if col not in df.columns:
            df[col] = features_df[col].to_numpy()
    return df
