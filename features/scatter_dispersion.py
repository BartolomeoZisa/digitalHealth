"""
Patient-level scatter-dispersion features (Sprint 5.5).

Clinician observation: plotting the per-window points
`(asymmetry_index, xcorr_max)` of one session reveals different cluster
spreads between dom-active and ndom-active sessions for UCP vs TD patients:

  * UCP -> ndom-active cluster is more dispersed than dom-active.
  * TD  -> both clusters are tight; if anything dom >= ndom.

We quantify the cluster spread two complementary ways and the inter-session
ratio:

  * `mean_pairwise_dist`    , symmetric, captures global spread.
  * `mean_dist_to_centroid` , more stable on small samples (N < 30).
  * `dispersion_ratio`       = disp(ndom) / disp(dom).

Inputs are the per-window DataFrames already produced by
`explainable.features.extractor` for each session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


# -----------------------------------------------------------------------------
# Pure dispersion primitives
# -----------------------------------------------------------------------------
def mean_pairwise_dist(points: np.ndarray) -> float:
    """Mean pairwise Euclidean distance of a (N, D) point cloud.

    Returns 0.0 when fewer than 2 valid points are available.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return 0.0
    return float(pdist(points, metric="euclidean").mean())


def mean_dist_to_centroid(points: np.ndarray) -> float:
    """Mean Euclidean distance from each point to the cluster centroid."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return 0.0
    centroid = points.mean(axis=0)
    return float(np.linalg.norm(points - centroid, axis=1).mean())


# -----------------------------------------------------------------------------
# Per-session DataFrame -> dispersion scalars
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionDispersion:
    """Spread measurements of one session's valid-window cluster."""
    n_points: int
    disp_pairwise: float
    disp_centroid: float


def _valid_points(df: pd.DataFrame,
                  x: str, y: str,
                  require_clean: bool) -> np.ndarray:
    """Return (N, 2) array of (x, y) values from clean rows."""
    if x not in df.columns or y not in df.columns:
        return np.zeros((0, 2), dtype=float)
    mask = pd.Series(True, index=df.index)
    if require_clean:
        if "is_boundary" in df.columns:
            mask &= ~df["is_boundary"].astype(bool)
        if "is_artifact" in df.columns:
            mask &= ~df["is_artifact"].astype(bool)
    pts = df.loc[mask, [x, y]].to_numpy(dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    # Drop any rows with NaN/Inf to keep distance computations finite.
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def compute_session_dispersion(df_windows: pd.DataFrame,
                               x: str = "asymmetry_index",
                               y: str = "xcorr_max",
                               require_clean: bool = True) -> SessionDispersion:
    """Compute both dispersion metrics on one session's DataFrame.

    `require_clean=True` keeps only rows with `is_boundary=False` and
    `is_artifact=False`, matching the clinician's visual inspection of the
    Sprint-4/5 scatter plots.
    """
    pts = _valid_points(df_windows, x, y, require_clean=require_clean)
    return SessionDispersion(
        n_points=int(len(pts)),
        disp_pairwise=mean_pairwise_dist(pts),
        disp_centroid=mean_dist_to_centroid(pts),
    )


def safe_ratio(num: float, den: float, eps: float = 1e-9) -> float:
    """Numerator over denominator, returning NaN if denominator is ~zero."""
    if abs(den) < eps:
        return float("nan")
    return num / den
