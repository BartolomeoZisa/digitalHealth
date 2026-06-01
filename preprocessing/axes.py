"""
Axis selection and vector magnitude utilities.

The clinical hint flagged the X axis as the noisiest (likely capturing
sensor-mount artefacts). For directional feature engineering we drop X and
keep Y/Z. Note that ENMO (in gravity.py) still uses XYZ as per the standard
formulation, we do not amputate it.
"""
from __future__ import annotations

import numpy as np

import config

_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def select_axes(accel: np.ndarray,
                keep: tuple[str, ...] | None = None) -> np.ndarray:
    """Slice columns of a (T, 3) array along the requested axes."""
    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError(f"expected (T, 3), got shape {accel.shape}")
    if keep is None:
        keep = config.KEEP_AXES_FOR_DIRECTIONAL
    idx = [_AXIS_INDEX[a.lower()] for a in keep]
    return accel[:, idx]


def vector_magnitude(accel: np.ndarray) -> np.ndarray:
    """Euclidean norm across channels, per sample. (T,k) -> (T,)."""
    if accel.ndim == 1:
        return np.abs(accel)
    return np.linalg.norm(accel, axis=1)
