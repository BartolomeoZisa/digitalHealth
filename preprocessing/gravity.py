"""
Gravity removal via ENMO (Euclidean Norm Minus One).

Definition (Van Hees et al.):
    ENMO(t) = max(sqrt(ax(t)^2 + ay(t)^2 + az(t)^2) - 1, 0)

ENMO is a scalar, rotation-invariant proxy for "amount of movement" once the
1 g static gravity component has been subtracted. The clipped variant is the
clinical standard. We also expose a signed variant (no clipping) for
diagnostics where negative residuals carry information (sensor at rest with
gravity magnitude < 1 g due to noise).
"""
from __future__ import annotations

import numpy as np

import config


def enmo(accel: np.ndarray, *, clip_zero: bool | None = None) -> np.ndarray:
    """ENMO on a (T, 3) acceleration array.

    Parameters
    ----------
    accel : np.ndarray, shape (T, 3)
        Columns: (ax, ay, az) in g.
    clip_zero : bool, optional
        If True, clip negative values to 0 (standard ENMO). If False, return
        the signed residual. Default falls back to `config.ENMO_CLIP_ZERO`.

    Returns
    -------
    np.ndarray, shape (T,)
        Scalar acceleration magnitude with gravity removed.
    """
    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError(f"expected (T, 3), got shape {accel.shape}")
    if clip_zero is None:
        clip_zero = config.ENMO_CLIP_ZERO
    if clip_zero:
        return np.maximum(np.sqrt(np.sum(accel ** 2, axis=1)) - 1.0, 0.0)
    return np.sqrt(np.sum(accel ** 2, axis=1)) - 1.0


def enmo_signed(accel: np.ndarray) -> np.ndarray:
    """Convenience wrapper for the unclipped variant."""
    return enmo(accel, clip_zero=False)
