"""
Feature extractor, orchestrates Families 1–3 (per window) and Family 4
(per session) on a `ProcessedSession` plus its `PatientArtifactReport`.

Per-window features are computed at a single reference window scale (default
1 s with 50% overlap, mirroring the Sprint 2 reference). The artifact mask
from Sprint 2 is propagated as a `is_outlier` boolean column.

Per-session features (Family 4) operate on the OR-merged sample-level
outlier mask, so they exploit information from all the scales used in
Sprint 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from artifact.filter import (
    PatientArtifactReport,
    SessionArtifactResult,
)
from features.registry import REGISTRY, WindowInputs, enabled_features
from features.temporal import (
    TEMPORAL_FEATURE_NAMES,
    temporal_features_from_sample_mask,
    window_mask_to_sample_mask,
)
from preprocessing.pipeline import ProcessedHand, ProcessedSession
from preprocessing.windowing import (
    sliding_windows_1d,
    sliding_windows_multi,
    window_starts,
)


REFERENCE_SCALE_S: float = 1.0   # window length used for per-window features
REFERENCE_OVERLAP: float = 0.50  # consistent with Sprint 2 chosen config


@dataclass(frozen=True)
class SessionFeatures:
    """Bundle of feature outputs for one BBT session of one patient."""
    patient_id: str
    group: str
    session_label: str
    fs: float
    scale_s: float
    overlap: float
    window_starts_idx: np.ndarray            # (n_windows,)
    df_windows: pd.DataFrame                 # rows: windows, cols: features + is_outlier
    temporal: dict[str, float]               # Family 4 (per session)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_window_inputs(still: ProcessedHand, active: ProcessedHand,
                         scale_s: float, overlap: float
                         ) -> tuple[np.ndarray, list[WindowInputs]]:
    """Window every signal needed by the registry and return WindowInputs."""
    fs = still.fs
    starts = window_starts(len(still.t), fs, scale_s, overlap)

    enmo_w_still = sliding_windows_1d(still.enmo, fs, scale_s, overlap)
    enmo_w_active = sliding_windows_1d(active.enmo, fs, scale_s, overlap)
    yz_bp_w_still = sliding_windows_multi(still.yz_bp, fs, scale_s, overlap)
    yz_bp_w_active = sliding_windows_multi(active.yz_bp, fs, scale_s, overlap)
    jerk_w_still = sliding_windows_1d(still.jerk_mag, fs, scale_s, overlap)
    jerk_w_active = sliding_windows_1d(active.jerk_mag, fs, scale_s, overlap)
    vmag_w_still = sliding_windows_1d(still.vec_mag, fs, scale_s, overlap)

    n = len(starts)
    inputs: list[WindowInputs] = [
        WindowInputs(
            enmo_still=enmo_w_still[i],
            enmo_active=enmo_w_active[i],
            yz_bp_still=yz_bp_w_still[i],
            yz_bp_active=yz_bp_w_active[i],
            jerk_mag_still=jerk_w_still[i],
            jerk_mag_active=jerk_w_active[i],
            vec_mag_still=vmag_w_still[i],
            fs=fs,
        )
        for i in range(n)
    ]
    return starts, inputs


def _compute_window_features(inputs: list[WindowInputs]) -> pd.DataFrame:
    """Apply every enabled feature in the registry to every window."""
    names = enabled_features()
    data: dict[str, list[float]] = {name: [] for name in names}
    for w in inputs:
        for name in names:
            _, fn = REGISTRY[name]
            data[name].append(fn(w))
    return pd.DataFrame(data)


def _is_outlier_for_reference_scale(sess_res: SessionArtifactResult,
                                    scale_s: float,
                                    n_windows: int) -> np.ndarray:
    """Take the window-level outlier mask of the chosen reference scale.

    Sprint 2 may not have used the same overlap; if the n_windows here does
    not match the artifact report we fall back to all-False (no flag).
    """
    if scale_s not in sess_res.per_scale:
        return np.zeros(n_windows, dtype=bool)
    sr = sess_res.per_scale[scale_s]
    mask = sr.mask_outlier_any
    if len(mask) != n_windows:
        return np.zeros(n_windows, dtype=bool)
    return mask


def _build_sample_level_outlier_mask(sess_res: SessionArtifactResult,
                                     n_samples: int,
                                     fs: float) -> np.ndarray:
    """OR-merge every scale's outlier windows onto a sample-level boolean array."""
    sm = np.zeros(n_samples, dtype=bool)
    for scale_s, sr in sess_res.per_scale.items():
        win_len = int(scale_s * fs)
        if len(sr.mask_outlier_any) == 0:
            continue
        out_idx = np.where(sr.mask_outlier_any)[0]
        for i in out_idx:
            start = int(sr.starts_idx[i])
            end = min(start + win_len, n_samples)
            sm[start:end] = True
    return sm


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def extract_session_features(
    proc_session: ProcessedSession,
    sess_res: SessionArtifactResult,
    *,
    patient_id: str,
    group: str,
    scale_s: float = REFERENCE_SCALE_S,
    overlap: float = REFERENCE_OVERLAP,
) -> SessionFeatures:
    """Compute Family 1–3 (per window) and Family 4 (per session) features."""
    still = proc_session.still
    active = proc_session.active

    starts, inputs = _build_window_inputs(still, active, scale_s, overlap)
    df = _compute_window_features(inputs)

    # Tag each window with the Sprint-2 outlier flag at the same scale.
    is_outlier = _is_outlier_for_reference_scale(sess_res, scale_s, len(df))
    df.insert(0, "is_outlier", is_outlier)
    t_start_s = starts / still.fs
    df.insert(0, "t_start_s", t_start_s)

    # Sprint 5, boundary tag. Windows whose start OR end falls in the first
    # / last `BOUNDARY_TRIM_S` seconds are flagged so the composite MM rule
    # can exclude them (the BBT setup/cleanup transients and filter edge
    # effects make those windows unreliable).
    duration_s = len(still.t) / still.fs
    win_end_s = t_start_s + scale_s
    is_boundary = (t_start_s < config.BOUNDARY_TRIM_S) | (
        win_end_s > duration_s - config.BOUNDARY_TRIM_S
    )
    df.insert(1, "is_boundary", is_boundary)

    # Per-session temporal-distribution features.
    n_samples = len(still.t)
    sample_mask = _build_sample_level_outlier_mask(sess_res, n_samples, still.fs)
    temporal = temporal_features_from_sample_mask(
        sample_mask, fs=still.fs, duration_s=n_samples / still.fs,
    )

    return SessionFeatures(
        patient_id=patient_id,
        group=group,
        session_label=proc_session.session_label,
        fs=still.fs,
        scale_s=scale_s,
        overlap=overlap,
        window_starts_idx=starts,
        df_windows=df,
        temporal=temporal,
    )


def extract_patient_features(
    proc: dict[str, ProcessedSession],
    report: PatientArtifactReport,
    *,
    patient_id: str,
    group: str,
    scale_s: float = REFERENCE_SCALE_S,
    overlap: float = REFERENCE_OVERLAP,
) -> dict[str, SessionFeatures]:
    """Run the extractor for every session of one patient."""
    out: dict[str, SessionFeatures] = {}
    for sess_label, sess in proc.items():
        sess_res = report.per_session[sess_label]
        out[sess_label] = extract_session_features(
            sess, sess_res,
            patient_id=patient_id, group=group,
            scale_s=scale_s, overlap=overlap,
        )
    return out


def temporal_to_dataframe(features_by_patient: list[SessionFeatures]) -> pd.DataFrame:
    """Stack Family-4 outputs of many sessions into a flat DataFrame."""
    rows = []
    for sf in features_by_patient:
        row = {
            "patient_id": sf.patient_id,
            "group": sf.group,
            "session": sf.session_label,
        }
        row.update({name: sf.temporal[name] for name in TEMPORAL_FEATURE_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)
