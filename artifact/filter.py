"""
Multi-scale, intra-patient artifact filter.

For each (patient, session) we compute the four indicators on overlapping
windows at several scales (e.g. 0.5, 1, 2 seconds), flag outliers
independently at each scale, and combine them with logical OR:

    window @ scale s is outlier
        iff any indicator passes its iterative robust threshold at scale s.

A window in the **time** domain is then declared outlier if any window at
any scale that covers its centre is outlier, this lets short scales catch
isolated spikes and long scales catch sustained artifacts.

The result is packaged in `PatientArtifactReport` so that downstream sprints
can ask "is window i of scale s a calm baseline window for patient X?" with
a single attribute lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

import config
from artifact.indicators import (
    INDICATOR_NAMES,
    batch_energy_ratio,
    batch_enmo_peak,
    batch_jerk_peak,
    batch_yz_bp_rms,
)
from artifact.robust_stats import (
    ThresholdResult,
    iterative_threshold,
)
from data_io.data_loader import PatientData
from preprocessing.pipeline import (
    ProcessedHand,
    ProcessedSession,
    preprocess_patient,
)
from preprocessing.windowing import (
    sliding_windows_1d,
    sliding_windows_multi,
    window_starts,
)


# -----------------------------------------------------------------------------
# Configuration of the filter
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterConfig:
    """Settings shared by all patients during a single ablation run."""
    k: float = 5.0
    max_iter: int = 5
    scales_s: tuple[float, ...] = (0.5, 1.0, 2.0)
    overlap: float = 0.75
    min_n_calm: int = 30
    relax_k: float = 7.0   # k used when min_n_calm not met (fallback)


# -----------------------------------------------------------------------------
# Per-(patient, session, scale) result
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ScaleResult:
    scale_s: float
    n_windows: int
    starts_idx: np.ndarray                       # sample indices of window start
    indicators: pd.DataFrame                     # rows: windows, cols: INDICATOR_NAMES
    threshold_per_indicator: dict[str, ThresholdResult]
    mask_outlier_any: np.ndarray                 # bool (n_windows,) OR-merge across indicators
    k_used: float                                # k actually applied (may be relax_k)


@dataclass(frozen=True)
class SessionArtifactResult:
    session_label: str
    fs: float
    per_scale: dict[float, ScaleResult]
    calm_idx_per_scale: dict[float, np.ndarray]
    outlier_idx_per_scale: dict[float, np.ndarray]
    n_total_per_scale: dict[float, int]


@dataclass(frozen=True)
class PatientArtifactReport:
    patient_id: str
    group: str
    fs: float
    per_session: dict[str, SessionArtifactResult]
    cfg: FilterConfig


# -----------------------------------------------------------------------------
# Computation of indicator matrices at one scale
# -----------------------------------------------------------------------------
def _indicators_at_scale(still: ProcessedHand, active: ProcessedHand,
                         scale_s: float, overlap: float) -> tuple[
                             np.ndarray, np.ndarray, pd.DataFrame]:
    """Window the still/active processed signals at `scale_s` and return
    (window_starts, indicators_df).
    """
    fs = still.fs
    starts = window_starts(len(still.t), fs, scale_s, overlap)

    enmo_w = sliding_windows_1d(still.enmo, fs, window_s=scale_s,
                                overlap=overlap)
    yz_bp_w_still = sliding_windows_multi(still.yz_bp, fs, window_s=scale_s,
                                          overlap=overlap)
    yz_bp_w_active = sliding_windows_multi(active.yz_bp, fs, window_s=scale_s,
                                           overlap=overlap)

    df = pd.DataFrame({
        "enmo_peak":   batch_enmo_peak(enmo_w),
        "jerk_peak":   batch_jerk_peak(yz_bp_w_still, fs),
        "yz_bp_rms":   batch_yz_bp_rms(yz_bp_w_still),
        "energy_ratio": batch_energy_ratio(yz_bp_w_still, yz_bp_w_active),
    })
    return starts, df


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def detect_artifacts_patient(
    p: PatientData,
    proc: Mapping[str, ProcessedSession],
    cfg: FilterConfig | None = None,
) -> PatientArtifactReport:
    """Run the multi-scale artifact filter on one patient."""
    if cfg is None:
        cfg = FilterConfig()

    per_session: dict[str, SessionArtifactResult] = {}

    for session_label, ps in proc.items():
        still, active = ps.still, ps.active
        per_scale: dict[float, ScaleResult] = {}
        calm_idx_per_scale: dict[float, np.ndarray] = {}
        outlier_idx_per_scale: dict[float, np.ndarray] = {}
        n_total_per_scale: dict[float, int] = {}

        for scale_s in cfg.scales_s:
            starts, ind_df = _indicators_at_scale(
                still, active, scale_s, cfg.overlap,
            )
            n_win = len(ind_df)

            # Try with primary k; if too few calm windows, retry with relax_k.
            k_used = cfg.k
            scale_res = _threshold_scale(
                starts, ind_df, scale_s, k_used, cfg.max_iter,
            )
            if int((~scale_res.mask_outlier_any).sum()) < cfg.min_n_calm:
                k_used = cfg.relax_k
                scale_res = _threshold_scale(
                    starts, ind_df, scale_s, k_used, cfg.max_iter,
                )

            per_scale[scale_s] = scale_res
            calm_idx_per_scale[scale_s] = np.where(~scale_res.mask_outlier_any)[0]
            outlier_idx_per_scale[scale_s] = np.where(scale_res.mask_outlier_any)[0]
            n_total_per_scale[scale_s] = n_win

        per_session[session_label] = SessionArtifactResult(
            session_label=session_label,
            fs=still.fs,
            per_scale=per_scale,
            calm_idx_per_scale=calm_idx_per_scale,
            outlier_idx_per_scale=outlier_idx_per_scale,
            n_total_per_scale=n_total_per_scale,
        )

    return PatientArtifactReport(
        patient_id=p.patient_id,
        group=p.group,
        fs=p.fs,
        per_session=per_session,
        cfg=cfg,
    )


def _threshold_scale(starts: np.ndarray, ind_df: pd.DataFrame,
                     scale_s: float, k: float, max_iter: int) -> ScaleResult:
    """Apply iterative robust thresholds to every indicator at one scale."""
    threshold_per_indicator: dict[str, ThresholdResult] = {}
    masks: list[np.ndarray] = []
    for name in INDICATOR_NAMES:
        res = iterative_threshold(ind_df[name].to_numpy(), k=k, max_iter=max_iter)
        threshold_per_indicator[name] = res
        masks.append(res.mask_outlier)

    if masks:
        mask_any = np.logical_or.reduce(masks)
    else:
        mask_any = np.zeros(len(ind_df), dtype=bool)

    return ScaleResult(
        scale_s=scale_s,
        n_windows=len(ind_df),
        starts_idx=starts,
        indicators=ind_df,
        threshold_per_indicator=threshold_per_indicator,
        mask_outlier_any=mask_any,
        k_used=k,
    )


# -----------------------------------------------------------------------------
# Time-domain outlier mask helper
# -----------------------------------------------------------------------------
def outlier_time_mask(report: PatientArtifactReport, session_label: str,
                      n_samples: int) -> np.ndarray:
    """Combine the per-scale outlier windows into a single sample-level mask.

    A sample t is marked outlier if any window at any scale that covers t is
    flagged outlier. Useful for plotting and for downstream sprints that
    operate on a time-aligned anomaly track.
    """
    sess = report.per_session[session_label]
    mask = np.zeros(n_samples, dtype=bool)
    for scale_s, scale_res in sess.per_scale.items():
        win_len = int(scale_s * sess.fs)
        for idx in scale_res.outlier_idx_per_scale_local(scale_res):
            start = scale_res.starts_idx[idx]
            end = min(start + win_len, n_samples)
            mask[start:end] = True
    return mask


# Convenience method patched onto ScaleResult for clarity in callers.
def _outlier_idx_local(scale_res: ScaleResult) -> np.ndarray:
    return np.where(scale_res.mask_outlier_any)[0]


ScaleResult.outlier_idx_per_scale_local = staticmethod(_outlier_idx_local)  # type: ignore[attr-defined]
