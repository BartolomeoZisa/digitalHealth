"""
Detector pipeline, fits and scores the 3 detectors on a per-patient basis.

For every (patient, session, feature_set) triple:
  1. Select the feature columns indicated by the active feature set.
  2. Slice the rows into a calm subset (`is_outlier == False`) and the full
     subset (all windows).
  3. Fit the three detectors on the calm features.
  4. Score the full subset with each detector and combine into a median
     ensemble.
  5. Build the final window-level DataFrame.

Outputs are returned as a `PatientDetectorOutput` dataclass keyed by
session label and feature set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from detectors.base import Detector
from detectors.ensemble import build_window_dataframe
from detectors.isolation_forest import IsolationForestDetector
from detectors.pca_reconstruction import PCAReconstructionDetector
from detectors.robust_quantile import RobustQuantileDetector


@dataclass(frozen=True)
class SessionDetectorOutput:
    patient_id: str
    group: str
    session_label: str
    feature_set: str
    feature_names: tuple[str, ...]
    df_windows: pd.DataFrame      # rows: windows, cols: score_*, is_artifact, features
    n_calm: int


@dataclass(frozen=True)
class PatientDetectorOutput:
    patient_id: str
    group: str
    per_session: dict[tuple[str, str], SessionDetectorOutput]  # (session, fset)


# -----------------------------------------------------------------------------
# Single-session run
# -----------------------------------------------------------------------------
def _run_session(
    *,
    patient_id: str,
    group: str,
    session_label: str,
    feature_set_name: str,
    feature_names: tuple[str, ...],
    features_df: pd.DataFrame,
    t_start_s: np.ndarray,
) -> SessionDetectorOutput:
    """Fit detectors on calm, score full, build the window DataFrame."""
    is_outlier = features_df["is_outlier"].to_numpy().astype(bool)
    is_boundary = (
        features_df["is_boundary"].to_numpy().astype(bool)
        if "is_boundary" in features_df.columns
        else np.zeros(len(features_df), dtype=bool)
    )
    X = features_df[list(feature_names)].to_numpy(dtype=float)
    X_calm = X[~is_outlier]

    if len(X_calm) < 5:
        # Fall back: emit zeros (downstream code can flag this as warning).
        zeros = np.zeros(len(X), dtype=float)
        df_out = build_window_dataframe(
            t_start_s=t_start_s,
            is_outlier=is_outlier,
            is_boundary=is_boundary,
            features_df=features_df[list(feature_names)],
            score_quantile=zeros, score_iforest=zeros, score_pca=zeros,
            attribution_quantile=[[] for _ in range(len(X))],
        )
        return SessionDetectorOutput(
            patient_id=patient_id, group=group,
            session_label=session_label,
            feature_set=feature_set_name,
            feature_names=feature_names,
            df_windows=df_out,
            n_calm=len(X_calm),
        )

    detectors: list[Detector] = [
        RobustQuantileDetector(feature_names),
        IsolationForestDetector(feature_names),
        PCAReconstructionDetector(feature_names),
    ]
    for d in detectors:
        d.fit(X_calm)

    sq = detectors[0].score(X)
    si = detectors[1].score(X)
    sp = detectors[2].score(X)
    # Attribution comes from the quantile detector, most interpretable.
    attr_q = detectors[0].attribution(X, top_k=3)

    df_out = build_window_dataframe(
        t_start_s=t_start_s,
        is_outlier=is_outlier,
        is_boundary=is_boundary,
        features_df=features_df[list(feature_names)],
        score_quantile=sq,
        score_iforest=si,
        score_pca=sp,
        attribution_quantile=attr_q,
    )
    return SessionDetectorOutput(
        patient_id=patient_id, group=group,
        session_label=session_label,
        feature_set=feature_set_name,
        feature_names=feature_names,
        df_windows=df_out,
        n_calm=int((~is_outlier).sum()),
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run_patient_detectors(
    *,
    patient_id: str,
    group: str,
    features_by_session: dict[str, pd.DataFrame],   # session_label -> df_windows
    starts_by_session: dict[str, np.ndarray],       # session_label -> sample-starts
    fs: float,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
) -> PatientDetectorOutput:
    """Run all feature sets on all sessions of one patient."""
    if feature_sets is None:
        feature_sets = config.DETECTOR_FEATURE_SETS

    per_session: dict[tuple[str, str], SessionDetectorOutput] = {}
    for session_label, df in features_by_session.items():
        starts = starts_by_session[session_label]
        t_start_s = starts / float(fs)
        for fset_name, fset_cols in feature_sets.items():
            per_session[(session_label, fset_name)] = _run_session(
                patient_id=patient_id,
                group=group,
                session_label=session_label,
                feature_set_name=fset_name,
                feature_names=tuple(fset_cols),
                features_df=df,
                t_start_s=t_start_s,
            )
    return PatientDetectorOutput(
        patient_id=patient_id, group=group, per_session=per_session,
    )
