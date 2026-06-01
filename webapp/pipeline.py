"""
Single-patient analysis pipeline for the Streamlit web tool.

Wraps the Sprint-0 through Sprint-5 building blocks into one call:

    analyse_patient(patient) -> PatientAnalysis

The returned dataclass carries everything the UI needs (per-session window
DataFrames, processed signals for the drill-down panel, scatter dispersion
ratio for the side info panel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from artifact.filter import detect_artifacts_patient
from detectors.pipeline import run_patient_detectors
from eda.artifact_demo import CHOSEN_CFG
from features.extractor import extract_patient_features
from features.scatter_dispersion import (
    compute_session_dispersion,
    safe_ratio,
)
from data_io.data_loader import PatientData
from preprocessing.pipeline import (
    ProcessedSession,
    preprocess_patient,
)


PRIMARY_FSET: str = "selected"


@dataclass(frozen=True)
class SessionAnalysis:
    """Everything the UI needs to render one session of one patient."""
    session_label: str
    fs: float
    df_windows: pd.DataFrame
    processed: ProcessedSession
    n_mm_candidate: int
    disp_pairwise: float


@dataclass(frozen=True)
class PatientAnalysis:
    """Top-level container returned by `analyse_patient`."""
    patient_id: str
    group: str
    hand_dominance: str
    fs: float
    sessions: dict[str, SessionAnalysis]
    dispersion_ratio: float


def analyse_patient(patient: PatientData) -> PatientAnalysis:
    """Run preprocessing → artifact → features → detectors for one patient.

    Returns a `PatientAnalysis` ready to be consumed by the UI layer.
    """
    proc = preprocess_patient(patient.sessions, patient.fs)
    report = detect_artifacts_patient(patient, proc, CHOSEN_CFG)
    feats = extract_patient_features(
        proc, report,
        patient_id=patient.patient_id, group=patient.group,
    )
    features_by_session = {lbl: sf.df_windows for lbl, sf in feats.items()}
    starts_by_session = {lbl: sf.window_starts_idx for lbl, sf in feats.items()}

    detector_out = run_patient_detectors(
        patient_id=patient.patient_id, group=patient.group,
        features_by_session=features_by_session,
        starts_by_session=starts_by_session, fs=patient.fs,
    )

    sessions: dict[str, SessionAnalysis] = {}
    disp_by_session: dict[str, float] = {}
    for sess_label, ps in proc.items():
        key = (sess_label, PRIMARY_FSET)
        if key not in detector_out.per_session:
            continue
        sdo = detector_out.per_session[key]
        df = sdo.df_windows
        n_mm = int(df.is_mm_candidate.sum())
        disp = compute_session_dispersion(df)
        disp_by_session[sess_label] = float(disp.disp_pairwise)
        sessions[sess_label] = SessionAnalysis(
            session_label=sess_label,
            fs=patient.fs,
            df_windows=df,
            processed=ps,
            n_mm_candidate=n_mm,
            disp_pairwise=float(disp.disp_pairwise),
        )

    dispersion_ratio = safe_ratio(
        disp_by_session.get("ndom", float("nan")),
        disp_by_session.get("dom", float("nan")),
    )

    return PatientAnalysis(
        patient_id=patient.patient_id,
        group=patient.group,
        hand_dominance=patient.hand_dominance,
        fs=patient.fs,
        sessions=sessions,
        dispersion_ratio=float(dispersion_ratio),
    )
