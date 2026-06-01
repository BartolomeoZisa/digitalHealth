"""
Dual-path preprocessing pipeline for one hand of one session.

Path A, Scalar ENMO
    ENMO is a single scalar trace that summarises movement amount after
    gravity removal. Optionally we band-pass it as well, so downstream
    activity statistics are not contaminated by slow drifts.

Path B, Per-axis directional (Y/Z)
    For features that depend on direction (bilateral cross-correlation,
    spectral content per axis), we drop the noisy X axis and band-pass Y
    and Z independently.

Both paths share the same time axis (t=0 at the start of the session) and
the same fs. The output is an immutable `ProcessedHand` dataclass that
carries every derived signal needed by later sprints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

import config
from data_io.data_loader import HandSignal, Session
from preprocessing.axes import select_axes, vector_magnitude
from preprocessing.derivatives import jerk_magnitude
from preprocessing.filters import bandpass
from preprocessing.gravity import enmo


@dataclass(frozen=True)
class ProcessedHand:
    """All derived signals for one hand within one session.

    Time-aligned with `t`. All entries are numpy arrays in their natural
    units (g for accelerations; g for ENMO; g/s for jerk magnitude).
    """
    hand_type: str            # 'dom' or 'ndom'
    is_active: bool           # True if performing BBT in this session
    fs: float
    t: np.ndarray             # (T,)
    raw: np.ndarray           # (T, 3), original signal (XYZ)
    enmo: np.ndarray          # (T,), clipped ENMO
    enmo_bp: np.ndarray       # (T,), band-passed ENMO
    yz: np.ndarray            # (T, 2), selected axes after X drop
    yz_bp: np.ndarray         # (T, 2), band-passed YZ
    jerk_mag: np.ndarray      # (T,), |d(accel)/dt| on YZ
    vec_mag: np.ndarray       # (T,), Euclidean norm of raw XYZ (incl. gravity)


@dataclass(frozen=True)
class ProcessedSession:
    """The dom / ndom processed pair for one BBT session."""
    session_label: str
    dom: ProcessedHand
    ndom: ProcessedHand

    @property
    def active(self) -> ProcessedHand:
        return self.dom if self.session_label == "dom" else self.ndom

    @property
    def still(self) -> ProcessedHand:
        return self.ndom if self.session_label == "dom" else self.dom


def preprocess_hand(h: HandSignal, fs: float,
                    *, bp_low_hz: float | None = None,
                    bp_high_hz: float | None = None) -> ProcessedHand:
    """Apply both preprocessing paths to a HandSignal."""
    if bp_low_hz is None:
        bp_low_hz = config.BANDPASS_LOW_HZ
    if bp_high_hz is None:
        bp_high_hz = config.BANDPASS_HIGH_HZ

    raw = np.asarray(h.accel, dtype=np.float64)
    vmag = vector_magnitude(raw)
    e = enmo(raw)

    # Path A: bandpass the scalar ENMO trace.
    e_bp = bandpass(e, fs, low_hz=bp_low_hz, high_hz=bp_high_hz)

    # Path B: directional Y/Z.
    yz = select_axes(raw, keep=("y", "z"))
    yz_bp = bandpass(yz, fs, low_hz=bp_low_hz, high_hz=bp_high_hz)
    j_mag = jerk_magnitude(yz_bp, fs)

    return ProcessedHand(
        hand_type=h.hand_type,
        is_active=h.is_active,
        fs=fs,
        t=h.t,
        raw=raw,
        enmo=e,
        enmo_bp=e_bp,
        yz=yz,
        yz_bp=yz_bp,
        jerk_mag=j_mag,
        vec_mag=vmag,
    )


def preprocess_session(s: Session, fs: float,
                       **kwargs) -> ProcessedSession:
    """Process both hands of a session and pack into a ProcessedSession."""
    return ProcessedSession(
        session_label=s.session_label,
        dom=preprocess_hand(s.dom, fs, **kwargs),
        ndom=preprocess_hand(s.ndom, fs, **kwargs),
    )


def preprocess_patient(sessions: Mapping[str, Session], fs: float,
                       **kwargs) -> dict[str, ProcessedSession]:
    """Preprocess all sessions of one patient."""
    return {lbl: preprocess_session(s, fs, **kwargs)
            for lbl, s in sessions.items()}
