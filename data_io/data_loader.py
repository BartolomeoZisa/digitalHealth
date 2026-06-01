"""
Patient-level data loading.

Each patient has 2 recording sessions in the BBT protocol:
    * session == 'dom'  → dominant hand is the ACTIVE hand performing the test;
                          non-dominant hand should stay still (target for MM).
    * session == 'ndom' → non-dominant hand is the active hand; dominant hand
                          should stay still.

For each session both hands are recorded simultaneously, so a session yields a
bilateral signal (active + still). This module exposes a `PatientData`
dataclass that holds, per session, the (T, 3) accelerometer signal for each
hand together with a common timestamp axis aligned to t=0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import config
from data_io import schema as sch


@dataclass(frozen=True)
class HandSignal:
    """Accelerometer signal of one hand within one session.

    Attributes
    ----------
    accel : np.ndarray of shape (T, 3)
        Columns: (x, y, z) in g.
    t : np.ndarray of shape (T,)
        Time axis in seconds, starting at 0.
    hand_type : str
        'dom' or 'ndom' (relative to the patient).
    is_active : bool
        True when this hand is the one performing the BBT in the session.
    """
    accel: np.ndarray
    t: np.ndarray
    hand_type: str
    is_active: bool

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


@dataclass(frozen=True)
class Session:
    """A single BBT session: one active hand, one still hand, both recorded."""
    session_label: str   # 'dom' or 'ndom' (which hand is active)
    dom: HandSignal
    ndom: HandSignal

    @property
    def active(self) -> HandSignal:
        return self.dom if self.session_label == "dom" else self.ndom

    @property
    def still(self) -> HandSignal:
        return self.ndom if self.session_label == "dom" else self.dom


@dataclass(frozen=True)
class PatientData:
    """All recordings for one patient."""
    patient_id: str
    group: str           # 'ucp' or 'td'
    hand_dominance: str  # 'L' or 'R'
    fs: float
    sessions: dict[str, Session] = field(default_factory=dict)

    def session(self, label: str) -> Session:
        return self.sessions[label]


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _read_group_csv(group: str) -> pd.DataFrame:
    """Read and validate the raw CSV for one group. Cached across calls."""
    path: Path = config.UCP_CSV if group == "ucp" else config.TD_CSV
    df = pd.read_csv(path)
    sch.validate_dataframe(df, expected_group=group)
    df[sch.COL_DT] = pd.to_datetime(df[sch.COL_DT])
    return df


def infer_fs(timestamps: pd.Series) -> float:
    """Infer sample rate from a monotonic datetime series.

    Uses the median inter-sample dt over the per-hand subseries to be robust to
    occasional missing rows.
    """
    if len(timestamps) < 2:
        raise ValueError("need >= 2 timestamps to infer fs")
    dt = timestamps.sort_values().diff().dropna().dt.total_seconds()
    median_dt = float(dt[dt > 0].median())
    if median_dt <= 0:
        raise ValueError("non-positive median dt")
    return 1.0 / median_dt


def list_patient_ids(group: str) -> list[str]:
    """Sorted list of patient ids for the given group."""
    df = _read_group_csv(group)
    ids = sorted(df[sch.COL_ID].unique(), key=_natural_id_key)
    return ids


def _natural_id_key(pid: str) -> tuple[str, int]:
    """Sort UCP0, UCP1, ..., UCP10 in human order."""
    prefix = "".join(ch for ch in pid if not ch.isdigit())
    digits = "".join(ch for ch in pid if ch.isdigit())
    return prefix, int(digits) if digits else 0


def _build_hand_signal(rows: pd.DataFrame, is_active: bool) -> HandSignal:
    rows = rows.sort_values(sch.COL_DT).reset_index(drop=True)
    accel = rows[[sch.COL_AX, sch.COL_AY, sch.COL_AZ]].to_numpy(dtype=np.float64)
    dt = (rows[sch.COL_DT] - rows[sch.COL_DT].iloc[0]).dt.total_seconds().to_numpy()
    return HandSignal(
        accel=accel,
        t=dt,
        hand_type=str(rows[sch.COL_HAND_TYPE].iloc[0]),
        is_active=is_active,
    )


def load_patient(group: str, patient_id: str) -> PatientData:
    """Load all sessions for one patient.

    Parameters
    ----------
    group:
        'ucp' or 'td'.
    patient_id:
        e.g. 'UCP0', 'TD3'.
    """
    if group not in config.GROUPS:
        raise ValueError(f"group must be one of {config.GROUPS}")
    df = _read_group_csv(group)
    sub = df[df[sch.COL_ID] == patient_id]
    if sub.empty:
        raise KeyError(f"patient {patient_id!r} not found in group {group!r}")

    hand_dominance = str(sub[sch.COL_HAND_DOMINANCE].iloc[0])
    fs = infer_fs(sub[sub[sch.COL_HAND_TYPE] == "dom"][sch.COL_DT])

    sessions: dict[str, Session] = {}
    for session_label in config.SESSION_LABELS:
        ses = sub[sub[sch.COL_SESSION] == session_label]
        if ses.empty:
            continue
        dom_rows = ses[ses[sch.COL_HAND_TYPE] == "dom"]
        ndom_rows = ses[ses[sch.COL_HAND_TYPE] == "ndom"]
        if dom_rows.empty or ndom_rows.empty:
            continue
        sessions[session_label] = Session(
            session_label=session_label,
            dom=_build_hand_signal(dom_rows, is_active=(session_label == "dom")),
            ndom=_build_hand_signal(ndom_rows, is_active=(session_label == "ndom")),
        )

    return PatientData(
        patient_id=patient_id,
        group=group,
        hand_dominance=hand_dominance,
        fs=fs,
        sessions=sessions,
    )


def load_group(group: str) -> Iterator[PatientData]:
    """Iterate over all patients in a group, in natural-sorted id order."""
    for pid in list_patient_ids(group):
        yield load_patient(group, pid)
