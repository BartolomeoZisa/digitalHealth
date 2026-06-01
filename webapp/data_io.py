"""
Input adapters for the Streamlit web tool.

Two entry points:

  * `load_demo_patient(group, patient_id)`: wraps the existing demo loader
    used in the Sprint 0-5 EDA scripts. Read-only path on local CSV.
  * `load_patient_from_uploaded_csv(upload, expected_id=None)`: validates
    a one-patient CSV uploaded through the Streamlit `st.file_uploader` and
    builds a `PatientData` using the same internal helpers as the demo
    loader.
"""
from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import IO

import pandas as pd

from data_io import schema as sch
from data_io.data_loader import (
    PatientData,
    Session,
    _build_hand_signal,
    infer_fs,
    list_patient_ids,
    load_patient,
)


def load_demo_patient(group: str, patient_id: str) -> PatientData:
    """Wrapper that re-exports the cached demo loader."""
    return load_patient(group, patient_id)


def list_demo_patients(group: str) -> list[str]:
    """Patient IDs available for the dropdown."""
    return list_patient_ids(group)


# -----------------------------------------------------------------------------
# CSV upload path
# -----------------------------------------------------------------------------
def _coerce_to_dataframe(file_like: IO | str | bytes) -> pd.DataFrame:
    """Read CSV from a Streamlit UploadedFile, a path, or a bytes object."""
    if hasattr(file_like, "read"):
        data = file_like.read()
        if isinstance(data, bytes):
            return pd.read_csv(BytesIO(data))
        return pd.read_csv(StringIO(data))
    if isinstance(file_like, (str, Path)):
        return pd.read_csv(file_like)
    if isinstance(file_like, (bytes, bytearray)):
        return pd.read_csv(BytesIO(file_like))
    raise TypeError(f"unsupported input type: {type(file_like)}")


def load_patient_from_uploaded_csv(file_like,
                                   patient_id: str | None = None
                                   ) -> PatientData:
    """Build a PatientData from a one-patient CSV uploaded by the user.

    Validation:
        * the dataframe must satisfy `schema.validate_dataframe`
        * if `patient_id` is provided, all rows must match it
        * if more than one `id` is present, raises ValueError so the user
          knows the upload should contain a single patient
    """
    df = _coerce_to_dataframe(file_like)
    sch.validate_dataframe(df)
    df[sch.COL_DT] = pd.to_datetime(df[sch.COL_DT])

    ids = df[sch.COL_ID].unique()
    if len(ids) > 1:
        raise ValueError(
            f"upload contains {len(ids)} patient ids; expected exactly 1: {ids}"
        )
    pid = str(ids[0])
    if patient_id is not None and pid != patient_id:
        raise ValueError(f"upload patient id {pid!r} differs from {patient_id!r}")

    group = str(df[sch.COL_TYPE].iloc[0])
    hand_dominance = str(df[sch.COL_HAND_DOMINANCE].iloc[0])

    # Use any rows with hand_type == "dom" to estimate fs; fallback to all rows.
    fs_source = df[df[sch.COL_HAND_TYPE] == "dom"][sch.COL_DT]
    if len(fs_source) < 2:
        fs_source = df[sch.COL_DT]
    fs = infer_fs(fs_source)

    sessions: dict[str, Session] = {}
    for session_label in ("dom", "ndom"):
        ses = df[df[sch.COL_SESSION] == session_label]
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

    if not sessions:
        raise ValueError("upload does not contain a valid dom or ndom session")

    return PatientData(
        patient_id=pid,
        group=group,
        hand_dominance=hand_dominance,
        fs=fs,
        sessions=sessions,
    )
