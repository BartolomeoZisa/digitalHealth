"""
Mirror Movement Explorer, Streamlit single-page web app (Sprint 6.1).

Layout:
1. Header with patient info and event counts.
2. Selected event drill-down (front and centre).
3. Events table.
4. Geometric pattern panel: per-window scatter, both sessions side by side.

Default focus is the non-dominant active session (clinical request).
Run with:
    streamlit run webapp/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explain.attribution import (  # noqa: E402
    explain_event, parse_attribution_string,
)
from explain.feature_glossary import clinical_sentence_en  # noqa: E402
from data_io.data_loader import PatientData  # noqa: E402
from webapp.data_io import (  # noqa: E402
    list_demo_patients, load_demo_patient, load_patient_from_uploaded_csv,
)
from webapp.pipeline import PatientAnalysis, analyse_patient  # noqa: E402
from webapp.plots import event_zoom_figure, scatter_pair_figure  # noqa: E402


WIN_S: float = 1.0
SESSION_ORDER: tuple[str, ...] = ("ndom", "dom")
SESSION_LABEL: dict[str, str] = {
    "ndom": "BBT with non-dominant hand (looking for MM on the dominant)",
    "dom":  "BBT with dominant hand (looking for MM on the non-dominant)",
}


# -----------------------------------------------------------------------------
# Page setup and data loading
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mirror Movement Explorer",
                   page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def _cached_demo(group: str, patient_id: str) -> PatientAnalysis:
    return analyse_patient(load_demo_patient(group, patient_id))


def _load_uploaded(file_bytes: bytes) -> PatientAnalysis:
    patient: PatientData = load_patient_from_uploaded_csv(file_bytes)
    return analyse_patient(patient)


def _sidebar_pick() -> tuple[PatientAnalysis | None, str | None]:
    st.sidebar.title("Mirror Movement Explorer")
    st.sidebar.caption("Decision support tool, not a diagnosis.")
    source = st.sidebar.radio(
        "Data source",
        ("Demo patient", "Upload CSV"),
        index=0,
    )
    if source == "Demo patient":
        group = st.sidebar.selectbox("Group", ("ucp", "td"), index=0)
        pid = st.sidebar.selectbox("Patient", list_demo_patients(group),
                                    index=0)
        try:
            return _cached_demo(group, pid), None
        except Exception as exc:  # noqa: BLE001
            return None, f"Cannot load {pid}: {exc}"
    upload = st.sidebar.file_uploader("Patient CSV", type=("csv",))
    if upload is None:
        return None, None
    try:
        return _load_uploaded(upload.getvalue()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Upload failed: {exc}"


# -----------------------------------------------------------------------------
# Render helpers (one section per function)
# -----------------------------------------------------------------------------
def render_header(a: PatientAnalysis) -> None:
    st.markdown(f"## 🧠 Patient `{a.patient_id}` (group `{a.group.upper()}`)")
    ndom_events = a.sessions.get("ndom")
    dom_events = a.sessions.get("dom")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant hand", a.hand_dominance)
    c2.metric("MM on dominant (BBT done with ndom)",
              str(ndom_events.n_mm_candidate) if ndom_events else "n/a")
    c3.metric("MM on non-dominant (BBT done with dom)",
              str(dom_events.n_mm_candidate) if dom_events else "n/a")
    ratio_text = (f"{a.dispersion_ratio:.2f}" if a.dispersion_ratio == a.dispersion_ratio  # noqa: PLR0124
                  else "n/a")
    c4.metric("Spread ratio (ndom / dom)", ratio_text)


def _selected_event(df: pd.DataFrame, key: str) -> pd.Series | None:
    mm_df = df[df.is_mm_candidate].sort_values("score_median", ascending=False)
    if mm_df.empty:
        return None
    state = st.session_state.get(key)
    idx = 0
    if state and state.selection.rows:
        idx = int(state.selection.rows[0])
    idx = min(idx, len(mm_df) - 1)
    return mm_df.iloc[idx]


def render_drilldown(session_analysis, event: pd.Series | None) -> None:
    st.markdown("### What this event looks like")
    if event is None:
        st.success("No mirror movement candidates in this session.")
        return
    contribs = parse_attribution_string(event.attribution_top3)
    evt = explain_event(event.t_start_s, event.score_median,
                        event.is_artifact, contribs)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Event start (s)", f"{evt.t_start_s:.2f}")
        st.metric("Severity score", f"{evt.score_median:.2f} / 1.00")
        st.markdown(f"**Tag:** {evt.tag}")
        st.markdown("**Clinical reading:**")
        for name, z in contribs:
            st.markdown(f"- {clinical_sentence_en(name, z)}")
    with c2:
        still = session_analysis.processed.still
        active = session_analysis.processed.active
        fig = event_zoom_figure(
            t=still.t,
            dom_yz_bp=active.yz_bp,
            ndom_yz_bp=still.yz_bp,
            t_start_s=float(evt.t_start_s),
            win_s=WIN_S,
        )
        st.plotly_chart(fig, width="stretch")


def render_events_table(df: pd.DataFrame, key: str) -> None:
    st.markdown("### Events found in this session")
    mm_df = df[df.is_mm_candidate].sort_values("score_median", ascending=False)
    if mm_df.empty:
        st.info("Nothing flagged in this session.")
        return
    table = mm_df[["t_start_s", "score_median",
                   "asymmetry_index", "xcorr_max"]].rename(columns={
        "t_start_s": "t (s)",
        "score_median": "severity",
        "asymmetry_index": "stillness balance",
        "xcorr_max": "hand sync",
    })
    table.insert(0, "id", range(len(table)))
    st.dataframe(
        table.style.format({
            "t (s)": "{:.2f}",
            "severity": "{:.2f}",
            "stillness balance": "{:.2f}",
            "hand sync": "{:.2f}",
        }),
        hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row",
        key=key,
    )


def render_geometric_panel(a: PatientAnalysis) -> None:
    st.markdown("### Window pattern across the session")
    dom_df = a.sessions["dom"].df_windows if "dom" in a.sessions else pd.DataFrame()
    ndom_df = a.sessions["ndom"].df_windows if "ndom" in a.sessions else pd.DataFrame()
    fig = scatter_pair_figure(dom_df, ndom_df)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each dot is a one-second window. Tighter clusters mean the patient "
        "behaves consistently across the session. A wider cluster on the "
        "non-dominant-active panel suggests intermittent mirror movements."
    )


# -----------------------------------------------------------------------------
# Page body
# -----------------------------------------------------------------------------
analysis, err = _sidebar_pick()
if err:
    st.error(err)
    st.stop()
if analysis is None:
    st.info("Pick a demo patient or upload a CSV in the sidebar to start.")
    st.stop()

render_header(analysis)
st.divider()

available_sessions = [s for s in SESSION_ORDER if s in analysis.sessions]
if not available_sessions:
    st.warning("No valid sessions in this patient.")
    st.stop()
tabs = st.tabs([SESSION_LABEL[s] for s in available_sessions])
for tab, sess in zip(tabs, available_sessions):
    with tab:
        session = analysis.sessions[sess]
        table_key = f"events_{sess}"
        event = _selected_event(session.df_windows, table_key)
        render_drilldown(session, event)
        st.markdown("")
        render_events_table(session.df_windows, table_key)

st.divider()
render_geometric_panel(analysis)
