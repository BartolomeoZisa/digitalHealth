"""
Plotly figure factories for the Streamlit web app (Sprint 6.1).

Two figures only:

* `event_zoom_figure`: 2 s zoom on the selected window with both hands
  overlaid. Used as the primary visual in the drill-down panel.
* `scatter_pair_figure`: side-by-side scatter of the per-window points,
  one panel per session. Used in the geometric pattern panel.

All labels are in English, non-technical. No em-dash characters anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


COLOR_DOM = "#1f77b4"
COLOR_NDOM = "#d62728"
COLOR_EVENT = "rgba(46, 204, 113, 0.30)"


def event_zoom_figure(*, t: np.ndarray,
                      dom_yz_bp: np.ndarray,
                      ndom_yz_bp: np.ndarray,
                      t_start_s: float,
                      win_s: float) -> go.Figure:
    """2 second zoom around the selected window, both hands overlaid."""
    centre = t_start_s + win_s / 2.0
    t0 = max(0.0, centre - 1.0)
    t1 = min(float(t[-1]), centre + 1.0)
    mask = (t >= t0) & (t <= t1)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.10,
        subplot_titles=("Active hand acceleration",
                        "Still hand acceleration"),
    )
    for j, axis in enumerate(("Y", "Z")):
        fig.add_trace(go.Scatter(
            x=t[mask], y=dom_yz_bp[mask, j], mode="lines",
            line=dict(color=COLOR_DOM,
                      dash="solid" if axis == "Y" else "dot", width=1.2),
            name=f"Active {axis}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t[mask], y=ndom_yz_bp[mask, j], mode="lines",
            line=dict(color=COLOR_NDOM,
                      dash="solid" if axis == "Y" else "dot", width=1.2),
            name=f"Still {axis}",
        ), row=2, col=1)

    for row in (1, 2):
        fig.add_vrect(x0=t_start_s, x1=t_start_s + win_s,
                      fillcolor=COLOR_EVENT, line_width=0,
                      layer="below", row=row, col=1)

    fig.update_xaxes(range=[t0, t1], title_text="time (s)", row=2, col=1)
    fig.update_yaxes(range=list(config.PLOT_YZ_FILTERED_YLIM),
                     title_text="acceleration (g)", row=1, col=1)
    fig.update_yaxes(range=list(config.PLOT_YZ_FILTERED_YLIM),
                     title_text="acceleration (g)", row=2, col=1)
    fig.update_layout(
        height=460, margin=dict(l=50, r=30, t=50, b=40),
        title=f"Selected event at t = {t_start_s:.2f} s "
              f"(window {win_s:.1f} s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5),
    )
    return fig


def _scatter_panel(fig: go.Figure, df: pd.DataFrame, col: int,
                   title: str) -> None:
    """Single scatter panel inside an existing subplot grid."""
    if df.empty:
        return
    points = df.copy()
    mask = (~points["is_boundary"].astype(bool)) & (
        ~points["is_artifact"].astype(bool)
    )
    base = points[mask]
    mm = points[mask & points["is_mm_candidate"].astype(bool)]

    fig.add_trace(go.Scatter(
        x=base["asymmetry_index"], y=base["xcorr_max"],
        mode="markers",
        marker=dict(size=7, color=base["score_median"],
                    colorscale="Viridis", cmin=0.0, cmax=1.0,
                    showscale=col == 2,
                    colorbar=dict(title="severity") if col == 2 else None,
                    opacity=0.75),
        name="windows", showlegend=False,
        hovertemplate=("t=%{customdata[0]:.1f}s<br>"
                       "stillness balance=%{x:.2f}<br>"
                       "hand sync=%{y:.2f}<br>"
                       "severity=%{marker.color:.2f}"),
        customdata=base[["t_start_s"]].to_numpy(),
    ), row=1, col=col)

    if not mm.empty:
        fig.add_trace(go.Scatter(
            x=mm["asymmetry_index"], y=mm["xcorr_max"],
            mode="markers",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(color="#27ae60", width=2)),
            name="MM event", showlegend=col == 1,
            hovertemplate=("MM at t=%{customdata[0]:.1f}s<br>"
                           "stillness balance=%{x:.2f}<br>"
                           "hand sync=%{y:.2f}"),
            customdata=mm[["t_start_s"]].to_numpy(),
        ), row=1, col=col)

    fig.update_xaxes(range=[-1.05, 1.05], title_text="stillness balance",
                     row=1, col=col)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="hand sync",
                     row=1, col=col)


def scatter_pair_figure(df_dom: pd.DataFrame,
                        df_ndom: pd.DataFrame) -> go.Figure:
    """Two scatter panels side by side, one per session."""
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            "BBT with dominant hand (MM watched on the non-dominant)",
            "BBT with non-dominant hand (MM watched on the dominant)",
        ),
        horizontal_spacing=0.10,
    )
    _scatter_panel(fig, df_dom, col=1, title="dom")
    _scatter_panel(fig, df_ndom, col=2, title="ndom")
    fig.update_layout(
        height=420, margin=dict(l=50, r=30, t=60, b=40),
        title="Window pattern: each dot is a 1 s window",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.15,
                    xanchor="center", x=0.5),
    )
    return fig
