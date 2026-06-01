"""
Sprint 0 EDA, bilateral raw-signal inspection per patient.

For every patient (UCP and TD) produces one PNG with two columns (one per
session: 'dom-active' and 'ndom-active') and four rows:
    row 1 : active hand X/Y/Z
    row 2 : still  hand X/Y/Z
    row 3 : active hand magnitude
    row 4 : still  hand magnitude

All plots use FIXED axis scales taken from `config` so that figures from
different patients are directly comparable. Timestamps within a session are
aligned to t=0.

A summary CSV is written with per-patient/session statistics (duration, fs,
peak amplitudes, naive outlier counts) so that the next sprint can pick up
quantitative observations.

Run:
    python -m eda.explore
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data_io.data_loader import (
    HandSignal,
    PatientData,
    Session,
    list_patient_ids,
    load_patient,
)


# Naive outlier threshold (g). Anything beyond this on the STILL hand is
# almost certainly a gross artifact (e.g. arm raised). Used only for counting
# in the EDA summary; proper artifact filter is implemented in Sprint 2.
NAIVE_OUTLIER_THRESHOLD_G: float = 2.0


# -----------------------------------------------------------------------------
# Per-patient summary record
# -----------------------------------------------------------------------------
@dataclass
class SessionSummary:
    patient_id: str
    group: str
    session: str                  # 'dom' or 'ndom' (active hand)
    fs_hz: float
    duration_s: float
    n_samples: int
    # Active hand
    active_peak_g: float
    active_rms_g: float
    # Still hand (target for MM detection)
    still_peak_g: float
    still_rms_g: float
    still_naive_outliers: int     # n samples where |a| > NAIVE_OUTLIER_THRESHOLD_G
    still_outlier_pct: float


def _vec_mag(accel: np.ndarray) -> np.ndarray:
    """Vector magnitude |a| over time."""
    return np.linalg.norm(accel, axis=1)


def _hand_stats(h: HandSignal) -> tuple[float, float, int]:
    mag = _vec_mag(h.accel)
    peak = float(np.max(np.abs(h.accel)))
    rms = float(np.sqrt(np.mean(h.accel ** 2)))
    n_out = int(np.sum(mag > NAIVE_OUTLIER_THRESHOLD_G))
    return peak, rms, n_out


def _summarize_session(p: PatientData, label: str) -> SessionSummary | None:
    if label not in p.sessions:
        return None
    s: Session = p.sessions[label]
    a_peak, a_rms, _ = _hand_stats(s.active)
    s_peak, s_rms, s_n_out = _hand_stats(s.still)
    n = len(s.active.t)
    return SessionSummary(
        patient_id=p.patient_id,
        group=p.group,
        session=label,
        fs_hz=p.fs,
        duration_s=s.active.duration_s,
        n_samples=n,
        active_peak_g=a_peak,
        active_rms_g=a_rms,
        still_peak_g=s_peak,
        still_rms_g=s_rms,
        still_naive_outliers=s_n_out,
        still_outlier_pct=100.0 * s_n_out / max(n, 1),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _plot_hand_axes(ax, h: HandSignal, *, ylim: tuple[float, float], title: str) -> None:
    """Plot X/Y/Z on the same axes with fixed scales."""
    for axis_name, idx in zip(("x", "y", "z"), (0, 1, 2)):
        ax.plot(
            h.t,
            h.accel[:, idx],
            color=config.AXIS_COLORS[axis_name],
            linewidth=0.6,
            label=axis_name.upper(),
        )
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("g")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)


def _plot_hand_magnitude(ax, h: HandSignal, *, ylim: tuple[float, float], color: str, title: str) -> None:
    mag = _vec_mag(h.accel)
    ax.plot(h.t, mag, color=color, linewidth=0.7)
    ax.axhline(
        NAIVE_OUTLIER_THRESHOLD_G,
        color="red",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
        label=f"naive outlier @ {NAIVE_OUTLIER_THRESHOLD_G}g",
    )
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("|a| (g)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)


def plot_patient(p: PatientData, out_path: Path) -> None:
    """Generate the per-patient bilateral inspection figure."""
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=config.PLOT_FIGSIZE_PATIENT,
        dpi=config.PLOT_DPI,
        sharex=True,
    )

    fig.suptitle(
        f"{p.patient_id} | group={p.group.upper()} | dominance={p.hand_dominance} | "
        f"fs={p.fs:.1f} Hz | naive outlier threshold = {NAIVE_OUTLIER_THRESHOLD_G} g",
        fontsize=11,
    )

    # Magnitude plot uses a tighter, FIXED scale so values are comparable
    # across patients without saturating active-hand bursts.
    mag_ylim = (0.0, 6.0)

    for col, sess_label in enumerate(config.SESSION_LABELS):
        if sess_label not in p.sessions:
            for r in range(4):
                axes[r, col].text(0.5, 0.5, f"no session '{sess_label}'",
                                  ha="center", va="center",
                                  transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue
        s = p.sessions[sess_label]
        active_label = f"ACTIVE hand ({s.active.hand_type}), session {sess_label}-active"
        still_label = f"STILL hand ({s.still.hand_type}), session {sess_label}-active"

        _plot_hand_axes(
            axes[0, col], s.active,
            ylim=config.PLOT_ACCEL_YLIM,
            title=active_label,
        )
        _plot_hand_axes(
            axes[1, col], s.still,
            ylim=config.PLOT_ACCEL_YLIM_STILL,
            title=still_label,
        )
        _plot_hand_magnitude(
            axes[2, col], s.active,
            ylim=mag_ylim,
            color=config.HAND_COLORS["dom"] if s.active.hand_type == "dom" else config.HAND_COLORS["ndom"],
            title=f"|a| {active_label}",
        )
        _plot_hand_magnitude(
            axes[3, col], s.still,
            ylim=mag_ylim,
            color=config.HAND_COLORS["dom"] if s.still.hand_type == "dom" else config.HAND_COLORS["ndom"],
            title=f"|a| {still_label}",
        )
        axes[3, col].set_xlabel("time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(group: str | None = None) -> pd.DataFrame:
    """Generate plots + summary for both groups (or just one if `group` set)."""
    groups = (group,) if group else config.GROUPS
    summaries: list[SessionSummary] = []
    fig_root = config.FIG_DIR / "sprint_00"

    for grp in groups:
        ids = list_patient_ids(grp)
        for pid in ids:
            p = load_patient(grp, pid)
            out_png = fig_root / grp / f"{pid}.png"
            plot_patient(p, out_png)
            for lbl in config.SESSION_LABELS:
                ss = _summarize_session(p, lbl)
                if ss is not None:
                    summaries.append(ss)
            print(f"  [{grp}] {pid} -> {out_png.relative_to(config.PKG_DIR)}")

    df = pd.DataFrame([asdict(s) for s in summaries])
    out_csv = fig_root / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv.relative_to(config.PKG_DIR)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 0 EDA")
    parser.add_argument("--group", choices=config.GROUPS, default=None,
                        help="restrict to one group; default = both")
    args = parser.parse_args()
    run(args.group)


if __name__ == "__main__":
    main()
