"""
Sprint 1 demo, apply the dual-path preprocessing to every patient and produce
a per-patient figure that shows, for each session (dom-active and ndom-active):

    row 1 : ACTIVE hand raw XYZ (X faded)
    row 2 : STILL  hand raw XYZ (X faded)              <-- target for MM
    row 3 : STILL  hand ENMO (gravity-free)
    row 4 : STILL  hand Y, Z band-passed [0.5, 15] Hz
    row 5 : STILL  hand jerk magnitude on band-passed YZ

All scales are FIXED, taken from `explainable.config`, so plots are directly
comparable across patients (see memory directive in MEMORY.md).

A sanity check is run first: a synthetic signal made of three sinusoids at
0.2 Hz, 5 Hz and 25 Hz is filtered through `bandpass(0.5, 15)`; the in-band
component (5 Hz) must be preserved and the out-of-band ones (0.2 Hz, 25 Hz)
must be strongly attenuated. The script aborts if this check fails.

Run:
    python -m eda.preprocess_demo
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data_io.data_loader import (
    list_patient_ids,
    load_patient,
    PatientData,
)
from preprocessing.filters import bandpass
from preprocessing.pipeline import (
    ProcessedHand,
    ProcessedSession,
    preprocess_patient,
)


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
def sanity_check_bandpass(fs: int = config.SAMPLE_RATE, dur_s: int = 10) -> None:
    """Verify that the band-pass preserves 5 Hz and attenuates 0.2 / 25 Hz."""
    t = np.arange(int(fs * dur_s)) / fs
    in_band = np.sin(2 * np.pi * 5.0 * t)
    low_oob = np.sin(2 * np.pi * 0.2 * t)
    high_oob = np.sin(2 * np.pi * 25.0 * t)
    signal = in_band + low_oob + high_oob

    filtered = bandpass(signal, fs=fs)
    rms_total = float(np.sqrt(np.mean(filtered ** 2)))
    # Compare against the rms of pure 5 Hz (expected ~ 1/sqrt(2)).
    expected_rms = float(np.sqrt(np.mean(in_band ** 2)))
    ratio = rms_total / expected_rms

    if not (0.90 <= ratio <= 1.10):
        raise RuntimeError(
            f"bandpass sanity failed: rms ratio {ratio:.3f} not in [0.9, 1.1]"
        )

    # Spot-check the out-of-band attenuation by filtering each component
    # separately.
    f_low = bandpass(low_oob, fs=fs)
    f_high = bandpass(high_oob, fs=fs)
    att_low = float(np.sqrt(np.mean(f_low ** 2)) /
                    np.sqrt(np.mean(low_oob ** 2)))
    att_high = float(np.sqrt(np.mean(f_high ** 2)) /
                     np.sqrt(np.mean(high_oob ** 2)))
    # Out-of-band tolerance is 0.20 (~14 dB): Butterworth order 4 has a
    # gradual transition band so 0.2 Hz and 25 Hz components are not
    # vanishingly small even though they are clearly outside [0.5, 15] Hz.
    if att_low > 0.20:
        raise RuntimeError(
            f"bandpass sanity failed: 0.2 Hz attenuation only {att_low:.3f}"
        )
    if att_high > 0.20:
        raise RuntimeError(
            f"bandpass sanity failed: 25 Hz attenuation only {att_high:.3f}"
        )

    print(f"  [sanity] bandpass OK | in-band ratio={ratio:.3f}, "
          f"0.2 Hz att={att_low:.4f}, 25 Hz att={att_high:.4f}")


# -----------------------------------------------------------------------------
# Per-patient summary
# -----------------------------------------------------------------------------
@dataclass
class SessionSummary:
    patient_id: str
    group: str
    session: str
    fs_hz: float
    # STILL hand only (target for MM)
    still_hand_type: str
    enmo_mean_g: float
    enmo_rms_g: float
    enmo_peak_g: float
    enmo_bp_std_g: float
    yz_bp_peak_g: float
    yz_bp_rms_g: float
    jerk_mag_mean_gps: float
    jerk_mag_peak_gps: float
    dominant_freq_hz: float       # peak frequency of |FFT(enmo_bp)|


def _dominant_frequency(signal: np.ndarray, fs: float,
                        f_low: float, f_high: float) -> float:
    """Frequency of the largest |FFT| bin within [f_low, f_high]."""
    n = len(signal)
    if n < 4:
        return float("nan")
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(signal - signal.mean()))
    band = (freqs >= f_low) & (freqs <= f_high)
    if not band.any():
        return float("nan")
    sub_freqs = freqs[band]
    sub_mag = mag[band]
    return float(sub_freqs[int(np.argmax(sub_mag))])


def _summarize_session(p: PatientData, ps: ProcessedSession) -> SessionSummary:
    s: ProcessedHand = ps.still
    return SessionSummary(
        patient_id=p.patient_id,
        group=p.group,
        session=ps.session_label,
        fs_hz=p.fs,
        still_hand_type=s.hand_type,
        enmo_mean_g=float(s.enmo.mean()),
        enmo_rms_g=float(np.sqrt(np.mean(s.enmo ** 2))),
        enmo_peak_g=float(s.enmo.max()),
        enmo_bp_std_g=float(s.enmo_bp.std()),
        yz_bp_peak_g=float(np.max(np.abs(s.yz_bp))),
        yz_bp_rms_g=float(np.sqrt(np.mean(s.yz_bp ** 2))),
        jerk_mag_mean_gps=float(s.jerk_mag.mean()),
        jerk_mag_peak_gps=float(s.jerk_mag.max()),
        dominant_freq_hz=_dominant_frequency(
            s.enmo_bp, p.fs, config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ
        ),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _plot_axes_with_x_faded(ax, t, accel, ylim, title):
    for axis_name, idx in (("x", 0), ("y", 1), ("z", 2)):
        alpha = config.PLOT_X_ALPHA if axis_name == "x" else config.PLOT_YZ_ALPHA
        lw = 0.4 if axis_name == "x" else 0.7
        ax.plot(t, accel[:, idx], color=config.AXIS_COLORS[axis_name],
                linewidth=lw, alpha=alpha, label=axis_name.upper())
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("g")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)


def _plot_scalar(ax, t, y, ylim, title, color, ylabel):
    ax.plot(t, y, color=color, linewidth=0.7)
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _plot_yz_bp(ax, t, yz_bp, ylim, title):
    for axis_name, idx in (("y", 0), ("z", 1)):
        ax.plot(t, yz_bp[:, idx], color=config.AXIS_COLORS[axis_name],
                linewidth=0.7, alpha=config.PLOT_YZ_ALPHA,
                label=axis_name.upper())
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("g (BP)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def plot_patient_processed(p: PatientData,
                           proc: dict[str, ProcessedSession],
                           out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=5, ncols=2,
        figsize=(14.0, 11.0),
        dpi=config.PLOT_DPI,
        sharex=True,
    )
    fig.suptitle(
        f"{p.patient_id} | group={p.group.upper()} | dominance={p.hand_dominance} "
        f"| fs={p.fs:.1f} Hz | BP [{config.BANDPASS_LOW_HZ}, "
        f"{config.BANDPASS_HIGH_HZ}] Hz | X axis faded (alpha={config.PLOT_X_ALPHA})",
        fontsize=11,
    )

    for col, sess_label in enumerate(config.SESSION_LABELS):
        if sess_label not in proc:
            for r in range(5):
                axes[r, col].text(0.5, 0.5, f"no session '{sess_label}'",
                                  ha="center", va="center",
                                  transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue

        ps = proc[sess_label]
        a, s = ps.active, ps.still
        sess_tag = f"session {sess_label}-active"
        still_color = (config.HAND_COLORS["dom"]
                       if s.hand_type == "dom" else config.HAND_COLORS["ndom"])

        _plot_axes_with_x_faded(
            axes[0, col], a.t, a.raw,
            ylim=config.PLOT_ACCEL_YLIM,
            title=f"ACTIVE hand ({a.hand_type}) raw XYZ, {sess_tag}",
        )
        _plot_axes_with_x_faded(
            axes[1, col], s.t, s.raw,
            ylim=config.PLOT_ACCEL_YLIM_STILL,
            title=f"STILL hand ({s.hand_type}) raw XYZ, {sess_tag}",
        )
        _plot_scalar(
            axes[2, col], s.t, s.enmo,
            ylim=config.PLOT_ENMO_YLIM,
            title=f"STILL ENMO, {sess_tag}",
            color=still_color, ylabel="ENMO (g)",
        )
        _plot_yz_bp(
            axes[3, col], s.t, s.yz_bp,
            ylim=config.PLOT_YZ_FILTERED_YLIM,
            title=f"STILL YZ band-passed, {sess_tag}",
        )
        _plot_scalar(
            axes[4, col], s.t, s.jerk_mag,
            ylim=config.PLOT_JERK_YLIM,
            title=f"STILL jerk magnitude (on BP YZ), {sess_tag}",
            color=still_color, ylabel="|jerk| (g/s)",
        )
        axes[4, col].set_xlabel("time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(group: str | None = None) -> pd.DataFrame:
    sanity_check_bandpass()

    groups = (group,) if group else config.GROUPS
    summaries: list[SessionSummary] = []
    fig_root = config.FIG_DIR / "sprint_01"

    for grp in groups:
        ids = list_patient_ids(grp)
        for pid in ids:
            p = load_patient(grp, pid)
            proc = preprocess_patient(p.sessions, p.fs)
            out_png = fig_root / grp / f"{pid}.png"
            plot_patient_processed(p, proc, out_png)
            for lbl, ps in proc.items():
                summaries.append(_summarize_session(p, ps))
            print(f"  [{grp}] {pid} -> {out_png.relative_to(config.PKG_DIR)}")

    df = pd.DataFrame([asdict(s) for s in summaries])
    out_csv = fig_root / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv.relative_to(config.PKG_DIR)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 1 preprocessing demo")
    parser.add_argument("--group", choices=config.GROUPS, default=None,
                        help="restrict to one group; default = both")
    args = parser.parse_args()
    try:
        run(args.group)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
