"""
Sprint 3 demo, compute every feature family on UCP and TD; visualise.

For every patient (UCP + TD) one PNG with the following layout:

    row 1 : Family 1 (activity), boxplots calm vs outlier (one box per
            feature) for both sessions
    row 2 : Family 2 (spectral), same layout, spectral features
    row 3 : Family 3 (bilateral mirror score), same layout
    row 4 : Family 4 (temporal distribution), bars for the per-session
            scalar features

A single aggregated PNG `cross_patient_temporal.png` shows for every patient
the four temporal indicators that the Sprint 2 evidence flagged as the most
likely discriminators:

    gini_temporal, burstiness_B, temporal_entropy, n_bursts

A `summary_features.csv` lists one row per (patient, session) with all
Family-4 scalars plus mean values of Families 1–3 over calm and outlier
windows separately.

A sanity check is run before the patient loop: a synthetic bilateral signal
with a known integer-sample lag must be detected within ±1 sample by the
`xcorr_max_dom_ndom` feature.

Run:
    python -m eda.feature_demo
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from artifact.filter import detect_artifacts_patient
from eda.artifact_demo import CHOSEN_CFG
from features.bilateral import xcorr_max_dom_ndom
from features.extractor import (
    SessionFeatures,
    extract_patient_features,
    temporal_to_dataframe,
)
from features.registry import REGISTRY, enabled_features
from features.temporal import TEMPORAL_FEATURE_NAMES
from data_io.data_loader import list_patient_ids, load_patient
from preprocessing.pipeline import preprocess_patient


# Per-family ordering used in the boxplot rows.
FAMILY_ORDER: dict[str, list[str]] = {
    "activity": [n for n, (fam, _) in REGISTRY.items() if fam == "activity"],
    "spectral": [n for n, (fam, _) in REGISTRY.items() if fam == "spectral"],
    "bilateral": [n for n, (fam, _) in REGISTRY.items() if fam == "bilateral"],
}


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
def sanity_check_bilateral(fs: int = config.SAMPLE_RATE,
                           dur_s: int = 5,
                           true_lag_samples: int = 10) -> None:
    """Aperiodic Gaussian bump + a delayed copy.

    A sinusoid would be periodic and the cross-correlation could lock onto
    a different lag modulo the period. We use a non-periodic test signal
    (single Gaussian pulse) so that the lag at the maximum is unambiguous.

    `xcorr_max_dom_ndom` must report (a) correlation close to 1 and (b) a
    lag within ±1 sample of `|true_lag_samples| * 1000 / fs` milliseconds.
    """
    t = np.arange(int(fs * dur_s)) / fs
    centre = dur_s / 2.0
    sigma = 0.05
    active = np.exp(-((t - centre) ** 2) / (2 * sigma ** 2))
    still = np.roll(active, true_lag_samples)
    corr, lag_ms = xcorr_max_dom_ndom(active, still, fs)
    expected_abs_ms = abs(true_lag_samples) * 1000.0 / fs
    tol_ms = 1000.0 / fs + 1e-6
    if corr < 0.95:
        raise RuntimeError(
            f"bilateral sanity failed: xcorr={corr:.3f}, expected > 0.95"
        )
    if abs(abs(lag_ms) - expected_abs_ms) > tol_ms:
        raise RuntimeError(
            f"bilateral sanity failed: lag_ms={lag_ms:.2f}, "
            f"expected magnitude ~{expected_abs_ms:.2f}"
        )
    print(f"  [sanity] bilateral OK | corr={corr:.3f}, "
          f"lag={lag_ms:.2f} ms (expected magnitude {expected_abs_ms:.2f})")


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _boxplot_family(ax, sf: SessionFeatures, family: str) -> None:
    names = FAMILY_ORDER[family]
    df = sf.df_windows
    data = []
    labels = []
    for name in names:
        calm = df.loc[~df.is_outlier, name].to_numpy()
        out = df.loc[df.is_outlier, name].to_numpy()
        data.append(calm)
        labels.append(f"{name}\ncalm")
        data.append(out)
        labels.append(f"{name}\nout")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                    patch_artist=True)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor("#7fcdbb" if "calm" in lab else "#fb6a4a")
        patch.set_alpha(0.6)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.set_title(f"{family}, {sf.session_label}-active", fontsize=9)
    ax.grid(alpha=0.25)


def _bars_temporal(ax, sf: SessionFeatures) -> None:
    names = list(TEMPORAL_FEATURE_NAMES)
    values = [sf.temporal[n] for n in names]
    bars = ax.bar(range(len(names)), values, color="#5a9bd5", alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
    ax.set_title(f"temporal, {sf.session_label}-active", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    # Annotate each bar with its value.
    for b, v in zip(bars, values):
        if np.isnan(v):
            txt = "nan"
        elif abs(v) >= 100:
            txt = f"{v:.0f}"
        elif abs(v) >= 1:
            txt = f"{v:.2f}"
        else:
            txt = f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                ha="center", va="bottom", fontsize=6)


def plot_patient_features(features: dict[str, SessionFeatures],
                          patient_id: str, group: str,
                          out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(16.0, 13.5),
        dpi=config.PLOT_DPI,
    )
    fig.suptitle(
        f"{patient_id} | group={group.upper()} | feature families 1–4 "
        f"(scale {list(features.values())[0].scale_s}s, overlap "
        f"{list(features.values())[0].overlap})",
        fontsize=11,
    )

    for col, sess_label in enumerate(config.SESSION_LABELS):
        if sess_label not in features:
            for r in range(4):
                axes[r, col].text(0.5, 0.5, f"no session '{sess_label}'",
                                  ha="center", va="center",
                                  transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue
        sf = features[sess_label]
        _boxplot_family(axes[0, col], sf, "activity")
        _boxplot_family(axes[1, col], sf, "spectral")
        _boxplot_family(axes[2, col], sf, "bilateral")
        _bars_temporal(axes[3, col], sf)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_patient_temporal(df_temporal: pd.DataFrame, out_path: Path) -> None:
    """Aggregate bars of the four most discriminative temporal indicators."""
    indicators = ["gini_temporal", "burstiness_B", "temporal_entropy", "n_bursts"]
    ylims = {
        "gini_temporal":    config.PLOT_GINI_YLIM,
        "burstiness_B":     config.PLOT_BURSTINESS_YLIM,
        "temporal_entropy": config.PLOT_TEMP_ENTROPY_YLIM,
        "n_bursts":         config.PLOT_NBURSTS_YLIM,
    }
    sessions = ["dom", "ndom"]
    fig, axes = plt.subplots(
        nrows=len(indicators), ncols=len(sessions),
        figsize=(16.0, 12.0), dpi=config.PLOT_DPI, sharex=False,
    )
    fig.suptitle("Cross-patient temporal-distribution indicators (UCP vs TD)",
                 fontsize=12)

    for r, ind in enumerate(indicators):
        for c, sess in enumerate(sessions):
            ax = axes[r, c]
            sub = df_temporal[df_temporal.session == sess].copy()
            sub = sub.sort_values(["group", "patient_id"])
            colors = [config.PLOT_GROUP_COLORS[g] for g in sub.group]
            x = np.arange(len(sub))
            ax.bar(x, sub[ind].to_numpy(), color=colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["patient_id"].to_list(), rotation=90,
                               fontsize=6)
            ax.set_ylim(ylims[ind])
            ax.set_ylabel(ind, fontsize=8)
            ax.set_title(f"{ind}, {sess}-active", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["ucp"]),
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["td"]),
    ]
    fig.legend(handles, ["UCP", "TD"], loc="lower center", ncol=2,
               fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(group: str) -> tuple[pd.DataFrame, list[SessionFeatures]]:
    fig_root = config.FIG_DIR / "sprint_03"
    all_sfs: list[SessionFeatures] = []
    summary_rows: list[dict] = []

    for pid in list_patient_ids(group):
        p = load_patient(group, pid)
        proc = preprocess_patient(p.sessions, p.fs)
        rep = detect_artifacts_patient(p, proc, CHOSEN_CFG)
        feats = extract_patient_features(
            proc, rep, patient_id=p.patient_id, group=p.group,
        )
        out_png = fig_root / group / f"{pid}.png"
        plot_patient_features(feats, p.patient_id, p.group, out_png)
        print(f"  [{group}] {pid} -> {out_png.relative_to(config.PKG_DIR)}")

        for sess_label, sf in feats.items():
            all_sfs.append(sf)
            row = {
                "patient_id": p.patient_id,
                "group": p.group,
                "session": sess_label,
                "n_windows": len(sf.df_windows),
                "n_outlier": int(sf.df_windows.is_outlier.sum()),
            }
            # Mean of each per-window feature, calm and outlier separately.
            df = sf.df_windows
            calm_df = df.loc[~df.is_outlier]
            out_df = df.loc[df.is_outlier]
            for name in enabled_features():
                row[f"{name}_mean_calm"] = (
                    float(calm_df[name].mean()) if len(calm_df) else float("nan")
                )
                row[f"{name}_mean_out"] = (
                    float(out_df[name].mean()) if len(out_df) else float("nan")
                )
            # Family 4 (per session) flat keys.
            for name in TEMPORAL_FEATURE_NAMES:
                row[name] = sf.temporal[name]
            summary_rows.append(row)

    return pd.DataFrame(summary_rows), all_sfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3 feature demo")
    parser.add_argument("--group", choices=("ucp", "td", "both"),
                        default="both")
    args = parser.parse_args()

    try:
        sanity_check_bilateral()
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    groups = ("ucp", "td") if args.group == "both" else (args.group,)
    sums = []
    all_sfs: list[SessionFeatures] = []
    for grp in groups:
        s, sfs = run(grp)
        sums.append(s)
        all_sfs.extend(sfs)

    fig_root = config.FIG_DIR / "sprint_03"
    fig_root.mkdir(parents=True, exist_ok=True)
    df_sum = pd.concat(sums, ignore_index=True)
    out_csv = fig_root / "summary_features.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv.relative_to(config.PKG_DIR)}")

    df_temporal = temporal_to_dataframe(all_sfs)
    out_tmp = fig_root / "temporal_features.csv"
    df_temporal.to_csv(out_tmp, index=False)
    print(f"Temporal long-form: {out_tmp.relative_to(config.PKG_DIR)}")

    out_png = fig_root / "cross_patient_temporal.png"
    plot_cross_patient_temporal(df_temporal, out_png)
    print(f"Cross-patient plot: {out_png.relative_to(config.PKG_DIR)}")


if __name__ == "__main__":
    main()
