"""
Sprint 2 demo, apply the artifact filter (chosen config) to every UCP
patient, generate per-patient figures and a summary CSV.

Per patient, one PNG with 4 rows × 2 cols (sessions dom-active and
ndom-active):
    row 1 : STILL ENMO with outlier-window time spans shaded red
    row 2 : STILL jerk magnitude with same spans
    row 3 : per-window indicator heatmap (n_windows × 4 indicators) at
            the canonical 1 s scale, normalised by intra-patient median +
            7 * MAD so the colour scale is interpretable
    row 4 : boxplot of each indicator on calm vs outlier windows
            (1 s scale)

The chosen configuration comes from `eda/artifact_ablation_run.py`
(k=7, min_n_calm=100, scales {0.5, 1, 2} s).

A flat `summary_artifact.csv` lists per (patient, session, scale):
    n_total, n_calm, n_outlier, pct_outlier, k_used,
    plus the threshold (median + k*MAD) and number of iterations for each
    of the four indicators.

A long-form `baseline_quantiles.csv` lists the intra-patient quantile
profile of each indicator on the calm pool, ready to be consumed by
Sprint 3 features and Sprint 4 detectors.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from artifact.baseline import (
    baseline_to_dataframe,
    compute_indicator_baseline,
)
from artifact.filter import (
    FilterConfig,
    PatientArtifactReport,
    detect_artifacts_patient,
)
from artifact.indicators import INDICATOR_NAMES
from data_io.data_loader import list_patient_ids, load_patient
from preprocessing.pipeline import preprocess_patient


# Chosen configuration (frozen here so the script is reproducible without
# re-running the ablation; mirrors `chosen_config.txt`).
# k=7 chosen for conservative outlier flagging while preserving UCP4 jerk
# spike. overlap=0.50 chosen over 0.75 because both produce nearly identical
# outlier rates (median |diff| = 0.43 pp at k=7) but 50% gives the robust
# MAD estimator more independent samples and halves the downstream cost.
CHOSEN_CFG = FilterConfig(
    k=7.0,
    max_iter=5,
    scales_s=(0.5, 1.0, 2.0),
    overlap=0.50,
    min_n_calm=50,
    relax_k=10.0,
)


# Reference scale used for the heatmap and boxplot rows in the figure.
REFERENCE_SCALE_S: float = 1.0


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _outlier_time_spans(scale_res, fs: float) -> list[tuple[float, float]]:
    """Convert window-level outlier mask into (t_start, t_end) spans."""
    win_len = int(scale_res.scale_s * fs)
    starts = scale_res.starts_idx
    out_idx = np.where(scale_res.mask_outlier_any)[0]
    return [(starts[i] / fs, (starts[i] + win_len) / fs) for i in out_idx]


def _shade_spans(ax, spans, color="red", alpha=0.15):
    for t0, t1 in spans:
        ax.axvspan(t0, t1, color=color, alpha=alpha)


def _heatmap_indicators(ax, scale_res, k_used, title):
    """Heatmap of the 4 indicators per window, normalised so colour 1.0
    matches the iterative-threshold value of each indicator.
    """
    df = scale_res.indicators
    norm = np.zeros(df.shape, dtype=float)
    for j, name in enumerate(INDICATOR_NAMES):
        tr = scale_res.threshold_per_indicator[name]
        denom = max(tr.threshold, 1e-12)
        norm[:, j] = df[name].to_numpy() / denom
    im = ax.imshow(norm.T, aspect="auto", cmap="magma",
                   vmin=0.0, vmax=2.0, interpolation="nearest")
    ax.set_yticks(range(len(INDICATOR_NAMES)))
    ax.set_yticklabels(INDICATOR_NAMES, fontsize=7)
    ax.set_xlabel("window index")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.025,
                 label="value / threshold (1.0 = threshold)")


def _boxplot_calm_vs_out(ax, scale_res, title):
    df = scale_res.indicators
    out_mask = scale_res.mask_outlier_any
    data = []
    labels = []
    for name in INDICATOR_NAMES:
        data.append(df.loc[~out_mask, name].to_numpy())
        labels.append(f"{name}\ncalm")
        data.append(df.loc[out_mask, name].to_numpy())
        labels.append(f"{name}\nout")
    bp = ax.boxplot(data, labels=labels, showfliers=False,
                    patch_artist=True)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor("#7fcdbb" if "calm" in lab else "#fb6a4a")
        patch.set_alpha(0.6)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)


# -----------------------------------------------------------------------------
# Per-patient figure
# -----------------------------------------------------------------------------
def plot_patient_artifacts(p, proc, report: PatientArtifactReport,
                           out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(15.0, 12.0),
        dpi=config.PLOT_DPI,
    )
    fig.suptitle(
        f"{p.patient_id} | artifact filter | k={CHOSEN_CFG.k}, "
        f"min_n_calm={CHOSEN_CFG.min_n_calm}, scales="
        f"{list(CHOSEN_CFG.scales_s)} s",
        fontsize=11,
    )

    fs = p.fs
    for col, sess_label in enumerate(config.SESSION_LABELS):
        if sess_label not in proc:
            for r in range(4):
                axes[r, col].text(0.5, 0.5, f"no session '{sess_label}'",
                                  ha="center", va="center",
                                  transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue

        ps = proc[sess_label]
        s = ps.still
        sess_res = report.per_session[sess_label]
        ref_res = sess_res.per_scale[REFERENCE_SCALE_S]

        # row 1 - ENMO with outlier spans at all scales overlaid (lightest at
        # the smallest scale, darkest at the largest).
        ax = axes[0, col]
        ax.plot(s.t, s.enmo, color="#1f77b4", linewidth=0.6)
        for alpha_, scale_s in zip((0.10, 0.18, 0.26), CHOSEN_CFG.scales_s):
            _shade_spans(ax, _outlier_time_spans(
                sess_res.per_scale[scale_s], fs), alpha=alpha_)
        ax.set_xlim(config.PLOT_TIME_XLIM)
        ax.set_ylim(config.PLOT_ENMO_YLIM)
        ax.set_title(f"STILL ENMO + outlier spans, {sess_label}-active",
                     fontsize=9)
        ax.set_ylabel("ENMO (g)")
        ax.grid(alpha=0.25)

        # row 2 - jerk magnitude with same spans
        ax = axes[1, col]
        ax.plot(s.t, s.jerk_mag, color="#d62728", linewidth=0.6)
        for alpha_, scale_s in zip((0.10, 0.18, 0.26), CHOSEN_CFG.scales_s):
            _shade_spans(ax, _outlier_time_spans(
                sess_res.per_scale[scale_s], fs), alpha=alpha_)
        ax.set_xlim(config.PLOT_TIME_XLIM)
        ax.set_ylim(config.PLOT_JERK_YLIM)
        ax.set_title(f"STILL |jerk| + outlier spans, {sess_label}-active",
                     fontsize=9)
        ax.set_ylabel("|jerk| (g/s)")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)

        # row 3 - heatmap of indicators @ reference scale
        _heatmap_indicators(
            axes[2, col], ref_res, ref_res.k_used,
            title=(f"Indicator heatmap @ {REFERENCE_SCALE_S}s, "
                   f"k_used={ref_res.k_used:.1f}, "
                   f"calm={int((~ref_res.mask_outlier_any).sum())}/"
                   f"{ref_res.n_windows}"),
        )

        # row 4 - calm vs outlier boxplots @ reference scale
        _boxplot_calm_vs_out(
            axes[3, col], ref_res,
            title=f"calm vs outlier indicators @ {REFERENCE_SCALE_S}s",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(group: str = "ucp") -> tuple[pd.DataFrame, pd.DataFrame]:
    fig_root = config.FIG_DIR / "sprint_02"

    summary_rows: list[dict] = []
    baselines_all: list = []

    for pid in list_patient_ids(group):
        p = load_patient(group, pid)
        proc = preprocess_patient(p.sessions, p.fs)
        rep = detect_artifacts_patient(p, proc, CHOSEN_CFG)

        out_png = fig_root / group / f"{pid}.png"
        plot_patient_artifacts(p, proc, rep, out_png)
        print(f"  [{group}] {pid} -> {out_png.relative_to(config.PKG_DIR)}")

        for sess_label, sess_res in rep.per_session.items():
            for scale_s, scale_res in sess_res.per_scale.items():
                row = {
                    "patient_id": pid,
                    "group": group,
                    "session": sess_label,
                    "scale_s": float(scale_s),
                    "n_total": scale_res.n_windows,
                    "n_outlier": int(scale_res.mask_outlier_any.sum()),
                    "n_calm": int((~scale_res.mask_outlier_any).sum()),
                    "k_used": scale_res.k_used,
                    "pct_outlier": 100.0 * scale_res.mask_outlier_any.mean(),
                }
                for name in INDICATOR_NAMES:
                    tr = scale_res.threshold_per_indicator[name]
                    row[f"{name}_thr"] = tr.threshold
                    row[f"{name}_median"] = tr.median
                    row[f"{name}_mad"] = tr.mad_value
                    row[f"{name}_n_iter"] = tr.n_iter
                summary_rows.append(row)

        # Add group tag to each IndicatorBaseline row so cross-group CSV is unambiguous.
        bls = compute_indicator_baseline(rep)
        for b in bls:
            baselines_all.append((group, b))

    df_sum = pd.DataFrame(summary_rows)

    # Wrap baselines so the long-form CSV has a 'group' column.
    bl_rows = []
    for grp_tag, b in baselines_all:
        row = {
            "patient_id": b.patient_id,
            "group": grp_tag,
            "session": b.session_label,
            "scale_s": b.scale_s,
            "indicator": b.indicator,
            "n_calm": b.n_calm,
        }
        for q, v in b.q.items():
            row[f"q{int(q * 100)}"] = v
        bl_rows.append(row)
    df_bl = pd.DataFrame(bl_rows)
    return df_sum, df_bl


def main() -> None:
    fig_root = config.FIG_DIR / "sprint_02"

    sums = []
    bls = []
    for grp in ("ucp", "td"):
        s, b = run(grp)
        sums.append(s)
        bls.append(b)

    df_sum = pd.concat(sums, ignore_index=True)
    out_csv = fig_root / "summary_artifact.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv.relative_to(config.PKG_DIR)}")

    df_bl = pd.concat(bls, ignore_index=True)
    out_bl = fig_root / "baseline_quantiles.csv"
    df_bl.to_csv(out_bl, index=False)
    print(f"Baseline quantiles: {out_bl.relative_to(config.PKG_DIR)}")


if __name__ == "__main__":
    main()
