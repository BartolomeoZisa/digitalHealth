"""
Sprint 4 demo, fit the 3 detectors on every patient (UCP + TD), score every
window, build per-patient figures and cross-patient summaries.

For each patient one PNG with 4 rows × 2 cols (one column per session):

    row 1 : ensemble timeline (score_median) + threshold line + is_artifact
            shading; top-5 events labelled with their dominant attribution
    row 2 : the three individual detector scores overlaid (agreement check)
    row 3 : asymmetry_index vs xcorr_max scatter, colour = score_median;
            artefact windows ringed in red
    row 4 : bar chart of the top-5 events for that session, bar height is
            the score, annotation is the attribution headlines

The plot uses the **selected** feature set by default because Sprint 3
showed it gives clinically meaningful attributions; both feature sets are
written to `summary_detector.csv` and compared via Spearman in
`agreement_full_vs_selected.csv`.

Cross-patient outputs:
    cross_patient_score_distribution.png, boxplots of score_median, UCP vs
        TD, per session, per feature set.

Run:
    python -m eda.detector_demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from artifact.filter import detect_artifacts_patient
from detectors.pipeline import (
    PatientDetectorOutput,
    run_patient_detectors,
)
from eda.artifact_demo import CHOSEN_CFG
from explain.attribution import explain_top_events
from explain.feature_glossary import headline
from features.extractor import extract_patient_features
from data_io.data_loader import list_patient_ids, load_patient
from preprocessing.pipeline import preprocess_patient


PRIMARY_FSET: str = "selected"


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
def _sanity_check() -> None:
    """Build a synthetic 100-window calm pool, then score one MM-like window.

    With `asymmetry_index = 0` (still hand fully participates) and high
    `xcorr_max`, the quantile detector must score the window > 0.5.
    """
    from detectors.robust_quantile import RobustQuantileDetector
    names = ("asymmetry_index", "xcorr_max", "enmo_peak")
    rng = np.random.default_rng(config.RANDOM_SEED)
    calm = np.column_stack([
        rng.normal(0.9, 0.05, size=100),    # asym ~ 0.9 (active dominates)
        rng.normal(0.3, 0.05, size=100),    # xcorr_max ~ 0.3
        rng.normal(0.05, 0.01, size=100),   # enmo_peak small
    ])
    det = RobustQuantileDetector(names)
    det.fit(calm)
    suspect = np.array([[0.0, 0.7, 0.2]])   # asym ≈ 0, xcorr high, peak high
    s = det.score(suspect)[0]
    if s < 0.5:
        raise RuntimeError(f"detector sanity failed: score={s:.3f} < 0.5")
    print(f"  [sanity] detector OK | score on MM-like sample = {s:.3f}")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _shade_artifact(ax, t_start_s: np.ndarray, is_artifact: np.ndarray,
                    win_s: float) -> None:
    for t, flag in zip(t_start_s, is_artifact):
        if flag:
            ax.axvspan(t, t + win_s, color="#bdbdbd", alpha=0.25)


def _shade_boundary(ax) -> None:
    """Light yellow shading on the first and last `BOUNDARY_TRIM_S` seconds."""
    trim = config.BOUNDARY_TRIM_S
    duration = config.SESSION_DURATION_S
    ax.axvspan(0.0, trim, color="#fff7bc", alpha=0.45)
    ax.axvspan(duration - trim, duration, color="#fff7bc", alpha=0.45)


def _row_score_timeline(ax, df: pd.DataFrame, win_s: float, title: str) -> None:
    _shade_boundary(ax)
    _shade_artifact(ax, df.t_start_s.to_numpy(),
                    df.is_artifact.to_numpy(), win_s)
    ax.plot(df.t_start_s, df.score_median, color="#c0392b", linewidth=1.2,
            label="score_median")
    # Highlight MM candidates as green vertical markers on the timeline.
    mm_t = df.loc[df.is_mm_candidate, "t_start_s"].to_numpy()
    for t in mm_t:
        ax.axvspan(t, t + win_s, color="#2ecc71", alpha=0.35)
    ax.axhline(config.ENSEMBLE_THRESHOLD, color="black",
               linestyle="--", linewidth=0.8, label="threshold")
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)


def _row_detector_overlay(ax, df: pd.DataFrame, win_s: float, title: str) -> None:
    _shade_boundary(ax)
    _shade_artifact(ax, df.t_start_s.to_numpy(),
                    df.is_artifact.to_numpy(), win_s)
    ax.plot(df.t_start_s, df.score_quantile, color="#1f77b4",
            linewidth=0.8, alpha=0.85, label="quantile")
    ax.plot(df.t_start_s, df.score_iforest, color="#2ca02c",
            linewidth=0.8, alpha=0.85, label="iforest")
    ax.plot(df.t_start_s, df.score_pca, color="#9467bd",
            linewidth=0.8, alpha=0.85, label="pca")
    ax.axhline(config.ENSEMBLE_THRESHOLD, color="black",
               linestyle="--", linewidth=0.7)
    ax.set_xlim(config.PLOT_TIME_XLIM)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("score (per detector)")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)


def _row_scatter(ax, df: pd.DataFrame, title: str) -> None:
    if "asymmetry_index" not in df.columns or "xcorr_max" not in df.columns:
        ax.text(0.5, 0.5, "feature set without asymmetry/xcorr",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8)
        ax.set_axis_off()
        return
    sc = ax.scatter(
        df["asymmetry_index"], df["xcorr_max"],
        c=df["score_median"], cmap="magma", vmin=0.0, vmax=1.0,
        s=18, alpha=0.85,
    )
    ax.scatter(
        df.loc[df.is_artifact, "asymmetry_index"],
        df.loc[df.is_artifact, "xcorr_max"],
        facecolors="none", edgecolors="red", linewidths=0.8, s=42,
        label="is_artifact",
    )
    plt.colorbar(sc, ax=ax, fraction=0.045, label="score_median")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("asymmetry_index")
    ax.set_ylabel("xcorr_max")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7)


def _row_top_events(ax, df: pd.DataFrame, title: str) -> None:
    # Restrict to MM candidates so artefacts and boundary windows are excluded.
    mm_df = df[df.is_mm_candidate].sort_values(
        "score_median", ascending=False).head(5)
    if mm_df.empty:
        ax.text(0.5, 0.5, "no is_mm_candidate windows",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return
    from explain.attribution import parse_attribution_string, explain_event
    events = [
        explain_event(
            row.t_start_s, row.score_median, row.is_artifact,
            parse_attribution_string(row.attribution_top3),
        )
        for _, row in mm_df.iterrows()
    ]
    xs = np.arange(len(events))
    scores = [e.score_median for e in events]
    colors = ["#27ae60"] * len(events)  # all MM candidates -> green
    ax.bar(xs, scores, color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"t={e.t_start_s:.1f}s\n[MM cand.]" for e in events],
                       rotation=0, fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    for x, e in zip(xs, events):
        # Render top contribs as headlines + sign.
        lines = []
        for (name, z), _ in zip(e.contribs, e.sentences):
            sign = "↑" if z > 0 else "↓"
            lines.append(f"{sign} {headline(name)}")
        ax.text(x, 0.05, "\n".join(lines), ha="center", va="bottom",
                fontsize=6, rotation=0)


def plot_patient_detector(out: PatientDetectorOutput, win_s: float,
                          out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(16.0, 13.5),
        dpi=config.PLOT_DPI,
    )
    fig.suptitle(
        f"{out.patient_id} | group={out.group.upper()} | feature set = "
        f"{PRIMARY_FSET} | ensemble = median(quantile, iforest, pca) | "
        f"threshold = {config.ENSEMBLE_THRESHOLD}",
        fontsize=11,
    )
    for col, sess in enumerate(config.SESSION_LABELS):
        key = (sess, PRIMARY_FSET)
        if key not in out.per_session:
            for r in range(4):
                axes[r, col].text(0.5, 0.5, f"no session '{sess}'",
                                  ha="center", va="center",
                                  transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue
        sdo = out.per_session[key]
        df = sdo.df_windows
        _row_score_timeline(
            axes[0, col], df, win_s,
            title=f"score_median timeline, {sess}-active "
                  f"(n_calm={sdo.n_calm})",
        )
        _row_detector_overlay(
            axes[1, col], df, win_s,
            title=f"3-detector overlay, {sess}-active",
        )
        _row_scatter(
            axes[2, col], df,
            title=f"asymmetry × xcorr scatter, {sess}-active",
        )
        _row_top_events(
            axes[3, col], df,
            title=f"top-5 events, {sess}-active",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Cross-patient aggregation
# -----------------------------------------------------------------------------
def _cross_patient_mm_plot(df_sum: pd.DataFrame, out_path: Path) -> None:
    sessions = sorted(df_sum.session.unique())
    fig, axes = plt.subplots(1, len(sessions),
                             figsize=(8.0 * len(sessions), 5.0),
                             dpi=config.PLOT_DPI, sharey=True)
    if len(sessions) == 1:
        axes = [axes]
    for ax, sess in zip(axes, sessions):
        sub = df_sum[df_sum.session == sess].copy()
        sub = sub.sort_values(["group", "patient_id"])
        colors = [config.PLOT_GROUP_COLORS[g] for g in sub.group]
        xs = np.arange(len(sub))
        ax.bar(xs, sub.n_mm_candidate.to_numpy(), color=colors, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub.patient_id.to_list(), rotation=90, fontsize=6)
        ax.set_title(f"n_mm_candidate, session {sess}-active", fontsize=10)
        ax.set_ylabel("n_mm_candidate")
        ax.grid(alpha=0.25, axis="y")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["ucp"]),
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["td"]),
    ]
    fig.legend(handles, ["UCP", "TD"], loc="lower center", ncol=2,
               fontsize=10, frameon=False)
    fig.suptitle(
        f"Cross-patient MM candidate count (asym<={config.ASYM_MM_CUTOFF}, "
        f"xcorr>={config.XCORR_MM_CUTOFF}, score>={config.ENSEMBLE_THRESHOLD}, "
        f"no artifact, no boundary)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def _cross_patient_distribution_plot(rows_long: pd.DataFrame, out_path: Path) -> None:
    fsets = sorted(rows_long.feature_set.unique())
    sessions = sorted(rows_long.session.unique())
    fig, axes = plt.subplots(
        len(fsets), len(sessions),
        figsize=(6.0 * len(sessions), 4.0 * len(fsets)),
        dpi=config.PLOT_DPI, sharey=True,
    )
    if len(fsets) == 1:
        axes = np.array([axes])
    if len(sessions) == 1:
        axes = axes[:, None]
    for r, fset in enumerate(fsets):
        for c, sess in enumerate(sessions):
            ax = axes[r, c]
            sub = rows_long[(rows_long.feature_set == fset) &
                            (rows_long.session == sess)]
            data = [
                sub[sub.group == "ucp"].score_median.to_numpy(),
                sub[sub.group == "td"].score_median.to_numpy(),
            ]
            ax.boxplot(data, tick_labels=["UCP", "TD"], showfliers=False,
                       patch_artist=True)
            ax.set_title(f"fset={fset}, session={sess}", fontsize=9)
            ax.set_ylabel("score_median")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(alpha=0.25)
    fig.suptitle(
        "Cross-patient distribution of score_median (per window, UCP vs TD)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(group: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig_root = config.FIG_DIR / "sprint_04"
    summary_rows: list[dict] = []
    agreement_rows: list[dict] = []
    long_rows: list[dict] = []

    win_s = 1.0  # reference scale used in features.extractor

    for pid in list_patient_ids(group):
        p = load_patient(group, pid)
        proc = preprocess_patient(p.sessions, p.fs)
        rep = detect_artifacts_patient(p, proc, CHOSEN_CFG)
        feats = extract_patient_features(
            proc, rep, patient_id=p.patient_id, group=p.group,
        )
        features_by_session = {lbl: sf.df_windows for lbl, sf in feats.items()}
        starts_by_session = {lbl: sf.window_starts_idx for lbl, sf in feats.items()}

        out = run_patient_detectors(
            patient_id=p.patient_id, group=p.group,
            features_by_session=features_by_session,
            starts_by_session=starts_by_session, fs=p.fs,
        )

        out_png = fig_root / group / f"{pid}.png"
        plot_patient_detector(out, win_s=win_s, out_path=out_png)
        print(f"  [{group}] {pid} -> {out_png.relative_to(config.PKG_DIR)}")

        # Summary rows + Spearman full vs selected per session.
        per_session_scores: dict[tuple[str, str], np.ndarray] = {}
        for (sess_label, fset_name), sdo in out.per_session.items():
            df = sdo.df_windows
            n_high = int((df.score_median >= config.ENSEMBLE_THRESHOLD).sum())
            n_total = len(df)
            n_artifact_high = int(
                ((df.score_median >= config.ENSEMBLE_THRESHOLD) &
                 df.is_artifact).sum()
            )
            n_mm_like = n_high - n_artifact_high
            n_boundary_high = int(
                ((df.score_median >= config.ENSEMBLE_THRESHOLD) &
                 df.get("is_boundary", False)).sum()
            ) if "is_boundary" in df.columns else 0
            n_mm_candidate = int(df.is_mm_candidate.sum()) if "is_mm_candidate" in df.columns else 0
            summary_rows.append({
                "patient_id": pid,
                "group": group,
                "session": sess_label,
                "feature_set": fset_name,
                "n_total": n_total,
                "n_calm": sdo.n_calm,
                "n_high_score": n_high,
                "n_mm_like": n_mm_like,
                "n_artifact_high": n_artifact_high,
                "n_boundary_high": n_boundary_high,
                "n_mm_candidate": n_mm_candidate,
                "score_median_mean": float(df.score_median.mean()),
                "score_median_max": float(df.score_median.max()),
                "score_median_p95": float(df.score_median.quantile(0.95)),
            })
            for _, row in df.iterrows():
                long_rows.append({
                    "patient_id": pid,
                    "group": group,
                    "session": sess_label,
                    "feature_set": fset_name,
                    "t_start_s": row.t_start_s,
                    "is_artifact": bool(row.is_artifact),
                    "score_median": float(row.score_median),
                })
            per_session_scores[(sess_label, fset_name)] = (
                df.score_median.to_numpy()
            )

        # Spearman full vs selected, per session.
        for sess_label in config.SESSION_LABELS:
            if ((sess_label, "full") in per_session_scores and
                    (sess_label, "selected") in per_session_scores):
                a = per_session_scores[(sess_label, "full")]
                b = per_session_scores[(sess_label, "selected")]
                if len(a) == len(b) and len(a) > 2:
                    rho, pval = spearmanr(a, b)
                    agreement_rows.append({
                        "patient_id": pid,
                        "group": group,
                        "session": sess_label,
                        "spearman_rho": float(rho),
                        "spearman_p": float(pval),
                    })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(agreement_rows),
        pd.DataFrame(long_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 4 detector demo")
    parser.add_argument("--group", choices=("ucp", "td", "both"),
                        default="both")
    args = parser.parse_args()

    try:
        _sanity_check()
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    groups = ("ucp", "td") if args.group == "both" else (args.group,)
    sums, agrs, longs = [], [], []
    for grp in groups:
        s, a, l = run(grp)
        sums.append(s)
        agrs.append(a)
        longs.append(l)

    fig_root = config.FIG_DIR / "sprint_04"
    fig_root.mkdir(parents=True, exist_ok=True)
    df_sum = pd.concat(sums, ignore_index=True)
    df_agr = pd.concat(agrs, ignore_index=True)
    df_long = pd.concat(longs, ignore_index=True)

    df_sum.to_csv(fig_root / "summary_detector.csv", index=False)
    df_agr.to_csv(fig_root / "agreement_full_vs_selected.csv", index=False)
    print(f"\nSummary written: "
          f"{(fig_root / 'summary_detector.csv').relative_to(config.PKG_DIR)}")
    print(f"Agreement written: "
          f"{(fig_root / 'agreement_full_vs_selected.csv').relative_to(config.PKG_DIR)}")

    out_png = fig_root / "cross_patient_score_distribution.png"
    _cross_patient_distribution_plot(df_long, out_png)
    print(f"Cross-patient plot: {out_png.relative_to(config.PKG_DIR)}")

    # Sprint-5, cross-patient n_mm_candidate bar chart.
    mm_sub = df_sum[df_sum.feature_set == "selected"].copy()
    if not mm_sub.empty:
        mm_png = fig_root / "cross_patient_mm_candidate_count.png"
        _cross_patient_mm_plot(mm_sub, mm_png)
        print(f"MM-candidate plot: {mm_png.relative_to(config.PKG_DIR)}")
        mm_csv = fig_root / "summary_mm.csv"
        mm_sub[["patient_id", "group", "session", "n_total",
                "n_calm", "n_high_score", "n_artifact_high",
                "n_boundary_high", "n_mm_candidate"]].to_csv(mm_csv, index=False)
        print(f"MM summary: {mm_csv.relative_to(config.PKG_DIR)}")


if __name__ == "__main__":
    main()
