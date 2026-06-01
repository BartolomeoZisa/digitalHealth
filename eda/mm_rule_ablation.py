"""
Sprint 5 ablation, grid search over the composite MM-rule cutoffs.

For every (ENSEMBLE_THRESHOLD, ASYM_MM_CUTOFF, XCORR_MM_CUTOFF) triple we
recompute `is_mm_candidate` on the already-saved Sprint-4 score table and
report:

    sensitivity_ucp : fraction of UCP patients with >=1 mm_candidate on
                      either session
    specificity_td  : fraction of TD patients with 0 mm_candidate on both
                      sessions
    youden_J        : sensitivity + specificity - 1

A single PNG heatmap (asym, xcorr) is generated for the chosen
ENSEMBLE_THRESHOLD, and the configuration that maximises Youden J is
written to `chosen_thresholds.txt`.

Run:
    python -m eda.mm_rule_ablation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from artifact.filter import detect_artifacts_patient
from detectors.ensemble import compute_is_mm_candidate
from detectors.pipeline import run_patient_detectors
from eda.artifact_demo import CHOSEN_CFG
from features.extractor import extract_patient_features
from data_io.data_loader import list_patient_ids, load_patient
from preprocessing.pipeline import preprocess_patient


ENSEMBLE_GRID: tuple[float, ...] = (0.7, 0.85, 0.95)
ASYM_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
XCORR_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
PRIMARY_FSET: str = "selected"


def _collect_per_session_data() -> dict[tuple[str, str, str], pd.DataFrame]:
    """Cache feature + score tables for every (group, patient, session)."""
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for group in ("ucp", "td"):
        for pid in list_patient_ids(group):
            p = load_patient(group, pid)
            proc = preprocess_patient(p.sessions, p.fs)
            rep = detect_artifacts_patient(p, proc, CHOSEN_CFG)
            feats = extract_patient_features(
                proc, rep, patient_id=p.patient_id, group=p.group,
            )
            fbs = {lbl: sf.df_windows for lbl, sf in feats.items()}
            sbs = {lbl: sf.window_starts_idx for lbl, sf in feats.items()}
            out = run_patient_detectors(
                patient_id=p.patient_id, group=p.group,
                features_by_session=fbs, starts_by_session=sbs, fs=p.fs,
            )
            for (sess, fset), sdo in out.per_session.items():
                if fset != PRIMARY_FSET:
                    continue
                cache[(group, pid, sess)] = sdo.df_windows
            print(f"  cached {group}/{pid}")
    return cache


def _count_mm_candidates(df: pd.DataFrame, score_cut: float,
                         asym_cut: float, xcorr_cut: float) -> int:
    mask = compute_is_mm_candidate(
        score_median=df.score_median.to_numpy(),
        asymmetry_index=df["asymmetry_index"].to_numpy()
            if "asymmetry_index" in df.columns else None,
        xcorr_max=df["xcorr_max"].to_numpy()
            if "xcorr_max" in df.columns else None,
        is_artifact=df.is_artifact.to_numpy().astype(bool),
        is_boundary=df.is_boundary.to_numpy().astype(bool),
        score_cutoff=score_cut,
        asym_cutoff=asym_cut,
        xcorr_cutoff=xcorr_cut,
    )
    return int(mask.sum())


def _evaluate_grid(cache: dict) -> pd.DataFrame:
    rows = []
    for score_cut in ENSEMBLE_GRID:
        for asym_cut in ASYM_GRID:
            for xcorr_cut in XCORR_GRID:
                # Per-patient flag: at least one session has >=1 candidate.
                ucp_pos = 0
                ucp_tot = 0
                td_neg = 0
                td_tot = 0
                per_patient: dict[tuple[str, str], list[int]] = {}
                for (grp, pid, sess), df in cache.items():
                    n = _count_mm_candidates(df, score_cut, asym_cut, xcorr_cut)
                    per_patient.setdefault((grp, pid), []).append(n)
                for (grp, pid), counts in per_patient.items():
                    pos = any(c >= 1 for c in counts)
                    if grp == "ucp":
                        ucp_tot += 1
                        if pos:
                            ucp_pos += 1
                    else:
                        td_tot += 1
                        if not pos:
                            td_neg += 1
                sens = ucp_pos / ucp_tot if ucp_tot else float("nan")
                spec = td_neg / td_tot if td_tot else float("nan")
                rows.append({
                    "score_cut": score_cut,
                    "asym_cut": asym_cut,
                    "xcorr_cut": xcorr_cut,
                    "sensitivity_ucp": sens,
                    "specificity_td": spec,
                    "youden_J": sens + spec - 1,
                })
    return pd.DataFrame(rows)


def _plot_heatmap(df_grid: pd.DataFrame, score_cut: float,
                  out_path: Path) -> None:
    sub = df_grid[df_grid.score_cut == score_cut]
    pivot = sub.pivot(index="asym_cut", columns="xcorr_cut", values="youden_J")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=config.PLOT_DPI)
    im = ax.imshow(pivot.values, origin="lower", cmap="RdYlGn",
                   vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
    ax.set_xlabel("xcorr_cut (>=)")
    ax.set_ylabel("asym_cut (<=)")
    ax.set_title(f"Youden J, score_cut={score_cut}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Youden J")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    out_dir = config.FIG_DIR / "sprint_05" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting per-session score tables...")
    cache = _collect_per_session_data()
    print(f"Cached {len(cache)} (group, patient, session) tables")

    print("Evaluating grid...")
    df_grid = _evaluate_grid(cache)
    df_grid.to_csv(out_dir / "grid_results.csv", index=False)
    print(df_grid.sort_values("youden_J", ascending=False).head(10).to_string(index=False))

    # Pick the row with the largest Youden J (ties broken by preferring
    # the more permissive, lower, score_cut, then more permissive xcorr).
    best_idx = df_grid.sort_values(
        ["youden_J", "score_cut", "xcorr_cut"],
        ascending=[False, True, True]
    ).index[0]
    best = df_grid.loc[best_idx]

    # Heatmap for the chosen score_cut.
    _plot_heatmap(df_grid, float(best.score_cut),
                  out_dir / "youden_heatmap.png")

    text = (
        f"# Sprint-5 chosen MM-rule cutoffs\n"
        f"# Selected feature set: {PRIMARY_FSET}\n"
        f"# Best Youden J = {best.youden_J:.3f} "
        f"(sensitivity_ucp={best.sensitivity_ucp:.3f}, "
        f"specificity_td={best.specificity_td:.3f})\n"
        f"ENSEMBLE_THRESHOLD = {best.score_cut}\n"
        f"ASYM_MM_CUTOFF = {best.asym_cut}\n"
        f"XCORR_MM_CUTOFF = {best.xcorr_cut}\n"
    )
    (out_dir / "chosen_thresholds.txt").write_text(text, encoding="utf-8")
    print(f"\nChosen:\n{text}")


if __name__ == "__main__":
    main()
