"""
Sprint 5.5, patient-level scatter-dispersion analysis on (asym, xcorr).

For every patient we compute, on each session, the mean-pairwise and the
mean-centroid dispersion of the per-window cluster (clean windows only,
i.e. `is_boundary=False` and `is_artifact=False`). The inter-session ratio
`dispersion_ratio = disp(ndom-active) / disp(dom-active)` is then expected
to be > 1 for UCP and ≈ 1 for TD.

Outputs:
    doc/figures/sprint_05/scatter_dispersion.csv
    doc/figures/sprint_05/dispersion_ratio_distribution.png
    doc/figures/sprint_05/dispersion_scatter_overlay.png

Run:
    python -m eda.scatter_dispersion_demo
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import config
from artifact.filter import detect_artifacts_patient
from eda.artifact_demo import CHOSEN_CFG
from features.extractor import extract_patient_features
from features.scatter_dispersion import (
    SessionDispersion,
    _valid_points,
    compute_session_dispersion,
    safe_ratio,
)
from data_io.data_loader import list_patient_ids, load_patient
from preprocessing.pipeline import preprocess_patient


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------
def _collect_rows() -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    """Return summary DataFrame + scatter points cache for overlay PNG."""
    rows: list[dict] = []
    pts_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for group in ("ucp", "td"):
        for pid in list_patient_ids(group):
            p = load_patient(group, pid)
            proc = preprocess_patient(p.sessions, p.fs)
            rep = detect_artifacts_patient(p, proc, CHOSEN_CFG)
            feats = extract_patient_features(
                proc, rep, patient_id=p.patient_id, group=p.group,
            )
            disps: dict[str, SessionDispersion] = {}
            for sess_label, sf in feats.items():
                df = sf.df_windows
                disps[sess_label] = compute_session_dispersion(df)
                pts_cache[(group, pid, sess_label)] = _valid_points(
                    df, x="asymmetry_index", y="xcorr_max",
                    require_clean=True,
                )

            disp_dom = disps.get("dom")
            disp_ndom = disps.get("ndom")
            rows.append({
                "patient_id": pid,
                "group": group,
                "n_dom": disp_dom.n_points if disp_dom else 0,
                "n_ndom": disp_ndom.n_points if disp_ndom else 0,
                "disp_pairwise_dom": disp_dom.disp_pairwise if disp_dom else float("nan"),
                "disp_pairwise_ndom": disp_ndom.disp_pairwise if disp_ndom else float("nan"),
                "disp_centroid_dom": disp_dom.disp_centroid if disp_dom else float("nan"),
                "disp_centroid_ndom": disp_ndom.disp_centroid if disp_ndom else float("nan"),
                "ratio_pairwise_ndom_over_dom": safe_ratio(
                    disp_ndom.disp_pairwise if disp_ndom else float("nan"),
                    disp_dom.disp_pairwise if disp_dom else float("nan"),
                ),
                "ratio_centroid_ndom_over_dom": safe_ratio(
                    disp_ndom.disp_centroid if disp_ndom else float("nan"),
                    disp_dom.disp_centroid if disp_dom else float("nan"),
                ),
            })
            print(f"  {group}/{pid} -> "
                  f"disp(dom)={disp_dom.disp_pairwise:.3f}, "
                  f"disp(ndom)={disp_ndom.disp_pairwise:.3f}")
    return pd.DataFrame(rows), pts_cache


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _bar_distribution(df: pd.DataFrame, metric_col: str,
                      out_path: Path, title: str) -> None:
    """Sorted bar chart of `metric_col` per patient, coloured by group."""
    sub = df[["patient_id", "group", metric_col]].dropna().copy()
    sub = sub.sort_values(metric_col, ascending=True)
    colors = [config.PLOT_GROUP_COLORS[g] for g in sub.group]
    xs = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(15, 5), dpi=config.PLOT_DPI)
    ax.bar(xs, sub[metric_col].to_numpy(), color=colors, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="ratio = 1")
    # Annotate TD26 and TD27 explicitly (clinician asked about them).
    for special in ("TD26", "TD27"):
        if special in sub.patient_id.values:
            idx = int(np.where(sub.patient_id.values == special)[0][0])
            ax.annotate(special,
                        xy=(idx, sub[metric_col].iloc[idx]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7, color="black",
                        fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(sub.patient_id.to_list(), rotation=90, fontsize=6)
    ax.set_ylabel(metric_col)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.0, max(2.0, float(sub[metric_col].max()) * 1.05))
    ax.grid(alpha=0.25, axis="y")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["ucp"]),
        plt.Rectangle((0, 0), 1, 1, color=config.PLOT_GROUP_COLORS["td"]),
    ]
    ax.legend(handles + [plt.Line2D([], [], color="black", linestyle="--")],
              ["UCP", "TD", "ratio=1"], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _overlay_scatter(pts_cache: dict, df: pd.DataFrame, out_path: Path,
                     metric_col: str = "ratio_pairwise_ndom_over_dom") -> None:
    """Side-by-side dom/ndom scatter overlay for top UCP and top TD ratios."""
    ucp_sub = df[df.group == "ucp"].sort_values(metric_col, ascending=False).head(3)
    td_sub = df[df.group == "td"].sort_values(metric_col, ascending=False).head(2)
    patients = list(ucp_sub.patient_id) + list(td_sub.patient_id)
    groups = ["ucp"] * len(ucp_sub) + ["td"] * len(td_sub)

    n = len(patients)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2),
                             dpi=config.PLOT_DPI, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, pid, grp in zip(axes, patients, groups):
        for sess_label, color, marker in (
            ("dom", "#1f77b4", "o"),
            ("ndom", "#d62728", "x"),
        ):
            pts = pts_cache.get((grp, pid, sess_label),
                                np.zeros((0, 2), dtype=float))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                           s=22, alpha=0.7,
                           label=f"{sess_label}-active (n={len(pts)})")
        ratio = float(df.loc[df.patient_id == pid,
                             metric_col].iloc[0])
        ax.set_title(f"{pid} ({grp.upper()})  ratio={ratio:.2f}",
                     fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("asymmetry_index")
        ax.set_ylabel("xcorr_max")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Per-session scatter overlay for top dispersion ratios "
                 "(UCP top 3, TD top 2)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Stats & orchestration
# -----------------------------------------------------------------------------
def _aggregate_stats(df: pd.DataFrame, metric_col: str) -> dict:
    ucp = df[df.group == "ucp"][metric_col].dropna()
    td = df[df.group == "td"][metric_col].dropna()
    stat = {
        "metric": metric_col,
        "ucp_median": float(ucp.median()),
        "td_median": float(td.median()),
        "ucp_mean": float(ucp.mean()),
        "td_mean": float(td.mean()),
        "ucp_frac_above_1": float((ucp > 1.0).mean()),
        "td_frac_below_1_2": float((td < 1.2).mean()),
        "td26": float(df.loc[df.patient_id == "TD26", metric_col].iloc[0])
                if "TD26" in df.patient_id.values else float("nan"),
        "td27": float(df.loc[df.patient_id == "TD27", metric_col].iloc[0])
                if "TD27" in df.patient_id.values else float("nan"),
        "td_median_baseline": float(td.median()),
    }
    if len(ucp) > 5 and len(td) > 5:
        u_stat, p = mannwhitneyu(ucp, td, alternative="greater")
        stat["mw_u"] = float(u_stat)
        stat["mw_p_one_sided_ucp_greater"] = float(p)
    else:
        stat["mw_u"] = float("nan")
        stat["mw_p_one_sided_ucp_greater"] = float("nan")
    return stat


def main() -> None:
    out_dir = config.FIG_DIR / "sprint_05"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting per-patient dispersions...")
    df, pts_cache = _collect_rows()
    df.to_csv(out_dir / "scatter_dispersion.csv", index=False)
    print(f"Saved: {(out_dir / 'scatter_dispersion.csv').relative_to(config.PKG_DIR)}")

    print("\n=== Aggregates ===")
    for col in ("ratio_pairwise_ndom_over_dom", "ratio_centroid_ndom_over_dom"):
        stats = _aggregate_stats(df, col)
        print(f"\n{col}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\nPlotting bar distribution...")
    _bar_distribution(
        df, "ratio_pairwise_ndom_over_dom",
        out_dir / "dispersion_ratio_distribution.png",
        title="Patient-level dispersion ratio "
              "(pairwise distance, ndom-active / dom-active)",
    )
    print(f"Saved: {(out_dir / 'dispersion_ratio_distribution.png').relative_to(config.PKG_DIR)}")

    print("Plotting top scatter overlay...")
    _overlay_scatter(
        pts_cache, df,
        out_dir / "dispersion_scatter_overlay.png",
    )
    print(f"Saved: {(out_dir / 'dispersion_scatter_overlay.png').relative_to(config.PKG_DIR)}")


if __name__ == "__main__":
    main()
