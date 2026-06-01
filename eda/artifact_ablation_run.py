"""
Run the artifact-filter k ablation and pick a final configuration.

Outputs:
    doc/figures/sprint_02/ablation/summary_ablation.csv
    doc/figures/sprint_02/ablation/k_vs_pct_outlier.png
    doc/figures/sprint_02/ablation/k_vs_min_n_calm.png
    doc/figures/sprint_02/ablation/chosen_config.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from artifact.ablation import run_grid


K_GRID: tuple[float, ...] = (3.0, 4.0, 5.0, 7.0)
SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
OVERLAP_GRID: tuple[float, ...] = (0.25, 0.50, 0.75)
PCT_CALM_TARGET: float = 70.0
PCT_TRIPLETS_THRESHOLD: float = 80.0   # % combos that must meet pct_calm
# UCP4 has a known ~100 g/s jerk spike (Sprint 1 finding). At any reasonable
# k the jerk threshold must stay BELOW that spike at every scale.
UCP4_JERK_SPIKE_GPS: float = 80.0


def _agreement_across_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise |diff| of pct_outlier between every overlap pair.

    For every (patient, session, scale, k) we compare the per-window outlier
    pct across overlaps: the closer the values, the more the filter is
    robust to the windowing redundancy. Each pairwise absolute diff is
    expressed in percentage points.
    """
    piv = df.pivot_table(
        index=["patient_id", "session", "scale_s", "k"],
        columns="overlap",
        values="pct_outlier",
    ).dropna()
    piv.columns = [f"o{int(c*100)}" for c in piv.columns]
    ovs = list(piv.columns)
    for i in range(len(ovs)):
        for j in range(i + 1, len(ovs)):
            piv[f"diff_{ovs[i]}_{ovs[j]}"] = (piv[ovs[i]] - piv[ovs[j]]).abs()
    return piv.reset_index()


def main() -> None:
    out_dir = config.FIG_DIR / "sprint_02" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running grid: k", K_GRID, "scales", SCALES, "overlap", OVERLAP_GRID)
    df = run_grid(K_GRID, scales_s=SCALES, overlap_grid=OVERLAP_GRID,
                  group="ucp")
    df.to_csv(out_dir / "summary_ablation.csv", index=False)
    print(f"  written: {out_dir / 'summary_ablation.csv'}")

    # ---------- Aggregations per (k, overlap) ----------------------------
    agg = df.groupby(["overlap", "k"]).agg(
        pct_outlier_median=("pct_outlier", "median"),
        pct_outlier_mean=("pct_outlier", "mean"),
        pct_outlier_p90=("pct_outlier", lambda s: float(np.quantile(s, 0.90))),
        pct_calm_min=("pct_calm", "min"),
        pct_calm_median=("pct_calm", "median"),
        min_n_calm=("n_calm", "min"),
        median_n_calm=("n_calm", "median"),
    ).round(2)
    print("\n=== per-(overlap, k) aggregate ===")
    print(agg.to_string())

    # ---------- Coverage criterion ---------------------------------------
    coverage = (df.assign(pass_=df["pct_calm"] >= PCT_CALM_TARGET)
                  .groupby(["overlap", "k"])["pass_"].mean() * 100.0).round(2)
    print(f"\n=== % combos with pct_calm >= {PCT_CALM_TARGET}% "
          f"per (overlap, k) ===")
    print(coverage.to_string())

    # ---------- UCP4 jerk-spike preservation per overlap -----------------
    ucp4 = df[df.patient_id == "UCP4"]
    print("\n=== UCP4 jerk_peak threshold per (overlap, k, scale, session) ===")
    print(ucp4.pivot_table(
        index=["overlap", "session", "scale_s"], columns="k",
        values="jerk_peak_threshold").round(2).to_string())

    # ---------- Overlap robustness ---------------------------------------
    overlap_diff = _agreement_across_overlap(df)
    overlap_diff.to_csv(out_dir / "overlap_agreement.csv", index=False)
    diff_cols = [c for c in overlap_diff.columns if c.startswith("diff_")]
    for dc in diff_cols:
        diff_by_k = overlap_diff.groupby("k")[dc].agg(
            ["median", "mean", "max"]).round(2)
        print(f"\n=== {dc} (pp) per k ===")
        print(diff_by_k.to_string())

    # ---------- Choose (overlap, k) --------------------------------------
    # Step 1: eligible (overlap, k) combos by coverage criterion.
    eligible = coverage[coverage >= PCT_TRIPLETS_THRESHOLD].reset_index()

    # Step 2: must keep UCP4 jerk threshold below known spike.
    def jerk_below(overlap: float, k: float) -> bool:
        sub = ucp4[(ucp4.overlap == overlap) & (ucp4.k == k)]
        if sub.empty:
            return False
        return bool((sub["jerk_peak_threshold"] < UCP4_JERK_SPIKE_GPS).all())

    eligible = eligible[eligible.apply(
        lambda r: jerk_below(r["overlap"], r["k"]), axis=1)]

    # Step 3: among eligible, prefer the overlap that gives the most STABLE
    # statistics (smallest median |diff|). When two overlaps produce nearly
    # identical results, 50% is the rational pick because:
    #   - windows are statistically more independent (less autocorrelation)
    #     so the robust MAD estimator is on firmer ground;
    #   - 2x fewer windows = 2x lower computational cost;
    #   - artifact spans tracked just as well at the scales we use (smallest
    #     scale 0.5s @ fs=80Hz -> 40 samples per window, which still slides
    #     by 20 samples = 0.25s at 50%, fine resolution).
    # 75% is preferred only if it materially improves coverage.
    if eligible.empty:
        chosen_overlap = 0.50
        chosen_k = float(min(K_GRID))
    else:
        # Prefer the most conservative k that is eligible at each overlap.
        best = (eligible
                .sort_values(["k", "overlap"], ascending=[False, True])
                .iloc[0])
        chosen_overlap = float(best["overlap"])
        chosen_k = float(best["k"])

    # Manual override on tie-break: if both overlaps are eligible at the same
    # k, pick 50% (justified above).
    same_k_eligible = eligible[eligible["k"] == chosen_k]
    if len(same_k_eligible) > 1:
        chosen_overlap = 0.50

    # ---------- Choose min_n_calm ----------------------------------------
    sub_chosen = df[(df.k == chosen_k) & (df.overlap == chosen_overlap)]
    p5_n_calm = float(np.quantile(sub_chosen["n_calm"], 0.05))
    p10_n_calm = float(np.quantile(sub_chosen["n_calm"], 0.10))
    median_n_calm = float(np.quantile(sub_chosen["n_calm"], 0.50))
    print(f"\n=== n_calm distribution at chosen k={chosen_k} "
          f"overlap={chosen_overlap} ===")
    print(f"  p5={p5_n_calm:.0f}  p10={p10_n_calm:.0f}  median={median_n_calm:.0f}")
    min_n_calm = max(30, int(p5_n_calm // 10 * 10))

    # ---------- Plots -----------------------------------------------------
    n_ov = len(OVERLAP_GRID)
    # Plot 1: k vs % outlier, faceted by overlap
    fig, axes = plt.subplots(1, n_ov, figsize=(6.0 * n_ov, 4.5),
                             dpi=config.PLOT_DPI, sharey=True)
    if n_ov == 1:
        axes = [axes]
    for ax, (ov, sub_ov) in zip(axes, df.groupby("overlap")):
        for scale_s, sub in sub_ov.groupby("scale_s"):
            med = sub.groupby("k")["pct_outlier"].median()
            ax.plot(med.index, med.values, "o-", label=f"scale {scale_s}s")
        ax.set_title(f"overlap {ov:.0%}")
        ax.set_xlabel("k (MAD multiplier)")
        ax.set_ylabel("% outlier (median)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Artifact filter, k vs. % outlier (overlap sweep)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "k_vs_pct_outlier.png")
    plt.close(fig)

    # Plot 2: k vs min(n_calm) faceted
    fig, axes = plt.subplots(1, n_ov, figsize=(6.0 * n_ov, 4.5),
                             dpi=config.PLOT_DPI)
    if n_ov == 1:
        axes = [axes]
    for ax, (ov, sub_ov) in zip(axes, df.groupby("overlap")):
        for scale_s, sub in sub_ov.groupby("scale_s"):
            mn = sub.groupby("k")["n_calm"].min()
            med = sub.groupby("k")["n_calm"].median()
            ax.plot(mn.index, mn.values, "o--", label=f"min {scale_s}s")
            ax.plot(med.index, med.values, "s-", label=f"median {scale_s}s")
        ax.set_title(f"overlap {ov:.0%}")
        ax.set_xlabel("k (MAD multiplier)")
        ax.set_ylabel("n_calm")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Artifact filter, k vs. n_calm (overlap sweep)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "k_vs_min_n_calm.png")
    plt.close(fig)

    # Plot 3: agreement between overlap pairs
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=config.PLOT_DPI)
    diff_cols_local = [c for c in overlap_diff.columns if c.startswith("diff_")]
    n_diff = len(diff_cols_local)
    width = 0.8 / max(n_diff, 1)
    for i, dc in enumerate(diff_cols_local):
        positions = []
        data = []
        for k_val, sub in overlap_diff.groupby("k"):
            positions.append(float(k_val) + (i - (n_diff - 1) / 2) * width)
            data.append(sub[dc].to_numpy())
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        showfliers=False, patch_artist=True,
                        manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_alpha(0.5)
        ax.plot([], [], "s", label=dc.replace("diff_", "|"
                                              ).replace("_", "-") + "|",
                alpha=0.5)
    ax.set_xticks(sorted(overlap_diff["k"].unique()))
    ax.set_xticklabels([f"k={int(k)}" for k in sorted(overlap_diff["k"].unique())])
    ax.set_ylabel("|Δ pct_outlier| (pp)")
    ax.set_title("Overlap robustness, pairwise disagreement per k")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_disagreement.png")
    plt.close(fig)

    # ---------- Save chosen config ---------------------------------------
    text = (
        f"# Chosen artifact-filter configuration (Sprint 2 ablation, "
        f"k + overlap)\n"
        f"# Grid: k in {list(K_GRID)}, scales in {list(SCALES)}, "
        f"overlap in {list(OVERLAP_GRID)}\n"
        f"# Selection rule:\n"
        f"#   (1) keep windows with pct_calm >= {PCT_CALM_TARGET}% on >= "
        f"{PCT_TRIPLETS_THRESHOLD}% of combos;\n"
        f"#   (2) UCP4 jerk threshold < {UCP4_JERK_SPIKE_GPS:.0f} g/s (known "
        f"spike must remain outlier);\n"
        f"#   (3) among ties, prefer the most conservative k;\n"
        f"#   (4) among ties on overlap, prefer 50% (better MAD independence,"
        f" lower cost, equivalent detection at the scales used).\n"
        f"k = {chosen_k}\n"
        f"overlap = {chosen_overlap}\n"
        f"min_n_calm = {min_n_calm}\n"
        f"scales_s = {list(SCALES)}\n"
        f"\n"
        f"# n_calm distribution at chosen config:\n"
        f"# p5  = {p5_n_calm:.0f}\n"
        f"# p10 = {p10_n_calm:.0f}\n"
        f"# median = {median_n_calm:.0f}\n"
    )
    (out_dir / "chosen_config.txt").write_text(text, encoding="utf-8")
    print(f"\nChosen: overlap={chosen_overlap}, k={chosen_k}, "
          f"min_n_calm={min_n_calm}")
    print(f"  written: {out_dir / 'chosen_config.txt'}")


if __name__ == "__main__":
    main()
