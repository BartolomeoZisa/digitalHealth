"""
Ablation over the artifact-filter hyperparameters.

We sweep `k` (the robust-threshold multiplier) and report, for every patient,
session and scale, how many windows are classified as calm vs outlier. The
ablation table is then used in `eda/artifact_ablation_run.py` to select a
sensible (k, min_n_calm) pair.

The metric used to pick a final k is recorded next to the table: we want the
known UCP4 jerk-peak spike (~100 g/s, seen in Sprint 1) to remain an outlier
at every scale, while keeping the share of calm windows at least 70% on at
least 80% of (patient, session, scale) triplets.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from artifact.filter import (
    FilterConfig,
    PatientArtifactReport,
    detect_artifacts_patient,
)
from data_io.data_loader import load_patient, list_patient_ids
from preprocessing.pipeline import preprocess_patient


def run_grid(k_grid: Iterable[float],
             scales_s: tuple[float, ...] = (0.5, 1.0, 2.0),
             overlap_grid: Iterable[float] = (0.50, 0.75),
             group: str = "ucp",
             max_iter: int = 5,
             ) -> pd.DataFrame:
    """Sweep `k` and `overlap` on every patient of the group.

    Returns a long-form DataFrame with columns:
        patient_id, session, scale_s, overlap, k,
        n_total, n_outlier, n_calm, pct_outlier, pct_calm,
        jerk_peak_threshold, jerk_peak_n_iter
    """
    rows: list[dict] = []
    for pid in list_patient_ids(group):
        p = load_patient(group, pid)
        proc = preprocess_patient(p.sessions, p.fs)
        for overlap in overlap_grid:
            for k in k_grid:
                cfg = FilterConfig(
                    k=float(k), max_iter=max_iter,
                    scales_s=scales_s, overlap=float(overlap),
                    min_n_calm=0, relax_k=float(k),
                )
                rep = detect_artifacts_patient(p, proc, cfg)
                for sl, sr in rep.per_session.items():
                    for scale_s, scale_res in sr.per_scale.items():
                        n = scale_res.n_windows
                        n_out = int(scale_res.mask_outlier_any.sum())
                        j = scale_res.threshold_per_indicator["jerk_peak"]
                        rows.append({
                            "patient_id": pid,
                            "session": sl,
                            "scale_s": float(scale_s),
                            "overlap": float(overlap),
                            "k": float(k),
                            "n_total": int(n),
                            "n_outlier": n_out,
                            "n_calm": int(n - n_out),
                            "pct_outlier": 100.0 * n_out / max(n, 1),
                            "pct_calm": 100.0 * (n - n_out) / max(n, 1),
                            "jerk_peak_threshold": j.threshold,
                            "jerk_peak_n_iter": j.n_iter,
                        })
    return pd.DataFrame(rows)
