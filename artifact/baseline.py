"""
Intra-patient reference distributions on the calm pool.

After the artifact filter has separated calm windows from outliers, we keep,
for every (patient, session, scale), the quantile profile of each indicator
on the calm subset. These quantiles will be the personal baseline used by
Sprint 3 features and by Sprint 4 detectors.

No cross-patient pooling. No TD involvement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from artifact.filter import PatientArtifactReport, ScaleResult
from artifact.indicators import INDICATOR_NAMES
from artifact.robust_stats import quantiles


DEFAULT_QUANTILES: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class IndicatorBaseline:
    patient_id: str
    session_label: str
    scale_s: float
    indicator: str
    n_calm: int
    q: dict[float, float]      # quantile -> value


def compute_indicator_baseline(report: PatientArtifactReport,
                               q: tuple[float, ...] = DEFAULT_QUANTILES,
                               ) -> list[IndicatorBaseline]:
    """Compute one IndicatorBaseline per (session, scale, indicator)."""
    out: list[IndicatorBaseline] = []
    for session_label, sess in report.per_session.items():
        for scale_s, scale_res in sess.per_scale.items():
            calm_idx = sess.calm_idx_per_scale[scale_s]
            df = scale_res.indicators.iloc[calm_idx]
            for name in INDICATOR_NAMES:
                vals = df[name].to_numpy()
                out.append(IndicatorBaseline(
                    patient_id=report.patient_id,
                    session_label=session_label,
                    scale_s=float(scale_s),
                    indicator=name,
                    n_calm=int(len(vals)),
                    q=quantiles(vals, q=q),
                ))
    return out


def baseline_to_dataframe(baselines: list[IndicatorBaseline]) -> pd.DataFrame:
    """Flatten a list of IndicatorBaseline into a long-form DataFrame."""
    rows = []
    for b in baselines:
        row = {
            "patient_id": b.patient_id,
            "session": b.session_label,
            "scale_s": b.scale_s,
            "indicator": b.indicator,
            "n_calm": b.n_calm,
        }
        for q, v in b.q.items():
            row[f"q{int(q * 100)}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
