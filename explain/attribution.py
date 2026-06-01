"""
Per-event explanation layer.

The detector pipeline already records `attribution_top3` as a compact
string of `feature(z)` pairs. This module turns the raw triples into
sentences a clinician can read, plus a structured object that the upcoming
web app can serialise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import config
from detectors.base import Contribution
from explain.feature_glossary import clinical_sentence, headline


@dataclass(frozen=True)
class EventExplanation:
    t_start_s: float
    score_median: float
    is_artifact: bool
    contribs: list[Contribution]
    sentences: list[str]
    tag: str   # "Likely mirror movement" | "Likely artifact" | "Borderline"


def _tag_event(score_median: float, is_artifact: bool) -> str:
    if is_artifact and score_median >= config.ENSEMBLE_THRESHOLD:
        return "Likely artifact"
    if score_median >= config.ENSEMBLE_THRESHOLD:
        return "Likely mirror movement"
    return "Borderline"


def explain_event(t_start_s: float, score_median: float, is_artifact: bool,
                  contribs: Iterable[Contribution]) -> EventExplanation:
    contribs = list(contribs)
    sentences = [clinical_sentence(name, z) for name, z in contribs]
    return EventExplanation(
        t_start_s=float(t_start_s),
        score_median=float(score_median),
        is_artifact=bool(is_artifact),
        contribs=contribs,
        sentences=sentences,
        tag=_tag_event(score_median, is_artifact),
    )


def parse_attribution_string(s: str) -> list[Contribution]:
    """Inverse of `ensemble.attribution_to_str`."""
    if not s:
        return []
    out: list[Contribution] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk or "(" not in chunk:
            continue
        name, rest = chunk.split("(", 1)
        val_str = rest.rstrip(")")
        try:
            val = float(val_str)
        except ValueError:
            continue
        out.append((name.strip(), val))
    return out


def explain_top_events(df_windows: pd.DataFrame,
                       threshold: float | None = None,
                       max_events: int = 10) -> list[EventExplanation]:
    """Return up to `max_events` events with `score_median >= threshold`."""
    if threshold is None:
        threshold = config.ENSEMBLE_THRESHOLD
    sub = df_windows[df_windows.score_median >= threshold].copy()
    sub = sub.sort_values("score_median", ascending=False).head(max_events)
    events: list[EventExplanation] = []
    for _, row in sub.iterrows():
        contribs = parse_attribution_string(row.attribution_top3)
        events.append(explain_event(
            row.t_start_s, row.score_median, row.is_artifact, contribs,
        ))
    return events
