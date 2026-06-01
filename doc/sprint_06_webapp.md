# Sprint 6.1, Web app for explainable MM detection

## Goal

A clinician-facing single-page web tool that, given a 60 s bilateral
recording of a patient, points out the moments most likely to contain
Mirror Movements, in plain language, with one zoomed snippet per event
and one visual snapshot of the whole session.

## How to start

```bash
streamlit run webapp/app.py
```

The app opens at `http://localhost:8501`. The backend pipeline (Sprints
0 to 5.5) runs on demand, no training at startup.

## Files

Four files under `explainable/webapp/`, less than 600 lines in total:

| File | Role |
|------|------|
| `pipeline.py` | Single-patient orchestrator. Calls preprocessing, artifact filter, feature extractor, the three anomaly detectors, and the dispersion ratio. Returns a `PatientAnalysis`. |
| `data_io.py` | Demo patient loader and uploaded-CSV validator (uses `io.schema`). |
| `plots.py` | Two Plotly figure factories: `event_zoom_figure` (2 s snippet around the selected window) and `scatter_pair_figure` (per-window scatter, both sessions side by side). |
| `app.py` | Streamlit page. Helpers `render_header`, `render_drilldown`, `render_events_table`, `render_geometric_panel`. |

The non-technical English wording lives in
`explainable/explain/feature_glossary.py` (`headline_en`,
`clinical_sentence_en`). The legacy Italian glossary stays for backward
compatibility with the analysis scripts.

## Expected CSV format

Same schema as `bbt_*_raw_anon.csv` (validated by
`explainable/io/schema.py`): columns `Accelerometer X/Y/Z`, `datetime`,
`hand`, `hand_dominance`, `type`, `hand_type`, `session`, `hand_label`,
`id`. Exactly one patient id per file. Both sessions `dom` and `ndom`
are recommended; at least one is required.

## Page layout (top to bottom)

1. **Header**: patient id and group, dominant hand, two MM-event
   counters (one per session, non-dominant first), and the spread ratio
   ndom over dom.
2. **Selected event drill-down (primary view)**: chosen row from the
   table below renders here. Shows event start time, severity score on
   a 0 to 1 scale, a clinical tag (likely mirror movement, likely
   artifact, or borderline), three plain-language sentences describing
   why the window is suspect, and a 2 s zoom plot with both hands
   overlaid.
3. **Events table**: sortable, click a row to update the drill-down.
   Columns are limited to `t (s)`, `severity`, `stillness balance`,
   `hand sync`. Tab order puts the non-dominant active session first
   (clinical request).
4. **Window pattern panel**: two scatter charts side by side, one per
   session, showing every clean window as a dot in the plane
   `stillness balance` vs `hand sync`. MM events are circled in green.
   Caption explains the geometric reading: tighter clusters mean a
   consistent behaviour, wider clusters on the ndom-active panel hint
   at intermittent mirror movements.

## What we removed compared to the first draft

* Italian copy, replaced with English everywhere.
* Em-dash characters, replaced with commas or colons across the whole
  project (project-wide style rule).
* Technical jargon: no more "ensemble", "z-score", "PCA", "MAD",
  "robust quantile".
* The ENMO and anomaly-score timeline plots, not informative at a
  glance for non-technical readers.
* The threshold banner with raw cutoffs.

## What we kept

* The full backend stack and the detection rule chosen in Sprint 5
  (sensitivity 82.4 percent, specificity 89.3 percent, Youden 0.716).
* The dispersion ratio header metric (Sprint 5.5).
* `@st.cache_data` for the demo dropdown, so repeated selections are
  instant.

## Verification

1. `streamlit run webapp/app.py` starts without errors.
2. No em-dash character in `explainable/webapp/*.py`, in
   `explainable/explain/feature_glossary.py`, or in this doc file.
3. Picking UCP5 in the demo dropdown shows the non-dominant tab with
   13 events in the table, a populated drill-down on the top event
   (around t = 33 s), and a scatter panel where the ndom side is
   visibly more spread than the dom side (Sprint 5.5 ratio 1.78).
4. Picking UCP7 (calm benchmark) shows at most 3 events and both
   scatter clusters compact.
5. Uploading a single-patient CSV of UCP4 finishes in under 10 s and
   matches the dropdown output.

## Clinical reading sample (UCP5, ndom-active)

* Top event at t = 33.0 s, severity 0.99, tag `Likely mirror movement`.
* "Still hand moves with energy comparable to the active hand (strong
  signal), a typical mirror pattern."
* "Both hands move in sync within about 100 ms (moderate signal),
  consistent with mirror coupling."
* "The still hand moves more than usual for this patient (mild signal)."

## Out of scope (Sprint 6.1)

* No PDF or PNG export.
* No multi-patient batch view.
* No authentication or persistent storage.
