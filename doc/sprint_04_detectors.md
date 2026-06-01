# Sprint 4, Unsupervised Intra-Patient Detector + Explainability

## Goal

Build a system that computes a per-window anomaly score in [0, 1] for each UCP
patient, with interpretable attribution through the top three contributing
features. TD patients are not used as baseline. Detectors are fitted on the
patient's own `calm` windows from Sprint 2 and then applied to all windows of
that same patient.

## Architecture

Three detectors live in `explainable/detectors/`:

1. **`RobustQuantileDetector`** estimates `median + k*MAD` per feature on the
   calm pool; score = `1 - exp(-max(|z|)/k)`. This is the most interpretable
   baseline.
2. **`IsolationForestDetector`** uses sklearn
   `IsolationForest(n_estimators=200, max_samples=256)` fitted on calm windows;
   `-decision_function` is rescaled through the empirical CDF of calm scores.
3. **`PCAReconstructionDetector`** uses robust z-scores plus PCA with
   `n_components<=8` on calm windows; score is residual SSE, normalized through
   the calm CDF.

The ensemble is the **median** of the three scores (`MedianEnsemble`).
Attribution comes from the quantile detector, preserving sign for the clinical
glossary.

`explainable/explain/feature_glossary.py` and `attribution.py` translate
`(feature_name, robust_z)` into clinical sentences, for example: "low asymmetry
index: the still hand participates with energy comparable to the active hand,
a mirror signature".

NaN handling: spectral features can produce NaN on flat windows.
`impute_nan_per_column` replaces them with the calm-pool column median before
sklearn fitting; `robust_z` clips NaN/Inf to 0.

## Feature-Set Ablation

Spearman rank correlation between the ensemble using **full** (17 features) and
**selected** (5 clinical features):

| group | session | median rho | min rho | max rho |
|---|---|---:|---:|---:|
| UCP | dom | 0.66 | 0.50 | 0.96 |
| UCP | ndom | 0.62 | 0.50 | 0.87 |
| TD | dom | 0.65 | 0.36 | 0.91 |
| TD | ndom | 0.62 | 0.40 | 0.96 |

Only **5.6%** of combinations have `rho > 0.85`, so the two feature sets rank
windows very differently. Spectral features (`band_power_*`, `dominant_freq`)
have very small calm-pool ranges, producing tiny MAD values and z-scores above
100, which dominate both score and attribution in the full set. UCP5 confirms
this in the Sprint 4 smoke test: full attribution is `band_power_fast(+111)`,
while selected attribution is `enmo_peak + asymmetry_index + jerk_mag_rms`, the
clinically sensible reading.

**Final choice**: use `selected` as the primary feature set in plots and
clinical sentences. `full` remains available for comparison in
`summary_detector.csv`.

## Cross-Patient UCP vs TD (selected feature set)

| Metric at p95 `score_median` | UCP median | TD median | MW p |
|---|---:|---:|---:|
| p95 per patient/session | 0.961 | 0.958 | 0.50 |
| `n_high_score` / session | 31 | 30 | n.s. |
| `n_mm_like` / session | 23 | 22 | n.s. |

**The score does not discriminate UCP vs TD at aggregate level**, consistent
with Sprint 3. The system is clinical decision support, not a classifier: it
identifies the most anomalous windows for that patient.

## Observed Limits

1. **A 0.7 threshold is too aggressive** for quantile-normalized detectors. By
   construction, the empirical CDF maps the 5th percentile of the calm pool to
   about 0.05, and windows outside the calm pool can approach 1.0. With about
   110 calm windows, 20-25% of windows can exceed the threshold even in calm
   benchmark patients (UCP7: 25 flagged windows, about 0 expected).
   - Operational solution: use **top-K events per patient** in reports, for
     example top-5 or top-10 windows, instead of a binary cutoff.
2. **UCP7 confirms this limit**: the patient has almost 0 outliers in Sprint 2
   but 25 `n_mm_like` windows here, meaning the detector amplifies
   micro-variations when the calm pool is extremely uniform.

## Key Patients for Manual Inspection

PNGs are under `doc/figures/sprint_04/ucp/`:

- **UCP5 `dom-active`**: `n_mm_like=27`, including 4 artefacts. Top-5 windows
  align with low `asymmetry_index` and high `enmo_peak`, making them target MM
  candidates.
- **UCP9 `dom-active`**: `n_mm_like=28`, artefact=1. Very few artefacts, so the
  score is almost pure MM signal.
- **UCP2 `dom-active`**: `n_mm_like=28`, artefact=8. Mixed case.
- **UCP4 `ndom-active`**: `n_mm_like=24`, artefact=16, strong known artefact
  component from the Sprint 1 100 g/s jerk spike.
- **UCP16 `dom-active`**: 12 `mm_like` among 78 high-score windows; the system
  recognizes that most high windows are artefacts (66/78).
- **TD12 `ndom-active`**: 79 high-score windows, 68 artefacts, only 11
  `mm_like`, confirming `is_artifact` as a useful filter.

## Outputs

- Code: `explainable/detectors/{base, robust_quantile, isolation_forest, pca_reconstruction, ensemble, pipeline}.py`;
  `explainable/explain/{feature_glossary, attribution}.py`;
  `explainable/eda/detector_demo.py`.
- 45 per-patient PNGs under `figures/sprint_04/{ucp,td}/`.
- `cross_patient_score_distribution.png`, boxplot UCP vs TD by feature set and
  session.
- `summary_detector.csv` (180 rows = 45 x 2 x 2 feature sets).
- `agreement_full_vs_selected.csv` (90 Spearman rows by patient/session).

## Consequences for Sprint 6 (Web App)

- Show **top-K events** per session instead of a simple binary cutoff.
- Default feature set is `selected`; `full` can remain available for advanced
  comparison.
- `is_artifact` must be visible for every event, distinguishing it from "likely
  MM".
- Clinical attribution from `RobustQuantileDetector` is the source of UI
  sentences.
