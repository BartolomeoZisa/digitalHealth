# Sprint 5, Boundary Trimming + Composite MM Rule + Ablation

## Goal

Solve two issues found during manual inspection of the Sprint 4 PNGs:

1. **Edge artefacts**: the first and last seconds of the timeseries contain BBT
   setup/cleanup (placing and lifting the hand) and `sosfiltfilt` Butterworth
   filter transients. Boundary windows were producing false positives.
2. **TD flagged as MM**: the intra-patient detector alone classified any window
   outside the calm pool as `mm_like`. TD recordings with generic gestures, not
   specific to MM, also ended up in that category.

The solution combined two fixes plus one ablation to choose the final cutoffs.

## Fix 1, Boundary Trimming

- New parameter: `config.BOUNDARY_TRIM_S = 3.0`.
- In `features/extractor.py`, every window receives the `is_boundary` column if
  it starts before 3 s or ends after 57 s.
- Boundary windows **remain in the data**, to preserve detector statistics for
  the intra-patient model, but are **excluded** from the new
  `is_mm_candidate` column.
- Visualization: light yellow shading on boundary regions in Sprint 4 timeline
  plots, plus green spans over `is_mm_candidate` windows.

## Fix 2, Composite MM Rule

New column in the window DataFrame
(`detectors/ensemble.py::compute_is_mm_candidate`):

```text
is_mm_candidate = (
    score_median  >= ENSEMBLE_THRESHOLD
    AND asymmetry_index <= ASYM_MM_CUTOFF       # still hand participates
    AND xcorr_max       >= XCORR_MM_CUTOFF      # bilateral synchrony
    AND not is_artifact
    AND not is_boundary
)
```

Rationale: the theoretical MM signature is "the still hand replicates the
active gesture with a small lag". This requires low asymmetry **and** high
bilateral synchrony, two properties that independent motor artefacts, including
top TD cases and UCP Sprint 2 outliers, do not have.

## Fix 3, Grid Ablation on Cutoffs

Grid tested by `eda/mm_rule_ablation.py`:

- `ENSEMBLE_THRESHOLD in {0.7, 0.85, 0.95}`
- `ASYM_MM_CUTOFF in {0.3, 0.4, 0.5, 0.6}`
- `XCORR_MM_CUTOFF in {0.3, 0.4, 0.5, 0.6}`

Metrics:

- `sensitivity_ucp`: percentage of UCP patients with at least one
  `is_mm_candidate` window in at least one session.
- `specificity_td`: percentage of TD patients with zero `is_mm_candidate`
  windows across both sessions.
- `youden_J = sensitivity + specificity - 1`.

**Top configurations**

| score_cut | asym_cut | xcorr_cut | sens UCP | spec TD | Youden J |
|---:|---:|---:|---:|---:|---:|
| **0.70** | **0.60** | **0.40** | **82.4%** | **89.3%** | **0.716** |
| 0.70 | 0.60 | 0.30 | 82.4% | 85.7% | 0.681 |
| 0.85 | 0.60 | 0.40 | 76.5% | 89.3% | 0.658 |
| 0.70 | 0.40 | 0.30 | 64.7% | 100.0% | 0.647 |
| 0.70 | 0.50 | 0.30 | 70.6% | 92.9% | 0.634 |

**Selected configuration**: `ENSEMBLE_THRESHOLD=0.70`,
`ASYM_MM_CUTOFF=0.60`, `XCORR_MM_CUTOFF=0.40`. This maximizes Youden J without
forcing sensitivity below 70%.

## Final Results (selected feature set)

Per patient, using the maximum `n_mm_candidate` across both sessions:

- **14/17 UCP** have at least one MM-candidate window (sensitivity 82.4%).
- **25/28 TD** have no MM-candidate windows (specificity 89.3%).
- Totals: UCP 77 windows, TD 6 windows.
- Mean per session: UCP 2.26, TD 0.11.

**Top UCP cases**, likely MM manifestations:

- **UCP5 `ndom-active`**: 13 candidates.
- **UCP6 `dom-active`**: 9 candidates.
- **UCP2 `ndom-active`**: 8 candidates.
- **UCP10 `ndom-active`**: 6 candidates.
- **UCP11 `ndom-active`**: 6 candidates.
- **UCP10 `dom`**, **UCP4 `dom`**: 5 candidates.

**UCP with no candidates**, possible non-expression within 60 s:

- **UCP14** (0+0), **UCP9** (0+0).
- **UCP16** (0+0): the `dom-active` session is saturated by artefacts
  (66/78 high-score windows are `is_artifact`), so no MM can be identified in
  this session.

**TD false positives**

- **TD26 `dom-active`**: 4 candidates, largest residual false positive.
- **TD12 `dom-active`**: 1 candidate.
- **TD17 `ndom-active`**: 1 candidate.

## Residual Limits

- Three UCP patients (UCP14, UCP9, UCP16) remain at zero. A 60 s recording is
  not always enough, and UCP16 has a very noisy session.
- Three TD patients produce 1-4 false positives. Their MM-like signatures need
  qualitative inspection.
- The ablation uses only the `selected` feature set. A later ablation on `full`
  could change the numbers; it was skipped in Sprint 5 for parsimony and can be
  revisited in Sprint 7.

## Outputs

- Code: `explainable/eda/mm_rule_ablation.py`; extensions to `config.py`,
  `features/extractor.py`, `detectors/ensemble.py`, `detectors/pipeline.py`,
  and `eda/detector_demo.py`.
- 45 PNGs regenerated with yellow boundary shading and green MM-candidate spans.
- `figures/sprint_05/ablation/grid_results.csv`, `youden_heatmap.png`,
  `chosen_thresholds.txt`.
- `figures/sprint_04/summary_mm.csv`, `cross_patient_mm_candidate_count.png`.

## Consequences for Sprint 6 (Web App)

- Show `is_mm_candidate` as a green timeline highlight, no longer the binary
  `score >= 0.7`.
- Event drill-down only on `is_mm_candidate` windows.
- Show separate counters in the summary panel: `n_mm_candidate`,
  `n_artifact_high`, and `n_boundary_high`.
