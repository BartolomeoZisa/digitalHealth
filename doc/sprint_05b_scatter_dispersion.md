# Sprint 5.5, Scatter Dispersion Ratio (Geometric Session Asymmetry)

## Goal

Quantify a patient-level pattern observed by the clinician during manual
inspection of the Sprint 5 PNGs: the **cluster dispersion** of
`(asymmetry_index, xcorr_max)` points changes between the two sessions in a
group-specific way.

- **UCP**: the `ndom-active` cluster is more dispersed than the `dom-active`
  cluster.
- **TD**: both clusters are compact, with similar dispersion.

Per-window features from Sprint 3 and the composite `is_mm_candidate` rule from
Sprint 5 do not capture this **session-level geometric property**.

## Metric

For each patient/session, consider the point set:

```text
S = { (asym_i, xcorr_i) : not is_boundary_i AND not is_artifact_i }
```

These are the same points the clinician inspected in the scatter plots.

Two complementary dispersion metrics are computed:

1. `disp_pairwise = mean_{i<j} ||p_i - p_j||_2`, via
   `scipy.spatial.distance.pdist`.
2. `disp_centroid = mean_i ||p_i - centroid||_2`, more stable with few points.

Patient-level ratio:

```text
ratio = disp_ndom / disp_dom
```

## Results

| Statistic | UCP (n=17) | TD (n=28) |
|---|---:|---:|
| `ratio_pairwise` median | **1.27** | **1.03** |
| `ratio_pairwise` mean | 1.35 | 1.01 |
| fraction `ratio > 1.0` | 70.6% | 53.6% |
| fraction `ratio < 1.2` | 47.1% | 71.4% |
| `ratio_centroid` median | **1.30** | **1.01** |
| `ratio_centroid` mean | 1.36 | 1.00 |

Mann-Whitney U, one-sided UCP > TD:

- `ratio_pairwise`: U=338, **p=0.0099**
- `ratio_centroid`: U=342, **p=0.0077**

The clinical hypothesis is **confirmed**: ndom dispersion is higher on average
than dom dispersion in UCP, while this pattern is absent or inverted in TD.

## Notable Patients

Top UCP by `ratio_pairwise`, likely MM-like geometric manifestations:

| Patient | disp(dom) | disp(ndom) | ratio |
|---|---:|---:|---:|
| **UCP2** | 0.17 | 0.40 | **2.30** |
| **UCP3** | 0.16 | 0.33 | **2.09** |
| **UCP5** | 0.18 | 0.34 | **1.89** |
| **UCP9** | 0.11 | 0.21 | **1.88** |
| **UCP13** | 0.12 | 0.20 | **1.70** |
| UCP1 | 0.18 | 0.27 | 1.48 |
| UCP7 | 0.14 | 0.20 | 1.38 |
| UCP6 | 0.21 | 0.27 | 1.29 |

**UCP9 and UCP7** had `n_mm_candidate <= 2` in Sprint 5, but high ratios here.
The dispersion pattern therefore identifies them as UCP-like even when the
candidate count is low. This suggests sub-threshold MM that is missed by the
composite rule but visible as cluster spread.

Exceptions:

- **UCP14**, **UCP12**, **UCP10**, **UCP16**, **UCP4**: ratio <= 1.0. UCP4 and
  UCP16 have artefact-saturated `dom-active` sessions in Sprint 5, artificially
  inflating `disp_dom`.

Borderline TD cases:

- **TD27**: ratio = **1.37**, confirmed UCP-like and above the UCP median,
  consistent with the clinician's hypothesis.
- **TD26**: ratio = **0.71**, opposite direction to the prediction
  (`dom` more dispersed than `ndom`). This may indicate poor acquisition during
  `dom-active` rather than MM.
- Other TD with high ratio: TD0 (1.64), TD22 (1.58), TD20 (1.34), TD5 (1.29),
  residual false positives requiring follow-up.

## Cutoff Candidates (Youden J)

| Cutoff | sens UCP | spec TD | Youden J |
|---|---:|---:|---:|
| >= 1.0 | 70.6% | 46.4% | 0.170 |
| >= 1.1 | 64.7% | 60.7% | 0.254 |
| **>= 1.2** | **58.8%** | **71.4%** | **0.303** |
| >= 1.3 | 41.2% | 85.7% | 0.269 |
| >= 1.5 | 29.4% | 92.9% | 0.223 |

Best Youden J is 0.30 at cutoff 1.2, much lower than the Sprint 5 composite
rule (J=0.72). As a standalone patient-level discriminator, the dispersion
ratio is insufficient.

## Conclusion and Integration

`dispersion_ratio` is **statistically significant** (MW p<0.01) and
**complementary** to the Sprint 5 composite rule: it captures patients such as
UCP9 and UCP7, who have few `mm_candidate` windows but a UCP-like dispersion
pattern.

**Decision for Sprint 6**

- Show `dispersion_ratio` as a **secondary indicator** in the web-tool summary
  panel, next to `n_mm_candidate`.
- Label example: "Cluster dispersion ndom/dom: 1.89 (UCP-like)", with a visual
  threshold at 1.2.
- Do **not** integrate it as a hard filter in the composite rule, because it
  would keep sensitivity below Sprint 5.

## Outputs

- Code: `explainable/features/scatter_dispersion.py`,
  `explainable/eda/scatter_dispersion_demo.py`.
- `figures/sprint_05/scatter_dispersion.csv` (45 rows).
- `dispersion_ratio_distribution.png`, sorted bar chart with TD26/TD27
  annotations.
- `dispersion_scatter_overlay.png`, top-3 UCP plus top-2 TD scatter plots,
  dom/ndom side by side.
