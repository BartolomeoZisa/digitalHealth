# Sprint 2, Artifact Filter + Intra-Patient Baseline

## Goal

For each UCP patient, separate `calm` windows, valid intra-patient stillness,
from `outlier` windows, gross motor artefacts or occasionally very strong MM,
before building richer features in Sprint 3. The baseline is intra-patient:
no cross-patient fitting and no TD baseline. TD recordings were not analysed
at this stage, by explicit request.

## Indicators on the Still Hand

1. **`enmo_peak`**, maximum clipped ENMO per window (g), capturing net movement
   peaks.
2. **`jerk_peak`**, maximum jerk magnitude on Y/Z bandpassed signals (g/s),
   capturing fast onsets.
3. **`yz_bp_rms`**, RMS of the Y/Z bandpassed signal (g), capturing directional
   energy.
4. **`energy_ratio`**, still-hand RMS^2 / active-hand RMS^2. In stillness this
   should be much smaller than 1; large values mean the still hand exceeds the
   active hand, a strong sign of artefact or strong MM.

## Algorithm

- Multi-scale windows `{0.5, 1, 2}` s, 75% overlap, for every session of every
  UCP patient.
- Iterative robust one-sided threshold per indicator: `median + k*MAD`,
  computed on current inliers and recomputed after each exclusion round until
  convergence, with a maximum of five iterations.
- A window is marked `outlier` if at least one indicator crosses threshold
  (OR merge).
- Fallback: if `n_calm < min_n_calm`, rerun with `relax_k=10`.

## Ablation (`figures/sprint_02/ablation/`)

Grid over `k in {3, 4, 5, 7}`, scales `{0.5, 1, 2}` s, and overlaps
`{25%, 50%, 75%}` across all UCP patients.

**Aggregates by (overlap, k)**, showing only the selected conservative `k=7`:

| Overlap | Median % outlier | min n_calm | median n_calm | Coverage @ pct_calm >= 70% |
|---|---:|---:|---:|---:|
| 25% | 5.66% | 30  | 75  | 96.1% |
| 50% | 6.08% | 42  | 113 | 95.1% |
| 75% | 5.82% | 83  | 224 | 95.1% |

**Pairwise agreement**, absolute difference in `pct_outlier` over all
patient x session x scale combinations, with `k=7`:

| Pair | median (pp) | mean (pp) | max (pp) |
|---|---:|---:|---:|
| 25 vs 50 | 0.90 | 1.90 | **55.85** |
| 25 vs 75 | 0.85 | 1.73 | **56.12** |
| 50 vs 75 | 0.43 | 0.72 | 5.84 |

For UCP4, the `jerk_peak` threshold at `k=7` is essentially identical across
overlaps (`dom`, 0.5 s: 27.16 / 27.67 / 26.74 g/s), so the known 100 g/s spike
is captured in every setup.

**Final criterion**

1. `pct_calm >= 70%` on at least 80% of combinations.
2. UCP4 jerk threshold < 80 g/s, below the known spike of about 100 g/s.
3. Among eligible `k` values, choose the most conservative one, to preserve a
   wider baseline.
4. Among eligible overlaps at equal `k`, choose **50%** because:
   - **25% discarded**: maximum disagreement is 55-56 pp on extreme patients
     (UCP16), MAD over about 60 windows/session is less robust, the calm pool is
     smaller (median 75), and a 0.75 s step at 1 s scale hurts temporal
     resolution.
   - **75% discarded**: detection is almost identical to 50% (median difference
     0.43 pp), but cost doubles and adjacent windows are strongly autocorrelated,
     which inflates the MAD spread estimate.
   - **50% wins as the middle ground**: useful statistical independence for MAD,
     equivalent detection to the more redundant 75% setup, wide calm pool
     (median 113), and half the cost.

**Selected configuration**: `k=7`, `overlap=0.50`, `min_n_calm=50`,
`scales={0.5, 1, 2}` s, `relax_k=10`.

## Final Results on 17 UCP at 1 s Scale

Session asymmetry is the main indicator of where to look for MM:

| Patient | dom-active % outlier | ndom-active % outlier | Pattern |
|---|---:|---:|---|
| **UCP16** | **55.5** | 9.2 | **atypical bilateral pattern, saturated dom-active** |
| **UCP10** | **19.3** | 3.4 | likely MM in dom-active |
| **UCP15** | **13.4** | 4.2 | likely MM in dom-active |
| **UCP11** | 10.1 | **20.2** | likely MM in ndom-active |
| **UCP4**  | 9.2 | **15.1** | likely MM in ndom-active plus known dom-active spike |
| **UCP14** | 10.1 | **16.8** | moderate bilateral pattern, higher in ndom |
| **UCP13** | 5.9 | **13.4** | likely MM in ndom-active |
| **UCP1**  | 2.5 | **9.2** | likely MM in ndom-active |
| UCP2  | 9.2 | 3.4 | mild dom asymmetry |
| UCP6  | 8.4 | 3.4 | mild dom asymmetry |
| UCP3, UCP12, UCP5, UCP8, UCP0, UCP9 | < 6% in both | calm |
| **UCP7**  | 0.0 | 2.5 | **extremely calm** |

The `relax_k=10` fallback was triggered 3 times out of 102 runs (UCP16
`ndom-active` at 1 s scale, plus two 2 s cases), indicating a stable algorithm.

With the final configuration, `n_calm` statistics are: min=53, p5=50,
median=113, giving a wide calm pool for Sprint 3.

## Key Evidence for Manual Follow-Up

1. **UCP16 `dom-active` is saturated (54% outlier)**. Median `enmo_peak` is near
   zero but has a very long tail. The `enmo_peak` threshold collapses to about
   `7e-12` because both median and MAD are tiny, so it cannot discriminate well.
   This suggests a bimodal intra-patient baseline: either long continuous MM
   stretches or an agitated non-task segment in the recording. PNG:
   `figures/sprint_02/ucp/UCP16.png`. This requires checking whether the
   patient performed BBT for the whole session.
2. **UCP4 confirms the isolated spike**: jerk threshold is 26.7 g/s at 0.5 s,
   31.8 g/s at 1 s, and 28.9 g/s at 2 s. The 100 g/s spike is correctly
   captured. The interesting pattern is that the `dom-active` jerk threshold is
   much higher (about 30 g/s) than `ndom-active` (about 5 g/s), suggesting that
   the ndom hand naturally moves more during `dom-active`, a possible MM
   candidate. PNG: `figures/sprint_02/ucp/UCP4.png`.
3. **UCP10 vs UCP15**, both with strong `dom-active` asymmetry (>15%). UCP10 has
   jerk threshold `dom=24.5` vs `ndom=5.0`, a 5x asymmetry. UCP15 shows the same
   pattern. These two should be visually compared for similar MM behavior.
4. **UCP11 and UCP13** show the inverse pattern, with noisier `ndom-active`
   sessions. It should be checked whether the dominant hand, expected to remain
   still during `ndom-active`, replicates the BBT gesture.
5. **UCP7** has almost 0% outliers and is an ideal patient for validating that
   Sprint 3 features stay near zero on the still hand; otherwise there is
   leakage.
6. **UCP14** has moderate bilateral activity (about 10% in both sessions) but a
   very low `ndom-active` `enmo_peak` threshold (0.009 g), meaning the dominant
   hand is almost perfectly still. The few flagged windows are therefore highly
   informative point MM candidates.

## Outputs

- Code: `explainable/artifact/{indicators, robust_stats, filter, baseline, ablation}.py`.
- Demo scripts: `explainable/eda/{artifact_ablation_run, artifact_demo}.py`.
- 17 PNGs under `doc/figures/sprint_02/ucp/`.
- `summary_artifact.csv` (102 rows = 17 x 2 x 3), `baseline_quantiles.csv`
  (1632 rows = 17 x 2 x 3 x 4 x 4), `ablation/summary_ablation.csv`,
  `ablation/chosen_config.txt`.

## UCP vs TD Check

As a control, the filter was also run on the 28 TD patients. The surprising
result was that intra-patient outlier percentage alone does **not** separate
UCP from TD.

| Statistic at 1 s scale | UCP | TD |
|---|---:|---:|
| median | 5.04% | 5.04% |
| q75 | 9.87% | 8.40% |
| q90 | 16.30% | 13.45% |
| q95 | 19.62% | 20.59% |
| max | 55.46% (UCP16) | 57.14% (TD12) |

**Most activated TD sessions** (`pct_outlier > 10%` at some scale): TD12
(`ndom` 57% at 1 s), TD7 (`dom` 44%), TD10 (`ndom` 27%), TD15 (`ndom` 18%),
TD2 (`dom` 16%), TD3, TD27, TD23, TD16, TD26. This is about 20% of TD
combinations (35/168).

**Interpretation**: the filter captures windows that are anomalous relative to
the patient's own baseline. Both UCP and TD can have involuntary movement of
the still hand, such as arm raises, gestures, or repositioning. The number of
intra-patient outliers is not MM-specific; it is a generic deviation-from-self
measure. Distinguishing MM from artefact requires bilateral directional
features in Sprint 3: the still hand replicating the active gesture at small lag
is MM, while independent still-hand movement is artefact.

**Practical consequence**: Sprint 3 must introduce bilateral synchrony features,
especially lag-aware cross-correlation between dom and ndom. Intra-patient
`pct_outlier` remains useful only to define the `calm` pool for robust baselines,
not as a disease signal.

## Known Limitations

- In the most MM-active patients (UCP16, UCP10, UCP15), the filter may classify
  true MM as `outlier`, and the MAD itself may be inflated by their presence.
  Sprint 3 will separate MM from artefact using directional features and
  bilateral cross-correlation. Here, `outlier` is a tool, not a diagnosis.
- `k=7` is conservative: we prefer keeping some residual artefacts in the calm
  pool over emptying the baseline. Sprint 3 benefits from a wider calm pool.
