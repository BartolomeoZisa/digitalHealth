# Sprint 3, Feature Engineering (4 Families, Intra-Patient)

## Goal

Build clinically interpretable features on top of Sprint 1 preprocessing and
the Sprint 2 outlier mask, evaluate them cross-patient, and identify the most
promising features for the Sprint 4 detector. This sprint directly tested the
clinical hypothesis raised in Sprint 2: TD outliers are clustered, while UCP
outliers are sparse.

## Architecture

`explainable/features/` contains six modules plus a registry pattern:

- `time_domain.py`, Family 1, per-window activity: RMS, peak, `jerk_rms`,
  `vec_mag_integral`, ZCR, ENMO mean/peak.
- `frequency.py`, Family 2, per-window spectral features: spectral centroid,
  dominant frequency, band power in three bands, spectral entropy.
- `bilateral.py`, Family 3, per-window mirror score: `xcorr_max` dom/ndom,
  lag at max correlation in ms, signed asymmetry index, bilateral jerk
  correlation. It reuses legacy `feature_extraction.py::_max_abs_xcorr`.
- `temporal.py`, Family 4, per-session temporal descriptors: outlier coverage
  percentage, number of bursts, burst durations, `gini_temporal`,
  `temporal_entropy`, Goh-Barabasi `burstiness_B`, and `autocorr_lag1`.
- `registry.py`, `dict[name -> (family, callable)]` plus a config toggle
  (`FEATURE_REGISTRY_ENABLED`).
- `extractor.py`, the orchestrator: `ProcessedSession` +
  `PatientArtifactReport` -> DataFrame (`n_windows x n_features` plus
  `is_outlier`) and temporal dictionary.

`explainable/eda/feature_demo.py` runs the full pipeline plus a bilateral sanity
check with a shifted Gaussian pulse: `corr=0.999`, `lag=+/-125 ms`, pass.

## Cross-Patient Results (45 UCP + TD)

**Mann-Whitney p-values for UCP vs TD on aggregate features** (`outlier` means
windows with `is_outlier=True`):

| Feature | UCP median | TD median | MW p |
|---|---:|---:|---:|
| **`asymmetry_index_mean_out`** | **0.477** | **0.733** | **0.0005** |
| `asymmetry_index_mean_calm` | 0.759 | 0.938 | 0.00001 |
| `outlier_coverage_pct` | 16.66% | 13.12% | 0.0324 |
| `xcorr_max_mean_out` | 0.387 | 0.339 | 0.485 |
| `xcorr_max_mean_calm` | 0.355 | 0.336 | 0.201 |
| `bilateral_jerk_corr_mean_out` | -0.005 | -0.015 | 0.266 |
| `dominant_freq_mean_out` | 3.57 Hz | 3.36 Hz | 0.667 |
| `gini_temporal` | 0.500 | 0.600 | 0.165 |
| `burstiness_B` | -0.495 | -0.481 | 0.646 |
| `temporal_entropy` | 1.000 | 1.000 | 0.309 |
| `n_bursts` | 5 | 4 | 0.387 |

## Main Findings

1. **Winning feature: `asymmetry_index`**, signed RMS asymmetry between active
   and still hand.
   - On `outlier` windows: UCP 0.48 vs TD 0.73 (MW p=0.0005).
   - On `calm` windows: same pattern, UCP 0.76 vs TD 0.94 (p about 0).
   - **Clinical rationale**: TD outliers usually mean only the active hand
     moves (`asym` close to +1, the active-only limit). UCP outliers have still
     hand energy comparable to active-hand energy (`asym` about +0.5), a direct
     MM signature.
   - The same pattern in `calm` windows suggests continuous sub-clinical mirror
     residuals in UCP.

2. **Maximum cross-correlation does not discriminate alone** (p=0.49 on
   outliers).
   - Higher UCP examples: UCP5 0.56, UCP2 0.53, UCP6 0.51.
   - Comparable TD examples: TD9 0.57, TD14 0.56.
   - At single-window level, mirror-like waveform similarity is confounded by
     noise, drift, and partial bilateral gestures. It must be combined with
     asymmetry.

3. **The Sprint 2 temporal hypothesis was falsified on aggregate data**.
   - `gini_temporal`, `burstiness_B`, `temporal_entropy`, and `n_bursts` are not
     significant.
   - UCP16 and TD12, the clinician's two focal cases, have nearly identical
     temporal values (gini 0.7 in both, B about -0.77 vs -0.80, entropy 1.0).
   - The visual observation "TD clustered vs UCP sparse" is not captured by the
     current temporal metrics. Possible causes: arbitrary 3 s burst gap, high
     intra-group variance, or the visual difference reflecting total outlier
     level more than temporal distribution.

4. **Outlier coverage is weakly discriminative** (p=0.032): UCP 16.7% vs TD
   13.1%. This confirms that UCP spend slightly more time in an outlier-like
   state, but overlap remains strong.

5. **Spectral features are weak alone**: dominant frequency, entropy, and band
   powers all have p > 0.5.

## Patients for Manual Inspection

To clinically validate `asymmetry_index_mean_out`:

- **UCP5 `dom-active`**: `asym_out=0.54`, `xcorr=0.56`, `jerk_corr=0.23`, high
  MM suspicion.
- **UCP2 `ndom-active`**: `asym_out=0.64`, `xcorr=0.53`.
- **UCP6 `dom-active`**: `asym_out=0.52`, lag 7.5 ms, almost instantaneous.
- **UCP8 `dom-active`**: `asym_out=0.13`, very close to 0, plus `xcorr=0.47`,
  strong bilateral synchrony.
- **TD2 / TD14**: high `asym_out` (0.85 / 0.77) but `xcorr` 0.49 / 0.56,
  control cases showing that TD with high correlation can remain asymmetric.

## End-of-Sprint Verification

- Bilateral sanity check: PASS (`corr=0.999`, lag magnitude 125 ms on a shifted
  aperiodic pulse).
- Feature extractor runs on 45 patients, 119 windows/session, 17 features per
  window plus 11 temporal descriptors per session.
- Cross-patient plot produced: `cross_patient_temporal.png`, visually
  confirming UCP/TD overlap on Gini/B.
- Outputs: `summary_features.csv` (90 rows), `temporal_features.csv` (90 rows).

## Consequences for Sprint 4

- The primary detector should focus on per-window `asymmetry_index`, detecting
  UCP patient windows where asymmetry drops below the intra-patient median by
  more than `k*MAD`.
- Combine with `xcorr_max` to reduce false positives: low asymmetry plus high
  cross-correlation means MM; low asymmetry alone remains ambiguous.
- Temporal features can remain output descriptors for the clinician, but not
  the main discriminative signal.
- Spectral features remain explanatory attribution candidates, not detector
  drivers.
