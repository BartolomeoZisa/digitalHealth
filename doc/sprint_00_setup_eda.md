# Sprint 0, Setup + EDA

## Goal

Build the `explainable/` repository foundation, validate the input data format,
and perform qualitative signal inspection for each patient (17 UCP + 28 TD).
The purpose was to understand the recordings before designing features or
detectors.

## What Was Done

- Created the `explainable/` package skeleton with `io`, `preprocessing`,
  `features`, `detectors`, `artifact`, `explain`, `eda`, and `webapp`.
- Added `config.py` as the single source of truth for constants: paths, sample
  rate, plot folders, fixed axis scales (`PLOT_ACCEL_YLIM=[-6,6]g`,
  `PLOT_ACCEL_YLIM_STILL=[-2.5,2.5]g`, `PLOT_TIME_XLIM=[0,60]s`), and
  axis/hand colors.
- Added `io/schema.py`, a CSV validator for required columns, allowed groups,
  and allowed labels.
- Added `io/data_loader.py` with immutable dataclasses `HandSignal`, `Session`,
  and `PatientData`; the loader separates rows by `session` and `hand_type`,
  with timestamps aligned to `t=0` within each session.
- Added `eda/explore.py`, which produces one PNG per patient with 4 x 2
  subplots (ACTIVE/STILL x X/Y/Z and magnitude, for `dom-active` and
  `ndom-active`) using the global fixed scales. It also writes `summary.csv`
  with statistics per patient/session.

## Design Choices

- **No combined PNG**, as requested: one file per patient, identical scale for
  direct cross-patient comparison.
- **Magnitude plotted separately** at the bottom, with a 2 g visual reference
  line. This was only a naive visual outlier threshold; the real filter was
  deferred to Sprint 2.
- **Fixed scales centralized in `config`**, so plots cannot silently diverge.
- **Frozen dataclasses** plus `_read_group_csv` caching: pure functions and
  easier future testing.

## Checks

- Inferred `fs` = 80.00 Hz for all patients.
- Every session has 4801 samples = 60 s x 80 Hz + 1, confirming the nominal
  60 s duration.
- Dataset contains 17 UCP and 28 TD patients.

## Qualitative Observations

1. **Active-hand range**: typically within [-5, +5] g during full BBT movement.
   The still hand stays within about [-2, +2] g in most cases.
2. **Gravity dominates the raw signal**: raw acceleration has a non-zero mean
   because gravity contributes about 1 g on an orientation-dependent axis. Raw
   RMS is not informative for UCP vs TD (about 0.58 in both groups, ratio 1.02).
   Sprint 1 therefore needed gravity removal (ENMO or detrend/high-pass) before
   computing activity features.
3. **TD are not a clean baseline**: several TD still hands have peaks above 2 g
   (for example TD10 peak 3.34 g, TD12, TD27, TD4, TD16, TD15). This confirms
   the clinical hint that TD recordings also contain random motor artefacts
   such as arm raises or involuntary gestures. A robust outlier filter was
   needed in Sprint 2.
4. **Within-patient session asymmetry**: some UCP patients show little ndom
   activity during `dom-active`, but high dom activity during `ndom-active`
   (for example UCP16). The MM-relevant session can vary by patient.
5. **Marked heterogeneity**: still-hand outlier percentages range from 0% to
   0.25%, consistent with patient-specific gravity and movement differences.

## Patients Worth Manual Timeseries Inspection

Suggested visual inspection targets, using PNGs under `doc/figures/sprint_00/`:

**Likely strong or informative MM**

- **UCP4**, `dom-active`, ndom peak 3.26 g and outlier 0.10%, likely evident MM.
- **UCP16**, `ndom-active`, dom outlier 0.21% (highest UCP value) and peak 2.70 g.
- **UCP15**, `ndom-active`, dom peak 2.33 g.
- **UCP5**, `ndom-active`, peak 1.56 g with diffuse activity.
- **UCP1**, `dom-active`, ndom peak 2.03 g.

**Likely absent or minimal MM**, useful as UCP-negative calibration cases

- **UCP2**, both sessions, among the lowest still-hand RMS values, peak < 1.4 g.
- **UCP9**, `ndom-active`, peak 1.01 g.
- **UCP14**, `ndom-active`, peak 0.96 g.

**Noisy TD cases to exclude from a clean baseline in Sprint 2**

- TD12, TD10, TD27, TD4, TD16, TD15, all with still-hand peaks > 2 g.

## Conclusion

The loading and inspection pipeline works. All plots use fixed, comparable
scales. Raw statistics are not useful before gravity removal, so Sprint 1 had
to prioritize detrending/bandpass preprocessing before feature extraction. The
data confirmed both patient heterogeneity and outliers in TD recordings.

## Outputs

- `doc/figures/sprint_00/{ucp,td}/<patient>.png`, 45 figures.
- `doc/figures/sprint_00/summary.csv`, 90 rows (2 sessions x 45 patients).
- Code: `explainable/{__init__, config}.py`, `explainable/io/{schema,data_loader}.py`,
  `explainable/eda/explore.py`.
