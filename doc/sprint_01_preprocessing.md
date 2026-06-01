# Sprint 1, Preprocessing (dual-path ENMO + Y/Z bandpass)

## Goal

Transform the raw signal, which is dominated by gravity and has a noisy X axis,
into two views useful for later analyses:

- **Path A, scalar ENMO**, to measure gravity-free, rotation-invariant movement
  amount.
- **Path B, Y/Z bandpassed signals**, for directional analysis and bilateral
  cross-correlation. The X axis is excluded from directional features but kept
  in plots with low opacity as visual reference.

## Design Choices

- **ENMO** (Van Hees et al.): `max(sqrt(ax^2 + ay^2 + az^2) - 1, 0)`.
  Zero clipping is enabled by default, following the canonical formulation.
  `A1.preprocessing.compute_enmo` was reused for consistency with the legacy
  base.
- **Fourth-order Butterworth bandpass** with scipy `sosfiltfilt`, zero phase,
  which is critical for the bilateral cross-correlation used in Sprint 3.
  Cutoffs are 0.5-15 Hz: 0.5 Hz removes drift and residual gravity bias, while
  15 Hz removes high-frequency noise. The band is intentionally broad because
  MM can be either fast or slow.
- **Jerk magnitude** is computed as `np.gradient(yz_bp, dt)` followed by the
  Euclidean norm over the two retained axes. Central differences preserve time
  alignment for plotting and windowing.
- **Multi-scale windows** `{0.5, 1, 2, 3, 5}` s are prepared but not yet used in
  this sprint.
- **Transparent X axis** in plots (`alpha=0.25`), full Y/Z opacity
  (`alpha=0.85`): the clinical hint is respected without changing ENMO, which
  by definition uses all three axes.
- **Fixed scales** (`PLOT_ENMO_YLIM=[0,4]`, `PLOT_YZ_FILTERED_YLIM=[-2,2]`,
  `PLOT_JERK_YLIM=[0,60]`) for cross-patient comparability.
- **Modular code** split across six files (`gravity`, `filters`, `derivatives`,
  `axes`, `windowing`, `pipeline`), with single-responsibility functions driven
  by `config.py`.

## Checks

- Sinusoid sanity check: input 0.2 Hz + 5 Hz + 25 Hz. The bandpass preserves
  5 Hz (RMS ratio 1.006), attenuates 0.2 Hz to 1.2%, and attenuates 25 Hz to
  17% (about 14 dB, fourth-order Butterworth transition band).
- Pipeline applied to all 45 patients, producing 5 x 2 PNGs per patient under
  `doc/figures/sprint_01/{group}/`.
- `summary.csv` contains statistics per patient/session: ENMO mean/peak/RMS,
  jerk mean/peak, and dominant frequency.

## Data Observations

1. **Gravity removal works**: ENMO mean now ranges from 0.0001 to 0.058 g
   (580x range) across sessions, while raw RMS was almost identical (0.58 g)
   for all patients. The Sprint 0 hypothesis was correct: raw signal statistics
   were not informative, ENMO is.
2. **Within-patient session asymmetry is the strongest signal**:
   - UCP14: ENMO mean `dom-active=0.0196` vs `ndom-active=0.0001`, ratio 198x.
     The ndom hand is highly active when the dom hand performs the task, while
     the dom hand is almost still in the opposite session.
   - UCP9: 14.7x, UCP4: 12.3x, UCP6: 8.8x, all suggesting evident MM during
     `dom-active`.
   - UCP13 and UCP5 peak during `ndom-active`.
   - UCP2, UCP11, and UCP3 have ratio <= 1.4, suggesting noisy baseline rather
     than session-specific MM.
3. **Dominant frequency** of bandpassed ENMO is typically in [1, 5] Hz, matching
   BBT cadence plus tremor-like activity, with a few extremes at 7-8 Hz
   (UCP7, UCP13).
4. **Jerk peak reveals isolated artefacts**: UCP4 `dom-active` has a 100 g/s
   single spike, likely a brief arm raise, to be excluded in Sprint 2.
5. **Y vs Z**: qualitative inspection confirms that Y and Z often look similar
   but shifted, consistent with wrist rotation. ENMO absorbs this rotation,
   while Y/Z bandpassed signals preserve it for directional analyses.

## Patients for Timeseries Inspection

**Likely strong MM**, PNGs under `figures/sprint_01/ucp/`:

- **UCP14**, `dom-active`, ratio 198x, ndom signal against near-zero baseline.
- **UCP9**, **UCP4**, **UCP6**, `dom-active`, ratio 8-15x.
- **UCP13**, **UCP5**, `ndom-active`, inverted asymmetry.
- **UCP0**, **UCP12**, **UCP1**, `dom-active`, less extreme but clear.

**Noisy intra-patient baseline**, to be handled in Sprint 2:

- **UCP16**, **UCP3**, **UCP15**, bilaterally active hard cases for the
  intra-patient artefact filter.
- **UCP4** has one 100 g/s jerk spike, a point artefact rather than MM.

**Likely no-MM or minimal-MM**

- **UCP2**, both sessions, extremely low activity.

## Outputs

- Code: `explainable/preprocessing/{gravity, filters, derivatives, axes, windowing, pipeline}.py`
  plus updates to `config.py`.
- 45 PNGs under `doc/figures/sprint_01/{ucp,td}/`.
- `doc/figures/sprint_01/summary.csv` (90 rows).
- Sinusoid sanity check included in the demo; execution stops if it fails.
