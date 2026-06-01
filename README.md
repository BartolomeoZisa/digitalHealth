# Mirror Movement Detection

![Patient-level dispersion overview](doc/figures/sprint_05/dispersion_scatter_overlay.png)

## Overview

This repository contains the final explainable decision-support pipeline for detecting candidate Mirror Movements (MM) in bilateral wrist accelerometer recordings collected during the Box and Blocks Test (BBT).

The package analyzes one patient at a time. It does not implement a UCP-vs-TD classifier as its final output. Instead, it highlights short windows compatible with MM and attaches event-level explanations based on clinically interpretable signal features.

The workflow is designed for technical review and clinical validation: detections are candidate events, not standalone diagnoses.

## Repository Scope

This GitHub release contains the pipeline files directly at repository root. The anonymized data required by the demos is included under `data/`; earlier modeling experiments are not part of this release.

Recommended local layout:

```text
digitalHealth/
  README.md
  config.py
  artifact/
  data_io/
  detectors/
  doc/
  eda/
  explain/
  features/
  preprocessing/
  webapp/
  data/
    UpdatedData/
      ucp/bbt_ucp_raw_anon.csv
      td/bbt_td_raw_anon.csv
    upload_examples/
      ucp5_webapp_upload.csv
```

Runtime commands should be executed from the repository root.

## Features

- **Single-patient analysis:** Loads one bilateral BBT recording and processes `dom-active` and `ndom-active` sessions separately.
- **Schema validation:** Checks that uploaded or local CSV files match the expected accelerometer schema.
- **Signal preprocessing:** Computes ENMO, zero-phase Y/Z Butterworth bandpass filtering, jerk magnitude, and sliding windows.
- **Artifact exclusion:** Removes gross movements using iterative robust thresholds over 0.5 s, 1 s, and 2 s scales.
- **Interpretable features:** Uses asymmetry, bilateral synchrony, jerk correlation, ENMO peak, and jerk RMS.
- **Intra-patient detectors:** Fits robust quantile, Isolation Forest, and PCA reconstruction detectors on each patient's calm windows.
- **Median ensemble:** Combines detector scores with a conservative median aggregation.
- **Composite MM rule:** Applies explicit score, asymmetry, synchrony, artifact, and boundary criteria.
- **Patient-level geometry:** Reports a dispersion ratio on the `(asymmetry_index, xcorr_max)` plane as secondary evidence.
- **Streamlit webapp:** Provides demo-patient loading, one-patient CSV upload, event inspection, signal zooms, and scatter plots.
- **Reproducible EDA scripts:** Regenerates sprint figures, ablations, CSV summaries, and the final project report.

## Tech Stack

- **Python:** Core implementation language.
- **NumPy:** Numerical array operations.
- **Pandas:** CSV loading, validation, and tabular feature processing.
- **SciPy:** Butterworth filtering, zero-phase filtering, and signal utilities.
- **scikit-learn:** Isolation Forest, PCA reconstruction, robust preprocessing utilities, and evaluation helpers.
- **Matplotlib:** Static sprint figures.
- **Plotly:** Interactive webapp charts.
- **Streamlit:** Single-patient decision-support interface.

## Sources and Data

No external API is required by the explainable pipeline.

The anonymized dataset is included in `data/`. By default, `config.py` resolves data paths relative to the repository root:

```text
data/UpdatedData/ucp/bbt_ucp_raw_anon.csv
data/UpdatedData/td/bbt_td_raw_anon.csv
```

The expected raw CSV schema is:

```text
Accelerometer X
Accelerometer Y
Accelerometer Z
datetime
hand
hand_dominance
type
hand_type
session
hand_label
id
```

Expected categorical values:

| Column           | Values        | Meaning                                |
| ---------------- | ------------- | -------------------------------------- |
| `type`           | `ucp`, `td`   | Cohort label in the source dataset     |
| `hand_type`      | `dom`, `ndom` | Dominant or non-dominant hand identity |
| `session`        | `dom`, `ndom` | Hand active during the BBT session     |
| `hand`           | `L`, `R`      | Anatomical hand                        |
| `hand_dominance` | `L`, `R`      | Patient dominant hand                  |

For webapp uploads, the CSV must contain exactly one patient id. Both `dom` and `ndom` sessions are recommended; at least one valid bilateral session is required.

## Method

The final pipeline follows this sequence:

1. Load a bilateral BBT recording for one patient.
2. Split the recording into `dom-active` and `ndom-active` sessions.
3. Compute per-hand ENMO:

```text
ENMO(t) = max(sqrt(ax^2 + ay^2 + az^2) - 1, 0)
```

4. Apply a zero-phase fourth-order Butterworth bandpass on Y/Z axes over 0.5-15 Hz.
5. Compute jerk magnitude from the bandpassed Y/Z signals.
6. Build 1 s windows with 50% overlap.
7. Detect gross motor artifacts with iterative `median + 7*MAD` thresholds at 0.5 s, 1 s, and 2 s scales.
8. Extract the selected feature vector:

```text
asymmetry_index
xcorr_max
bilateral_jerk_corr
enmo_peak
jerk_mag_rms
```

9. Fit three detectors on the patient's calm windows:

```text
robust quantile detector
Isolation Forest
PCA reconstruction detector
```

10. Combine detector scores with the median ensemble.
11. Apply the final MM-candidate rule.
12. Compute the patient-level dispersion ratio as a secondary indicator.

## Final MM-Candidate Rule

A window is marked as `is_mm_candidate` when:

```text
score_median >= 0.70
AND asymmetry_index <= 0.60
AND xcorr_max >= 0.40
AND not is_artifact
AND not is_boundary
```

Where:

| Field             | Description                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| `score_median`    | Median score from the robust quantile, Isolation Forest, and PCA reconstruction detectors               |
| `asymmetry_index` | Signed energy asymmetry between active and still hand                                                   |
| `xcorr_max`       | Maximum bilateral cross-correlation within the configured lag range                                     |
| `is_artifact`     | Gross movement flag used as an exclusion mask                                                           |
| `is_boundary`     | First and last 3 seconds of each session, excluded because of BBT setup/cleanup and filter edge effects |

The selected cutoffs were chosen by grid ablation on the available cohort:

```text
sensitivity UCP = 82.4%
specificity TD = 89.3%
Youden J = 0.716
```

## Patient-Level Dispersion Indicator

The package computes a geometric dispersion ratio on the `(asymmetry_index, xcorr_max)` plane:

```text
dispersion_ratio = dispersion_ndom / dispersion_dom
```

This value is displayed in the webapp as secondary patient-level evidence. It is not a hard filter in the window-level MM rule.

## Getting Started

The commands below assume the layout described in `Repository Scope` and are executed from the repository root.

### Prerequisites

- Python 3.10 or newer.
- Access to the anonymized BBT CSV files listed in `Sources and Data`.
- A shell running from the repository root.

### Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the minimal runtime dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Webapp

Start Streamlit from the repository root:

```bash
streamlit run webapp/app.py
```

The app supports:

- Selecting demo patients from the local anonymized dataset.
- Uploading a one-patient CSV.
- Viewing MM-candidate counts by session.
- Inspecting event-level explanations.
- Viewing a 2 s signal zoom around a selected event.
- Comparing session scatter plots on the `(asymmetry_index, xcorr_max)` plane.

An upload-ready test file is available at:

```text
data/upload_examples/ucp5_webapp_upload.csv
```

Expected output for UCP5:

```text
dom  -> 0 MM candidates
ndom -> 13 MM candidates
dispersion_ratio -> about 1.78
```

### Reproducing Sprint Outputs

Run the sprint scripts from the repository root:

```bash
python -m eda.explore
python -m eda.preprocess_demo
python -m eda.artifact_demo
python -m eda.feature_demo
python -m eda.detector_demo
python -m eda.mm_rule_ablation
python -m eda.scatter_dispersion_demo
```

Outputs are written under this folder:

```text
doc/figures/
```

## Project Structure

| Path             | Purpose                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `config.py`      | Central configuration for paths, sampling rate, thresholds, feature sets, and plotting constants |
| `data_io/`       | CSV schema validation and patient/session loading                                                |
| `preprocessing/` | ENMO, Y/Z bandpass filtering, jerk, axis handling, and windowing                                 |
| `artifact/`      | Robust intra-patient artifact detection                                                          |
| `features/`      | Per-window features and patient-level scatter dispersion                                         |
| `detectors/`     | Robust quantile, Isolation Forest, PCA reconstruction, median ensemble, and MM rule              |
| `explain/`       | Feature glossary and event-level clinical explanations                                           |
| `eda/`           | Reproducible scripts for figures, ablations, and CSV summaries                                   |
| `webapp/`        | Streamlit single-patient decision-support app                                                    |
| `doc/`           | Sprint documentation, figures, CSV outputs, and final report                                     |

## Documentation and Outputs

| File                                              | Content                                                 |
| ------------------------------------------------- | ------------------------------------------------------- |
| `doc/final_report.md`                             | Final technical report for the explainable pipeline     |
| `doc/sprint_00_setup_eda.md`                      | Raw-signal inspection and setup notes                   |
| `doc/sprint_01_preprocessing.md`                  | Gravity removal, filtering, and preprocessing rationale |
| `doc/sprint_02_artifact_baseline.md`              | Artifact filtering design and ablation                  |
| `doc/sprint_03_features.md`                       | Feature engineering and statistical evidence            |
| `doc/sprint_04_detectors.md`                      | Detector design and ensemble scoring                    |
| `doc/sprint_05_ablation.md`                       | MM-rule threshold ablation                              |
| `doc/sprint_05b_scatter_dispersion.md`            | Patient-level dispersion indicator                      |
| `doc/sprint_06_webapp.md`                         | Streamlit webapp implementation notes                   |
| `doc/figures/sprint_04/summary_mm.csv`            | Per-session MM-candidate summary                        |
| `doc/figures/sprint_05/ablation/grid_results.csv` | Composite-rule cutoff ablation                          |
| `doc/figures/sprint_05/scatter_dispersion.csv`    | Patient-level dispersion values                         |

## Other Experiments

The full research workspace also contained earlier experiments outside this release. They are not part of this GitHub branch, but they document the research path that led to the current design.

| Area                            | Description                                                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Classical ML                    | Supervised UCP-vs-TD experiments on engineered features, including selected-feature variants and raw/preprocessed pipelines |
| Split generation                | Patient-level K-fold and leave-one-out split generation used to avoid subject leakage                                       |
| 1D deep learning                | ENMO-based CNN experiments: shared-weight two-branch CNN, independent-branch CNN, and CNN-LSTM cascade                      |
| Spectrogram deep learning       | STFT and EfficientNet experiments using dual-branch, signed-delta, and RGB-like spectrogram inputs                          |
| SimCLR and transfer experiments | Representation-learning and fine-tuning experiments evaluated against the supervised baselines                              |

## Issues and Known Constraints

- The final rule is calibrated on the available anonymized cohort and should be revalidated before use on a different acquisition protocol.
- The first and last 3 seconds of each session are excluded from MM-candidate decisions because of setup/cleanup transients and filter edge effects.
- A 60 s BBT recording may not contain visible MM events for every UCP patient.
- TD sessions can contain gross movement artifacts, so the pipeline uses intra-patient baselines instead of a TD-trained anomaly baseline.
- Upload mode accepts one patient id per CSV.
