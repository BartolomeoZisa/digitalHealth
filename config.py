"""
Centralized configuration for the explainable MM detection pipeline.

All hyperparameters, paths, and plot conventions live here so that no magic
numbers leak into the rest of the codebase.

Memory directive: every multi-patient comparable plot must use FIXED axis
scales taken from this module (see PLOT_* constants).
"""
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data" / "UpdatedData"
UCP_CSV: Path = DATA_DIR / "ucp" / "bbt_ucp_raw_anon.csv"
TD_CSV: Path = DATA_DIR / "td" / "bbt_td_raw_anon.csv"

PKG_DIR: Path = Path(__file__).resolve().parent
DOC_DIR: Path = PKG_DIR / "doc"
FIG_DIR: Path = DOC_DIR / "figures"
RESULTS_DIR: Path = ROOT / "results"

# -----------------------------------------------------------------------------
# Acquisition
# -----------------------------------------------------------------------------
SAMPLE_RATE: int = 80                  # Hz, verified via inter-sample dt = 12.5 ms
SESSION_DURATION_S: float = 60.0       # nominal duration of one BBT session
EXPECTED_ROWS_PER_SESSION: int = 4801  # 60 s * 80 Hz + 1

GROUPS: tuple[str, str] = ("ucp", "td")
SESSION_LABELS: tuple[str, str] = ("dom", "ndom")    # which hand was active
HAND_TYPES: tuple[str, str] = ("dom", "ndom")        # per-row hand identity

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
# Bandpass cutoffs for the per-axis directional path (Y, Z). Lower bound
# removes residual drift / slow postural shifts; upper bound removes
# high-frequency noise above plausible motor frequencies.
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 15.0
FILTER_ORDER: int = 4                  # Butterworth filter order

# Gravity removal: ENMO = max(sqrt(ax^2+ay^2+az^2) - 1, 0). Clipping at 0 is
# the canonical formulation (Van Hees et al.). We keep clipping ON by default
# but expose both clipped and signed variants in preprocessing/gravity.py.
ENMO_CLIP_ZERO: bool = True

# Directional analysis (Sprint 3 features) uses Y and Z only, clinical hint
# says X carries most of the noise. ENMO itself still uses XYZ standard.
KEEP_AXES_FOR_DIRECTIONAL: tuple[str, ...] = ("y", "z")
DROP_X_AXIS_DEFAULT: bool = True       # for feature engineering only

# Multi-scale windows (seconds), MM kinematics unknown a priori, so we keep
# both short (fast MM) and long (slow MM) scales available.
WINDOW_SIZES_S: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_WINDOW_S: float = 1.0
DEFAULT_OVERLAP: float = 0.75

# -----------------------------------------------------------------------------
# Plot conventions (FIXED scales, never use auto-scaling for comparable plots)
# -----------------------------------------------------------------------------
# Accelerometer signals are in g. Empirical inspection shows |a| < ~3 g for
# normal activity; a wider symmetric range guarantees no clipping while keeping
# patients visually comparable.
PLOT_ACCEL_YLIM: tuple[float, float] = (-6.0, 6.0)
# Tight scale for the STILL hand (where MM live). Inspected empirically: still
# hand rarely exceeds ~2 g; tighter ylim makes small MM more visible.
PLOT_ACCEL_YLIM_STILL: tuple[float, float] = (-2.5, 2.5)
PLOT_TIME_XLIM: tuple[float, float] = (0.0, SESSION_DURATION_S)
PLOT_DPI: int = 110
PLOT_FIGSIZE_PATIENT: tuple[float, float] = (14.0, 8.0)

# Fixed scales for Sprint 1 derived signals (in their natural units).
PLOT_ENMO_YLIM: tuple[float, float] = (0.0, 4.0)              # g, after gravity removal
PLOT_YZ_FILTERED_YLIM: tuple[float, float] = (-2.0, 2.0)      # g, bandpassed
PLOT_JERK_YLIM: tuple[float, float] = (0.0, 60.0)             # g / s

# Per-axis colour map kept consistent across the project.
AXIS_COLORS: dict[str, str] = {
    "x": "#d62728",   # red   , the noisy axis
    "y": "#1f77b4",   # blue
    "z": "#2ca02c",   # green
}

# Alpha for the noisy X axis when plotted alongside Y/Z: low so it does not
# dominate the visual but stays available as reference.
PLOT_X_ALPHA: float = 0.25
PLOT_YZ_ALPHA: float = 0.85

# Hand colour map (used wherever dom/ndom are overlaid).
HAND_COLORS: dict[str, str] = {
    "dom": "#ff7f0e",     # orange, active hand
    "ndom": "#1f77b4",    # blue  , still hand (target for MM detection)
}

# -----------------------------------------------------------------------------
# Feature engineering (Sprint 3)
# -----------------------------------------------------------------------------
# Bands (Hz) for spectral power features on the still hand.
FREQ_BANDS_HZ: dict[str, tuple[float, float]] = {
    "slow":   (0.5, 2.0),    # BBT cadence range
    "medium": (2.0, 6.0),    # plausible MM range
    "fast":   (6.0, 15.0),   # tremor-like range
}

# Bilateral mirror-score cross-correlation: maximum lag tested (ms).
# 200 ms covers reaction-time-like delays plausible for mirror movements.
XCORR_MAX_LAG_MS: float = 200.0

# Temporal-distribution features (Family 4) operate on the outlier mask of a
# session. A burst is a contiguous group of outlier windows; two bursts are
# considered separate when the gap between them exceeds this many seconds.
BURST_GAP_S: float = 3.0

# Number of bins used to compute the Shannon entropy of the outlier time
# distribution across the 60-second session.
TEMPORAL_ENTROPY_BINS: int = 10

# Active feature names, toggled here to enable/disable from a single place.
# Empty tuple means "use all registered features".
FEATURE_REGISTRY_ENABLED: tuple[str, ...] = ()

# Plot conventions for Sprint 3.
PLOT_GINI_YLIM: tuple[float, float] = (0.0, 1.0)
PLOT_BURSTINESS_YLIM: tuple[float, float] = (-1.0, 1.0)
PLOT_TEMP_ENTROPY_YLIM: tuple[float, float] = (0.0, 1.0)   # normalised by log(N_bins)
PLOT_NBURSTS_YLIM: tuple[float, float] = (0.0, 20.0)
PLOT_GROUP_COLORS: dict[str, str] = {
    "ucp": "#d62728",   # red
    "td":  "#1f77b4",   # blue
}

# -----------------------------------------------------------------------------
# Detectors (Sprint 4)
# -----------------------------------------------------------------------------
# k for robust-quantile detector threshold (median + k*MAD per feature).
DETECTOR_KMAD: float = 5.0

# IsolationForest hyperparameters.
IFOREST_N_ESTIMATORS: int = 200
IFOREST_MAX_SAMPLES: int = 256

# PCA reconstruction detector, max components used when fitting calm pool.
PCA_N_COMPONENTS_MAX: int = 8

# Anomaly score above which a window is tagged as a likely MM event in the UI.
ENSEMBLE_THRESHOLD: float = 0.7

# Sprint 5, boundary trim. The first and last `BOUNDARY_TRIM_S` seconds of
# each session are tagged `is_boundary=True` because they contain the
# setup/cleanup transients of the BBT (hand placement, hand removal) plus
# the edge transient of the zero-phase Butterworth filter. They are kept in
# the data (so the detector statistics remain comparable) but excluded from
# `is_mm_candidate`.
BOUNDARY_TRIM_S: float = 3.0

# Composite MM rule cutoffs, initial defaults; finalised by
# `eda/mm_rule_ablation.py` and written into `chosen_thresholds.txt`.
# Chosen by Sprint-5 ablation (`eda/mm_rule_ablation.py`):
#   Youden J = 0.716, sensitivity_ucp = 0.824, specificity_td = 0.893.
ASYM_MM_CUTOFF: float = 0.6     # asymmetry_index <= this → still hand participates
XCORR_MM_CUTOFF: float = 0.4    # xcorr_max >= this → bilateral synchrony

# Feature sets for the Sprint-4 ablation. The "selected" subset is the one
# the Sprint-3 evidence flagged as most discriminative.
DETECTOR_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full": (
        "enmo_mean", "enmo_peak", "yz_bp_rms", "yz_bp_peak",
        "jerk_mag_rms", "vec_mag_integral", "yz_bp_zcr",
        "spectral_centroid", "dominant_freq",
        "band_power_slow", "band_power_medium", "band_power_fast",
        "spectral_entropy",
        "xcorr_max", "xcorr_lag_ms", "asymmetry_index", "bilateral_jerk_corr",
    ),
    "selected": (
        "asymmetry_index",
        "xcorr_max",
        "bilateral_jerk_corr",
        "enmo_peak",
        "jerk_mag_rms",
    ),
}

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42
