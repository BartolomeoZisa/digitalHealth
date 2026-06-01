"""
Feature glossary for event explanations.

Each function maps a `(feature_name, signed_robust_z)` pair to a human
readable label or sentence. The wording avoids implementation jargon so the
same text can be used in plots and in the Streamlit app.
"""
from __future__ import annotations


HEADLINES: dict[str, str] = {
    "enmo_mean": "average still-hand motion",
    "enmo_peak": "still-hand motion peak",
    "yz_bp_rms": "filtered still-hand amplitude",
    "yz_bp_peak": "filtered still-hand peak",
    "jerk_mag_rms": "still-hand acceleration speed",
    "vec_mag_integral": "total still-hand motion",
    "yz_bp_zcr": "still-hand vibration rate",
    "spectral_centroid": "still-hand frequency centre",
    "dominant_freq": "still-hand dominant frequency",
    "band_power_slow": "slow motion energy (0.5-2 Hz)",
    "band_power_medium": "medium motion energy (2-6 Hz)",
    "band_power_fast": "fast motion energy (6-15 Hz)",
    "spectral_entropy": "still-hand frequency spread",
    "xcorr_max": "hand sync",
    "xcorr_lag_ms": "hand sync delay",
    "asymmetry_index": "stillness balance",
    "bilateral_jerk_corr": "joint-onset alignment",
}

# Kept as an alias for older call sites.
HEADLINES_EN = HEADLINES


def headline(name: str) -> str:
    return HEADLINES.get(name, name)


def headline_en(name: str) -> str:
    return headline(name)


def _intensity_en(z: float) -> str:
    z = abs(z)
    if z >= 5:
        return "strong"
    if z >= 2.5:
        return "moderate"
    return "mild"


def clinical_sentence(name: str, robust_z: float) -> str:
    """Short English sentence describing a feature contribution.

    `robust_z > 0` means the feature is above the patient baseline, and
    `robust_z < 0` means it is below the patient baseline.
    """
    intensity = _intensity_en(robust_z)

    if name == "asymmetry_index":
        if robust_z < 0:
            return (
                "Still hand moves with energy comparable to the active hand "
                f"({intensity} signal), a typical mirror pattern."
            )
        return (
            f"Active hand clearly dominates ({intensity} signal), this window "
            "is unlikely to be a mirror movement."
        )

    if name == "xcorr_max":
        if robust_z > 0:
            return (
                "Both hands move in sync within about 100 ms "
                f"({intensity} signal), consistent with mirror coupling."
            )
        return f"Both hands move with low coordination ({intensity} signal)."

    if name == "bilateral_jerk_corr":
        if robust_z > 0:
            return (
                "Sudden movements occur on both hands at the same moment "
                f"({intensity} signal)."
            )
        return f"Sudden movements are not aligned between hands ({intensity})."

    if name in {
        "enmo_peak",
        "enmo_mean",
        "yz_bp_rms",
        "yz_bp_peak",
        "vec_mag_integral",
    }:
        if robust_z > 0:
            return (
                "The still hand moves more than usual for this patient "
                f"({intensity} signal)."
            )
        return f"The still hand is unusually quiet ({intensity} signal)."

    if name == "jerk_mag_rms":
        if robust_z > 0:
            return (
                "Acceleration changes faster than usual on the still hand "
                f"({intensity} signal)."
            )
        return f"Acceleration changes are very smooth ({intensity} signal)."

    if name in {
        "band_power_slow",
        "band_power_medium",
        "band_power_fast",
        "spectral_centroid",
        "dominant_freq",
        "spectral_entropy",
    }:
        direction = "higher" if robust_z > 0 else "lower"
        return (
            f"{HEADLINES[name].capitalize()} is {direction} than the "
            f"patient baseline ({intensity} signal)."
        )

    return (
        f"{HEADLINES.get(name, name).capitalize()} differs from baseline "
        f"({intensity} signal)."
    )


def clinical_sentence_en(name: str, robust_z: float) -> str:
    return clinical_sentence(name, robust_z)
