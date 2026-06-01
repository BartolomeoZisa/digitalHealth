"""
CSV schema definition and validation for raw bilateral accelerometer recordings.

A valid recording file must contain one row per (timestamp, hand) pair with the
columns listed below. Both UCP and TD groups share the same schema.
"""
from __future__ import annotations

import pandas as pd

# Column names exactly as found in the source CSVs.
COL_AX = "Accelerometer X"
COL_AY = "Accelerometer Y"
COL_AZ = "Accelerometer Z"
COL_DT = "datetime"
COL_HAND = "hand"                  # L | R (anatomical)
COL_HAND_DOMINANCE = "hand_dominance"  # L | R (patient's dominant hand)
COL_TYPE = "type"                  # ucp | td
COL_HAND_TYPE = "hand_type"        # dom | ndom (per row, this row's hand)
COL_SESSION = "session"            # dom | ndom (which hand was active in test)
COL_HAND_LABEL = "hand_label"      # mirror of hand_type
COL_ID = "id"                      # patient id, e.g. UCP0, TD3

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_AX, COL_AY, COL_AZ, COL_DT, COL_HAND, COL_HAND_DOMINANCE,
    COL_TYPE, COL_HAND_TYPE, COL_SESSION, COL_HAND_LABEL, COL_ID,
)

ALLOWED_HAND = frozenset({"L", "R"})
ALLOWED_GROUP = frozenset({"ucp", "td"})
ALLOWED_SESSION = frozenset({"dom", "ndom"})
ALLOWED_HAND_TYPE = frozenset({"dom", "ndom"})


class SchemaError(ValueError):
    """Raised when a dataframe does not satisfy the expected schema."""


def validate_dataframe(df: pd.DataFrame, *, expected_group: str | None = None) -> None:
    """Raise SchemaError if df does not match the required schema.

    Parameters
    ----------
    df:
        Dataframe loaded from one of the raw CSVs.
    expected_group:
        Optional 'ucp' or 'td'; if provided, the `type` column must be uniform
        and match this value.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    if df.empty:
        raise SchemaError("dataframe is empty")

    bad_hand = set(df[COL_HAND].unique()) - ALLOWED_HAND
    if bad_hand:
        raise SchemaError(f"unexpected values in '{COL_HAND}': {bad_hand}")

    bad_group = set(df[COL_TYPE].unique()) - ALLOWED_GROUP
    if bad_group:
        raise SchemaError(f"unexpected values in '{COL_TYPE}': {bad_group}")

    bad_session = set(df[COL_SESSION].unique()) - ALLOWED_SESSION
    if bad_session:
        raise SchemaError(f"unexpected values in '{COL_SESSION}': {bad_session}")

    bad_ht = set(df[COL_HAND_TYPE].unique()) - ALLOWED_HAND_TYPE
    if bad_ht:
        raise SchemaError(f"unexpected values in '{COL_HAND_TYPE}': {bad_ht}")

    if expected_group is not None:
        groups = set(df[COL_TYPE].unique())
        if groups != {expected_group}:
            raise SchemaError(
                f"expected group '{expected_group}' but found {groups}"
            )
