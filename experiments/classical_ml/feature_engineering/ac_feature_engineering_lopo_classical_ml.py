#!/usr/bin/env python3
"""
Mirror Movements classification - LOPO on precomputed feature-engineered datasets
---------------------------------------------------------------------------------
This script reuses the already feature-engineered datasets:
- ac_wide_timeseries.csv
- ac_session_features.csv
- ac_subject_features.csv

It keeps exactly the same evaluation logic and models as the previous script,
changing only the data source.

Datasets evaluated:
- session_level_all  -> all rows from ac_session_features.csv
- session_level_dom  -> rows with session == "dom"
- session_level_ndom -> rows with session == "ndom"
- subject_level      -> all rows from ac_subject_features.csv

Models:
- Logistic Regression
- SVM (RBF)
- Random Forest
- XGBoost (if available)

Outputs:
- results_from_feature_datasets/results_all.csv
- results_from_feature_datasets/predictions_all.csv
- results_from_feature_datasets/best_models_top5.csv
- results_from_feature_datasets/summary.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42

WIDE_PATH = "Dataset/CSV/ac_wide_timeseries.csv"
SESSION_PATH = "Dataset/CSV/ac_session_features.csv"
SUBJECT_PATH = "Dataset/CSV/ac_subject_features.csv"

RESULTS_DIR = Path("results_from_feature_datasets")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

LABEL_MAP = {"td": 0, "ucp": 1}


def load_precomputed_datasets(
    wide_path: str | Path,
    session_path: str | Path,
    subject_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide_path = Path(wide_path)
    session_path = Path(session_path)
    subject_path = Path(subject_path)

    for p in [wide_path, session_path, subject_path]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    wide_df = pd.read_csv(wide_path)
    session_df = pd.read_csv(session_path)
    subject_df = pd.read_csv(subject_path)

    required_session = {"id", "type", "session"}
    required_subject = {"id", "type"}
    if not required_session.issubset(session_df.columns):
        raise ValueError(f"{session_path} must contain columns: {sorted(required_session)}")
    if not required_subject.issubset(subject_df.columns):
        raise ValueError(f"{subject_path} must contain columns: {sorted(required_subject)}")

    return wide_df, session_df, subject_df


def summarize_input_datasets(
    wide_df: pd.DataFrame,
    session_df: pd.DataFrame,
    subject_df: pd.DataFrame,
) -> Dict[str, object]:
    out = {
        "wide_shape": [int(wide_df.shape[0]), int(wide_df.shape[1])],
        "session_shape": [int(session_df.shape[0]), int(session_df.shape[1])],
        "subject_shape": [int(subject_df.shape[0]), int(subject_df.shape[1])],
        "n_subjects_session": int(session_df["id"].astype(str).nunique()),
        "n_subjects_subject": int(subject_df["id"].astype(str).nunique()),
        "session_counts": session_df["session"].astype(str).value_counts(dropna=False).to_dict(),
        "label_counts_session": session_df["type"].astype(str).value_counts(dropna=False).to_dict(),
        "label_counts_subject": subject_df["type"].astype(str).value_counts(dropna=False).to_dict(),
    }
    return out


def feature_family_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    meta_cols = {"id", "type", "session"}
    excluded_exact = {"n_seconds", "dom__n_seconds", "ndom__n_seconds"}
    numeric_cols = [
        c for c in df.columns
        if c not in meta_cols and c not in excluded_exact and pd.api.types.is_numeric_dtype(df[c])
    ]
    return {
        "active_only": [c for c in numeric_cols if "_active_" in c],
        "active_mirror": [c for c in numeric_cols if "_active_" in c or "_mirror_" in c],
        "bilateral_only": [c for c in numeric_cols if "_bilateral_" in c],
        "all_features": numeric_cols,
    }


def build_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=4,
            )),
        ])

    return models


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_dataset_lopo(df: pd.DataFrame, dataset_name: str, models: Dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = feature_family_columns(df)
    logo = LeaveOneGroupOut()

    y_str = df["type"].astype(str).str.lower().to_numpy()
    y = np.array([LABEL_MAP[v] for v in y_str], dtype=int)
    groups = df["id"].astype(str).to_numpy()

    metrics_rows = []
    predictions_rows = []

    for family_name, cols in families.items():
        if not cols:
            continue

        X = df[cols].to_numpy(dtype=float)
        X = np.nan_to_num(X)

        for model_name, model in models.items():
            y_pred = np.empty(len(df), dtype=int)
            fold_metrics = []

            for train_idx, test_idx in logo.split(X, y, groups):
                model_fold = clone(model)
                model_fold.fit(X[train_idx], y[train_idx])
                pred_fold = model_fold.predict(X[test_idx]).astype(int)
                y_pred[test_idx] = pred_fold
                fold_metrics.append(compute_fold_metrics(y[test_idx], pred_fold))

            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            acc_mean = float(np.mean([m["acc"] for m in fold_metrics]))
            acc_std = float(np.std([m["acc"] for m in fold_metrics], ddof=1))
            f1_mean = float(np.mean([m["f1"] for m in fold_metrics]))
            f1_std = float(np.std([m["f1"] for m in fold_metrics], ddof=1))
            rec_mean = float(np.mean([m["rec"] for m in fold_metrics]))
            rec_std = float(np.std([m["rec"] for m in fold_metrics], ddof=1))
            prec_mean = float(np.mean([m["prec"] for m in fold_metrics]))
            prec_std = float(np.std([m["prec"] for m in fold_metrics], ddof=1))

            metrics_rows.append({
                "dataset": dataset_name,
                "feature_family": family_name,
                "model": model_name,
                "n_samples": int(len(df)),
                "n_groups": int(pd.Series(groups).nunique()),
                "n_features": int(len(cols)),
                "accuracy": float(accuracy_score(y, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
                "macro_f1": float(f1_score(y, y_pred, average="macro")),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1": float(f1_score(y, y_pred, zero_division=0)),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "rec_mean": rec_mean,
                "rec_std": rec_std,
                "prec_mean": prec_mean,
                "prec_std": prec_std,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            })

            pred_df = df[["id", "type"]].copy()
            if "session" in df.columns:
                pred_df["session"] = df["session"]
            pred_df["dataset"] = dataset_name
            pred_df["feature_family"] = family_name
            pred_df["model"] = model_name
            pred_df["true_label"] = y
            pred_df["pred_label"] = y_pred
            predictions_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    preds_df = pd.concat(predictions_rows, ignore_index=True)
    return metrics_df, preds_df


def main() -> None:
    print("Loading precomputed datasets...")
    wide_df, session_df, subject_df = load_precomputed_datasets(
        WIDE_PATH, SESSION_PATH, SUBJECT_PATH
    )

    summary = summarize_input_datasets(wide_df, session_df, subject_df)

    datasets = {
        "session_level_all": session_df.copy(),
        "session_level_dom": session_df.loc[session_df["session"].astype(str).str.lower() == "dom"].reset_index(drop=True),
        "session_level_ndom": session_df.loc[session_df["session"].astype(str).str.lower() == "ndom"].reset_index(drop=True),
        "subject_level": subject_df.copy(),
    }

    models = build_models(random_state=RANDOM_STATE)

    all_results = []
    all_predictions = []

    for dataset_name, df in datasets.items():
        print(f"\n=== DATASET: {dataset_name} ===")
        metrics_df, preds_df = evaluate_dataset_lopo(df, dataset_name, models)
        all_results.append(metrics_df)
        all_predictions.append(preds_df)

    results_df = pd.concat(all_results, ignore_index=True).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    preds_df = pd.concat(all_predictions, ignore_index=True)

    best_models_df = (
        results_df.sort_values(
            ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    results_df.to_csv(RESULTS_DIR / "results_all.csv", index=False)
    preds_df.to_csv(RESULTS_DIR / "predictions_all.csv", index=False)
    best_models_df.to_csv(RESULTS_DIR / "best_models_top5.csv", index=False)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Top results per dataset ===")
    print(best_models_df.to_string(index=False))
    print(f"\nOutput saved in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
