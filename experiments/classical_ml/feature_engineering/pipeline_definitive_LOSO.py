#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


REQUIRED_COLUMNS = {
    "Axis1", "Axis2", "Axis3", "datetime",
    "type", "session", "hand_label", "id",
}


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible wrapper around xgboost.XGBClassifier.

    XGBoost works most robustly with integer labels. This wrapper lets the rest of
    the pipeline keep the original labels ("td", "ucp") and converts them only
    inside the estimator. This is useful because the metrics and saved predictions
    remain readable.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        auto_scale_pos_weight: bool = True,
        scale_pos_weight: Optional[float] = None,
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.auto_scale_pos_weight = auto_scale_pos_weight
        self.scale_pos_weight = scale_pos_weight
        self.verbosity = verbosity

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost non è installato. Installa la dipendenza con: pip install xgboost"
            ) from exc

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        scale_pos_weight = self.scale_pos_weight
        if self.auto_scale_pos_weight and scale_pos_weight is None and len(self.classes_) == 2:
            counts = np.bincount(y_enc, minlength=2)
            n_negative = counts[0]
            n_positive = counts[1]
            scale_pos_weight = float(n_negative / n_positive) if n_positive > 0 else 1.0

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            objective="binary:logistic",
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=scale_pos_weight,
            verbosity=self.verbosity,
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc_pred = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(np.asarray(y_enc_pred, dtype=int))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Movements pipeline for Activity Count data.")
    parser.add_argument("--td", required=True, help="Path al CSV TD activity count (1 sec).")
    parser.add_argument("--ucp", required=True, help="Path al CSV UCP activity count (1 sec).")
    parser.add_argument("--outdir", default="results_ac_xgb", help="Cartella di output.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed randomica.")
    parser.add_argument("--rf-n-estimators", type=int, default=200, help="Numero di alberi Random Forest.")
    parser.add_argument("--max-lag", type=int, default=3, help="Lag massimo per cross-correlation.")
    parser.add_argument("--knn-k", type=int, default=5, help="Numero di vicini per KNN.")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Numero di alberi boosting XGBoost.")
    parser.add_argument("--xgb-max-depth", type=int, default=2, help="Profondità massima alberi XGBoost.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="Learning rate XGBoost.")
    return parser.parse_args()


def load_single_csv(path: str | Path, expected_type: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} non contiene le colonne richieste: {sorted(missing)}")

    if df["type"].nunique() != 1:
        raise ValueError(f"{path} contiene più label nel campo 'type': {df['type'].unique()}")

    found_type = str(df["type"].iloc[0]).lower()
    if found_type != expected_type.lower():
        raise ValueError(f"{path} ha type='{found_type}', atteso '{expected_type}'")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        raise ValueError(f"{path} contiene datetime non validi")

    for col in ["Axis1", "Axis2", "Axis3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["Axis1", "Axis2", "Axis3"]].isna().any().any():
        raise ValueError(f"{path} contiene valori non numerici o NaN nelle colonne Axis")

    return df


def load_activity_count_data(td_path: str | Path, ucp_path: str | Path) -> pd.DataFrame:
    td = load_single_csv(td_path, "td")
    ucp = load_single_csv(ucp_path, "ucp")
    df = pd.concat([td, ucp], ignore_index=True)

    df["VM"] = np.sqrt(df["Axis1"] ** 2 + df["Axis2"] ** 2 + df["Axis3"] ** 2)
    df["Total"] = df["Axis1"] + df["Axis2"] + df["Axis3"]
    return df


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, int]:
    counts = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["id"].nunique()),
        "n_td_subjects": int(df.loc[df["type"] == "td", "id"].nunique()),
        "n_ucp_subjects": int(df.loc[df["type"] == "ucp", "id"].nunique()),
        "n_sessions_total": int(df[["id", "session"]].drop_duplicates().shape[0]),
    }

    per_subject_session_hand = (
        df.groupby(["id", "session", "hand_label"]).size().reset_index(name="n_rows")
    )
    counts["min_rows_per_subject_session_hand"] = int(per_subject_session_hand["n_rows"].min())
    counts["max_rows_per_subject_session_hand"] = int(per_subject_session_hand["n_rows"].max())
    counts["expected_seconds_detected"] = int(per_subject_session_hand["n_rows"].mode().iloc[0])
    return counts


def make_wide_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["Axis1", "Axis2", "Axis3", "VM", "Total"]
    wide = (
        df.pivot_table(
            index=["id", "type", "session", "datetime"],
            columns="hand_label",
            values=value_cols,
            aggfunc="first",
        )
        .sort_index()
    )

    wide.columns = [f"{signal}_{hand}" for signal, hand in wide.columns]
    wide = wide.reset_index().sort_values(["id", "session", "datetime"]).reset_index(drop=True)

    for signal in value_cols:
        wide[f"{signal}_active"] = np.where(
            wide["session"].eq("dom"),
            wide[f"{signal}_dom"],
            wide[f"{signal}_ndom"],
        )
        wide[f"{signal}_mirror"] = np.where(
            wide["session"].eq("dom"),
            wide[f"{signal}_ndom"],
            wide[f"{signal}_dom"],
        )

    return wide


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def max_abs_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> Tuple[float, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    best_corr = None
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[:lag], b[-lag:]
        elif lag > 0:
            aa, bb = a[lag:], b[:-lag]
        else:
            aa, bb = a, b
        if len(aa) < 2:
            continue
        corr = safe_corr(aa, bb)
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    if best_corr is None:
        return 0.0, 0
    return float(best_corr), int(best_lag)


def summarize_signal(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=0)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
        f"{prefix}_sum": float(np.sum(x)),
        f"{prefix}_nonzero_frac": float(np.mean(x > 0)),
    }


def extract_session_features(wide: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    signals = ["Axis1", "Axis2", "Axis3", "VM", "Total"]
    eps = 1e-8

    for (sid, label, session), grp in wide.groupby(["id", "type", "session"], sort=True):
        grp = grp.sort_values("datetime")
        row: Dict[str, float] = {
            "id": sid,
            "type": label,
            "session": session,
            "n_seconds": int(len(grp)),
        }

        for signal in signals:
            active = grp[f"{signal}_active"].to_numpy(dtype=float)
            mirror = grp[f"{signal}_mirror"].to_numpy(dtype=float)

            row.update(summarize_signal(active, f"{signal}_active"))
            row.update(summarize_signal(mirror, f"{signal}_mirror"))

            diff = active - mirror
            row[f"{signal}_bilateral_diff_mean"] = float(np.mean(diff))
            row[f"{signal}_bilateral_absdiff_mean"] = float(np.mean(np.abs(diff)))
            row[f"{signal}_bilateral_ratio_sum"] = float(np.sum(mirror) / (np.sum(active) + eps))
            row[f"{signal}_bilateral_asymmetry"] = float((np.sum(active) - np.sum(mirror)) / (np.sum(active) + np.sum(mirror) + eps))
            row[f"{signal}_bilateral_corr0"] = safe_corr(active, mirror)

            maxcorr, lag = max_abs_xcorr(active, mirror, max_lag=max_lag)
            row[f"{signal}_bilateral_maxcorr_abs"] = maxcorr
            row[f"{signal}_bilateral_lag_at_maxcorr"] = lag

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["id", "session"]).reset_index(drop=True)


def make_subject_level_dataset(session_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in session_df.columns if c not in {"id", "type", "session"}]
    subject_df = session_df.pivot(index=["id", "type"], columns="session", values=feature_cols)
    subject_df.columns = [f"{session}__{feature}" for feature, session in subject_df.columns]
    return subject_df.reset_index().sort_values("id").reset_index(drop=True)


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
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


def build_models(
    random_state: int = 42,
    knn_k: int = 5,
    rf_n_estimators: int = 200,
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 2,
    xgb_learning_rate: float = 0.05,
) -> Dict[str, Pipeline]:
    return {
        "knn": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=knn_k)),
        ]),
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=random_state)),
        ]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=rf_n_estimators, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBStringClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }


def evaluate_loso(df: pd.DataFrame, dataset_name: str, models: Dict[str, Pipeline]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    families = feature_families(df)
    logo = LeaveOneGroupOut()
    y = df["type"].astype(str).to_numpy()
    groups = df["id"].astype(str).to_numpy()

    metrics_rows: List[Dict[str, float]] = []
    predictions_rows: List[pd.DataFrame] = []

    for family_name, cols in families.items():
        if not cols:
            continue
        X = df[cols]

        for model_name, model in models.items():
            y_pred = np.empty(len(df), dtype=object)
            for train_idx, test_idx in logo.split(X, y, groups):
                model_fold = clone(model)
                model_fold.fit(X.iloc[train_idx], y[train_idx])
                y_pred[test_idx] = model_fold.predict(X.iloc[test_idx])

            cm = confusion_matrix(y, y_pred, labels=["td", "ucp"])
            tn, fp, fn, tp = cm.ravel()
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
            pred_df["pred_label"] = y_pred
            predictions_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    preds_df = pd.concat(predictions_rows, ignore_index=True)
    return metrics_df, preds_df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    datasets_dir = outdir / "datasets"
    metrics_dir = outdir / "metrics"
    predictions_dir = outdir / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)

    ac = load_activity_count_data(args.td, args.ucp)
    structure = validate_dataset_structure(ac)

    wide = make_wide_timeseries(ac)
    session_df = extract_session_features(wide, max_lag=args.max_lag)
    subject_df = make_subject_level_dataset(session_df)

    save_dataframe(wide, datasets_dir / "ac_wide_timeseries.csv")
    save_dataframe(session_df, datasets_dir / "ac_session_features.csv")
    save_dataframe(subject_df, datasets_dir / "ac_subject_features.csv")

    models = build_models(
        random_state=args.random_state,
        knn_k=args.knn_k,
        rf_n_estimators=args.rf_n_estimators,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
    )

    results_all = []
    preds_all = []
    datasets_to_eval = {
        "session_level_all": session_df,
        "session_level_dom": session_df.loc[session_df["session"] == "dom"].reset_index(drop=True),
        "session_level_ndom": session_df.loc[session_df["session"] == "ndom"].reset_index(drop=True),
        "subject_level": subject_df,
    }

    for dataset_name, dataset_df in datasets_to_eval.items():
        metrics_df, preds_df = evaluate_loso(dataset_df, dataset_name, models=models)
        results_all.append(metrics_df)
        preds_all.append(preds_df)

    results_df = pd.concat(results_all, ignore_index=True).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    preds_df = pd.concat(preds_all, ignore_index=True)

    best_models_df = (
        results_df.sort_values(
            ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    save_dataframe(results_df, metrics_dir / "results_all.csv")
    save_dataframe(best_models_df, metrics_dir / "best_models_top5.csv")
    save_dataframe(preds_df, predictions_dir / "predictions_all.csv")

    summary = {
        "validation": "LOSO / Leave-One-Subject-Out",
        "added_model": "xgboost",
        "structure": structure,
        "paths": {
            "wide_timeseries": str((datasets_dir / "ac_wide_timeseries.csv").resolve()),
            "session_features": str((datasets_dir / "ac_session_features.csv").resolve()),
            "subject_features": str((datasets_dir / "ac_subject_features.csv").resolve()),
            "results_all": str((metrics_dir / "results_all.csv").resolve()),
            "best_models_top5": str((metrics_dir / "best_models_top5.csv").resolve()),
            "predictions_all": str((predictions_dir / "predictions_all.csv").resolve()),
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset structure ===")
    for key, value in structure.items():
        print(f"{key}: {value}")

    print("\n=== Top results per dataset ===")
    print(best_models_df.to_string(index=False))
    print(f"\nOutput salvato in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
