#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

try:
    from pipeline_definitive_LOSO import (
        build_models,
        extract_session_features,
        feature_families,
        load_activity_count_data,
        make_subject_level_dataset,
        make_wide_timeseries,
        save_dataframe,
        validate_dataset_structure,
    )
except ImportError as exc:
    raise ImportError(
        "Questo script deve trovarsi nella stessa cartella di "
        "'mirror_movements_ac_pipeline_xgboost.py'."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Movements nested CV pipeline for Activity Count data.")
    parser.add_argument("--td", required=True, help="Path al CSV TD activity count (1 sec).")
    parser.add_argument("--ucp", required=True, help="Path al CSV UCP activity count (1 sec).")
    parser.add_argument("--outdir", default="results_ac_nested", help="Cartella di output.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed randomica.")
    parser.add_argument("--rf-n-estimators", type=int, default=200, help="Numero base di alberi Random Forest.")
    parser.add_argument("--max-lag", type=int, default=3, help="Lag massimo per cross-correlation.")
    parser.add_argument("--knn-k", type=int, default=5, help="Numero base di vicini per KNN.")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Numero base di alberi boosting XGBoost.")
    parser.add_argument("--xgb-max-depth", type=int, default=2, help="Profondità base alberi XGBoost.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="Learning rate base XGBoost.")
    parser.add_argument("--inner-splits", type=int, default=5, help="Numero massimo di fold GroupKFold nell'inner loop.")
    parser.add_argument(
        "--scoring",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1_macro", "accuracy"],
        help="Metrica usata per scegliere gli iperparametri nell'inner loop.",
    )
    parser.add_argument(
        "--grid-n-jobs",
        type=int,
        default=1,
        help=(
            "Parallelismo per GridSearchCV. Tienilo a 1 se usi modelli con n_jobs=-1 "
            "per evitare oversubscription."
        ),
    )
    return parser.parse_args()


def build_param_grids(rf_n_estimators: int, xgb_n_estimators: int) -> Dict[str, Dict[str, List]]:
    """Griglie volutamente compatte, adatte a dataset piccolo e nested CV.

    Con 45 soggetti, griglie troppo grandi possono produrre risultati instabili
    e tempi lunghi. Qui si ottimizzano pochi iperparametri essenziali per modello.
    """
    return {
        "knn": {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        },
        "logreg": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        },
        "svm_rbf": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "random_forest": {
            "clf__n_estimators": [max(50, rf_n_estimators // 2), rf_n_estimators],
            "clf__max_depth": [None, 3, 5],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "clf__n_estimators": [max(50, xgb_n_estimators // 2), xgb_n_estimators],
            "clf__max_depth": [1, 2, 3],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.8],
            "clf__colsample_bytree": [0.8],
            "clf__reg_lambda": [1.0, 5.0],
        },
    }


def make_inner_cv(groups_train: np.ndarray, max_splits: int) -> GroupKFold:
    n_groups_train = int(pd.Series(groups_train).nunique())
    n_splits = min(max_splits, n_groups_train)
    if n_splits < 2:
        raise ValueError("Servono almeno 2 gruppi nel training set per fare nested CV.")
    return GroupKFold(n_splits=n_splits)


def evaluate_nested_cv(
    df: pd.DataFrame,
    dataset_name: str,
    models: Dict,
    param_grids: Dict[str, Dict[str, List]],
    inner_splits: int = 5,
    scoring: str = "balanced_accuracy",
    grid_n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = feature_families(df)
    outer_cv = LeaveOneGroupOut()

    y = df["type"].astype(str).to_numpy()
    groups = df["id"].astype(str).to_numpy()

    metrics_rows: List[Dict[str, float]] = []
    predictions_rows: List[pd.DataFrame] = []
    selected_params_rows: List[Dict[str, object]] = []

    for family_name, cols in families.items():
        if not cols:
            continue

        X = df[cols]

        for model_name, model in models.items():
            y_pred = np.empty(len(df), dtype=object)

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
                groups_train = groups[train_idx]
                inner_cv = make_inner_cv(groups_train, inner_splits)

                search = GridSearchCV(
                    estimator=clone(model),
                    param_grid=param_grids[model_name],
                    scoring=scoring,
                    cv=inner_cv,
                    refit=True,
                    n_jobs=grid_n_jobs,
                    error_score=np.nan,
                )

                search.fit(X.iloc[train_idx], y[train_idx], groups=groups_train)
                y_pred[test_idx] = search.predict(X.iloc[test_idx])

                selected_params_rows.append({
                    "dataset": dataset_name,
                    "feature_family": family_name,
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "test_ids": ",".join(pd.Series(groups[test_idx]).astype(str).unique()),
                    "n_train_samples": int(len(train_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "n_train_groups": int(pd.Series(groups_train).nunique()),
                    "n_inner_splits": int(inner_cv.n_splits),
                    "scoring": scoring,
                    "best_inner_score": float(search.best_score_) if search.best_score_ is not None else np.nan,
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                })

            cm = confusion_matrix(y, y_pred, labels=["td", "ucp"])
            tn, fp, fn, tp = cm.ravel()

            metrics_rows.append({
                "dataset": dataset_name,
                "feature_family": family_name,
                "model": model_name,
                "validation": "nested_outer_LOSO_inner_GroupKFold",
                "inner_scoring": scoring,
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
            pred_df["validation"] = "nested_outer_LOSO_inner_GroupKFold"
            pred_df["pred_label"] = y_pred
            predictions_rows.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    preds_df = pd.concat(predictions_rows, ignore_index=True)
    selected_params_df = pd.DataFrame(selected_params_rows)
    return metrics_df, preds_df, selected_params_df


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    datasets_dir = outdir / "datasets"
    metrics_dir = outdir / "metrics"
    predictions_dir = outdir / "predictions"
    nested_dir = outdir / "nested"
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
    param_grids = build_param_grids(
        rf_n_estimators=args.rf_n_estimators,
        xgb_n_estimators=args.xgb_n_estimators,
    )

    datasets_to_eval = {
        "session_level_all": session_df,
        "session_level_dom": session_df.loc[session_df["session"] == "dom"].reset_index(drop=True),
        "session_level_ndom": session_df.loc[session_df["session"] == "ndom"].reset_index(drop=True),
        "subject_level": subject_df,
    }

    results_all = []
    preds_all = []
    selected_params_all = []

    for dataset_name, dataset_df in datasets_to_eval.items():
        metrics_df, preds_df, selected_params_df = evaluate_nested_cv(
            dataset_df,
            dataset_name,
            models=models,
            param_grids=param_grids,
            inner_splits=args.inner_splits,
            scoring=args.scoring,
            grid_n_jobs=args.grid_n_jobs,
        )
        results_all.append(metrics_df)
        preds_all.append(preds_df)
        selected_params_all.append(selected_params_df)

    results_df = pd.concat(results_all, ignore_index=True).sort_values(
        ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    preds_df = pd.concat(preds_all, ignore_index=True)
    selected_params_df = pd.concat(selected_params_all, ignore_index=True)

    best_models_df = (
        results_df.sort_values(
            ["dataset", "balanced_accuracy", "macro_f1", "accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    save_dataframe(results_df, metrics_dir / "results_all_nested.csv")
    save_dataframe(best_models_df, metrics_dir / "best_models_top5_nested.csv")
    save_dataframe(preds_df, predictions_dir / "predictions_all_nested.csv")
    save_dataframe(selected_params_df, nested_dir / "selected_params_by_outer_fold.csv")

    summary = {
        "validation": "Nested CV: outer Leave-One-Subject-Out, inner GroupKFold",
        "inner_splits_requested": args.inner_splits,
        "inner_scoring": args.scoring,
        "models": sorted(models.keys()),
        "structure": structure,
        "paths": {
            "wide_timeseries": str((datasets_dir / "ac_wide_timeseries.csv").resolve()),
            "session_features": str((datasets_dir / "ac_session_features.csv").resolve()),
            "subject_features": str((datasets_dir / "ac_subject_features.csv").resolve()),
            "results_all_nested": str((metrics_dir / "results_all_nested.csv").resolve()),
            "best_models_top5_nested": str((metrics_dir / "best_models_top5_nested.csv").resolve()),
            "predictions_all_nested": str((predictions_dir / "predictions_all_nested.csv").resolve()),
            "selected_params_by_outer_fold": str((nested_dir / "selected_params_by_outer_fold.csv").resolve()),
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Dataset structure ===")
    for key, value in structure.items():
        print(f"{key}: {value}")

    print("\n=== Top nested results per dataset ===")
    print(best_models_df.to_string(index=False))
    print(f"\nOutput salvato in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
