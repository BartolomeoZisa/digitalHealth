#!/usr/bin/env python3
"""
Mirror Movements classification - Nested CV version
---------------------------------------------------
Pipeline:
- Outer test: GroupKFold subject-wise
- Inner validation: GroupKFold subject-wise
- Data source: 1-second AC dataset
- Representation:
    * compute VM = sqrt(Axis1^2 + Axis2^2 + Axis3^2)
    * define active hand and mirror hand based on session
    * extract session-level bilateral/statistical features
- Models:
    1) KMedoids
       - preprocessing: Euclidean, DTW
       - no preprocessing: Euclidean, DTW
    2) BOSSEnsemble
       - preprocessing
       - no preprocessing with univariate concatenation

Expected files:
- dataset/csv/bbt_td_1sec_anon.csv
- dataset/csv/bbt_ucp_1sec_anon.csv

Outputs:
- results_nested/results_summary.csv
- results_nested/results_summary.json
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
EPS = 1e-8
MAX_LAG = 3

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4

RESULTS_DIR = Path("results_nested")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

LABEL_MAP = {
    "td": 0,
    "ucp": 1,
}

KMEDOIDS_GRID = {
    "medoids_per_class": [1, 2, 3],
}

SIGNALS = ["Axis1", "Axis2", "Axis3", "VM", "Total"]


# ============================================================
# DISTANCES
# ============================================================

def dtw_distance_1d(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    n = len(a)
    m = len(b)

    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = (ai - b[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(np.sqrt(dp[n, m]))


def pairwise_multivariate_dtw(
    X_a: np.ndarray,
    X_b: np.ndarray | None = None,
    window: int | None = None,
) -> np.ndarray:
    if X_b is None:
        X_b = X_a
        symmetric = True
    else:
        symmetric = False

    n_a = len(X_a)
    n_b = len(X_b)
    D = np.zeros((n_a, n_b), dtype=float)

    for i in range(n_a):
        j_start = i if symmetric else 0
        for j in range(j_start, n_b):
            d = 0.0
            for c in range(X_a.shape[1]):
                d += dtw_distance_1d(X_a[i, c], X_b[j, c], window=window)
            D[i, j] = d
            if symmetric:
                D[j, i] = d
    return D


def pairwise_distance(
    X_a: np.ndarray,
    X_b: np.ndarray | None = None,
    metric: str = "euclidean",
    dtw_window: int | None = None,
) -> np.ndarray:
    if metric == "euclidean":
        A = X_a.reshape(X_a.shape[0], -1)
        if X_b is None:
            return cdist(A, A, metric="euclidean")
        B = X_b.reshape(X_b.shape[0], -1)
        return cdist(A, B, metric="euclidean")

    if metric == "dtw":
        return pairwise_multivariate_dtw(X_a, X_b, window=dtw_window)

    raise ValueError(f"Unsupported metric: {metric}")


# ============================================================
# DATA LOADING
# ============================================================

def load_single_csv(path: str | Path, expected_type: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = [
        "Axis1", "Axis2", "Axis3", "datetime",
        "type", "session", "hand_label", "id"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")

    for c in ["Axis1", "Axis2", "Axis3"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if df[["Axis1", "Axis2", "Axis3"]].isna().any().any():
        raise ValueError(f"{path} contains NaN in axis columns")

    unique_types = df["type"].astype(str).str.lower().unique().tolist()
    if len(unique_types) != 1:
        raise ValueError(f"{path} must contain a single type, found: {unique_types}")
    if unique_types[0] != expected_type:
        raise ValueError(f"{path} expected type '{expected_type}', found '{unique_types[0]}'")

    return df


def load_activity_count_data(td_file: str | Path, ucp_file: str | Path) -> pd.DataFrame:
    td = load_single_csv(td_file, expected_type="td")
    ucp = load_single_csv(ucp_file, expected_type="ucp")

    data = pd.concat([td, ucp], ignore_index=True).copy()

    data["VM"] = np.sqrt(
        data["Axis1"] ** 2 +
        data["Axis2"] ** 2 +
        data["Axis3"] ** 2
    )
    data["Total"] = data["Axis1"] + data["Axis2"] + data["Axis3"]

    return data


# ============================================================
# WIDE FORMAT + ACTIVE/MIRROR
# ============================================================

def _detect_dom_ndom_columns(wide: pd.DataFrame) -> Tuple[str, str]:
    suffixes = list(wide.columns)
    has_dom = any(col.endswith("_dom") for col in suffixes)
    has_ndom = any(col.endswith("_ndom") for col in suffixes)
    if has_dom and has_ndom:
        return "dom", "ndom"
    raise ValueError(
        "Could not find dominant/non-dominant columns after pivot. "
        "Expected hand_label values to produce *_dom and *_ndom columns."
    )


def make_wide_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    pivot = data.pivot_table(
        index=["id", "type", "session", "datetime"],
        columns="hand_label",
        values=SIGNALS,
        aggfunc="first"
    )

    pivot.columns = [f"{signal}_{hand}" for signal, hand in pivot.columns]
    wide = pivot.reset_index().sort_values(["id", "session", "datetime"]).copy()

    dom_suffix, ndom_suffix = _detect_dom_ndom_columns(wide)

    for signal in SIGNALS:
        dom_col = f"{signal}_{dom_suffix}"
        ndom_col = f"{signal}_{ndom_suffix}"

        active_col = f"{signal}_active"
        mirror_col = f"{signal}_mirror"

        wide[active_col] = np.where(
            wide["session"].astype(str).str.lower() == "dom",
            wide[dom_col],
            wide[ndom_col],
        )
        wide[mirror_col] = np.where(
            wide["session"].astype(str).str.lower() == "dom",
            wide[ndom_col],
            wide[dom_col],
        )

    return wide


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def summarize_signal(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
        f"{prefix}_sum": float(np.sum(x)),
        f"{prefix}_nonzero_frac": float(np.mean(x > 0)),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def max_abs_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 3) -> Tuple[float, int]:
    best_abs_corr = -1.0
    best_signed_corr = 0.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_lag = x[:lag]
            y_lag = y[-lag:]
        elif lag > 0:
            x_lag = x[lag:]
            y_lag = y[:-lag]
        else:
            x_lag = x
            y_lag = y

        if len(x_lag) < 2 or len(y_lag) < 2:
            corr = 0.0
        else:
            corr = safe_corr(x_lag, y_lag)

        if abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_signed_corr = corr
            best_lag = lag

    return float(best_signed_corr), int(best_lag)


def extract_session_features(wide: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    rows = []

    for (subject_id, typ, session), g in wide.groupby(["id", "type", "session"], sort=True):
        g = g.sort_values("datetime")
        row = {
            "id": subject_id,
            "type": typ,
            "session": session,
            "n_seconds": int(len(g)),
        }

        for signal in SIGNALS:
            active = g[f"{signal}_active"].to_numpy(dtype=float)
            mirror = g[f"{signal}_mirror"].to_numpy(dtype=float)

            row.update(summarize_signal(active, f"{signal}_active"))
            row.update(summarize_signal(mirror, f"{signal}_mirror"))

            row[f"{signal}_bilateral_mean_diff"] = float(np.mean(active - mirror))
            row[f"{signal}_bilateral_abs_mean_diff"] = float(np.mean(np.abs(active - mirror)))
            row[f"{signal}_bilateral_sum_ratio"] = float(np.sum(mirror) / (np.sum(active) + EPS))
            row[f"{signal}_bilateral_asymmetry_index"] = float(
                (np.sum(active) - np.sum(mirror)) /
                (np.sum(active) + np.sum(mirror) + EPS)
            )
            row[f"{signal}_bilateral_corr0"] = safe_corr(active, mirror)

            maxcorr, lag = max_abs_xcorr(active, mirror, max_lag=max_lag)
            row[f"{signal}_bilateral_maxcorr_abs"] = float(abs(maxcorr))
            row[f"{signal}_bilateral_lag_at_maxcorr"] = int(lag)

        rows.append(row)

    return pd.DataFrame(rows)


def feature_family_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    meta_cols = {"id", "type", "session", "n_seconds"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    active_only = [c for c in feature_cols if "_active_" in c]
    active_mirror = [c for c in feature_cols if ("_active_" in c or "_mirror_" in c)]
    bilateral_only = [c for c in feature_cols if "_bilateral_" in c]
    all_features = feature_cols[:]

    return {
        "active_only": active_only,
        "active_mirror": active_mirror,
        "bilateral_only": bilateral_only,
        "all_features": all_features,
    }


def session_df_to_tensor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)
    X = X[:, np.newaxis, :]
    y = df["type"].astype(str).str.lower().map(LABEL_MAP).to_numpy(dtype=int)
    groups = df["id"].astype(str).to_numpy()
    return X, y, groups


# ============================================================
# PREPROCESSING
# ============================================================

def z_normalize_per_channel_per_sample(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=True)
    mu = X.mean(axis=2, keepdims=True)
    sigma = X.std(axis=2, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma


def to_univariate_concatenation(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], 1, X.shape[1] * X.shape[2])


# ============================================================
# SUPERVISED K-MEDOIDS
# ============================================================

class PerClassKMedoidsClassifier:
    def __init__(
        self,
        medoids_per_class: int = 1,
        metric: str = "euclidean",
        dtw_window: int | None = None,
        random_state: int = 42,
        max_iter: int = 50,
    ) -> None:
        self.medoids_per_class = medoids_per_class
        self.metric = metric
        self.dtw_window = dtw_window
        self.random_state = random_state
        self.max_iter = max_iter
        self.medoid_series_ = None
        self.medoid_labels_ = None

    def _fit_class_medoids(self, X_class: np.ndarray) -> np.ndarray:
        D = pairwise_distance(X_class, metric=self.metric, dtw_window=self.dtw_window)
        n = len(X_class)
        k = min(self.medoids_per_class, n)

        rng = np.random.default_rng(self.random_state)
        medoids = rng.choice(n, size=k, replace=False)

        for _ in range(self.max_iter):
            assignments = np.argmin(D[:, medoids], axis=1)
            new_medoids = medoids.copy()

            for cluster_id in range(k):
                idx = np.where(assignments == cluster_id)[0]
                if len(idx) == 0:
                    continue
                intra = D[np.ix_(idx, idx)]
                costs = intra.sum(axis=1)
                new_medoids[cluster_id] = idx[np.argmin(costs)]

            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        return medoids

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerClassKMedoidsClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        all_medoids = []
        all_labels = []

        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            medoids_local = self._fit_class_medoids(X[idx])
            all_medoids.append(X[idx][medoids_local])
            all_labels.extend([cls] * len(medoids_local))

        self.medoid_series_ = np.concatenate(all_medoids, axis=0)
        self.medoid_labels_ = np.asarray(all_labels, dtype=int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.medoid_series_ is None or self.medoid_labels_ is None:
            raise RuntimeError("Model not fitted yet.")

        D = pairwise_distance(
            np.asarray(X),
            self.medoid_series_,
            metric=self.metric,
            dtw_window=self.dtw_window,
        )
        nearest = np.argmin(D, axis=1)
        return self.medoid_labels_[nearest]


# ============================================================
# BOSS WRAPPER
# ============================================================

def build_boss_classifier():
    try:
        from aeon.classification.dictionary_based import BOSSEnsemble
    except Exception as e:
        raise ImportError(
            "BOSSEnsemble requires aeon. Install it with: pip install aeon"
        ) from e
    return BOSSEnsemble(random_state=RANDOM_STATE)


# ============================================================
# EVALUATION
# ============================================================

def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }


def summarize_across_folds(folds: List[Dict[str, float]]) -> Dict[str, float]:
    keys = folds[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in folds], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def global_metrics(y_true_all: np.ndarray, y_pred_all: np.ndarray) -> Dict[str, float]:
    return {
        "test_accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "test_f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
        "test_recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "test_precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
    }


def run_nested_cv_kmedoids(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metric: str,
    medoids_grid: List[int],
    outer_splits: int = 5,
    inner_splits: int = 4,
    dtw_window: int | None = None,
) -> Dict[str, object]:
    outer_cv = GroupKFold(n_splits=outer_splits)

    all_test_true = []
    all_test_pred = []
    outer_fold_metrics = []
    validation_f1_per_outer_fold = []
    selected_k_values = []

    outer_fold_total = outer_splits

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        print(f"[KMedoids-{metric}] Outer fold {fold_idx}/{outer_fold_total}")

        X_train_all = X[outer_train_idx]
        y_train_all = y[outer_train_idx]
        groups_train_all = groups[outer_train_idx]

        X_test = X[outer_test_idx]
        y_test = y[outer_test_idx]

        inner_cv = GroupKFold(n_splits=inner_splits)
        grid_scores = {k: [] for k in medoids_grid}

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_all, y_train_all, groups_train_all):
            X_inner_train = X_train_all[inner_train_idx]
            y_inner_train = y_train_all[inner_train_idx]
            X_inner_val = X_train_all[inner_val_idx]
            y_inner_val = y_train_all[inner_val_idx]

            for k in medoids_grid:
                clf = PerClassKMedoidsClassifier(
                    medoids_per_class=k,
                    metric=metric,
                    dtw_window=dtw_window,
                    random_state=RANDOM_STATE,
                )
                clf.fit(X_inner_train, y_inner_train)
                y_val_pred = clf.predict(X_inner_val)
                val_f1 = f1_score(y_inner_val, y_val_pred, zero_division=0)
                grid_scores[k].append(val_f1)

        best_k = max(medoids_grid, key=lambda k: (np.mean(grid_scores[k]), -k))
        selected_k_values.append(best_k)
        validation_f1_per_outer_fold.append(float(np.mean(grid_scores[best_k])))

        final_clf = PerClassKMedoidsClassifier(
            medoids_per_class=best_k,
            metric=metric,
            dtw_window=dtw_window,
            random_state=RANDOM_STATE,
        )
        final_clf.fit(X_train_all, y_train_all)
        y_test_pred = final_clf.predict(X_test)

        all_test_true.extend(y_test.tolist())
        all_test_pred.extend(y_test_pred.tolist())
        outer_fold_metrics.append(fold_metrics(y_test, y_test_pred))

    y_true_all = np.asarray(all_test_true)
    y_pred_all = np.asarray(all_test_pred)

    return {
        **global_metrics(y_true_all, y_pred_all),
        **summarize_across_folds(outer_fold_metrics),
        "validation_f1": float(np.mean(validation_f1_per_outer_fold)),
        "validation_f1_std": float(np.std(validation_f1_per_outer_fold, ddof=1))
        if len(validation_f1_per_outer_fold) > 1 else 0.0,
        "selected_medoids_per_class": dict(Counter(selected_k_values)),
        "n_samples": int(len(X)),
        "n_groups": int(len(np.unique(groups))),
        "n_features_per_sample": int(X.shape[1] * X.shape[2]),
        "outer_cv_splits": int(outer_splits),
        "inner_cv_splits": int(inner_splits),
    }


def run_nested_cv_boss(
    X_uni: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 4,
) -> Dict[str, object]:
    outer_cv = GroupKFold(n_splits=outer_splits)

    all_test_true = []
    all_test_pred = []
    outer_fold_metrics = []
    validation_f1_per_outer_fold = []

    outer_fold_total = outer_splits

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X_uni, y, groups), start=1
    ):
        print(f"[BOSSEnsemble] Outer fold {fold_idx}/{outer_fold_total}")

        X_train_all = X_uni[outer_train_idx]
        y_train_all = y[outer_train_idx]
        groups_train_all = groups[outer_train_idx]

        X_test = X_uni[outer_test_idx]
        y_test = y[outer_test_idx]

        inner_cv = GroupKFold(n_splits=inner_splits)
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_all, y_train_all, groups_train_all):
            X_inner_train = X_train_all[inner_train_idx]
            y_inner_train = y_train_all[inner_train_idx]
            X_inner_val = X_train_all[inner_val_idx]
            y_inner_val = y_train_all[inner_val_idx]

            clf = build_boss_classifier()
            clf.fit(X_inner_train, y_inner_train)
            y_val_pred = clf.predict(X_inner_val)
            inner_scores.append(float(f1_score(y_inner_val, y_val_pred, zero_division=0)))

        validation_f1_per_outer_fold.append(float(np.mean(inner_scores)))

        final_clf = build_boss_classifier()
        final_clf.fit(X_train_all, y_train_all)
        y_test_pred = final_clf.predict(X_test)

        all_test_true.extend(y_test.tolist())
        all_test_pred.extend(y_test_pred.tolist())
        outer_fold_metrics.append(fold_metrics(y_test, y_test_pred))

    y_true_all = np.asarray(all_test_true)
    y_pred_all = np.asarray(all_test_pred)

    return {
        **global_metrics(y_true_all, y_pred_all),
        **summarize_across_folds(outer_fold_metrics),
        "validation_f1": float(np.mean(validation_f1_per_outer_fold)),
        "validation_f1_std": float(np.std(validation_f1_per_outer_fold, ddof=1))
        if len(validation_f1_per_outer_fold) > 1 else 0.0,
        "n_samples": int(len(X_uni)),
        "n_groups": int(len(np.unique(groups))),
        "n_features_per_sample": int(X_uni.shape[1] * X_uni.shape[2]),
        "outer_cv_splits": int(outer_splits),
        "inner_cv_splits": int(inner_splits),
    }


# ============================================================
# OUTPUT FORMAT
# ============================================================

def format_result_block(name: str, result: Dict[str, object]) -> str:
    lines = [
        "=" * 60,
        name,
        "=" * 60,
        f"Test Accuracy: {result['test_accuracy']:.6f}",
        f"Test F1: {result['test_f1']:.6f}",
        f"Test Recall: {result['test_recall']:.6f}",
        f"Test Precision: {result['test_precision']:.6f}",
        f"Validation F1: {result['validation_f1']:.6f}",
        "",
        f"Test Accuracy std (outer folds): {result['accuracy_std']:.6f}",
        f"Test F1 std (outer folds): {result['f1_std']:.6f}",
        f"Test Recall std (outer folds): {result['recall_std']:.6f}",
        f"Test Precision std (outer folds): {result['precision_std']:.6f}",
        f"Validation F1 std: {result['validation_f1_std']:.6f}",
    ]

    if "selected_medoids_per_class" in result:
        lines.append(f"Selected medoids_per_class counts: {result['selected_medoids_per_class']}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    td_path = "dataset/csv/bbt_td_1sec_anon.csv"
    ucp_path = "dataset/csv/bbt_ucp_1sec_anon.csv"

    print("Loading data...")
    data = load_activity_count_data(td_path, ucp_path)

    print("Making wide timeseries...")
    wide = make_wide_timeseries(data)

    print("Extracting session-level features...")
    session_df = extract_session_features(wide, max_lag=MAX_LAG)

    families = feature_family_columns(session_df)
    feature_cols = families["bilateral_only"]

    X_raw, y, groups = session_df_to_tensor(session_df, feature_cols)
    X_pre = z_normalize_per_channel_per_sample(X_raw)
    X_boss_pre = to_univariate_concatenation(X_pre)
    X_boss_raw = to_univariate_concatenation(X_raw)

    print("Dataset summary")
    print(f"Session-level samples: {len(session_df)}")
    print(f"Feature family used: bilateral_only")
    print(f"Tensor shape: {X_raw.shape}")
    print(f"Labels distribution: {Counter(y)}")
    print(f"Subjects: {len(np.unique(groups))}")
    print(f"Outer CV folds: {OUTER_N_SPLITS}")
    print(f"Inner CV folds: {INNER_N_SPLITS}")
    print()

    results = {}

    print("Running KMedoids_AC_Preprocessing_Euclidean...")
    results["KMedoids_AC_Preprocessing_Euclidean"] = run_nested_cv_kmedoids(
        X=X_pre,
        y=y,
        groups=groups,
        metric="euclidean",
        medoids_grid=KMEDOIDS_GRID["medoids_per_class"],
        outer_splits=OUTER_N_SPLITS,
        inner_splits=INNER_N_SPLITS,
    )

    print("Running KMedoids_AC_Preprocessing_DTW...")
    results["KMedoids_AC_Preprocessing_DTW"] = run_nested_cv_kmedoids(
        X=X_pre,
        y=y,
        groups=groups,
        metric="dtw",
        medoids_grid=KMEDOIDS_GRID["medoids_per_class"],
        outer_splits=OUTER_N_SPLITS,
        inner_splits=INNER_N_SPLITS,
        dtw_window=None,
    )

    print("Running KMedoids_AC_NoPreprocessing_Euclidean...")
    results["KMedoids_AC_NoPreprocessing_Euclidean"] = run_nested_cv_kmedoids(
        X=X_raw,
        y=y,
        groups=groups,
        metric="euclidean",
        medoids_grid=KMEDOIDS_GRID["medoids_per_class"],
        outer_splits=OUTER_N_SPLITS,
        inner_splits=INNER_N_SPLITS,
    )

    print("Running KMedoids_AC_NoPreprocessing_DTW...")
    results["KMedoids_AC_NoPreprocessing_DTW"] = run_nested_cv_kmedoids(
        X=X_raw,
        y=y,
        groups=groups,
        metric="dtw",
        medoids_grid=KMEDOIDS_GRID["medoids_per_class"],
        outer_splits=OUTER_N_SPLITS,
        inner_splits=INNER_N_SPLITS,
        dtw_window=None,
    )

    print("Running BOSSEnsemble_AC_Preprocessing / NoPreprocessing...")
    boss_available = True
    try:
        _ = build_boss_classifier()
    except Exception as e:
        boss_available = False
        print("\n[WARNING] BOSSEnsemble skipped.")
        print(str(e))
        print("Install aeon with: pip install aeon\n")

    if boss_available:
        results["BOSSEnsemble_AC_Preprocessing"] = run_nested_cv_boss(
            X_uni=X_boss_pre,
            y=y,
            groups=groups,
            outer_splits=OUTER_N_SPLITS,
            inner_splits=INNER_N_SPLITS,
        )
        results["BOSSEnsemble_AC_NoPreprocessing"] = run_nested_cv_boss(
            X_uni=X_boss_raw,
            y=y,
            groups=groups,
            outer_splits=OUTER_N_SPLITS,
            inner_splits=INNER_N_SPLITS,
        )

    rows = []
    for name, res in results.items():
        row = {"experiment": name}
        row.update(res)
        rows.append(row)

    df_results = pd.DataFrame(rows).sort_values("experiment")
    df_results.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    with open(RESULTS_DIR / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "#" * 80)
    print("FINAL RESULTS")
    print("#" * 80)
    for name in sorted(results):
        print(format_result_block(name, results[name]))
        print()

    print(f"Saved CSV  -> {RESULTS_DIR / 'results_summary.csv'}")
    print(f"Saved JSON -> {RESULTS_DIR / 'results_summary.json'}")


if __name__ == "__main__":
    main()
