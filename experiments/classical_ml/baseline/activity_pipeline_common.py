
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


TD_CSV_DEFAULT = "bbt_td_1sec_anon.csv"
UCP_CSV_DEFAULT = "bbt_ucp_1sec_anon.csv"


@dataclass
class Dataset:
    X_raw: np.ndarray         # shape: (n_samples, T, 6)
    y: np.ndarray             # 0=TD, 1=UCP
    groups: np.ndarray        # subject IDs for LOSO
    sample_ids: np.ndarray    # subject_session
    subject_labels: Dict[str, int]


def parse_args_base(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--td_csv", default=TD_CSV_DEFAULT)
    p.add_argument("--ucp_csv", default=UCP_CSV_DEFAULT)
    p.add_argument("--outdir", required=True)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_folds", type=int, default=None, help="Run only the first N LOSO folds.")
    return p


def load_activity_count_dataset(td_csv: str, ucp_csv: str) -> Dataset:
    td = pd.read_csv(td_csv)
    ucp = pd.read_csv(ucp_csv)
    df = pd.concat([td, ucp], ignore_index=True)

    required = {"Axis1", "Axis2", "Axis3", "datetime", "type", "session", "hand_label", "id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    samples_X = []
    samples_y = []
    groups = []
    sample_ids = []

    subject_labels = {}
    for subject_id, g_subj in df.groupby("id"):
        subj_type = g_subj["type"].iloc[0].lower()
        if subj_type not in {"td", "ucp"}:
            raise ValueError(f"Unexpected type for {subject_id}: {subj_type}")
        y = 0 if subj_type == "td" else 1
        subject_labels[subject_id] = y

        for session, g_sess in g_subj.groupby("session"):
            # pivot to align the two hands over time
            g = g_sess.copy()
            g["dt"] = pd.to_datetime(g["datetime"])
            g = g.sort_values(["dt", "hand_label"])
            wide = (
                g.pivot_table(
                    index="dt",
                    columns="hand_label",
                    values=["Axis1", "Axis2", "Axis3"],
                    aggfunc="first",
                )
                .sort_index()
            )

            expected_cols = [
                ("Axis1", "dom"), ("Axis2", "dom"), ("Axis3", "dom"),
                ("Axis1", "ndom"), ("Axis2", "ndom"), ("Axis3", "ndom"),
            ]
            for col in expected_cols:
                if col not in wide.columns:
                    raise ValueError(f"Missing hand/time data for sample {subject_id}_{session}: {col}")

            arr = np.column_stack([wide[col].to_numpy(dtype=float) for col in expected_cols])

            if arr.shape[0] != 61:
                raise ValueError(f"Expected 61 timesteps, got {arr.shape[0]} for {subject_id}_{session}")

            samples_X.append(arr)
            samples_y.append(y)
            groups.append(subject_id)
            sample_ids.append(f"{subject_id}_{session}")

    X_raw = np.stack(samples_X, axis=0)
    y = np.asarray(samples_y, dtype=int)
    groups = np.asarray(groups)
    sample_ids = np.asarray(sample_ids)

    return Dataset(X_raw=X_raw, y=y, groups=groups, sample_ids=sample_ids, subject_labels=subject_labels)


def make_representation(X_raw: np.ndarray, name: str) -> np.ndarray:
    name = name.lower()
    if name == "no_preprocessing":
        return X_raw.astype(float)

    dom = X_raw[:, :, 0:3]
    ndom = X_raw[:, :, 3:6]
    norm_dom = np.linalg.norm(dom, axis=2)
    norm_ndom = np.linalg.norm(ndom, axis=2)
    bilateral_norm = np.sqrt((dom ** 2).sum(axis=2) + (ndom ** 2).sum(axis=2))

    if name == "norm_diff":
        diff = np.abs(norm_dom - norm_ndom)
        return np.stack([bilateral_norm, diff], axis=2)

    if name == "norm_p2p":
        p2p = np.abs(np.diff(bilateral_norm, axis=1, prepend=bilateral_norm[:, :1]))
        return np.stack([bilateral_norm, p2p], axis=2)

    raise ValueError(f"Unknown representation: {name}")


def z_normalize_per_sample(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    return (X - mu) / np.maximum(sigma, eps)


def flatten_time_series(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def subject_level_val_split(
    train_subjects: Sequence[str],
    subject_labels: Dict[str, int],
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    train_subjects = np.asarray(sorted(train_subjects))
    y_subj = np.asarray([subject_labels[s] for s in train_subjects], dtype=int)

    cls_counts = np.bincount(y_subj, minlength=2)
    if np.any(cls_counts < 2):
        # Fallback: take one subject per class for validation when possible.
        val_subjects = []
        tr_subjects = []
        for cls in [0, 1]:
            cls_subj = train_subjects[y_subj == cls]
            if len(cls_subj) > 0:
                val_subjects.append(cls_subj[0])
                tr_subjects.extend(cls_subj[1:])
        return np.asarray(sorted(tr_subjects)), np.asarray(sorted(val_subjects))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(train_subjects, y_subj))
    return train_subjects[train_idx], train_subjects[val_idx]


def loso_subject_folds(groups: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    unique_subjects = np.asarray(sorted(np.unique(groups)))
    folds = []
    for subject in unique_subjects:
        test_mask = groups == subject
        train_mask = ~test_mask
        folds.append((subject, np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_pred_0": int(np.sum(y_pred == 0)),
        "n_pred_1": int(np.sum(y_pred == 1)),
        "n_true_0": int(np.sum(y_true == 0)),
        "n_true_1": int(np.sum(y_true == 1)),
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
        "test_f1", "test_recall", "test_precision", "validation_f1"
    ]
    rows = []
    for (experiment, representation), g in df.groupby(["experiment", "representation"]):
        row = {
            "experiment": experiment,
            "representation": representation,
            "n_folds": int(len(g)),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["experiment", "representation"]).reset_index(drop=True)


def save_results(fold_df: pd.DataFrame, outdir: str) -> Tuple[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fold_path = out / "fold_results.csv"
    summary_path = out / "summary_results.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df = summarize_results(fold_df)
    summary_df.to_csv(summary_path, index=False)
    return str(fold_path), str(summary_path)


def dtw_distance_multivariate(
    a: np.ndarray,
    b: np.ndarray,
    radius: int | None = None,
) -> float:
    """
    Multivariate DTW with Euclidean local cost.
    a, b: (T, C)
    """
    n, m = len(a), len(b)
    if radius is None:
        radius = max(n, m)
    radius = max(radius, abs(n - m))

    inf = float("inf")
    dp = np.full((n + 1, m + 1), inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def local_shape_descriptors(X: np.ndarray, reach: int = 2) -> np.ndarray:
    """
    Build simple shape descriptors for shapeDTW:
    each time step becomes the flattened local window around it.
    X: (n_samples, T, C) -> (n_samples, T, (2*reach+1)*C)
    """
    n, T, C = X.shape
    padded = np.pad(X, ((0, 0), (reach, reach), (0, 0)), mode="edge")
    desc = np.empty((n, T, (2 * reach + 1) * C), dtype=float)
    for t in range(T):
        desc[:, t, :] = padded[:, t : t + 2 * reach + 1, :].reshape(n, -1)
    return desc


def knn_predict_from_distance_matrix(
    D_query_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    k = min(k, len(y_train))
    idx = np.argsort(D_query_train, axis=1)[:, :k]
    neigh = y_train[idx]
    # binary majority vote, tie -> 1 if mean >= 0.5 else 0
    pred = (neigh.mean(axis=1) >= 0.5).astype(int)
    return pred


def compute_distance_matrix(
    X_a: np.ndarray,
    X_b: np.ndarray,
    metric: str,
    radius: int | None = None,
    shape_reach: int = 2,
) -> np.ndarray:
    metric = metric.lower()
    if metric == "euclidean":
        A = flatten_time_series(X_a)
        B = flatten_time_series(X_b)
        AA = (A ** 2).sum(axis=1, keepdims=True)
        BB = (B ** 2).sum(axis=1, keepdims=True).T
        D2 = np.maximum(AA + BB - 2 * A @ B.T, 0.0)
        return np.sqrt(D2)

    if metric == "dtw":
        D = np.empty((len(X_a), len(X_b)), dtype=float)
        for i in range(len(X_a)):
            for j in range(len(X_b)):
                D[i, j] = dtw_distance_multivariate(X_a[i], X_b[j], radius=radius)
        return D

    if metric == "shapedtw":
        A = local_shape_descriptors(X_a, reach=shape_reach)
        B = local_shape_descriptors(X_b, reach=shape_reach)
        D = np.empty((len(A), len(B)), dtype=float)
        for i in range(len(A)):
            for j in range(len(B)):
                D[i, j] = dtw_distance_multivariate(A[i], B[j], radius=radius)
        return D

    raise ValueError(f"Unsupported metric: {metric}")


class DTWMedoidClusterer:
    """
    KMeans-like clustering under DTW using medoid updates.
    This is intentionally runnable without tslearn.
    """
    def __init__(self, n_clusters: int = 2, max_iter: int = 10, radius: int | None = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.radius = radius
        self.random_state = random_state
        self.medoid_indices_: np.ndarray | None = None
        self.medoids_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        medoid_idx = rng.choice(n, size=self.n_clusters, replace=False)
        labels = np.zeros(n, dtype=int)

        for _ in range(self.max_iter):
            medoids = X[medoid_idx]
            D = compute_distance_matrix(X, medoids, metric="dtw", radius=self.radius)
            new_labels = D.argmin(axis=1)

            new_medoid_idx = medoid_idx.copy()
            for k in range(self.n_clusters):
                members = np.where(new_labels == k)[0]
                if len(members) == 0:
                    remaining = np.setdiff1d(np.arange(n), new_medoid_idx, assume_unique=False)
                    if len(remaining) == 0:
                        remaining = np.arange(n)
                    new_medoid_idx[k] = rng.choice(remaining)
                    continue

                D_within = compute_distance_matrix(X[members], X[members], metric="dtw", radius=self.radius)
                best_local = members[np.argmin(D_within.sum(axis=1))]
                new_medoid_idx[k] = best_local

            if np.array_equal(new_labels, labels) and np.array_equal(new_medoid_idx, medoid_idx):
                labels = new_labels
                medoid_idx = new_medoid_idx
                break

            labels = new_labels
            medoid_idx = new_medoid_idx

        self.medoid_indices_ = medoid_idx
        self.medoids_ = X[medoid_idx]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.medoids_ is None:
            raise RuntimeError("Call fit before predict")
        D = compute_distance_matrix(X, self.medoids_, metric="dtw", radius=self.radius)
        return D.argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)


def cluster_majority_label_map(cluster_ids: np.ndarray, y_true: np.ndarray, n_clusters: int) -> Dict[int, int]:
    out = {}
    for k in range(n_clusters):
        members = y_true[cluster_ids == k]
        if len(members) == 0:
            out[k] = 0
        else:
            out[k] = int(np.mean(members) >= 0.5)
    return out


def apply_cluster_label_map(cluster_ids: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(c)] for c in cluster_ids], dtype=int)


def print_summary_table(summary_df: pd.DataFrame):
    if summary_df.empty:
        print("No results.")
        return
    cols = [
        "experiment", "representation",
        "test_accuracy_mean", "test_accuracy_std",
        "test_balanced_accuracy_mean", "test_balanced_accuracy_std",
        "test_macro_f1_mean", "test_macro_f1_std",
        "test_f1_mean", "test_f1_std",
        "test_recall_mean", "test_recall_std",
        "test_precision_mean", "test_precision_std",
        "validation_f1_mean", "validation_f1_std",
    ]
    printable = summary_df[cols].copy()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(printable.round(4).to_string(index=False))
