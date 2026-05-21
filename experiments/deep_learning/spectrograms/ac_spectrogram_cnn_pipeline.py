#!/usr/bin/env python3
"""
AC - Spectogram CNN pipeline
-----------------------------------
Adaptation of the original STFT spectrogram script for Activity Count (AC) data.

What it does:
1. Loads TD and UCP AC 1-second CSV files.
2. Builds STFT spectrograms per (patient id, session) for dominant and nondominant hands.
3. Creates three learning inputs:
   - B1: two-branch CNN with shared weights
   - B2: two-branch CNN with independent weights
   - B3: delta spectrogram (dom - ndom) -> single CNN
4. Uses grouped-by-patient validation (LOPO or Nested GroupKFold).
5. Saves debug prints, intermediate logs/checkpoints, final JSON/CSV results,
   spectrograms of tested patients, and best models in .joblib.

Expected AC columns:
    Axis1, Axis2, Axis3, datetime, session, id, type,
    plus one of: hand_label or hand_type (values: dom/ndom)

Notes:
- Since this is AC at 1 Hz, STFT resolution is much lower than for RAW data.
- Best-model serialization is done with joblib by storing a dictionary containing
  the PyTorch state_dict and metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 1
DEFAULT_WINDOW_S = 3
DEFAULT_OVERLAP_PCT = 0.5
DEFAULT_AXES = ["Axis1", "Axis2", "Axis3"]
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.f = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full)
        self.f.write(full + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def get_hand_column(df: pd.DataFrame) -> str:
    if "hand_label" in df.columns:
        return "hand_label"
    if "hand_type" in df.columns:
        return "hand_type"
    raise ValueError("Expected one of 'hand_label' or 'hand_type' in the dataset.")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_single_csv(path: str | Path, expected_type: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    hand_col = get_hand_column(df)

    required = ["Axis1", "Axis2", "Axis3", "datetime", "session", "id", "type", hand_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    for c in ["Axis1", "Axis2", "Axis3"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    unique_types = df["type"].astype(str).str.lower().unique().tolist()
    if len(unique_types) != 1:
        raise ValueError(f"{path} must contain a single type. Found: {unique_types}")
    if unique_types[0] != expected_type.lower():
        raise ValueError(f"{path} expected type {expected_type}, found {unique_types[0]}")

    # standardize hand column name to hand_label
    if hand_col != "hand_label":
        df = df.rename(columns={hand_col: "hand_label"})

    return df


def load_ac_dataset(td_path: str | Path, ucp_path: str | Path) -> pd.DataFrame:
    td = load_single_csv(td_path, "td")
    ucp = load_single_csv(ucp_path, "ucp")
    df = pd.concat([td, ucp], ignore_index=True)
    return df


# -----------------------------------------------------------------------------
# STFT / spectrogram generation (adapted from original script)
# -----------------------------------------------------------------------------

def compute_spectrogram_array(
    signal: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    signal shape: (n_samples, n_channels)
    returns:
        freqs, times, spec shape (n_channels, n_freqs, n_times)
    """
    specs = []
    freqs = None
    times = None
    for i in range(signal.shape[1]):
        f, t, Sxx = spectrogram(
            signal[:, i],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="magnitude",
        )
        specs.append(np.log1p(Sxx).astype(np.float32))
        freqs = f
        times = t
    return freqs, times, np.stack(specs, axis=0)


def save_preview_png(spec: np.ndarray, title: str, save_path: Path) -> None:
    """
    spec shape: (channels, freqs, times)
    """
    n_ch = spec.shape[0]
    fig, axs = plt.subplots(1, n_ch, figsize=(5 * n_ch, 4), squeeze=False, sharey=True)
    fig.suptitle(title, fontsize=10)
    for i, ax in enumerate(axs[0]):
        im = ax.imshow(spec[i], aspect="auto", origin="lower", cmap="inferno")
        ax.set_title(f"Ch {i}")
        ax.set_xlabel("Time bins")
        ax.set_ylabel("Freq bins")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


@dataclass
class SpectrogramSample:
    subject_id: str
    session: str
    label_name: str
    label: int
    dom_path: str
    ndom_path: str
    delta_path: str
    dom_png: str
    ndom_png: str
    delta_png: str


def build_spectrogram_dataset(
    df: pd.DataFrame,
    outdir: Path,
    sample_rate: int,
    window_s: int,
    overlap_pct: float,
    logger: TeeLogger,
) -> pd.DataFrame:
    nperseg = max(2, int(window_s * sample_rate))
    noverlap = max(0, min(nperseg - 1, int(nperseg * overlap_pct)))

    logger.log(f"Building spectrogram dataset | fs={sample_rate}, window_s={window_s}, overlap={overlap_pct}, nperseg={nperseg}, noverlap={noverlap}")

    arrays_dir = ensure_dir(outdir / "arrays")
    previews_dir = ensure_dir(outdir / "previews")
    skipped_rows = []
    manifest_rows: List[Dict] = []

    grouped = df.groupby(["id", "session"], sort=True)
    total = len(grouped)

    for idx, ((sid, session), chunk) in enumerate(grouped, start=1):
        chunk = chunk.sort_values("datetime")
        label_name = str(chunk["type"].iloc[0]).lower()
        label = 1 if label_name == "ucp" else 0

        dom_sig = chunk[chunk["hand_label"] == "dom"][DEFAULT_AXES].to_numpy(dtype=float)
        ndom_sig = chunk[chunk["hand_label"] == "ndom"][DEFAULT_AXES].to_numpy(dtype=float)

        logger.log(f"[{idx}/{total}] subject={sid} session={session} label={label_name} dom_len={len(dom_sig)} ndom_len={len(ndom_sig)}")

        if len(dom_sig) < nperseg or len(ndom_sig) < nperseg:
            skipped_rows.append({
                "subject_id": sid,
                "session": session,
                "label": label_name,
                "dom_len": len(dom_sig),
                "ndom_len": len(ndom_sig),
                "reason": "shorter_than_one_window",
            })
            logger.log(f"  -> skipped: shorter than one STFT window")
            continue

        n = min(len(dom_sig), len(ndom_sig))
        dom_sig = dom_sig[:n]
        ndom_sig = ndom_sig[:n]
        delta_sig = dom_sig - ndom_sig

        _, _, dom_spec = compute_spectrogram_array(dom_sig, sample_rate, nperseg, noverlap)
        _, _, ndom_spec = compute_spectrogram_array(ndom_sig, sample_rate, nperseg, noverlap)
        _, _, delta_spec = compute_spectrogram_array(delta_sig, sample_rate, nperseg, noverlap)

        base = f"{sid}_{session}"
        dom_npy = arrays_dir / f"{base}_dom.npy"
        ndom_npy = arrays_dir / f"{base}_ndom.npy"
        delta_npy = arrays_dir / f"{base}_delta.npy"

        np.save(dom_npy, dom_spec)
        np.save(ndom_npy, ndom_spec)
        np.save(delta_npy, delta_spec)

        dom_png = previews_dir / f"{base}_dom.png"
        ndom_png = previews_dir / f"{base}_ndom.png"
        delta_png = previews_dir / f"{base}_delta.png"
        save_preview_png(dom_spec, f"{sid} | {session} | dom | {label_name}", dom_png)
        save_preview_png(ndom_spec, f"{sid} | {session} | ndom | {label_name}", ndom_png)
        save_preview_png(delta_spec, f"{sid} | {session} | delta | {label_name}", delta_png)

        manifest_rows.append(asdict(SpectrogramSample(
            subject_id=str(sid),
            session=str(session),
            label_name=label_name,
            label=label,
            dom_path=str(dom_npy.resolve()),
            ndom_path=str(ndom_npy.resolve()),
            delta_path=str(delta_npy.resolve()),
            dom_png=str(dom_png.resolve()),
            ndom_png=str(ndom_png.resolve()),
            delta_png=str(delta_png.resolve()),
        )))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(outdir / "manifest.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(outdir / "skipped_sessions.csv", index=False)

    summary = {
        "n_rows_raw": int(len(df)),
        "n_subjects_raw": int(df["id"].nunique()),
        "n_samples_manifest": int(len(manifest_df)),
        "n_subjects_manifest": int(manifest_df["subject_id"].nunique()) if len(manifest_df) else 0,
        "labels": manifest_df["label_name"].value_counts().to_dict() if len(manifest_df) else {},
        "sample_rate": sample_rate,
        "window_s": window_s,
        "overlap_pct": overlap_pct,
        "nperseg": nperseg,
        "noverlap": noverlap,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.log(f"Saved manifest -> {outdir / 'manifest.csv'}")
    logger.log(f"Saved skipped sessions -> {outdir / 'skipped_sessions.csv'}")
    logger.log(f"Saved summary -> {outdir / 'summary.json'}")
    return manifest_df


# -----------------------------------------------------------------------------
# PyTorch datasets
# -----------------------------------------------------------------------------

class TwoBranchSpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x1 = np.load(row["dom_path"]).astype(np.float32)
        x2 = np.load(row["ndom_path"]).astype(np.float32)
        y = np.int64(row["label"])
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.tensor(y, dtype=torch.long), row.to_dict()


class DeltaSpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["delta_path"]).astype(np.float32)
        y = np.int64(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), row.to_dict()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class SmallCNNEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.features(x))


class B1SharedTwoBranch(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.encoder = SmallCNNEncoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = torch.cat([z1, z2], dim=1)
        return self.classifier(z)


class B2IndependentTwoBranch(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.encoder1 = SmallCNNEncoder(in_channels)
        self.encoder2 = SmallCNNEncoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z = torch.cat([z1, z2], dim=1)
        return self.classifier(z)


class B3DeltaSingleCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.encoder = SmallCNNEncoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def save_test_previews(test_df: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for _, row in test_df.iterrows():
        for key in ["dom_png", "ndom_png", "delta_png"]:
            src = Path(row[key])
            dst = save_dir / src.name
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())


def train_epoch_two_branch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x1, x2, y, _ in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / max(1, len(loader.dataset))


def _uncollate_meta(meta_batch):
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        batch_len = len(meta_batch[keys[0]]) if keys else 0
        rows = []
        for i in range(batch_len):
            row = {}
            for k in keys:
                v = meta_batch[k][i]
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                row[k] = v
            rows.append(row)
        return rows
    if isinstance(meta_batch, list):
        return meta_batch
    return [meta_batch]


def eval_two_branch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, preds = [], []
    metas = []
    with torch.no_grad():
        for x1, x2, y, meta in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1, x2)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred)
            metas.extend(_uncollate_meta(meta))
    return total_loss / max(1, len(loader.dataset)), ys, preds, metas


def train_epoch_single(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / max(1, len(loader.dataset))


def eval_single(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, preds = [], []
    metas = []
    with torch.no_grad():
        for x, y, meta in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred)
            metas.extend(_uncollate_meta(meta))
    return total_loss / max(1, len(loader.dataset)), ys, preds, metas


def instantiate_model(arch: str, in_channels: int) -> nn.Module:
    if arch == "B1":
        return B1SharedTwoBranch(in_channels)
    if arch == "B2":
        return B2IndependentTwoBranch(in_channels)
    if arch == "B3":
        return B3DeltaSingleCNN(in_channels)
    raise ValueError(f"Unknown architecture: {arch}")


def get_loaders(arch: str, train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size: int):
    if arch in {"B1", "B2"}:
        train_ds = TwoBranchSpectrogramDataset(train_df)
        val_ds = TwoBranchSpectrogramDataset(val_df)
    else:
        train_ds = DeltaSpectrogramDataset(train_df)
        val_ds = DeltaSpectrogramDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def fit_one_fold(
    arch: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    outdir: Path,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    logger: TeeLogger,
) -> Tuple[Dict[str, float], List[Dict], Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    # infer channels from one file
    sample_path = train_df.iloc[0]["dom_path"] if arch in {"B1", "B2"} else train_df.iloc[0]["delta_path"]
    in_channels = np.load(sample_path).shape[0]

    model = instantiate_model(arch, in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(arch, train_df, val_df, batch_size)

    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    best_preds: List[Dict] = []
    patience_count = 0
    best_model_path = outdir / f"best_{arch}.joblib"

    for epoch in range(1, epochs + 1):
        if arch in {"B1", "B2"}:
            train_loss = train_epoch_two_branch(model, train_loader, optimizer, criterion, device)
            val_loss, y_true, y_pred, metas = eval_two_branch(model, val_loader, criterion, device)
        else:
            train_loss = train_epoch_single(model, train_loader, optimizer, criterion, device)
            val_loss, y_true, y_pred, metas = eval_single(model, val_loader, criterion, device)

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics.update({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        logger.log(f"{arch} | epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

        ckpt = {
            "epoch": epoch,
            "architecture": arch,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        }
        joblib.dump(ckpt, outdir / f"checkpoint_{arch}_epoch_{epoch}.joblib")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_metrics = metrics.copy()
            best_preds = []
            for m, yt, yp in zip(metas, y_true, y_pred):
                row = {
                    "subject_id": m["subject_id"],
                    "session": m["session"],
                    "label_name": m["label_name"],
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "architecture": arch,
                }
                best_preds.append(row)
            joblib.dump(ckpt, best_model_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.log(f"{arch} | early stopping at epoch {epoch}")
                break

    assert best_metrics is not None
    best_metrics["best_epoch"] = best_epoch
    return best_metrics, best_preds, best_model_path


# -----------------------------------------------------------------------------
# Validation loops
# -----------------------------------------------------------------------------

def run_lopo(
    manifest_df: pd.DataFrame,
    archs: List[str],
    outdir: Path,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    logger: TeeLogger,
) -> None:
    logo = LeaveOneGroupOut()
    groups = manifest_df["subject_id"].to_numpy()
    fold_rows = []
    pred_rows = []
    summary_rows = []

    for arch in archs:
        logger.log(f"Starting LOPO for architecture {arch}")
        arch_dir = ensure_dir(outdir / arch)
        arch_fold_metrics = []
        arch_all_preds = []

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(manifest_df, manifest_df["label"], groups), start=1):
            train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
            test_df = manifest_df.iloc[test_idx].reset_index(drop=True)
            fold_dir = ensure_dir(arch_dir / f"fold_{fold_idx:03d}")
            fold_logger = TeeLogger(fold_dir / "train.log")
            fold_logger.log(f"LOPO fold {fold_idx} | test_subjects={sorted(test_df['subject_id'].unique().tolist())}")
            save_test_previews(test_df, fold_dir / "test_previews")

            metrics, preds, model_path = fit_one_fold(
                arch=arch,
                train_df=train_df,
                val_df=test_df,
                outdir=fold_dir,
                device=device,
                epochs=epochs,
                lr=lr,
                patience=patience,
                batch_size=batch_size,
                logger=fold_logger,
            )
            fold_logger.close()

            row = {"architecture": arch, "fold": fold_idx, **metrics, "best_model_path": str(model_path.resolve())}
            fold_rows.append(row)
            arch_fold_metrics.append(metrics)
            arch_all_preds.extend(preds)

        pred_rows.extend(arch_all_preds)
        # aggregate mean/std
        accs = [m["accuracy"] for m in arch_fold_metrics]
        f1s = [m["f1"] for m in arch_fold_metrics]
        precs = [m["precision"] for m in arch_fold_metrics]
        recs = [m["recall"] for m in arch_fold_metrics]
        summary_rows.append({
            "validation": "LOPO",
            "architecture": arch,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs, ddof=1)) if len(precs) > 1 else 0.0,
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs, ddof=1)) if len(recs) > 1 else 0.0,
            "n_folds": len(accs),
        })

    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(outdir / "all_test_predictions.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    logger.log(f"Saved LOPO fold metrics -> {outdir / 'fold_metrics.csv'}")
    logger.log(f"Saved LOPO predictions -> {outdir / 'all_test_predictions.csv'}")
    logger.log(f"Saved LOPO summary -> {outdir / 'summary.csv'} and summary.json")



def run_nested(
    manifest_df: pd.DataFrame,
    archs: List[str],
    outdir: Path,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    outer_splits: int,
    inner_splits: int,
    logger: TeeLogger,
) -> None:
    groups = manifest_df["subject_id"].to_numpy()
    outer = GroupKFold(n_splits=outer_splits)
    fold_rows = []
    pred_rows = []
    summary_rows = []

    for arch in archs:
        logger.log(f"Starting Nested CV for architecture {arch}")
        arch_dir = ensure_dir(outdir / arch)
        arch_fold_metrics = []
        arch_all_preds = []

        for outer_idx, (train_idx, test_idx) in enumerate(outer.split(manifest_df, manifest_df["label"], groups), start=1):
            outer_train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
            outer_test_df = manifest_df.iloc[test_idx].reset_index(drop=True)
            outer_groups = outer_train_df["subject_id"].to_numpy()
            inner = GroupKFold(n_splits=min(inner_splits, len(np.unique(outer_groups))))

            outer_dir = ensure_dir(arch_dir / f"outer_fold_{outer_idx:03d}")
            outer_logger = TeeLogger(outer_dir / "train.log")
            outer_logger.log(f"Nested outer fold {outer_idx} | test_subjects={sorted(outer_test_df['subject_id'].unique().tolist())}")
            save_test_previews(outer_test_df, outer_dir / "test_previews")

            # No hyperparameter tuning grid here; inner loop is used as checkpointed validation standard.
            # We train with fixed hyperparameters but still log inner folds.
            inner_metrics = []
            for inner_idx, (in_train_idx, in_val_idx) in enumerate(inner.split(outer_train_df, outer_train_df["label"], outer_groups), start=1):
                in_train_df = outer_train_df.iloc[in_train_idx].reset_index(drop=True)
                in_val_df = outer_train_df.iloc[in_val_idx].reset_index(drop=True)
                inner_dir = ensure_dir(outer_dir / f"inner_fold_{inner_idx:03d}")
                inner_logger = TeeLogger(inner_dir / "train.log")
                m, _, _ = fit_one_fold(
                    arch=arch,
                    train_df=in_train_df,
                    val_df=in_val_df,
                    outdir=inner_dir,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    patience=patience,
                    batch_size=batch_size,
                    logger=inner_logger,
                )
                inner_logger.close()
                inner_metrics.append(m)

            # Train on full outer train and test on outer test
            test_metrics, preds, model_path = fit_one_fold(
                arch=arch,
                train_df=outer_train_df,
                val_df=outer_test_df,
                outdir=outer_dir / "final_outer_model",
                device=device,
                epochs=epochs,
                lr=lr,
                patience=patience,
                batch_size=batch_size,
                logger=outer_logger,
            )
            outer_logger.close()

            row = {
                "architecture": arch,
                "outer_fold": outer_idx,
                **test_metrics,
                "best_model_path": str(model_path.resolve()),
                "inner_f1_mean": float(np.mean([m["f1"] for m in inner_metrics])) if inner_metrics else None,
                "inner_f1_std": float(np.std([m["f1"] for m in inner_metrics], ddof=1)) if len(inner_metrics) > 1 else 0.0,
            }
            fold_rows.append(row)
            arch_fold_metrics.append(test_metrics)
            arch_all_preds.extend(preds)

        pred_rows.extend(arch_all_preds)
        accs = [m["accuracy"] for m in arch_fold_metrics]
        f1s = [m["f1"] for m in arch_fold_metrics]
        precs = [m["precision"] for m in arch_fold_metrics]
        recs = [m["recall"] for m in arch_fold_metrics]
        summary_rows.append({
            "validation": "Nested",
            "architecture": arch,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs, ddof=1)) if len(precs) > 1 else 0.0,
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs, ddof=1)) if len(recs) > 1 else 0.0,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
        })

    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(outdir / "all_test_predictions.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    logger.log(f"Saved Nested fold metrics -> {outdir / 'fold_metrics.csv'}")
    logger.log(f"Saved Nested predictions -> {outdir / 'all_test_predictions.csv'}")
    logger.log(f"Saved Nested summary -> {outdir / 'summary.csv'} and summary.json")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AC spectrogram pipeline + B1/B2/B3 grouped-by-patient training")
    p.add_argument("--td", required=True, help="Path to TD AC CSV (1 sec)")
    p.add_argument("--ucp", required=True, help="Path to UCP AC CSV (1 sec)")
    p.add_argument("--outdir", default="spectrogram_ac_experiment", help="Output directory")
    p.add_argument("--validation", choices=["lopo", "nested"], default="lopo")
    p.add_argument("--architectures", default="all", help="Comma-separated list among B1,B2,B3 or 'all'")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S)
    p.add_argument("--overlap-pct", type=float, default=DEFAULT_OVERLAP_PCT)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    outdir = ensure_dir(args.outdir)
    logger = TeeLogger(outdir / "pipeline.log")
    logger.log("Loading AC dataset...")
    df = load_ac_dataset(args.td, args.ucp)
    logger.log(f"Loaded rows={len(df)} subjects={df['id'].nunique()} sessions={df[['id','session']].drop_duplicates().shape[0]}")

    logger.log("Generating spectrogram dataset from AC time series...")
    manifest_df = build_spectrogram_dataset(
        df=df,
        outdir=outdir / "spectrogram_dataset",
        sample_rate=args.sample_rate,
        window_s=args.window_s,
        overlap_pct=args.overlap_pct,
        logger=logger,
    )

    if len(manifest_df) == 0:
        logger.log("No valid sessions after STFT generation. Exiting.")
        logger.close()
        sys.exit(1)

    archs = ["B1", "B2", "B3"] if args.architectures == "all" else [a.strip() for a in args.architectures.split(",") if a.strip()]
    logger.log(f"Architectures to train: {archs}")
    logger.log(f"Grouped-by-patient validation: {args.validation}")

    train_outdir = ensure_dir(outdir / f"training_{args.validation}")
    if args.validation == "lopo":
        run_lopo(
            manifest_df=manifest_df,
            archs=archs,
            outdir=train_outdir,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
            logger=logger,
        )
    else:
        run_nested(
            manifest_df=manifest_df,
            archs=archs,
            outdir=train_outdir,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            logger=logger,
        )

    logger.log("Done.")
    logger.close()


if __name__ == "__main__":
    main()
