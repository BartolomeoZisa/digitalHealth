#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_MODELS = [
    "efficientnet_b0_two_branch",
    "resnet18_two_branch",
    "efficientnet_b0_delta",
    "single_cnn_concat",
    "single_cnn_concat_delta",
]

TWO_BRANCH_MODELS = {
    "efficientnet_b0_two_branch",
    "resnet18_two_branch",
}

SINGLE_INPUT_MODELS = {
    "efficientnet_b0_delta",
    "single_cnn_concat",
    "single_cnn_concat_delta",
}


# =============================================================================
# Config
# =============================================================================

@dataclass
class RunConfig:
    manifest: str
    out_dir: str
    run_name: str
    models: list[str]
    seed: int
    outer_folds: int
    inner_folds: int
    epochs: int
    inner_epochs: int
    patience: int
    image_size: int
    batch_sizes: list[int]
    learning_rates: list[float]
    weight_decays: list[float]
    pretrained: bool
    num_workers: int
    monitor_metric: str
    save_joblib: bool
    device: str


# =============================================================================
# Logging and reproducibility
# =============================================================================

def setup_logger(log_path: Path, console: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ac_spectrogram_training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More reproducible, slightly slower.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Useful for Mac. Harmless elsewhere.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Dataset
# =============================================================================

class ACSpectrogramDataset(Dataset):
    """
    Dataset for Activity Count spectrograms.

    Returned tensors are resized to image_size x image_size and normalized with
    safe per-channel standardization.

    For flat channels, std is close to zero. In that case the standardized
    channel is set to zero instead of dividing by zero.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        model_name: str,
        image_size: int = 224,
        normalize: str = "per_sample",
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.label_to_idx = label_to_idx
        self.model_name = model_name
        self.image_size = image_size
        self.normalize = normalize

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_npy(path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Expected shape from dataset creation: (C, F, T).
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        if arr.ndim != 3:
            raise ValueError(f"Expected npy with shape (C,F,T), got {arr.shape} for {path}")

        return torch.from_numpy(arr)

    @staticmethod
    def _safe_per_channel_standardize(x: torch.Tensor) -> torch.Tensor:
        # x shape: (C, H, W)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)

        x_norm = torch.where(
            std > 1e-6,
            (x - mean) / (std + 1e-6),
            torch.zeros_like(x),
        )

        return x_norm

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, F, T) -> (C, image_size, image_size)
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "per_sample":
            x = self._safe_per_channel_standardize(x)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        x = self._resize(x)
        return x

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        active = self._load_npy(row["active_path"])
        mirror = self._load_npy(row["mirror_path"])
        delta = self._load_npy(row["delta_path"]) if isinstance(row["delta_path"], str) and row["delta_path"] else None

        y = self.label_to_idx[str(row["label"])]

        sample: dict[str, Any] = {
            "y": torch.tensor(y, dtype=torch.long),
            "id": str(row["id"]),
            "session": str(row["session"]),
            "label": str(row["label"]),
            "row_index": int(row.name),
        }

        if self.model_name in TWO_BRANCH_MODELS:
            sample["active"] = self._process(active)
            sample["mirror"] = self._process(mirror)
            return sample

        if self.model_name == "efficientnet_b0_delta":
            if delta is None:
                raise ValueError("efficientnet_b0_delta requires delta_path in manifest.")

            
            x = torch.cat([active, mirror, delta], dim=0)
            sample["x"] = self._process(x)
            return sample

        if self.model_name == "single_cnn_concat":
            
            x = torch.cat([active, mirror], dim=0)
            sample["x"] = self._process(x)
            return sample

        if self.model_name == "single_cnn_concat_delta":
            
            if delta is None:
                raise ValueError("single_cnn_concat_delta requires delta_path in manifest.")

            x = torch.cat([active, mirror, delta], dim=0)
            sample["x"] = self._process(x)
            return sample

        raise ValueError(f"Unsupported model_name: {self.model_name}")


# =============================================================================
# Models
# =============================================================================

def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Adapt an existing Conv2d to a different number of input channels.

    If the original conv has 3 channels and pretrained weights, the new channels
    are initialized using the mean over RGB channels. This allows 1-channel,
    2-channel, or 3-channel spectrogram inputs.
    """
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        old_w = conv.weight.data

        if old_w.shape[1] == 3:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_w = mean_w.repeat(1, in_channels, 1, 1)
        else:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_w = mean_w.repeat(1, in_channels, 1, 1)

        # Keep the overall activation scale approximately stable.
        new_w = new_w * (old_w.shape[1] / float(in_channels))
        new_conv.weight.copy_(new_w)

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)

    return new_conv


def build_resnet18_backbone(in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            print(f"[WARN] Could not load pretrained ResNet18 weights ({exc}). Falling back to random init.")
        model = resnet18(weights=None)

    model.conv1 = adapt_first_conv(model.conv1, in_channels)

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    return model, feature_dim


def build_efficientnet_b0_backbone(in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

    try:
        model = efficientnet_b0(weights=weights)
    except Exception as exc:
        if pretrained:
            print(f"[WARN] Could not load pretrained EfficientNet-B0 weights ({exc}). Falling back to random init.")
        model = efficientnet_b0(weights=None)

    # torchvision EfficientNet-B0 first conv is usually model.features[0][0].
    first_conv = model.features[0][0]
    model.features[0][0] = adapt_first_conv(first_conv, in_channels)

    if isinstance(model.classifier, nn.Sequential):
        feature_dim = None
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                feature_dim = layer.in_features
                break
        if feature_dim is None:
            raise RuntimeError("Could not infer EfficientNet classifier input size.")
    else:
        feature_dim = model.classifier.in_features

    model.classifier = nn.Identity()

    return model, feature_dim


def build_backbone(name: str, in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet18":
        return build_resnet18_backbone(in_channels=in_channels, pretrained=pretrained)

    if name == "efficientnet_b0":
        return build_efficientnet_b0_backbone(in_channels=in_channels, pretrained=pretrained)

    raise ValueError(f"Unknown backbone: {name}")


class TwoBranchSharedWeights(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.backbone, feature_dim = build_backbone(
            name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, active: torch.Tensor, mirror: torch.Tensor) -> torch.Tensor:
        fa = self.backbone(active)
        fm = self.backbone(mirror)

        z = torch.cat([fa, fm], dim=1)
        return self.classifier(z)


class SingleBackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.backbone, feature_dim = build_backbone(
            name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.classifier(z)


class SmallCNNConcat(nn.Module):
    """
    Lightweight CNN for the [active, mirror] concatenation baseline.

    This is intentionally smaller than EfficientNet/ResNet because the dataset
    is very small and the original spectrograms are only 9 x 6 before resizing.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.30) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "efficientnet_b0_two_branch":
        return TwoBranchSharedWeights(
            backbone_name="efficientnet_b0",
            in_channels=1,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "resnet18_two_branch":
        return TwoBranchSharedWeights(
            backbone_name="resnet18",
            in_channels=1,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "efficientnet_b0_delta":
        return SingleBackboneClassifier(
            backbone_name="efficientnet_b0",
            in_channels=3,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "single_cnn_concat":
        return SmallCNNConcat(
            in_channels=2,
            num_classes=num_classes,
        )

    if model_name == "single_cnn_concat_delta":
        return SmallCNNConcat(
            in_channels=3,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[np.ndarray],
    labels: list[int],
    idx_to_label: dict[int, str],
) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }

    if len(labels) == 2:
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update(
                {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

    per_class = {}
    for class_idx in labels:
        class_name = idx_to_label[class_idx]
        support = int(np.sum(np.array(y_true) == class_idx))
        correct = int(np.sum((np.array(y_true) == class_idx) & (np.array(y_pred) == class_idx)))
        per_class[class_name] = {
            "support": support,
            "correct": correct,
            "accuracy": float(correct / support) if support > 0 else None,
        }

    metrics["per_class"] = per_class

    return metrics


# =============================================================================
# Training utilities
# =============================================================================

def make_loader(
    df: pd.DataFrame,
    label_to_idx: dict[str, int],
    model_name: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = ACSpectrogramDataset(
        df=df,
        label_to_idx=label_to_idx,
        model_name=model_name,
        image_size=image_size,
        normalize="per_sample",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def class_weights_from_df(df: pd.DataFrame, label_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    labels = df["label"].astype(str).map(label_to_idx).to_numpy()
    num_classes = len(label_to_idx)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def forward_model(model: nn.Module, model_name: str, batch: dict[str, Any]) -> torch.Tensor:
    if model_name in TWO_BRANCH_MODELS:
        return model(batch["active"], batch["mirror"])

    return model(batch["x"])


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    losses = []
    y_true = []
    y_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = forward_model(model, model_name, batch)
        loss = criterion(logits, batch["y"])

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(batch["y"].detach().cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else math.nan,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else math.nan,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    idx_to_label: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()

    losses = []
    y_true = []
    y_pred = []
    y_prob_rows = []

    meta_rows = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = forward_model(model, model_name, batch)

        if criterion is not None:
            loss = criterion(logits, batch["y"])
            losses.append(float(loss.item()))

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_batch = batch["y"].detach().cpu().numpy()
        p_batch = preds.detach().cpu().numpy()
        prob_batch = probs.detach().cpu().numpy()

        y_true.extend(y_batch.tolist())
        y_pred.extend(p_batch.tolist())
        y_prob_rows.append(prob_batch)

        batch_size = len(y_batch)
        for i in range(batch_size):
            row = {
                "id": batch["id"][i],
                "session": batch["session"][i],
                "true_label": idx_to_label[int(y_batch[i])],
                "pred_label": idx_to_label[int(p_batch[i])],
                "true_idx": int(y_batch[i]),
                "pred_idx": int(p_batch[i]),
            }

            for class_idx, class_name in idx_to_label.items():
                row[f"prob_{class_name}"] = float(prob_batch[i, class_idx])

            meta_rows.append(row)

    y_prob = np.concatenate(y_prob_rows, axis=0) if y_prob_rows else None
    labels = list(idx_to_label.keys())

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=labels,
        idx_to_label=idx_to_label,
    )

    metrics["loss"] = float(np.mean(losses)) if losses else None

    return metrics, pd.DataFrame(meta_rows)


def metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return -float("inf")
    return float(value)


def run_training_with_early_stopping(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    params: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
    logger: logging.Logger,
    save_best: bool = False,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame]:
    set_seed(args.seed + int(params.get("seed_offset", 0)))

    model = build_model(
        model_name=model_name,
        num_classes=len(label_to_idx),
        pretrained=args.pretrained,
    ).to(device)

    train_loader = make_loader(
        df=train_df,
        label_to_idx=label_to_idx,
        model_name=model_name,
        image_size=args.image_size,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = make_loader(
        df=val_df,
        label_to_idx=label_to_idx,
        model_name=model_name,
        image_size=args.image_size,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_df(train_df, label_to_idx, device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    history = []
    patience_counter = 0

    epochs = int(params["epochs"])

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            model_name=model_name,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics, _ = evaluate(
            model=model,
            model_name=model_name,
            loader=val_loader,
            criterion=criterion,
            device=device,
            idx_to_label=idx_to_label,
        )

        score = metric_value(val_metrics, args.monitor_metric)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "monitor_score": score,
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "batch_size": params["batch_size"],
        }
        history.append(row)

        logger.info(
            "%s | epoch=%03d/%03d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            model_name,
            epoch,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"] if val_metrics["loss"] is not None else -1.0,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if save_best:
                torch.save(
                    {
                        "model_name": model_name,
                        "state_dict": best_state,
                        "label_to_idx": label_to_idx,
                        "params": params,
                        "epoch": epoch,
                        "best_score": best_score,
                        "monitor_metric": args.monitor_metric,
                    },
                    fold_dir / "best_model.pth",
                )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(
                "%s | early stopping at epoch=%d | best_epoch=%d | best_%s=%.4f",
                model_name,
                epoch,
                best_epoch,
                args.monitor_metric,
                best_score,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, {
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "monitor_metric": args.monitor_metric,
    }, history_df


# =============================================================================
# Nested CV
# =============================================================================

def build_hyperparameter_grid(args: argparse.Namespace, inner_epochs: bool = True) -> list[dict[str, Any]]:
    grid = []

    for lr in args.learning_rates:
        for wd in args.weight_decays:
            for bs in args.batch_sizes:
                grid.append(
                    {
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "batch_size": int(bs),
                        "epochs": int(args.inner_epochs if inner_epochs else args.epochs),
                    }
                )

    return grid


def make_stratified_group_splitter(n_splits: int, seed: int):
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )


def split_train_val_for_final(
    outer_train_df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    
    y = outer_train_df["label"].astype(str).to_numpy()
    groups = outer_train_df["id"].astype(str).to_numpy()

    n_groups = outer_train_df["id"].nunique()
    n_splits = min(max(2, n_splits), n_groups)

    splitter = make_stratified_group_splitter(n_splits=n_splits, seed=seed)

    train_idx, val_idx = next(splitter.split(outer_train_df, y, groups))
    return train_idx, val_idx


def copy_test_spectrograms(test_df: pd.DataFrame, fold_dir: Path, logger: logging.Logger) -> None:
    dest_dir = fold_dir / "test_spectrograms"
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for _, row in test_df.iterrows():
        for col in ["active_png_path", "mirror_png_path", "delta_png_path"]:
            path = row.get(col, "")
            if not isinstance(path, str) or not path:
                continue

            src = Path(path)
            if not src.exists():
                continue

            dst_name = f'{row["id"]}_{row["session"]}_{src.name}'
            dst = dest_dir / dst_name

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as exc:
                logger.warning("Could not copy %s -> %s: %s", src, dst, exc)

    logger.info("Copied %d test spectrogram PNGs to %s", copied, dest_dir)


def save_joblib_backup(
    model: nn.Module,
    model_name: str,
    label_to_idx: dict[str, int],
    params: dict[str, Any],
    path: Path,
    logger: logging.Logger,
) -> None:
    try:
        import joblib

        package = {
            "model_name": model_name,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "label_to_idx": label_to_idx,
            "params": params,
            "note": (
                "PyTorch .pth is the recommended format. "
                "This .joblib file is provided only as an additional backup."
            ),
        }

        joblib.dump(package, path)
    except Exception as exc:
        logger.warning("Could not save joblib backup at %s: %s", path, exc)


def run_nested_cv_for_model(
    model_name: str,
    manifest: pd.DataFrame,
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    y = manifest["label"].astype(str).to_numpy()
    groups = manifest["id"].astype(str).to_numpy()

    outer_splitter = make_stratified_group_splitter(
        n_splits=args.outer_folds,
        seed=args.seed,
    )

    outer_summary_rows = []
    all_pred_rows = []

    logger.info("=" * 90)
    logger.info("MODEL START: %s", model_name)
    logger.info("=" * 90)

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(manifest, y, groups),
        start=1,
    ):
        fold_start = time.time()

        fold_dir = model_dir / f"outer_fold_{outer_fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        outer_train_df = manifest.iloc[outer_train_idx].reset_index(drop=True)
        outer_test_df = manifest.iloc[outer_test_idx].reset_index(drop=True)

        logger.info(
            "[%s] Outer fold %d/%d | train_rows=%d | test_rows=%d | train_patients=%d | test_patients=%d",
            model_name,
            outer_fold,
            args.outer_folds,
            len(outer_train_df),
            len(outer_test_df),
            outer_train_df["id"].nunique(),
            outer_test_df["id"].nunique(),
        )

        fold_indices = pd.concat(
            [
                outer_train_df.assign(split="outer_train"),
                outer_test_df.assign(split="outer_test"),
            ],
            ignore_index=True,
        )
        fold_indices.to_csv(fold_dir / "fold_indices.csv", index=False)

        # ---------------------------------------------------------------------
        # Inner CV for hyperparameter selection
        # ---------------------------------------------------------------------
        hyper_grid = build_hyperparameter_grid(args, inner_epochs=True)

        inner_results = []

        inner_y = outer_train_df["label"].astype(str).to_numpy()
        inner_groups = outer_train_df["id"].astype(str).to_numpy()

        inner_splitter = make_stratified_group_splitter(
            n_splits=args.inner_folds,
            seed=args.seed + outer_fold,
        )

        logger.info(
            "[%s] Outer fold %d | inner CV | combinations=%d",
            model_name,
            outer_fold,
            len(hyper_grid),
        )

        for combo_idx, params in enumerate(hyper_grid, start=1):
            combo_scores = []

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_splitter.split(outer_train_df, inner_y, inner_groups),
                start=1,
            ):
                inner_train_df = outer_train_df.iloc[inner_train_idx].reset_index(drop=True)
                inner_val_df = outer_train_df.iloc[inner_val_idx].reset_index(drop=True)

                params_inner = dict(params)
                params_inner["seed_offset"] = outer_fold * 1000 + combo_idx * 100 + inner_fold

                logger.info(
                    "[%s] Outer %d | combo %d/%d | inner %d/%d | params=%s",
                    model_name,
                    outer_fold,
                    combo_idx,
                    len(hyper_grid),
                    inner_fold,
                    args.inner_folds,
                    params,
                )

                inner_tmp_dir = fold_dir / "inner_cv_tmp"
                inner_tmp_dir.mkdir(exist_ok=True)

                model, best_info, history_df = run_training_with_early_stopping(
                    model_name=model_name,
                    train_df=inner_train_df,
                    val_df=inner_val_df,
                    label_to_idx=label_to_idx,
                    idx_to_label=idx_to_label,
                    params=params_inner,
                    args=args,
                    device=device,
                    fold_dir=inner_tmp_dir,
                    logger=logger,
                    save_best=False,
                )

                val_loader = make_loader(
                    df=inner_val_df,
                    label_to_idx=label_to_idx,
                    model_name=model_name,
                    image_size=args.image_size,
                    batch_size=params["batch_size"],
                    shuffle=False,
                    num_workers=args.num_workers,
                )

                criterion = nn.CrossEntropyLoss(
                    weight=class_weights_from_df(inner_train_df, label_to_idx, device)
                )

                val_metrics, _ = evaluate(
                    model=model,
                    model_name=model_name,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    idx_to_label=idx_to_label,
                )

                score = metric_value(val_metrics, args.monitor_metric)
                combo_scores.append(score)

                inner_results.append(
                    {
                        "outer_fold": outer_fold,
                        "combo_idx": combo_idx,
                        "inner_fold": inner_fold,
                        "lr": params["lr"],
                        "weight_decay": params["weight_decay"],
                        "batch_size": params["batch_size"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "monitor_score": score,
                        "best_epoch": best_info["best_epoch"],
                    }
                )

            logger.info(
                "[%s] Outer %d | combo %d/%d | mean_%s=%.4f",
                model_name,
                outer_fold,
                combo_idx,
                len(hyper_grid),
                args.monitor_metric,
                float(np.mean(combo_scores)),
            )

        inner_results_df = pd.DataFrame(inner_results)
        inner_results_df.to_csv(fold_dir / "inner_results.csv", index=False)

        # Select best hyperparameters.
        combo_summary = (
            inner_results_df
            .groupby(["combo_idx", "lr", "weight_decay", "batch_size"], as_index=False)
            ["monitor_score"]
            .mean()
            .sort_values("monitor_score", ascending=False)
        )

        combo_summary.to_csv(fold_dir / "inner_combo_summary.csv", index=False)

        best_combo = combo_summary.iloc[0].to_dict()

        best_params = {
            "lr": float(best_combo["lr"]),
            "weight_decay": float(best_combo["weight_decay"]),
            "batch_size": int(best_combo["batch_size"]),
            "epochs": int(args.epochs),
        }

        logger.info(
            "[%s] Outer fold %d | selected params=%s | inner_mean_%s=%.4f",
            model_name,
            outer_fold,
            best_params,
            args.monitor_metric,
            float(best_combo["monitor_score"]),
        )

        # ---------------------------------------------------------------------
        # Final model for this outer fold.
        # Use one grouped validation split from outer_train for early stopping.
        # Outer test remains untouched until the final evaluation.
        # ---------------------------------------------------------------------
        final_train_idx, final_val_idx = split_train_val_for_final(
            outer_train_df=outer_train_df,
            n_splits=args.inner_folds,
            seed=args.seed + 10_000 + outer_fold,
        )

        final_train_df = outer_train_df.iloc[final_train_idx].reset_index(drop=True)
        final_val_df = outer_train_df.iloc[final_val_idx].reset_index(drop=True)

        best_params["seed_offset"] = outer_fold * 10_000

        model, best_info, train_log_df = run_training_with_early_stopping(
            model_name=model_name,
            train_df=final_train_df,
            val_df=final_val_df,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            params=best_params,
            args=args,
            device=device,
            fold_dir=fold_dir,
            logger=logger,
            save_best=True,
        )

        train_log_df.to_csv(fold_dir / "train_log.csv", index=False)

        test_loader = make_loader(
            df=outer_test_df,
            label_to_idx=label_to_idx,
            model_name=model_name,
            image_size=args.image_size,
            batch_size=best_params["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
        )

        test_criterion = nn.CrossEntropyLoss(
            weight=class_weights_from_df(final_train_df, label_to_idx, device)
        )

        test_metrics, predictions_df = evaluate(
            model=model,
            model_name=model_name,
            loader=test_loader,
            criterion=test_criterion,
            device=device,
            idx_to_label=idx_to_label,
        )

        predictions_df.insert(0, "model", model_name)
        predictions_df.insert(1, "outer_fold", outer_fold)
        predictions_df.to_csv(fold_dir / "predictions.csv", index=False)

        for _, pred_row in predictions_df.iterrows():
            all_pred_rows.append(pred_row.to_dict())

        copy_test_spectrograms(outer_test_df, fold_dir, logger)

        if args.save_joblib:
            save_joblib_backup(
                model=model,
                model_name=model_name,
                label_to_idx=label_to_idx,
                params=best_params,
                path=fold_dir / "best_model.joblib",
                logger=logger,
            )

        fold_elapsed = time.time() - fold_start

        metrics_package = {
            "model": model_name,
            "outer_fold": outer_fold,
            "test_metrics": test_metrics,
            "selected_params": best_params,
            "inner_best_score": float(best_combo["monitor_score"]),
            "final_best_info": best_info,
            "elapsed_sec": float(fold_elapsed),
            "num_train_patients": int(outer_train_df["id"].nunique()),
            "num_test_patients": int(outer_test_df["id"].nunique()),
            "num_train_rows": int(len(outer_train_df)),
            "num_test_rows": int(len(outer_test_df)),
        }

        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_package, f, indent=4)

        summary_row = {
            "model": model_name,
            "outer_fold": outer_fold,
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "loss": test_metrics["loss"],
            "tn": test_metrics.get("tn"),
            "fp": test_metrics.get("fp"),
            "fn": test_metrics.get("fn"),
            "tp": test_metrics.get("tp"),
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"],
            "batch_size": best_params["batch_size"],
            "best_epoch": best_info["best_epoch"],
            "inner_best_score": float(best_combo["monitor_score"]),
            "elapsed_sec": float(fold_elapsed),
        }

        outer_summary_rows.append(summary_row)

        logger.info(
            "[%s] Outer fold %d DONE | acc=%.4f | bal_acc=%.4f | macro_f1=%.4f | elapsed=%.1fs",
            model_name,
            outer_fold,
            test_metrics["accuracy"],
            test_metrics["balanced_accuracy"],
            test_metrics["macro_f1"],
            fold_elapsed,
        )

    model_summary_df = pd.DataFrame(outer_summary_rows)
    model_predictions_df = pd.DataFrame(all_pred_rows)

    model_summary_df.to_csv(model_dir / "summary.csv", index=False)
    model_predictions_df.to_csv(model_dir / "all_predictions.csv", index=False)

    aggregate = {
        "model": model_name,
        "accuracy_mean": float(model_summary_df["accuracy"].mean()),
        "accuracy_std": float(model_summary_df["accuracy"].std(ddof=1)),
        "balanced_accuracy_mean": float(model_summary_df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(model_summary_df["balanced_accuracy"].std(ddof=1)),
        "macro_f1_mean": float(model_summary_df["macro_f1"].mean()),
        "macro_f1_std": float(model_summary_df["macro_f1"].std(ddof=1)),
    }

    with open(model_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)

    logger.info(
        "MODEL DONE: %s | acc=%.4f±%.4f | bal_acc=%.4f±%.4f | macro_f1=%.4f±%.4f",
        model_name,
        aggregate["accuracy_mean"],
        aggregate["accuracy_std"],
        aggregate["balanced_accuracy_mean"],
        aggregate["balanced_accuracy_std"],
        aggregate["macro_f1_mean"],
        aggregate["macro_f1_std"],
    )

    return model_summary_df, model_predictions_df


# =============================================================================
# Manifest validation
# =============================================================================

def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = [
        "id",
        "session",
        "label",
        "active_path",
        "mirror_path",
        "delta_path",
    ]

    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    manifest = manifest.copy()
    manifest["label"] = manifest["label"].astype(str)
    manifest["id"] = manifest["id"].astype(str)
    manifest["session"] = manifest["session"].astype(str)

    for col in ["active_path", "mirror_path", "delta_path"]:
        missing_files = []
        for path in manifest[col].astype(str):
            if path and not Path(path).exists():
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(
                f"Column {col} contains missing files. First examples: {missing_files[:5]}"
            )

    return manifest


def inspect_flat_channels(manifest: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Lightweight debug inspection to confirm flat channels are handled.
    """
    active_flat = 0
    mirror_flat = 0
    delta_flat = 0

    for _, row in manifest.iterrows():
        active = np.load(row["active_path"])
        mirror = np.load(row["mirror_path"])
        delta = np.load(row["delta_path"])

        if float(np.std(active)) < 1e-6:
            active_flat += 1
        if float(np.std(mirror)) < 1e-6:
            mirror_flat += 1
        if float(np.std(delta)) < 1e-6:
            delta_flat += 1

    logger.info(
        "Flat-channel inspection | active=%d/%d | mirror=%d/%d | delta=%d/%d",
        active_flat,
        len(manifest),
        mirror_flat,
        len(manifest),
        delta_flat,
        len(manifest),
    )


# =============================================================================
# Main
# =============================================================================

def parse_float_list(values: list[str]) -> list[float]:
    return [float(v) for v in values]


def parse_int_list(values: list[str]) -> list[int]:
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN models on Activity Count spectrograms with nested grouped CV."
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/ac_spectrogram_dataset/manifest.csv",
        help="Path to manifest.csv produced by create_ac_spectrogram_dataset.py.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ac_spectrogram_training",
        help="Directory where training outputs will be saved.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Default: timestamped run.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help="Models to train.",
    )

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Max epochs for final outer-fold training.",
    )

    parser.add_argument(
        "--inner-epochs",
        type=int,
        default=8,
        help="Max epochs for each inner-CV training run.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize spectrograms to image_size x image_size.",
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=["8"],
        help="Batch sizes to try in inner CV.",
    )

    parser.add_argument(
        "--learning-rates",
        nargs="+",
        default=["0.001", "0.0003"],
        help="Learning rates to try in inner CV.",
    )

    parser.add_argument(
        "--weight-decays",
        nargs="+",
        default=["0.0001"],
        help="Weight decays to try in inner CV.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Use ImageNet pretrained torchvision weights. "
            "If weights are not available locally and internet is unavailable, the script falls back to random init."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for maximum compatibility.",
    )

    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "balanced_accuracy", "accuracy"],
        help="Metric used for model selection.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, ...",
    )

    parser.add_argument(
        "--save-joblib",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a .joblib backup in addition to the recommended .pth checkpoint.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console logs. File logs are always saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.batch_sizes = parse_int_list(args.batch_sizes)
    args.learning_rates = parse_float_list(args.learning_rates)
    args.weight_decays = parse_float_list(args.weight_decays)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log", console=not args.quiet)

    set_seed(args.seed)

    device = get_device(args.device)

    logger.info("Training started.")
    logger.info("Device: %s", device)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Arguments: %s", vars(args))

    manifest = pd.read_csv(args.manifest)
    manifest = validate_manifest(manifest)

    labels = sorted(manifest["label"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    logger.info("Manifest rows: %d", len(manifest))
    logger.info("Patients: %d", manifest["id"].nunique())
    logger.info("Labels: %s", label_to_idx)
    logger.info("Rows by label:\n%s", manifest["label"].value_counts().to_string())
    logger.info("Patients by label:\n%s", manifest.drop_duplicates("id")["label"].value_counts().to_string())

    inspect_flat_channels(manifest, logger)

    if args.outer_folds > manifest["id"].nunique():
        raise ValueError("--outer-folds cannot exceed the number of patients/groups.")

    config = RunConfig(
        manifest=args.manifest,
        out_dir=args.out_dir,
        run_name=run_name,
        models=args.models,
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        epochs=args.epochs,
        inner_epochs=args.inner_epochs,
        patience=args.patience,
        image_size=args.image_size,
        batch_sizes=args.batch_sizes,
        learning_rates=args.learning_rates,
        weight_decays=args.weight_decays,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
        monitor_metric=args.monitor_metric,
        save_joblib=args.save_joblib,
        device=str(device),
    )

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)

    all_summaries = []
    all_predictions = []

    total_start = time.time()

    for model_name in args.models:
        model_summary_df, model_predictions_df = run_nested_cv_for_model(
            model_name=model_name,
            manifest=manifest,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            args=args,
            run_dir=run_dir,
            device=device,
            logger=logger,
        )

        all_summaries.append(model_summary_df)
        all_predictions.append(model_predictions_df)

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    final_summary.to_csv(run_dir / "final_summary.csv", index=False)
    final_predictions.to_csv(run_dir / "all_predictions.csv", index=False)

    aggregate_rows = []

    for model_name, g in final_summary.groupby("model"):
        aggregate_rows.append(
            {
                "model": model_name,
                "accuracy_mean": float(g["accuracy"].mean()),
                "accuracy_std": float(g["accuracy"].std(ddof=1)),
                "balanced_accuracy_mean": float(g["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(g["balanced_accuracy"].std(ddof=1)),
                "macro_f1_mean": float(g["macro_f1"].mean()),
                "macro_f1_std": float(g["macro_f1"].std(ddof=1)),
                "mean_elapsed_sec": float(g["elapsed_sec"].mean()),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("macro_f1_mean", ascending=False)
    aggregate_df.to_csv(run_dir / "aggregate_summary.csv", index=False)

    final_json = {
        "run_name": run_name,
        "elapsed_sec": float(time.time() - total_start),
        "aggregate_summary": aggregate_rows,
    }

    with open(run_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    logger.info("=" * 90)
    logger.info("FINAL AGGREGATE SUMMARY")
    logger.info("=" * 90)
    logger.info("\n%s", aggregate_df.to_string(index=False))
    logger.info("Saved outputs in: %s", run_dir)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
