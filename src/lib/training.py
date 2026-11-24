"""Piezas de apoyo para entrenar los clasificadores (Etapa 2)."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lib.preprocessing import preprocess_bgr

if TYPE_CHECKING:
    from lib.config import Settings


@dataclass
class TrainConfig:
    """Hiperparámetros de entrenamiento (vienen del .env, ver Settings)."""

    splits_file: Path
    embedding_dim: int = 512
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3
    patience: int = 3
    seed: int = 42
    resnet_finetune: str = "head"
    limit_per_class: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> TrainConfig:
        return cls(
            splits_file=settings.splits_file,
            embedding_dim=int(settings.embedding_dim),
            epochs=int(settings.train_epochs),
            batch_size=int(settings.train_batch_size),
            lr=float(settings.train_lr),
            weight_decay=float(settings.train_weight_decay),
            label_smoothing=float(settings.train_label_smoothing),
            dropout=float(settings.train_dropout),
            patience=int(settings.train_patience),
            seed=int(settings.train_seed),
            resnet_finetune=settings.resnet_finetune,
            limit_per_class=int(settings.train_limit_per_class),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(
    splits_file: Path, dataset_path: Path, split: str, limit_per_class: int = 0
) -> list[tuple[Path, str]]:
    """Lee del CSV de splits las (ruta, raza) de un split (train | valid | test)."""
    items: list[tuple[Path, str]] = []
    per_class: dict[str, int] = defaultdict(int)
    with open(splits_file, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["data set"] != split:
                continue
            # Normaliza espacios: el dataset trae "American  Spaniel" (doble espacio) en valid
            # y "American Spaniel" en train; sin esto quedarian como dos razas distintas.
            breed = " ".join(row["labels"].split())
            if limit_per_class and per_class[breed] >= limit_per_class:
                continue
            per_class[breed] += 1
            items.append((dataset_path / row["filepaths"], breed))
    return items


class DogDataset(Dataset):
    """Imagen BGR (OpenCV) -> augmentation opcional -> mismo preprocess_bgr que usa la app."""

    def __init__(
        self,
        items: list[tuple[Path, str]],
        classes: list[str],
        image_size: int,
        augment: Optional[Callable[..., dict]] = None,
    ) -> None:
        self.items = items
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, breed = self.items[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        if self.augment is not None:
            image = self.augment(image=image)["image"]
        tensor = preprocess_bgr(image, self.image_size).squeeze(0)
        return tensor, self.class_to_idx[breed]


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """BatchNorm de capas congeladas en eval: que no actualicen sus estadísticas de ImageNet."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    """Una pasada sobre `loader`. Entrena si recibe optimizer; si no, solo evalúa.

    Retorna (loss medio, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    if training:
        set_frozen_batchnorm_eval(model)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)
