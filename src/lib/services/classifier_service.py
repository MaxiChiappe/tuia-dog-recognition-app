from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import onnxruntime
from torch import nn
from torch.utils.data import DataLoader

from lib.models import BreedClassifier, build_cnn_custom_classifier, build_resnet18_classifier
from lib.preprocessing import build_train_augmentation
from lib.training import DogDataset, TrainConfig, load_split, run_epoch, set_seed

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
        train_config: Optional[TrainConfig] = None,
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        # Hiperparametros de entrenamiento (Etapa 2), leidos del .env en bootstrap
        self.train_config = train_config or TrainConfig(splits_file=dataset_path / "dogs.csv")
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        cfg = self.train_config
        name = self.active_model_name
        set_seed(cfg.seed)

        train_items = load_split(cfg.splits_file, self.dataset_path, "train", cfg.limit_per_class)
        valid_items = load_split(cfg.splits_file, self.dataset_path, "valid", cfg.limit_per_class)
        if not train_items or not valid_items:
            raise ValueError(f"Splits train/valid vacios en {cfg.splits_file}")
        classes = sorted({breed for _, breed in train_items})
        logger.info(
            "Training %s: %d train / %d valid images, %d breeds",
            name, len(train_items), len(valid_items), len(classes),
        )

        model = self._build_model(name, classes)
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "Trainable params: %d / %d",
            sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters()),
        )

        generator = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            DogDataset(train_items, classes, self.image_size, build_train_augmentation(cfg.seed)),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=generator,
        )
        valid_loader = DataLoader(
            DogDataset(valid_items, classes, self.image_size), batch_size=cfg.batch_size
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        checkpoint = self.active_checkpoint
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        best_acc, epochs_without_improvement = -1.0, 0

        for epoch in range(1, cfg.epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, valid_loader, criterion)
            scheduler.step()
            history.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "seconds": time.time() - start,
            })
            logger.info(
                "[%s] epoch %d/%d - train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.0fs",
                name, epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
                time.time() - start,
            )

            if val_acc > best_acc:
                best_acc, epochs_without_improvement = val_acc, 0
                torch.save(model, checkpoint)  # modelo completo, como espera load_model
                logger.info("New best val acc %.4f -> saved %s", val_acc, checkpoint)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._save_history(name, history, best_acc)
        self._loaded.pop(name, None)  # que load_model levante el checkpoint recien entrenado

    def _build_model(self, name: str, classes: list[str]) -> BreedClassifier:
        """Arma el modelo a entrenar segun el nombre logico (Modelo A o Modelo B)."""
        cfg = self.train_config
        if name == "cnn_custom":
            return build_cnn_custom_classifier(classes, cfg.embedding_dim, cfg.dropout)

        model = build_resnet18_classifier(classes, cfg.dropout)
        for param in model.backbone.parameters():
            param.requires_grad = False
        if cfg.resnet_finetune == "layer4":
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
        elif cfg.resnet_finetune != "head":
            raise ValueError(f"RESNET_FINETUNE invalido: {cfg.resnet_finetune} (head | layer4)")
        return model

    def _save_history(self, name: str, history: list[dict[str, float]], best_acc: float) -> None:
        """Historial por epoca en output/ para graficar las curvas en 2-train.ipynb."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"history_{name}.json"
        payload = {
            "model": name,
            "best_val_acc": best_acc,
            "config": {k: str(v) for k, v in asdict(self.train_config).items()},
            "history": history,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Training history saved to %s", path)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        raise NotImplementedError("Etapa 2: implementar evaluate_classifier")

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        raise NotImplementedError("Etapa 2: implementar extract_custom_embedding")
