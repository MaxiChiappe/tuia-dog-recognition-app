from __future__ import annotations

import copy
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import onnxruntime
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.models import BreedClassifier, build_cnn_custom_classifier, build_resnet18_classifier
from lib.preprocessing import build_train_augmentation, preprocess_bgr
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

        # GPU si esta disponible (ej. Colab); si no, CPU. El checkpoint se guarda siempre en CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = device.type == "cuda"
        model = self._build_model(name, classes).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "Trainable params: %d / %d | device: %s",
            sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters()), device,
        )

        # Con GPU, 2 procesos leen imagenes en paralelo para no frenarla; en CPU no hace falta.
        loader_kwargs = {"num_workers": 2 if use_cuda else 0, "pin_memory": use_cuda}
        generator = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            DogDataset(train_items, classes, self.image_size, build_train_augmentation(cfg.seed)),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=generator,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            DogDataset(valid_items, classes, self.image_size), batch_size=cfg.batch_size,
            **loader_kwargs,
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
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, valid_loader, criterion, device=device)
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
                # modelo completo (como espera load_model) y en CPU, para que la app lo cargue sin GPU
                torch.save(copy.deepcopy(model).to("cpu"), checkpoint)
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
        items = load_split(self.train_config.splits_file, self.dataset_path, "test")
        if not items:
            raise ValueError(f"Split test vacio en {self.train_config.splits_file}")
        return self.evaluate_items(items, tag="test")

    def evaluate_items(self, items: list[tuple[Path, str]], tag: str) -> dict[str, float]:
        """Evalua el modelo activo sobre una lista de (ruta, raza): test o conjunto independiente.

        Precision, recall, F1 y specificity se calculan por raza (one-vs-rest) y se promedian
        (macro) sobre las razas presentes en `items`. Guarda las predicciones en
        output/predictions_<modelo>_<tag>.csv para el analisis de errores en 2-train.ipynb.
        """
        model = self.get_torch_model()
        classes = model.classes
        missing = sorted({breed for _, breed in items if breed not in classes})
        if missing:
            raise ValueError(f"Razas que el modelo no conoce: {missing}")

        loader = DataLoader(
            DogDataset(items, classes, self.image_size), batch_size=self.train_config.batch_size
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        scores: list[float] = []
        with torch.no_grad():
            for images, labels in loader:
                probs = torch.softmax(model(images), dim=1)
                score, pred = probs.max(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(pred.tolist())
                scores.extend(score.tolist())

        metrics = self._classification_metrics(y_true, y_pred, len(classes))
        self._save_predictions(tag, items, classes, y_true, y_pred, scores)
        logger.info("[%s] metrics on %s: %s", self.active_model_name, tag, metrics)
        return metrics

    @staticmethod
    def _classification_metrics(
        y_true: list[int], y_pred: list[int], n_classes: int
    ) -> dict[str, float]:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        total = int(cm.sum())
        precisions, recalls, f1s, specificities = [], [], [], []
        for c in sorted(set(y_true)):  # razas presentes en el conjunto evaluado
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum()) - tp
            fn = int(cm[c, :].sum()) - tp
            tn = total - tp - fp - fn
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            specificities.append(specificity(tn, fp))
        return {
            "accuracy": float(np.trace(cm) / total) if total else 0.0,
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "specificity": float(np.mean(specificities)),
            "f1": float(np.mean(f1s)),
        }

    def _save_predictions(
        self,
        tag: str,
        items: list[tuple[Path, str]],
        classes: list[str],
        y_true: list[int],
        y_pred: list[int],
        scores: list[float],
    ) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"predictions_{self.active_model_name}_{tag}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true", "pred", "score"])
            for (image_path, _), t, p, s in zip(items, y_true, y_pred, scores):
                writer.writerow([image_path.as_posix(), classes[t], classes[p], round(s, 6)])
        logger.info("Predictions saved to %s", path)

    def get_torch_model(self) -> BreedClassifier:
        """Modelo activo (.pth) en modo evaluacion. Lo usan la evaluacion, los embeddings y la
        clasificacion de la Etapa 3."""
        model = self.load_model()
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"{self.active_checkpoint}: se esperaba un checkpoint .pth (modelo PyTorch completo)"
            )
        model.eval()
        return model

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        model = self.get_torch_model()
        tensor = preprocess_bgr(image, self.image_size)  # mismo preprocesamiento que al entrenar
        with torch.no_grad():
            embedding = model.embed(tensor).squeeze(0).numpy()
        if embedding.shape[0] != self.train_config.embedding_dim:
            raise ValueError(
                f"El embedding de {self.active_model_name} tiene {embedding.shape[0]} dimensiones "
                f"y EMBEDDING_DIM es {self.train_config.embedding_dim}"
            )
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(float).tolist()
