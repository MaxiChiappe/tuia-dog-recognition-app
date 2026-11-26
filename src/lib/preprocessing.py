"""Preprocesamiento compartido entre notebooks y servicios de la app.

Toda imagen que entra a un modelo (embeddings, clasificador, recortes de YOLO)
pasa por acá, para que entrenamiento e inferencia usen exactamente el mismo pipeline.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(image: np.ndarray, size: int) -> torch.Tensor:
    """Imagen BGR uint8 (OpenCV) -> tensor float32 (1, 3, size, size) normalizado con ImageNet.

    Resize directo (sin center crop) para no cortar al perro en recortes alargados de YOLO.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def build_train_augmentation(seed: int | None = None) -> A.Compose:
    """Augmentation SOLO para train: se aplica sobre la imagen BGR uint8, antes de preprocess_bgr.

    Variaciones realistas que no cambian la raza aparente (sin flips verticales,
    rotaciones de 90° ni cambios de color fuertes que alteren el pelaje).
    `seed` hace reproducible la augmentation (albumentations usa su propio generador aleatorio).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        ],
        seed=seed,
    )
