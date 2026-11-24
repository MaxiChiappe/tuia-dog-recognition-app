"""Preprocesamiento compartido entre notebooks y servicios de la app.

Toda imagen que entra a un modelo (embeddings, clasificador, recortes de YOLO)
pasa por acá, para que entrenamiento e inferencia usen exactamente el mismo pipeline.
"""
from __future__ import annotations

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
