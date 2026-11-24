"""Envoltorio común para los dos clasificadores de razas (Modelo A y Modelo B).

backbone (imagen -> embedding) + head (embedding -> una salida por raza).
La lista de razas viaja dentro del modelo, así la app traduce índice -> raza
sin depender de que el dataset esté en el contenedor.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from lib.models.cnn_custom import CustomCNN

RESNET18_EMBEDDING_DIM = 512


class BreedClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        classes: list[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, len(classes)))
        self.classes = list(classes)
        self.embedding_dim = embedding_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding de la capa previa al clasificador (para la búsqueda por similitud)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def build_resnet18_classifier(classes: list[str], dropout: float = 0.0) -> BreedClassifier:
    """Modelo A: ResNet18 preentrenada en ImageNet, con fc reemplazada por la cabeza de razas."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    return BreedClassifier(backbone, RESNET18_EMBEDDING_DIM, classes, dropout)


def build_cnn_custom_classifier(
    classes: list[str], embedding_dim: int, dropout: float = 0.0
) -> BreedClassifier:
    """Modelo B: CNN propia desde cero, con embedding de `embedding_dim` dimensiones."""
    return BreedClassifier(CustomCNN(embedding_dim), embedding_dim, classes, dropout)
