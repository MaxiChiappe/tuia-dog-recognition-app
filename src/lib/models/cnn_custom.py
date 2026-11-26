"""Modelo B: CNN propia entrenada desde cero."""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool (reduce el tamaño a la mitad)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """5 bloques convolucionales + global average pooling + capa de embedding.

    Con entrada 224x224 los mapas quedan en 7x7 antes del pooling. Devuelve un
    embedding de `embedding_dim` dimensiones (sin la capa de clasificación).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        channels: tuple[int, ...] = (32, 64, 128, 256, 256),
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(_conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.pool(self.features(x)))
