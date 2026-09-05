"""
Контракт детектора: кадр -> карта меток той же формы + карта уверенности.

Ровно один интерфейс на все реализации. Благодаря этому HTTP-слой, замеры,
полигоны и привязка к местности не знают, кто именно сегментировал: настоящий
SegFormer «Чистого берега» или правиловый бэйзлайн, работающий без весов и
без GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class Segmentation:
    """Результат сегментации кадра."""

    #: (H, W) uint8, значения — id классов из `app.taxonomy`, 0 = фон.
    labels: np.ndarray
    #: (H, W) float32 0..1 — уверенность в выбранной метке для каждого пикселя.
    probs: np.ndarray


@dataclass(frozen=True)
class BackendInfo:
    """Паспорт реализации. Уходит в ответ: клиент обязан видеть, кто считал."""

    name: str
    version: str
    weights: str | None
    device: str
    tile_px: int
    stride_px: int
    trained: bool
    notes: str


class Detector(Protocol):
    """Всё, что умеет сегментировать мусор на кадре."""

    def info(self) -> BackendInfo: ...

    def segment(self, image: PILImage) -> Segmentation: ...
