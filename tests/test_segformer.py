"""
Партнёрская модель. Тесты пропускаются, если нет torch или весов — CI слим-образа
их не имеет, и это штатная ситуация, а не поломка.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="requirements-model.txt не установлен")
pytest.importorskip("transformers", reason="requirements-model.txt не установлен")

WEIGHTS = Path(os.environ.get("MODEL_WEIGHTS", "model.pt"))
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason=f"нет весов {WEIGHTS}: python scripts/download_model.py"
)


@pytest.fixture(scope="module")
def detector():
    from app.backends.segformer import SegformerDetector

    return SegformerDetector(WEIGHTS)


def test_weights_fit_the_declared_architecture(detector):
    """Главный тест бэкенда: `strict=True` в загрузке — и он не упал.

    Если апстрим перевыпустит веса под другую форму сети или transformers
    переименует модули, конструктор бросит исключение, и мы узнаем об этом
    здесь, а не на проде посреди инференса.
    """
    info = detector.info()

    assert info.trained is True
    assert info.name == "segformer"


def test_output_shape_matches_input(detector):
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (256, 320, 3), dtype=np.uint8)
    )
    result = detector.segment(image)

    assert result.labels.shape == (256, 320)
    assert result.probs.shape == (256, 320)


def test_labels_stay_inside_the_taxonomy(detector):
    image = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    labels = detector.segment(image).labels

    assert set(np.unique(labels)) <= {0, 1, 2, 3, 4, 5, 6}


def test_probabilities_are_probabilities(detector):
    image = Image.fromarray(np.full((256, 256, 3), 120, dtype=np.uint8))
    probs = detector.segment(image).probs

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_tiling_covers_frames_larger_than_a_tile():
    """Кадр шире тайла должен разбираться целиком, без дыр и без промахов осей."""
    from app.backends.segformer import STRIDE_PX, TILE_PX, _starts

    starts = _starts(2688, TILE_PX, STRIDE_PX)

    assert starts[0] == 0
    assert starts[-1] + TILE_PX == 2688  # последний тайл прижат к краю
    assert all(b - a <= STRIDE_PX for a, b in zip(starts, starts[1:]))


def test_small_frame_needs_a_single_tile():
    from app.backends.segformer import STRIDE_PX, TILE_PX, _starts

    assert _starts(512, TILE_PX, STRIDE_PX) == [0]
