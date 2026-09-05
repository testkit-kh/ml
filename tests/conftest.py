from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Клиент API поверх правилового бэйзлайна.

    Бэкенд фиксируется явно, а не берётся из окружения: тесты ручек проверяют
    контракт и конвейер, а не качество распознавания. С партнёрской моделью на
    синтетических кадрах находок не будет (она обучена на настоящем берегу),
    и падал бы не код, а несоответствие данных модели — это не то, что должен
    ловить тест API. Качество модели проверяется отдельно, в test_segformer.
    """
    from app.config import settings

    settings.MODEL_BACKEND = "heuristic"
    from app.main import app

    return TestClient(app)


@pytest.fixture
def frame() -> tuple[Image.Image, np.ndarray]:
    """Синтетический кадр с эталонной маской — тот же генератор, что и в testdata."""
    from make_testdata import make_frame

    image, truth, _ = make_frame(seed=42, width=480, height=360, objects=4)
    return image, truth


@pytest.fixture(autouse=True)
def _clear_artifacts():
    from app.router import artifacts

    artifacts.clear()
    yield
    artifacts.clear()
