"""
Точка входа ML-сервиса.

Запуск локально:
    uvicorn app.main:app --reload --port 8001

Сервис намеренно stateless: ни БД, ни очередей, ни знания о пользователях.
Он принимает кадр и возвращает разметку; кто её сохранит и кому покажет —
дело бэкенда. Из-за этого его можно масштабировать репликами и выключать,
не роняя платформу: без него остаётся ручной ввод типа мусора волонтёром.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backends import get_detector
from app.config import settings
from app.router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Прогреваем модель на старте: первый запрос не должен ждать веса.

    Отказ загрузки не валит процесс — сервис поднимется и честно ответит
    `backend_ready: false` в /health, а платформа переживёт это деградацией
    до ручного ввода, а не пятисотками.
    """
    try:
        info = get_detector(settings.MODEL_BACKEND).info()
        log.info(
            "backend=%s version=%s device=%s trained=%s",
            info.name,
            info.version,
            info.device,
            info.trained,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("бэкенд %s не загрузился: %s", settings.MODEL_BACKEND, exc)
    yield


app = FastAPI(
    title="Eco-Project ML",
    description=(
        "Детекция мусора на снимках БПЛА. Сегментация моделью проекта "
        "«Чистый берег» (Yandex DataSphere) либо правиловым бэйзлайном; "
        "на выходе — полигоны, оценка объёма и массы, кандидаты в точки ООПТ."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
