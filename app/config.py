"""Конфигурация сервиса. Всё через переменные окружения, дефолты рабочие."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "development"

    #: `heuristic` — без весов и torch; `segformer` — партнёрская модель.
    MODEL_BACKEND: str = "heuristic"
    MODEL_WEIGHTS: str = "model.pt"

    #: Кто может звать сервис. В проде — только бэкенд, не браузер.
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    #: Общий секрет для вызовов бэкенд -> ML. Пусто = проверка выключена (dev).
    API_KEY: str = ""

    #: Предел размера кадра, МБ. Ортофото больше режется клиентом на тайлы.
    MAX_UPLOAD_MB: int = 40

    #: Предел стороны кадра, px: длинная сторона ужимается до него перед
    #: инференсом, координаты потом масштабируются обратно.
    MAX_SIDE_PX: int = 4096

    #: Сколько держать оверлеи в памяти, секунд, и сколько штук максимум.
    ARTIFACT_TTL_SECONDS: int = 900
    ARTIFACT_MAX_ITEMS: int = 64

    #: Порог уверенности: ниже него объект в ответ не попадает.
    MIN_CONFIDENCE: float = 0.0

    #: Минимальная площадь объекта, px.
    MIN_AREA_PX: int = 80

    #: Антифрод: расхождение EXIF GPS и присланных координат больше — флаг.
    FRAUD_GPS_DELTA_M: float = 500.0

    #: Антифрод: съёмка старше стольких дней — флаг.
    FRAUD_MAX_AGE_DAYS: int = 365

    # --- тайловые подложки -------------------------------------------------

    #: Дополнительные источники тайлов, JSON:
    #: {"oopt_ortho": {"url_template": "https://.../{z}/{x}/{y}.png",
    #:                 "attribution": "ФГБУ ...", "max_zoom": 22}}
    #: Именно сюда подключается своя ортофото-мозаика ООПТ — правка кода не нужна.
    TILE_SOURCES: dict[str, dict] = {}

    #: Источник по умолчанию, если клиент не назвал свой.
    TILE_DEFAULT_SOURCE: str = "esri"

    #: Куда складывать скачанные тайлы. Пусто — не кешировать (тесты, CI).
    TILE_CACHE_DIR: str = "tilecache"

    #: Сколько тайлов качать одновременно.
    TILE_WORKERS: int = 8

    #: Потолок тайлов на один запрос. Защита и от случайного «выдели всю
    #: Камчатку», и от того, чтобы не долбить чужой сервис тысячами запросов.
    TILE_MAX_PER_REQUEST: int = 256

    #: Запрещать инференс по слишком грубой подложке. Спутниковые основы дают
    #: 0.2–0.6 м/px, а модель обучена на сантиметрах: находки с такой подложки
    #: в очередь модератора пускать нельзя.
    TILE_BLOCK_COARSE_CANDIDATES: bool = True


settings = Settings()
