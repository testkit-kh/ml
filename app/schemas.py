"""
Контракты сервиса. Это же — контракт с бэкендом и фронтом, поэтому названия
полей совпадают с моделью мусора бэкенда (`trash_category`, `fraction`,
`estimated_volume_m3`), а не изобретают свои.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

LatLon = tuple[float, float]


class CameraParams(BaseModel):
    """Параметры съёмки — если GSD не известен, он выводится отсюда."""

    altitude_m: float = Field(gt=0, description="Высота над поверхностью, м")
    focal_mm: float = Field(gt=0, description="Фокусное расстояние, мм")
    sensor_width_mm: float = Field(gt=0, description="Ширина матрицы, мм")


class DetectOptions(BaseModel):
    """Метаданные кадра. Передаются полем формы `meta` как JSON-строка."""

    gsd_m_per_px: float | None = Field(
        default=None, gt=0, description="Размер пикселя на местности, м/px"
    )
    camera: CameraParams | None = Field(
        default=None, description="Альтернатива gsd_m_per_px: посчитаем сами"
    )
    bounds: list[LatLon] | None = Field(
        default=None,
        description="Четыре угла кадра (lat, lon) по обходу: TL, TR, BR, BL",
    )
    center: LatLon | None = Field(
        default=None, description="Координата центра кадра, если углы неизвестны"
    )
    heading_deg: float = Field(default=0.0, description="Курс камеры, град. от севера")
    reported_at: str | None = Field(
        default=None, description="Когда волонтёр заявил съёмку, ISO-8601 — для антифрода"
    )
    territory_id: int | None = Field(default=None, description="ООПТ, к которой относится кадр")
    reporter_id: int | None = Field(default=None, description="Автор загрузки")
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int | None = Field(default=None, ge=1)
    suppress_candidates: bool = Field(
        default=False,
        description=(
            "Не создавать кандидатов в точки: снимок слишком грубый, "
            "чтобы пускать находки в очередь модератора"
        ),
    )
    render_overlay: bool = Field(default=True, description="Готовить PNG-оверлей")
    overlay_by_class: bool = Field(
        default=True, description="Цвет по классу; иначе один акцентный фиолетовый"
    )
    include_geojson: bool = Field(default=True)

    @model_validator(mode="after")
    def _check_bounds(self) -> DetectOptions:
        if self.bounds is not None and len(self.bounds) != 4:
            raise ValueError("bounds должен содержать ровно 4 угла: TL, TR, BR, BL")
        return self


class AreaRequest(BaseModel):
    """Запрос по участку карты: сервис сам заберёт тайлы подложки.

    Углы кадра здесь не нужны — координаты тайловой сетки заданы формулой,
    поэтому привязка получается точной, а не приблизительной.
    """

    bbox: tuple[float, float, float, float] = Field(
        description="min_lon, min_lat, max_lon, max_lat"
    )
    zoom: int = Field(default=18, ge=1, le=23, description="Зум тайловой сетки")
    source: str | None = Field(default=None, description="Имя источника; None — по умолчанию")

    territory_id: int | None = None
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    min_area_px: int | None = Field(default=None, ge=1)
    render_overlay: bool = True
    overlay_by_class: bool = True
    include_geojson: bool = True

    @model_validator(mode="after")
    def _check_bbox(self) -> AreaRequest:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError("bbox должен быть min_lon < max_lon и min_lat < max_lat")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("долгота вне диапазона")
        if not (-85.05 <= min_lat <= 85.05 and -85.05 <= max_lat <= 85.05):
            raise ValueError("широта вне диапазона Web Mercator")
        return self


class TileSourceOut(BaseModel):
    name: str
    attribution: str = Field(description="Обязателен к показу на карте")
    max_zoom: int
    tile_size: int


class ImageryInfo(BaseModel):
    """Откуда взялась картинка. Есть только у запросов по участку карты."""

    source: str
    attribution: str
    zoom: int
    tiles_total: int
    tiles_missing: int
    gsd_m_per_px: float
    too_coarse: bool = Field(
        description="Пиксель крупнее, чем различимый моделью объект — находкам верить нельзя"
    )
    candidates_suppressed: bool = Field(
        description="Кандидаты не создавались из-за грубой подложки"
    )


class ModelInfo(BaseModel):
    backend: str
    version: str
    weights: str | None
    device: str
    trained: bool = Field(description="False = правиловый бэйзлайн, а не обученная модель")
    notes: str


class ClassInfoOut(BaseModel):
    id: int
    label: str
    label_ru: str
    trash_category: str
    color_hex: str


class ImageInfo(BaseModel):
    width: int
    height: int
    processed_width: int
    processed_height: int
    gsd_m_per_px: float | None
    gsd_source: Literal["provided", "camera", "bounds", "tiles", "unknown"]
    georeferenced: bool
    georeference_source: Literal["bounds", "exif", "center", "tiles", "none"]
    georeference_approximate: bool


class Detection(BaseModel):
    """Один объект мусора. Единица, из которой бэкенд делает кандидата в точку."""

    id: int
    class_id: int
    label: str
    label_ru: str
    trash_category: str
    color_hex: str
    confidence: float

    area_px: int
    area_m2: float | None
    depth_m: float | None = Field(description="Эффективная высота слоя — оценка, не замер")
    volume_m3: float | None
    mass_kg: float | None
    fraction: str | None

    centroid_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int] = Field(description="(x0, y0, x1, y1)")
    polygon_px: list[tuple[float, float]]

    centroid: LatLon | None = Field(default=None, description="(lat, lon) центра объекта")
    geometry: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon, lon/lat")


class ClassSummary(BaseModel):
    class_id: int
    label: str
    trash_category: str
    count: int
    area_px: int
    area_m2: float | None
    volume_m3: float | None
    mass_kg: float | None


class Summary(BaseModel):
    count: int
    by_class: list[ClassSummary]
    total_area_px: int
    total_area_m2: float | None
    total_volume_m3: float | None
    total_mass_kg: float | None
    coverage_ratio: float = Field(description="Доля кадра под мусором, 0..1")
    dominant_category: str | None
    fraction: str | None


class PointCandidate(BaseModel):
    """Готовая полезная нагрузка для `point_created(mode='uav_auto')` бэкенда.

    Бэкенд не должен пересобирать её из детекций сам: сервис знает свою
    таксономию и свои допущения, бэкенд — только принимает и валидирует.
    """

    source: Literal["uav_auto"] = "uav_auto"
    detection_ids: list[int]
    lat: float | None
    lon: float | None
    geometry: dict[str, Any] | None
    trash_categories: list[str]
    dominant_category: str | None
    fraction: str | None
    estimated_volume_m3: float | None
    estimated_mass_kg: float | None
    confidence: float
    territory_id: int | None
    needs_human_validation: Literal[True] = True


class FraudFlag(BaseModel):
    """Сигнал антифрода. Решение принимает человек, сервис только помечает."""

    code: str
    severity: Literal["info", "warning"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ExifOut(BaseModel):
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    captured_at: str | None = None
    focal_mm: float | None = None
    make: str | None = None
    model: str | None = None


class OverlayOut(BaseModel):
    """Всё, что нужно фронту, чтобы положить фиолетовый слой на карту."""

    png_url: str | None
    blend_url: str | None
    bounds: list[list[float]] | None = Field(
        default=None, description="Углы для image-источника MapLibre: TL, TR, BR, BL в [lon, lat]"
    )
    accent_hex: str
    expires_in_seconds: int


class DetectResponse(BaseModel):
    job_id: str
    model: ModelInfo
    image: ImageInfo
    detections: list[Detection]
    summary: Summary
    point_candidates: list[PointCandidate]
    fraud_flags: list[FraudFlag]
    exif: ExifOut
    overlay: OverlayOut
    geojson: dict[str, Any] | None
    assumptions: dict[str, Any]
    timing_ms: dict[str, float]
    imagery: ImageryInfo | None = Field(
        default=None, description="Заполняется только для запросов по участку карты"
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str
    backend_ready: bool
    trained: bool
    version: str


class ModelResponse(BaseModel):
    model: ModelInfo
    classes: list[ClassInfoOut]
    assumptions: dict[str, Any]
