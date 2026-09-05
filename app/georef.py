"""
Привязка кадра к местности: пиксель → широта/долгота.

Три источника, по убыванию доверия:
1. `bounds` — четыре угла кадра, заданные клиентом (ортофото из QGIS/Agisoft).
   Точнее всего: геометрия уже посчитана фотограмметрией, мы её не пересчитываем.
2. EXIF GPS + GSD — одиночный кадр БПЛА. Считаем север-ориентированный
   прямоугольник вокруг точки съёмки. Крен и тангаж не учитываются, поэтому
   результат помечается `approximate=True`.
3. Ничего — работаем в пикселях, физических величин и геометрии не отдаём.

GSD (ground sample distance, м/px) можно не передавать, а вывести из высоты
полёта и параметров камеры.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL.ExifTags import GPSTAGS, TAGS
from PIL.Image import Image as PILImage

EARTH_RADIUS_M = 6_378_137.0

#: Угол кадра как (широта, долгота) — порядок «человеческий», не GeoJSON.
Corner = tuple[float, float]


@dataclass(frozen=True)
class ExifInfo:
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    captured_at: str | None = None
    focal_mm: float | None = None
    make: str | None = None
    model: str | None = None


def _ratio(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _dms_to_deg(dms: object, ref: object) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, str) and ref.upper() in {"S", "W"}:
        deg = -deg
    return deg


def read_exif(image: PILImage) -> ExifInfo:
    """EXIF без сторонних зависимостей. Отсутствие тегов — не ошибка."""
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 — битый EXIF не должен ронять инференс
        return ExifInfo()
    if not exif:
        return ExifInfo()

    flat = {TAGS.get(k, k): v for k, v in exif.items()}
    gps_raw = exif.get_ifd(0x8825) or {}
    gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    lat = _dms_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = _dms_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    alt = _ratio(gps.get("GPSAltitude"))
    if alt is not None and gps.get("GPSAltitudeRef") in (1, b"\x01"):
        alt = -alt

    return ExifInfo(
        lat=lat,
        lon=lon,
        altitude_m=alt,
        captured_at=flat.get("DateTimeOriginal") or flat.get("DateTime"),
        focal_mm=_ratio(flat.get("FocalLength")),
        make=str(flat["Make"]).strip() if flat.get("Make") else None,
        model=str(flat["Model"]).strip() if flat.get("Model") else None,
    )


def gsd_from_camera(
    *,
    altitude_m: float,
    focal_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """Размер пикселя на местности, м/px. Классическая формула аэросъёмки."""
    if focal_mm <= 0 or image_width_px <= 0:
        raise ValueError("focal_mm и image_width_px должны быть положительными")
    return altitude_m * sensor_width_mm / (focal_mm * image_width_px)


@dataclass(frozen=True)
class Georeference:
    """Билинейная привязка кадра по четырём углам.

    Углы — в порядке обхода: верх-лево, верх-право, низ-право, низ-лево,
    каждый как (lat, lon). Билинейная интерполяция вместо аффинной, чтобы
    поддержать слегка непрямоугольные footprint'ы без матричной алгебры.
    """

    width_px: int
    height_px: int
    corners: tuple[Corner, Corner, Corner, Corner]
    approximate: bool = False
    source: str = "bounds"

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Пиксель → (lon, lat) — порядок GeoJSON, а не «широта-долгота»."""
        u = x / max(self.width_px - 1, 1)
        v = y / max(self.height_px - 1, 1)
        tl, tr, br, bl = self.corners
        top_lat = tl[0] + (tr[0] - tl[0]) * u
        top_lon = tl[1] + (tr[1] - tl[1]) * u
        bot_lat = bl[0] + (br[0] - bl[0]) * u
        bot_lon = bl[1] + (br[1] - bl[1]) * u
        lat = top_lat + (bot_lat - top_lat) * v
        lon = top_lon + (bot_lon - top_lon) * v
        return (round(lon, 7), round(lat, 7))

    def polygon(self, points_px: list[tuple[float, float]]) -> list[list[float]]:
        return [list(self.to_lonlat(x, y)) for x, y in points_px]

    def bounds_lonlat(self) -> list[list[float]]:
        """Углы для `image`-источника MapLibre: TL, TR, BR, BL в [lon, lat]."""
        return [[round(lon, 7), round(lat, 7)] for lat, lon in self.corners]

    def gsd_m_per_px(self) -> float:
        """Размер пикселя на местности: средняя длина стороны кадра на её пиксели."""
        tl, tr, br, bl = self.corners
        per_px_x = haversine_m(*tl, *tr) / max(self.width_px - 1, 1)
        per_px_y = haversine_m(*tl, *bl) / max(self.height_px - 1, 1)
        return (per_px_x + per_px_y) / 2.0


def georeference_from_center(
    *,
    lat: float,
    lon: float,
    gsd_m_per_px: float,
    width_px: int,
    height_px: int,
    heading_deg: float = 0.0,
    source: str = "exif",
) -> Georeference:
    """Север-ориентированный (или повёрнутый на `heading_deg`) footprint.

    Плоская аппроксимация: на кадре БПЛА в сотни метров кривизна Земли ниже
    ошибки самого GPS, поэтому метры переводятся в градусы линейно.
    """
    half_w_m = width_px * gsd_m_per_px / 2.0
    half_h_m = height_px * gsd_m_per_px / 2.0
    theta = math.radians(heading_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    deg_per_m_lat = 180.0 / (math.pi * EARTH_RADIUS_M)
    deg_per_m_lon = deg_per_m_lat / max(math.cos(math.radians(lat)), 1e-6)

    offsets = (
        (-half_w_m, -half_h_m),
        (half_w_m, -half_h_m),
        (half_w_m, half_h_m),
        (-half_w_m, half_h_m),
    )
    corners: list[Corner] = []
    for dx, dy in offsets:
        east = dx * cos_t + dy * sin_t
        north = -(dy * cos_t - dx * sin_t)
        corners.append((lat + north * deg_per_m_lat, lon + east * deg_per_m_lon))

    return Georeference(
        width_px=width_px,
        height_px=height_px,
        corners=(corners[0], corners[1], corners[2], corners[3]),
        approximate=True,
        source=source,
    )


@dataclass(frozen=True)
class WebMercatorGeoreference:
    """Привязка мозаики XYZ-тайлов. Точная, а не интерполированная.

    У тайловой сетки координаты выводятся формулой, а не натягиваются по углам:
    пиксель -> метры Web Mercator -> широта/долгота. Это важно именно здесь —
    Меркатор нелинеен по широте, и билинейная интерполяция углов на высоких
    широтах (а это Арктика, ЗФИ, Кольский) даёт ошибку в метры.

    `origin_x_m`, `origin_y_m` — левый верхний угол мозаики в EPSG:3857.
    `resolution_m_per_px` — размер пикселя в проекции; на местности он в
    cos(широта) раз меньше, и это возвращает `gsd_m_per_px()`.
    """

    width_px: int
    height_px: int
    origin_x_m: float
    origin_y_m: float
    resolution_m_per_px: float
    approximate: bool = False
    source: str = "tiles"

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        mx = self.origin_x_m + x * self.resolution_m_per_px
        my = self.origin_y_m - y * self.resolution_m_per_px
        lon = math.degrees(mx / EARTH_RADIUS_M)
        lat = math.degrees(2 * math.atan(math.exp(my / EARTH_RADIUS_M)) - math.pi / 2)
        return (round(lon, 7), round(lat, 7))

    def polygon(self, points_px: list[tuple[float, float]]) -> list[list[float]]:
        return [list(self.to_lonlat(x, y)) for x, y in points_px]

    @property
    def corners(self) -> tuple[Corner, Corner, Corner, Corner]:
        w, h = self.width_px, self.height_px
        return tuple(  # type: ignore[return-value]
            (lat, lon)
            for lon, lat in (
                self.to_lonlat(0, 0),
                self.to_lonlat(w, 0),
                self.to_lonlat(w, h),
                self.to_lonlat(0, h),
            )
        )

    def bounds_lonlat(self) -> list[list[float]]:
        return [[round(lon, 7), round(lat, 7)] for lat, lon in self.corners]

    def rescaled(self, scale: float) -> WebMercatorGeoreference:
        """Та же привязка для ужатой копии кадра.

        Конвейер ужимает большие мозаики перед инференсом; привязка обязана
        поехать вместе с ними, иначе полигоны лягут не туда. Начало координат
        не меняется — меняется цена пикселя.
        """
        if scale == 1.0:
            return self
        return WebMercatorGeoreference(
            width_px=max(int(self.width_px * scale), 1),
            height_px=max(int(self.height_px * scale), 1),
            origin_x_m=self.origin_x_m,
            origin_y_m=self.origin_y_m,
            resolution_m_per_px=self.resolution_m_per_px / scale,
            approximate=self.approximate,
            source=self.source,
        )

    def gsd_m_per_px(self) -> float:
        """Размер пикселя на местности в центре мозаики.

        Масштаб Меркатора растёт как 1/cos(широта): на экваторе пиксель зума 18
        это 0.6 м, на широте Мурманска — уже 0.26 м. Берём широту центра —
        по кадру в сотни метров разброс пренебрежим.
        """
        _, centre_lat = self.to_lonlat(self.width_px / 2, self.height_px / 2)
        return self.resolution_m_per_px * math.cos(math.radians(centre_lat))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между точками, м. Нужно антифроду: EXIF GPS vs присланные."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
