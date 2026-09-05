"""
Загрузка подложки из XYZ-тайлов и сшивка в мозаику.

Зачем это сервису. Кадр БПЛА приходит от волонтёра или сотрудника ООПТ, и его
надо сначала где-то взять. Тайловый источник снимает этот шаг: клиент называет
прямоугольник на карте и зум, сервис сам забирает нужные тайлы, склеивает и
считает по ним. Привязка при этом получается не приблизительная, а точная —
координаты тайловой сетки заданы формулой.

Главное ограничение, и его надо понимать до, а не после:

    **Публичные спутниковые подложки для этой модели слишком грубые.**

Модель «Чистого берега» обучена на снимках БПЛА с разрешением в единицы
сантиметров на пиксель. Лучшие открытые подложки дают 0.3–0.6 м/px на зуме 18–19:
один пиксель — это полметра берега, и пластиковой бутылки там просто нет. На
такой подложке ловятся только крупные объекты — бетонные плиты, кучи в десятки
метров, ржавые баржи. Поэтому сервис считает GSD мозаики и, если он грубее
`COARSE_IMAGERY_GSD_M`, возвращает предупреждение в `fraud_flags` вместо того,
чтобы делать вид, что всё в порядке.

Реальный сценарий с высоким разрешением — не публичная подложка, а **своя
ортофотоплан-мозаика ООПТ, выложенная как XYZ** (QGIS, GeoServer, mbtiles).
Она даёт те же сантиметры, что и исходные кадры, и с ней ограничения нет.
Источник задаётся конфигом, поэтому такая мозаика подключается без правки кода.

Про лицензии. Тайлы чужие. Условия у каждого провайдера свои, и часть из них
прямо запрещает машинную выкачку; здесь по умолчанию только источники с
публично разрешённым доступом, а `attribution` возвращается в ответе, чтобы его
было чем показать на карте. Google и Яндекс не добавлены намеренно.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from app.georef import EARTH_RADIUS_M, WebMercatorGeoreference

#: Полный обхват Web Mercator по любой оси, м.
WORLD_SIZE_M = 2 * math.pi * EARTH_RADIUS_M

#: Предел широты в Web Mercator: полюса в проекции уходят в бесконечность.
MAX_LATITUDE = 85.05112878

#: Грубее этого GSD искать мусор моделью бессмысленно, м/px. Порог взят по
#: нижней границе: объект должен занимать хотя бы несколько пикселей, а самый
#: мелкий класс таксономии («пластик») — это десятки сантиметров.
COARSE_IMAGERY_GSD_M = 0.15


@dataclass(frozen=True)
class TileSource:
    """Тайловый источник. Шаблон в формате XYZ: `{z}`, `{x}`, `{y}`."""

    name: str
    url_template: str
    attribution: str
    max_zoom: int = 19
    tile_size: int = 256
    #: Часть сервисов (ArcGIS) отдаёт `/{z}/{y}/{x}` — порядок задаётся шаблоном,
    #: а не флагом: так видно, что именно уходит в запрос.
    api_key_env: str | None = None

    def url(self, x: int, y: int, z: int) -> str:
        url = self.url_template.format(x=x, y=y, z=z)
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            url = url.replace("{key}", key)
        return url


#: Источники по умолчанию. Свою ортофото-мозаику ООПТ добавляют через
#: переменную `TILE_SOURCES` — правка кода для этого не нужна.
DEFAULT_SOURCES: dict[str, TileSource] = {
    "esri": TileSource(
        name="esri",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        max_zoom=19,
    ),
}


class TileError(RuntimeError):
    """Тайл не удалось получить. Отдельный тип: наружу это 502, а не 500."""


# ---------------------------------------------------------------------------
# Математика тайловой сетки
# ---------------------------------------------------------------------------


def resolution_m_per_px(zoom: int, tile_size: int) -> float:
    """Размер пикселя в проекции (не на местности), м."""
    return WORLD_SIZE_M / (tile_size * 2**zoom)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Долгота/широта -> дробные координаты тайла."""
    lat = max(min(lat, MAX_LATITUDE), -MAX_LATITUDE)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_mercator(x: float, y: float, zoom: int, tile_size: int) -> tuple[float, float]:
    """Координаты тайла -> метры EPSG:3857 (левый верхний угол тайла)."""
    res = resolution_m_per_px(zoom, tile_size)
    return (
        x * tile_size * res - WORLD_SIZE_M / 2,
        WORLD_SIZE_M / 2 - y * tile_size * res,
    )


@dataclass(frozen=True)
class TileRange:
    zoom: int
    x0: int
    y0: int
    x1: int  # включительно
    y1: int  # включительно

    @property
    def count(self) -> int:
        return (self.x1 - self.x0 + 1) * (self.y1 - self.y0 + 1)

    def coords(self):
        for y in range(self.y0, self.y1 + 1):
            for x in range(self.x0, self.x1 + 1):
                yield x, y


def tile_range(bbox: tuple[float, float, float, float], zoom: int) -> TileRange:
    """bbox (min_lon, min_lat, max_lon, max_lat) -> диапазон тайлов, покрывающий его."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x_left, y_top = lonlat_to_tile(min_lon, max_lat, zoom)
    x_right, y_bottom = lonlat_to_tile(max_lon, min_lat, zoom)
    n = 2**zoom
    return TileRange(
        zoom=zoom,
        x0=max(int(math.floor(x_left)), 0),
        y0=max(int(math.floor(y_top)), 0),
        x1=min(int(math.floor(x_right)), n - 1),
        y1=min(int(math.floor(y_bottom)), n - 1),
    )


def zoom_for_gsd(target_gsd_m: float, lat: float, tile_size: int = 256) -> int:
    """Минимальный зум, дающий пиксель не крупнее заданного. Полезно клиенту."""
    scale = math.cos(math.radians(max(min(lat, MAX_LATITUDE), -MAX_LATITUDE)))
    for zoom in range(0, 24):
        if resolution_m_per_px(zoom, tile_size) * scale <= target_gsd_m:
            return zoom
    return 23


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------


class TileFetcher:
    """Качает тайлы параллельно и кеширует их на диске.

    Кеш обязателен, а не приятен: соседние запросы по одной ООПТ перекрываются,
    подложка не меняется месяцами, а выкачивать её заново — это и время, и
    нагрузка на чужой сервис, которую нам никто не разрешал создавать.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        workers: int = 8,
        timeout: float = 20.0,
        # Только ASCII: HTTP-заголовки кодируются latin-1, кириллица в
        # User-Agent роняет запрос ещё до сети.
        user_agent: str = "eco-project-ml/0.1 (protected-areas cleanup platform)",
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.workers = workers
        self.timeout = timeout
        self.user_agent = user_agent
        self._lock = threading.Lock()

    def _cache_path(self, source: TileSource, x: int, y: int, z: int) -> Path | None:
        if self.cache_dir is None:
            return None
        # Имя источника в путь не кладём сырым: оно приходит из конфига.
        digest = hashlib.sha256(source.url_template.encode()).hexdigest()[:12]
        return self.cache_dir / digest / str(z) / str(x) / f"{y}.img"

    def fetch_one(self, source: TileSource, x: int, y: int, z: int) -> bytes:
        path = self._cache_path(source, x, y, z)
        if path is not None and path.exists():
            return path.read_bytes()

        request = Request(source.url(x, y, z), headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = response.read()
        except HTTPError as exc:
            raise TileError(f"тайл {z}/{x}/{y}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TileError(f"тайл {z}/{x}/{y}: {exc}") from exc

        if path is not None:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(path)
        return payload

    def fetch_range(self, source: TileSource, rng: TileRange) -> dict[tuple[int, int], bytes]:
        """Все тайлы диапазона. Параллельно, потому что это сеть, а не счёт."""
        results: dict[tuple[int, int], bytes] = {}
        errors: list[str] = []

        def job(coord: tuple[int, int]) -> None:
            x, y = coord
            try:
                results[coord] = self.fetch_one(source, x, y, rng.zoom)
            except TileError as exc:
                errors.append(str(exc))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(job, rng.coords()))

        if not results:
            raise TileError(f"не удалось получить ни одного тайла: {errors[:3]}")
        return results


# ---------------------------------------------------------------------------
# Мозаика
# ---------------------------------------------------------------------------


#: Ниже стольких уникальных цветов мозаика считается заглушкой провайдера.
#: Замер по факту: настоящий снимок берега на зуме 18 даёт десятки тысяч
#: оттенков, плитка «Map data not yet available» — около сотни.
PLACEHOLDER_MAX_COLOURS = 2048


@dataclass(frozen=True)
class Mosaic:
    image: Image.Image
    georeference: WebMercatorGeoreference
    zoom: int
    tiles_total: int
    tiles_missing: int
    source: TileSource

    @property
    def gsd_m_per_px(self) -> float:
        return self.georeference.gsd_m_per_px()

    @property
    def too_coarse(self) -> bool:
        return self.gsd_m_per_px > COARSE_IMAGERY_GSD_M

    @property
    def looks_like_placeholder(self) -> bool:
        """Провайдер отдал заглушку «съёмки нет», а не снимок.

        Проверять обязательно: заглушка приходит с кодом 200 и обычной
        картинкой, поэтому по HTTP её не отличить. Запустить модель по ней —
        значит получить пустой ответ и решить, что берег чистый, хотя на самом
        деле мы просто смотрели в серый прямоугольник. Так бывает часто: на
        Балтике у Esri зум 19 не покрыт, на зуме 18 — покрыт.
        """
        return count_colours(self.image) <= PLACEHOLDER_MAX_COLOURS


def count_colours(image: Image.Image, limit: int = 1 << 16) -> int:
    """Сколько разных цветов в картинке. `limit` — потолок, дальше не считаем."""
    colours = image.convert("RGB").getcolors(maxcolors=limit)
    return limit if colours is None else len(colours)


def build_mosaic(
    source: TileSource,
    bbox: tuple[float, float, float, float],
    zoom: int,
    fetcher: TileFetcher,
    *,
    crop_to_bbox: bool = True,
) -> Mosaic:
    """Забирает тайлы и склеивает их в один кадр с точной привязкой.

    Недостающие тайлы (провайдер отдал 404 на краю покрытия) оставляют чёрные
    дыры, а не роняют запрос: край съёмки — обычное дело, а модель на чёрном
    прямоугольнике ничего не находит.
    """
    rng = tile_range(bbox, zoom)
    payloads = fetcher.fetch_range(source, rng)

    tile_px = source.tile_size
    width = (rng.x1 - rng.x0 + 1) * tile_px
    height = (rng.y1 - rng.y0 + 1) * tile_px
    canvas = Image.new("RGB", (width, height), (0, 0, 0))

    missing = 0
    for (x, y), payload in payloads.items():
        try:
            tile = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            missing += 1
            continue
        if tile.size != (tile_px, tile_px):
            tile = tile.resize((tile_px, tile_px), Image.Resampling.BILINEAR)
        canvas.paste(tile, ((x - rng.x0) * tile_px, (y - rng.y0) * tile_px))
    missing += rng.count - len(payloads)

    res = resolution_m_per_px(zoom, tile_px)
    origin_x, origin_y = tile_to_mercator(rng.x0, rng.y0, zoom, tile_px)

    if crop_to_bbox:
        # Тайловая сетка почти всегда шире запрошенного прямоугольника.
        # Лишнее — это чужие метры, лишний инференс и лишние находки за
        # пределами участка, поэтому мозаика режется по bbox.
        min_lon, min_lat, max_lon, max_lat = bbox
        fx0, fy0 = lonlat_to_tile(min_lon, max_lat, zoom)
        fx1, fy1 = lonlat_to_tile(max_lon, min_lat, zoom)
        left = int(round((fx0 - rng.x0) * tile_px))
        top = int(round((fy0 - rng.y0) * tile_px))
        right = int(round((fx1 - rng.x0) * tile_px))
        bottom = int(round((fy1 - rng.y0) * tile_px))
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, width), min(bottom, height)
        if right - left >= 16 and bottom - top >= 16:
            canvas = canvas.crop((left, top, right, bottom))
            origin_x += left * res
            origin_y -= top * res

    georeference = WebMercatorGeoreference(
        width_px=canvas.width,
        height_px=canvas.height,
        origin_x_m=origin_x,
        origin_y_m=origin_y,
        resolution_m_per_px=res,
        approximate=False,
        source=f"tiles:{source.name}",
    )
    return Mosaic(
        image=canvas,
        georeference=georeference,
        zoom=zoom,
        tiles_total=rng.count,
        tiles_missing=missing,
        source=source,
    )
