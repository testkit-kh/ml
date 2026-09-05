"""
Тайловая подложка: математика сетки, сшивка мозаики, привязка.

Сеть здесь не трогается: тайлы подменяются заглушкой. Тест, ходящий в чужой
сервис, ломается не тогда, когда сломали мы.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image

from app import tiles
from app.georef import haversine_m

CURONIAN = (20.9950, 55.3020, 21.0030, 55.3070)

#: Крошечный участок для тестов на больших зумах: на 22-м зуме даже сотня
#: метров берега — это мозаика в десятки тысяч пикселей.
TINY = (20.9950, 55.3020, 20.9952, 55.3021)


def _tile_bytes(colour: tuple[int, int, int], size: int = 256, noisy: bool = True) -> bytes:
    rng = np.random.default_rng(abs(hash(colour)) % 2**31)
    arr = np.full((size, size, 3), colour, dtype=np.int16)
    if noisy:
        arr = arr + rng.integers(-40, 40, arr.shape)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeFetcher:
    """Отдаёт цветной шум вместо снимка. Считает, сколько тайлов запросили."""

    def __init__(self, *, fail: set[tuple[int, int]] | None = None, flat: bool = False):
        self.requested: list[tuple[int, int, int]] = []
        self.fail = fail or set()
        self.flat = flat

    def fetch_range(self, source, rng):
        result = {}
        for x, y in rng.coords():
            self.requested.append((x, y, rng.zoom))
            if (x, y) in self.fail:
                continue
            result[(x, y)] = _tile_bytes((60 + x % 100, 90, 110), source.tile_size, not self.flat)
        if not result:
            raise tiles.TileError("все тайлы недоступны")
        return result


@pytest.fixture
def source() -> tiles.TileSource:
    return tiles.TileSource(
        name="test", url_template="https://example.invalid/{z}/{x}/{y}", attribution="test"
    )


# --- математика сетки -------------------------------------------------------


def test_tile_of_null_island_is_the_middle_of_the_world():
    x, y = tiles.lonlat_to_tile(0.0, 0.0, 1)

    assert (x, y) == pytest.approx((1.0, 1.0))


def test_zoom_increases_tile_count_fourfold():
    low = tiles.tile_range(CURONIAN, 16).count
    high = tiles.tile_range(CURONIAN, 18).count

    assert high > low


def test_resolution_halves_with_each_zoom_level():
    assert tiles.resolution_m_per_px(18, 256) == pytest.approx(
        tiles.resolution_m_per_px(17, 256) / 2
    )


def test_ground_resolution_shrinks_towards_the_poles():
    """Меркатор растягивает высокие широты: пиксель на ЗФИ мельче, чем в Сочи."""
    res = tiles.resolution_m_per_px(18, 256)
    sochi = res * math.cos(math.radians(43.6))
    franz_josef = res * math.cos(math.radians(80.7))

    assert franz_josef < sochi


@pytest.mark.parametrize(("target", "expected_max"), [(0.05, 0.05), (0.3, 0.3), (1.0, 1.0)])
def test_zoom_for_gsd_returns_a_zoom_that_meets_the_target(target: float, expected_max: float):
    lat = 55.3
    zoom = tiles.zoom_for_gsd(target, lat)
    actual = tiles.resolution_m_per_px(zoom, 256) * math.cos(math.radians(lat))

    assert actual <= expected_max


def test_bbox_is_fully_covered_by_the_tile_range():
    zoom = 18
    rng = tiles.tile_range(CURONIAN, zoom)
    min_lon, min_lat, max_lon, max_lat = CURONIAN
    x_left, y_top = tiles.lonlat_to_tile(min_lon, max_lat, zoom)
    x_right, y_bottom = tiles.lonlat_to_tile(max_lon, min_lat, zoom)

    assert rng.x0 <= x_left and rng.x1 + 1 >= x_right
    assert rng.y0 <= y_top and rng.y1 + 1 >= y_bottom


# --- мозаика ----------------------------------------------------------------


def test_mosaic_is_cropped_to_the_requested_bbox(source):
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())
    west, south, east, north = CURONIAN
    (tl_lon, tl_lat), (_, _), (br_lon, br_lat), _ = [
        tuple(c) for c in mosaic.georeference.bounds_lonlat()
    ]

    # Обрезка по bbox: углы мозаики совпадают с запросом с точностью до пикселя.
    assert tl_lon == pytest.approx(west, abs=1e-4)
    assert br_lon == pytest.approx(east, abs=1e-4)
    assert tl_lat == pytest.approx(north, abs=1e-4)
    assert br_lat == pytest.approx(south, abs=1e-4)


def test_mosaic_width_matches_ground_distance(source):
    """Ширина мозаики в пикселях, умноженная на GSD, — это ширина участка."""
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())
    west, south, east, north = CURONIAN
    ground_m = haversine_m(north, west, north, east)

    assert mosaic.image.width * mosaic.gsd_m_per_px == pytest.approx(ground_m, rel=0.02)


def test_georeference_is_exact_not_approximate(source):
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())

    assert mosaic.georeference.approximate is False
    assert mosaic.georeference.source == "tiles:test"


def test_missing_tiles_leave_holes_but_do_not_fail(source):
    """Край покрытия провайдера — обычное дело, а не повод отдать пятисотку."""
    rng = tiles.tile_range(CURONIAN, 18)
    fetcher = FakeFetcher(fail={(rng.x0, rng.y0)})
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, fetcher)

    assert mosaic.tiles_missing == 1
    assert mosaic.tiles_total == rng.count


def test_all_tiles_missing_is_an_error(source):
    rng = tiles.tile_range(CURONIAN, 18)
    fetcher = FakeFetcher(fail=set(rng.coords()))

    with pytest.raises(tiles.TileError):
        tiles.build_mosaic(source, CURONIAN, 18, fetcher)


def test_rescaled_georeference_keeps_the_same_ground_footprint(source):
    """Конвейер ужимает большие мозаики — привязка обязана ехать вместе с ними."""
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())
    full = mosaic.georeference
    half = full.rescaled(0.5)

    assert half.width_px == full.width_px // 2
    assert half.to_lonlat(0, 0) == full.to_lonlat(0, 0)
    # Допуск 1e-5 градуса (~1 м): ширина ужатой мозаики округляется вниз до
    # целого пикселя, и правый край уезжает на полпикселя. Для полигонов на
    # карте это ниже точности самой подложки.
    assert half.to_lonlat(half.width_px, half.height_px) == pytest.approx(
        full.to_lonlat(full.width_px, full.height_px), abs=1e-5
    )


# --- предупреждения о качестве подложки --------------------------------------


def test_satellite_basemap_is_flagged_as_too_coarse(source):
    """Ключевое ограничение: подложка 0.3 м/px для этой модели слишком груба."""
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())

    assert mosaic.gsd_m_per_px > tiles.COARSE_IMAGERY_GSD_M
    assert mosaic.too_coarse is True


def test_centimetre_imagery_is_not_flagged(source):
    """Своя ортофото ООПТ на большом зуме проходит порог — ради неё всё и делалось."""
    high_res = tiles.TileSource(
        name="ortho",
        url_template="https://example.invalid/{z}/{x}/{y}",
        attribution="ООПТ",
        max_zoom=23,
    )
    mosaic = tiles.build_mosaic(high_res, TINY, 22, FakeFetcher())

    assert mosaic.gsd_m_per_px < tiles.COARSE_IMAGERY_GSD_M
    assert mosaic.too_coarse is False


def test_provider_placeholder_is_detected(source):
    """Заглушка «съёмки нет» приходит с кодом 200 — по HTTP её не отличить."""
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher(flat=True))

    assert mosaic.looks_like_placeholder is True


def test_real_imagery_is_not_mistaken_for_a_placeholder(source):
    mosaic = tiles.build_mosaic(source, CURONIAN, 18, FakeFetcher())

    assert mosaic.looks_like_placeholder is False
