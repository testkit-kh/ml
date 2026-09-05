"""
Ручка детекции по участку карты. Тайлы подменяются: сеть в тестах не трогаем.
"""

from __future__ import annotations

import pytest

from app import tiles
from tests.test_tiles import CURONIAN, TINY, FakeFetcher

BBOX = list(CURONIAN)


@pytest.fixture(autouse=True)
def fake_tiles(monkeypatch):
    """Подменяем загрузчик, оставляя настоящую сшивку, привязку и пороги."""
    from app import router

    monkeypatch.setattr(router, "_fetcher", lambda: FakeFetcher())
    router.available_sources.cache_clear()
    yield
    router.available_sources.cache_clear()


def test_sources_are_listed_with_attribution(client):
    sources = client.get("/api/v1/tiles/sources").json()

    assert sources
    assert all(s["attribution"] for s in sources), "подложку нельзя показывать без копирайта"


def test_area_request_returns_the_usual_contract(client):
    body = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18}).json()

    assert body["image"]["georeference_source"] == "tiles"
    assert body["image"]["georeference_approximate"] is False
    assert body["imagery"]["tiles_total"] > 0
    assert body["overlay"]["bounds"] is not None


def test_area_georeference_matches_the_requested_bbox(client):
    body = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18}).json()
    (tl_lon, tl_lat), _, (br_lon, br_lat), _ = body["overlay"]["bounds"]

    assert tl_lon == pytest.approx(BBOX[0], abs=1e-4)
    assert br_lat == pytest.approx(BBOX[1], abs=1e-4)
    assert br_lon == pytest.approx(BBOX[2], abs=1e-4)
    assert tl_lat == pytest.approx(BBOX[3], abs=1e-4)


def test_coarse_basemap_suppresses_candidates(client):
    """Главная защита: находки с 0.3-метровой подложки в очередь ООПТ не идут."""
    body = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18}).json()

    assert body["imagery"]["too_coarse"] is True
    assert body["imagery"]["candidates_suppressed"] is True
    assert body["point_candidates"] == []
    assert "imagery_too_coarse" in {f["code"] for f in body["fraud_flags"]}


def test_high_resolution_source_produces_candidates(client, monkeypatch):
    """Своя ортофото ООПТ порог проходит — ради неё тайловый путь и сделан."""
    from app import router

    monkeypatch.setitem(
        router.available_sources(),
        "ortho",
        tiles.TileSource(
            name="ortho",
            url_template="https://example.invalid/{z}/{x}/{y}",
            attribution="ФГБУ, ортофотоплан БПЛА",
            max_zoom=23,
        ),
    )
    body = client.post(
        "/api/v1/imagery/detect/area",
        json={"bbox": list(TINY), "zoom": 22, "source": "ortho", "territory_id": 4},
    ).json()

    assert body["imagery"]["too_coarse"] is False
    assert body["imagery"]["candidates_suppressed"] is False


def test_unknown_source_is_a_404(client):
    response = client.post(
        "/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18, "source": "yandex"}
    )

    assert response.status_code == 404


def test_zoom_above_source_maximum_is_rejected(client):
    response = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 23})

    assert response.status_code == 422


def test_area_too_large_is_rejected_before_any_download(client):
    """Предел тайлов защищает и нас, и чужой сервис — проверяется до сети."""
    response = client.post(
        "/api/v1/imagery/detect/area",
        json={"bbox": [39.0, 43.0, 40.0, 44.0], "zoom": 18},
    )

    assert response.status_code == 422
    assert "тайлов" in str(response.json()["detail"])


@pytest.mark.parametrize(
    "bbox",
    [
        [21.0, 55.30, 20.99, 55.31],  # запад восточнее востока
        [20.99, 55.31, 21.0, 55.30],  # юг севернее севера
        [20.99, 89.0, 21.0, 89.5],  # за пределом Меркатора
    ],
)
def test_broken_bbox_is_rejected(client, bbox):
    response = client.post("/api/v1/imagery/detect/area", json={"bbox": bbox, "zoom": 18})

    assert response.status_code == 422


def test_missing_tiles_are_reported(client, monkeypatch):
    from app import router

    rng = tiles.tile_range(CURONIAN, 18)
    monkeypatch.setattr(router, "_fetcher", lambda: FakeFetcher(fail={(rng.x0, rng.y0)}))
    body = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18}).json()

    assert body["imagery"]["tiles_missing"] == 1
    assert "imagery_incomplete" in {f["code"] for f in body["fraud_flags"]}


def test_provider_placeholder_is_flagged(client, monkeypatch):
    """Пустой результат по заглушке не должен читаться как «берег чистый»."""
    from app import router

    monkeypatch.setattr(router, "_fetcher", lambda: FakeFetcher(flat=True))
    body = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18}).json()

    assert "imagery_unavailable" in {f["code"] for f in body["fraud_flags"]}


def test_tile_failure_becomes_a_bad_gateway(client, monkeypatch):
    from app import router

    rng = tiles.tile_range(CURONIAN, 18)
    monkeypatch.setattr(router, "_fetcher", lambda: FakeFetcher(fail=set(rng.coords())))
    response = client.post("/api/v1/imagery/detect/area", json={"bbox": BBOX, "zoom": 18})

    assert response.status_code == 502
