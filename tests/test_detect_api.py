"""Ручка детекции целиком: контракт ответа, привязка, оверлей, антифрод."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

CENTER = [55.2833, 20.9167]


def _post(client, image: Image.Image, meta: dict | None = None):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    data = {"meta": json.dumps(meta)} if meta is not None else {}
    return client.post(
        "/api/v1/imagery/detect",
        files={"image": ("frame.png", buffer, "image/png")},
        data=data,
    )


def test_health_reports_backend(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["backend_ready"] is True


def test_model_card_lists_six_classes_and_assumptions(client):
    body = client.get("/api/v1/model").json()

    assert len(body["classes"]) == 6
    assert {c["trash_category"] for c in body["classes"]} == {
        "metal",
        "fishing_gear",
        "plastic",
        "wood",
        "construction",
        "rubber",
    }
    assert "depth_method" in body["assumptions"]


def test_baseline_is_flagged_as_untrained(client, frame):
    """Ответ обязан говорить, что считал бэйзлайн, а не партнёрская модель."""
    body = _post(client, frame[0]).json()

    assert body["model"]["backend"] == "heuristic"
    assert body["model"]["trained"] is False


def test_detects_the_planted_objects(client, frame):
    image, truth = frame
    body = _post(client, image, {"gsd_m_per_px": 0.05, "center": CENTER}).json()

    planted = len(np.unique(truth)) - 1
    assert body["summary"]["count"] >= planted - 1
    assert body["summary"]["coverage_ratio"] > 0


def test_georeferenced_frame_yields_geojson_and_coordinates(client, frame):
    body = _post(client, frame[0], {"gsd_m_per_px": 0.05, "center": CENTER}).json()

    assert body["image"]["georeferenced"] is True
    assert body["geojson"]["type"] == "FeatureCollection"
    for detection in body["detections"]:
        lat, lon = detection["centroid"]
        assert abs(lat - CENTER[0]) < 0.01 and abs(lon - CENTER[1]) < 0.01
        assert detection["geometry"]["type"] == "Polygon"


def test_frame_without_coordinates_stays_in_pixels(client, frame):
    """Без привязки сервис не выдумывает ни метры, ни геометрию."""
    body = _post(client, frame[0]).json()

    assert body["image"]["georeferenced"] is False
    assert body["geojson"] is None
    for detection in body["detections"]:
        assert detection["area_px"] > 0
        assert detection["area_m2"] is None
        assert detection["volume_m3"] is None
        assert detection["centroid"] is None


def test_physical_estimates_appear_with_gsd(client, frame):
    body = _post(client, frame[0], {"gsd_m_per_px": 0.05, "center": CENTER}).json()

    assert body["summary"]["total_volume_m3"] > 0
    for detection in body["detections"]:
        assert detection["area_m2"] > 0
        assert detection["mass_kg"] >= 0
        assert detection["fraction"] in {"mega", "macro", "meso", "micro"}


def test_gsd_can_be_derived_from_camera(client, frame):
    body = _post(
        client,
        frame[0],
        {
            "camera": {"altitude_m": 100, "focal_mm": 24, "sensor_width_mm": 13.2},
            "center": CENTER,
        },
    ).json()

    assert body["image"]["gsd_source"] == "camera"
    assert body["image"]["gsd_m_per_px"] > 0


def test_point_candidate_matches_backend_contract(client, frame):
    body = _post(
        client, frame[0], {"gsd_m_per_px": 0.05, "center": CENTER, "territory_id": 7}
    ).json()
    candidate = body["point_candidates"][0]

    assert candidate["source"] == "uav_auto"
    assert candidate["needs_human_validation"] is True
    assert candidate["territory_id"] == 7
    assert candidate["dominant_category"] in {
        "metal",
        "fishing_gear",
        "plastic",
        "wood",
        "construction",
        "rubber",
    }
    assert candidate["detection_ids"] == [d["id"] for d in body["detections"]]


def test_overlay_png_is_served_and_transparent(client, frame):
    body = _post(client, frame[0], {"gsd_m_per_px": 0.05, "center": CENTER}).json()
    response = client.get(body["overlay"]["png_url"])
    overlay = np.asarray(Image.open(io.BytesIO(response.content)))

    assert response.headers["content-type"] == "image/png"
    assert overlay.shape[2] == 4
    assert overlay[..., 3].min() == 0  # фон прозрачен — подложка карты видна
    assert overlay[..., 3].max() > 0
    assert body["overlay"]["bounds"] is not None


def test_overlay_can_be_disabled(client, frame):
    body = _post(client, frame[0], {"render_overlay": False}).json()

    assert body["overlay"]["png_url"] is None


def test_missing_artifact_returns_404(client):
    assert client.get("/api/v1/imagery/deadbeef/overlay.png").status_code == 404


def test_exif_gps_far_from_reported_position_is_flagged(client, frame, monkeypatch):
    """Антифрод B2: снимок сделан не там, где заявлено."""
    from app import pipeline

    monkeypatch.setattr(
        pipeline.georef,
        "read_exif",
        lambda image: pipeline.georef.ExifInfo(lat=CENTER[0] + 0.5, lon=CENTER[1]),
    )
    body = _post(client, frame[0], {"gsd_m_per_px": 0.05, "center": CENTER}).json()

    assert "exif_gps_mismatch" in {flag["code"] for flag in body["fraud_flags"]}


def test_missing_exif_is_flagged_as_info(client, frame):
    body = _post(client, frame[0]).json()

    flags = {flag["code"]: flag["severity"] for flag in body["fraud_flags"]}
    assert flags.get("exif_stripped") == "info"


def test_clean_frame_reports_no_detections(client):
    plain = Image.new("RGB", (256, 256), (198, 186, 158))
    body = _post(client, plain, {"gsd_m_per_px": 0.05, "center": CENTER}).json()

    assert body["summary"]["count"] == 0
    assert body["point_candidates"] == []
    assert "no_detections" in {flag["code"] for flag in body["fraud_flags"]}


def test_broken_meta_is_rejected(client, frame):
    buffer = io.BytesIO()
    frame[0].save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post(
        "/api/v1/imagery/detect",
        files={"image": ("frame.png", buffer, "image/png")},
        data={"meta": "{not json"},
    )

    assert response.status_code == 422


def test_non_image_payload_is_rejected(client):
    response = client.post(
        "/api/v1/imagery/detect",
        files={"image": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("bounds", [[[1, 2], [3, 4]], []])
def test_bounds_must_have_four_corners(client, frame, bounds):
    assert _post(client, frame[0], {"bounds": bounds}).status_code == 422


def test_same_frame_gives_same_answer(client, frame):
    """Детерминизм: одинаковый кадр — одинаковые находки, иначе метрики не воспроизвести."""
    first = _post(client, frame[0], {"gsd_m_per_px": 0.05}).json()
    second = _post(client, frame[0], {"gsd_m_per_px": 0.05}).json()

    assert [d["area_px"] for d in first["detections"]] == [
        d["area_px"] for d in second["detections"]
    ]
    assert [d["label"] for d in first["detections"]] == [d["label"] for d in second["detections"]]
