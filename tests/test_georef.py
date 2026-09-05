"""Привязка кадра к местности."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app import georef


def test_center_of_frame_maps_to_center_coordinate():
    gr = georef.georeference_from_center(
        lat=55.2833, lon=20.9167, gsd_m_per_px=0.05, width_px=1000, height_px=800
    )
    lon, lat = gr.to_lonlat(499.5, 399.5)

    assert lat == pytest.approx(55.2833, abs=1e-6)
    assert lon == pytest.approx(20.9167, abs=1e-6)


def test_frame_width_matches_gsd_on_the_ground():
    """1000 px по 0.05 м — это 50 м местности, с точностью до плоской аппроксимации."""
    gr = georef.georeference_from_center(
        lat=55.2833, lon=20.9167, gsd_m_per_px=0.05, width_px=1000, height_px=800
    )
    tl, tr, _, _ = gr.corners

    assert georef.haversine_m(*tl, *tr) == pytest.approx(50.0, rel=0.01)


def test_north_up_frame_has_north_at_the_top():
    gr = georef.georeference_from_center(
        lat=55.0, lon=20.0, gsd_m_per_px=0.1, width_px=200, height_px=200
    )
    _, top_lat = gr.to_lonlat(100, 0)
    _, bottom_lat = gr.to_lonlat(100, 199)

    assert top_lat > bottom_lat


def test_heading_rotates_the_footprint():
    straight = georef.georeference_from_center(
        lat=55.0, lon=20.0, gsd_m_per_px=0.1, width_px=200, height_px=100
    )
    turned = georef.georeference_from_center(
        lat=55.0, lon=20.0, gsd_m_per_px=0.1, width_px=200, height_px=100, heading_deg=90
    )

    assert straight.corners != turned.corners


def test_bounds_are_interpolated_bilinearly():
    gr = georef.Georeference(
        width_px=101,
        height_px=101,
        corners=((10.0, 20.0), (10.0, 21.0), (9.0, 21.0), (9.0, 20.0)),
    )

    assert gr.to_lonlat(0, 0) == (20.0, 10.0)
    assert gr.to_lonlat(100, 100) == (21.0, 9.0)
    assert gr.to_lonlat(50, 50) == (20.5, 9.5)


def test_gsd_from_camera_matches_textbook_formula():
    # 100 м высоты, 24 мм объектив, матрица 13.2 мм, 4000 px -> 1.375 см/px.
    gsd = georef.gsd_from_camera(
        altitude_m=100, focal_mm=24, sensor_width_mm=13.2, image_width_px=4000
    )

    assert gsd == pytest.approx(0.01375, rel=1e-6)


def test_image_without_exif_is_not_an_error():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buffer, format="PNG")
    info = georef.read_exif(Image.open(buffer))

    assert info.lat is None and info.captured_at is None


def test_haversine_is_symmetric_and_zero_on_itself():
    assert georef.haversine_m(55.0, 20.0, 55.0, 20.0) == pytest.approx(0.0)
    assert georef.haversine_m(55.0, 20.0, 55.1, 20.1) == pytest.approx(
        georef.haversine_m(55.1, 20.1, 55.0, 20.0)
    )
