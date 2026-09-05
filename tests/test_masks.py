"""Разбор маски на объекты и отрисовка оверлея."""

from __future__ import annotations

import numpy as np

from app import masks
from app.taxonomy import PALETTE


def _labels() -> np.ndarray:
    canvas = np.zeros((200, 240), dtype=np.uint8)
    canvas[20:60, 20:70] = 3  # пластик
    canvas[120:170, 140:230] = 5  # бетон
    canvas[5:8, 5:8] = 1  # шум: 9 px
    return canvas


def test_blobs_are_split_by_class_and_sorted_by_area():
    blobs = masks.extract_blobs(_labels())

    assert [b.class_id for b in blobs] == [5, 3]
    assert blobs[0].area_px > blobs[1].area_px


def test_small_specks_are_dropped():
    """Девять пикселей — это шум сегментации, а не находка для модератора."""
    assert all(b.class_id != 1 for b in masks.extract_blobs(_labels()))


def test_min_area_is_configurable():
    blobs = masks.extract_blobs(_labels(), min_area_px=4)

    assert any(b.class_id == 1 for b in blobs)


def test_polygon_is_closed_and_inside_bbox():
    blob = next(b for b in masks.extract_blobs(_labels()) if b.class_id == 3)
    y0, x0, y1, x1 = blob.bbox_px

    assert blob.polygon_px[0] == blob.polygon_px[-1]
    assert all(x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1 for x, y in blob.polygon_px)


def test_confidence_comes_from_probability_map():
    labels = _labels()
    probs = np.where(labels == 3, 0.8, 0.5).astype(np.float32)
    blob = next(b for b in masks.extract_blobs(labels, prob_map=probs) if b.class_id == 3)

    assert blob.confidence == 0.8


def test_overlay_is_transparent_outside_garbage():
    overlay = np.asarray(masks.render_overlay(_labels()))
    labels = _labels()

    assert overlay.shape == (200, 240, 4)
    assert overlay[labels == 0, 3].max() == 0
    assert overlay[labels == 3, 3].min() > 0


def test_overlay_uses_class_colours():
    overlay = np.asarray(masks.render_overlay(_labels(), outline=False))
    labels = _labels()

    assert tuple(overlay[labels == 3][0][:3]) == PALETTE[3]
    assert tuple(overlay[labels == 5][0][:3]) == PALETTE[5]


def test_single_accent_mode_paints_everything_one_colour():
    overlay = np.asarray(masks.render_overlay(_labels(), by_class=False, outline=False))
    labels = _labels()
    colours = {tuple(px[:3]) for px in overlay[labels > 0]}

    assert len(colours) == 1


def test_empty_mask_produces_empty_overlay():
    empty = np.zeros((32, 32), dtype=np.uint8)

    assert masks.extract_blobs(empty) == []
    assert np.asarray(masks.render_overlay(empty))[..., 3].max() == 0
