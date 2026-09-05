"""Замеры объёма и массы: проверяем поведение, а не конкретные числа."""

from __future__ import annotations

import numpy as np
import pytest

from app import estimate


def _square(size: int, canvas: int = 128) -> np.ndarray:
    mask = np.zeros((canvas, canvas), dtype=bool)
    start = (canvas - size) // 2
    mask[start : start + size, start : start + size] = True
    return mask


def test_thickness_grows_with_compactness():
    """Компактная куча «толще» вытянутой ленты той же площади — на этом стоит объём."""
    blob = _square(32)  # 32 x 32 = 1024 px
    strip = np.zeros((128, 128), dtype=bool)
    strip[60:68, :] = True  # 8 x 128 = 1024 px, площадь та же

    assert blob.sum() == strip.sum()
    assert estimate.thickness_blocks(blob, 3) > estimate.thickness_blocks(strip, 3)


def test_no_gsd_no_physics():
    """Без размера пикселя физические величины не выдумываются."""
    m = estimate.measure(_square(32), 3, gsd_m_per_px=None)

    assert m.area_px == 32 * 32
    assert (m.area_m2, m.volume_m3, m.mass_kg, m.fraction) == (None, None, None, None)


def test_area_scales_with_gsd_squared():
    mask = _square(40)
    coarse = estimate.measure(mask, 3, gsd_m_per_px=0.10)
    fine = estimate.measure(mask, 3, gsd_m_per_px=0.05)

    assert coarse.area_m2 == pytest.approx(fine.area_m2 * 4)


def test_mass_follows_density():
    """Одна и та же куча из бетона тяжелее, чем из пластика."""
    mask = _square(40)
    plastic = estimate.measure(mask, 3, gsd_m_per_px=0.05)
    concrete = estimate.measure(mask, 5, gsd_m_per_px=0.05)

    assert concrete.mass_kg > plastic.mass_kg


def test_depth_is_capped():
    """Пятно в пол-кадра не превращается в многометровую кучу."""
    huge = np.ones((512, 512), dtype=bool)
    m = estimate.measure(huge, 5, gsd_m_per_px=1.0)

    assert m.depth_m == estimate.MAX_DEPTH_M


@pytest.mark.parametrize(
    ("extent_m", "expected"),
    [(3.0, "mega"), (0.5, "macro"), (0.01, "meso"), (0.001, "micro")],
)
def test_fraction_bounds(extent_m: float, expected: str):
    assert estimate.fraction_of(extent_m) == expected


def test_empty_mask_is_not_a_crash():
    m = estimate.measure(np.zeros((64, 64), dtype=bool), 1, gsd_m_per_px=0.05)

    assert m.area_px == 0
    assert m.volume_m3 == 0.0
