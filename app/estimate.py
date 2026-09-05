"""
Из маски — в объём, массу и фракцию. Все коэффициенты названы и объяснены.

Метод толщины взят из `calc_coefs` апстрима: модель видит мусор сверху, высоту
кучи измерить нечем, поэтому за прокси высоты берётся её *горизонтальный*
характерный размер — насколько глубоко внутрь пятна можно уйти в четырёх
направлениях. Плоская плёнка на песке даёт тонкий отклик, куча покрышек —
толстый. Это допущение, а не измерение, и оно возвращается клиенту в
`assumptions`, чтобы на защите не выдавать его за факт.

Отличие от апстрима: там `volume_coef` — безразмерное число в пикселях, годное
только для сравнения куч между собой. Здесь оно доводится до метров через GSD
(размер пикселя на местности), а без GSD физические величины честно не
считаются вовсе.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Сторона блока, которым маска огрубляется перед замером толщины, px.
BLOCK_PX = 4

#: Предел «заглядывания» внутрь пятна, в блоках. Больше — дороже и бессмысленно:
#: у плёнки толщины нет, у бетонной плиты она упирается в размер плиты.
MAX_WINDOW_BLOCKS: dict[int, int] = {1: 2, 2: 2, 3: 2, 4: 4, 5: 6, 6: 3}

#: Доля описанного объёма, которую реально занимает материал. Сеть — это
#: в основном воздух, бетонная плита — почти сплошная.
FILL_COEF: dict[int, float] = {1: 0.025, 2: 0.125, 3: 0.1, 4: 0.125, 5: 0.5, 6: 0.05}

#: Плотность материала, кг/м³ (значения апстрима; пластик — влажный).
DENSITY_KG_M3: dict[int, float] = {1: 2000, 2: 100, 3: 100, 4: 200, 5: 700, 6: 700}

#: Потолок эффективной высоты, м. Страховка от артефактов сегментации: пятно
#: в пол-кадра не должно превратиться в двухэтажную кучу.
MAX_DEPTH_M = 3.0

#: Границы фракции по наибольшему габариту объекта, м (классификация морского
#: мусора: mega > 1 м, macro 2.5 см – 1 м, meso 0.5 – 2.5 см, micro < 0.5 см).
FRACTION_BOUNDS_M: list[tuple[float, str]] = [
    (1.0, "mega"),
    (0.025, "macro"),
    (0.005, "meso"),
]


@dataclass(frozen=True)
class Measurement:
    """Замер одного пятна. `*_m2/_m3/_kg` = None, если кадр без привязки."""

    area_px: int
    thickness_blocks: float
    area_m2: float | None
    depth_m: float | None
    volume_m3: float | None
    mass_kg: float | None
    max_extent_m: float | None
    fraction: str | None


def _block_fill(mask: np.ndarray) -> np.ndarray:
    """Доля заполнения в блоках BLOCK_PX x BLOCK_PX, значения 0..1."""
    h, w = mask.shape
    bh, bw = h // BLOCK_PX, w // BLOCK_PX
    if bh == 0 or bw == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cropped = mask[: bh * BLOCK_PX, : bw * BLOCK_PX].astype(np.float64)
    return cropped.reshape(bh, BLOCK_PX, bw, BLOCK_PX).mean(axis=(1, 3))


def _directional_min(fill: np.ndarray, window: int) -> np.ndarray:
    """Минимум по четырём направлениям от суммы заполнения в окне `window`.

    Для каждого блока считается, сколько «вещества» лежит слева, справа, сверху
    и снизу от него в пределах окна; берётся минимум. У края пятна хотя бы одно
    направление пустое, поэтому вклад края мал, а у ядра кучи велик.
    """
    if fill.size == 0:
        return fill
    k = window - 1
    if k == 0:
        return fill.copy()
    pad = np.pad(fill, ((k, k), (k, k)), mode="constant")
    # Скользящие суммы по строкам и по столбцам за O(N) через кумулятивные суммы.
    csum_h = np.cumsum(np.pad(pad, ((0, 0), (1, 0)), mode="constant"), axis=1)
    hor = csum_h[:, window:] - csum_h[:, :-window]
    csum_v = np.cumsum(np.pad(pad, ((1, 0), (0, 0)), mode="constant"), axis=0)
    vert = csum_v[window:, :] - csum_v[:-window, :]

    left = hor[k:-k, : -k or None]
    right = hor[k:-k, k:]
    up = vert[: -k or None, k:-k]
    down = vert[k:, k:-k]
    n = min(left.shape[0], right.shape[0], up.shape[0], down.shape[0])
    m = min(left.shape[1], right.shape[1], up.shape[1], down.shape[1])
    return np.stack([left[:n, :m], right[:n, :m], up[:n, :m], down[:n, :m]]).min(axis=0)


def thickness_blocks(mask: np.ndarray, class_id: int) -> float:
    """Характерная толщина пятна в блоках. 0 — если пятно меньше блока."""
    fill = _block_fill(mask)
    total = float(fill.sum())
    if total <= 0:
        return 0.0
    inner = _directional_min(fill, MAX_WINDOW_BLOCKS[class_id])
    return float(inner.sum()) / total


def measure(
    mask: np.ndarray,
    class_id: int,
    *,
    gsd_m_per_px: float | None,
    bbox_px: tuple[int, int, int, int] | None = None,
) -> Measurement:
    """Полный замер пятна. Без `gsd_m_per_px` отдаёт только пиксели."""
    area_px = int(mask.sum())
    thick = thickness_blocks(mask, class_id)

    if gsd_m_per_px is None or gsd_m_per_px <= 0:
        return Measurement(area_px, thick, None, None, None, None, None, None)

    area_m2 = area_px * gsd_m_per_px**2
    depth_m = min(thick * BLOCK_PX * gsd_m_per_px * FILL_COEF[class_id], MAX_DEPTH_M)
    volume_m3 = area_m2 * depth_m
    mass_kg = volume_m3 * DENSITY_KG_M3[class_id]

    if bbox_px is not None:
        y0, x0, y1, x1 = bbox_px
        max_extent_m = max(y1 - y0, x1 - x0) * gsd_m_per_px
    else:
        max_extent_m = float(np.sqrt(area_m2))

    return Measurement(
        area_px=area_px,
        thickness_blocks=thick,
        area_m2=area_m2,
        depth_m=depth_m,
        volume_m3=volume_m3,
        mass_kg=mass_kg,
        max_extent_m=max_extent_m,
        fraction=fraction_of(max_extent_m),
    )


def fraction_of(max_extent_m: float | None) -> str | None:
    """Фракция по наибольшему габариту."""
    if max_extent_m is None:
        return None
    for bound, name in FRACTION_BOUNDS_M:
        if max_extent_m >= bound:
            return name
    return "micro"


def assumptions() -> dict[str, object]:
    """Всё, что сервис принял на веру, — в ответ, а не в комментарий в коде."""
    return {
        "block_px": BLOCK_PX,
        "max_window_blocks": MAX_WINDOW_BLOCKS,
        "fill_coef": FILL_COEF,
        "density_kg_m3": DENSITY_KG_M3,
        "max_depth_m": MAX_DEPTH_M,
        "depth_method": (
            "высота кучи не измеряется — за прокси взят горизонтальный "
            "характерный размер пятна (метод calc_coefs апстрима), "
            "переведённый в метры через GSD"
        ),
    }
