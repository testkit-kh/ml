"""
Маска сегментации → отдельные объекты, полигоны и картинка оверлея.

Модель отдаёт одну карту меток на кадр. Карта как таковая карте на фронте не
нужна: нужны отдельные объекты (кандидаты в точки) и их контуры (фиолетовые
полигоны поверх подложки). Здесь этот перевод и живёт.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

from app.taxonomy import OVERLAY_ACCENT, PALETTE

#: Меньше этого пятно считается шумом сегментации, px. Значение апстрима.
MIN_AREA_PX = 80

#: Допуск упрощения контура, px. Douglas–Peucker: полигон в сотни вершин
#: карту не улучшает, а вес тайла увеличивает.
POLYGON_TOLERANCE_PX = 2.0

#: Предел вершин на полигон — страховка от изрезанных контуров.
MAX_POLYGON_VERTICES = 200

#: Потолок объектов на кадр. Больше — это не находки, а шум сегментации.
MAX_BLOBS = 500


@dataclass(frozen=True)
class Blob:
    """Одно связное пятно одного класса.

    `mask` — локальная, размером с bbox, а не с кадр. Это не микрооптимизация:
    на мозаике 2600x2000 с четырьмя сотнями пятен полнокадровые маски — это
    два гигабайта и полминуты на одни только `np.where`. Все координаты
    наружу (`centroid_px`, `bbox_px`, `polygon_px`) при этом глобальные.
    """

    class_id: int
    mask: np.ndarray  # bool, размером с bbox
    offset_yx: tuple[int, int]  # где лежит левый верхний угол mask в кадре
    area_px: int
    centroid_px: tuple[float, float]  # (x, y)
    bbox_px: tuple[int, int, int, int]  # (y0, x0, y1, x1), y1/x1 не включая
    polygon_px: list[tuple[float, float]]  # [(x, y), ...], замкнут
    confidence: float

    def full_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Маска во весь кадр. Нужна сравнению с эталоном, не инференсу."""
        canvas = np.zeros(shape, dtype=bool)
        y0, x0 = self.offset_yx
        canvas[y0 : y0 + self.mask.shape[0], x0 : x0 + self.mask.shape[1]] = self.mask
        return canvas


def _largest_contour(mask: np.ndarray) -> list[tuple[float, float]]:
    """Внешний контур пятна в координатах (x, y). Дырки игнорируются."""
    padded = np.pad(mask.astype(np.float64), 1, mode="constant")
    contours = find_contours(padded, 0.5)
    if not contours:
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

    contour = max(contours, key=len)
    tol = POLYGON_TOLERANCE_PX
    simplified = approximate_polygon(contour, tolerance=tol)
    while len(simplified) > MAX_POLYGON_VERTICES:
        tol *= 2
        simplified = approximate_polygon(contour, tolerance=tol)

    pts = [(float(x - 1), float(y - 1)) for y, x in simplified]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def extract_blobs(
    label_mask: np.ndarray,
    *,
    prob_map: np.ndarray | None = None,
    min_area_px: int = MIN_AREA_PX,
    max_blobs: int = MAX_BLOBS,
) -> list[Blob]:
    """Карта меток -> список объектов, отсортированный по площади вниз.

    `prob_map` (H, W) — уверенность модели по пикселю; без неё уверенность
    объекта считается неизвестной и выставляется в 1.0: эвристический бэкенд
    свою уверенность передаёт явно.

    `max_blobs` — потолок на кадр. На грубой подложке модель может насыпать
    тысячи мелких пятен; отдавать их все бессмысленно (ответ на мегабайты) и
    вредно (модератор такое не разберёт). Оставляем самые крупные.
    """
    blobs: list[Blob] = []
    for class_id in np.unique(label_mask):
        class_id = int(class_id)
        if class_id not in PALETTE:
            continue
        labelled, count = ndimage.label(label_mask == class_id)
        if count == 0:
            continue
        # find_objects сразу отдаёт bbox каждой компоненты — дальше вся работа
        # идёт внутри него, а не по всему кадру.
        for index, window in enumerate(ndimage.find_objects(labelled), start=1):
            if window is None:
                continue
            ys, xs = window
            local = labelled[ys, xs] == index
            area = int(local.sum())
            if area < min_area_px:
                continue

            local_ys, local_xs = np.nonzero(local)
            y0, x0 = ys.start, xs.start
            conf = 1.0
            if prob_map is not None:
                conf = float(prob_map[ys, xs][local].mean())

            polygon = [
                (x + x0, y + y0) for x, y in _largest_contour(local)
            ]
            blobs.append(
                Blob(
                    class_id=class_id,
                    mask=local,
                    offset_yx=(y0, x0),
                    area_px=area,
                    centroid_px=(float(local_xs.mean()) + x0, float(local_ys.mean()) + y0),
                    bbox_px=(y0, x0, ys.stop, xs.stop),
                    polygon_px=polygon,
                    confidence=round(conf, 4),
                )
            )
    blobs.sort(key=lambda b: b.area_px, reverse=True)
    return blobs[:max_blobs]


def render_overlay(
    label_mask: np.ndarray,
    *,
    alpha: int = 140,
    outline: bool = True,
    by_class: bool = True,
) -> Image.Image:
    """RGBA-оверлей: прозрачный фон, фиолетовые пятна мусора.

    Кладётся на карту как `image`-источник MapLibre по четырём углам кадра —
    подложка сквозь него видна, потому что фон полностью прозрачный.
    """
    h, w = label_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        sel = label_mask == class_id
        if not sel.any():
            continue
        rgba[sel, :3] = color if by_class else OVERLAY_ACCENT
        rgba[sel, 3] = alpha

    if outline:
        garbage = label_mask > 0
        eroded = ndimage.binary_erosion(garbage, iterations=2, border_value=0)
        edge = garbage & ~eroded
        rgba[edge, 3] = 255

    return Image.fromarray(rgba)


def render_blend(image: Image.Image, label_mask: np.ndarray, *, weight: float = 0.3) -> Image.Image:
    """Кадр со впечатанным оверлеем — для отладки и для отчёта ООПТ."""
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    overlay = np.zeros_like(base)
    for class_id, color in PALETTE.items():
        overlay[label_mask == class_id] = color
    mixed = np.where(
        (label_mask > 0)[..., None],
        base * (1 - weight) + overlay * weight,
        base,
    )
    return Image.fromarray(mixed.astype(np.uint8))
