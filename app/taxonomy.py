"""
Таксономия классов модели «Чистый берег» и её стык с моделью мусора бэкенда.

Источник истины по классам — `model-server/code/server.py` из
https://github.com/yandex-datasphere/garbage-detection : SegFormer с 7 метками,
где 0 — фон. Порядок меток менять нельзя: он зашит в веса `model.pt`.

Здесь же — единственное место, где классы модели переводятся в `TrashCategory`
бэкенда. Если бэкенд поменяет свой enum, правится только `TO_TRASH_CATEGORY`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

BACKGROUND_ID = 0


class GarbageClass(int, enum.Enum):
    """Метки модели. Значения = индексы каналов в весах, не переставлять."""

    iron = 1
    fishing_gear = 2
    plastic = 3
    tree = 4
    concrete = 5
    rubber = 6


#: Метки в том виде, в каком их ждёт `SegformerForSemanticSegmentation`.
ID2LABEL: dict[int, str] = {
    0: "background",
    1: "iron",
    2: "fishing gear",
    3: "plastic",
    4: "tree",
    5: "concrete",
    6: "rubber",
}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

#: Русские названия — для карточки точки и отчётов ООПТ.
LABEL_RU: dict[int, str] = {
    1: "металл",
    2: "рыболовные снасти",
    3: "пластик",
    4: "дерево",
    5: "бетон",
    6: "резина",
}

#: Перевод в `app.cleanup_cost.TrashCategory` бэкенда. Строки, а не импорт:
#: микросервис не должен зависеть от кода бэкенда.
TO_TRASH_CATEGORY: dict[int, str] = {
    1: "metal",
    2: "fishing_gear",
    3: "plastic",
    4: "wood",
    5: "construction",
    6: "rubber",
}

#: Палитра апстрима — нужна, чтобы наши картинки совпадали с их QGIS-плагином.
UPSTREAM_PALETTE: dict[int, tuple[int, int, int]] = {
    1: (255, 255, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 0, 0),
    5: (255, 0, 255),
    6: (255, 255, 0),
}

#: Палитра продукта: фиолетовый оверлей мусора на карте. Один тон (violet),
#: классы различаются светлотой — на спутниковой подложке это читается лучше,
#: чем шесть конкурирующих чистых цветов.
PALETTE: dict[int, tuple[int, int, int]] = {
    1: (0x4C, 0x1D, 0x95),  # металл — самый тёмный
    2: (0x6D, 0x28, 0xD9),
    3: (0x8B, 0x5C, 0xF6),  # пластик — базовый акцент
    4: (0xA7, 0x8B, 0xFA),
    5: (0xC4, 0xB5, 0xFD),
    6: (0x5B, 0x21, 0xB6),
}

#: Цвет агрегированного слоя «здесь мусор» без разбивки по классам.
OVERLAY_ACCENT = (0x8B, 0x5C, 0xF6)


@dataclass(frozen=True)
class ClassInfo:
    id: int
    label: str
    label_ru: str
    trash_category: str
    color_rgb: tuple[int, int, int]
    color_hex: str


def to_hex(rgb: tuple[int, int, int]) -> str:
    """RGB -> #RRGGBB: фронт красит слои этой строкой."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def class_info(class_id: int) -> ClassInfo:
    rgb = PALETTE[class_id]
    return ClassInfo(
        id=class_id,
        label=ID2LABEL[class_id],
        label_ru=LABEL_RU[class_id],
        trash_category=TO_TRASH_CATEGORY[class_id],
        color_rgb=rgb,
        color_hex=to_hex(rgb),
    )


def all_classes() -> list[ClassInfo]:
    return [class_info(c.value) for c in GarbageClass]
