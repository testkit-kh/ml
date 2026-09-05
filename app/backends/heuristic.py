"""
Правиловый бэйзлайн: сегментация мусора без весов, без torch и без GPU.

Зачем он есть. Веса «Чистого берега» — сотня мегабайт и torch в зависимостях;
на защите и в CI это лишняя точка отказа, а до подключения GPU весь остальной
сервис (разбор кадра, замеры, полигоны, привязка, контракт с бэком) должен
проверяться сквозь. Поэтому бэйзлайн честно помечен `trained=False`: его
качество меряется на своих тестовых данных и не выдаётся за качество
партнёрской модели.

Как он работает. Берег сверху — это три-четыре доминирующих цвета: вода, мокрый
песок, сухой песок, камни. Мусор в эту палитру не укладывается. Поэтому фон
моделируется не пространственно, а в цветовом пространстве: k-means по кадру,
доминирующие кластеры объявляются фоном, а признак — расстояние пикселя до
ближайшего фонового цвета. Второй признак — зернистость поверхности: мусор
шероховатый, песок и вода гладкие. Класс назначается по тону, светлоте и форме
пятна.

Почему не «яркий цвет = мусор», как напрашивается: вода на снимке насыщенно
синяя и занимает треть кадра — абсолютный порог насыщенности объявляет мусором
море.

Почему фон не пространственный (размытие кадра, скользящее окно): граница воды
и песка — резкий перепад в сотню уровней, любое окно размазывает её на свою
ширину, и вдоль всей береговой линии появляется полоса ложных срабатываний.
У цветовой модели такого артефакта нет по построению: она не знает, где пиксель
лежит.
"""

from __future__ import annotations

import colorsys

import numpy as np
from PIL.Image import Image as PILImage
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from app.backends.base import BackendInfo, Segmentation
from app.taxonomy import GarbageClass

VERSION = "0.1.0"

#: Порог расстояния до ближайшего фонового цвета (евклид в RGB 0..1).
#: 0.12 ≈ 30 уровней из 255: ниже этого разница тонет в шуме сенсора и тенях.
BACKGROUND_DELTA_THRESHOLD = 0.12

#: Сколько цветовых кластеров искать в кадре.
PALETTE_CLUSTERS = 6

#: Доля кадра, начиная с которой кластер считается фоном, а не находкой.
#: Кластер меньше — это как раз мусор, и вычитать его из самого себя нельзя.
PALETTE_BACKGROUND_SHARE = 0.08

#: На скольких пикселях по короткой стороне оценивается палитра. Полный кадр
#: для k-means избыточен: цвета фона видны и на прореженной выборке.
PALETTE_SAMPLE_SIDE = 160

#: Порог зернистости для ахроматичного мусора: металл и покрышки шершавые,
#: песок и вода — нет. Значение в долях яркости, ~7 уровней из 255.
TEXTURE_THRESHOLD = 0.028

#: Окно локальной статистики, px.
TEXTURE_WINDOW = 9

#: Порог насыщенности, выше которого пятно считается искусственно цветным.
#: Применяется к уже найденному объекту, а не к каждому пикселю кадра.
SATURATION_THRESHOLD = 0.30

#: Насыщенность тёплого пятна, начиная с которой оно считается древесиной.
WOOD_SATURATION = 0.25

#: Границы светлоты: темнее — резина, светлее — бетон, между ними металл.
DARK_VALUE = 0.28
BRIGHT_VALUE = 0.70

#: Радиус морфологического закрытия/открытия — сшивает разрывы и снимает соль.
MORPH_RADIUS = 2

#: Доля кадра, больше которой пятно считается фоном (вода, поле, лес), а не
#: находкой. Куча мусора в четверть кадра БПЛА — это уже не куча, а ошибка.
MAX_BLOB_FRAME_RATIO = 0.25


def _rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Векторный RGB->HSV на float 0..1: `colorsys` поэлементно слишком медленный."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    safe = diff > 1e-6
    idx = safe & (mx == r)
    hue[idx] = ((g - b)[idx] / diff[idx]) % 6
    idx = safe & (mx == g) & ~(mx == r)
    hue[idx] = ((b - r)[idx] / diff[idx]) + 2
    idx = safe & (mx == b) & ~(mx == r) & ~(mx == g)
    hue[idx] = ((r - g)[idx] / diff[idx]) + 4
    hue = (hue / 6.0) % 1.0
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return hue, sat, mx


def _local_texture(gray: np.ndarray, window: int) -> np.ndarray:
    """Зернистость поверхности: усреднённый высокочастотный остаток.

    Не локальное СКО, хотя оно и напрашивается: у СКО максимум приходится на
    границу воды и песка, из-за чего вдоль всей береговой линии тянется полоса
    «шероховатости», сшивающая разные объекты в одно пятно. Медиана 3x3 резкий
    перепад сохраняет (это её свойство), поэтому в остатке от края почти ничего
    не остаётся, а случайное зерно материала остаётся целиком.
    """
    residual = np.abs(gray - ndimage.median_filter(gray, size=3, mode="reflect"))
    return ndimage.uniform_filter(residual, size=window, mode="reflect")


def background_palette(arr: np.ndarray, *, clusters: int = PALETTE_CLUSTERS) -> np.ndarray:
    """Доминирующие цвета кадра — модель фона. (N, 3), N >= 1.

    Детерминирована: `seed` зафиксирован, инициализация k-means++ от него же,
    поэтому один и тот же кадр всегда даёт одну и ту же палитру.
    """
    height, width = arr.shape[:2]
    step = max(min(height, width) // PALETTE_SAMPLE_SIDE, 1)
    sample = arr[::step, ::step].reshape(-1, 3).astype(np.float64)
    k = min(clusters, len(sample))
    if k < 2:
        return sample[:1] if len(sample) else np.zeros((1, 3))

    centers, labels = kmeans2(sample, k, minit="++", seed=0, iter=25, missing="warn")
    share = np.bincount(labels, minlength=k) / len(labels)
    background = centers[share >= PALETTE_BACKGROUND_SHARE]
    if len(background) == 0:
        # Кадр без единого доминирующего цвета (сплошная мозаика) — берём
        # самый населённый кластер, иначе фона не останется вовсе.
        background = centers[[int(share.argmax())]]
    return background


def _background_delta(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Расстояние до ближайшего фонового цвета, 0..~1.7 (евклид в RGB)."""
    # По одному центру за раз: матрица (H*W, N, 3) на ортофото не влезает в память.
    delta = np.full(arr.shape[:2], np.inf, dtype=np.float32)
    for center in palette:
        diff = arr - center.astype(np.float32)
        np.minimum(delta, np.sqrt((diff * diff).sum(axis=-1)), out=delta)
    return delta


def classify(hue: float, sat: float, val: float, elongation: float, fill: float) -> int:
    """Класс пятна по усреднённым признакам.

    Порядок проверок — по убыванию надёжности признака. Цвет плавника узнаётся
    надёжнее его формы, поэтому дерево проверяется раньше формы: иначе бревно,
    вытянутое ровно как трос, уезжает в рыболовные снасти.
    """
    # Тёмное — первым делом. Насыщенность считается как (max-min)/max, и на
    # тёмном пикселе знаменатель мал: шум сенсора в пару уровней даёт
    # «насыщенный цвет» на ровном месте, и покрышка уезжает в пластик.
    if val <= DARK_VALUE:
        return GarbageClass.rubber.value
    # Плавник узнаётся по цвету надёжнее, чем по форме: бревно вытянуто ровно
    # как трос, поэтому тёплый тон проверяется раньше вытянутости.
    if sat >= WOOD_SATURATION and 0.01 <= hue <= 0.15:
        return GarbageClass.tree.value
    # Сеть, трос, ярус: сильно вытянутое и рыхлое пятно — площадь много меньше bbox.
    if elongation >= 3.0 and fill <= 0.55:
        return GarbageClass.fishing_gear.value
    if sat >= SATURATION_THRESHOLD:
        return GarbageClass.plastic.value
    if val >= BRIGHT_VALUE:
        return GarbageClass.concrete.value
    return GarbageClass.iron.value


class HeuristicDetector:
    """Бэйзлайн на цвете и текстуре. Детерминирован: тот же вход — тот же выход."""

    def __init__(
        self,
        *,
        delta_threshold: float = BACKGROUND_DELTA_THRESHOLD,
        texture_threshold: float = TEXTURE_THRESHOLD,
        saturation_threshold: float = SATURATION_THRESHOLD,
        min_area_px: int = 80,
    ) -> None:
        self.delta_threshold = delta_threshold
        self.texture_threshold = texture_threshold
        self.saturation_threshold = saturation_threshold
        self.min_area_px = min_area_px

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="heuristic",
            version=VERSION,
            weights=None,
            device="cpu",
            tile_px=0,
            stride_px=0,
            trained=False,
            notes=(
                "Правиловый бэйзлайн на цвете и текстуре, без обучения. "
                "Нужен для сквозной проверки конвейера и как fallback, когда "
                "веса «Чистого берега» недоступны."
            ),
        )

    def segment(self, image: PILImage) -> Segmentation:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        hue, sat, val = _rgb_to_hsv(arr)
        gray = arr.mean(axis=-1)
        texture = _local_texture(gray, TEXTURE_WINDOW)
        palette = background_palette(arr)
        delta = _background_delta(arr, palette)

        foreground = (delta >= self.delta_threshold) | (texture >= self.texture_threshold)

        structure = ndimage.generate_binary_structure(2, 2)
        foreground = ndimage.binary_closing(foreground, structure, iterations=MORPH_RADIUS)
        foreground = ndimage.binary_opening(foreground, structure, iterations=MORPH_RADIUS)
        foreground = ndimage.binary_fill_holes(foreground)

        labels = np.zeros(gray.shape, dtype=np.uint8)
        probs = np.zeros(gray.shape, dtype=np.float32)

        frame_area = float(gray.size)
        components, count = ndimage.label(foreground, structure=structure)
        for idx in range(1, count + 1):
            blob = components == idx
            area = int(blob.sum())
            if area < self.min_area_px or area > frame_area * MAX_BLOB_FRAME_RATIO:
                continue
            ys, xs = np.where(blob)
            h = int(ys.max() - ys.min()) + 1
            w = int(xs.max() - xs.min()) + 1
            elongation = max(h, w) / max(min(h, w), 1)
            fill = area / float(h * w)

            mean_sat = float(sat[blob].mean())
            mean_val = float(val[blob].mean())
            # Циклическое среднее тона: 0.99 и 0.01 — соседние оттенки, а не
            # противоположные, поэтому усредняем углы, а не числа.
            angles = hue[blob] * 2 * np.pi
            mean_hue = float(
                (np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) / (2 * np.pi)) % 1.0
            )

            labels[blob] = classify(mean_hue, mean_sat, mean_val, elongation, fill)

            # «Уверенность» = запас, с которым пятно прошло пороги. Это не
            # вероятность модели; в ответе она помечена источником `heuristic`.
            delta_margin = float(delta[blob].mean()) / max(self.delta_threshold, 1e-6)
            texture_margin = float(texture[blob].mean()) / max(self.texture_threshold, 1e-6)
            probs[blob] = float(np.clip(0.40 + 0.20 * max(delta_margin, texture_margin), 0.0, 0.99))

        return Segmentation(labels=labels, probs=probs)


def hsv_of(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Утилита для тестов и подбора порогов: цвет -> его HSV."""
    return colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
