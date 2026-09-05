"""
Синтетические кадры БПЛА с известной разметкой.

Зачем синтетика, а не реальные снимки: датасет «Чистого берега» не опубликован
(лицензия), а без разметки нельзя посчитать ни IoU, ни precision — то есть
нечем показать, что конвейер вообще работает. Синтетика даёт ground truth
бесплатно и детерминированно: один и тот же seed — один и тот же кадр.

Чего синтетика не даёт: она не доказывает качество распознавания на настоящем
берегу. Она проверяет конвейер (кадр -> маска -> объекты -> объём -> полигоны
-> привязка) и служит регрессионным тестом. Качество партнёрской модели меряется
её же авторами на её же данных.

Запуск:
    python scripts/make_testdata.py --count 6 --out testdata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.taxonomy import GarbageClass

#: Базовые цвета сцены. Берег снят сверху: вода, мокрый песок, сухой песок, камни.
SCENE_BANDS: list[tuple[str, tuple[int, int, int], float]] = [
    ("water", (58, 92, 110), 0.30),
    ("wet_sand", (150, 138, 116), 0.18),
    ("dry_sand", (198, 186, 158), 0.34),
    ("stones", (128, 124, 118), 0.18),
]

#: Как выглядит мусор каждого класса: цвет, разброс цвета, зернистость.
CLASS_APPEARANCE: dict[int, tuple[tuple[int, int, int], int, float]] = {
    GarbageClass.iron.value: ((118, 116, 112), 18, 0.10),
    GarbageClass.fishing_gear.value: ((40, 190, 180), 26, 0.06),
    GarbageClass.plastic.value: ((45, 205, 90), 30, 0.04),
    GarbageClass.tree.value: ((132, 92, 48), 22, 0.07),
    GarbageClass.concrete.value: ((214, 212, 206), 12, 0.03),
    GarbageClass.rubber.value: ((34, 32, 34), 10, 0.04),
}

#: Центр сцены — Куршская коса: реальная ООПТ из сида, координаты правдоподобны.
DEFAULT_CENTER = (55.2833, 20.9167)
DEFAULT_GSD = 0.05  # м/px — типичный кадр БПЛА с 100 м


@dataclass
class GroundTruthObject:
    class_id: int
    label: str
    shape: str
    area_px: int
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1)


def _noise(rng: np.random.Generator, shape: tuple[int, int], strength: float) -> np.ndarray:
    """Зерно, одинаковое по всем каналам: имитирует шум сенсора и текстуру."""
    return rng.normal(0.0, strength * 255.0, shape)[..., None]


def _draw_scene(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Полосы берега сверху вниз + плавные переходы между ними."""
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    y = 0
    for _, colour, share in SCENE_BANDS:
        band_h = int(height * share)
        canvas[y : y + band_h] = colour
        y += band_h
    canvas[y:] = SCENE_BANDS[-1][1]

    # Береговая линия не прямая: сдвигаем полосы синусоидой.
    shift = (np.sin(np.linspace(0, 3 * np.pi, width)) * height * 0.03).astype(int)
    for x in range(width):
        canvas[:, x] = np.roll(canvas[:, x], shift[x], axis=0)

    canvas += _noise(rng, (height, width), 0.012)
    return canvas


def _blob_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    shape: str,
) -> np.ndarray:
    """Маска одного объекта. Форма зависит от типа мусора."""
    yy, xx = np.mgrid[0:height, 0:width]
    if shape == "strip":
        # Сеть/трос: длинная тонкая изогнутая полоса.
        thickness = max(height // 6, 2)
        curve = (np.sin(np.linspace(0, rng.uniform(1.5, 3.0) * np.pi, width)) + 1) / 2
        centre = curve * (height - thickness) + thickness / 2
        return np.abs(yy - centre[None, :]) <= thickness / 2
    if shape == "rect":
        # Плита/бочка: прямоугольник со срезанными углами.
        mask = np.ones((height, width), dtype=bool)
        cut = min(height, width) // 5
        mask[:cut, :cut] = mask[:cut, -cut:] = False
        mask[-cut:, :cut] = mask[-cut:, -cut:] = False
        return mask
    # Куча: эллипс с рваным краем.
    cy, cx = height / 2, width / 2
    ry, rx = height / 2.1, width / 2.1
    base = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    jitter = rng.normal(0, 0.06, base.shape)
    return base + jitter <= 1.0


SHAPE_BY_CLASS: dict[int, str] = {
    GarbageClass.iron.value: "rect",
    GarbageClass.fishing_gear.value: "strip",
    GarbageClass.plastic.value: "blob",
    GarbageClass.tree.value: "strip",
    GarbageClass.concrete.value: "rect",
    GarbageClass.rubber.value: "blob",
}

#: Диапазон размеров объекта, доля от стороны кадра.
SIZE_RANGE: dict[str, tuple[float, float]] = {
    "strip": (0.18, 0.34),
    "rect": (0.07, 0.15),
    "blob": (0.06, 0.14),
}


def make_frame(
    seed: int,
    *,
    width: int = 900,
    height: int = 700,
    objects: int = 5,
) -> tuple[Image.Image, np.ndarray, list[GroundTruthObject]]:
    """Кадр, его эталонная маска и список объектов."""
    rng = np.random.default_rng(seed)
    canvas = _draw_scene(rng, height, width)
    truth = np.zeros((height, width), dtype=np.uint8)
    records: list[GroundTruthObject] = []

    classes = list(CLASS_APPEARANCE)
    rng.shuffle(classes)
    placed: list[tuple[int, int, int, int]] = []

    for i in range(objects):
        class_id = classes[i % len(classes)]
        shape = SHAPE_BY_CLASS[class_id]
        lo, hi = SIZE_RANGE[shape]

        for _ in range(40):  # попытки поставить объект, не задев уже стоящие
            bw = int(width * rng.uniform(lo, hi))
            bh = int(bw * (rng.uniform(0.2, 0.4) if shape == "strip" else rng.uniform(0.7, 1.3)))
            bh = max(bh, 12)
            x0 = int(rng.integers(0, max(width - bw, 1)))
            # Мусор лежит на суше, а не в воде: начинаем ниже водной полосы.
            y0 = int(rng.integers(int(height * 0.34), max(height - bh, int(height * 0.35) + 1)))
            box = (x0, y0, x0 + bw, y0 + bh)
            if all(not _overlaps(box, other) for other in placed):
                placed.append(box)
                break
        else:
            continue

        mask = _blob_mask(rng, bh, bw, shape=shape)
        colour, spread, grain = CLASS_APPEARANCE[class_id]
        patch = np.array(colour, dtype=np.float64) + rng.normal(0, spread, 3)
        region = canvas[y0 : y0 + bh, x0 : x0 + bw]
        texture = rng.normal(0, grain * 255.0, (bh, bw))[..., None]
        region[mask] = (np.broadcast_to(patch, region.shape) + texture)[mask]

        truth[y0 : y0 + bh, x0 : x0 + bw][mask] = class_id
        ys, xs = np.where(mask)
        records.append(
            GroundTruthObject(
                class_id=class_id,
                label=GarbageClass(class_id).name,
                shape=shape,
                area_px=int(mask.sum()),
                bbox_px=(
                    x0 + int(xs.min()),
                    y0 + int(ys.min()),
                    x0 + int(xs.max()) + 1,
                    y0 + int(ys.max()) + 1,
                ),
            )
        )

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return image, truth, records


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 14) -> bool:
    return not (
        a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор тестовых кадров с разметкой")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--objects", type=int, default=5)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("testdata"))
    args = parser.parse_args()

    frames_dir = args.out / "frames"
    truth_dir = args.out / "truth"
    frames_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(args.count):
        seed = args.seed + i
        image, truth, records = make_frame(
            seed, width=args.width, height=args.height, objects=args.objects
        )
        name = f"frame_{i:02d}"
        image.save(frames_dir / f"{name}.png")
        Image.fromarray(truth).save(truth_dir / f"{name}.png")
        manifest.append(
            {
                "name": name,
                "seed": seed,
                "frame": f"frames/{name}.png",
                "truth": f"truth/{name}.png",
                "width": args.width,
                "height": args.height,
                "gsd_m_per_px": DEFAULT_GSD,
                "center": list(DEFAULT_CENTER),
                "objects": [asdict(r) for r in records],
            }
        )

    (args.out / "manifest.json").write_text(
        json.dumps({"frames": manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(len(f["objects"]) for f in manifest)
    print(f"готово: {len(manifest)} кадров, {total} объектов -> {args.out}")


if __name__ == "__main__":
    main()
