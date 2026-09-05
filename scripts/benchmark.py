"""
Замер скорости инференса. Нужен, чтобы про ускорение говорить числами.

Меряет то, что видит клиент: сегментацию кадра целиком, включая тайлинг и
перегон данных, а не отдельный прогон сети. Первый прогон отбрасывается —
на GPU он включает разогрев CUDA-контекста и аллокатора и к установившейся
скорости отношения не имеет.

Запуск:
    PYTHONPATH=. python scripts/benchmark.py --backend segformer --size 2688x1792
    MODEL_DEVICE=cpu PYTHONPATH=. python scripts/benchmark.py --backend segformer
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.backends import get_detector


def make_frame(width: int, height: int, seed: int = 0) -> Image.Image:
    """Синтетический кадр. Для замера скорости важен только размер."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def bench(detector, image: Image.Image, runs: int) -> dict:
    detector.segment(image)  # разогрев: не считается
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        detector.segment(image)
        samples.append(time.perf_counter() - start)

    megapixels = image.width * image.height / 1e6
    best = min(samples)
    return {
        "runs": runs,
        "best_s": round(best, 3),
        "median_s": round(statistics.median(samples), 3),
        "megapixels": round(megapixels, 2),
        "mpx_per_s": round(megapixels / best, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Скорость инференса")
    parser.add_argument("--backend", default="segformer")
    parser.add_argument("--size", default="2688x1792", help="ШхВ кадра")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--image", type=Path, default=None, help="взять настоящий кадр")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        width, height = (int(v) for v in args.size.lower().split("x"))
        image = make_frame(width, height)

    detector = get_detector(args.backend)
    info = detector.info()
    print(f"бэкенд {info.name} {info.version}, устройство {info.device}")
    print(f"тайл {info.tile_px} шаг {info.stride_px}, кадр {image.width}x{image.height}")

    result = bench(detector, image, args.runs)
    print(
        f"лучшее {result['best_s']} с, медиана {result['median_s']} с "
        f"({result['mpx_per_s']} Мпикс/с на {result['megapixels']} Мпикс)"
    )


if __name__ == "__main__":
    main()
