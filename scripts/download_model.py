"""
Загрузка весов «Чистого берега».

Веса не лежат в репозитории намеренно: это ~100 МБ бинарника, который git
хранить не умеет, а пересобирать образ из-за них не нужно. Скрипт качает их
в том же виде, в каком их публикует Yandex DataSphere, и проверяет, что файл
действительно загружается как `state_dict`.

Запуск:
    PYTHONPATH=. python scripts/download_model.py --out model.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://storage.yandexcloud.net/socialtech/garbage-detect/seg-model/model.pt"


def download(url: str, target: Path, *, force: bool = False) -> Path:
    if target.exists() and not force:
        print(f"уже на месте: {target} ({target.stat().st_size / 1e6:.1f} МБ)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"качаю {url}")

    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} МБ", end="", flush=True)
    print()

    # Переименование в самом конце: недокачанный файл не должен выглядеть готовым.
    shutil.move(str(tmp), str(target))
    print(f"готово: {target} ({target.stat().st_size / 1e6:.1f} МБ)")
    return target


def verify(path: Path) -> None:
    """Проверяем, что это действительно state_dict SegFormer, а не HTML-заглушка."""
    try:
        import torch
    except ImportError:
        print("torch не установлен — проверка пропущена (pip install -r requirements-model.txt)")
        return

    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise SystemExit(f"ожидался state_dict, получено {type(state).__name__}")
    keys = list(state)
    print(f"тензоров в чекпоинте: {len(keys)}; первый ключ: {keys[0]}")

    head = [k for k in keys if k.endswith("classifier.weight")]
    if head:
        classes = state[head[0]].shape[0]
        print(f"классов в голове: {classes} (ожидается 7: фон + 6 типов мусора)")
        if classes != 7:
            raise SystemExit("число классов не совпадает с таксономией сервиса")


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать веса модели «Чистый берег»")
    parser.add_argument("--url", default=WEIGHTS_URL)
    parser.add_argument("--out", type=Path, default=Path("model.pt"))
    parser.add_argument("--force", action="store_true", help="перекачать поверх существующего")
    args = parser.parse_args()

    try:
        path = download(args.url, args.out, force=args.force)
    except OSError as exc:
        print(f"не удалось скачать: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    verify(path)


if __name__ == "__main__":
    main()
