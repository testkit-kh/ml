"""
Собрать окружение с GPU, переиспользуя torch из чужой установки.

Зачем. CUDA-сборка torch — это ~3 ГБ загрузки. Если на машине уже стоит что-то
тяжёлое с CUDA (ComfyUI, Automatic1111, любой локальный инференс), торч там уже
есть и работает. Качать второй экземпляр незачем, особенно на плохом канале:
достаточно создать своё окружение и дописать в него путь к чужому
`site-packages`.

Как это работает. Путь добавляется файлом `.pth` в `site-packages` нашего venv.
Пути из `.pth` попадают в конец `sys.path` — **после** наших собственных
пакетов. Это принципиально: у нас свой `transformers` ветки 4.x (под опубликованные
веса «Чистого берега»), а в чужом окружении может стоять 5.x, и он не должен
нас перекрывать. Наши пакеты выигрывают, torch берётся оттуда.

Что нужно совпасть: версия Python. Колёса с расширениями (`cp313`) под другую
версию не подходят, поэтому venv создаётся тем же интерпретатором.

Чего скрипт не делает: он ничего не меняет в чужой установке. Только читает.

Проверено на: ComfyUI (Python 3.13, torch 2.12.1+cu130, RTX 4060 Ti).
Итог — 2.5 с на кадр 2688x1792 вместо 126 с на процессоре, при загрузке из
сети около 30 МБ вместо трёх гигабайт.

Запуск:
    python scripts/setup_gpu_env.py \
        --donor "C:/AI/comfy/ComfyUI/ComfyUI/.venv" \
        --venv .venv-gpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: Что ставим сами. Всё остальное (torch, numpy, scipy, PIL, pydantic)
#: подтягивается из донора, если там есть.
OUR_PACKAGES = [
    "fastapi[standard]",
    "uvicorn[standard]",
    "python-multipart",
    "scikit-image",
    "structlog",
    "pytest",
    "httpx",
    # Ветка 4.x обязательна: в 5.x переименованы внутренние модули SegFormer,
    # и опубликованный model.pt под новые имена не встаёт.
    "transformers==4.46.*",
]

PTH_NAME = "_donor_site_packages.pth"


def donor_site_packages(donor: Path) -> Path:
    for candidate in (donor / "Lib" / "site-packages", donor / "lib" / "site-packages"):
        if candidate.is_dir():
            return candidate
    # На Linux путь содержит версию: lib/python3.13/site-packages.
    matches = sorted((donor / "lib").glob("python*/site-packages"))
    if matches:
        return matches[0]
    raise SystemExit(f"в {donor} не нашёл site-packages")


def donor_python(donor: Path) -> Path:
    for candidate in (donor / "Scripts" / "python.exe", donor / "bin" / "python"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"в {donor} не нашёл интерпретатор")


def check_donor_torch(python: Path) -> str:
    probe = (
        "import torch, sys; "
        "print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
    )
    result = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"у донора не импортируется torch:\n{result.stderr.strip()[:500]}")
    version, torch_version, cuda = result.stdout.split()
    print(f"донор: Python {version}, torch {torch_version}, CUDA доступна: {cuda}")
    if cuda != "True":
        print("  ВНИМАНИЕ: у донора CUDA недоступна — ускорения не будет")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Окружение с GPU поверх чужого torch")
    parser.add_argument("--donor", type=Path, required=True, help="venv с рабочим CUDA-torch")
    parser.add_argument("--venv", type=Path, default=Path(".venv-gpu"))
    parser.add_argument("--skip-install", action="store_true", help="только прописать путь")
    args = parser.parse_args()

    donor = args.donor.resolve()
    packages_dir = donor_site_packages(donor)
    python = donor_python(donor)
    check_donor_torch(python)

    if not args.venv.exists():
        print(f"создаю {args.venv} интерпретатором донора")
        subprocess.run([str(python), "-m", "venv", str(args.venv)], check=True)

    venv_python = donor_python(args.venv)
    our_packages_dir = donor_site_packages(args.venv)
    (our_packages_dir / PTH_NAME).write_text(f"{packages_dir}\n", encoding="utf-8")
    print(f"путь донора прописан: {our_packages_dir / PTH_NAME}")

    if not args.skip_install:
        print("ставлю недостающие пакеты")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--quiet", *OUR_PACKAGES], check=True
        )

    verify = (
        "import torch, transformers, fastapi, skimage; "
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
        "print('transformers', transformers.__version__)"
    )
    subprocess.run([str(venv_python), "-c", verify], check=True)

    print(
        "\nготово. Запуск:\n"
        f"    MODEL_BACKEND=segformer MODEL_DEVICE=cuda "
        f"{venv_python} -m uvicorn app.main:app --port 8001\n"
        "\nОкружение зависит от чужой установки: если донора удалят или обновят,\n"
        "сервис перестанет находить torch. Для продакшена ставьте torch честно."
    )


if __name__ == "__main__":
    sys.exit(main())
