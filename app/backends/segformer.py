"""
Партнёрская модель «Чистый берег» (Yandex DataSphere), SegFormer B2.

Веса: https://storage.yandexcloud.net/socialtech/garbage-detect/seg-model/model.pt
Это `state_dict`, а не сериализованный модуль, поэтому архитектура собирается
отдельно, а веса грузятся поверх.

Три отличия от `model-server` апстрима, все сознательные:

1. **Архитектура из конфига, а не `from_pretrained("nvidia/mit-b2")`.**
   Апстрим на каждом старте тянет с Hugging Face чекпоинт mit-b2 (сотня
   мегабайт), чтобы тут же затереть его своим `state_dict`. Нам от него нужна
   только форма сети, и она выписана здесь константами. Итог: старт без сети,
   образ без кеша HF, сборка без ещё одной загрузки. Совпадение формы проверяется
   жёстко — `load_state_dict(strict=True)` падает при любом расхождении.

2. **Препроцессинг в numpy, без `SegformerImageProcessor`.** Процессор делает
   ровно три вещи: ресайз до 1024, деление на 255 и нормализацию по ImageNet.
   Ради них тянуть `torchvision` в образ незачем.

3. **Тайлинг с перекрытием и усреднением логитов.** У апстрима при
   `stride < tile` соседний тайл затирал уже посчитанный кусок, из-за чего на
   границах тайлов рвались объекты; там же индексы тайлов считались по чужой
   оси (`row_num = index // n`, где `n` — число тайлов по высоте), что
   промахивается на неквадратных кадрах.

Torch импортируется внутри модуля: без выбора этого бэкенда он не нужен.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL.Image import Image as PILImage

from app.backends.base import BackendInfo, Segmentation
from app.taxonomy import ID2LABEL, LABEL2ID

VERSION = "0.1.0"

WEIGHTS_URL = "https://storage.yandexcloud.net/socialtech/garbage-detect/seg-model/model.pt"

#: Размер тайла. Значение апстрима: модель обучалась на 1024 px, менять нельзя.
TILE_PX = 1024

#: Шаг тайлов. 512 (половина тайла) — как у апстрима: каждый пиксель попадает
#: в четыре тайла, границы усредняются, швов не видно. Но и считается вчетверо
#: больше. `STRIDE_PX = TILE_PX` даёт ускорение в 4 раза ценой возможных швов
#: на границах — для предварительного прохода по большой площади это разумный
#: размен, поэтому шаг вынесен в переменную окружения.
STRIDE_PX = int(os.environ.get("MODEL_STRIDE_PX", TILE_PX // 2))

#: Сколько тайлов уходит в сеть за один вызов.
#:
#: Замерено на RTX 4060 Ti (16 ГБ), кадр 2688x1792: батчи 1, 2, 4 и 6 дают
#: одинаковые 2.55-2.67 с — тайл 1024x1024 сам по себе загружает карту целиком,
#: и батчить нечего. А вот батч 8 обваливается до 17.7 с: первая стадия
#: SegFormer держит внимание на 65536 токенов при 8192 ключах, и на восьми
#: тайлах эта матрица перестаёт помещаться в память, начинается своп.
#:
#: Поэтому по умолчанию 2, а не «побольше»: выигрыша от большого батча нет,
#: а обрыв есть. Увеличивать имеет смысл только на карте с заметно большим
#: объёмом памяти и только вместе с замером.
BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", 0))

#: cuda / cpu / auto. Явное значение нужно, когда на хосте есть GPU, но занят.
DEVICE = os.environ.get("MODEL_DEVICE", "auto")

#: Половинная точность на GPU. Сегментация к ней устойчива: argmax по семи
#: классам не меняется от четвёртого знака. Выигрыш скромнее ожидаемого —
#: 2.92 с против 2.55 с на том же кадре, то есть около 13 %, а не вдвое:
#: заметная часть времени уходит не на матмулы, а на интерполяцию логитов и
#: перегон в память хоста.
USE_FP16 = os.environ.get("MODEL_FP16", "1") == "1"

#: Нормализация ImageNet — та же, что зашита в `SegformerImageProcessor`.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Форма SegFormer B2 (`nvidia/mit-b2`). Менять нельзя: под неё обучены веса,
#: и `strict=True` при загрузке это проверит.
MIT_B2 = {
    "num_channels": 3,
    "num_encoder_blocks": 4,
    "depths": [3, 4, 6, 3],
    "sr_ratios": [8, 4, 2, 1],
    "hidden_sizes": [64, 128, 320, 512],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "num_attention_heads": [1, 2, 5, 8],
    "mlp_ratios": [4, 4, 4, 4],
    "hidden_act": "gelu",
    "decoder_hidden_size": 768,
    "classifier_dropout_prob": 0.1,
    "drop_path_rate": 0.1,
}


class SegformerDetector:
    """Инференс партнёрской модели. Веса читаются один раз при создании."""

    def __init__(self, weights_path: str | os.PathLike[str] | None = None) -> None:
        import torch  # noqa: PLC0415 — зависимость только этого бэкенда
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        self._torch = torch
        self.weights_path = Path(weights_path or os.environ.get("MODEL_WEIGHTS", "model.pt"))
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Веса не найдены: {self.weights_path}. "
                f"Скачать: python scripts/download_model.py (источник {WEIGHTS_URL})"
            )

        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)
        self.fp16 = USE_FP16 and self.device.type == "cuda"
        self.batch_size = BATCH_SIZE or (2 if self.device.type == "cuda" else 1)

        # TF32 на матмулах: для инференса сегментации разницы в ответе нет,
        # а на Ada это заметно быстрее fp32.
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        config = SegformerConfig(
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            **MIT_B2,
        )
        self.model = SegformerForSemanticSegmentation(config)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        if self.device.type == "cuda":
            # channels_last — раскладка, под которую написаны свёрточные ядра
            # тензорных блоков; на CPU она, наоборот, обычно вредит.
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="segformer",
            version=VERSION,
            weights=str(self.weights_path),
            device=f"{self.device}{' fp16' if self.fp16 else ''} batch={self.batch_size}",
            tile_px=TILE_PX,
            stride_px=STRIDE_PX,
            trained=True,
            notes=(
                "Модель проекта «Чистый берег», Yandex DataSphere, SegFormer B2. "
                "Дообучение не проводилось: используется как есть."
            ),
        )

    def _preprocess(self, tile: np.ndarray) -> np.ndarray:
        """Тайл -> (3, TILE, TILE), нормализован как при обучении.

        Недостающее до полного тайла добивается нулями, а не растягивается
        ресайзом: у ресайза меняется масштаб объектов, а модель обучена на
        конкретном сантиметраже пикселя — растянутый край кадра она видит как
        другую съёмку. Нулевой паддинг после нормализации — это средний серый,
        сеть на нём ничего не находит, и лишних объектов он не создаёт.
        """
        arr = (tile.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        h, w = arr.shape[:2]
        if (h, w) != (TILE_PX, TILE_PX):
            padded = np.zeros((TILE_PX, TILE_PX, 3), dtype=np.float32)
            padded[:h, :w] = arr
            arr = padded
        return arr.transpose(2, 0, 1)

    def _run_batch(self, tiles: list[np.ndarray]) -> np.ndarray:
        """Батч тайлов -> логиты (N, C, TILE, TILE) в исходном разрешении тайла."""
        torch = self._torch
        batch = np.stack([self._preprocess(tile) for tile in tiles])
        pixel_values = torch.from_numpy(batch).to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            if self.fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(pixel_values=pixel_values).logits
            else:
                logits = self.model(pixel_values=pixel_values).logits
            # SegFormer отдаёт логиты в 1/4 разрешения входа — поднимаем их
            # ещё на GPU: интерполяция там дешевле, чем перегон в память.
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(TILE_PX, TILE_PX), mode="bilinear", align_corners=False
            )
        return logits.cpu().numpy()

    def segment(self, image: PILImage) -> Segmentation:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = arr.shape[:2]
        n_classes = len(ID2LABEL)

        acc = np.zeros((n_classes, height, width), dtype=np.float32)
        weight = np.zeros((height, width), dtype=np.float32)

        boxes = [
            (y0, x0, min(y0 + TILE_PX, height), min(x0 + TILE_PX, width))
            for y0 in _starts(height, TILE_PX, STRIDE_PX)
            for x0 in _starts(width, TILE_PX, STRIDE_PX)
        ]

        for start in range(0, len(boxes), self.batch_size):
            chunk = boxes[start : start + self.batch_size]
            logits = self._run_batch([arr[y0:y1, x0:x1] for y0, x0, y1, x1 in chunk])
            for (y0, x0, y1, x1), tile_logits in zip(chunk, logits):
                # Обрезаем паддинг: за пределами тайла логитов не существует.
                acc[:, y0:y1, x0:x1] += tile_logits[:, : y1 - y0, : x1 - x0]
                weight[y0:y1, x0:x1] += 1.0

        acc /= np.maximum(weight, 1.0)

        # Softmax по классам вручную: тащить torch ради одной операции незачем.
        shifted = acc - acc.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        probs_all = exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-9)

        labels = probs_all.argmax(axis=0).astype(np.uint8)
        probs = probs_all.max(axis=0).astype(np.float32)
        # На фоне уверенность не нужна — она путает агрегаты по объектам.
        probs[labels == 0] = 0.0
        return Segmentation(labels=labels, probs=probs)


def _starts(size: int, tile: int, stride: int) -> list[int]:
    """Начала тайлов вдоль оси. Последний тайл прижимается к краю кадра."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts
