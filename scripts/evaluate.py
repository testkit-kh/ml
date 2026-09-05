"""
Оценка конвейера на тестовых данных: IoU по классам, precision/recall объектов.

Считает две разные вещи, и их нельзя путать:

* **Пиксельные метрики** (IoU, precision, recall по маске) — насколько точно
  обведён мусор. Это качество сегментации.
* **Объектные метрики** — сколько реальных объектов найдено и сколько находок
  ложные, при сопоставлении по IoU >= порога. Это то, что увидит модератор в
  очереди: пропущенный объект — недособранный мусор, ложный — потраченное
  время сотрудника ООПТ. Именно объектная precision идёт в KPI автодетекции.

Запуск:
    PYTHONPATH=. python scripts/evaluate.py --data testdata
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app import masks
from app.backends import get_detector
from app.taxonomy import ID2LABEL, LABEL_RU

#: Порог IoU, при котором находка считается тем же объектом, что и эталон.
MATCH_IOU = 0.3


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def evaluate(data_dir: Path, backend: str, *, min_area_px: int = 80) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    detector = get_detector(backend)

    pixel = defaultdict(lambda: {"inter": 0, "union": 0, "pred": 0, "true": 0})
    objects = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    class_agnostic = {"tp": 0, "fp": 0, "fn": 0}
    confusion: dict[tuple[int, int], int] = defaultdict(int)

    for entry in manifest["frames"]:
        image = Image.open(data_dir / entry["frame"]).convert("RGB")
        truth = np.asarray(Image.open(data_dir / entry["truth"]))
        predicted = detector.segment(image).labels

        for class_id in ID2LABEL:
            if class_id == 0:
                continue
            t = truth == class_id
            p = predicted == class_id
            row = pixel[class_id]
            row["inter"] += int(np.logical_and(t, p).sum())
            row["union"] += int(np.logical_or(t, p).sum())
            row["pred"] += int(p.sum())
            row["true"] += int(t.sum())

        shape = truth.shape
        truth_blobs = masks.extract_blobs(truth, min_area_px=min_area_px)
        pred_blobs = masks.extract_blobs(predicted, min_area_px=min_area_px)
        matched: set[int] = set()

        for pred in pred_blobs:
            best_idx, best_iou = -1, 0.0
            for idx, gt in enumerate(truth_blobs):
                if idx in matched:
                    continue
                score = iou(pred.full_mask(shape), gt.full_mask(shape))
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_iou >= MATCH_IOU:
                matched.add(best_idx)
                gt = truth_blobs[best_idx]
                class_agnostic["tp"] += 1
                confusion[(gt.class_id, pred.class_id)] += 1
                if gt.class_id == pred.class_id:
                    objects[pred.class_id]["tp"] += 1
                else:
                    objects[pred.class_id]["fp"] += 1
                    objects[gt.class_id]["fn"] += 1
            else:
                class_agnostic["fp"] += 1
                objects[pred.class_id]["fp"] += 1

        for idx, gt in enumerate(truth_blobs):
            if idx not in matched:
                class_agnostic["fn"] += 1
                objects[gt.class_id]["fn"] += 1

    return {
        "backend": backend,
        "frames": len(manifest["frames"]),
        "pixel": {k: dict(v) for k, v in pixel.items()},
        "objects": {k: dict(v) for k, v in objects.items()},
        "class_agnostic": class_agnostic,
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
    }


def _ratio(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def report(result: dict) -> str:
    lines = [
        f"бэкенд: {result['backend']}   кадров: {result['frames']}",
        "",
        f"{'класс':<16}{'IoU':>7}{'P(px)':>8}{'R(px)':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'P(об)':>8}{'R(об)':>8}",
        "-" * 70,
    ]
    for class_id, row in sorted(result["pixel"].items(), key=lambda kv: int(kv[0])):
        class_id = int(class_id)
        obj = result["objects"].get(class_id, result["objects"].get(str(class_id), {}))
        tp, fp, fn = obj.get("tp", 0), obj.get("fp", 0), obj.get("fn", 0)
        lines.append(
            f"{LABEL_RU[class_id]:<16}"
            f"{_ratio(row['inter'], row['union']):>7}"
            f"{_ratio(row['inter'], row['pred']):>8}"
            f"{_ratio(row['inter'], row['true']):>8}"
            f"{tp:>5}{fp:>5}{fn:>5}"
            f"{_ratio(tp, tp + fp):>8}{_ratio(tp, tp + fn):>8}"
        )

    ca = result["class_agnostic"]
    lines += [
        "-" * 70,
        "«найден объект» без учёта класса — это и есть работа для модератора:",
        f"  precision {_ratio(ca['tp'], ca['tp'] + ca['fp'])}   "
        f"recall {_ratio(ca['tp'], ca['tp'] + ca['fn'])}   "
        f"(TP {ca['tp']}, FP {ca['fp']}, FN {ca['fn']})",
    ]
    if result["confusion"]:
        lines += ["", "путаница классов (эталон -> предсказание):"]
        for key, count in result["confusion"].items():
            a, b = (int(x) for x in key.split("->"))
            mark = "  ok" if a == b else "  !!"
            lines.append(f"{mark} {ID2LABEL[a]:>14} -> {ID2LABEL[b]:<14} {count}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики на тестовых данных")
    parser.add_argument("--data", type=Path, default=Path("testdata"))
    parser.add_argument("--backend", default="heuristic")
    parser.add_argument("--json", type=Path, default=None, help="куда сложить сырые числа")
    args = parser.parse_args()

    result = evaluate(args.data, args.backend)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсырые метрики -> {args.json}")


if __name__ == "__main__":
    main()
