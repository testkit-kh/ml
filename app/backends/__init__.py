"""Реализации сегментации. Выбор — через `get_detector()`, не импортом напрямую."""

from __future__ import annotations

import threading

from app.backends.base import BackendInfo, Detector, Segmentation

__all__ = ["BackendInfo", "Detector", "Segmentation", "get_detector", "reset_detector"]

_lock = threading.Lock()
_cache: dict[str, Detector] = {}


def _build(name: str) -> Detector:
    if name == "heuristic":
        from app.backends.heuristic import HeuristicDetector

        return HeuristicDetector()
    if name == "segformer":
        # Импорт внутри ветки: torch и transformers не нужны, пока бэкенд не выбран.
        from app.backends.segformer import SegformerDetector

        return SegformerDetector()
    raise ValueError(f"неизвестный бэкенд сегментации: {name!r}")


def get_detector(name: str) -> Detector:
    """Ленивая синглтон-загрузка. Веса читаются с диска один раз на процесс."""
    detector = _cache.get(name)
    if detector is not None:
        return detector
    with _lock:
        if name not in _cache:
            _cache[name] = _build(name)
        return _cache[name]


def reset_detector(name: str | None = None) -> None:
    """Сброс кэша — нужен тестам и горячей смене весов."""
    with _lock:
        if name is None:
            _cache.clear()
        else:
            _cache.pop(name, None)
