"""
Временное хранилище артефактов инференса (оверлей, размеченный кадр).

Осознанно в памяти процесса: картинка нужна ровно на время, пока модератор
смотрит карточку кандидата. Класть её в S3 или в БД — это состояние, которое
надо чистить, бэкапить и согласовывать; сервис должен оставаться stateless и
горизонтально масштабируемым. Если ответ понадобился надолго, клиент забирает
PNG сразу и хранит у себя.

При нескольких репликах за балансировщиком ссылка на артефакт может уйти не в
ту реплику — поэтому она помечена как best-effort, а сами детекции всегда
приходят в JSON и от артефакта не зависят.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    content: bytes
    media_type: str
    created_at: float


class ArtifactStore:
    """LRU с TTL. Потокобезопасен: uvicorn отдаёт sync-эндпоинты в пул потоков."""

    def __init__(self, *, ttl_seconds: int, max_items: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, Artifact] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, content: bytes, media_type: str) -> None:
        with self._lock:
            self._evict_expired()
            self._items[key] = Artifact(content, media_type, time.monotonic())
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, key: str) -> Artifact | None:
        with self._lock:
            self._evict_expired()
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def _evict_expired(self) -> None:
        deadline = time.monotonic() - self.ttl_seconds
        for key in [k for k, v in self._items.items() if v.created_at < deadline]:
            del self._items[key]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
