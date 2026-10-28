from __future__ import annotations

import threading
from collections import OrderedDict

from .models import AnyPiiResponse


class ResultCache:
    def __init__(self, max_entries: int, max_bytes: int):
        self.max_entries = max(0, max_entries)
        self.max_bytes = max(0, max_bytes)
        self._entries: OrderedDict[str, tuple[AnyPiiResponse, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> AnyPiiResponse | None:
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            self._entries.move_to_end(key)
            response, _ = item
            return response.model_copy(deep=True)

    def put(self, key: str, response: AnyPiiResponse) -> None:
        if self.max_entries <= 0 or self.max_bytes <= 0:
            return
        payload = response.model_dump_json()
        size = len(payload.encode("utf-8"))
        if size > self.max_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            while self._entries and (
                len(self._entries) + 1 > self.max_entries or self._bytes + size > self.max_bytes
            ):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed[1]
            self._entries[key] = (response.model_copy(deep=True), size)
            self._bytes += size

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
