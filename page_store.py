"""lidarv2 page store — every GUI page lives in an mmap, not a Python string.

`PageStore` memory-maps each file under `gui_pages/` (HTML fragments and JS) and
serves bytes straight from the mapping. The frontend is therefore 100%
modular: the shell loads one fragment per page from `/page/<name>`.
"""

import mmap
import os
import threading


class PageStore:
    def __init__(self, pages_dir):
        self.dir = os.path.abspath(pages_dir)
        self._lock = threading.RLock()
        self._maps = {}
        self.reload()

    # ── lifecycle ──────────────────────────────────────────────────────
    def reload(self):
        with self._lock:
            self._close_all()
            for rel, full in self._walk():
                self._map(rel, full)
            return sorted(self._maps)

    def _walk(self):
        for root, _dirs, files in os.walk(self.dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, self.dir).replace(os.sep, "/")
                yield rel, full

    def _map(self, name, full):
        size = os.path.getsize(full)
        if size == 0:
            self._maps[name] = (None, b"")
            return
        fh = open(full, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps[name] = (fh, mm)

    def _close_all(self):
        for fh, mm in self._maps.values():
            if mm is not None and not isinstance(mm, bytes):
                try:
                    mm.close()
                except (BufferError, ValueError):
                    pass
            if fh is not None:
                fh.close()
        self._maps.clear()

    def close(self):
        with self._lock:
            self._close_all()

    # ── access ─────────────────────────────────────────────────────────
    def get(self, name):
        name = name.lstrip("/")
        with self._lock:
            entry = self._maps.get(name)
            if entry is None:
                return None
            _fh, mm = entry
            if isinstance(mm, bytes):
                return mm
            return mm[:]

    def has(self, name):
        with self._lock:
            return name.lstrip("/") in self._maps

    def list(self):
        with self._lock:
            return sorted(self._maps)

    def size(self, name):
        data = self.get(name)
        return None if data is None else len(data)
