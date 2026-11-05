"""lidarv2 store — SQLite persistence for map points and inventory.

Pure stdlib (`sqlite3`, `threading`). One connection guarded by a lock, so
the ThreadingHTTPServer can share it safely. Tests use `:memory:`.
"""

import sqlite3
import threading
import time

from geospatial import haversine_m

SCHEMA = """
CREATE TABLE IF NOT EXISTS map_points (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, session TEXT, room TEXT,
  lat REAL, lon REAL, alt_m REAL, radius_m REAL,
  east_m REAL, north_m REAL, up_m REAL,
  tag TEXT, depth_m REAL, ci_low_m REAL, ci_high_m REAL,
  confidence REAL, rejected INTEGER, flags TEXT
);
CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tag TEXT, label TEXT, room TEXT,
  count INTEGER DEFAULT 1,
  w_mm REAL, h_mm REAL, d_mm REAL,
  lat REAL, lon REAL, alt_m REAL, radius_m REAL,
  confidence REAL, first_seen REAL, last_seen REAL
);
CREATE INDEX IF NOT EXISTS idx_items_tag ON items(tag);
CREATE INDEX IF NOT EXISTS idx_items_room ON items(room);
CREATE INDEX IF NOT EXISTS idx_map_room ON map_points(room);
CREATE INDEX IF NOT EXISTS idx_map_tag ON map_points(tag);
"""

_MAP_FIELDS = (
    "ts", "session", "room", "lat", "lon", "alt_m", "radius_m",
    "east_m", "north_m", "up_m", "tag", "depth_m", "ci_low_m",
    "ci_high_m", "confidence", "rejected", "flags",
)


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ── map points ─────────────────────────────────────────────────────
    def add_map_point(self, **kw):
        now = kw.pop("ts", None) or time.time()
        values = {f: kw.get(f) for f in _MAP_FIELDS}
        values["ts"] = now
        values["rejected"] = 1 if values["rejected"] else 0
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO map_points (%s) VALUES (%s)"
                % (", ".join(_MAP_FIELDS), ", ".join("?" * len(_MAP_FIELDS))),
                tuple(values[f] for f in _MAP_FIELDS),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_map_points(self, room=None, tag=None, limit=500, include_rejected=False):
        clauses, params = [], []
        if room:
            clauses.append("room = ?")
            params.append(room)
        if tag:
            clauses.append("tag = ?")
            params.append(tag)
        if not include_rejected:
            clauses.append("rejected = 0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM map_points%s ORDER BY id DESC LIMIT ?" % where, params
            ).fetchall()
        return [dict(r) for r in rows]

    # ── inventory ──────────────────────────────────────────────────────
    def get_item(self, item_id):
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return dict(row) if row else None

    def upsert_item(self, tag, label, room, w_mm, h_mm, d_mm, lat, lon, alt_m,
                    radius_m, confidence, merge_radius_m=2.0, now=None):
        """Merge into an existing item when tag + room + place all line up."""
        now = now or time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM items WHERE tag = ? AND room = ?", (tag, room)
            ).fetchall()
            for r in rows:
                same_place = haversine_m(r["lat"], r["lon"], lat, lon) <= merge_radius_m
                if not same_place:
                    continue
                n = int(r["count"]) + 1
                nw = _blend(r["w_mm"], w_mm, n)
                nh = _blend(r["h_mm"], h_mm, n)
                nd = _blend(r["d_mm"], d_mm, n)
                conf = (float(r["confidence"] or 0.0) * (n - 1) + float(confidence)) / n
                self._conn.execute(
                    "UPDATE items SET count=?, w_mm=?, h_mm=?, d_mm=?, lat=?, lon=?, "
                    "alt_m=?, radius_m=?, confidence=?, last_seen=? WHERE id=?",
                    (n, nw, nh, nd, lat, lon, alt_m,
                     max(float(r["radius_m"] or 0.0), float(radius_m)), conf, now, r["id"]),
                )
                self._conn.commit()
                return self.get_item(r["id"])
            cur = self._conn.execute(
                "INSERT INTO items (tag, label, room, count, w_mm, h_mm, d_mm, lat, lon, "
                "alt_m, radius_m, confidence, first_seen, last_seen) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tag, label, room, 1, w_mm, h_mm, d_mm, lat, lon, alt_m,
                 radius_m, confidence, now, now),
            )
            self._conn.commit()
            return self.get_item(cur.lastrowid)

    def list_items(self, room=None, tag=None):
        clauses, params = [], []
        if room:
            clauses.append("room = ?")
            params.append(room)
        if tag:
            clauses.append("tag = ?")
            params.append(tag)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM items%s ORDER BY tag, room, id" % where, params
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self):
        with self._lock:
            by_tag = self._conn.execute(
                "SELECT tag, SUM(count) AS n FROM items GROUP BY tag ORDER BY n DESC"
            ).fetchall()
            by_room = self._conn.execute(
                "SELECT room, SUM(count) AS n FROM items GROUP BY room ORDER BY n DESC"
            ).fetchall()
            total = self._conn.execute("SELECT COALESCE(SUM(count), 0) AS n FROM items").fetchone()
            points = self._conn.execute("SELECT COUNT(*) AS n FROM map_points").fetchone()
        return {
            "total_items": int(total["n"]),
            "map_points": int(points["n"]),
            "by_tag": [dict(r) for r in by_tag],
            "by_room": [dict(r) for r in by_room],
        }

    # ── lifecycle ──────────────────────────────────────────────────────
    def reset(self):
        with self._lock:
            self._conn.execute("DELETE FROM map_points")
            self._conn.execute("DELETE FROM items")
            self._conn.commit()

    def health(self):
        with self._lock:
            self._conn.execute("SELECT 1").fetchone()
        return {"ok": True, "path": self.path}

    def close(self):
        with self._lock:
            self._conn.close()


def _blend(old, new, n):
    """Running mean that tolerates either side being None."""
    if new is None:
        return old
    if old is None:
        return new
    return (float(old) * (n - 1) + float(new)) / n
