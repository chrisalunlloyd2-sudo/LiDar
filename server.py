#!/usr/bin/env python3
"""lidarv2 backend — emulative LiDAR + spatial inventory (pure stdlib).

Serves the camera frontend and a JSON API:

  GET  /                     camera frontend
  GET  /api/health           liveness
  GET  /api/config           version + tag catalog
  POST /api/observe          one detected object -> depth, CI, illusions, geo, item
  POST /api/scan             ground-plane point cloud (emulated LiDAR sweep)
  POST /api/demo             seed a synthetic room so the UI is non-empty
  GET  /api/map              accepted map points
  GET  /api/inventory        tagged inventory + summary
  POST /api/reset            clear map + inventory

Run:  python3 server.py [--port 8790] [--db lidarv2.db]
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from depth_engine import DepthEngine  # noqa: E402
from geospatial import scan_ground  # noqa: E402
from illusion_filter import analyze as analyze_illusions  # noqa: E402
from inventory import (  # noqa: E402
    TAG_CATALOG,
    classify_observation,
    emulated_dimensions_mm,
    fill_thickness,
)
from models import CameraIntrinsics, CameraPose, Observation  # noqa: E402
from store import Store  # noqa: E402

VERSION = "2.0.0"
STATIC_DIR = os.path.join(HERE, "static")
_DEFAULT_INTR = CameraIntrinsics.from_fov(720, 1280, 60.0)


def _opt_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_intrinsics(payload):
    intr = payload.get("intrinsics") or {}
    image_w = int(intr.get("image_w") or payload.get("image_w") or 720)
    image_h = int(intr.get("image_h") or payload.get("image_h") or 1280)
    fov = float(intr.get("fov_deg") or payload.get("fov_deg") or 60.0)
    if intr.get("f_px"):
        return CameraIntrinsics(
            f_px=float(intr["f_px"]),
            cx=float(intr.get("cx", image_w / 2.0)),
            cy=float(intr.get("cy", image_h / 2.0)),
            image_w=image_w,
            image_h=image_h,
            fov_deg=fov,
        )
    return CameraIntrinsics.from_fov(image_w, image_h, fov)


def parse_pose(payload):
    p = payload.get("pose") or {}
    if not p:
        return None
    return CameraPose(
        lat=float(p.get("lat", 0.0)),
        lon=float(p.get("lon", 0.0)),
        alt_m=float(p.get("alt_m", 0.0)),
        heading_deg=float(p.get("heading_deg", 0.0)),
        pitch_deg=float(p.get("pitch_deg", 0.0)),
        roll_deg=float(p.get("roll_deg", 0.0)),
        accuracy_m=float(p.get("accuracy_m", 8.0)),
        has_gps=bool(p.get("has_gps", False)),
    )


def parse_observation(payload):
    intr = parse_intrinsics(payload)
    obs = Observation(
        object_px_width=float(payload.get("object_px_width") or 0.0),
        object_px_height=float(payload.get("object_px_height") or 0.0),
        center_x=float(payload.get("center_x", intr.cx)),
        center_y=float(payload.get("center_y", intr.cy)),
        intrinsics=intr,
        pose=parse_pose(payload),
        known_width_mm=_opt_float(payload.get("known_width_mm")),
        object_class=str(payload.get("object_class") or "unknown"),
        camera_height_m=float(payload.get("camera_height_m") or 1.5),
        texture_density=_opt_float(payload.get("texture_density")),
        sharpness=_opt_float(payload.get("sharpness")),
        convergence=_opt_float(payload.get("convergence")),
        arrowhead=_opt_float(payload.get("arrowhead")),
        specular=_opt_float(payload.get("specular")),
        contrast=_opt_float(payload.get("contrast")),
        room=str(payload.get("room") or "unknown"),
        timestamp=float(payload.get("timestamp") or time.time()),
    )
    return obs


def process_observation(obs, store, engine, level=0.95, session="default"):
    """Run the full inference pipeline for one observation."""
    cues = engine.cues(obs)
    if not cues:
        return {"error": "no depth cues (need known_width_mm and/or ground geometry)"}

    estimate = engine.estimate(obs, level=level)
    illusions = analyze_illusions(obs, cues)
    dims = emulated_dimensions_mm(obs, estimate.depth_m)
    tag = classify_observation(obs, dims[:2])
    dims = fill_thickness(dims, tag)

    geo = None
    if obs.pose is not None and obs.pose.has_gps:
        from geospatial import project_point

        geo = project_point(
            obs.pose, obs.intrinsics, obs.center_x, obs.center_y,
            estimate.depth_m, sigma_depth_m=estimate.sigma_m,
        )

    store.add_map_point(
        ts=obs.timestamp, session=session, room=obs.room,
        lat=geo.lat if geo else 0.0, lon=geo.lon if geo else 0.0,
        alt_m=geo.alt_m if geo else 0.0, radius_m=geo.radius_m if geo else 0.0,
        east_m=geo.east_m if geo else 0.0, north_m=geo.north_m if geo else 0.0,
        up_m=geo.up_m if geo else 0.0,
        tag=tag["tag"], depth_m=estimate.depth_m,
        ci_low_m=estimate.ci_low_m, ci_high_m=estimate.ci_high_m,
        confidence=estimate.confidence,
        rejected=illusions.rejected,
        flags=json.dumps([f.code for f in illusions.flags]),
    )

    item = None
    accepted = not illusions.rejected
    if accepted and tag["tag"] != "unknown" and geo is not None:
        item = store.upsert_item(
            tag=tag["tag"], label=tag["label"], room=obs.room,
            w_mm=dims[0], h_mm=dims[1], d_mm=dims[2],
            lat=geo.lat, lon=geo.lon, alt_m=geo.alt_m,
            radius_m=geo.radius_m, confidence=estimate.confidence,
        )

    return {
        "accepted": accepted,
        "room": obs.room,
        "estimate": asdict(estimate),
        "illusions": asdict(illusions),
        "dimensions_mm": {"w": dims[0], "h": dims[1], "d": dims[2]},
        "tag": tag,
        "geo": asdict(geo) if geo else None,
        "item": item,
    }


def _demo_observation(intr, heading_deg, known_width_mm, depth_m, tag, room,
                      camera_height_m=1.5):
    f = intr.f_px
    px_w = known_width_mm * f / (depth_m * 1000.0)
    px_h = max(4.0, px_w * 0.45)
    # Put the object's bottom edge on the floor so the ground cue agrees with
    # the known-size cue instead of manufacturing a false conflict.
    depression = math.atan(camera_height_m / depth_m)
    v_bottom = intr.cy + f * math.tan(depression)
    center_y = max(px_h / 2.0, min(intr.image_h - px_h / 2.0, v_bottom - px_h / 2.0))
    pose = CameraPose(
        lat=45.0, lon=10.0, alt_m=50.0, heading_deg=heading_deg,
        pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True,
    )
    return Observation(
        object_px_width=px_w, object_px_height=px_h,
        center_x=intr.cx, center_y=center_y,
        intrinsics=intr, pose=pose, known_width_mm=known_width_mm,
        object_class=tag, camera_height_m=camera_height_m,
        texture_density=0.4, contrast=0.45,
        room=room,
    )


class LidarHandler(BaseHTTPRequestHandler):
    server_version = "lidarv2/" + VERSION

    # ── helpers ────────────────────────────────────────────────────────
    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _serve_file(self, rel):
        safe = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not safe.startswith(STATIC_DIR) or not os.path.isfile(safe):
            return self._send(404, {"error": "not found"})
        ctype = "text/html; charset=utf-8" if safe.endswith(".html") else "application/octet-stream"
        with open(safe, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # ── GET ────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._serve_file("index.html")
        if path.startswith("/static/"):
            return self._serve_file(path[len("/static/"):])
        if path == "/api/health":
            return self._send(200, {"status": "ok", "version": VERSION, "db": self.server.store.health()})
        if path == "/api/config":
            return self._send(200, {
                "version": VERSION,
                "tags": TAG_CATALOG,
                "defaults": {"fov_deg": 60.0, "camera_height_m": 1.5, "level": 0.95},
            })
        if path == "/api/map":
            q = parse_qs(parsed.query)
            points = self.server.store.list_map_points(
                room=_first(q, "room"), tag=_first(q, "tag"),
                limit=int(_first(q, "limit") or 500),
                include_rejected=bool(_first(q, "include_rejected")),
            )
            return self._send(200, {"count": len(points), "points": points})
        if path == "/api/inventory":
            q = parse_qs(parsed.query)
            items = self.server.store.list_items(room=_first(q, "room"), tag=_first(q, "tag"))
            return self._send(200, {"count": len(items), "items": items,
                                    "summary": self.server.store.summary()})
        return self._send(404, {"error": "not found"})

    # ── POST ───────────────────────────────────────────────────────────
    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": "bad json: %s" % exc})
        try:
            if parsed.path == "/api/observe":
                return self._handle_observe(payload)
            if parsed.path == "/api/scan":
                return self._handle_scan(payload)
            if parsed.path == "/api/demo":
                return self._handle_demo(payload)
            if parsed.path == "/api/reset":
                self.server.store.reset()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as exc:  # keep the API returning JSON on any failure
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def _handle_observe(self, payload):
        obs = parse_observation(payload)
        result = process_observation(
            obs, self.server.store, self.server.engine,
            level=float(payload.get("level") or 0.95),
            session=str(payload.get("session") or "default"),
        )
        return self._send(200 if "error" not in result else 422, result)

    def _handle_scan(self, payload):
        intr = parse_intrinsics(payload)
        pose = parse_pose(payload)
        if pose is None:
            pose = CameraPose()
        camera_height = float(payload.get("camera_height_m") or 1.5)
        cols = int(payload.get("cols") or 24)
        rows = int(payload.get("rows") or 16)
        points = scan_ground(pose, intr, camera_height, cols=cols, rows=rows)
        if payload.get("store", True):
            room = str(payload.get("room") or "unknown")
            now = time.time()
            for gp in points:
                self.server.store.add_map_point(
                    ts=now, session=str(payload.get("session") or "scan"), room=room,
                    lat=gp.lat, lon=gp.lon, alt_m=gp.alt_m, radius_m=gp.radius_m,
                    east_m=gp.east_m, north_m=gp.north_m, up_m=gp.up_m,
                    tag="ground", depth_m=gp.distance_m,
                    ci_low_m=gp.distance_m, ci_high_m=gp.distance_m,
                    confidence=0.5, rejected=0, flags="[]",
                )
        return self._send(200, {"count": len(points), "points": [asdict(p) for p in points]})

    def _handle_demo(self, payload):
        intr = parse_intrinsics(payload)
        room = str(payload.get("room") or "basement B")
        seeds = [
            ("2x4", 88.9, 3.0, 80.0),
            ("2x6", 139.7, 4.5, 105.0),
            ("paint_can", 168.0, 2.2, 60.0),
            ("box", 400.0, 5.0, 130.0),
        ]
        results = []
        for tag, width_mm, depth_m, heading in seeds:
            obs = _demo_observation(intr, heading, width_mm, depth_m, tag, room)
            results.append(process_observation(
                obs, self.server.store, self.server.engine, level=0.95, session="demo",
            ))
        return self._send(200, {
            "seeded": len(results),
            "results": results,
            "inventory": self.server.store.list_items(room=room),
            "summary": self.server.store.summary(),
        })

    def log_message(self, fmt, *args):
        pass


def _first(query, key):
    values = query.get(key)
    return values[0] if values else None


def build_server(host="0.0.0.0", port=8790, db_path=":memory:"):
    store = Store(db_path)
    httpd = ThreadingHTTPServer((host, port), LidarHandler)
    httpd.store = store
    httpd.engine = DepthEngine()
    return httpd


def main():
    parser = argparse.ArgumentParser(description="lidarv2 emulative LiDAR backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--db", default=os.path.join(HERE, "lidarv2.db"))
    parser.add_argument("--memory", action="store_true", help="use an in-memory database")
    args = parser.parse_args()

    db_path = ":memory:" if args.memory else args.db
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = build_server(args.host, args.port, db_path)

    print("lidarv2 %s — emulative LiDAR backend" % VERSION)
    print("  serving : http://%s:%d" % (args.host, args.port))
    print("  database: %s" % db_path)
    print("  open the URL in Chrome on the phone, allow camera + location")
    print("  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.store.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
