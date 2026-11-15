#!/usr/bin/env python3
"""lidarv2 backend — emulative LiDAR + spatial inventory (pure stdlib).

Serves an mmap-backed modular GUI and a JSON API:

  GET  /                    shell (gui_pages/shell.html, mmap)
  GET  /page/<name>         one GUI page fragment (mmap)
  GET  /api/health          liveness
  GET  /api/config          version + tag catalog
  GET  /api/pages           list of mapped GUI pages
  GET  /api/illusions       the canonical 5-illusion registry
  POST /api/observe         one observation -> depth+CI, 5 illusions, geo, item
  POST /api/scan            ground-plane point cloud (emulated LiDAR sweep)
  POST /api/fit             fit a rectangle/cuboid model to corners
  POST /api/triangulate     two-view triangulation (point or size)
  POST /api/premade         emulate a known premade object, report % error
  POST /api/benchmark       MAPE + CI coverage over all premade fixtures
  POST /api/plan            active view plan for a budget (FOV-capped)
  POST /api/flash           torch-pulse depth grid (demo scene or supplied grids)
  POST /api/flash/calibrate check an ambient/torch pair and suggest the gauge k
  POST /api/flash/observe   fuse the flash cue with the monocular depth cues
  POST /api/demo            seed a synthetic room
  GET  /api/map             accepted map points
  GET  /api/inventory       tagged inventory + summary
  POST /api/reset           clear map + inventory

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

from accuracy import percent_error, run_benchmark  # noqa: E402
from arena.arena import run_arena  # noqa: E402
from arena.hardware_probe import probe as probe_hardware  # noqa: E402
from arena.planner import DEFAULT_SIGMA_PX, plan_views  # noqa: E402
from depth_engine import DepthEngine  # noqa: E402
from flash_lidar import (  # noqa: E402
    FlashCalibration,
    analyse_grid,
    calibrate_from_grids,
    estimate_with_flash,
    run_demo as run_flash_demo,
    valid_fraction,
)
from fit_engine import fit_cuboid, fit_rectangle, fit_observation  # noqa: E402
from fixtures.premade_objects import by_tag, synthesize, truth_dims  # noqa: E402
from geospatial import project_point, scan_ground  # noqa: E402
from illusion_filter import analyze as analyze_illusions  # noqa: E402
from illusions import ILLUSION_REGISTRY, correct_size, evaluate as evaluate_illusions  # noqa: E402
from inventory import (  # noqa: E402
    TAG_CATALOG,
    classify_observation,
    emulated_dimensions_mm,
    fill_thickness,
)
from models import CameraIntrinsics, CameraPose, Observation  # noqa: E402
from page_store import PageStore  # noqa: E402
from shapes import classify_shape  # noqa: E402
from store import Store  # noqa: E402
from triangulate import triangulate_point, triangulate_size  # noqa: E402

VERSION = "2.5.3"
GUI_DIR = os.path.join(HERE, "gui_pages")
_DEFAULT_INTR = CameraIntrinsics.from_fov(720, 1280, 60.0)

# ── wire limits ──────────────────────────────────────────────────────
# The largest legitimate body is a full flash grid (~16k floats), so 4 MiB
# leaves ample room while keeping a hostile client from exhausting the heap.
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_MAP_LIMIT = 5000
MAX_SCAN_CELLS = 16384
SOCKET_TIMEOUT_S = 30


class PayloadTooLarge(ValueError):
    """Request body longer than MAX_BODY_BYTES."""


class BadRequest(ValueError):
    """A semantically invalid field in an otherwise well-formed request."""


_REQUIRED = object()


def _number(value, default=_REQUIRED, name="value", lo=None, hi=None):
    """Parse a finite request float, refusing non-numbers and out-of-range.

    Absent/empty values fall back to `default`; `default` may be `_REQUIRED`
    to make the field mandatory. JSON `true` is not a number.
    """
    if isinstance(value, bool):
        raise BadRequest("%s must be a number" % name)
    if value is None or value == "":
        if default is _REQUIRED:
            raise BadRequest("%s is required" % name)
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequest("%s must be a number" % name)
    if not math.isfinite(number):
        raise BadRequest("%s must be finite" % name)
    if lo is not None and number < lo:
        raise BadRequest("%s must be >= %g" % (name, lo))
    if hi is not None and number > hi:
        raise BadRequest("%s must be <= %g" % (name, hi))
    return number


def _opt_float(value, name="value", lo=None, hi=None):
    if value is None or value == "":
        return None
    return _number(value, None, name, lo=lo, hi=hi)


def _level(value, default=0.95):
    """Confidence level, bounded to the range the CI math actually uses."""
    return _number(value, default, "level", lo=0.01, hi=0.9999)


def _flag(value, default=False, name="flag"):
    """Strict JSON boolean: the string "false" is a client bug, not False."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    raise BadRequest("%s must be true or false" % name)


_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")

MAX_TEXT_LEN = 128


def _text(value, default="", name="value", max_len=MAX_TEXT_LEN):
    """A short printable label from the wire (room, session, tag, kind)."""
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise BadRequest("%s must be a string" % name)
    text = value.strip()
    if not text:
        return default
    if len(text) > max_len:
        raise BadRequest("%s must be at most %d characters" % (name, max_len))
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise BadRequest("%s must not contain control characters" % name)
    return text


def _query_flag(value, default=False, name="flag"):
    """Boolean from a query string; only the listed words are accepted."""
    if value is None or value == "":
        return default
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise BadRequest("%s must be one of: %s" % (
        name, ", ".join(_TRUE_WORDS + _FALSE_WORDS)))


def _bounded_int(value, default, lo, hi, name):
    """Parse a request integer, refusing anything outside [lo, hi]."""
    if isinstance(value, bool):
        raise BadRequest("%s must be an integer" % name)
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise BadRequest("%s must be an integer" % name)
    if not lo <= number <= hi:
        raise BadRequest("%s must be in [%d, %d]" % (name, lo, hi))
    return number


def _pick(primary, fallback, key):
    """First non-empty value, preferring the nested object over the payload."""
    for source in (primary, fallback):
        if not source:
            continue
        value = source.get(key)
        if value is not None and value != "":
            return value
    return None


def parse_intrinsics(payload):
    intr = payload.get("intrinsics")
    if intr is None:
        intr = {}
    elif not isinstance(intr, dict):
        raise BadRequest("intrinsics must be an object")
    image_w = _bounded_int(_pick(intr, payload, "image_w"), 720, 1, 16384, "image_w")
    image_h = _bounded_int(_pick(intr, payload, "image_h"), 1280, 1, 16384, "image_h")
    fov = _number(_pick(intr, payload, "fov_deg"), 60.0, "fov_deg", lo=1.0, hi=179.0)
    f_px = _opt_float(_pick(intr, None, "f_px"), "f_px", lo=1.0, hi=1e6)
    if f_px is None:
        return CameraIntrinsics.from_fov(image_w, image_h, fov)
    return CameraIntrinsics(
        f_px=f_px,
        cx=_number(intr.get("cx"), image_w / 2.0, "cx",
                   lo=-image_w, hi=2 * image_w),
        cy=_number(intr.get("cy"), image_h / 2.0, "cy",
                   lo=-image_h, hi=2 * image_h),
        image_w=image_w,
        image_h=image_h,
        fov_deg=fov,
    )


def parse_pose(payload):
    p = payload.get("pose")
    if p is None:
        return None
    if not isinstance(p, dict):
        raise BadRequest("pose must be an object")
    if not p:
        return None
    return CameraPose(
        lat=_number(p.get("lat"), 0.0, "pose.lat", lo=-90.0, hi=90.0),
        lon=_number(p.get("lon"), 0.0, "pose.lon", lo=-180.0, hi=180.0),
        alt_m=_number(p.get("alt_m"), 0.0, "pose.alt_m", lo=-1e5, hi=1e6),
        heading_deg=_number(p.get("heading_deg"), 0.0, "pose.heading_deg",
                            lo=-360.0, hi=360.0),
        pitch_deg=_number(p.get("pitch_deg"), 0.0, "pose.pitch_deg",
                          lo=-180.0, hi=180.0),
        roll_deg=_number(p.get("roll_deg"), 0.0, "pose.roll_deg",
                         lo=-180.0, hi=180.0),
        accuracy_m=_number(p.get("accuracy_m"), 8.0, "pose.accuracy_m", lo=0.0),
        has_gps=_flag(p.get("has_gps"), False, "pose.has_gps"),
    )


def parse_observation(payload):
    intr = parse_intrinsics(payload)
    return Observation(
        object_px_width=_number(payload.get("object_px_width"), 0.0,
                                "object_px_width", lo=0.0, hi=1e6),
        object_px_height=_number(payload.get("object_px_height"), 0.0,
                                 "object_px_height", lo=0.0, hi=1e6),
        center_x=_number(payload.get("center_x"), intr.cx, "center_x",
                         lo=-intr.image_w, hi=2 * intr.image_w),
        center_y=_number(payload.get("center_y"), intr.cy, "center_y",
                         lo=-intr.image_h, hi=2 * intr.image_h),
        intrinsics=intr,
        pose=parse_pose(payload),
        known_width_mm=_opt_float(payload.get("known_width_mm"),
                                  "known_width_mm", lo=0.01, hi=100000.0),
        object_class=_text(payload.get("object_class"), "unknown",
                           "object_class"),
        camera_height_m=_number(payload.get("camera_height_m"), 1.5,
                                "camera_height_m", lo=0.01, hi=1000.0),
        texture_density=_opt_float(payload.get("texture_density"),
                                   "texture_density", lo=0.0, hi=1.0),
        sharpness=_opt_float(payload.get("sharpness"), "sharpness", lo=0.0, hi=1.0),
        convergence=_opt_float(payload.get("convergence"), "convergence", lo=0.0, hi=1.0),
        arrowhead=_opt_float(payload.get("arrowhead"), "arrowhead", lo=0.0, hi=1.0),
        specular=_opt_float(payload.get("specular"), "specular", lo=0.0, hi=1.0),
        contrast=_opt_float(payload.get("contrast"), "contrast", lo=0.0, hi=1.0),
        room=_text(payload.get("room"), "unknown", "room"),
        timestamp=_number(payload.get("timestamp"), time.time(), "timestamp", lo=0.0),
    )


def process_observation(obs, store, engine, level=0.95, session="default", do_fit=False):
    """Run the full inference pipeline for one observation."""
    cues = engine.cues(obs)
    if not cues:
        return {"error": "no depth cues (need known_width_mm and/or ground geometry)"}

    try:
        estimate = engine.estimate(obs, level=level)
    except ValueError as exc:
        return {"error": str(exc)}
    illusions = analyze_illusions(obs, cues)
    corrections = evaluate_illusions(obs, cues)

    raw = emulated_dimensions_mm(obs, estimate.depth_m)
    tag = classify_observation(obs, raw[:2])
    raw = fill_thickness(raw, tag)
    corrected = correct_size(raw, corrections["combined_factor"])
    shape = classify_shape(corrected[0], corrected[1], corrected[2])

    geo = None
    if obs.pose is not None and obs.pose.has_gps:
        geo = project_point(
            obs.pose, obs.intrinsics, obs.center_x, obs.center_y,
            estimate.depth_m, sigma_depth_m=estimate.sigma_m,
        )

    fit = fit_observation(obs, estimate.depth_m) if do_fit else None

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
            w_mm=corrected[0], h_mm=corrected[1], d_mm=corrected[2],
            lat=geo.lat, lon=geo.lon, alt_m=geo.alt_m,
            radius_m=geo.radius_m, confidence=estimate.confidence,
        )

    return {
        "accepted": accepted,
        "room": obs.room,
        "estimate": asdict(estimate),
        "illusions": asdict(illusions),
        "corrections": corrections,
        "dimensions_mm": {"w": raw[0], "h": raw[1], "d": raw[2]},
        "corrected_dimensions_mm": {"w": corrected[0], "h": corrected[1], "d": corrected[2]},
        "shape": shape,
        "fit": fit,
        "tag": tag,
        "geo": asdict(geo) if geo else None,
        "item": item,
    }


def _demo_observation(intr, heading_deg, known_width_mm, depth_m, tag, room,
                      camera_height_m=1.5):
    f = intr.f_px
    px_w = known_width_mm * f / (depth_m * 1000.0)
    px_h = max(4.0, px_w * 0.45)
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
        texture_density=0.45, contrast=0.5,
        convergence=0.02, arrowhead=0.02, specular=0.02,
        room=room,
    )


class LidarHandler(BaseHTTPRequestHandler):
    server_version = "lidarv2/" + VERSION
    timeout = SOCKET_TIMEOUT_S

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
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return {}
        try:
            length = int(raw_len)
        except ValueError:
            raise ValueError("Content-Length must be an integer")
        if length < 0:
            raise ValueError("Content-Length must not be negative")
        if length > MAX_BODY_BYTES:
            raise PayloadTooLarge("body exceeds %d bytes" % MAX_BODY_BYTES)
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    def _serve_page(self, name):
        data = self.server.page_store.get(name)
        if data is None:
            return self._send(404, {"error": "page not found: %s" % name})
        if name.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        elif name.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif name.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        else:
            ctype = "text/plain; charset=utf-8"
        return self._send(200, data, ctype)

    # ── GET ────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._serve_page("shell.html")
        if path.startswith("/page/"):
            return self._serve_page(path[len("/page/"):])
        if path == "/api/health":
            return self._send(200, {"status": "ok", "version": VERSION,
                                    "db": self.server.store.health(),
                                    "pages": len(self.server.page_store.list())})
        if path == "/api/pages":
            return self._send(200, {"pages": self.server.page_store.list()})
        if path == "/api/config":
            return self._send(200, {
                "version": VERSION,
                "tags": TAG_CATALOG,
                "defaults": {"fov_deg": 60.0, "camera_height_m": 1.5, "level": 0.95},
            })
        if path == "/api/illusions":
            return self._send(200, {
                "illusions": [dict(e, severity=0.0) for e in ILLUSION_REGISTRY],
                "note": "5 illusions + correction math; live severities appear on /api/observe",
            })
        if path == "/api/hardware":
            return self._send(200, probe_hardware(seconds=0.2))
        if path == "/api/survivors":
            surv = os.path.join(HERE, "arena_report_survivors.json")
            if os.path.exists(surv):
                with open(surv) as fh:
                    return self._send(200, json.load(fh))
            return self._send(200, {"keep": {}, "prune": {}})
        if path == "/api/map":
            q = parse_qs(parsed.query)
            try:
                limit = _bounded_int(_first(q, "limit"), 500, 1,
                                     MAX_MAP_LIMIT, "limit")
                include_rejected = _query_flag(
                    _first(q, "include_rejected"), False, "include_rejected")
                room = _text(_first(q, "room"), None, "room")
                tag = _text(_first(q, "tag"), None, "tag")
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            points = self.server.store.list_map_points(
                room=room, tag=tag,
                limit=limit,
                include_rejected=include_rejected,
            )
            return self._send(200, {"count": len(points), "points": points})
        if path == "/api/inventory":
            q = parse_qs(parsed.query)
            try:
                room = _text(_first(q, "room"), None, "room")
                tag = _text(_first(q, "tag"), None, "tag")
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            items = self.server.store.list_items(room=room, tag=tag)
            return self._send(200, {"count": len(items), "items": items,
                                    "summary": self.server.store.summary()})
        return self._send(404, {"error": "not found"})

    # ── POST ───────────────────────────────────────────────────────────
    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except PayloadTooLarge as exc:
            # the unread body would corrupt keep-alive framing
            self.close_connection = True
            return self._send(413, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self.close_connection = True
            return self._send(400, {"error": "bad request: %s" % exc})
        try:
            if parsed.path == "/api/observe":
                return self._handle_observe(payload)
            if parsed.path == "/api/scan":
                return self._handle_scan(payload)
            if parsed.path == "/api/fit":
                return self._handle_fit(payload)
            if parsed.path == "/api/triangulate":
                return self._handle_triangulate(payload)
            if parsed.path == "/api/premade":
                return self._handle_premade(payload)
            if parsed.path == "/api/benchmark":
                return self._handle_benchmark(payload)
            if parsed.path == "/api/arena":
                return self._send(200, run_arena(quick=True, top_k=3))
            if parsed.path == "/api/plan":
                return self._handle_plan(payload)
            if parsed.path == "/api/flash":
                return self._handle_flash(payload)
            if parsed.path == "/api/flash/calibrate":
                return self._handle_flash_calibrate(payload)
            if parsed.path == "/api/flash/observe":
                return self._handle_flash_observe(payload)
            if parsed.path == "/api/demo":
                return self._handle_demo(payload)
            if parsed.path == "/api/reset":
                self.server.store.reset()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except BadRequest as exc:
            return self._send(422, {"error": str(exc)})
        except Exception as exc:
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def _handle_observe(self, payload):
        obs = parse_observation(payload)
        result = process_observation(
            obs, self.server.store, self.server.engine,
            level=_level(payload.get("level")),
            session=_text(payload.get("session"), "default", "session"),
            do_fit=_flag(payload.get("fit"), False, "fit"),
        )
        return self._send(200 if "error" not in result else 422, result)

    def _handle_scan(self, payload):
        intr = parse_intrinsics(payload)
        pose = parse_pose(payload) or CameraPose()
        camera_height = _number(payload.get("camera_height_m"), 1.5,
                                "camera_height_m", lo=0.01, hi=1000.0)
        try:
            cols = _bounded_int(payload.get("cols"), 24, 1, 256, "cols")
            rows = _bounded_int(payload.get("rows"), 16, 1, 256, "rows")
            if cols * rows > MAX_SCAN_CELLS:
                raise ValueError("cols*rows must be <= %d" % MAX_SCAN_CELLS)
            store = _flag(payload.get("store"), True, "store")
            if store:
                room = _text(payload.get("room"), "unknown", "room")
                session = _text(payload.get("session"), "scan", "session")
        except ValueError as exc:
            return self._send(422, {"error": str(exc)})
        points = scan_ground(pose, intr, camera_height, cols=cols, rows=rows)
        if store:
            now = time.time()
            self.server.store.add_map_points([{
                "ts": now, "session": session,
                "room": room,
                "lat": gp.lat, "lon": gp.lon, "alt_m": gp.alt_m,
                "radius_m": gp.radius_m,
                "east_m": gp.east_m, "north_m": gp.north_m, "up_m": gp.up_m,
                "tag": "ground", "depth_m": gp.distance_m,
                "ci_low_m": gp.distance_m, "ci_high_m": gp.distance_m,
                "confidence": 0.5, "rejected": 0, "flags": "[]",
            } for gp in points])
        return self._send(200, {"count": len(points), "points": [asdict(p) for p in points]})

    def _handle_fit(self, payload):
        intr = parse_intrinsics(payload)
        pts = _pairs(payload.get("corners_px") or [], "corners_px")
        depth_m = _number(payload.get("depth_m"), 2.0, "depth_m", lo=0.01, hi=1000.0)
        kind = _text(payload.get("kind"), "rectangle", "kind")
        if kind not in ("rectangle", "cuboid"):
            raise BadRequest("kind must be rectangle or cuboid")
        if kind == "cuboid":
            distance_m = _number(payload.get("distance_m"), depth_m,
                                 "distance_m", lo=0.01, hi=1000.0)
            result = fit_cuboid(pts, intr, distance_m)
        else:
            result = fit_rectangle(pts, intr, depth_m)
        if result is None:
            return self._send(422, {"error": "not enough corners for %s fit" % kind})
        return self._send(200, result)

    def _handle_triangulate(self, payload):
        intr = parse_intrinsics(payload)
        pose1 = parse_pose({"pose": payload.get("pose1")})
        pose2 = parse_pose({"pose": payload.get("pose2")})
        if pose1 is None or pose2 is None:
            return self._send(422, {"error": "pose1 and pose2 required"})
        if payload.get("corners1") and payload.get("corners2"):
            size = triangulate_size(pose1, _pairs(payload["corners1"], "corners1"),
                                    pose2, _pairs(payload["corners2"], "corners2"), intr)
            if size is None:
                return self._send(422, {"error": "triangulation failed"})
            return self._send(200, size)
        uv1 = payload.get("uv1")
        uv2 = payload.get("uv2")
        if not uv1 or not uv2:
            return self._send(422, {"error": "provide corners1/corners2 or uv1/uv2"})
        point = triangulate_point(pose1, _pixel(uv1, "uv1"),
                                  pose2, _pixel(uv2, "uv2"), intr)
        if point is None:
            return self._send(422, {"error": "triangulation failed"})
        return self._send(200, point)

    def _handle_premade(self, payload):
        obj = by_tag(_text(payload.get("tag"), "2x4", "tag"))
        if obj is None:
            return self._send(404, {"error": "unknown premade tag"})
        intr = parse_intrinsics(payload)
        depth_m = _number(payload.get("depth_m"), 2.0, "depth_m", lo=0.01, hi=1000.0)
        room = _text(payload.get("room"), "premade", "room")
        noise_frac = _number(payload.get("noise_frac"), 0.0,
                             "noise_frac", lo=0.0, hi=1.0)
        obs = synthesize(obj, depth_m, intrinsics=intr, room=room,
                         noise_frac=noise_frac)
        result = process_observation(obs, self.server.store, self.server.engine,
                                     level=_level(payload.get("level")),
                                     session="premade")
        truth = truth_dims(obj)
        corrected = result.get("corrected_dimensions_mm", {})
        errors = {
            "w": round(percent_error(corrected.get("w", 0), truth[0]), 3),
            "h": round(percent_error(corrected.get("h", 0), truth[1]), 3),
        }
        return self._send(200, {
            "tag": obj["tag"],
            "label": obj["label"],
            "truth_mm": {"w": truth[0], "h": truth[1], "d": truth[2]},
            "error_pct": errors,
            "result": result,
        })

    def _handle_benchmark(self, payload):
        report = run_benchmark(
            depth_m=_number(payload.get("depth_m"), 2.0, "depth_m", lo=0.01, hi=1000.0),
            noise_frac=_number(payload.get("noise_frac"), 0.0,
                               "noise_frac", lo=0.0, hi=1.0),
            level=_level(payload.get("level")),
            intrinsics=parse_intrinsics(payload) if payload.get("intrinsics") else None,
        )
        return self._send(200, report)

    def _handle_plan(self, payload):
        depth_m = _number(payload.get("depth_m"), 2.0, "depth_m", lo=0.01, hi=1000.0)
        budget = _bounded_int(payload.get("budget"), 6, 1, 16, "budget")
        tag = _text(payload.get("tag"), "", "tag")
        obj = by_tag(tag) if tag else None
        w_mm = _opt_float(payload.get("object_w_mm"), "object_w_mm",
                          lo=0.0, hi=100000.0)
        h_mm = _opt_float(payload.get("object_h_mm"), "object_h_mm",
                          lo=0.0, hi=100000.0)
        if obj is not None:
            w_mm = obj["w_mm"] if w_mm is None else w_mm
            h_mm = obj["h_mm"] if h_mm is None else h_mm
        plan = plan_views(
            parse_intrinsics(payload), depth_m, budget,
            object_w_mm=w_mm or 0.0,
            object_h_mm=h_mm or 88.9,
            sigma_px=_number(payload.get("sigma_px"), DEFAULT_SIGMA_PX,
                             "sigma_px", lo=0.001, hi=100.0),
        )
        plan["tag"] = obj["tag"] if obj is not None else None
        plan["depth_m"] = depth_m
        return self._send(200, plan)

    def _handle_flash(self, payload):
        intr = parse_intrinsics(payload)
        pose = parse_pose(payload) or CameraPose()
        try:
            if payload.get("ambient") is not None or payload.get("flash") is not None:
                cols, rows, ambient, flash = _parse_flash_grids(payload)
                cal = _parse_flash_calibration(payload)
                result = analyse_grid(ambient, flash, cols, rows, intr, cal,
                                      pose=pose,
                                      level=_level(payload.get("level")))
            else:
                cols = _bounded_int(payload.get("cols"), 96, 2, 256, "cols")
                rows = _bounded_int(payload.get("rows"), 54, 2, 256, "rows")
                if cols * rows > MAX_FLASH_CELLS:
                    raise ValueError("grid too large: cols*rows must be <= %d"
                                     % MAX_FLASH_CELLS)
                result = run_flash_demo(cols, rows,
                                        _flag(payload.get("mixed_albedo"), False,
                                              "mixed_albedo"),
                                        _bounded_int(payload.get("seed"), 0,
                                                     0, 2 ** 31 - 1, "seed"),
                                        width=cols)
        except ValueError as exc:
            return self._send(422, {"error": str(exc)})
        return self._send(200, _flash_payload(
            result, _flag(payload.get("cells"), False, "cells")))

    def _handle_flash_calibrate(self, payload):
        try:
            cols, rows, ambient, flash = _parse_flash_grids(payload)
            known = _opt_float(payload.get("reference_m"), "reference_m",
                               lo=0.01, hi=1000.0)
            result = calibrate_from_grids(
                ambient, flash, cols, rows, parse_intrinsics(payload),
                known_range_m=known,
            )
        except ValueError as exc:
            return self._send(422, {"error": str(exc)})
        return self._send(200, result)

    def _handle_flash_observe(self, payload):
        try:
            cols, rows, ambient, flash = _parse_flash_grids(payload)
            cal = _parse_flash_calibration(payload)
            obs = parse_observation(payload)
            merged = estimate_with_flash(obs, ambient, flash, cal, cols, rows,
                                         level=_level(payload.get("level")))
        except ValueError as exc:
            return self._send(422, {"error": str(exc)})
        estimate = merged["estimate"]
        cue = merged["flash_cue"]
        return self._send(200, {
            "estimate": asdict(estimate),
            "cues": [asdict(c) for c in estimate.cues],
            "flash_cue": asdict(cue) if cue else None,
            "cols": cols,
            "rows": rows,
            "valid_pct": round(valid_fraction(merged["cells"]) * 100, 2),
        })

    def _handle_demo(self, payload):
        intr = parse_intrinsics(payload)
        room = _text(payload.get("room"), "basement B", "room")
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


def _pixel(value, name):
    """One (u, v) pair from `[u, v]` or `{u, v}`; both numbers are required."""
    if isinstance(value, dict):
        return (_number(value.get("u"), name=name + ".u", lo=-1e6, hi=1e6),
                _number(value.get("v"), name=name + ".v", lo=-1e6, hi=1e6))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise BadRequest("%s must be [u, v]" % name)
    return (_number(value[0], name=name + ".u", lo=-1e6, hi=1e6),
            _number(value[1], name=name + ".v", lo=-1e6, hi=1e6))


def _pairs(corners, name="corners"):
    if not isinstance(corners, list):
        raise BadRequest("%s must be a list of [u, v] pairs" % name)
    return [_pixel(c, "%s[%d]" % (name, i)) for i, c in enumerate(corners)]


MAX_FLASH_CELLS = 16384


def _grid_values(payload, name, expected):
    """One ambient/flash grid: a list of `expected` finite numbers."""
    raw = payload.get(name)
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("%s must be a list of %d numbers" % (name, expected))
    if len(raw) != expected:
        raise ValueError("ambient and flash must each hold cols*rows values")
    return [_number(x, name="%s[%d]" % (name, i), lo=-1e6, hi=1e6)
            for i, x in enumerate(raw)]


def _parse_flash_grids(payload):
    """cols, rows, ambient[] and flash[] from a /api/flash request body."""
    cols = _bounded_int(payload.get("cols"), 96, 2, 256, "cols")
    rows = _bounded_int(payload.get("rows"), 54, 2, 256, "rows")
    if cols * rows > MAX_FLASH_CELLS:
        raise ValueError("grid too large: cols*rows must be <= %d" % MAX_FLASH_CELLS)
    expected = cols * rows
    ambient = _grid_values(payload, "ambient", expected)
    flash = _grid_values(payload, "flash", expected)
    return cols, rows, ambient, flash


def _parse_flash_calibration(payload):
    raw = payload.get("calibration")
    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raise ValueError("calibration must be an object")
    depth = _opt_float(raw.get("reference_depth_m"),
                       "calibration.reference_depth_m", lo=0.01, hi=1000.0)
    delta = _opt_float(raw.get("reference_delta_dn"),
                       "calibration.reference_delta_dn", lo=1e-9, hi=1e6)
    if not depth or not delta:
        raise ValueError("calibration.reference_depth_m and .reference_delta_dn "
                         "must be positive (anchor the light ping first)")
    min_depth = _number(raw.get("min_depth_m"), 0.3,
                        "calibration.min_depth_m", lo=0.01, hi=1000.0)
    max_depth = _number(raw.get("max_depth_m"), 8.0,
                        "calibration.max_depth_m", lo=0.01, hi=1000.0)
    if max_depth <= min_depth:
        raise ValueError("calibration.max_depth_m must exceed min_depth_m")
    return FlashCalibration(
        reference_depth_m=depth,
        reference_delta_dn=delta,
        sigma_delta_dn=_number(raw.get("sigma_delta_dn"), 1.0,
                               "calibration.sigma_delta_dn", lo=0.001, hi=1e6),
        albedo_rel_sigma=_number(raw.get("albedo_rel_sigma"), 0.15,
                                 "calibration.albedo_rel_sigma", lo=0.0, hi=10.0),
        min_depth_m=min_depth,
        max_depth_m=max_depth,
    )


def _flash_payload(result, include_cells=False):
    """Trim the flash analysis for the wire: drop the 3-D cloud by default."""
    out = {key: result.get(key) for key in (
        "cols", "rows", "valid_pct", "accuracy", "calibration", "ascii",
        "scene", "mixed_albedo", "targets", "planes", "flash_cue", "level",
    )}
    out["n_targets"] = len(result.get("targets") or [])
    if include_cells:
        out["cells"] = result.get("cells")
    return out


def build_server(host="0.0.0.0", port=8790, db_path=":memory:"):
    store = Store(db_path)
    httpd = ThreadingHTTPServer((host, port), LidarHandler)
    httpd.store = store
    httpd.engine = DepthEngine()
    httpd.page_store = PageStore(GUI_DIR)
    return httpd


def main():
    parser = argparse.ArgumentParser(description="lidarv2 emulative LiDAR backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--db", default=os.path.join(HERE, "lidarv2.db"))
    parser.add_argument("--memory", action="store_true", help="use an in-memory database")
    parser.add_argument("--version", action="version", version="lidarv2 " + VERSION)
    args = parser.parse_args()

    db_path = ":memory:" if args.memory else args.db
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = build_server(args.host, args.port, db_path)

    print("lidarv2 %s — emulative LiDAR backend" % VERSION)
    print("  serving : http://%s:%d" % (args.host, args.port))
    print("  database: %s" % db_path)
    print("  gui     : %d mmap pages" % len(httpd.page_store.list()))
    print("  open the URL in Chrome on the phone, allow camera + location")
    print("  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.page_store.close()
        httpd.store.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
