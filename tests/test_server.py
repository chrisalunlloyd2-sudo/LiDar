import json
import os
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import build_server  # noqa: E402
from server import MAX_BODY_BYTES, MAX_MAP_LIMIT  # noqa: E402
import flash_lidar  # noqa: E402
from models import CameraIntrinsics, CameraPose  # noqa: E402
from geospatial import enu_to_geodetic  # noqa: E402
from triangulate import enu_direction_to_pixel  # noqa: E402


def request_json(base, method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def request_text(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


def raw_request(port, request, timeout=5):
    """Send a hand-built request and read until the server closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


def observe_payload(depth_m=1.0):
    f = 554.2562584220408  # 640px wide, 60 deg FOV
    known_mm = 88.9
    px = known_mm * f / (depth_m * 1000.0)
    return {
        "object_px_width": px,
        "object_px_height": px * 0.45,
        "center_x": 320.0,
        "center_y": 240.0,
        "intrinsics": {"f_px": f, "cx": 320.0, "cy": 240.0,
                       "image_w": 640, "image_h": 480, "fov_deg": 60.0},
        "pose": {"lat": 45.0, "lon": 10.0, "alt_m": 50.0, "heading_deg": 0.0,
                 "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 5.0, "has_gps": True},
        "known_width_mm": known_mm,
        "object_class": "2x4",
        "camera_height_m": 1.5,
        "texture_density": 0.4,
        "sharpness": 0.4,
        "convergence": 0.05,
        "specular": 0.02,
        "contrast": 0.4,
        "room": "basement B",
        "level": 0.95,
    }


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = build_server("127.0.0.1", 0, ":memory:")
        cls.port = cls.httpd.server_address[1]
        cls.base = "http://127.0.0.1:%d" % cls.port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.httpd.page_store.close()
        cls.httpd.store.close()

    def test_health(self):
        status, body = request_json(self.base, "GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["db"]["ok"])

    def test_config_lists_tags(self):
        status, body = request_json(self.base, "GET", "/api/config")
        self.assertEqual(status, 200)
        tags = [t["tag"] for t in body["tags"]]
        self.assertIn("2x4", tags)

    def test_index_served(self):
        status, html = request_text(self.base, "/")
        self.assertEqual(status, 200)
        self.assertIn("LIDARV2", html)

    def test_observe_returns_interval_and_geo(self):
        status, body = request_json(self.base, "POST", "/api/observe", observe_payload(1.0))
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])
        self.assertAlmostEqual(body["estimate"]["depth_m"], 1.0, delta=0.05)
        self.assertLess(body["estimate"]["ci_low_m"], body["estimate"]["depth_m"])
        self.assertGreater(body["estimate"]["ci_high_m"], body["estimate"]["depth_m"])
        self.assertGreater(body["estimate"]["confidence"], 0.5)
        self.assertEqual(body["tag"]["tag"], "2x4")
        self.assertIsNotNone(body["geo"])
        self.assertAlmostEqual(body["geo"]["lat"], 45.0, places=4)

    def test_inventory_after_observe(self):
        request_json(self.base, "POST", "/api/reset", {})
        status, body = request_json(self.base, "POST", "/api/observe", observe_payload(2.0))
        self.assertEqual(status, 200)
        status, inv = request_json(self.base, "GET", "/api/inventory?room=basement%20B")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(inv["count"], 1)
        self.assertEqual(inv["items"][0]["tag"], "2x4")
        self.assertEqual(inv["items"][0]["room"], "basement B")

    def test_demo_seeds_inventory(self):
        request_json(self.base, "POST", "/api/reset", {})
        status, body = request_json(self.base, "POST", "/api/demo", {"room": "basement B"})
        self.assertEqual(status, 200)
        self.assertEqual(body["seeded"], 4)
        tags = {it["tag"] for it in body["inventory"]}
        self.assertTrue({"2x4", "2x6", "paint_can", "box"} <= tags)

    def test_scan_returns_points(self):
        status, body = request_json(self.base, "POST", "/api/scan", {
            "intrinsics": {"f_px": 554.256, "cx": 160.0, "cy": 120.0,
                           "image_w": 320, "image_h": 240, "fov_deg": 60.0},
            "pose": {"lat": 45.0, "lon": 10.0, "alt_m": 50.0, "heading_deg": 0.0,
                     "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 5.0, "has_gps": True},
            "camera_height_m": 1.5, "cols": 8, "rows": 6, "store": True,
            "room": "basement B",
        })
        self.assertEqual(status, 200)
        self.assertGreater(body["count"], 0)
        self.assertGreater(len(body["points"]), 0)

    def test_reset_clears(self):
        request_json(self.base, "POST", "/api/demo", {"room": "basement B"})
        request_json(self.base, "POST", "/api/reset", {})
        status, inv = request_json(self.base, "GET", "/api/inventory")
        self.assertEqual(status, 200)
        self.assertEqual(inv["count"], 0)

    def test_unknown_route_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "GET", "/api/nope")
        self.assertEqual(ctx.exception.code, 404)
        ctx.exception.close()

    # ── request framing ────────────────────────────────────────────────
    def test_non_object_json_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/observe", [])
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("JSON object", ctx.exception.read().decode("utf-8"))
        ctx.exception.close()

    def test_negative_content_length_is_rejected_without_hanging(self):
        reply = raw_request(self.port, (
            b"POST /api/observe HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: -1\r\n\r\n{}"
        ))
        self.assertIn(b"400", reply.split(b"\r\n", 1)[0])

    def test_malformed_content_length_is_rejected(self):
        reply = raw_request(self.port, (
            b"POST /api/observe HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: nope\r\n\r\n{}"
        ))
        self.assertIn(b"400", reply.split(b"\r\n", 1)[0])

    def test_oversized_body_is_refused_before_it_is_read(self):
        reply = raw_request(self.port, (
            b"POST /api/observe HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: %d\r\n\r\n" % (MAX_BODY_BYTES + 1)
        ))
        self.assertIn(b"413", reply.split(b"\r\n", 1)[0])

    def test_map_rejects_non_integer_limit(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "GET", "/api/map?limit=abc")
        self.assertEqual(ctx.exception.code, 400)
        ctx.exception.close()

    def test_map_rejects_out_of_range_limit(self):
        for limit in ("0", "-5", str(MAX_MAP_LIMIT + 1)):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                request_json(self.base, "GET", "/api/map?limit=" + limit)
            self.assertEqual(ctx.exception.code, 400)
            ctx.exception.close()

    def test_scan_rejects_oversized_grid(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/scan", {"cols": 256, "rows": 256})
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_scan_rejects_non_integer_grid(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/scan", {"cols": "many"})
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_scan_rejects_empty_grid(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/scan", {"rows": 0})
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_scan_stores_the_whole_grid(self):
        status, body = request_json(self.base, "POST", "/api/scan", {
            "intrinsics": {"f_px": 554.256, "cx": 160.0, "cy": 120.0,
                           "image_w": 320, "image_h": 240, "fov_deg": 60.0},
            "pose": {"lat": 45.0, "lon": 10.0, "alt_m": 50.0, "heading_deg": 0.0,
                     "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 5.0, "has_gps": True},
            "camera_height_m": 1.5, "cols": 8, "rows": 6, "store": True,
            "room": "batch room",
        })
        self.assertEqual(status, 200)
        self.assertGreater(body["count"], 0)
        status, mapped = request_json(
            self.base, "GET", "/api/map?room=batch%20room&limit=500")
        self.assertEqual(status, 200)
        self.assertEqual(mapped["count"], body["count"])

    # ── numeric body fields ────────────────────────────────────────────
    def assert_rejected(self, path, payload, code=422):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", path, payload)
        self.assertEqual(ctx.exception.code, code,
                         ctx.exception.read().decode("utf-8"))
        ctx.exception.close()

    def test_scan_rejects_non_numeric_intrinsics(self):
        for value in ("wide", -5, 0, float("nan")):
            self.assert_rejected("/api/scan", {"intrinsics": {"f_px": value}})

    def test_scan_rejects_non_numeric_camera_height(self):
        for value in ("high", -1, 0, float("inf")):
            self.assert_rejected("/api/scan", {"camera_height_m": value})

    def test_observe_rejects_out_of_range_pose(self):
        for key, value in (("lat", 91.0), ("lon", -181.0), ("accuracy_m", -1.0)):
            payload = observe_payload()
            payload["pose"][key] = value
            self.assert_rejected("/api/observe", payload)

    def test_observe_rejects_non_numeric_pose_accuracy(self):
        payload = observe_payload()
        payload["pose"]["accuracy_m"] = "high"
        self.assert_rejected("/api/observe", payload)

    def test_observe_rejects_out_of_range_level(self):
        for value in (0.0, 1.5, float("nan")):
            self.assert_rejected("/api/observe", {"level": value})

    def test_observe_rejects_non_finite_optional_cue(self):
        for key in ("known_width_mm", "texture_density", "specular"):
            payload = observe_payload()
            payload[key] = float("nan")
            self.assert_rejected("/api/observe", payload)

    def test_observe_rejects_out_of_range_cue(self):
        for key, value in (("texture_density", 2.0), ("object_px_width", -4.0)):
            payload = observe_payload()
            payload[key] = value
            self.assert_rejected("/api/observe", payload)

    def test_fit_rejects_malformed_corners(self):
        self.assert_rejected("/api/fit", {"corners_px": "nope"})
        self.assert_rejected("/api/fit", {"corners_px": [[1.0, 2.0], [3.0]]})
        self.assert_rejected("/api/fit", {"corners_px": [{"u": "left", "v": 2}]})

    def test_fit_rejects_non_positive_depth(self):
        corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        for depth in (0, -2.0, "far"):
            self.assert_rejected("/api/fit", {"corners_px": corners, "depth_m": depth})

    def test_triangulate_rejects_malformed_uv(self):
        self.assert_rejected("/api/triangulate", {
            "pose1": {"lat": 0.0}, "pose2": {"lat": 0.001},
            "uv1": [1.0], "uv2": [1.0, 2.0],
        })

    def test_premade_rejects_bad_noise(self):
        for value in (-0.5, 2.0, float("nan")):
            self.assert_rejected("/api/premade", {"noise_frac": value})

    def test_benchmark_rejects_non_numeric_depth(self):
        self.assert_rejected("/api/benchmark", {"depth_m": "deep"})

    def test_plan_rejects_bad_budget(self):
        for value in ("many", 0, 99):
            self.assert_rejected("/api/plan", {"budget": value})

    def test_plan_rejects_non_positive_sigma(self):
        for value in (0, -1.0, float("nan")):
            self.assert_rejected("/api/plan", {"sigma_px": value})

    def flash_grid_payload(self, **extra):
        payload = {"cols": 4, "rows": 4,
                   "ambient": [10.0] * 16, "flash": [60.0] * 16,
                   "calibration": {"reference_depth_m": 1.0,
                                   "reference_delta_dn": 50.0}}
        payload.update(extra)
        return payload

    def test_flash_rejects_out_of_range_level(self):
        self.assert_rejected("/api/flash", self.flash_grid_payload(level=2.0))

    def test_flash_rejects_non_finite_calibration(self):
        for key, value in (("reference_delta_dn", float("inf")),
                           ("sigma_delta_dn", float("nan")),
                           ("min_depth_m", -1.0)):
            cal = dict(self.flash_grid_payload()["calibration"])
            cal[key] = value
            self.assert_rejected("/api/flash", self.flash_grid_payload(calibration=cal))
        inverted = dict(self.flash_grid_payload()["calibration"],
                        min_depth_m=5.0, max_depth_m=2.0)
        self.assert_rejected("/api/flash", self.flash_grid_payload(calibration=inverted))

    def test_flash_rejects_non_finite_cell(self):
        ambient = [10.0] * 16
        ambient[7] = float("nan")
        self.assert_rejected("/api/flash", self.flash_grid_payload(ambient=ambient))

    # ── boolean body fields ────────────────────────────────────────────
    def test_scan_rejects_non_boolean_store(self):
        for value in ("false", 0, "no"):
            self.assert_rejected("/api/scan", {"store": value})

    def test_scan_store_false_writes_nothing(self):
        status, body = request_json(self.base, "POST", "/api/scan", {
            "intrinsics": {"f_px": 554.256, "cx": 160.0, "cy": 120.0,
                           "image_w": 320, "image_h": 240, "fov_deg": 60.0},
            "pose": {"lat": 45.0, "lon": 10.0, "alt_m": 50.0, "heading_deg": 0.0,
                     "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 5.0,
                     "has_gps": True},
            "camera_height_m": 1.5, "cols": 4, "rows": 3, "store": False,
            "room": "flag room",
        })
        self.assertEqual(status, 200)
        self.assertGreater(body["count"], 0)
        status, mapped = request_json(
            self.base, "GET", "/api/map?room=flag%20room")
        self.assertEqual(status, 200)
        self.assertEqual(mapped["count"], 0)

    def test_observe_rejects_non_boolean_fit(self):
        payload = observe_payload()
        payload["fit"] = "yes"
        self.assert_rejected("/api/observe", payload)

    def test_pose_rejects_non_boolean_has_gps(self):
        payload = observe_payload()
        payload["pose"]["has_gps"] = "false"   # used to be read as True
        self.assert_rejected("/api/observe", payload)

    def test_has_gps_false_yields_no_geopoint(self):
        payload = observe_payload()
        payload["pose"]["has_gps"] = False
        status, body = request_json(self.base, "POST", "/api/observe", payload)
        self.assertEqual(status, 200)
        self.assertIsNone(body["geo"])

    def test_flash_rejects_non_boolean_flags(self):
        self.assert_rejected("/api/flash", {"cols": 48, "rows": 27,
                                            "mixed_albedo": "false"})
        self.assert_rejected("/api/flash", {"cols": 48, "rows": 27,
                                            "cells": "true"})

    def test_map_rejects_unknown_include_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "GET", "/api/map?include_rejected=maybe")
        self.assertEqual(ctx.exception.code, 400)
        ctx.exception.close()

    def test_map_accepts_query_boolean_words(self):
        for word in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            status, body = request_json(
                self.base, "GET", "/api/map?include_rejected=" + word)
            self.assertEqual(status, 200)
            self.assertIn("points", body)

    # ── modular mmap GUI ───────────────────────────────────────────────
    def test_pages_listed(self):
        status, body = request_json(self.base, "GET", "/api/pages")
        self.assertEqual(status, 200)
        for name in ("shell.html", "app.js", "controls.html", "measure.html",
                     "plan.html", "arena.html", "hardware.html"):
            self.assertIn(name, body["pages"])

    def test_page_fragment_served(self):
        status, html = request_text(self.base, "/page/controls.html")
        self.assertEqual(status, 200)
        self.assertIn("btnStart", html)

    def test_app_js_served(self):
        status, js = request_text(self.base, "/page/app.js")
        self.assertEqual(status, 200)
        self.assertIn("loadPages", js)

    # ── new engines ────────────────────────────────────────────────────
    def test_illusions_registry(self):
        status, body = request_json(self.base, "GET", "/api/illusions")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["illusions"]), 5)
        ids = [i["id"] for i in body["illusions"]]
        for expected in ("ponzo", "muller_lyer", "vertical_horizontal",
                         "ebbinghaus", "checker_shadow"):
            self.assertIn(expected, ids)

    def test_observe_includes_corrections_and_shape(self):
        status, body = request_json(self.base, "POST", "/api/observe", observe_payload(1.0))
        self.assertEqual(status, 200)
        self.assertIn("corrections", body)
        self.assertEqual(len(body["corrections"]["illusions"]), 5)
        self.assertIn("combined_factor", body["corrections"])
        self.assertIn("corrected_dimensions_mm", body)
        self.assertIn("shape", body)
        self.assertIn(body["shape"]["shape"], ("square", "rectangle", "cube", "cuboid", "lumber", "unknown"))

    def test_fit_endpoint(self):
        intr = {"f_px": 554.256, "cx": 320.0, "cy": 240.0,
                "image_w": 640, "image_h": 480, "fov_deg": 60.0}
        f = 554.256
        hw = 50.0 / 1000.0 * f / 5.0
        hh = 25.0 / 1000.0 * f / 5.0
        corners = [[320 - hw, 240 - hh], [320 + hw, 240 - hh],
                   [320 + hw, 240 + hh], [320 - hw, 240 + hh]]
        status, body = request_json(self.base, "POST", "/api/fit", {
            "corners_px": corners, "intrinsics": intr, "depth_m": 5.0, "kind": "rectangle",
        })
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["w_mm"], 100.0, delta=2.0)
        self.assertAlmostEqual(body["h_mm"], 50.0, delta=2.0)

    def test_triangulate_endpoint(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        pose1 = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                           pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
        lat2, lon2, alt2 = enu_to_geodetic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        pose2 = CameraPose(lat=lat2, lon=lon2, alt_m=alt2, heading_deg=0.0,
                           pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
        uv1 = enu_direction_to_pixel(0.0, 5.0, 0.0, pose1, intr)
        uv2 = enu_direction_to_pixel(-1.0, 5.0, 0.0, pose2, intr)
        status, body = request_json(self.base, "POST", "/api/triangulate", {
            "intrinsics": {"f_px": intr.f_px, "cx": intr.cx, "cy": intr.cy,
                           "image_w": 640, "image_h": 480, "fov_deg": 60.0},
            "pose1": {"lat": 0.0, "lon": 0.0, "alt_m": 0.0, "heading_deg": 0.0,
                      "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 1.0, "has_gps": True},
            "pose2": {"lat": lat2, "lon": lon2, "alt_m": alt2, "heading_deg": 0.0,
                      "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 1.0, "has_gps": True},
            "uv1": list(uv1), "uv2": list(uv2),
        })
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["north_m"], 5.0, delta=0.05)
        self.assertAlmostEqual(body["east_m"], 0.0, delta=0.05)

    def test_premade_endpoint(self):
        status, body = request_json(self.base, "POST", "/api/premade", {
            "tag": "2x4", "depth_m": 2.0, "room": "premade",
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["tag"], "2x4")
        self.assertLess(body["error_pct"]["w"], 2.0)
        self.assertIn("result", body)

    def test_benchmark_endpoint(self):
        status, body = request_json(self.base, "POST", "/api/benchmark", {
            "depth_m": 2.0, "noise_frac": 0.02,
        })
        self.assertEqual(status, 200)
        self.assertEqual(len(body["results"]), 6)
        self.assertIn("mean_mape_pct", body)
        self.assertIn("coverage_w_pct", body)

    def test_hardware_endpoint(self):
        status, body = request_json(self.base, "GET", "/api/hardware")
        self.assertEqual(status, 200)
        self.assertIn("cpu_count", body)
        self.assertIn("fusion_ops_per_sec", body)
        self.assertIn("budget", body)

    def test_arena_endpoint(self):
        status, body = request_json(self.base, "POST", "/api/arena", {})
        self.assertEqual(status, 200)
        for stage in ("fusers", "sizes", "illusions", "triangulators"):
            self.assertIn(stage, body)
            self.assertIn(stage, body["survivors"])
            self.assertIn(stage, body["pruned"])

    def test_survivors_endpoint(self):
        status, body = request_json(self.base, "GET", "/api/survivors")
        self.assertEqual(status, 200)
        self.assertIn("keep", body)
        self.assertIn("prune", body)

    def test_plan_endpoint(self):
        status, body = request_json(self.base, "POST", "/api/plan", {
            "depth_m": 1.5, "budget": 6, "tag": "2x6",
            "intrinsics": {"f_px": 554.256, "cx": 320.0, "cy": 240.0,
                           "image_w": 640, "image_h": 480, "fov_deg": 60.0},
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["tag"], "2x6")
        self.assertEqual(body["budget"], 6)
        self.assertEqual(len(body["offsets_m"]), 6)
        self.assertEqual(len(body["steps"]), 6)
        self.assertGreater(body["fov_cap_m"], 0.0)
        self.assertLessEqual(max(abs(b) for b in body["offsets_m"]),
                             body["fov_cap_m"] + 1e-6)
        self.assertGreater(body["predicted_error_pct"], 0.0)
        self.assertEqual(len(body["acquisition_order_m"]), 6)

    def test_plan_rejects_non_positive_depth(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/plan", {"depth_m": 0.0})
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_flash_demo_endpoint(self):
        status, body = request_json(self.base, "POST", "/api/flash", {
            "cols": 48, "rows": 27, "seed": 0,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["cols"], 48)
        self.assertEqual(body["rows"], 27)
        self.assertIn("ascii", body)
        self.assertEqual(len(body["ascii"]), int(round(27 * 0.5)))
        self.assertAlmostEqual(body["valid_pct"], 100.0)
        self.assertEqual(body["n_targets"], len(body["targets"]))
        self.assertIn("mape_pct", body["accuracy"])
        self.assertNotIn("cells", body)          # trimmed unless asked

    def test_flash_grid_endpoint_uses_supplied_frames(self):
        intr = flash_lidar.CameraIntrinsics.from_fov(640, 480, 60.0)
        frame = flash_lidar.render_scene(
            flash_lidar.FlashScene("wall", [], wall_depth_m=2.0),
            intr, cols=32, rows=18, seed=0,
        )
        cal = flash_lidar.calibrate_scene(frame, intr)
        status, body = request_json(self.base, "POST", "/api/flash", {
            "cols": 32, "rows": 18,
            "ambient": frame["ambient"], "flash": frame["flash"],
            "intrinsics": {"f_px": intr.f_px, "cx": intr.cx, "cy": intr.cy,
                           "image_w": intr.image_w, "image_h": intr.image_h,
                           "fov_deg": 60.0},
            "calibration": {"reference_depth_m": cal.reference_depth_m,
                            "reference_delta_dn": cal.reference_delta_dn},
        })
        self.assertEqual(status, 200)
        self.assertIsNone(body["accuracy"])
        self.assertEqual(body["valid_pct"], 100.0)
        self.assertEqual(len(body["planes"]), 1)
        self.assertAlmostEqual(abs(body["planes"][0]["offset_m"]), 2.0,
                               delta=0.15)
        self.assertIsNotNone(body["flash_cue"])
        self.assertEqual(body["flash_cue"]["name"], "flash_falloff")

    def test_flash_grid_requires_calibration(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/flash", {
                "cols": 4, "rows": 4, "ambient": [10.0] * 16,
                "flash": [60.0] * 16,
            })
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_flash_calibrate_endpoint_suggests_a_gauge(self):
        intr = flash_lidar.CameraIntrinsics.from_fov(640, 480, 60.0)
        frame = flash_lidar.render_scene(
            flash_lidar.FlashScene("wall", [], wall_depth_m=1.5),
            intr, cols=32, rows=18, seed=0,
        )
        status, body = request_json(self.base, "POST", "/api/flash/calibrate", {
            "cols": 32, "rows": 18,
            "ambient": frame["ambient"], "flash": frame["flash"],
            "intrinsics": {"f_px": intr.f_px, "cx": intr.cx, "cy": intr.cy,
                           "image_w": intr.image_w, "image_h": intr.image_h,
                           "fov_deg": 60.0},
            "reference_m": 1.5,
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["anchored"])
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["cols"], 32)
        self.assertGreater(body["centre_median_dn"], 20.0)
        suggested = body["suggested_calibration"]
        self.assertEqual(suggested["reference_depth_m"], 1.5)
        self.assertAlmostEqual(suggested["reference_delta_dn"],
                               body["centre_median_dn"], places=3)
        self.assertGreater(suggested["constant"], 0.0)

    def test_flash_calibrate_dark_pair_abstains(self):
        status, body = request_json(self.base, "POST", "/api/flash/calibrate", {
            "cols": 8, "rows": 8,
            "ambient": [10.0] * 64, "flash": [10.0] * 64,
            "reference_m": 1.0,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "dark")
        self.assertIsNone(body["suggested_calibration"])

    def test_flash_calibrate_rejects_wrong_grid_size(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request_json(self.base, "POST", "/api/flash/calibrate", {
                "cols": 8, "rows": 8, "ambient": [10.0] * 4,
                "flash": [60.0] * 4, "reference_m": 1.0,
            })
        self.assertEqual(ctx.exception.code, 422)
        ctx.exception.close()

    def test_flash_observe_fuses_the_flash_cue(self):
        intr = flash_lidar.CameraIntrinsics.from_fov(640, 480, 60.0)
        scene = flash_lidar.FlashScene("centred 2x6", [
            flash_lidar.ScenePatch("2x6", "2x6 lumber", 0.0, 0.0,
                                   0.1397, 0.62, 1.5),
        ])
        cols, rows = 48, 27
        frame = flash_lidar.render_scene(scene, intr, cols=cols, rows=rows, seed=0)
        cal = flash_lidar.calibrate_scene(frame, intr)
        payload = observe_payload(depth_m=1.5)
        payload.update(cols=cols, rows=rows, intrinsics={
            "f_px": intr.f_px, "cx": intr.cx, "cy": intr.cy,
            "image_w": intr.image_w, "image_h": intr.image_h, "fov_deg": 60.0,
        }, ambient=frame["ambient"], flash=frame["flash"],
           calibration={"reference_depth_m": cal.reference_depth_m,
                        "reference_delta_dn": cal.reference_delta_dn})
        status, body = request_json(self.base, "POST", "/api/flash/observe", payload)
        self.assertEqual(status, 200)
        names = [c["name"] for c in body["cues"]]
        self.assertIn("flash_falloff", names)
        self.assertAlmostEqual(body["flash_cue"]["depth_m"], 1.5, delta=0.2)
        self.assertLess(body["estimate"]["ci_low_m"],
                        body["estimate"]["depth_m"])
        self.assertAlmostEqual(body["valid_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
