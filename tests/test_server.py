import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import build_server  # noqa: E402
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

    # ── modular mmap GUI ───────────────────────────────────────────────
    def test_pages_listed(self):
        status, body = request_json(self.base, "GET", "/api/pages")
        self.assertEqual(status, 200)
        for name in ("shell.html", "app.js", "controls.html", "measure.html"):
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


if __name__ == "__main__":
    unittest.main()
