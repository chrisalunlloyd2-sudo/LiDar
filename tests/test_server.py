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


if __name__ == "__main__":
    unittest.main()
