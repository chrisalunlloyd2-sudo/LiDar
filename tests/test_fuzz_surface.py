"""Property test: no POST payload makes the server crash or emit bad JSON.

Each round takes a valid baseline body, replaces one field (or adds an unknown
one) with an adversarial value, posts it, and checks the response. The server
may accept the body or refuse it with a 4xx, but it must never answer 5xx,
never close the connection without a reply, and never emit a NaN/Infinity token
that a strict JSON parser would reject.
"""

import json
import os
import random
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request
from urllib.parse import urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import build_server  # noqa: E402

SEED = 20261115
ROUNDS = 250

INTR = {"f_px": 554.256, "cx": 320.0, "cy": 240.0,
        "image_w": 640, "image_h": 480, "fov_deg": 60.0}
POSE = {"lat": 45.0, "lon": 10.0, "alt_m": 50.0, "heading_deg": 0.0,
        "pitch_deg": 0.0, "roll_deg": 0.0, "accuracy_m": 5.0, "has_gps": True}
CORNERS = [[300.0, 210.0], [340.0, 210.0], [340.0, 250.0], [300.0, 250.0]]
CAL = {"reference_depth_m": 1.0, "reference_delta_dn": 50.0}


def observe_payload():
    return {
        "object_px_width": 40.0, "object_px_height": 20.0,
        "center_x": 320.0, "center_y": 240.0,
        "intrinsics": dict(INTR), "pose": dict(POSE),
        "known_width_mm": 88.9, "object_class": "2x4",
        "camera_height_m": 1.5, "texture_density": 0.4, "sharpness": 0.4,
        "convergence": 0.05, "specular": 0.02, "contrast": 0.4,
        "room": "fuzz room", "level": 0.95,
    }


def flash_payload():
    return {"cols": 4, "rows": 4,
            "ambient": [10.0] * 16, "flash": [60.0] * 16,
            "calibration": dict(CAL)}


def flash_observe_payload():
    payload = observe_payload()
    payload.update(flash_payload())
    return payload


# Expensive or side-effecting routes are exercised with a few random bodies
# only; the rest get the full mutation budget.
BASELINES = {
    "/api/observe": observe_payload,
    "/api/scan": lambda: {"intrinsics": dict(INTR), "pose": dict(POSE),
                          "cols": 4, "rows": 3, "store": False,
                          "room": "fuzz room"},
    "/api/fit": lambda: {"corners_px": [list(c) for c in CORNERS],
                         "intrinsics": dict(INTR), "depth_m": 2.0,
                         "kind": "rectangle"},
    "/api/triangulate": lambda: {
        "intrinsics": dict(INTR),
        "pose1": dict(POSE),
        "pose2": dict(POSE, lat=45.00001),
        "uv1": [320.0, 240.0], "uv2": [319.0, 240.0],
    },
    "/api/premade": lambda: {"tag": "2x4", "depth_m": 2.0, "room": "fuzz room"},
    "/api/benchmark": lambda: {"depth_m": 1.5, "noise_frac": 0.01},
    "/api/plan": lambda: {"depth_m": 1.5, "budget": 4, "tag": "2x4"},
    "/api/flash": flash_payload,
    "/api/flash/calibrate": lambda: dict(flash_payload(), reference_m=1.0),
    "/api/flash/observe": flash_observe_payload,
}

WEIRD = [
    None, True, False, -1, 0, 1, 1e308, -1e308, 1e-308,
    float("nan"), float("inf"), -float("inf"),
    "", " ", "x", "x" * 200, "true", "0",
    [], {}, [1, 2], ["a"], {"u": 1}, {"x": [1]}, [[1], [2]],
]


def strict_loads(raw):
    def reject(token):
        raise ValueError("non-standard JSON constant: %s" % token)

    return json.loads(raw.decode("utf-8"), parse_constant=reject)


class TestPostSurfaceFuzz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = build_server("127.0.0.1", 0, ":memory:")
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.httpd.page_store.close()
        cls.httpd.store.close()

    def post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")   # NaN/Infinity allowed here
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            finally:
                exc.close()
            return exc.code, exc.headers.get("Content-Type", ""), body

    def check(self, path, payload):
        status, ctype, raw = self.post(path, payload)
        detail = "%s -> %s for %r" % (path, status, payload)
        self.assertLess(status, 500, detail)
        self.assertIn(status, (200, 400, 404, 413, 422), detail)
        self.assertIn("application/json", ctype, detail)
        try:
            parsed = strict_loads(raw)
        except ValueError as exc:
            self.fail("%s returned non-standard JSON (%s): %r"
                      % (detail, exc, raw[:300]))
        self.assertIsInstance(parsed, dict, detail)

    def mutate(self, rng, baseline):
        payload = json.loads(json.dumps(baseline))   # deep copy
        keys = list(payload)
        key = rng.choice(keys)
        if isinstance(payload[key], dict) and payload[key] and rng.random() < 0.5:
            inner = rng.choice(list(payload[key]))
            payload[key] = dict(payload[key])
            payload[key][inner] = rng.choice(WEIRD)
        else:
            payload[key] = rng.choice(WEIRD)
        if rng.random() < 0.25:
            payload["unknown_%d" % rng.randrange(1000)] = rng.choice(WEIRD)
        return payload

    def test_mutated_payloads_never_crash_the_handler(self):
        rng = random.Random(SEED)
        paths = sorted(BASELINES)
        for i in range(ROUNDS):
            path = paths[i % len(paths)]
            self.check(path, self.mutate(rng, BASELINES[path]()))

    def test_random_bodies_on_every_route(self):
        rng = random.Random(SEED + 1)
        paths = sorted(BASELINES) + ["/api/demo", "/api/arena", "/api/reset"]
        for path in paths:
            for _ in range(2):
                payload = {("k%d" % i): rng.choice(WEIRD)
                           for i in range(rng.randrange(0, 4))}
                self.check(path, payload)

    def test_full_size_boundaries_stay_finite(self):
        # values at the field bounds must still not overflow the math
        cases = [
            ("/api/observe", dict(observe_payload(), center_x=-640.0,
                                  center_y=960.0)),
            ("/api/observe", dict(observe_payload(), object_px_width=1e6,
                                  object_px_height=1e6)),
            ("/api/observe", {"known_width_mm": 88.9, "texture_density": 0.2,
                              "object_px_width": 1e-308}),
            ("/api/fit", {"corners_px": [[-1e6, -1e6], [1e6, -1e6],
                                         [1e6, 1e6], [-1e6, 1e6]],
                          "intrinsics": dict(INTR), "depth_m": 0.01}),
            ("/api/flash", dict(flash_payload(), ambient=[1e6] * 16,
                                flash=[1e6] * 16)),
            ("/api/observe", dict(observe_payload(), pose=dict(POSE, lat=90.0))),
            ("/api/observe", dict(observe_payload(), pose=dict(POSE, alt_m=-1e5))),
        ]
        for path, payload in cases:
            self.check(path, payload)


QUERY_VALUES = [
    "", "x", "x" * 129, "%00", "a%0Ab", "true", "false", "1", "0", "-1",
    "abc", "1e308", "NaN", "999999", "5001", "₩", "…", "%ff", "r" * 2000,
]
QUERY_KEYS = ["room", "tag", "limit", "include_rejected", "unknown"]


class TestGetSurfaceFuzz(unittest.TestCase):
    """GET routes: pages, API reads, query combinations and raw request lines."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = build_server("127.0.0.1", 0, ":memory:")
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        with urllib.request.urlopen(cls.base + "/api/pages", timeout=10) as resp:
            cls.pages = json.loads(resp.read())["pages"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.httpd.page_store.close()
        cls.httpd.store.close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.headers.get("Content-Type", ""), exc.read()
            finally:
                exc.close()

    def check_any(self, path):
        status, ctype, raw = self.get(path)
        detail = "%s -> %s" % (path, status)
        self.assertLess(status, 500, detail)
        self.assertIn(status, (200, 400, 404), detail)
        if "application/json" in ctype:
            try:
                body = strict_loads(raw)
            except ValueError as exc:
                self.fail("%s returned non-standard JSON (%s)" % (detail, exc))
            self.assertIsInstance(body, dict, detail)
        else:
            self.assertTrue("text/html" in ctype or "javascript" in ctype,
                            "%s ctype %r" % (detail, ctype))
            self.assertTrue(raw, detail)

    def check_api(self, path):
        status, ctype, raw = self.get(path)
        detail = "%s -> %s" % (path, status)
        self.assertLess(status, 500, detail)
        self.assertIn(status, (200, 400, 404), detail)
        self.assertIn("application/json", ctype, detail)
        try:
            body = strict_loads(raw)
        except ValueError as exc:
            self.fail("%s returned non-standard JSON (%s)" % (detail, exc))
        self.assertIsInstance(body, dict, detail)

    def test_known_pages_and_entry_point_serve(self):
        status, ctype, raw = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"LIDARV2", raw)
        for name in self.pages:
            self.check_any("/page/" + name)

    def test_every_api_route_answers(self):
        for path in ("/api/health", "/api/pages", "/api/config",
                     "/api/illusions", "/api/survivors", "/api/map",
                     "/api/inventory", "/api/hardware"):
            self.check_api(path)

    def test_page_paths_cannot_escape_the_page_store(self):
        for path in ("/page/../server.py", "/page/..%2fserver.py",
                     "/page/%2e%2e/%2e%2e/etc/passwd",
                     "/page/....//server.py", "/page//shell.html"):
            status, ctype, raw = self.get(path)
            self.assertIn(status, (200, 404), path)
            self.assertNotIn(b"#!/usr/bin/env", raw, path)
            self.assertNotIn(b"import sqlite3", raw, path)

    def test_random_queries_on_read_routes(self):
        rng = random.Random(SEED + 2)
        for _ in range(120):
            roll = rng.random()
            if roll < 0.45:
                pairs = [(rng.choice(QUERY_KEYS), rng.choice(QUERY_VALUES))
                         for _ in range(rng.randrange(1, 4))]
                self.check_api("/api/map?" + urlencode(pairs))
            elif roll < 0.9:
                pairs = [(rng.choice(QUERY_KEYS[:2]), rng.choice(QUERY_VALUES))
                         for _ in range(rng.randrange(1, 3))]
                self.check_api("/api/inventory?" + urlencode(pairs))
            else:
                path = "/api/" + "".join(
                    rng.choice("abcdefghijklmnopqrstuvwxyz0123456789_-")
                    for _ in range(rng.randrange(1, 20)))
                self.check_api(path)

    def test_random_paths_never_crash(self):
        rng = random.Random(SEED + 3)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789._-%/"
        for _ in range(120):
            suffix = "".join(rng.choice(alphabet)
                             for _ in range(rng.randrange(1, 40)))
            path = ("/page/" if rng.random() < 0.5 else "/api/") + suffix
            self.check_any(path)

    def test_overlong_request_line_is_refused(self):
        request = (b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\nHost: x\r\n\r\n")
        with socket.create_connection(
                ("127.0.0.1", self.httpd.server_address[1]), timeout=10) as sock:
            sock.sendall(request)
            reply = sock.recv(400)
        status_line = reply.split(b"\r\n", 1)[0]
        self.assertTrue(status_line.startswith(b"HTTP/"), status_line)
        self.assertGreaterEqual(int(status_line.split()[1]), 400, status_line)
        self.assertLess(int(status_line.split()[1]), 500, status_line)

    def test_nul_byte_in_path_is_a_404(self):
        request = b"GET /page/a\x00b HTTP/1.1\r\nHost: x\r\n\r\n"
        with socket.create_connection(
                ("127.0.0.1", self.httpd.server_address[1]), timeout=10) as sock:
            sock.sendall(request)
            reply = sock.recv(400)
        status_line = reply.split(b"\r\n", 1)[0]
        self.assertIn(b" 404 ", status_line, status_line)

    def test_nul_byte_in_query_is_400(self):
        request = b"GET /api/map?room=a\x00b HTTP/1.1\r\nHost: x\r\n\r\n"
        with socket.create_connection(
                ("127.0.0.1", self.httpd.server_address[1]), timeout=10) as sock:
            sock.sendall(request)
            reply = sock.recv(400)
        status_line = reply.split(b"\r\n", 1)[0]
        self.assertIn(b" 400 ", status_line, status_line)



if __name__ == "__main__":
    unittest.main()
