import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geospatial import enu_to_geodetic  # noqa: E402
from models import CameraIntrinsics, CameraPose  # noqa: E402
from triangulate import (  # noqa: E402
    enu_direction_to_pixel,
    geodetic_to_enu,
    triangulate_point,
    triangulate_size,
)


class TestTriangulation(unittest.TestCase):
    def setUp(self):
        self.intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        self.pose1 = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                                pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
        lat2, lon2, alt2 = enu_to_geodetic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.pose2 = CameraPose(lat=lat2, lon=lon2, alt_m=alt2, heading_deg=0.0,
                                pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)

    def test_geodetic_to_enu_roundtrip(self):
        e, n, u = geodetic_to_enu(0.0, 0.0, 0.0,
                                  self.pose2.lat, self.pose2.lon, self.pose2.alt_m)
        self.assertAlmostEqual(e, 1.0, places=3)
        self.assertAlmostEqual(n, 0.0, places=3)

    def test_point_center(self):
        uv1 = enu_direction_to_pixel(0.0, 5.0, 0.0, self.pose1, self.intr)
        uv2 = enu_direction_to_pixel(-1.0, 5.0, 0.0, self.pose2, self.intr)
        self.assertIsNotNone(uv1)
        self.assertIsNotNone(uv2)
        point = triangulate_point(self.pose1, uv1, self.pose2, uv2, self.intr)
        self.assertAlmostEqual(point["east_m"], 0.0, delta=0.01)
        self.assertAlmostEqual(point["north_m"], 5.0, delta=0.01)
        self.assertAlmostEqual(point["up_m"], 0.0, delta=0.01)
        self.assertLess(point["residual_m"], 1e-3)

    def test_point_offset(self):
        target = (0.5, 4.0, 0.2)  # east, north, up
        uv1 = enu_direction_to_pixel(*target, self.pose1, self.intr)
        uv2 = enu_direction_to_pixel(target[0] - 1.0, target[1], target[2],
                                     self.pose2, self.intr)
        point = triangulate_point(self.pose1, uv1, self.pose2, uv2, self.intr)
        self.assertAlmostEqual(point["east_m"], 0.5, delta=0.02)
        self.assertAlmostEqual(point["north_m"], 4.0, delta=0.02)
        self.assertAlmostEqual(point["up_m"], 0.2, delta=0.02)

    def test_size(self):
        w, h = 0.10, 0.05
        center = (0.0, 5.0, 0.0)
        corners_enu = [
            (center[0] - w / 2, center[1], center[2] + h / 2),
            (center[0] + w / 2, center[1], center[2] + h / 2),
            (center[0] + w / 2, center[1], center[2] - h / 2),
            (center[0] - w / 2, center[1], center[2] - h / 2),
        ]
        c1 = [enu_direction_to_pixel(*c, self.pose1, self.intr) for c in corners_enu]
        c2 = [enu_direction_to_pixel(c[0] - 1.0, c[1], c[2], self.pose2, self.intr)
              for c in corners_enu]
        size = triangulate_size(self.pose1, c1, self.pose2, c2, self.intr)
        self.assertAlmostEqual(size["w_mm"], 100.0, delta=1.0)
        self.assertAlmostEqual(size["h_mm"], 50.0, delta=1.0)
        self.assertLess(size["residual_m"], 1e-3)


if __name__ == "__main__":
    unittest.main()
