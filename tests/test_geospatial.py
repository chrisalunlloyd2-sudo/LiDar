import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geospatial import (  # noqa: E402
    enu_to_geodetic,
    ground_plane_point,
    haversine_m,
    initial_bearing_deg,
    pixel_to_ray,
    project_point,
    ray_camera_to_enu,
    uncertainty_radius_m,
    wgs84_radii,
)
from models import CameraIntrinsics, CameraPose  # noqa: E402


class TestRadiiAndGeodesy(unittest.TestCase):
    def test_radii_positive(self):
        r_n, r_e = wgs84_radii(0.0)
        self.assertGreater(r_n, 6_300_000)
        self.assertGreater(r_e, r_n)
        self.assertAlmostEqual(r_e, 6378137.0, delta=1.0)

    def test_north_offset_one_degree(self):
        r_n, _ = wgs84_radii(0.0)
        lat, lon, alt = enu_to_geodetic(0.0, 0.0, 0.0, 0.0, math.radians(1.0) * r_n, 0.0)
        self.assertAlmostEqual(lat, 1.0, places=6)
        self.assertAlmostEqual(lon, 0.0, places=6)
        self.assertAlmostEqual(alt, 0.0, places=6)

    def test_east_offset_one_degree(self):
        _, r_e = wgs84_radii(0.0)
        lat, lon, _ = enu_to_geodetic(0.0, 0.0, 0.0, math.radians(1.0) * r_e, 0.0, 0.0)
        self.assertAlmostEqual(lat, 0.0, places=6)
        self.assertAlmostEqual(lon, 1.0, places=6)

    def test_altitude_offset(self):
        _, _, alt = enu_to_geodetic(10.0, 20.0, 100.0, 0.0, 0.0, 2.5)
        self.assertAlmostEqual(alt, 102.5, places=6)

    def test_haversine_one_degree(self):
        self.assertAlmostEqual(haversine_m(0, 0, 0, 1), 111195, delta=500)

    def test_haversine_symmetric(self):
        self.assertAlmostEqual(haversine_m(1, 2, 3, 4), haversine_m(3, 4, 1, 2), places=6)

    def test_bearing(self):
        self.assertAlmostEqual(initial_bearing_deg(0, 0, 1, 0), 0.0, places=3)
        self.assertAlmostEqual(initial_bearing_deg(0, 0, 0, 1), 90.0, places=3)


class TestRays(unittest.TestCase):
    def test_center_pixel_ray(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        self.assertEqual(pixel_to_ray(intr.cx, intr.cy, intr), (0.0, 0.0, 1.0))

    def test_forward_ray_north(self):
        e, n, u = ray_camera_to_enu((0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
        self.assertAlmostEqual(e, 0.0, places=6)
        self.assertAlmostEqual(n, 1.0, places=6)
        self.assertAlmostEqual(u, 0.0, places=6)

    def test_forward_ray_east(self):
        e, n, u = ray_camera_to_enu((0.0, 0.0, 1.0), 90.0, 0.0, 0.0)
        self.assertAlmostEqual(e, 1.0, places=6)
        self.assertAlmostEqual(n, 0.0, places=6)

    def test_pitch_up_raises_ray(self):
        _, _, u = ray_camera_to_enu((0.0, 0.0, 1.0), 0.0, 20.0, 0.0)
        self.assertGreater(u, 0.0)

    def test_downward_pixel(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        _, _, u = ray_camera_to_enu(pixel_to_ray(intr.cx, intr.cy + 100, intr), 0.0, 0.0, 0.0)
        self.assertLess(u, 0.0)


class TestProjectPoint(unittest.TestCase):
    def setUp(self):
        self.intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        self.pose = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                               pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True)

    def test_center_ray_projects_north(self):
        g = project_point(self.pose, self.intr, self.intr.cx, self.intr.cy, 10.0, sigma_depth_m=0.2)
        self.assertAlmostEqual(g.north_m, 10.0, places=4)
        self.assertAlmostEqual(g.east_m, 0.0, places=4)
        self.assertAlmostEqual(g.distance_m, 10.0, places=4)
        self.assertAlmostEqual(g.bearing_deg, 0.0, places=3)
        self.assertGreater(g.lat, 0.0)
        self.assertAlmostEqual(g.lon, 0.0, places=6)

    def test_heading_east(self):
        self.pose.heading_deg = 90.0
        g = project_point(self.pose, self.intr, self.intr.cx, self.intr.cy, 10.0)
        self.assertAlmostEqual(g.east_m, 10.0, places=4)
        self.assertAlmostEqual(g.north_m, 0.0, places=4)
        self.assertAlmostEqual(g.bearing_deg, 90.0, places=3)

    def test_radius_combines_errors(self):
        g = project_point(self.pose, self.intr, self.intr.cx, self.intr.cy, 10.0,
                          sigma_depth_m=0.3, sigma_angle_deg=2.0)
        expected = uncertainty_radius_m(10.0, 0.3, 2.0, 5.0)
        self.assertAlmostEqual(g.radius_m, expected, places=6)
        self.assertGreater(g.radius_m, 5.0)


class TestGroundPlane(unittest.TestCase):
    def test_downward_pixel_hits_floor(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        pose = CameraPose(lat=45.0, lon=10.0, alt_m=50.0, heading_deg=0.0,
                          pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True)
        gp = ground_plane_point(pose, intr, intr.cx, intr.cy + 100.0, 2.0)
        self.assertIsNotNone(gp)
        self.assertAlmostEqual(gp.up_m, -2.0, places=6)
        self.assertAlmostEqual(gp.alt_m, 48.0, places=6)
        self.assertGreater(gp.distance_m, 0.0)
        self.assertAlmostEqual(gp.north_m, gp.distance_m, places=3)

    def test_upward_pixel_returns_none(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        pose = CameraPose(pitch_deg=0.0)
        self.assertIsNone(ground_plane_point(pose, intr, intr.cx, intr.cy - 100.0, 1.5))


if __name__ == "__main__":
    unittest.main()
