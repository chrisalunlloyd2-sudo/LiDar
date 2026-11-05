import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fit_engine import (  # noqa: E402
    bbox_corners,
    fit_cuboid,
    fit_observation,
    fit_rectangle,
    nelder_mead,
    project_cuboid,
    project_rectangle,
)
from fixtures.premade_objects import by_tag, synthesize  # noqa: E402
from models import CameraIntrinsics  # noqa: E402


class TestNelderMead(unittest.TestCase):
    def test_quadratic(self):
        x, f = nelder_mead(lambda p: (p[0] - 3.0) ** 2 + (p[1] + 1.0) ** 2,
                           [0.0, 0.0], [1.0, 1.0], max_iter=200)
        self.assertAlmostEqual(x[0], 3.0, places=2)
        self.assertAlmostEqual(x[1], -1.0, places=2)
        self.assertLess(f, 1e-4)


class TestRectangleFit(unittest.TestCase):
    def setUp(self):
        self.intr = CameraIntrinsics.from_fov(640, 480, 60.0)

    def test_projection_counts(self):
        self.assertEqual(len(project_rectangle(100, 50, 0, 5.0, self.intr)), 4)
        self.assertEqual(len(project_cuboid(100, 100, 100, 0, 5.0, self.intr)), 8)

    def test_recovers_size(self):
        truth = project_rectangle(100.0, 50.0, 0.0, 5.0, self.intr)
        fit = fit_rectangle(truth, self.intr, 5.0, init=(80.0, 80.0, 0.0))
        self.assertAlmostEqual(fit["w_mm"], 100.0, delta=2.0)
        self.assertAlmostEqual(fit["h_mm"], 50.0, delta=2.0)
        self.assertAlmostEqual(fit["yaw_deg"], 0.0, delta=2.0)
        self.assertLess(fit["rms_px"], 1.0)
        self.assertTrue(fit["ok"])

    def test_off_center_bbox(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        # rectangle translated down by 40 px at 5 m
        center = (0.0, (280.0 - intr.cy) / intr.f_px * 5.0)
        truth = project_rectangle(120.0, 60.0, 0.0, 5.0, intr, center)
        fit = fit_rectangle(truth, intr, 5.0, init=(100.0, 100.0, 0.0))
        self.assertAlmostEqual(fit["w_mm"], 120.0, delta=3.0)
        self.assertAlmostEqual(fit["h_mm"], 60.0, delta=3.0)
        self.assertLess(fit["rms_px"], 1.0)

    def test_fit_observation(self):
        obj = by_tag("2x4")
        obs = synthesize(obj, 2.0)
        fit = fit_observation(obs, 2.0)
        self.assertAlmostEqual(fit["w_mm"], 88.9, delta=2.0)
        self.assertAlmostEqual(fit["h_mm"], 38.1, delta=2.0)
        self.assertLess(fit["rms_px"], 2.0)

    def test_too_few_corners(self):
        self.assertIsNone(fit_rectangle([(0, 0), (1, 1)], self.intr, 5.0))


class TestCuboidFit(unittest.TestCase):
    def setUp(self):
        self.intr = CameraIntrinsics.from_fov(640, 480, 60.0)

    def test_recovers_cube(self):
        truth = project_cuboid(100.0, 100.0, 100.0, 0.0, 5.0, self.intr)
        fit = fit_cuboid(truth, self.intr, 5.0, init=(80.0, 80.0, 80.0, 0.0))
        self.assertAlmostEqual(fit["w_mm"], 100.0, delta=4.0)
        self.assertAlmostEqual(fit["h_mm"], 100.0, delta=4.0)
        self.assertAlmostEqual(fit["d_mm"], 100.0, delta=4.0)
        self.assertLess(fit["rms_px"], 1.0)

    def test_too_few_corners(self):
        self.assertIsNone(fit_cuboid([(0, 0)] * 4, self.intr, 5.0))


if __name__ == "__main__":
    unittest.main()
