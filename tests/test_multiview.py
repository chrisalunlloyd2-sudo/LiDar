import os
import statistics
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.multiview import (  # noqa: E402
    refine_3d,
    refine_aligned,
    refine_temporal,
    refine_triangulated,
    run_multiview_benchmark,
    simulate_views,
)
from fixtures.premade_objects import by_tag  # noqa: E402


class TestSimulateViews(unittest.TestCase):
    def test_view_count_and_corners(self):
        obj = by_tag("2x6")
        views = simulate_views(obj, depth_m=1.5, n_views=5, baseline_m=0.2, noise_px=0.0)
        self.assertEqual(len(views), 5)
        for v in views:
            self.assertEqual(len(v["corners"]), 4)
            self.assertIn("pose", v)
            self.assertIn("intrinsics", v)

    def test_minimum_two_views(self):
        views = simulate_views(by_tag("2x4"), n_views=1)
        self.assertEqual(len(views), 2)


class TestRefiners(unittest.TestCase):
    def setUp(self):
        self.obj = by_tag("2x6")
        self.views = simulate_views(self.obj, depth_m=1.5, n_views=9,
                                    baseline_m=0.2, noise_px=0.3, seed=1)

    def test_temporal(self):
        r = refine_temporal(self.views, self.obj["w_mm"], 1.5)
        self.assertGreater(r["w_mm"], 0.0)
        self.assertGreater(r["h_mm"], 0.0)

    def test_aligned(self):
        r = refine_aligned(self.views, self.obj["w_mm"], 1.5)
        self.assertGreater(r["w_mm"], 0.0)
        self.assertGreater(r["h_mm"], 0.0)

    def test_triangulated(self):
        for method in ("ransac", "midpoint"):
            r = refine_triangulated(self.views, method)
            self.assertIsNotNone(r)
            self.assertGreater(r["w_mm"], 0.0)
            self.assertGreater(r["h_mm"], 0.0)

    def test_three_d(self):
        r = refine_3d(self.views)
        self.assertIsNotNone(r)
        self.assertGreater(r["w_mm"], 0.0)
        self.assertGreater(r["h_mm"], 0.0)


class TestMultiviewBenchmark(unittest.TestCase):
    def test_three_d_drives_2x6_height_below_one_percent(self):
        report = run_multiview_benchmark(n_views=15, baseline_m=0.2,
                                         noise_px=0.3, depth_m=1.5)
        row = next(r for r in report["results"] if r["tag"] == "2x6")
        self.assertLess(row["three_d_error_pct"]["h"], 1.0)
        self.assertLess(report["three_d_mape_pct"], 1.0)

    def test_more_views_reduce_three_d_error(self):
        obj = by_tag("2x6")
        e5, e15 = [], []
        for seed in range(3):
            v5 = simulate_views(obj, depth_m=1.5, n_views=5,
                                baseline_m=0.2, noise_px=0.3, seed=seed)
            v15 = simulate_views(obj, depth_m=1.5, n_views=15,
                                 baseline_m=0.2, noise_px=0.3, seed=seed)
            e5.append(abs(refine_3d(v5)["h_mm"] - obj["h_mm"]))
            e15.append(abs(refine_3d(v15)["h_mm"] - obj["h_mm"]))
        self.assertLess(statistics.median(e15), statistics.median(e5))


if __name__ == "__main__":
    unittest.main()
