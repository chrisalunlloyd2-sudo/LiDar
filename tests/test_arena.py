import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.arena import run_arena, snip  # noqa: E402
from arena.depth_fusers import FUSERS  # noqa: E402
from arena.illusion_models import ALL_ILLUSIONS, ILLUSION_MODELS  # noqa: E402
from arena.size_estimators import SIZE_ESTIMATORS  # noqa: E402
from arena.triangulators import TRIANGULATORS  # noqa: E402
from depth_engine import DepthEngine  # noqa: E402
from fixtures.premade_objects import PREMADE, by_tag, synthesize  # noqa: E402
from geospatial import enu_to_geodetic  # noqa: E402
from models import CameraIntrinsics, CameraPose  # noqa: E402
from triangulate import enu_direction_to_pixel  # noqa: E402

INTR = CameraIntrinsics.from_fov(640, 480, 60.0)


def _cues():
    obs = synthesize(by_tag("2x4"), 2.0)
    return obs, DepthEngine().cues(obs)


class TestDepthFusers(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(FUSERS), 15)

    def test_all_finite_and_ordered(self):
        obs, cues = _cues()
        for name, fn in FUSERS.items():
            r = fn(cues, 0.95)
            self.assertTrue(math.isfinite(r["depth_m"]), name)
            self.assertGreater(r["depth_m"], 0.0, name)
            self.assertLessEqual(r["ci_low_m"], r["depth_m"] + 1e-9, name)
            self.assertGreaterEqual(r["ci_high_m"], r["depth_m"] - 1e-9, name)
            self.assertGreaterEqual(r["confidence"], 0.0, name)
            self.assertLessEqual(r["confidence"], 1.0, name)
            self.assertAlmostEqual(r["depth_m"], 2.0, delta=1.0, msg=name)

    def test_inverse_variance_is_accurate(self):
        obs, cues = _cues()
        r = FUSERS["inverse_variance"](cues, 0.95)
        self.assertAlmostEqual(r["depth_m"], 2.0, delta=0.05)


class TestSizeEstimators(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(SIZE_ESTIMATORS), 10)

    def test_all_positive(self):
        obs, cues = _cues()
        for name, fn in SIZE_ESTIMATORS.items():
            dims = fn(obs, 2.0, 0.95)
            self.assertGreater(dims["w_mm"], 0.0, name)
            self.assertGreater(dims["h_mm"], 0.0, name)
            self.assertTrue(math.isfinite(dims["w_mm"]), name)
            self.assertTrue(math.isfinite(dims["h_mm"]), name)


class TestIllusionModels(unittest.TestCase):
    def test_three_models(self):
        self.assertEqual(set(ILLUSION_MODELS), {"none", "canonical5", "extended10"})

    def test_extended_has_ten(self):
        self.assertEqual(len(ALL_ILLUSIONS), 10)
        obs, cues = _cues()
        report = ILLUSION_MODELS["extended10"](obs, cues)
        self.assertEqual(len(report["illusions"]), 10)

    def test_none_is_unity(self):
        obs, cues = _cues()
        self.assertEqual(ILLUSION_MODELS["none"](obs, cues)["combined_factor"], 1.0)

    def test_canonical_bounded(self):
        obs, cues = _cues()
        report = ILLUSION_MODELS["canonical5"](obs, cues)
        self.assertLessEqual(report["combined_factor"], 1.0)


class TestTriangulators(unittest.TestCase):
    def setUp(self):
        self.pose1 = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                                pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
        lat2, lon2, alt2 = enu_to_geodetic(0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
        self.pose2 = CameraPose(lat=lat2, lon=lon2, alt_m=alt2, heading_deg=0.0,
                                pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)

    def test_count(self):
        self.assertEqual(len(TRIANGULATORS), 5)

    def test_all_recover_size(self):
        obj = by_tag("2x4")
        w = obj["w_mm"] / 1000.0
        h = obj["h_mm"] / 1000.0
        depth = 5.0
        corners = [(-w / 2, depth, h / 2), (w / 2, depth, h / 2),
                   (w / 2, depth, -h / 2), (-w / 2, depth, -h / 2)]
        c1 = [enu_direction_to_pixel(*c, self.pose1, INTR) for c in corners]
        c2 = [enu_direction_to_pixel(c[0] - 0.5, c[1], c[2], self.pose2, INTR) for c in corners]
        for name, fn in TRIANGULATORS.items():
            size = fn(self.pose1, c1, self.pose2, c2, INTR)
            self.assertIsNotNone(size, name)
            self.assertAlmostEqual(size["w_mm"], obj["w_mm"], delta=obj["w_mm"] * 0.10, msg=name)
            self.assertAlmostEqual(size["h_mm"], obj["h_mm"], delta=obj["h_mm"] * 0.10, msg=name)


class TestArena(unittest.TestCase):
    def test_quick_arena_ranks_and_snips(self):
        report = run_arena(quick=True, top_k=2)
        for stage in ("fusers", "sizes", "illusions", "triangulators"):
            rows = report[stage]
            self.assertGreater(len(rows), 0)
            scores = [r["score"] for r in rows]
            self.assertEqual(scores, sorted(scores))
            self.assertGreaterEqual(len(report["survivors"][stage]), 1)
            kept = set(report["survivors"][stage])
            pruned = set(report["pruned"][stage])
            self.assertEqual(kept & pruned, set())
            self.assertEqual(kept | pruned, {r["method"] for r in rows})

    def test_snip_matches_report(self):
        report = run_arena(quick=True, top_k=2)
        cut = snip(report, top_k=2)
        self.assertEqual(cut["keep"]["fusers"], report["survivors"]["fusers"])
        self.assertEqual(cut["prune"]["fusers"], report["pruned"]["fusers"])


if __name__ == "__main__":
    unittest.main()
