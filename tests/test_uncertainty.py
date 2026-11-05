import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import CueEstimate  # noqa: E402
from uncertainty import (  # noqa: E402
    bootstrap_ci,
    fuse_cues,
    informative_cues,
    interval_width_ratio,
    z_for_level,
)


class TestZScore(unittest.TestCase):
    def test_95_percent(self):
        self.assertAlmostEqual(z_for_level(0.95), 1.959964, places=4)

    def test_monotonic(self):
        self.assertLess(z_for_level(0.80), z_for_level(0.95))
        self.assertLess(z_for_level(0.95), z_for_level(0.99))


class TestFuseCues(unittest.TestCase):
    def test_single_cue_keeps_point_and_sigma(self):
        est = fuse_cues([CueEstimate("a", 3.0, 0.3)])
        self.assertAlmostEqual(est.depth_m, 3.0, places=6)
        self.assertAlmostEqual(est.sigma_m, 0.3, places=6)
        self.assertLess(est.ci_low_m, est.depth_m)
        self.assertGreater(est.ci_high_m, est.depth_m)

    def test_two_identical_cues_reduce_sigma(self):
        est = fuse_cues([CueEstimate("a", 3.0, 1.0), CueEstimate("b", 3.0, 1.0)])
        self.assertAlmostEqual(est.depth_m, 3.0, places=6)
        self.assertAlmostEqual(est.sigma_m, math.sqrt(0.5), places=6)

    def test_disagreement_inflates_interval(self):
        agree = fuse_cues([CueEstimate("a", 3.0, 0.1), CueEstimate("b", 3.0, 0.1)])
        disagree = fuse_cues([CueEstimate("a", 2.0, 0.1), CueEstimate("b", 4.0, 0.1)])
        self.assertGreater(disagree.sigma_m, agree.sigma_m)
        self.assertGreater(disagree.ci_high_m - disagree.ci_low_m,
                           agree.ci_high_m - agree.ci_low_m)

    def test_confidence_bounded(self):
        est = fuse_cues([CueEstimate("a", 1.0, 10.0)])
        self.assertGreaterEqual(est.confidence, 0.0)
        self.assertLessEqual(est.confidence, 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fuse_cues([])

    def test_zero_depth_cue_ignored(self):
        est = fuse_cues([CueEstimate("a", 0.0, 0.1), CueEstimate("b", 2.0, 0.2)])
        self.assertAlmostEqual(est.depth_m, 2.0, places=6)


class TestInformativeCues(unittest.TestCase):
    def test_far_ground_cue_is_dropped(self):
        cues = [
            CueEstimate("pixel", 1.0, 0.02, 1.0),
            CueEstimate("ground", 80.0, 20.0, 0.6),
        ]
        names = [c.name for c in informative_cues(cues)]
        self.assertIn("pixel", names)
        self.assertNotIn("ground", names)

    def test_far_ground_cue_does_not_inflate_interval(self):
        cues = [
            CueEstimate("pixel", 1.0, 0.02, 1.0),
            CueEstimate("angular", 1.0, 0.05, 0.8),
            CueEstimate("ground", 80.0, 20.0, 0.6),
        ]
        est = fuse_cues(cues)
        self.assertAlmostEqual(est.depth_m, 1.0, places=3)
        self.assertLess(est.sigma_m, 0.1)


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_brackets_point(self):
        cues = [CueEstimate("a", 3.0, 0.2), CueEstimate("b", 3.2, 0.3)]
        lo, hi = bootstrap_ci(cues, level=0.95, n=200, seed=7)
        self.assertLessEqual(lo, hi)
        self.assertLess(lo, 3.2)
        self.assertGreater(hi, 3.0)

    def test_interval_width_ratio(self):
        est = fuse_cues([CueEstimate("a", 4.0, 0.4)])
        self.assertAlmostEqual(interval_width_ratio(est), est.ci_half_m / 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
