import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from illusions import (  # noqa: E402
    ILLUSION_REGISTRY,
    as_flags,
    correct_size,
    evaluate,
)
from models import CameraIntrinsics, CueEstimate, Observation  # noqa: E402


def make_obs(**kwargs):
    base = dict(
        object_px_width=60.0, object_px_height=30.0,
        center_x=320.0, center_y=240.0,
        intrinsics=CameraIntrinsics.from_fov(640, 480, 60.0),
        convergence=0.02, arrowhead=0.02, specular=0.02,
        texture_density=0.45, contrast=0.45,
    )
    base.update(kwargs)
    return Observation(**base)


CLEAN = [CueEstimate("pixel_scaling", 3.0, 0.10), CueEstimate("angular_size", 3.0, 0.10)]


class TestRegistry(unittest.TestCase):
    def test_exactly_five(self):
        self.assertEqual(len(ILLUSION_REGISTRY), 5)

    def test_ids(self):
        ids = [e["id"] for e in ILLUSION_REGISTRY]
        self.assertEqual(ids, ["ponzo", "muller_lyer", "vertical_horizontal",
                               "ebbinghaus", "checker_shadow"])

    def test_every_entry_has_correction_math(self):
        for e in ILLUSION_REGISTRY:
            self.assertIn("formula", e)
            self.assertGreater(e["max_correction"], 0.0)
            self.assertIn(e["kind"], ("geometry", "light"))


class TestEvaluate(unittest.TestCase):
    def test_clean_near_unity(self):
        report = evaluate(make_obs(), CLEAN)
        self.assertGreater(report["combined_factor"], 0.98)
        self.assertLess(report["correction_percent"], 2.0)

    def test_ponzo_lowers_factor(self):
        base = evaluate(make_obs(), CLEAN)["combined_factor"]
        ponzo = evaluate(make_obs(convergence=0.9), CLEAN)["combined_factor"]
        self.assertLess(ponzo, base)

    def test_muller_lyer_lowers_factor(self):
        base = evaluate(make_obs(), CLEAN)["combined_factor"]
        ml = evaluate(make_obs(arrowhead=0.8), CLEAN)["combined_factor"]
        self.assertLess(ml, base)

    def test_vertical_horizontal(self):
        report = evaluate(make_obs(object_px_width=30.0, object_px_height=90.0), CLEAN)
        entry = [e for e in report["illusions"] if e["id"] == "vertical_horizontal"][0]
        self.assertGreater(entry["severity"], 0.0)

    def test_ebbinghaus_from_cue_conflict(self):
        cues = [CueEstimate("pixel_scaling", 2.0, 0.1), CueEstimate("angular_size", 5.0, 0.1)]
        report = evaluate(make_obs(), cues)
        entry = [e for e in report["illusions"] if e["id"] == "ebbinghaus"][0]
        self.assertGreater(entry["severity"], 0.0)

    def test_checker_shadow_from_light(self):
        report = evaluate(make_obs(contrast=0.0, specular=1.0), CLEAN)
        entry = [e for e in report["illusions"] if e["id"] == "checker_shadow"][0]
        self.assertGreater(entry["severity"], 0.5)

    def test_combined_bounded(self):
        report = evaluate(make_obs(convergence=1.0, arrowhead=1.0, contrast=0.0,
                                   specular=1.0, object_px_width=10.0, object_px_height=200.0),
                          [CueEstimate("pixel_scaling", 1.0, 0.1), CueEstimate("angular_size", 9.0, 0.1)])
        self.assertGreaterEqual(report["combined_factor"], 0.30)
        self.assertLessEqual(report["combined_factor"], 1.20)

    def test_top_three(self):
        report = evaluate(make_obs(convergence=0.9, arrowhead=0.8, contrast=0.0), CLEAN)
        self.assertLessEqual(len(report["top"]), 3)
        self.assertEqual(report["top"][0]["severity"],
                         max(e["severity"] for e in report["illusions"]))

    def test_as_flags(self):
        report = evaluate(make_obs(convergence=0.9), CLEAN)
        flags = as_flags(report)
        self.assertTrue(any(f.code == "ponzo" for f in flags))


class TestCorrectSize(unittest.TestCase):
    def test_scales_all_dims(self):
        w, h, d = correct_size((100.0, 50.0, 20.0), 0.9)
        self.assertAlmostEqual(w, 90.0)
        self.assertAlmostEqual(h, 45.0)
        self.assertAlmostEqual(d, 18.0)

    def test_none_thickness(self):
        w, h, d = correct_size((100.0, 50.0, None), 0.9)
        self.assertIsNone(d)


if __name__ == "__main__":
    unittest.main()
