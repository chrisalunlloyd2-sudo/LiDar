import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from illusion_filter import analyze, top_flags  # noqa: E402
from models import CameraIntrinsics, CueEstimate, Observation  # noqa: E402


def make_obs(**kwargs):
    base = dict(
        object_px_width=60.0, object_px_height=30.0,
        center_x=320.0, center_y=240.0,
        intrinsics=CameraIntrinsics.from_fov(640, 480, 60.0),
        convergence=0.05, arrowhead=None, specular=0.02,
        texture_density=0.5, contrast=0.4,
    )
    base.update(kwargs)
    return Observation(**base)


CLEAN_CUES = [
    CueEstimate("pixel_scaling", 3.0, 0.10),
    CueEstimate("angular_size", 3.05, 0.15),
]


class TestIllusionFilter(unittest.TestCase):
    def test_clean_observation_accepted(self):
        report = analyze(make_obs(), CLEAN_CUES)
        self.assertFalse(report.rejected)
        self.assertLess(report.risk, 0.3)
        self.assertEqual(report.flags, [])

    def test_severe_cue_conflict_rejected(self):
        cues = [CueEstimate("pixel_scaling", 2.0, 0.05), CueEstimate("ground_plane", 8.0, 0.05)]
        report = analyze(make_obs(), cues)
        self.assertTrue(report.rejected)
        codes = [f.code for f in report.flags]
        self.assertIn("cue_conflict", codes)
        self.assertGreaterEqual(report.flags[0].severity, 0.75)

    def test_far_ground_cue_ignored_for_conflict(self):
        cues = [
            CueEstimate("pixel_scaling", 1.0, 0.02),
            CueEstimate("angular_size", 1.0, 0.05),
            CueEstimate("ground_plane", 80.0, 20.0, 0.6),
        ]
        report = analyze(make_obs(), cues)
        self.assertFalse(report.rejected)
        self.assertNotIn("cue_conflict", [f.code for f in report.flags])

    def test_size_constancy_flag(self):
        cues = [CueEstimate("pixel_scaling", 2.0, 0.1), CueEstimate("angular_size", 5.0, 0.1)]
        report = analyze(make_obs(), cues)
        self.assertIn("size_constancy", [f.code for f in report.flags])

    def test_reflection_rejected(self):
        report = analyze(make_obs(specular=1.0, texture_density=0.0), CLEAN_CUES)
        self.assertTrue(report.rejected)
        self.assertIn("reflection", [f.code for f in report.flags])

    def test_ponzo_rejected(self):
        report = analyze(make_obs(convergence=0.9), CLEAN_CUES)
        self.assertTrue(report.rejected)
        self.assertIn("ponzo", [f.code for f in report.flags])

    def test_muller_lyer_flagged(self):
        report = analyze(make_obs(arrowhead=0.5), CLEAN_CUES)
        self.assertIn("muller_lyer", [f.code for f in report.flags])

    def test_low_contrast_rejected(self):
        report = analyze(make_obs(contrast=0.0), CLEAN_CUES)
        self.assertTrue(report.rejected)
        self.assertIn("aerial_perspective", [f.code for f in report.flags])

    def test_moderate_contrast_not_rejected(self):
        report = analyze(make_obs(contrast=0.05), CLEAN_CUES)
        self.assertIn("aerial_perspective", [f.code for f in report.flags])
        self.assertFalse(report.rejected)

    def test_foreshortening_flag(self):
        report = analyze(make_obs(object_px_width=10.0, object_px_height=100.0), CLEAN_CUES)
        self.assertIn("foreshortening", [f.code for f in report.flags])

    def test_risk_bounded(self):
        report = analyze(make_obs(specular=1.0, convergence=1.0, contrast=0.0,
                                  object_px_width=1.0, object_px_height=1.0), CLEAN_CUES)
        self.assertGreaterEqual(report.risk, 0.0)
        self.assertLessEqual(report.risk, 1.0)

    def test_top_flags_limit(self):
        report = analyze(make_obs(specular=1.0, convergence=0.8, arrowhead=0.7,
                                  contrast=0.0), CLEAN_CUES)
        self.assertLessEqual(len(top_flags(report, 2)), 2)
        severities = [f.severity for f in report.flags]
        self.assertEqual(severities, sorted(severities, reverse=True))


if __name__ == "__main__":
    unittest.main()
