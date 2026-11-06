import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.hardware_probe import probe, stress, suggest  # noqa: E402


class TestHardwareProbe(unittest.TestCase):
    def test_probe_returns_limits(self):
        result = probe(seconds=0.05, workers=2)
        for key in ("cpu_count", "mem_total_mb", "single_core_ops_per_sec",
                    "fusion_ops_per_sec", "fusion_peak_mem_mb", "budget"):
            self.assertIn(key, result)
        self.assertGreater(result["cpu_count"], 0)
        self.assertGreater(result["fusion_ops_per_sec"], 0.0)
        self.assertGreaterEqual(result["fusion_peak_mem_mb"], 0.0)
        self.assertIn("recommended_threads", result["budget"])

    def test_stress_is_bounded_and_positive(self):
        result = stress(seconds=0.05, workers=2)
        self.assertEqual(result["workers"], 2)
        self.assertGreater(result["iterations"], 0)
        self.assertGreater(result["ops_per_sec"], 0.0)

    def test_suggest_from_report(self):
        report = probe(seconds=0.05, workers=1)
        report["fusers"] = [{"method": "inverse_variance"}, {"method": "huber"}]
        report["sizes"] = [{"method": "pixel_scale"}]
        report["illusions"] = [{"method": "canonical5"}]
        report["triangulators"] = [{"method": "midpoint"}]
        picked = suggest(report, top_k=1)
        self.assertEqual(picked["keep"]["fusers"], ["inverse_variance"])
        self.assertEqual(picked["keep"]["sizes"], ["pixel_scale"])
        self.assertIn("budget", picked)


if __name__ == "__main__":
    unittest.main()
