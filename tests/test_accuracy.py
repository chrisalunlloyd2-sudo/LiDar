import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from accuracy import mape, percent_error, run_benchmark  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_percent_error(self):
        self.assertAlmostEqual(percent_error(105.0, 100.0), 5.0)
        self.assertAlmostEqual(percent_error(95.0, 100.0), 5.0)

    def test_mape(self):
        self.assertAlmostEqual(mape([(105.0, 100.0), (90.0, 100.0)]), 7.5)

    def test_zero_truth_is_infinite(self):
        self.assertTrue(math.isinf(percent_error(1.0, 0.0)))


class TestBenchmark(unittest.TestCase):
    def test_clean_benchmark_is_accurate(self):
        report = run_benchmark(depth_m=2.0, noise_frac=0.0)
        self.assertEqual(len(report["results"]), 6)
        self.assertLess(report["mean_mape_pct"], 1.0)
        self.assertGreaterEqual(report["coverage_w_pct"], 0.0)
        self.assertLessEqual(report["coverage_w_pct"], 100.0)

    def test_noise_increases_error(self):
        clean = run_benchmark(depth_m=2.0, noise_frac=0.0)
        noisy = run_benchmark(depth_m=2.0, noise_frac=0.10)
        self.assertGreater(noisy["mean_mape_pct"], clean["mean_mape_pct"])

    def test_every_result_brackets_dimensions(self):
        report = run_benchmark(depth_m=3.0, noise_frac=0.02)
        for r in report["results"]:
            self.assertIn("ci_mm", r)
            self.assertLessEqual(r["ci_mm"]["w"][0], r["ci_mm"]["w"][1])
            self.assertIsNotNone(r["shape"]["shape"])

    def test_max_error_is_max(self):
        report = run_benchmark(depth_m=2.0, noise_frac=0.05)
        errors = [r["error_pct"]["w"] for r in report["results"]] + \
                 [r["error_pct"]["h"] for r in report["results"]]
        self.assertAlmostEqual(report["max_error_pct"], max(errors), places=2)


if __name__ == "__main__":
    unittest.main()
