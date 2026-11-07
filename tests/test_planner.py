import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.multiview import refine_3d, simulate_views  # noqa: E402
from arena.planner import (  # noqa: E402
    acquisition_order,
    evaluate_plan,
    even_plan,
    extremes_plan,
    fov_cap_m,
    is_visible,
    plan_views,
    predicted_error_pct,
    random_plan,
    run_plan_benchmark,
    simulate_plan,
)
from fixtures.premade_objects import by_tag  # noqa: E402
from models import CameraIntrinsics  # noqa: E402

INTR = CameraIntrinsics.from_fov(640, 480, 60.0)
OBJ = by_tag("2x6")


class TestVisibility(unittest.TestCase):
    def test_cap_is_positive_and_within_the_frame(self):
        cap = fov_cap_m(INTR, 1.5, OBJ["w_mm"] / 1000.0)
        self.assertGreater(cap, 0.2)
        self.assertTrue(is_visible(INTR, 1.5, 0.1397, 0.0381, cap * 0.99))
        self.assertFalse(is_visible(INTR, 1.5, 0.1397, 0.0381, cap * 1.10))

    def test_wider_object_and_closer_depth_shrink_the_cap(self):
        narrow = fov_cap_m(INTR, 1.5, 0.05)
        wide = fov_cap_m(INTR, 1.5, 0.40)
        near = fov_cap_m(INTR, 0.5, 0.05)
        self.assertGreater(narrow, wide)
        self.assertGreater(narrow, near)

    def test_degenerate_depth_returns_zero(self):
        self.assertEqual(fov_cap_m(INTR, 0.0, 0.1), 0.0)
        self.assertEqual(fov_cap_m(INTR, -1.0, 0.1), 0.0)

    def test_offset_outside_the_cap_is_clipped_from_the_plan(self):
        cap = fov_cap_m(INTR, 1.5, OBJ["w_mm"] / 1000.0)
        views = simulate_plan(OBJ, [0.0, cap * 1.5], depth_m=1.5, noise_px=0.0)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["offset_m"], 0.0)


class TestPlans(unittest.TestCase):
    def test_even_plan_spans_the_arc(self):
        cap = 0.69
        plan = even_plan(cap, 6)
        self.assertEqual(len(plan), 6)
        self.assertAlmostEqual(plan[0], -cap, places=6)
        self.assertAlmostEqual(plan[-1], cap, places=6)
        self.assertEqual(len(set(plan)), 6)
        steps = [plan[i + 1] - plan[i] for i in range(5)]
        for s in steps:
            self.assertAlmostEqual(s, steps[0], places=6)
        self.assertAlmostEqual(even_plan(cap, 5)[2], 0.0, places=6)

    def test_single_view_is_the_centre(self):
        self.assertEqual(even_plan(0.5, 1), [0.0])
        self.assertEqual(even_plan(0.5, 0), [0.0])

    def test_acquisition_starts_at_the_centre_and_expands(self):
        order = acquisition_order(even_plan(0.6, 7))
        self.assertEqual(order[0], 0.0)
        mags = [abs(b) for b in order]
        self.assertEqual(mags, sorted(mags))
        self.assertAlmostEqual(mags[-1], 0.6, places=6)

    def test_extremes_alternate_signs_within_the_cap(self):
        plan = extremes_plan(0.6, 4)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0], 0.0)
        self.assertEqual(plan[1], 0.6)
        self.assertEqual(plan[2], -0.6)
        for b in plan:
            self.assertLessEqual(abs(b), 0.6 + 1e-9)

    def test_random_plan_is_bounded(self):
        plan = random_plan(0.4, 5)
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan, sorted(plan))
        self.assertTrue(all(abs(b) <= 0.4 for b in plan))


class TestPrediction(unittest.TestCase):
    def test_more_views_predict_a_smaller_error(self):
        cap = fov_cap_m(INTR, 1.5, 0.1397)
        two = predicted_error_pct(INTR, 1.5, 38.1, even_plan(cap, 2))
        eight = predicted_error_pct(INTR, 1.5, 38.1, even_plan(cap, 8))
        self.assertLess(eight, two)

    def test_baseline_matters_at_fixed_view_count(self):
        collapsed = predicted_error_pct(INTR, 1.5, 38.1, [0.0, 0.001])
        spread = predicted_error_pct(INTR, 1.5, 38.1, [0.0, 0.6])
        self.assertLess(spread, collapsed)

    def test_zero_baseline_has_no_prediction(self):
        self.assertIsNone(predicted_error_pct(INTR, 1.5, 38.1, [0.0, 0.0]))

    def test_plan_steps_improve_with_each_view(self):
        plan = plan_views(INTR, 1.5, 6, OBJ["w_mm"], OBJ["h_mm"])
        preds = [s["predicted_error_pct"] for s in plan["steps"]]
        self.assertEqual(len(preds), 6)
        for a, b in zip(preds, preds[1:]):
            self.assertLessEqual(b, a)
        innermost = min(abs(b) for b in plan["offsets_m"])
        self.assertAlmostEqual(abs(plan["steps"][0]["offset_m"]), innermost,
                               places=6)
        self.assertGreater(plan["fov_cap_m"], 0.0)
        self.assertTrue(all(abs(b) <= plan["fov_cap_m"] + 1e-9
                            for b in plan["offsets_m"]))


class TestMeasured(unittest.TestCase):
    def test_even_plan_improves_with_budget(self):
        cap = fov_cap_m(INTR, 1.5, OBJ["w_mm"] / 1000.0)
        two = evaluate_plan(OBJ, even_plan(cap, 2), seeds=8)
        eight = evaluate_plan(OBJ, even_plan(cap, 8), seeds=8)
        self.assertIsNotNone(two["median_pct"])
        self.assertLess(eight["median_pct"], two["median_pct"])

    def test_even_beats_random_at_the_same_budget(self):
        cap = fov_cap_m(INTR, 1.5, OBJ["w_mm"] / 1000.0)
        even = evaluate_plan(OBJ, even_plan(cap, 6), seeds=10)
        rand = evaluate_plan(OBJ, random_plan(cap, 6), seeds=10)
        self.assertLess(even["median_pct"], rand["median_pct"])

    def test_anchor_only_cannot_triangulate(self):
        result = evaluate_plan(OBJ, [0.0, 0.0, 0.0], seeds=4)
        self.assertIsNone(result["median_pct"])
        self.assertEqual(result["skipped"], 4)

    def test_measurement_is_close_to_the_prediction(self):
        cap = fov_cap_m(INTR, 1.5, OBJ["w_mm"] / 1000.0)
        offsets = even_plan(cap, 6)
        predicted = predicted_error_pct(INTR, 1.5, OBJ["h_mm"], offsets)
        measured = evaluate_plan(OBJ, offsets, seeds=10)["median_pct"]
        ratio = measured / predicted
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 2.5)

    def test_degenerate_pairs_are_rejected(self):
        views = simulate_views(OBJ, depth_m=1.5, n_views=3,
                               baseline_m=0.0, noise_px=0.3, seed=2)
        self.assertIsNone(refine_3d(views))


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_plan_benchmark(budgets=(2, 4), seeds=4,
                                        fixtures=[OBJ])

    def test_shape(self):
        self.assertIn("strategies", self.report)
        self.assertIn("calibration", self.report)
        self.assertEqual(len(self.report["rows"]), 4 * 2)
        for name in ("even", "extremes", "random", "anchor"):
            self.assertIn(name, self.report["strategies"])

    def test_anchor_never_solves(self):
        anchor = self.report["strategies"]["anchor"]["by_budget"]
        for cell in anchor.values():
            self.assertEqual(cell["solved"], 0)

    def test_even_plan_solves_every_cell(self):
        even = self.report["strategies"]["even"]["by_budget"]
        for cell in even.values():
            self.assertEqual(cell["solved"], cell["cells"])
            self.assertIsNotNone(cell["hard_median_pct"])

    def test_calibration_ratio_is_finite(self):
        cal = self.report["calibration"]
        self.assertTrue(math.isfinite(cal["median_ratio"]))
        self.assertGreater(cal["median_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
