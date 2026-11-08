import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import flash_lidar as fl  # noqa: E402
from models import CameraIntrinsics, Observation  # noqa: E402

INTR = CameraIntrinsics.from_fov(640, 480, 60.0)
COLS, ROWS = 64, 40


def wall_frame(depth_m=2.0, cols=COLS, rows=ROWS, seed=0, noise_dn=1.0):
    scene = fl.FlashScene("flat wall", [], wall_depth_m=depth_m)
    return fl.render_scene(scene, INTR, cols=cols, rows=rows, seed=seed,
                           noise_dn=noise_dn)


def board_frame(depth_m=1.5, cols=COLS, rows=ROWS, seed=0):
    scene = fl.FlashScene("centred 2x6", [
        fl.ScenePatch("2x6", "2x6 lumber", 0.0, 0.0, 0.1397, 0.62, depth_m),
    ])
    return fl.render_scene(scene, INTR, cols=cols, rows=rows, seed=seed)


class TestDifferential(unittest.TestCase):
    def test_subtracts_ambient_and_clips_noise(self):
        diff = fl.differential_map([10.0, 20.0, 30.0], [40.0, 15.0, 30.0])
        self.assertEqual(diff, [30.0, 0.0, 0.0])

    def test_grid_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fl.differential_map([1.0, 2.0], [1.0])


class TestInverseSquare(unittest.TestCase):
    def setUp(self):
        self.cal = fl.FlashCalibration(reference_depth_m=2.0,
                                       reference_delta_dn=100.0,
                                       sigma_delta_dn=1.0,
                                       albedo_rel_sigma=0.0)

    def test_reference_flux_returns_reference_depth(self):
        depth, sigma, reason = fl.range_from_flux(100.0, self.cal)
        self.assertAlmostEqual(depth, 2.0, places=6)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(sigma, 2.0 * 0.5 * (1.0 / 100.0), places=9)

    def test_quarter_flux_doubles_range(self):
        depth, _, reason = fl.range_from_flux(25.0, self.cal)
        self.assertAlmostEqual(depth, 4.0, places=6)
        self.assertEqual(reason, "ok")

    def test_dark_cell_abstains(self):
        depth, sigma, reason = fl.range_from_flux(1.0, self.cal)
        self.assertIsNone(depth)
        self.assertIsNone(sigma)
        self.assertEqual(reason, "dark")

    def test_too_near_and_too_far_flagged(self):
        near = fl.range_from_flux(100000.0, self.cal)
        far = fl.range_from_flux(4.0, self.cal)
        self.assertEqual(near[2], "too_near")
        self.assertIsNone(near[1])
        self.assertEqual(far[2], "too_far")

    def test_albedo_prior_widens_sigma(self):
        clean, _, _ = fl.range_from_flux(100.0, self.cal)
        noisy = fl.FlashCalibration(2.0, 100.0, sigma_delta_dn=1.0,
                                    albedo_rel_sigma=0.30)
        _, widened, _ = fl.range_from_flux(100.0, noisy)
        _, plain, _ = fl.range_from_flux(100.0, self.cal)
        self.assertGreater(widened, plain)

    def test_calibration_constant(self):
        self.assertAlmostEqual(self.cal.constant(), 400.0, places=6)


class TestFrameAnalysis(unittest.TestCase):
    def test_cell_centres_row_major(self):
        centres = fl.cell_centres(2, 2, INTR)
        self.assertEqual(len(centres), 4)
        self.assertLess(centres[0][0], centres[1][0])
        self.assertLess(centres[0][1], centres[2][1])

    def test_saturated_cells_are_invalid(self):
        cal = fl.FlashCalibration(2.0, 100.0)
        ambient = [10.0] * 4
        flash = [255.0, 110.0, 10.0, 60.0]
        cells = fl.analyse_frame(ambient, flash, 2, 2, cal, INTR)
        self.assertFalse(cells[0]["valid"])
        self.assertEqual(cells[0]["reason"], "saturated")
        self.assertTrue(cells[1]["valid"])
        self.assertEqual(cells[2]["reason"], "dark")
        self.assertAlmostEqual(fl.valid_fraction(cells), 0.5)

    def test_flat_wall_recovers_one_plane(self):
        frame = wall_frame(2.0)
        cal = fl.calibrate_scene(frame, INTR, sample="wall")
        result = fl.analyse_grid(frame["ambient"], frame["flash"], COLS, ROWS,
                                 INTR, cal)
        self.assertEqual(len(result["planes"]), 1)
        plane = result["planes"][0]
        self.assertEqual(plane["cells"], COLS * ROWS)
        # the normal is oriented toward the camera, so the offset is negative
        self.assertAlmostEqual(abs(plane["offset_m"]), 2.0, delta=0.12)
        self.assertGreater(plane["length_m"], 1.8)
        self.assertLess(plane["length_m"], 2.6)

    def test_grid_length_is_enforced(self):
        cal = fl.FlashCalibration(2.0, 100.0)
        with self.assertRaises(ValueError):
            fl.analyse_grid([1.0], [1.0], 4, 4, INTR, cal)


class TestTemporalFusion(unittest.TestCase):
    def test_identical_frames_converge_and_shrink(self):
        cell = {"index": 0, "depth_m": 2.0, "sigma_m": 0.05, "valid": True}
        frames = [[dict(cell)] for _ in range(9)]
        fused = fl.fuse_depth_frames(frames, min_frames=3)
        self.assertTrue(fused[0]["converged"])
        self.assertAlmostEqual(fused[0]["depth_m"], 2.0, places=6)
        self.assertAlmostEqual(fused[0]["sigma_m"], 0.05 / 3.0, places=6)
        self.assertEqual(fused[0]["n"], 9)

    def test_disagreeing_frames_do_not_converge(self):
        frames = [[{"index": 0, "depth_m": d, "sigma_m": 0.02, "valid": True}]
                  for d in (1.0, 3.0, 5.0)]
        fused = fl.fuse_depth_frames(frames, min_frames=3)
        self.assertFalse(fused[0]["converged"])
        self.assertGreater(fused[0]["spread_m"], 0.5)

    def test_invalid_cells_are_preserved(self):
        frames = [[{"index": 0, "depth_m": None, "sigma_m": None, "valid": False}]
                  for _ in range(3)]
        fused = fl.fuse_depth_frames(frames)
        self.assertFalse(fused[0]["valid"])
        self.assertEqual(fused[0]["n"], 0)


class TestTargetMatching(unittest.TestCase):
    def plane(self, length_m, width_m, rms_m=0.01, sigma_m=0.03):
        return {"length_m": length_m, "width_m": width_m,
                "aspect": max(length_m, width_m) / min(length_m, width_m),
                "rms_m": rms_m, "median_sigma_m": sigma_m,
                "cell_m": 0.025, "cells": 150}

    def test_exact_lumber_face_matches(self):
        match = fl.match_target(self.plane(0.62, 0.1397))
        self.assertEqual(match["tag"], "2x6")
        self.assertEqual(match["reason"], "ok")

    def test_between_two_widths_abstains(self):
        match = fl.match_target(self.plane(0.50, 0.16))
        self.assertIsNone(match["tag"])
        self.assertEqual(match["reason"], "ambiguous")
        self.assertLessEqual(match["confidence"], 0.5)

    def test_unknown_cross_section_abstains(self):
        match = fl.match_target(self.plane(0.30, 0.055))
        self.assertIsNone(match["tag"])
        self.assertEqual(match["reason"], "no_catalog_match")

    def test_shape_prior_blocks_square_face_for_a_board(self):
        # 140 mm wide but square-ish: aspect 1.1 cannot be 2x6 lumber
        match = fl.match_target(self.plane(0.154, 0.1397))
        self.assertNotEqual(match["tag"], "2x6")


class TestDemoScene(unittest.TestCase):
    def test_depth_accuracy_and_targets(self):
        result = fl.run_demo(COLS, ROWS, seed=0, width=COLS)
        self.assertEqual(len(result["cells"]), COLS * ROWS)
        self.assertEqual(result["valid_pct"], 100.0)
        self.assertLess(result["accuracy"]["mape_pct"], 2.5)
        self.assertEqual(len(result["planes"]), 3)
        self.assertEqual(sorted(t["tag"] for t in result["targets"]),
                         ["2x6", "2x8"])
        self.assertEqual(len(result["ascii"]), int(COLS * ROWS / COLS * 0.5))

    def test_mixed_albedo_degrades_honestly(self):
        uniform = fl.run_demo(96, 54, mixed_albedo=False, seed=0, width=8)
        mixed = fl.run_demo(96, 54, mixed_albedo=True, seed=0, width=8)
        self.assertGreater(mixed["accuracy"]["mape_pct"],
                           1.5 * uniform["accuracy"]["mape_pct"])
        self.assertTrue(mixed["mixed_albedo"])
        # a 0.30 albedo face reads sqrt(0.55/0.30) = 35 % too far, so the near
        # 2x6's 140 mm face measures ~103 mm and drops out of the catalog.
        self.assertNotIn("2x6", [t["tag"] for t in mixed["targets"]])

    def test_flash_cue_merges_with_monocular_depth(self):
        frame = board_frame(1.5)
        cal = fl.calibrate_scene(frame, INTR, sample="wall")
        f_px = INTR.f_px
        obs = Observation(
            object_px_width=88.9 * f_px / 1500.0,
            object_px_height=38.1 * f_px / 1500.0,
            center_x=INTR.cx, center_y=INTR.cy, intrinsics=INTR,
            known_width_mm=88.9, camera_height_m=1.5,
        )
        merged = fl.estimate_with_flash(obs, frame["ambient"], frame["flash"],
                                        cal, COLS, ROWS)
        names = [c.name for c in merged["estimate"].cues]
        self.assertIn("flash_falloff", names)
        self.assertAlmostEqual(merged["flash_cue"].depth_m, 1.5, delta=0.15)
        self.assertAlmostEqual(merged["estimate"].depth_m, 1.5, delta=0.15)
        self.assertLess(merged["estimate"].ci_low_m, merged["estimate"].depth_m)


class TestAscii(unittest.TestCase):
    def test_renders_valid_cells_and_handles_empty(self):
        frame = wall_frame(2.0)
        cal = fl.calibrate_scene(frame, INTR, sample="wall")
        cells = fl.analyse_frame(frame["ambient"], frame["flash"], COLS, ROWS,
                                 cal, INTR)
        lines = fl.ascii_depth(cells, COLS, ROWS, width=32)
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(len(line) == 32 for line in lines))
        self.assertEqual(fl.ascii_depth([], 4, 4), ["(no valid flash cells)"])


if __name__ == "__main__":
    unittest.main()
