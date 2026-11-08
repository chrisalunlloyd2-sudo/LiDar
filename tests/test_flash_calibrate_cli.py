import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import flash_lidar as fl  # noqa: E402
from models import CameraIntrinsics  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "flash_calibrate.py")
INTR = CameraIntrinsics.from_fov(640, 480, 60.0)


def write_grids(path, wall_m=1.5, cols=32, rows=18, dark=False):
    frame = fl.render_scene(fl.FlashScene("wall", [], wall_depth_m=wall_m),
                            INTR, cols=cols, rows=rows, seed=0)
    ambient = [10.0] * (cols * rows) if dark else frame["ambient"]
    flash = [10.0] * (cols * rows) if dark else frame["flash"]
    payload = {
        "cols": cols, "rows": rows, "ambient": ambient, "flash": flash,
        "intrinsics": {"f_px": INTR.f_px, "cx": INTR.cx, "cy": INTR.cy,
                       "image_w": INTR.image_w, "image_h": INTR.image_h,
                       "fov_deg": 60.0},
        "reference_m": wall_m,
        "samples": {"ambient": 3, "torch": 3},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def run_script(*args):
    return subprocess.run([sys.executable, SCRIPT] + list(args), cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=60)


class TestFlashCalibrateCli(unittest.TestCase):
    def test_markdown_row_for_a_healthy_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grids.json")
            write_grids(path)
            proc = run_script("--grids", path, "--range", "1.5", "--markdown")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = proc.stdout.strip().splitlines()[-1]
        self.assertTrue(row.startswith("| 1.50 |"), row)
        self.assertIn("| ok |", row)
        self.assertIn("self-anchored", row)
        # six pipe-separated cells plus the leading/trailing pipes
        self.assertEqual(row.count("|"), 8, row)

    def test_dark_pair_reports_no_constant_with_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grids.json")
            write_grids(path, dark=True)
            proc = run_script("--grids", path, "--range", "1.0", "--markdown")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = proc.stdout.strip().splitlines()[-1]
        self.assertIn("| dark |", row)
        self.assertIn("| — |", row)

    def test_text_mode_prints_the_suggested_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grids.json")
            write_grids(path, wall_m=2.0)
            proc = run_script("--grids", path, "--range", "2.0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("calibration : reference 2.00 m", proc.stdout)
        self.assertIn("status      : ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
