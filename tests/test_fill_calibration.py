import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "fill_calibration.py")
DOC = os.path.join(ROOT, "CALIBRATION.md")

ROWS = """\
| target range | centre DN | k = S0·d0² | cue range | error % | status | notes |
|---|---|---|---|---|---|---|
| 1.00 | 57.1 | 57.1 | 1.004 | +0.38 | ok | locked k 2026-11-08 |
| 1.50 | 25.4 | 57.1 | 1.512 | +0.80 | ok | locked k 2026-11-08 |
| 2.00 | 14.2 | 57.1 | 2.041 | +2.05 | marginal | locked k 2026-11-08 |
| 3.00 | 6.3 | 57.1 | — | — | dark | torch out of range |
"""


class TestFillCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.doc = os.path.join(self.tmp, "CALIBRATION.md")
        shutil.copy(DOC, self.doc)
        self.rows = os.path.join(self.tmp, "rows.md")
        with open(self.rows, "w", encoding="utf-8") as fh:
            fh.write(ROWS)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *args):
        return subprocess.run([sys.executable, SCRIPT] + list(args),
                              cwd=self.tmp, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=60)

    def read(self):
        with open(self.doc, encoding="utf-8") as fh:
            return fh.read()

    def test_write_replaces_the_blank_rows(self):
        before = self.read()
        proc = self.run_tool("--rows", self.rows, "--doc", self.doc, "--write")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        after = self.read()
        self.assertIn("| 1.00 m | 57.1 | 57.1 | 1.004 | +0.38 | ok |", after)
        self.assertIn("| 3.00 m | 6.3 | 57.1 | — | — | dark |", after)
        self.assertNotIn("| 1.0 m | | | | | | |", after)
        # the rest of the document must survive untouched
        self.assertIn("## Failure modes", after)
        self.assertIn("## Device pass", after)
        self.assertLess(len(before), len(after) + 2000)

    def test_preview_does_not_modify_the_document(self):
        before = self.read()
        proc = self.run_tool("--rows", self.rows, "--doc", self.doc)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("| 2.00 m | 14.2 |", proc.stdout)
        self.assertIn("preview only", proc.stderr)
        self.assertEqual(self.read(), before)

    def test_rows_are_sorted_by_range(self):
        shuffled = os.path.join(self.tmp, "shuffled.md")
        with open(shuffled, "w", encoding="utf-8") as fh:
            lines = ROWS.strip().splitlines()
            fh.write("\n".join([lines[0], lines[1], lines[4], lines[2],
                                lines[5], lines[3]]) + "\n")
        proc = self.run_tool("--rows", shuffled, "--doc", self.doc, "--write")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        after = self.read()
        self.assertLess(after.index("| 1.00 m |"), after.index("| 1.50 m |"))
        self.assertLess(after.index("| 1.50 m |"), after.index("| 2.00 m |"))
        self.assertLess(after.index("| 2.00 m |"), after.index("| 3.00 m |"))

    def test_rejects_a_short_row(self):
        bad = os.path.join(self.tmp, "bad.md")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("| 1.00 | 57.1 | 57.1 | 1.004 | +0.38 |\n")
        proc = self.run_tool("--rows", bad, "--doc", self.doc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("at least 6 columns", proc.stderr)

    def test_rejects_an_unknown_status(self):
        bad = os.path.join(self.tmp, "bad-status.md")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("| 1.00 | 57.1 | 57.1 | 1.004 | +0.38 | great | x |\n")
        proc = self.run_tool("--rows", bad, "--doc", self.doc)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown status", proc.stderr)


if __name__ == "__main__":
    unittest.main()
