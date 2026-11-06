import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.arena import run_arena  # noqa: E402
from arena.parallel_arena import run_parallel_arena  # noqa: E402


class TestParallelArena(unittest.TestCase):
    def test_quick_parallel_covers_same_methods_as_serial(self):
        par = run_parallel_arena(workers=2, quick=True, top_k=2)
        ser = run_arena(quick=True, top_k=2)
        for stage in ("fusers", "sizes", "illusions", "triangulators"):
            self.assertEqual({r["method"] for r in par[stage]},
                             {r["method"] for r in ser[stage]})
        self.assertIn("survivors", par)
        self.assertIn("pruned", par)
        self.assertGreaterEqual(par["wall_ms"], 0.0)

    def test_all_fusers_evaluated(self):
        par = run_parallel_arena(workers=2, quick=True)
        self.assertEqual(len(par["fusers"]), 17)
        self.assertGreaterEqual(len(par["sizes"]), 10)

    def test_survivors_and_pruned_are_disjoint(self):
        par = run_parallel_arena(workers=2, quick=True, top_k=2)
        for stage in ("fusers", "sizes", "illusions", "triangulators"):
            kept = set(par["survivors"][stage])
            pruned = set(par["pruned"][stage])
            self.assertEqual(kept & pruned, set())
            self.assertEqual(kept | pruned,
                             {r["method"] for r in par[stage]})


if __name__ == "__main__":
    unittest.main()
