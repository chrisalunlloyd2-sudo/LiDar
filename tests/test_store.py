import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from store import Store  # noqa: E402


def point(**kw):
    row = dict(session="scan", room="r", lat=1.0, lon=2.0, alt_m=0.0,
               radius_m=1.0, east_m=0.0, north_m=0.0, up_m=0.0,
               tag="ground", depth_m=1.0, ci_low_m=1.0, ci_high_m=1.0,
               confidence=0.5, rejected=0, flags="[]")
    row.update(kw)
    return row


class TestStoreBatch(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_batch_writes_every_row(self):
        written = self.store.add_map_points([point() for _ in range(200)])
        self.assertEqual(written, 200)
        self.assertEqual(len(self.store.list_map_points(room="r", limit=500)), 200)

    def test_batch_of_nothing_is_a_no_op(self):
        self.assertEqual(self.store.add_map_points([]), 0)
        self.assertEqual(self.store.summary()["map_points"], 0)

    def test_batch_leaves_the_caller_rows_untouched(self):
        row = point(ts=5.0)
        self.store.add_map_points([row])
        self.assertEqual(row["ts"], 5.0)
        self.assertEqual(len(row), 17)

    def test_batch_defaults_ts_and_coerces_rejected(self):
        self.store.add_map_points([point(ts=None, rejected=True)])
        stored = self.store.list_map_points(limit=10, include_rejected=True)[0]
        self.assertGreater(stored["ts"], 0)
        self.assertEqual(stored["rejected"], 1)

    def test_single_insert_still_works(self):
        point_id = self.store.add_map_point(**point())
        self.assertIsNotNone(point_id)
        self.assertEqual(self.store.summary()["map_points"], 1)


if __name__ == "__main__":
    unittest.main()
