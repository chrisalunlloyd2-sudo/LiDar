import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from inventory import (  # noqa: E402
    classify_dimensions,
    classify_observation,
    emulated_dimensions_mm,
    fill_thickness,
)
from models import CameraIntrinsics, Observation  # noqa: E402
from store import Store  # noqa: E402


class TestClassification(unittest.TestCase):
    def test_2x4(self):
        self.assertEqual(classify_dimensions(88.9, 38.1)["tag"], "2x4")

    def test_2x6(self):
        self.assertEqual(classify_dimensions(139.7, 38.1)["tag"], "2x6")

    def test_orientation_agnostic(self):
        self.assertEqual(classify_dimensions(38.1, 88.9)["tag"], "2x4")

    def test_unknown(self):
        self.assertIsNone(classify_dimensions(10.0, 10.0))

    def test_invalid(self):
        self.assertIsNone(classify_dimensions(0.0, 38.1))

    def test_explicit_class(self):
        obs = Observation(60, 30, 320, 240, CameraIntrinsics.from_fov(640, 480, 60),
                          object_class="2x6")
        self.assertEqual(classify_observation(obs)["tag"], "2x6")

    def test_custom_falls_back_to_dimensions(self):
        obs = Observation(60, 30, 320, 240, CameraIntrinsics.from_fov(640, 480, 60),
                          object_class="custom")
        self.assertEqual(classify_observation(obs, (88.9, 38.1))["tag"], "2x4")

    def test_unknown_when_no_match(self):
        obs = Observation(60, 30, 320, 240, CameraIntrinsics.from_fov(640, 480, 60),
                          object_class="custom")
        self.assertEqual(classify_observation(obs, (5.0, 5.0))["tag"], "unknown")


class TestDimensions(unittest.TestCase):
    def test_emulated_dimensions(self):
        intr = CameraIntrinsics(f_px=1000.0, cx=500.0, cy=500.0, image_w=1000, image_h=1000)
        obs = Observation(100.0, 46.0, 500.0, 500.0, intr)
        w, h, d = emulated_dimensions_mm(obs, 2.0)
        self.assertAlmostEqual(w, 200.0, places=1)
        self.assertAlmostEqual(h, 92.0, places=1)
        self.assertIsNone(d)

    def test_fill_thickness_from_catalog(self):
        w, h, d = fill_thickness((88.0, 38.0, None), {"tag": "2x4"})
        self.assertEqual(d, 38.1)

    def test_fill_thickness_keeps_existing(self):
        w, h, d = fill_thickness((88.0, 38.0, 12.0), {"tag": "2x4"})
        self.assertEqual(d, 12.0)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_upsert_merges_same_place(self):
        a = self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                                   45.0, 10.0, 50.0, 1.0, 0.9)
        b = self.store.upsert_item("2x4", "2x4 lumber", "basement B", 89.1, 38.2, 38.1,
                                   45.000001, 10.000001, 50.0, 1.1, 0.8)
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(b["count"], 2)

    def test_upsert_separates_rooms(self):
        a = self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                                   45.0, 10.0, 50.0, 1.0, 0.9)
        b = self.store.upsert_item("2x4", "2x4 lumber", "garage", 88.9, 38.1, 38.1,
                                   45.0, 10.0, 50.0, 1.0, 0.9)
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(len(self.store.list_items()), 2)

    def test_upsert_separates_far_locations(self):
        a = self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                                   45.0, 10.0, 50.0, 1.0, 0.9)
        b = self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                                   45.01, 10.01, 50.0, 1.0, 0.9)
        self.assertNotEqual(a["id"], b["id"])

    def test_summary_and_reset(self):
        self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                               45.0, 10.0, 50.0, 1.0, 0.9)
        self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                               45.0, 10.0, 50.0, 1.0, 0.9)
        summary = self.store.summary()
        self.assertEqual(summary["total_items"], 2)
        self.assertEqual(summary["by_tag"][0]["tag"], "2x4")
        self.store.reset()
        self.assertEqual(self.store.list_items(), [])
        self.assertEqual(self.store.summary()["total_items"], 0)

    def test_map_points_roundtrip(self):
        self.store.add_map_point(room="basement B", tag="2x4", lat=45.0, lon=10.0,
                                 alt_m=50.0, radius_m=0.5, rejected=0, flags="[]")
        self.store.add_map_point(room="basement B", tag="2x4", lat=45.0, lon=10.0,
                                 alt_m=50.0, radius_m=0.5, rejected=1, flags='["ponzo"]')
        self.assertEqual(len(self.store.list_map_points(room="basement B")), 1)
        self.assertEqual(len(self.store.list_map_points(room="basement B", include_rejected=True)), 2)

    def test_find_by_tag_and_room(self):
        self.store.upsert_item("2x4", "2x4 lumber", "basement B", 88.9, 38.1, 38.1,
                               45.0, 10.0, 50.0, 1.0, 0.9)
        self.store.upsert_item("paint_can", "Paint can", "basement B", 168.0, 190.0, 168.0,
                               45.0, 10.0, 50.0, 1.0, 0.8)
        found = self.store.list_items(tag="2x4", room="basement B")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["tag"], "2x4")


if __name__ == "__main__":
    unittest.main()
