import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shapes import classify_shape  # noqa: E402


class TestShapes(unittest.TestCase):
    def test_square(self):
        s = classify_shape(100.0, 100.0, 10.0)
        self.assertEqual(s["shape"], "square")
        self.assertTrue(s["is_square"])
        self.assertFalse(s["is_cube"])

    def test_cube(self):
        s = classify_shape(100.0, 100.0, 100.0)
        self.assertEqual(s["shape"], "cube")
        self.assertTrue(s["is_cube"])

    def test_rectangle(self):
        s = classify_shape(200.0, 100.0, 20.0)
        self.assertEqual(s["shape"], "rectangle")
        self.assertTrue(s["is_rectangle"])

    def test_lumber(self):
        s = classify_shape(400.0, 50.0, 50.0)
        self.assertEqual(s["shape"], "lumber")
        self.assertAlmostEqual(s["aspect"], 8.0)

    def test_cuboid(self):
        s = classify_shape(100.0, 120.0, 100.0)
        self.assertEqual(s["shape"], "cuboid")

    def test_unknown(self):
        s = classify_shape(0.0, 100.0)
        self.assertEqual(s["shape"], "unknown")

    def test_faces(self):
        self.assertEqual(classify_shape(100, 100, 100)["faces"], 6)
        self.assertEqual(classify_shape(100, 100, 5)["faces"], 1)


if __name__ == "__main__":
    unittest.main()
