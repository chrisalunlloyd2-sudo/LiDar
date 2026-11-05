import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from page_store import PageStore  # noqa: E402


class TestPageStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lidarv2_pages_")
        self._write("a.html", b"<h1>A</h1>")
        self._write("app.js", b"console.log(1);")
        os.mkdir(os.path.join(self.dir, "sub"))
        self._write(os.path.join("sub", "b.html"), b"<p>B</p>")
        self.store = PageStore(self.dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, rel, data):
        with open(os.path.join(self.dir, rel), "wb") as fh:
            fh.write(data)

    def test_list_includes_nested_relative_paths(self):
        pages = self.store.list()
        self.assertIn("a.html", pages)
        self.assertIn("app.js", pages)
        self.assertIn("sub/b.html", pages)

    def test_get_returns_mapped_bytes(self):
        self.assertEqual(self.store.get("a.html"), b"<h1>A</h1>")
        self.assertEqual(self.store.get("sub/b.html"), b"<p>B</p>")

    def test_get_missing(self):
        self.assertIsNone(self.store.get("nope.html"))

    def test_has_and_size(self):
        self.assertTrue(self.store.has("app.js"))
        self.assertEqual(self.store.size("a.html"), len(b"<h1>A</h1>"))

    def test_reload_picks_up_new_file(self):
        self._write("c.html", b"<b>C</b>")
        self.store.reload()
        self.assertEqual(self.store.get("c.html"), b"<b>C</b>")

    def test_zero_length_file(self):
        self._write("empty.html", b"")
        self.store.reload()
        self.assertEqual(self.store.get("empty.html"), b"")


if __name__ == "__main__":
    unittest.main()
