import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import server  # noqa: E402


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


class TestReleaseDocs(unittest.TestCase):
    """Drift guards: shipping a version means the docs say the same thing."""

    def setUp(self):
        self.version = server.VERSION
        self.changelog = read("CHANGELOG.md")
        self.readme = read("README.md")
        self.shell = read(os.path.join("gui_pages", "shell.html"))

    def test_changelog_has_a_section_for_the_shipped_version(self):
        self.assertIn("## [%s]" % self.version, self.changelog)

    def test_unreleased_section_comes_first(self):
        self.assertIn("## [Unreleased]", self.changelog)
        self.assertLess(self.changelog.index("## [Unreleased]"),
                        self.changelog.index("## [%s]" % self.version))

    def test_gui_footer_shows_the_version(self):
        self.assertIn("v%s" % self.version, self.shell)

    def test_install_docs_do_not_pin_a_version(self):
        # a hardcoded lidar_X.Y.Z_all.deb goes stale on every release
        self.assertIsNone(re.search(r"lidar_\d+\.\d+\.\d+_all\.deb",
                                    self.readme))


if __name__ == "__main__":
    unittest.main()
