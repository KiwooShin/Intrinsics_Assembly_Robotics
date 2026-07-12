"""Import-and-structure smoke test for profile_loader.py.

The benchmark itself needs torch + GPU + the smoke dataset, so only import and the
presence of a callable entry point are checked here.
"""
from __future__ import annotations

import unittest

import profile_loader


class TestProfileLoaderSmoke(unittest.TestCase):
    def test_importable_with_main(self) -> None:
        self.assertTrue(callable(profile_loader.main))


if __name__ == "__main__":
    unittest.main()
