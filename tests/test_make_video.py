"""Smoke + pure-helper tests for make_video.py.

The heavy render path (ROS/cv2/ffmpeg) is not exercised; only the pure trimming
and nearest-neighbour helpers plus module structure are tested.
"""
from __future__ import annotations

import unittest

import make_video as mv


class TestNearest(unittest.TestCase):
    def test_picks_closest_payload(self) -> None:
        lst = [(0.0, "a"), (1.0, "b"), (2.0, "c")]
        self.assertEqual(mv.nearest(lst, 0.9), "b")
        self.assertEqual(mv.nearest(lst, 0.1), "a")
        self.assertEqual(mv.nearest(lst, 5.0), "c")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            mv.nearest([], 1.0)


class TestSelectWindow(unittest.TestCase):
    def test_inclusive_bounds(self) -> None:
        frames = [(0.0, "a"), (1.0, "b"), (2.0, "c"), (3.0, "d")]
        kept = mv.select_window(frames, 1.0, 2.0)
        self.assertEqual([p for _, p in kept], ["b", "c"])

    def test_preserves_order(self) -> None:
        frames = [(2.0, "c"), (0.5, "a"), (1.5, "b")]
        kept = mv.select_window(frames, 0.0, 2.0)
        self.assertEqual([p for _, p in kept], ["c", "a", "b"])

    def test_empty_when_out_of_range(self) -> None:
        self.assertEqual(mv.select_window([(5.0, "x")], 0.0, 1.0), [])


class TestModuleStructure(unittest.TestCase):
    def test_public_callables_present(self) -> None:
        self.assertTrue(callable(mv.render_video))
        self.assertTrue(callable(mv.main))


if __name__ == "__main__":
    unittest.main()
