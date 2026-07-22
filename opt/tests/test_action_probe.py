"""Unit tests for opt.action_probe pure logic (no torch/ROS/GPU)."""
from __future__ import annotations

import unittest

import numpy as np

from opt.action_probe import decompose_first_action


class DecomposeFirstActionTest(unittest.TestCase):
    def test_splits_lateral_and_vertical(self) -> None:
        chunk = np.array([[3.0, 4.0, -2.0, 0, 0, 0],
                          [9.0, 9.0, 9.0, 0, 0, 0]], dtype=np.float64)
        lat, vert = decompose_first_action(chunk)
        self.assertAlmostEqual(lat, 5.0)      # hypot(3,4) of the FIRST step only
        self.assertAlmostEqual(vert, -2.0)    # signed vz of the first step

    def test_pure_vertical_has_zero_lateral(self) -> None:
        lat, vert = decompose_first_action(np.array([[0.0, 0.0, -30.0, 1, 2, 3]]))
        self.assertAlmostEqual(lat, 0.0)
        self.assertAlmostEqual(vert, -30.0)

    def test_lateral_is_nonnegative_magnitude(self) -> None:
        lat, _ = decompose_first_action(np.array([[-3.0, -4.0, 0.0, 0, 0, 0]]))
        self.assertAlmostEqual(lat, 5.0)

    def test_rejects_bad_shape(self) -> None:
        for bad in (np.zeros((4,)), np.zeros((2, 3)), np.zeros((0, 6))):
            with self.assertRaises(ValueError):
                decompose_first_action(bad)


if __name__ == "__main__":
    unittest.main()
