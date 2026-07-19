#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free policy state assembly used by ``DeployACT``.

These exercise :mod:`aic_example_policies.ros.state_assembly` -- assembling the
7-D (pose) or 13-D (pose + wrist wrench) policy state and the
normalize/de-normalize round-trip -- and require neither ROS, Gazebo, torch, nor
a GPU.

Run with::

    python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from aic_example_policies.ros import state_assembly


class AssembleStateTest(unittest.TestCase):
    """7-D backward compatibility and 13-D pose+wrench assembly."""

    _POSE = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _WRENCH = np.array([19.0, -1.0, 21.0, 0.1, -0.2, 0.3], dtype=np.float32)

    def test_seven_dim_returns_pose_unchanged(self) -> None:
        # 7-D checkpoint: byte-identical to feeding the raw pose (legacy path).
        out = state_assembly.assemble_state(self._POSE, None, state_dim=7)
        np.testing.assert_array_equal(out, self._POSE)

    def test_seven_dim_ignores_wrench(self) -> None:
        out = state_assembly.assemble_state(self._POSE, self._WRENCH, state_dim=7)
        np.testing.assert_array_equal(out, self._POSE)

    def test_thirteen_dim_appends_wrench_in_order(self) -> None:
        out = state_assembly.assemble_state(self._POSE, self._WRENCH, state_dim=13)
        self.assertEqual(out.shape, (13,))
        np.testing.assert_array_equal(out[:7], self._POSE)
        # Wrench appended as [fx, fy, fz, tx, ty, tz], matching prepare_dataset.
        np.testing.assert_array_equal(out[7:], self._WRENCH)

    def test_thirteen_dim_requires_wrench(self) -> None:
        with self.assertRaises(ValueError):
            state_assembly.assemble_state(self._POSE, None, state_dim=13)

    def test_bad_pose_shape(self) -> None:
        with self.assertRaises(ValueError):
            state_assembly.assemble_state(np.zeros(6), None, state_dim=7)

    def test_bad_wrench_shape(self) -> None:
        with self.assertRaises(ValueError):
            state_assembly.assemble_state(self._POSE, np.zeros(5), state_dim=13)

    def test_unsupported_state_dim(self) -> None:
        with self.assertRaises(ValueError):
            state_assembly.assemble_state(self._POSE, self._WRENCH, state_dim=9)


class NormalizeRoundTripTest(unittest.TestCase):
    """normalize_state / denormalize_state invert each other for 7-D and 13-D."""

    def _roundtrip(self, dim: int) -> None:
        rng = np.random.default_rng(dim)
        state = rng.normal(size=dim).astype(np.float32)
        mean = rng.normal(size=dim).astype(np.float32)
        std = np.abs(rng.normal(size=dim)).astype(np.float32) + 1e-3
        norm = state_assembly.normalize_state(state, mean, std)
        recon = state_assembly.denormalize_state(norm, mean, std)
        np.testing.assert_allclose(recon, state, rtol=1e-5, atol=1e-5)

    def test_roundtrip_7d(self) -> None:
        self._roundtrip(7)

    def test_roundtrip_13d(self) -> None:
        self._roundtrip(13)

    def test_normalize_matches_formula(self) -> None:
        state = np.array([2.0, 4.0], dtype=np.float32)
        mean = np.array([1.0, 2.0], dtype=np.float32)
        std = np.array([0.5, 2.0], dtype=np.float32)
        # (2-1)/0.5 = 2 ; (4-2)/2 = 1
        np.testing.assert_allclose(
            state_assembly.normalize_state(state, mean, std), [2.0, 1.0]
        )

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            state_assembly.normalize_state(
                np.zeros(7), np.zeros(13), np.ones(13)
            )


class AssembleThenNormalizeTest(unittest.TestCase):
    """End-to-end: assemble a 13-D state then normalize/de-normalize it back."""

    def test_full_pipeline_roundtrip(self) -> None:
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        wrench = np.array([19.0, -1.0, 21.0, 0.1, -0.2, 0.3], dtype=np.float32)
        state = state_assembly.assemble_state(pose, wrench, 13)
        mean = np.zeros(13, dtype=np.float32)
        std = np.ones(13, dtype=np.float32)
        norm = state_assembly.normalize_state(state, mean, std)
        recon = state_assembly.denormalize_state(norm, mean, std)
        np.testing.assert_allclose(recon[:7], pose, rtol=1e-6)
        np.testing.assert_allclose(recon[7:], wrench, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
