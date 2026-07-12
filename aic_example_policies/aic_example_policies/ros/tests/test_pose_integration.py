#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free pose-integration math used by ``DeployACT``.

These exercise :mod:`aic_example_policies.ros.pose_integration` -- the pure
function that turns a predicted base_link twist chunk into absolute pose targets
-- and require neither ROS, Gazebo, torch, nor a GPU.

Run with::

    python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aic_example_policies.ros.pose_integration import (
    expand_twists,
    integrate_twist_chunk,
    quaternion_from_angular_velocity,
    quaternion_multiply,
)


class QuaternionHelpersTest(unittest.TestCase):
    """Tests for the quaternion helper functions."""

    def test_identity_angular_velocity(self) -> None:
        """Near-zero angular velocity yields the identity quaternion."""
        q = quaternion_from_angular_velocity(np.zeros(3), 0.1)
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0])

    def test_z_rotation_matches_analytic(self) -> None:
        """A z-axis angular velocity integrates to the expected half-angle quat."""
        # 1 rad/s about +z for 0.5 s -> 0.5 rad rotation about z.
        q = quaternion_from_angular_velocity(np.array([0.0, 0.0, 1.0]), 0.5)
        expected = [0.0, 0.0, math.sin(0.25), math.cos(0.25)]
        np.testing.assert_allclose(q, expected, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)

    def test_multiply_identity(self) -> None:
        """Multiplying by the identity quaternion returns the original."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        ident = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(quaternion_multiply(ident, q), q, atol=1e-12)
        np.testing.assert_allclose(quaternion_multiply(q, ident), q, atol=1e-12)

    def test_multiply_known_product(self) -> None:
        """i * j == k under the Hamilton product ([x, y, z, w] order)."""
        qi = np.array([1.0, 0.0, 0.0, 0.0])
        qj = np.array([0.0, 1.0, 0.0, 0.0])
        qk = np.array([0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(quaternion_multiply(qi, qj), qk, atol=1e-12)


class IntegrateTwistChunkTest(unittest.TestCase):
    """Tests for :func:`integrate_twist_chunk`."""

    def test_pure_translation(self) -> None:
        """Zero angular velocity: positions accumulate v*dt, orientation held."""
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        twists = np.array(
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        out = integrate_twist_chunk(pos, quat, twists, dt=0.5)
        self.assertEqual(out.shape, (2, 7))
        # x advances by 0.1*0.5 = 0.05 each step.
        np.testing.assert_allclose(out[0, :3], [1.05, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(out[1, :3], [1.10, 2.0, 3.0], atol=1e-12)
        # Orientation unchanged (identity) at every step.
        np.testing.assert_allclose(out[:, 3:], [[0, 0, 0, 1]] * 2, atol=1e-12)

    def test_start_pose_not_included(self) -> None:
        """The first row is one step ahead of the start pose, not the start pose."""
        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        twists = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out = integrate_twist_chunk(pos, quat, twists, dt=0.25)
        np.testing.assert_allclose(out[0, :3], [0.25, 0.0, 0.0], atol=1e-12)

    def test_net_translation_equals_velocity_time(self) -> None:
        """Constant velocity over n steps nets v * (n * dt) total displacement."""
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        v = np.array([0.03, -0.01, 0.02])
        n, dt = 8, 0.05
        twists = np.tile(np.concatenate([v, np.zeros(3)]), (n, 1))
        out = integrate_twist_chunk(pos, quat, twists, dt)
        np.testing.assert_allclose(out[-1, :3], v * (n * dt), atol=1e-12)

    def test_quaternions_stay_unit_norm(self) -> None:
        """Every produced orientation is a unit quaternion."""
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        rng = np.random.default_rng(0)
        twists = rng.normal(scale=0.05, size=(16, 6))
        out = integrate_twist_chunk(pos, quat, twists, dt=0.275)
        norms = np.linalg.norm(out[:, 3:], axis=1)
        np.testing.assert_allclose(norms, np.ones(16), atol=1e-9)

    def test_rotation_accumulates(self) -> None:
        """Two 0.25 rad z-steps compose to a single 0.5 rad z-rotation."""
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        # 1 rad/s about z, dt=0.25 -> 0.25 rad per step, two steps -> 0.5 rad.
        twists = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        out = integrate_twist_chunk(pos, quat, twists, dt=0.25)
        expected = [0.0, 0.0, math.sin(0.25), math.cos(0.25)]
        np.testing.assert_allclose(out[-1, 3:], expected, atol=1e-9)

    def test_invalid_dt_raises(self) -> None:
        """A non-positive dt raises ValueError."""
        with self.assertRaises(ValueError):
            integrate_twist_chunk(
                np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros((1, 6)), dt=0.0,
            )

    def test_invalid_twist_shape_raises(self) -> None:
        """A twist array without 6 columns raises ValueError."""
        with self.assertRaises(ValueError):
            integrate_twist_chunk(
                np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros((4, 3)), dt=0.1,
            )

    def test_zero_norm_quaternion_raises(self) -> None:
        """A degenerate zero-norm start quaternion raises ValueError."""
        with self.assertRaises(ValueError):
            integrate_twist_chunk(
                np.zeros(3), np.zeros(4), np.zeros((1, 6)), dt=0.1
            )


class ExpandTwistsTest(unittest.TestCase):
    """Tests for :func:`expand_twists`."""

    def test_expands_row_count(self) -> None:
        """Expanding by k multiplies the row count and repeats contiguously."""
        twists = np.array([[1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0]])
        out = expand_twists(twists, 3)
        self.assertEqual(out.shape, (6, 6))
        np.testing.assert_array_equal(out[:3, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(out[3:, 0], [2.0, 2.0, 2.0])

    def test_substeps_preserve_net_trajectory(self) -> None:
        """Sub-stepping yields the same net displacement as the coarse chunk."""
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        twists = np.array([[0.02, 0.01, -0.03, 0.0, 0.0, 0.0]] * 4)
        coarse = integrate_twist_chunk(pos, quat, twists, dt=0.275)
        fine = integrate_twist_chunk(
            pos, quat, expand_twists(twists, 5), dt=0.275 / 5
        )
        np.testing.assert_allclose(coarse[-1, :3], fine[-1, :3], atol=1e-9)

    def test_invalid_substeps_raises(self) -> None:
        """A sub-step count below 1 raises ValueError."""
        with self.assertRaises(ValueError):
            expand_twists(np.zeros((2, 6)), 0)


if __name__ == "__main__":
    unittest.main()
