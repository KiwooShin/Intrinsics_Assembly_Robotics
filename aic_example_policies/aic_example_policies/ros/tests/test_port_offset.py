#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free port-offset geometry (``port_offset.py``).

These exercise :mod:`aic_example_policies.ros.port_offset` -- quaternion vector
rotation and its inverse, the TCP-frame <-> base_link round trip, the robust
terminal target, per-frame offset/axis labels, and the deploy prediction record
with its plausibility gate. They require neither ROS, Gazebo, torch, nor a GPU.

Run with::

    PYTHONPATH=<repo>/aic_example_policies python -m unittest \
        aic_example_policies.ros.tests.test_port_offset -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aic_example_policies.ros.pose_integration import quaternion_from_angular_velocity
from aic_example_policies.ros.port_offset import (
    PortOffsetPrediction,
    base_approach_axis,
    base_target_from_tcp_offset,
    per_frame_axis_labels,
    per_frame_tcp_offsets,
    predict_from_aux,
    quaternion_conjugate,
    robust_terminal,
    rotate_vector_by_quat,
    rotate_vector_by_quat_inverse,
    rotate_vectors_by_quats,
    tcp_frame_offset,
)


def _quat_z(theta: float) -> np.ndarray:
    """Return the ``[x, y, z, w]`` quaternion for a rotation ``theta`` about +Z."""
    return np.array([0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)])


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Return a random unit quaternion ``[x, y, z, w]``."""
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class RotationTest(unittest.TestCase):
    """Quaternion vector rotation, its inverse, and the batch form."""

    def test_rotate_ninety_about_z(self) -> None:
        """A +90 deg Z rotation maps +X onto +Y."""
        out = rotate_vector_by_quat(_quat_z(math.pi / 2.0), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-12)

    def test_inverse_undoes_rotation(self) -> None:
        """The inverse rotation recovers the original vector."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = _random_unit_quat(rng)
            v = rng.normal(size=3)
            back = rotate_vector_by_quat_inverse(q, rotate_vector_by_quat(q, v))
            np.testing.assert_allclose(back, v, atol=1e-12)

    def test_rotation_preserves_norm(self) -> None:
        """Rotation is an isometry -- it preserves vector length."""
        rng = np.random.default_rng(1)
        q = _random_unit_quat(rng)
        v = rng.normal(size=3)
        self.assertAlmostEqual(
            float(np.linalg.norm(rotate_vector_by_quat(q, v))),
            float(np.linalg.norm(v)),
            places=12,
        )

    def test_non_unit_quat_is_normalized(self) -> None:
        """A non-unit quaternion gives the same rotation as its normalization."""
        q = _quat_z(0.7) * 3.5
        v = [0.2, -0.3, 0.5]
        np.testing.assert_allclose(
            rotate_vector_by_quat(q, v),
            rotate_vector_by_quat(_quat_z(0.7), v),
            atol=1e-12,
        )

    def test_zero_norm_quat_raises(self) -> None:
        """A zero quaternion cannot define a rotation."""
        with self.assertRaises(ValueError):
            rotate_vector_by_quat([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    def test_batch_matches_scalar(self) -> None:
        """The vectorized rotation matches the scalar one row-for-row."""
        rng = np.random.default_rng(2)
        quats = np.stack([_random_unit_quat(rng) for _ in range(16)])
        vecs = rng.normal(size=(16, 3))
        batch = rotate_vectors_by_quats(quats, vecs)
        for i in range(16):
            np.testing.assert_allclose(
                batch[i], rotate_vector_by_quat(quats[i], vecs[i]), atol=1e-12
            )
        inv = rotate_vectors_by_quats(quats, vecs, inverse=True)
        for i in range(16):
            np.testing.assert_allclose(
                inv[i], rotate_vector_by_quat_inverse(quats[i], vecs[i]), atol=1e-12
            )

    def test_batch_bad_shapes_raise(self) -> None:
        """Wrong-shaped batch inputs raise ValueError."""
        with self.assertRaises(ValueError):
            rotate_vectors_by_quats(np.zeros((3, 3)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            rotate_vectors_by_quats(np.zeros((3, 4)), np.zeros((2, 3)))

    def test_conjugate(self) -> None:
        """The conjugate negates the vector part only."""
        np.testing.assert_allclose(
            quaternion_conjugate([0.1, 0.2, 0.3, 0.9]), [-0.1, -0.2, -0.3, 0.9]
        )


class RoundTripTest(unittest.TestCase):
    """``tcp_frame_offset`` and ``base_target_from_tcp_offset`` are inverses."""

    def test_identity_round_trip(self) -> None:
        """base -> tcp-offset -> base recovers the original target for random poses."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            pos = rng.normal(size=3)
            quat = _random_unit_quat(rng)
            target = rng.normal(size=3)
            offset = tcp_frame_offset(pos, quat, target)
            recovered = base_target_from_tcp_offset(pos, quat, offset)
            np.testing.assert_allclose(recovered, target, atol=1e-12)

    def test_offset_is_relative_to_tcp(self) -> None:
        """With identity orientation the TCP offset is target minus position."""
        pos = np.array([0.1, 0.2, 0.3])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        target = np.array([0.15, 0.20, 0.25])
        np.testing.assert_allclose(
            tcp_frame_offset(pos, quat, target), [0.05, 0.0, -0.05], atol=1e-12
        )

    def test_offset_uses_tcp_frame(self) -> None:
        """Under a +90 deg Z TCP a base +X displacement reads as -Y in TCP frame."""
        pos = np.zeros(3)
        quat = _quat_z(math.pi / 2.0)
        target = np.array([1.0, 0.0, 0.0])  # +X in base
        # R(q)^T maps base +X to TCP -Y for a +90 deg Z rotation.
        np.testing.assert_allclose(
            tcp_frame_offset(pos, quat, target), [0.0, -1.0, 0.0], atol=1e-12
        )


class RobustTerminalTest(unittest.TestCase):
    """The robust terminal target medians the tail and takes the end quaternion."""

    def _poses(self, n: int) -> np.ndarray:
        p = np.zeros((n, 7))
        p[:, 0] = np.linspace(0.0, 1.0, n)
        p[:, 6] = 1.0  # identity quaternion
        return p

    def test_median_of_tail_position(self) -> None:
        """The target position is the component-wise median of the last M frames."""
        p = self._poses(10)
        # Corrupt one tail frame; the median rejects the outlier.
        p[-2, 0] = 100.0
        pos, quat = robust_terminal(p, terminal_frames=5)
        self.assertLess(pos[0], 2.0)  # outlier rejected by the median
        np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])

    def test_terminal_frames_clamped_to_length(self) -> None:
        """Asking for more frames than exist clamps to the episode length."""
        p = self._poses(3)
        pos, _ = robust_terminal(p, terminal_frames=10)
        np.testing.assert_allclose(pos, np.median(p[:, :3], axis=0))

    def test_offset_at_terminal_is_zero(self) -> None:
        """The per-frame offset at the terminal frame is ~0 (identity orientation)."""
        p = self._poses(20)
        target, _ = robust_terminal(p, terminal_frames=1)  # exact last position
        offs = per_frame_tcp_offsets(p, target)
        np.testing.assert_allclose(offs[-1], [0.0, 0.0, 0.0], atol=1e-12)
        # A mid-approach frame still points toward the target (+X here).
        self.assertGreater(offs[0][0], 0.0)

    def test_bad_shapes_and_args_raise(self) -> None:
        with self.assertRaises(ValueError):
            robust_terminal(np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            robust_terminal(np.zeros((0, 7)))
        with self.assertRaises(ValueError):
            robust_terminal(np.zeros((5, 7)), terminal_frames=0)


class PerFrameOffsetTest(unittest.TestCase):
    """Per-frame offset labels in TCP and base frames."""

    def test_base_frame_is_target_minus_position(self) -> None:
        rng = np.random.default_rng(4)
        p = np.zeros((6, 7))
        p[:, :3] = rng.normal(size=(6, 3))
        p[:, 6] = 1.0
        target = np.array([0.5, -0.2, 0.1])
        offs = per_frame_tcp_offsets(p, target, frame="base")
        np.testing.assert_allclose(offs, target[None, :] - p[:, :3], atol=1e-12)

    def test_tcp_and_base_magnitudes_match(self) -> None:
        """Rotation preserves length, so |offset_tcp| == |offset_base| per frame."""
        rng = np.random.default_rng(5)
        p = np.zeros((8, 7))
        p[:, :3] = rng.normal(size=(8, 3))
        p[:, 3:] = np.stack([_random_unit_quat(rng) for _ in range(8)])
        target = rng.normal(size=3)
        tcp = per_frame_tcp_offsets(p, target, frame="tcp")
        base = per_frame_tcp_offsets(p, target, frame="base")
        np.testing.assert_allclose(
            np.linalg.norm(tcp, axis=1), np.linalg.norm(base, axis=1), atol=1e-12
        )

    def test_bad_frame_raises(self) -> None:
        with self.assertRaises(ValueError):
            per_frame_tcp_offsets(np.zeros((3, 7)), np.zeros(3), frame="world")


class ApproachAxisTest(unittest.TestCase):
    """Base-frame approach axis and per-frame axis labels."""

    def test_straight_line_axis(self) -> None:
        """A pure -Z descent yields the -Z unit axis."""
        p = np.zeros((12, 7))
        p[:, 2] = np.linspace(0.5, 0.38, 12)  # descending in Z
        p[:, 6] = 1.0
        axis = base_approach_axis(p, axis_frames=5, min_displacement=1e-4)
        assert axis is not None
        np.testing.assert_allclose(axis, [0.0, 0.0, -1.0], atol=1e-9)

    def test_stalled_tail_ignored(self) -> None:
        """Sub-threshold seated frames at the end do not flip the axis estimate."""
        p = np.zeros((20, 7))
        move = np.linspace(0.5, 0.40, 12)
        p[:12, 2] = move
        p[12:, 2] = move[-1]  # perfectly stalled tail (zero displacement)
        p[:, 6] = 1.0
        axis = base_approach_axis(p, axis_frames=5, min_displacement=1e-4)
        assert axis is not None
        self.assertLess(axis[2], -0.99)

    def test_no_motion_returns_none(self) -> None:
        p = np.zeros((5, 7))
        p[:, 6] = 1.0
        self.assertIsNone(base_approach_axis(p))

    def test_per_frame_axis_tcp_unit_norm(self) -> None:
        """Per-frame TCP axis labels stay unit norm (rotation preserves length)."""
        rng = np.random.default_rng(6)
        p = np.zeros((7, 7))
        p[:, 3:] = np.stack([_random_unit_quat(rng) for _ in range(7)])
        axis = np.array([0.0, 0.0, -1.0])
        labels = per_frame_axis_labels(p, axis, frame="tcp")
        np.testing.assert_allclose(np.linalg.norm(labels, axis=1), 1.0, atol=1e-12)

    def test_per_frame_axis_none_is_zero(self) -> None:
        p = np.zeros((4, 7))
        p[:, 6] = 1.0
        labels = per_frame_axis_labels(p, None, frame="tcp")
        np.testing.assert_allclose(labels, 0.0)


class PredictionTest(unittest.TestCase):
    """The deploy-time prediction record and its plausibility gate."""

    def test_tcp_frame_prediction_matches_round_trip(self) -> None:
        """A TCP-frame aux vector resolves to the base target via the round trip."""
        rng = np.random.default_rng(7)
        pos = rng.normal(size=3)
        quat = _random_unit_quat(rng)
        target = rng.normal(size=3)
        offset_tcp = tcp_frame_offset(pos, quat, target)
        pred = predict_from_aux(pos, quat, offset_tcp, frame="tcp")
        np.testing.assert_allclose(pred.target_base, target, atol=1e-12)
        self.assertAlmostEqual(
            pred.magnitude, float(np.linalg.norm(target - pos)), places=12
        )

    def test_base_frame_prediction_is_direct(self) -> None:
        pos = np.array([0.1, 0.2, 0.3])
        quat = _quat_z(0.9)  # ignored for base-frame input
        offset = np.array([0.02, -0.01, 0.03])
        pred = predict_from_aux(pos, quat, offset, frame="base")
        np.testing.assert_allclose(pred.offset_base, offset, atol=1e-12)
        np.testing.assert_allclose(pred.target_base, pos + offset, atol=1e-12)

    def test_six_dim_axis_is_unit(self) -> None:
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        aux = np.array([0.0, 0.0, -0.05, 0.0, 0.0, -2.0])  # offset + non-unit axis
        pred = predict_from_aux(pos, quat, aux, frame="tcp")
        assert pred.axis_base is not None
        np.testing.assert_allclose(pred.axis_base, [0.0, 0.0, -1.0], atol=1e-12)

    def test_plausible_gate_boundaries(self) -> None:
        pred = PortOffsetPrediction(
            offset_base=np.array([0.0, 0.0, -0.05]),
            target_base=np.array([0.0, 0.0, -0.05]),
            magnitude=0.05,
        )
        self.assertTrue(pred.plausible(0.005, 0.12))
        self.assertFalse(pred.plausible(0.06, 0.12))  # below min
        self.assertFalse(pred.plausible(0.005, 0.04))  # above max
        # inclusive boundaries
        self.assertTrue(pred.plausible(0.05, 0.05))

    def test_plausible_rejects_non_finite(self) -> None:
        pred = PortOffsetPrediction(
            offset_base=np.array([np.nan, 0.0, 0.0]),
            target_base=np.zeros(3),
            magnitude=float("nan"),
        )
        self.assertFalse(pred.plausible(0.0, 1.0))

    def test_bad_aux_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            predict_from_aux(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
