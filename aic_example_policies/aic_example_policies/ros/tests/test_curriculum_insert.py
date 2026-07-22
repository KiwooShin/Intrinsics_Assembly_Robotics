#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the pure helpers in ``CurriculumInsert`` (ROS/torch-free)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from aic_example_policies.ros import CurriculumInsert as ci


class LateralOffsetVectorTest(unittest.TestCase):
    """``lateral_offset_vector`` geometry and validation."""

    def test_zero_offset_is_zero_vector(self) -> None:
        np.testing.assert_allclose(ci.lateral_offset_vector(0.0, 37.0), [0.0, 0.0])

    def test_magnitude_and_azimuth(self) -> None:
        v = ci.lateral_offset_vector(13.0, 0.0)
        np.testing.assert_allclose(v, [0.013, 0.0], atol=1e-12)
        v = ci.lateral_offset_vector(5.0, 90.0)
        np.testing.assert_allclose(v, [0.0, 0.005], atol=1e-12)
        v = ci.lateral_offset_vector(2.0, 45.0)
        self.assertAlmostEqual(float(np.hypot(*v)), 0.002, places=12)
        self.assertAlmostEqual(math.degrees(math.atan2(v[1], v[0])), 45.0, places=9)

    def test_negative_offset_raises(self) -> None:
        with self.assertRaises(ValueError):
            ci.lateral_offset_vector(-1.0, 0.0)


class IntegrateChunkTargetsTest(unittest.TestCase):
    """``integrate_chunk_targets`` receding-horizon integration."""

    def _chunk(self, vz: float, k: int = 4) -> np.ndarray:
        ch = np.zeros((k, 6))
        ch[:, 2] = vz
        return ch

    def test_shape_and_orientation_held(self) -> None:
        quat = np.array([0.1, 0.2, 0.3, 0.9])
        out = ci.integrate_chunk_targets(
            np.zeros(3), quat, self._chunk(-0.01), exec_steps=4, dt=0.275, substeps=5
        )
        self.assertEqual(out.shape, (20, 7))
        np.testing.assert_allclose(out[:, 3:], np.tile(quat, (20, 1)))

    def test_pure_descent_integrates_cumulatively(self) -> None:
        dt = 0.275
        out = ci.integrate_chunk_targets(
            np.array([0.4, 0.1, 0.2]), np.array([0, 0, 0, 1.0]),
            self._chunk(-0.012), exec_steps=4, dt=dt, substeps=5,
        )
        # Final target: z descends by 4 * v * dt; xy unchanged.
        np.testing.assert_allclose(out[-1, :3],
                                   [0.4, 0.1, 0.2 - 4 * 0.012 * dt], atol=1e-12)
        # Monotone descent across the interpolated stream.
        self.assertTrue(np.all(np.diff(out[:, 2]) < 0))

    def test_substep_interpolation_is_linear(self) -> None:
        out = ci.integrate_chunk_targets(
            np.zeros(3), np.array([0, 0, 0, 1.0]),
            self._chunk(-0.01, k=1), exec_steps=1, dt=1.0, substeps=4,
        )
        np.testing.assert_allclose(out[:, 2], [-0.0025, -0.005, -0.0075, -0.01],
                                   atol=1e-12)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            ci.integrate_chunk_targets(np.zeros(3), np.zeros(4),
                                       np.zeros((4, 5)), 4, 0.1, 5)
        with self.assertRaises(ValueError):
            ci.integrate_chunk_targets(np.zeros(3), np.zeros(4),
                                       self._chunk(-0.01), 5, 0.1, 5)
        with self.assertRaises(ValueError):
            ci.integrate_chunk_targets(np.zeros(3), np.zeros(4),
                                       self._chunk(-0.01), 4, 0.0, 5)
        with self.assertRaises(ValueError):
            ci.integrate_chunk_targets(np.zeros(3), np.zeros(4),
                                       self._chunk(-0.01), 4, 0.1, 0)


class WrenchForceMagTest(unittest.TestCase):
    """``wrench_force_mag`` magnitude + validation for the contact gate."""

    def test_magnitude(self) -> None:
        self.assertAlmostEqual(ci.wrench_force_mag((3.0, 4.0, 0.0)), 5.0)
        self.assertAlmostEqual(ci.wrench_force_mag((0.0, 0.0, 0.0)), 0.0)
        self.assertAlmostEqual(ci.wrench_force_mag((2.0, 3.0, 6.0)), 7.0)

    def test_accepts_ndarray(self) -> None:
        self.assertAlmostEqual(ci.wrench_force_mag(np.array([0.0, -5.0, 12.0])), 13.0)

    def test_gate_threshold_semantics(self) -> None:
        # A free-space reading stays under a typical threshold; a rim contact clears it.
        self.assertLess(ci.wrench_force_mag((0.3, 0.2, 0.5)), 6.0)
        self.assertGreater(ci.wrench_force_mag((1.0, 2.0, 8.0)), 6.0)

    def test_bad_length_raises(self) -> None:
        for bad in ((1.0, 2.0), (1.0, 2.0, 3.0, 4.0), np.zeros(1)):
            with self.assertRaises(ValueError):
                ci.wrench_force_mag(bad)


class ContactSpikeTest(unittest.TestCase):
    """``contact_spike`` = |mag change| > threshold, catching the rig's contact DROP."""

    def test_detects_contact_drop(self) -> None:
        # Free-space ~20 N -> contact ~7 N (a ~13 N drop) fires at threshold 6.
        self.assertTrue(ci.contact_spike((0.0, 0.0, 7.0), (0.0, 0.0, 20.0), 6.0))

    def test_ignores_small_change(self) -> None:
        # Free-space jitter around the ~20 N baseline does not fire.
        self.assertFalse(ci.contact_spike((0.3, 0.0, 20.2), (0.0, 0.0, 20.0), 6.0))

    def test_detects_upward_change_too(self) -> None:
        # Deviation is direction-agnostic: a spike up also fires.
        self.assertTrue(ci.contact_spike((0.0, 0.0, 30.0), (0.0, 0.0, 20.0), 6.0))

    def test_exactly_threshold_does_not_fire(self) -> None:
        self.assertFalse(ci.contact_spike((0.0, 0.0, 14.0), (0.0, 0.0, 20.0), 6.0))


def _yaw_quat(angle: float) -> list[float]:
    """Quaternion [x, y, z, w] for a rotation of ``angle`` rad about world +z."""
    return [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]


def _roll_quat(angle: float) -> list[float]:
    """Quaternion [x, y, z, w] for a rotation of ``angle`` rad about world +x."""
    return [math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)]


_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


class ScAxisFrameTest(unittest.TestCase):
    """``sc_axis_frame`` pose-conditioned SC staging geometry + lateral basis."""

    def test_identity_reduces_to_world_z_line(self) -> None:
        # Identity entrance orientation: the entrance-frame local +z (into the port)
        # maps to world +z, so the insertion axis is +z and the hover steps one
        # standoff back OUT of the mouth -- a pure world-z displacement (the same
        # vertical-line shape as the legacy world-z stage). The sign follows the
        # tested sc_entrance_waypoint convention (local +z = INTO the port), so for
        # this identity pose "out of the mouth" is -z (the physical SC mount is
        # rotated so its real axis points down; see test_downward_axis_*).
        entrance = np.array([0.1, -0.2, 0.3])
        hover, axis, u, v = ci.sc_axis_frame(entrance, _IDENTITY_QUAT, 0.2)
        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            hover, entrance - np.array([0.0, 0.0, 0.2]), atol=1e-12
        )
        # The hover displacement from the entrance is purely along world z.
        disp = hover - entrance
        self.assertAlmostEqual(disp[0], 0.0, places=12)
        self.assertAlmostEqual(disp[1], 0.0, places=12)

    def test_downward_axis_matches_legacy_sfp_hover(self) -> None:
        # A 180-deg roll maps local +z -> world -z: an upward-facing port whose
        # insertion axis points straight DOWN -- exactly the legacy SFP geometry. The
        # hover then sits ``standoff`` ABOVE the entrance (+z), reproducing the world-z
        # ``port.z + standoff`` staging point the SFP branch uses.
        entrance = np.array([0.1, -0.2, 0.3])
        hover, axis, u, v = ci.sc_axis_frame(entrance, _roll_quat(math.pi), 0.05)
        np.testing.assert_allclose(axis, [0.0, 0.0, -1.0], atol=1e-9)
        np.testing.assert_allclose(
            hover, entrance + np.array([0.0, 0.0, 0.05]), atol=1e-9
        )

    def test_hover_steps_back_one_standoff_along_axis(self) -> None:
        # The defining relation holds for an arbitrary (unnormalized) quaternion:
        # hover = entrance - standoff * unit_insertion_axis.
        entrance = np.array([0.13, -0.42, 0.31])
        hover, axis, u, v = ci.sc_axis_frame(entrance, [0.1, -0.3, 0.5, 0.8], 0.037)
        np.testing.assert_allclose(hover, entrance - 0.037 * axis, atol=1e-12)

    def test_rotated_axis_tracks_orientation(self) -> None:
        # 90-deg roll about +x maps local +z -> world -y: the insertion axis rotates
        # accordingly and the hover steps back along it (out of the mouth toward +y).
        # This is the case a fixed world-z waypoint gets wrong for a rotated SC port.
        entrance = np.array([1.0, 1.0, 1.0])
        hover, axis, u, v = ci.sc_axis_frame(entrance, _roll_quat(math.pi / 2), 0.2)
        np.testing.assert_allclose(axis, [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            hover, entrance + np.array([0.0, 0.2, 0.0]), atol=1e-9
        )

    def test_yaw_leaves_axis_vertical(self) -> None:
        # Board yaw is a rotation about world +z; a vertical insertion axis is
        # invariant, so yawed SC poses still reduce to a world-z staging line.
        for yaw in (0.6, 1.32, -1.8):
            hover, axis, u, v = ci.sc_axis_frame(
                np.array([0.1, 0.2, 0.3]), _yaw_quat(yaw), 0.2
            )
            np.testing.assert_allclose(
                axis, [0.0, 0.0, 1.0], atol=1e-9, err_msg=f"yaw={yaw}"
            )
            np.testing.assert_allclose(
                hover, [0.1, 0.2, 0.1], atol=1e-9, err_msg=f"yaw={yaw}"
            )

    def test_insertion_axis_is_unit(self) -> None:
        for quat in (_IDENTITY_QUAT, _yaw_quat(1.0), _roll_quat(0.7), [1.0, 2.0, 3.0, 4.0]):
            _, axis, _, _ = ci.sc_axis_frame([0.0, 0.0, 0.0], quat, 0.2)
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)

    def test_lateral_basis_unit_and_orthogonal_to_axis(self) -> None:
        for quat in (
            _IDENTITY_QUAT,
            _yaw_quat(0.9),
            _roll_quat(0.7),
            _roll_quat(math.pi / 2),
            [0.2, -0.5, 0.3, 0.8],
        ):
            _, axis, u, v = ci.sc_axis_frame([0.3, 0.1, -0.2], quat, 0.15)
            self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=9)
            self.assertAlmostEqual(float(np.dot(u, axis)), 0.0, places=9)
            self.assertAlmostEqual(float(np.dot(v, axis)), 0.0, places=9)
            self.assertAlmostEqual(float(np.dot(u, v)), 0.0, places=9)

    def test_lateral_offset_maps_onto_world_xy_when_vertical(self) -> None:
        # With a vertical (identity) axis the lateral basis is world (x, y), so a
        # world-parameterized curriculum offset lands on world xy exactly as SFP does.
        _, axis, u, v = ci.sc_axis_frame([0.0, 0.0, 0.0], _IDENTITY_QUAT, 0.2)
        np.testing.assert_allclose(u, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)
        dxy = ci.lateral_offset_vector(3.0, 90.0)  # 3 mm along +y
        lat_vec = dxy[0] * u + dxy[1] * v
        np.testing.assert_allclose(lat_vec, [0.0, 0.003, 0.0], atol=1e-12)

    def test_propagates_invalid_pose_and_standoff(self) -> None:
        with self.assertRaises(ValueError):
            ci.sc_axis_frame([0.0, 0.0], _IDENTITY_QUAT, 0.2)
        with self.assertRaises(ValueError):
            ci.sc_axis_frame([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 0.2)
        with self.assertRaises(ValueError):
            ci.sc_axis_frame([0.0, 0.0, 0.0], _IDENTITY_QUAT, -0.1)


class LateralBasisTest(unittest.TestCase):
    """``_lateral_basis`` orthonormality for the SC port-face plane."""

    def test_orthonormal_and_orthogonal_to_axis(self) -> None:
        for raw in ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0],
                    [0.3, -0.4, 0.866]):
            axis = np.asarray(raw, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            u, v = ci._lateral_basis(axis)
            self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=9)
            self.assertAlmostEqual(float(np.dot(u, axis)), 0.0, places=9)
            self.assertAlmostEqual(float(np.dot(v, axis)), 0.0, places=9)
            self.assertAlmostEqual(float(np.dot(u, v)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
