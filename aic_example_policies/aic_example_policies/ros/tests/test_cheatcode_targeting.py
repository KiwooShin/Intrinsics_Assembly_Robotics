#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free CheatCode target-frame resolution.

Exercises :mod:`aic_example_policies.ros.cheatcode_targeting` -- the pure mapping
from a task's ``port_type`` to the TF frame ``CheatCode`` should approach and the
descent floor it should stop at. Requires neither ROS, Gazebo, torch, nor a GPU.

Run with::

    python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""
from __future__ import annotations

import dataclasses
import math
import unittest

import numpy as np

from aic_example_policies.ros import cheatcode_targeting as ct


class ResolvePortApproachTest(unittest.TestCase):
    """Port-type -> (frame, descent floor) mapping."""

    def test_sfp_targets_plain_link_and_deep_floor(self) -> None:
        a = ct.resolve_port_approach("nic_card_mount_2", "sfp_port_0", "sfp")
        self.assertEqual(a.frame, "task_board/nic_card_mount_2/sfp_port_0_link")
        self.assertEqual(a.descent_floor_z, ct.SFP_DESCENT_FLOOR_Z)
        self.assertEqual(a.descent_floor_z, -0.015)

    def test_sfp_behaviour_is_byte_identical_to_legacy(self) -> None:
        # The pre-change primitive built exactly this frame and stopped at -0.015.
        for module, port in (
            ("nic_card_mount_0", "sfp_port_0"),
            ("nic_card_mount_4", "sfp_port_1"),
        ):
            a = ct.resolve_port_approach(module, port, "sfp")
            self.assertEqual(a.frame, f"task_board/{module}/{port}_link")
            self.assertEqual(a.descent_floor_z, -0.015)

    def test_sc_retargets_to_entrance_and_shallow_floor(self) -> None:
        a = ct.resolve_port_approach("sc_port_0", "sc_port_base", "sc")
        self.assertEqual(
            a.frame, "task_board/sc_port_0/sc_port_base_link_entrance"
        )
        self.assertEqual(a.descent_floor_z, ct.SC_DESCENT_FLOOR_Z)
        # Floor fix: -0.005 was too shallow (partial-insert, ins==0); -0.007 seats
        # ~2 mm deeper while staying well inside the -0.01564 entrance offset.
        self.assertEqual(a.descent_floor_z, -0.007)

    def test_sc_descent_floor_constant_value(self) -> None:
        self.assertAlmostEqual(ct.SC_DESCENT_FLOOR_Z, -0.007)
        # Must stay shallower than the -0.01564 entrance offset so it never reaches
        # the port body (which caused the original -0.015 ram).
        self.assertGreater(ct.SC_DESCENT_FLOOR_Z, -0.01564)

    def test_sc_frame_matches_model_sdf_link_name(self) -> None:
        # SC Port/model.sdf publishes link ``sc_port_base_link_entrance`` -- the
        # resolved frame's leaf must be exactly that link under the SC module.
        a = ct.resolve_port_approach("sc_port_1", "sc_port_base", "sc")
        self.assertTrue(a.frame.endswith("/sc_port_base_link_entrance"))
        self.assertEqual(a.frame, "task_board/sc_port_1/sc_port_base_link_entrance")

    def test_sc_floor_is_shallower_than_sfp(self) -> None:
        sc = ct.resolve_port_approach("sc_port_0", "sc_port_base", "sc")
        sfp = ct.resolve_port_approach("nic_card_mount_0", "sfp_port_0", "sfp")
        self.assertGreater(sc.descent_floor_z, sfp.descent_floor_z)

    def test_port_type_is_case_and_whitespace_insensitive(self) -> None:
        for raw in ("sc", "SC", " Sc ", "sc\n"):
            a = ct.resolve_port_approach("sc_port_0", "sc_port_base", raw)
            self.assertTrue(a.frame.endswith("_entrance"), msg=f"raw={raw!r}")

    def test_unknown_port_type_falls_back_to_sfp_behaviour(self) -> None:
        a = ct.resolve_port_approach("some_module", "some_port", "usb")
        self.assertEqual(a.frame, "task_board/some_module/some_port_link")
        self.assertEqual(a.descent_floor_z, ct.SFP_DESCENT_FLOOR_Z)

    def test_empty_identifiers_raise(self) -> None:
        with self.assertRaises(ValueError):
            ct.resolve_port_approach("", "sfp_port_0", "sfp")
        with self.assertRaises(ValueError):
            ct.resolve_port_approach("nic_card_mount_2", "", "sfp")
        with self.assertRaises(ValueError):
            ct.resolve_port_approach("nic_card_mount_2", "sfp_port_0", "")

    def test_port_approach_is_frozen(self) -> None:
        a = ct.resolve_port_approach("sc_port_0", "sc_port_base", "sc")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.frame = "x"  # type: ignore[misc]


def _yaw_quat(angle: float) -> list[float]:
    """Quaternion [x, y, z, w] for a rotation of ``angle`` rad about world +Z."""
    return [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]


def _roll_quat(angle: float) -> list[float]:
    """Quaternion [x, y, z, w] for a rotation of ``angle`` rad about world +X."""
    return [math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)]


_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


class ScEntranceWaypointTest(unittest.TestCase):
    """Pose-conditioned SC pre-insertion waypoint geometry + validation."""

    def test_geometry_constants(self) -> None:
        self.assertEqual(ct.PORT_INSERTION_AXIS_LOCAL, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(ct.SC_APPROACH_STANDOFF_M, 0.2)

    def test_identity_pose_steps_back_along_local_axis(self) -> None:
        # Identity orientation: into-port axis is world +z, so the staging point is
        # one standoff below the entrance and the axis is +z.
        wp = ct.sc_entrance_waypoint([0.0, 0.0, 0.0], _IDENTITY_QUAT, standoff_m=0.2)
        np.testing.assert_allclose(wp.insertion_axis, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(wp.approach_point, [0.0, 0.0, -0.2], atol=1e-12)
        self.assertEqual(wp.standoff_m, 0.2)

    def test_default_standoff_is_named_constant(self) -> None:
        wp = ct.sc_entrance_waypoint([0.0, 0.0, 0.0], _IDENTITY_QUAT)
        self.assertEqual(wp.standoff_m, ct.SC_APPROACH_STANDOFF_M)
        np.testing.assert_allclose(
            wp.approach_point, [0.0, 0.0, -ct.SC_APPROACH_STANDOFF_M], atol=1e-12
        )

    def test_translated_port_translates_waypoint(self) -> None:
        wp = ct.sc_entrance_waypoint([0.5, -0.3, 0.2], _IDENTITY_QUAT, standoff_m=0.05)
        np.testing.assert_allclose(wp.insertion_axis, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(wp.approach_point, [0.5, -0.3, 0.15], atol=1e-12)

    def test_yaw_about_vertical_leaves_axis_vertical(self) -> None:
        # Board yaw rotates about world +z; a vertical insertion axis is invariant.
        for yaw in (0.6, 1.32, -1.8, 3.1):
            wp = ct.sc_entrance_waypoint([0.1, 0.2, 0.3], _yaw_quat(yaw), standoff_m=0.2)
            np.testing.assert_allclose(
                wp.insertion_axis, [0.0, 0.0, 1.0], atol=1e-9, err_msg=f"yaw={yaw}"
            )
            np.testing.assert_allclose(
                wp.approach_point, [0.1, 0.2, 0.1], atol=1e-9, err_msg=f"yaw={yaw}"
            )

    def test_rolled_port_tilts_axis_and_waypoint(self) -> None:
        # A 90 deg roll about +x maps local +z -> world -y, so the staging point
        # offsets in +y (out of the mouth), never straight down -- this is the case
        # a fixed world-z waypoint gets wrong for a mounted-rotated SC port.
        wp = ct.sc_entrance_waypoint([1.0, 1.0, 1.0], _roll_quat(math.pi / 2), standoff_m=0.2)
        np.testing.assert_allclose(wp.insertion_axis, [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(wp.approach_point, [1.0, 1.2, 1.0], atol=1e-9)

    def test_insertion_axis_is_unit(self) -> None:
        for quat in (_IDENTITY_QUAT, _yaw_quat(1.0), _roll_quat(0.7), [1.0, 2.0, 3.0, 4.0]):
            wp = ct.sc_entrance_waypoint([0.0, 0.0, 0.0], quat)
            self.assertAlmostEqual(float(np.linalg.norm(wp.insertion_axis)), 1.0, places=9)

    def test_approach_equals_entrance_minus_standoff_axis(self) -> None:
        # The defining relation holds for an arbitrary (unnormalized) quaternion.
        pos = np.array([0.13, -0.42, 0.31])
        quat = [0.1, -0.3, 0.5, 0.8]
        wp = ct.sc_entrance_waypoint(pos, quat, standoff_m=0.037)
        np.testing.assert_allclose(
            wp.approach_point, pos - 0.037 * wp.insertion_axis, atol=1e-12
        )

    def test_zero_standoff_returns_entrance_point(self) -> None:
        wp = ct.sc_entrance_waypoint([0.4, 0.5, 0.6], _yaw_quat(1.0), standoff_m=0.0)
        np.testing.assert_allclose(wp.approach_point, [0.4, 0.5, 0.6], atol=1e-12)

    def test_bad_position_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([0.0, 0.0], _IDENTITY_QUAT)
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([0.0, 0.0, 0.0, 0.0], _IDENTITY_QUAT)

    def test_nonfinite_position_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([0.0, float("nan"), 0.0], _IDENTITY_QUAT)
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([float("inf"), 0.0, 0.0], _IDENTITY_QUAT)

    def test_negative_or_nonfinite_standoff_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([0.0, 0.0, 0.0], _IDENTITY_QUAT, standoff_m=-0.01)
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint(
                [0.0, 0.0, 0.0], _IDENTITY_QUAT, standoff_m=float("nan")
            )

    def test_zero_norm_quaternion_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.sc_entrance_waypoint([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

    def test_entrance_approach_is_frozen(self) -> None:
        wp = ct.sc_entrance_waypoint([0.0, 0.0, 0.0], _IDENTITY_QUAT)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            wp.standoff_m = 0.1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
