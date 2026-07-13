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
import unittest

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


if __name__ == "__main__":
    unittest.main()
