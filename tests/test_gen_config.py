"""Unit tests for gen_config.py config randomization.

Verifies the perturbation stays within the documented eval ranges, is deterministic
under a seed, preserves the aic_engine YAML structure, and does not mutate the base.
Runs without ROS/Gazebo/GPU (pure dict + PyYAML).
"""
from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from typing import Any

import yaml

import gen_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = REPO_ROOT / "aic_engine" / "config" / "eval_config.yaml"

# Documented eval ranges (see mission / eval_config task_board_limits).
NIC_MIN, NIC_MAX = -0.0215, 0.0234
CARD_YAW_LIMIT = math.radians(10.0)  # +/- 10 degrees
GRASP_Z_MIN, GRASP_Z_MAX = 0.040, 0.046


def _synthetic_base() -> dict[str, Any]:
    """Build a minimal base config with every field perturb() touches."""
    return {
        "trials": {
            "trial_1": {
                "scene": {
                    "task_board": {
                        "pose": {"x": 0.16, "y": -0.21, "z": 1.14,
                                 "roll": 0.0, "pitch": 0.0, "yaw": 3.1},
                        "nic_rail_0": {"entity_present": False},
                        "nic_rail_1": {"entity_present": False},
                        "nic_rail_2": {
                            "entity_present": True,
                            "entity_name": "nic_card_2",
                            "entity_pose": {"translation": 0.016, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0},
                        },
                        "nic_rail_3": {"entity_present": False},
                        "nic_rail_4": {"entity_present": False},
                    },
                    "cables": {
                        "cable_0": {
                            "pose": {
                                "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04245},
                                "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
                            }
                        }
                    },
                },
                "tasks": {
                    "task_1": {"port_name": "sfp_port_0",
                               "target_module_name": "nic_card_mount_2"}
                },
            }
        },
        "robot": {
            "home_joint_positions": {
                "shoulder_pan_joint": -0.1597, "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648, "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710, "wrist_3_joint": 1.4110,
            }
        },
    }


def _load_real_base() -> dict[str, Any] | None:
    if not EVAL_CONFIG.exists():
        return None
    with open(EVAL_CONFIG) as fh:
        return yaml.safe_load(fh)


class TestStructureAndDeterminism(unittest.TestCase):
    """Structure preservation, determinism and non-mutation (synthetic base)."""

    def setUp(self) -> None:
        self.base = _synthetic_base()

    def test_structure_preserved(self) -> None:
        c = gen_config.perturb(self.base, 0, seed=1, mode="near")
        self.assertEqual(list(c["trials"].keys()), ["trial_1"])
        tb = c["trials"]["trial_1"]["scene"]["task_board"]
        self.assertIn("pose", tb)
        self.assertIn("task_1", c["trials"]["trial_1"]["tasks"])
        self.assertIn("home_joint_positions", c["robot"])

    def test_yaml_roundtrip(self) -> None:
        c = gen_config.perturb(self.base, 2, seed=7, mode="near")
        reloaded = yaml.safe_load(yaml.safe_dump(c, sort_keys=False))
        self.assertEqual(
            reloaded["trials"]["trial_1"]["tasks"]["task_1"]["port_name"],
            "sfp_port_0",
        )

    def test_deterministic_under_seed(self) -> None:
        a = gen_config.perturb(self.base, 3, seed=42, mode="near")
        b = gen_config.perturb(self.base, 3, seed=42, mode="near")
        self.assertEqual(yaml.safe_dump(a), yaml.safe_dump(b))

    def test_seed_changes_output(self) -> None:
        xs = {
            gen_config.perturb(self.base, 0, seed=s, mode="near")
            ["trials"]["trial_1"]["scene"]["task_board"]["pose"]["x"]
            for s in range(5)
        }
        self.assertGreater(len(xs), 1, "different seeds should vary board x")

    def test_base_not_mutated(self) -> None:
        snapshot = copy.deepcopy(self.base)
        gen_config.perturb(self.base, 0, seed=1, mode="wide")
        self.assertEqual(self.base, snapshot)

    def test_wide_single_rail_and_valid_port(self) -> None:
        for i in range(30):
            c = gen_config.perturb(self.base, i, seed=100, mode="wide")
            tb = c["trials"]["trial_1"]["scene"]["task_board"]
            present = [k for k in range(5) if tb[f"nic_rail_{k}"].get("entity_present")]
            self.assertEqual(len(present), 1, "exactly one NIC rail must be present")
            port = c["trials"]["trial_1"]["tasks"]["task_1"]["port_name"]
            self.assertIn(port, ("sfp_port_0", "sfp_port_1"))


class TestEvalRanges(unittest.TestCase):
    """Randomized values stay within the documented eval ranges (real base)."""

    def setUp(self) -> None:
        base = _load_real_base()
        if base is None:
            self.skipTest(f"eval_config not found at {EVAL_CONFIG}")
        self.base = base

    def test_near_mode_ranges(self) -> None:
        base_z = self.base["trials"]["trial_1"]["scene"]["cables"]["cable_0"]["pose"][
            "gripper_offset"]["z"]
        base_x = self.base["trials"]["trial_1"]["scene"]["task_board"]["pose"]["x"]
        base_y = self.base["trials"]["trial_1"]["scene"]["task_board"]["pose"]["y"]
        for i in range(200):
            c = gen_config.perturb(self.base, i, seed=2026, mode="near")
            tb = c["trials"]["trial_1"]["scene"]["task_board"]
            nic = tb["nic_rail_2"]["entity_pose"]
            self.assertGreaterEqual(nic["translation"], NIC_MIN)
            self.assertLessEqual(nic["translation"], NIC_MAX)
            self.assertLessEqual(abs(nic["yaw"]), CARD_YAW_LIMIT)
            gz = c["trials"]["trial_1"]["scene"]["cables"]["cable_0"]["pose"][
                "gripper_offset"]["z"]
            self.assertGreaterEqual(gz, GRASP_Z_MIN)
            self.assertLessEqual(gz, GRASP_Z_MAX)
            self.assertAlmostEqual(tb["pose"]["x"], base_x, delta=0.015 + 1e-9)
            self.assertAlmostEqual(tb["pose"]["y"], base_y, delta=0.015 + 1e-9)
        # base grasp z within the documented window (sanity on the fixture itself).
        self.assertTrue(GRASP_Z_MIN <= base_z <= GRASP_Z_MAX)

    def test_wide_mode_ranges(self) -> None:
        for i in range(200):
            c = gen_config.perturb(self.base, i, seed=2026, mode="wide")
            tb = c["trials"]["trial_1"]["scene"]["task_board"]
            self.assertGreaterEqual(tb["pose"]["x"], 0.16 - 0.03 - 1e-9)
            self.assertLessEqual(tb["pose"]["x"], 0.16 + 0.04 + 1e-9)
            self.assertGreaterEqual(tb["pose"]["y"], -0.21 - 0.02 - 1e-9)
            self.assertLessEqual(tb["pose"]["y"], -0.21 + 0.26 + 1e-9)
            rail = next(k for k in range(5)
                        if tb[f"nic_rail_{k}"].get("entity_present"))
            nic = tb[f"nic_rail_{rail}"]["entity_pose"]
            self.assertGreaterEqual(nic["translation"], NIC_MIN)
            self.assertLessEqual(nic["translation"], NIC_MAX)
            self.assertLessEqual(abs(nic["yaw"]), CARD_YAW_LIMIT)

    def test_home_joints_bounded(self) -> None:
        base_joints = self.base["robot"]["home_joint_positions"]
        for i in range(50):
            c = gen_config.perturb(self.base, i, seed=5, mode="near")
            for j, v in c["robot"]["home_joint_positions"].items():
                self.assertAlmostEqual(v, base_joints[j], delta=0.02 + 1e-9)


if __name__ == "__main__":
    unittest.main()
