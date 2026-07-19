"""Unit tests for gen_config.py config randomization.

Verifies the perturbation stays within the documented eval ranges, is deterministic
under a seed, preserves the aic_engine YAML structure, and does not mutate the base.
Runs without ROS/Gazebo/GPU (pure dict + PyYAML).
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import math
import random
import tempfile
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


def _yaw_in_eval_band(yaw: float) -> bool:
    """Return True if ``yaw`` lies in either eval-band cluster.

    The eval band is the near-+/-pi cluster (``|yaw|`` in
    ``[YAW_PI_BAND_MIN, pi]``) or the ``~-1.8`` side cluster
    (``[YAW_SIDE_BAND_MIN, YAW_SIDE_BAND_MAX]``); see ``gen_config``.
    """
    in_pi_band = gen_config.YAW_PI_BAND_MIN - 1e-9 <= abs(yaw) <= math.pi + 1e-9
    in_side_band = (gen_config.YAW_SIDE_BAND_MIN - 1e-9
                    <= yaw <= gen_config.YAW_SIDE_BAND_MAX + 1e-9)
    return in_pi_band or in_side_band


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


def _trial_sfp() -> dict[str, Any]:
    """A minimal SFP template trial (mirrors eval_config trial_1)."""
    tb: dict[str, Any] = {
        "pose": {"x": 0.16, "y": -0.21, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": 3.1},
        "sc_rail_0": {"entity_present": True, "entity_name": "sc_mount_0",
                      "entity_pose": {"translation": 0.042, "roll": 0.0,
                                      "pitch": 0.0, "yaw": 0.1}},
        "sc_rail_1": {"entity_present": False},
        "lc_mount_rail_0": {"entity_present": True, "entity_name": "lc_mount_0",
                            "entity_pose": {"translation": 0.02, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0}},
        "sfp_mount_rail_0": {"entity_present": True, "entity_name": "sfp_mount_0",
                             "entity_pose": {"translation": 0.03, "roll": 0.0,
                                             "pitch": 0.0, "yaw": 0.0}},
        "sc_mount_rail_0": {"entity_present": True, "entity_name": "sc_mount_0",
                            "entity_pose": {"translation": -0.02, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0}},
        "lc_mount_rail_1": {"entity_present": True, "entity_name": "lc_mount_1",
                            "entity_pose": {"translation": -0.01, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0}},
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }
    for k in range(5):
        tb[f"nic_rail_{k}"] = {"entity_present": False}
    tb["nic_rail_2"] = {"entity_present": True, "entity_name": "nic_card_2",
                        "entity_pose": {"translation": 0.016, "roll": 0.0,
                                        "pitch": 0.0, "yaw": 0.0}}
    return {
        "scene": {
            "task_board": tb,
            "cables": {"cable_0": {
                "pose": {"gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04245},
                         "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
                "attach_cable_to_gripper": True, "cable_type": "sfp_sc_cable"}},
        },
        "tasks": {"task_1": {"cable_type": "sfp_sc", "cable_name": "cable_0",
                             "plug_type": "sfp", "plug_name": "sfp_tip",
                             "port_type": "sfp", "port_name": "sfp_port_0",
                             "target_module_name": "nic_card_mount_2", "time_limit": 180}},
    }


def _trial_sc() -> dict[str, Any]:
    """A minimal SC template trial (mirrors eval_config trial_3, reversed cable)."""
    tb: dict[str, Any] = {
        "pose": {"x": 0.2, "y": -0.15, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": -1.8},
        "sc_rail_0": {"entity_present": True, "entity_name": "sc_mount_0",
                      "entity_pose": {"translation": 0.0, "roll": 0.0,
                                      "pitch": 0.0, "yaw": 0.0}},
        "sc_rail_1": {"entity_present": True, "entity_name": "sc_mount_1",
                      "entity_pose": {"translation": -0.04, "roll": 0.0,
                                      "pitch": 0.0, "yaw": 0.0}},
        "lc_mount_rail_0": {"entity_present": False},
        "sfp_mount_rail_0": {"entity_present": True, "entity_name": "sfp_mount_0",
                             "entity_pose": {"translation": 0.05, "roll": 0.0,
                                             "pitch": 0.0, "yaw": 0.0}},
        "sc_mount_rail_0": {"entity_present": True, "entity_name": "sc_mount_2",
                            "entity_pose": {"translation": -0.03, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0}},
        "lc_mount_rail_1": {"entity_present": True, "entity_name": "lc_mount_1",
                            "entity_pose": {"translation": 0.04, "roll": 0.0,
                                            "pitch": 0.0, "yaw": 0.0}},
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }
    for k in range(5):
        tb[f"nic_rail_{k}"] = {"entity_present": False}
    return {
        "scene": {
            "task_board": tb,
            "cables": {"cable_1": {
                "pose": {"gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04045},
                         "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303},
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable_reversed"}},
        },
        "tasks": {"task_1": {"cable_type": "sfp_sc_cable_reversed", "cable_name": "cable_1",
                             "plug_type": "sc", "plug_name": "sc_tip",
                             "port_type": "sc", "port_name": "sc_port_base",
                             "target_module_name": "sc_port_1", "time_limit": 180}},
    }


def _synthetic_base_full() -> dict[str, Any]:
    """Full synthetic base with both SFP (trial_1) and SC (trial_3) templates."""
    return {
        "trials": {"trial_1": _trial_sfp(), "trial_3": _trial_sc()},
        "robot": {"home_joint_positions": {
            "shoulder_pan_joint": -0.1597, "shoulder_lift_joint": -1.3542,
            "elbow_joint": -1.6648, "wrist_1_joint": -1.6933,
            "wrist_2_joint": 1.5710, "wrist_3_joint": 1.4110}},
    }


class TestEnumerateStrata(unittest.TestCase):
    def test_covers_full_product(self) -> None:
        cells = gen_config.enumerate_strata()
        self.assertEqual(len(cells), 12)
        sfp = {(c.target_rail, c.port_name) for c in cells if c.plug == "sfp"}
        self.assertEqual(
            sfp,
            {(r, p) for r in range(5) for p in ("sfp_port_0", "sfp_port_1")},
        )
        sc = {c.target_rail for c in cells if c.plug == "sc"}
        self.assertEqual(sc, {0, 1})

    def test_names_unique(self) -> None:
        names = [c.name for c in gen_config.enumerate_strata()]
        self.assertEqual(len(names), len(set(names)))


class TestBuildStrataConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.base = _synthetic_base_full()
        self.cells = gen_config.enumerate_strata()

    def test_sfp_target_and_task_fields_no_distractors(self) -> None:
        for c in (x for x in self.cells if x.plug == "sfp"):
            cfg, _ = gen_config.build_strata_config(
                self.base, c, 0, seed=1, distractors=False)
            tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
            task = cfg["trials"]["trial_1"]["tasks"]["task_1"]
            self.assertTrue(tb[f"nic_rail_{c.target_rail}"]["entity_present"])
            self.assertEqual(
                tb[f"nic_rail_{c.target_rail}"]["entity_name"],
                f"nic_card_{c.target_rail}")
            self.assertEqual(task["target_module_name"],
                             f"nic_card_mount_{c.target_rail}")
            self.assertEqual(task["port_name"], c.port_name)
            present = [k for k in range(5)
                       if tb[f"nic_rail_{k}"].get("entity_present")]
            self.assertEqual(present, [c.target_rail])

    def test_sc_target_and_task_fields_no_distractors(self) -> None:
        for c in (x for x in self.cells if x.plug == "sc"):
            cfg, _ = gen_config.build_strata_config(
                self.base, c, 0, seed=1, distractors=False)
            tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
            task = cfg["trials"]["trial_1"]["tasks"]["task_1"]
            self.assertTrue(tb[f"sc_rail_{c.target_rail}"]["entity_present"])
            self.assertEqual(task["target_module_name"], f"sc_port_{c.target_rail}")
            self.assertEqual(task["port_name"], "sc_port_base")
            self.assertEqual(task["cable_type"], "sfp_sc_cable_reversed")
            sc_present = [k for k in range(2)
                          if tb[f"sc_rail_{k}"].get("entity_present")]
            self.assertEqual(sc_present, [c.target_rail])
            nic_present = [k for k in range(5)
                           if tb[f"nic_rail_{k}"].get("entity_present")]
            self.assertEqual(nic_present, [])

    def test_distractors_never_collide_with_target(self) -> None:
        for seed in range(15):
            for idx, c in enumerate(self.cells):
                cfg, row = gen_config.build_strata_config(
                    self.base, c, idx, seed=seed, distractors=True)
                tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
                self.assertGreaterEqual(row.n_distractors, 1)
                if c.plug == "sfp":
                    present = [k for k in range(5)
                               if tb[f"nic_rail_{k}"].get("entity_present")]
                    self.assertIn(c.target_rail, present)
                    distractors = [k for k in present if k != c.target_rail]
                    self.assertTrue(1 <= len(distractors) <= 2)
                else:
                    self.assertTrue(tb[f"sc_rail_{c.target_rail}"]["entity_present"])
                    other = 1 - c.target_rail
                    self.assertTrue(tb[f"sc_rail_{other}"]["entity_present"])
                    nic = [k for k in range(5)
                           if tb[f"nic_rail_{k}"].get("entity_present")]
                    self.assertTrue(1 <= len(nic) <= 2)

    def test_distractors_off_gives_zero_distractors(self) -> None:
        for c in self.cells:
            _, row = gen_config.build_strata_config(
                self.base, c, 0, seed=3, distractors=False)
            self.assertEqual(row.n_distractors, 0)
            self.assertEqual(row.distractor_rails, "")

    def test_continuous_axes_within_ranges(self) -> None:
        for seed in range(8):
            for idx, c in enumerate(self.cells):
                cfg, row = gen_config.build_strata_config(
                    self.base, c, idx, seed=seed)
                tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
                self.assertGreaterEqual(tb["pose"]["x"], 0.15 - 1e-9)
                self.assertLessEqual(tb["pose"]["x"], 0.20 + 1e-9)
                self.assertGreaterEqual(tb["pose"]["y"], -0.21 - 1e-9)
                self.assertLessEqual(tb["pose"]["y"], 0.05 + 1e-9)
                self.assertTrue(
                    _yaw_in_eval_band(tb["pose"]["yaw"]),
                    msg=f"yaw {tb['pose']['yaw']} outside eval band (seed={seed} idx={idx})")
                self.assertGreaterEqual(row.grasp_z, GRASP_Z_MIN - 1e-9)
                self.assertLessEqual(row.grasp_z, GRASP_Z_MAX + 1e-9)
                if c.plug == "sfp":
                    lo, hi = gen_config.NIC_MIN, gen_config.NIC_MAX
                else:
                    lo, hi = gen_config.SC_MIN, gen_config.SC_MAX
                self.assertGreaterEqual(row.target_translation, lo - 1e-9)
                self.assertLessEqual(row.target_translation, hi + 1e-9)

    def test_yaw_always_in_eval_band(self) -> None:
        """Every strata board yaw stays inside the eval band (never yaw~0)."""
        for seed in range(12):
            for idx, c in enumerate(self.cells):
                cfg, row = gen_config.build_strata_config(
                    self.base, c, idx, seed=seed)
                yaw = cfg["trials"]["trial_1"]["scene"]["task_board"]["pose"]["yaw"]
                self.assertTrue(
                    _yaw_in_eval_band(yaw),
                    msg=f"yaw {yaw} outside eval band (seed={seed} idx={idx})")
                self.assertEqual(row.board_yaw, yaw)

    def test_yaw_both_subbands_reachable(self) -> None:
        """Both the +/-pi cluster (both signs) and the ~-1.8 side band are sampled."""
        saw_pi_pos = saw_pi_neg = saw_side = False
        for seed in range(60):
            for idx, c in enumerate(self.cells):
                cfg, _ = gen_config.build_strata_config(self.base, c, idx, seed=seed)
                yaw = cfg["trials"]["trial_1"]["scene"]["task_board"]["pose"]["yaw"]
                if yaw >= gen_config.YAW_PI_BAND_MIN:
                    saw_pi_pos = True
                elif yaw <= -gen_config.YAW_PI_BAND_MIN:
                    saw_pi_neg = True
                elif (gen_config.YAW_SIDE_BAND_MIN <= yaw
                      <= gen_config.YAW_SIDE_BAND_MAX):
                    saw_side = True
            if saw_pi_pos and saw_pi_neg and saw_side:
                break
        self.assertTrue(saw_pi_pos, "positive +/-pi cluster never sampled")
        self.assertTrue(saw_pi_neg, "negative +/-pi cluster never sampled")
        self.assertTrue(saw_side, "~-1.8 side band never sampled")

    def test_sample_eval_band_yaw_unit(self) -> None:
        """The standalone sampler only ever returns in-band yaws."""
        rng = random.Random(0)
        for _ in range(2000):
            self.assertTrue(_yaw_in_eval_band(gen_config.sample_eval_band_yaw(rng)))

    def test_deterministic_under_seed(self) -> None:
        c = self.cells[0]
        a, ra = gen_config.build_strata_config(self.base, c, 3, seed=42)
        b, rb = gen_config.build_strata_config(self.base, c, 3, seed=42)
        self.assertEqual(yaml.safe_dump(a), yaml.safe_dump(b))
        self.assertEqual(dataclasses.asdict(ra), dataclasses.asdict(rb))

    def test_base_not_mutated(self) -> None:
        snapshot = copy.deepcopy(self.base)
        gen_config.build_strata_config(self.base, self.cells[0], 0, seed=1)
        gen_config.build_strata_config(
            self.base, self.cells[-1], 1, seed=1, distractors=True)
        self.assertEqual(self.base, snapshot)

    def test_manifest_row_matches_config(self) -> None:
        c = self.cells[3]
        cfg, row = gen_config.build_strata_config(self.base, c, 5, seed=9)
        tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
        self.assertEqual(row.board_x, tb["pose"]["x"])
        self.assertEqual(row.board_y, tb["pose"]["y"])
        self.assertEqual(row.board_yaw, tb["pose"]["yaw"])
        self.assertEqual(row.stratum, c.name)
        self.assertEqual(row.seed, 9)
        self.assertEqual(row.index, 5)

    def test_single_trial_output(self) -> None:
        cfg, _ = gen_config.build_strata_config(self.base, self.cells[0], 0, seed=1)
        self.assertEqual(list(cfg["trials"].keys()), ["trial_1"])

    def test_missing_template_trial_raises(self) -> None:
        base_no_sc = {"trials": {"trial_1": self.base["trials"]["trial_1"]},
                      "robot": self.base["robot"]}
        sc_cell = next(c for c in self.cells if c.plug == "sc")
        with self.assertRaises(KeyError):
            gen_config.build_strata_config(base_no_sc, sc_cell, 0, seed=1)


class TestManifest(unittest.TestCase):
    def test_write_and_reload_covers_all_cells(self) -> None:
        base = _synthetic_base_full()
        cells = gen_config.enumerate_strata()
        rows = []
        for idx, c in enumerate(cells):
            _, row = gen_config.build_strata_config(base, c, idx, seed=1)
            row.config = f"/tmp/cfg_{idx}.yaml"
            rows.append(row)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.csv"
            gen_config.write_manifest(rows, str(path))
            with open(path, newline="") as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(len(got), 12)
        self.assertEqual({r["stratum"] for r in got}, {c.name for c in cells})
        # every column of StrataManifestRow is present in the CSV header
        expected_cols = {f.name for f in dataclasses.fields(gen_config.StrataManifestRow)}
        self.assertEqual(set(got[0].keys()), expected_cols)


class TestStrataStructureVsEval(unittest.TestCase):
    """Diff generated strata configs against the real eval_config topology (item 4)."""

    def setUp(self) -> None:
        base = _load_real_base()
        if base is None:
            self.skipTest(f"eval_config not found at {EVAL_CONFIG}")
        self.base = base

    def test_sfp_matches_trial_2_topology(self) -> None:
        cell = gen_config.Stratum("sfp", 4, "sfp_port_1")  # like trial_2 target
        cfg, _ = gen_config.build_strata_config(self.base, cell, 0, seed=1)
        gen_tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
        real_tb = self.base["trials"]["trial_2"]["scene"]["task_board"]
        self.assertEqual(set(gen_tb.keys()), set(real_tb.keys()))
        gen_task = cfg["trials"]["trial_1"]["tasks"]["task_1"]
        real_task = self.base["trials"]["trial_2"]["tasks"]["task_1"]
        self.assertEqual(set(gen_task.keys()), set(real_task.keys()))
        self.assertEqual(gen_task["target_module_name"], "nic_card_mount_4")

    def test_sc_matches_trial_3_topology(self) -> None:
        cell = gen_config.Stratum("sc", 1, "sc_port_base")
        cfg, _ = gen_config.build_strata_config(self.base, cell, 0, seed=1)
        gen_tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
        real_tb = self.base["trials"]["trial_3"]["scene"]["task_board"]
        self.assertEqual(set(gen_tb.keys()), set(real_tb.keys()))
        gen_task = cfg["trials"]["trial_1"]["tasks"]["task_1"]
        self.assertEqual(gen_task["cable_type"], "sfp_sc_cable_reversed")
        self.assertEqual(gen_task["target_module_name"], "sc_port_1")
        cable = next(iter(cfg["trials"]["trial_1"]["scene"]["cables"].values()))
        self.assertEqual(cable["cable_type"], "sfp_sc_cable_reversed")
        self.assertTrue(GRASP_Z_MIN <= cable["pose"]["gripper_offset"]["z"] <= GRASP_Z_MAX)


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


class TestParseRepsSpec(unittest.TestCase):
    """Parsing of the ``--weights`` per-cell reps-plan string."""

    def test_empty_and_blank_give_empty_map(self) -> None:
        self.assertEqual(gen_config.parse_reps_spec(""), {})
        self.assertEqual(gen_config.parse_reps_spec("   "), {})

    def test_parses_multiple_items_and_strips(self) -> None:
        got = gen_config.parse_reps_spec(" sc_rail0=4 , sfp_rail0_sfp_port_0=3 ")
        self.assertEqual(got, {"sc_rail0": 4, "sfp_rail0_sfp_port_0": 3})

    def test_zero_reps_allowed(self) -> None:
        self.assertEqual(gen_config.parse_reps_spec("sc_rail1=0"), {"sc_rail1": 0})

    def test_trailing_comma_ignored(self) -> None:
        self.assertEqual(gen_config.parse_reps_spec("sc_rail0=2,"), {"sc_rail0": 2})

    def test_missing_equals_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.parse_reps_spec("sc_rail0")

    def test_non_integer_reps_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.parse_reps_spec("sc_rail0=two")

    def test_negative_reps_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.parse_reps_spec("sc_rail0=-1")

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.parse_reps_spec("=4")

    def test_duplicate_stratum_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.parse_reps_spec("sc_rail0=2,sc_rail0=3")


class TestBuildStrataPlan(unittest.TestCase):
    """Per-cell rep planning that drives failure-driven oversampling."""

    def setUp(self) -> None:
        self.strata = gen_config.enumerate_strata()

    def test_default_reps_applied_uniformly(self) -> None:
        plan = gen_config.build_strata_plan(self.strata, 2)
        self.assertEqual([r for _, r in plan], [2] * 12)
        self.assertEqual([s.name for s, _ in plan],
                         [s.name for s in self.strata])

    def test_overrides_take_precedence(self) -> None:
        plan = gen_config.build_strata_plan(
            self.strata, 2, {"sc_rail0": 4, "sfp_rail0_sfp_port_0": 0})
        reps = {s.name: r for s, r in plan}
        self.assertEqual(reps["sc_rail0"], 4)
        self.assertEqual(reps["sfp_rail0_sfp_port_0"], 0)
        self.assertEqual(reps["sc_rail1"], 2)  # untouched -> default

    def test_unknown_stratum_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.build_strata_plan(self.strata, 1, {"sfp_rail9_sfp_port_0": 3})

    def test_negative_default_raises(self) -> None:
        with self.assertRaises(ValueError):
            gen_config.build_strata_plan(self.strata, -1)

    def test_plan_preserves_stratum_order(self) -> None:
        plan = gen_config.build_strata_plan(self.strata, 1, {"sc_rail1": 5})
        self.assertEqual([s.name for s, _ in plan],
                         [s.name for s in self.strata])

    def test_phase2_plan_totals(self) -> None:
        """The committed Phase-2 plan yields 40 configs with an 8-SC / 32-SFP split."""
        overrides = gen_config.parse_reps_spec(
            "sc_rail0=4,sc_rail1=4,"
            "sfp_rail0_sfp_port_0=4,sfp_rail1_sfp_port_0=4,sfp_rail2_sfp_port_0=4,"
            "sfp_rail3_sfp_port_0=4,sfp_rail4_sfp_port_0=4,"
            "sfp_rail0_sfp_port_1=3,sfp_rail1_sfp_port_1=2,sfp_rail2_sfp_port_1=2,"
            "sfp_rail3_sfp_port_1=2,sfp_rail4_sfp_port_1=3")
        plan = gen_config.build_strata_plan(self.strata, 0, overrides)
        reps = {s.name: r for s, r in plan}
        total = sum(reps.values())
        sc_total = sum(r for s, r in plan if s.plug == "sc")
        sfp_total = sum(r for s, r in plan if s.plug == "sfp")
        self.assertEqual(total, 40)
        self.assertEqual(sc_total, 8)
        self.assertEqual(sfp_total, 32)
        # every cell is explicitly planned (no cell falls back to the 0 default).
        self.assertTrue(all(r > 0 for r in reps.values()))


class TestRunStrataWeighted(unittest.TestCase):
    """End-to-end strata emission honours the per-cell plan on disk + manifest."""

    def _args(self, outdir: str, **kw: Any) -> argparse.Namespace:
        base = dict(o=outdir, seed=7, prefix="p2", reps=0, weights="",
                    distractors=True, manifest="")
        base.update(kw)
        return argparse.Namespace(**base)

    def test_uniform_matches_legacy_rep_major(self) -> None:
        """No overrides + reps=R reproduces the R-per-cell rep-major layout."""
        base = _synthetic_base_full()
        with tempfile.TemporaryDirectory() as d:
            gen_config._run_strata(base, self._args(d, reps=2))
            with open(Path(d) / "manifest.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 24)  # 12 cells x 2 reps
        # rep-major: first 12 rows are r0 for every cell, next 12 are r1.
        self.assertTrue(all(r["config"].endswith("_r0.yaml") for r in rows[:12]))
        self.assertTrue(all(r["config"].endswith("_r1.yaml") for r in rows[12:]))

    def test_weighted_emits_planned_counts(self) -> None:
        base = _synthetic_base_full()
        weights = ("sc_rail0=4,sc_rail1=4,sfp_rail0_sfp_port_0=4,"
                   "sfp_rail1_sfp_port_0=1,sfp_rail0_sfp_port_1=0")
        with tempfile.TemporaryDirectory() as d:
            gen_config._run_strata(base, self._args(d, reps=0, weights=weights))
            manifest = Path(d) / "manifest.csv"
            with open(manifest, newline="") as fh:
                rows = list(csv.DictReader(fh))
            files = sorted(p.name for p in Path(d).glob("p2_*.yaml"))
        per_cell: dict[str, int] = {}
        for r in rows:
            per_cell[r["stratum"]] = per_cell.get(r["stratum"], 0) + 1
        self.assertEqual(per_cell.get("sc_rail0"), 4)
        self.assertEqual(per_cell.get("sc_rail1"), 4)
        self.assertEqual(per_cell.get("sfp_rail0_sfp_port_0"), 4)
        self.assertEqual(per_cell.get("sfp_rail1_sfp_port_0"), 1)
        # reps=0 cells are omitted entirely (default reps=0 too).
        self.assertNotIn("sfp_rail0_sfp_port_1", per_cell)
        self.assertNotIn("sfp_rail2_sfp_port_0", per_cell)
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(files), 13)  # one YAML per manifest row
        # config filenames are resumable (unique) and end in _r<rep>.
        self.assertEqual(len(set(files)), 13)

    def test_manifest_configs_parse_as_campaign_tasks(self) -> None:
        """collect_campaign's parser consumes the emitted manifest unchanged."""
        import campaign_lib
        base = _synthetic_base_full()
        with tempfile.TemporaryDirectory() as d:
            gen_config._run_strata(
                base, self._args(d, weights="sc_rail0=2,sfp_rail0_sfp_port_0=2"))
            manifest = str(Path(d) / "manifest.csv")
            tasks = campaign_lib.load_tasks(manifest)
            sc_tasks = campaign_lib.load_tasks(manifest, plug="sc")
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len(sc_tasks), 2)
        self.assertEqual({t.episode_dir for t in tasks},
                         {"ep_sc_rail0_r0", "ep_sc_rail0_r1",
                          "ep_sfp_rail0_sfp_port_0_r0", "ep_sfp_rail0_sfp_port_0_r1"})


if __name__ == "__main__":
    unittest.main()
