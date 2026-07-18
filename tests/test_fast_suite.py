"""Unit tests for ``eval_lib.fast_suite`` (fast-protocol suite derivation).

All tests are pure file/YAML logic and run without ROS, Gazebo, or a GPU
(CLAUDE.md section 2).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from eval_lib import fast_suite


def _sample_config(time_limit: int = 180) -> dict:
    """Build a minimal engine-config-shaped document with one timed task."""
    return {
        "scoring": {"weights": {"tier_1": 1.0}},
        "trials": {
            "trial_1": {
                "scene": {"task_board": {"pose": {"x": 0.17}}},
                "tasks": {
                    "task_1": {
                        "cable_type": "sfp_sc",
                        "port_name": "sfp_port_0",
                        "time_limit": time_limit,
                    }
                },
            }
        },
    }


class SetTaskTimeLimitsTest(unittest.TestCase):
    """Tests for the in-place time-limit rewriter."""

    def test_rewrites_single_task(self) -> None:
        cfg = _sample_config(180)
        n = fast_suite.set_task_time_limits(cfg, 60)
        self.assertEqual(n, 1)
        self.assertEqual(cfg["trials"]["trial_1"]["tasks"]["task_1"]["time_limit"], 60)

    def test_rewrites_multiple_trials_and_tasks(self) -> None:
        cfg = {
            "trials": {
                "trial_1": {"tasks": {"task_1": {"time_limit": 180}}},
                "trial_2": {
                    "tasks": {
                        "task_1": {"time_limit": 200},
                        "task_2": {"time_limit": 90},
                    }
                },
            }
        }
        n = fast_suite.set_task_time_limits(cfg, 60)
        self.assertEqual(n, 3)
        for trial in cfg["trials"].values():
            for task in trial["tasks"].values():
                self.assertEqual(task["time_limit"], 60)

    def test_leaves_tasks_without_limit_untouched(self) -> None:
        cfg = {"trials": {"trial_1": {"tasks": {"task_1": {"port_name": "p"}}}}}
        n = fast_suite.set_task_time_limits(cfg, 60)
        self.assertEqual(n, 0)
        self.assertNotIn("time_limit", cfg["trials"]["trial_1"]["tasks"]["task_1"])

    def test_rejects_nonpositive_limit(self) -> None:
        with self.assertRaises(ValueError):
            fast_suite.set_task_time_limits(_sample_config(), 0)
        with self.assertRaises(ValueError):
            fast_suite.set_task_time_limits(_sample_config(), -5)

    def test_rejects_config_without_trials(self) -> None:
        with self.assertRaises(ValueError):
            fast_suite.set_task_time_limits({"scoring": {}}, 60)


class RewriteConfigFileTest(unittest.TestCase):
    """Tests for the single-file rewrite helper."""

    def test_rewrite_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.yaml"
            dst = Path(td) / "out" / "dst.yaml"
            with src.open("w") as fh:
                yaml.safe_dump(_sample_config(180), fh)
            n = fast_suite.rewrite_config_file(src, dst, 60)
            self.assertEqual(n, 1)
            with dst.open() as fh:
                out = yaml.safe_load(fh)
            self.assertEqual(
                out["trials"]["trial_1"]["tasks"]["task_1"]["time_limit"], 60
            )

    def test_missing_source_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                fast_suite.rewrite_config_file(
                    Path(td) / "nope.yaml", Path(td) / "o.yaml", 60
                )

    def test_config_without_limit_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.yaml"
            with src.open("w") as fh:
                yaml.safe_dump({"trials": {"t": {"tasks": {"a": {}}}}}, fh)
            with self.assertRaises(ValueError):
                fast_suite.rewrite_config_file(src, Path(td) / "o.yaml", 60)


class BuildFastSuiteTest(unittest.TestCase):
    """Tests for the end-to-end suite derivation."""

    def _make_src_suite(self, root: Path, n_configs: int = 3) -> Path:
        """Create a minimal source suite on disk and return its directory."""
        src = root / "suite_src"
        (src / "configs").mkdir(parents=True)
        for i in range(n_configs):
            with (src / "configs" / f"cfg_{i:03d}.yaml").open("w") as fh:
                yaml.safe_dump(_sample_config(180), fh)
        (src / "manifest.csv").write_text("config_id,config_file\ncfg_000,configs/cfg_000.yaml\n")
        with (src / "suite_meta.yaml").open("w") as fh:
            yaml.safe_dump({"seed": 20260712, "n_total": n_configs}, fh)
        return src

    def test_builds_suite_with_shortened_limits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = self._make_src_suite(Path(td), n_configs=3)
            out = Path(td) / "suite_fast"
            result = fast_suite.build_fast_suite(src, out, time_limit_s=60)

            self.assertEqual(result.n_configs, 3)
            self.assertEqual(result.n_limits_rewritten, 3)
            self.assertEqual(result.time_limit_s, 60)

            for cfg in sorted((out / "configs").glob("*.yaml")):
                with cfg.open() as fh:
                    doc = yaml.safe_load(fh)
                self.assertEqual(
                    doc["trials"]["trial_1"]["tasks"]["task_1"]["time_limit"], 60
                )

    def test_manifest_copied_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = self._make_src_suite(Path(td))
            out = Path(td) / "suite_fast"
            fast_suite.build_fast_suite(src, out, time_limit_s=60)
            self.assertEqual(
                (out / "manifest.csv").read_text(), (src / "manifest.csv").read_text()
            )

    def test_meta_carries_fast_protocol_tag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = self._make_src_suite(Path(td))
            out = Path(td) / "suite_fast"
            fast_suite.build_fast_suite(src, out, time_limit_s=60)
            with (out / "suite_meta.yaml").open() as fh:
                meta = yaml.safe_load(fh)
            self.assertEqual(meta[fast_suite.FAST_PROTOCOL_KEY], "time_limit=60")
            self.assertEqual(meta["fast_time_limit_s"], 60)
            self.assertEqual(meta["official_time_limit_s"], 180)
            # Source provenance preserved.
            self.assertEqual(meta["seed"], 20260712)

    def test_missing_source_manifest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "empty" / "configs").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                fast_suite.build_fast_suite(
                    Path(td) / "empty", Path(td) / "o", time_limit_s=60
                )


if __name__ == "__main__":
    unittest.main()
