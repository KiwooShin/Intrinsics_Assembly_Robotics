"""Unit tests for the matched-seed evaluation harness (``eval_suite`` / ``eval_lib``).

Covers: suite-generation determinism and strata coverage; ``scoring.yaml``
parsing against synthetic fixtures built to the real engine schema; outcome
classification; Wilson / IQM / bootstrap math against hand-computed values;
compare-mode verdict logic; and a dry-run end-to-end pass. All tests run without
ROS, Gazebo, or a GPU (CLAUDE.md section 2): the sim seam is never invoked here.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

import eval_suite
from eval_lib import report, runner, scoring, stats, suite

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = REPO_ROOT / "aic_engine" / "config" / "eval_config.yaml"


def _fixture_scoring(
    tier_3: float = 75.0,
    force: float = 0.0,
    contacts: float = 0.0,
    duration: float = 9.0,
    smoothness: float = 5.0,
    efficiency: float = 6.0,
    tier_1: float = 1.0,
    trial_id: str = "trial_1",
) -> dict:
    """Build a ``scoring.yaml`` document matching the real engine schema."""
    tier_2 = force + contacts + duration + smoothness + efficiency
    return {
        "total": tier_1 + tier_2 + tier_3,
        trial_id: {
            "tier_1": {"score": tier_1, "message": "Model validation succeeded."},
            "tier_2": {
                "score": tier_2,
                "message": "Scoring succeeded.",
                "categories": {
                    scoring.CAT_INSERTION_FORCE: {"score": force, "message": "f"},
                    scoring.CAT_CONTACTS: {"score": contacts, "message": "c"},
                    scoring.CAT_DURATION: {"score": duration, "message": "d"},
                    scoring.CAT_SMOOTHNESS: {"score": smoothness, "message": "s"},
                    scoring.CAT_EFFICIENCY: {"score": efficiency, "message": "e"},
                },
            },
            "tier_3": {"score": tier_3, "message": "msg"},
        },
    }


class TestStatsMath(unittest.TestCase):
    """Wilson / IQM / bootstrap estimators vs hand-computed values."""

    def test_iqm_even_and_odd(self) -> None:
        # trim_mean(0.25): drop int(0.25*n) each tail.
        self.assertAlmostEqual(stats.iqm(np.arange(1, 9)), 4.5)   # keep [3,4,5,6]
        self.assertAlmostEqual(stats.iqm(np.arange(1, 11)), 5.5)  # keep [3..8]
        self.assertAlmostEqual(stats.iqm([10, 10, 10, 10]), 10.0)

    def test_iqm_axis_matches_1d(self) -> None:
        rows = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        out = stats.iqm(rows, axis=1)
        self.assertTrue(np.allclose(out, [4.5, 4.5]))

    def test_iqm_small_n_is_mean(self) -> None:
        # Fewer than 4 values: nothing to trim -> plain mean (like trim_mean).
        self.assertAlmostEqual(stats.iqm(np.array([1.0, 2.0])), 1.5)
        self.assertAlmostEqual(stats.iqm(np.array([1.0, 2.0, 9.0])), 4.0)
        with self.assertRaises(ValueError):
            stats.iqm(np.array([]))

    def test_wilson_hand_values(self) -> None:
        ci = stats.wilson_ci(50, 100)
        self.assertAlmostEqual(ci.point, 0.5)
        self.assertAlmostEqual(ci.lo, 0.4038, places=3)
        self.assertAlmostEqual(ci.hi, 0.5962, places=3)

    def test_wilson_edges(self) -> None:
        self.assertEqual(stats.wilson_ci(0, 10).lo, 0.0)
        self.assertAlmostEqual(stats.wilson_ci(10, 10).hi, 1.0, places=9)
        self.assertGreater(stats.wilson_ci(0, 10).hi, 0.0)

    def test_wilson_validation(self) -> None:
        with self.assertRaises(ValueError):
            stats.wilson_ci(1, 0)
        with self.assertRaises(ValueError):
            stats.wilson_ci(11, 10)

    def test_bootstrap_determinism(self) -> None:
        x = np.array([1.0, 5.0, 3.0, 9.0, 2.0, 7.0])
        a = stats.bootstrap_ci(x, seed=42)
        b = stats.bootstrap_ci(x, seed=42)
        self.assertEqual(a.as_tuple(), b.as_tuple())
        # Different seed -> a different resample distribution (checked on the
        # full interval rather than a single percentile, which can collide).
        other = stats.bootstrap_ci(x, seed=1)
        self.assertNotEqual(other.as_tuple(), a.as_tuple())

    def test_bootstrap_point_is_mean(self) -> None:
        x = np.array([2.0, 4.0, 6.0, 8.0])
        ci = stats.bootstrap_ci(x, seed=0)
        self.assertAlmostEqual(ci.point, 5.0)
        self.assertLessEqual(ci.lo, ci.point)
        self.assertGreaterEqual(ci.hi, ci.point)

    def test_bootstrap_degenerate(self) -> None:
        ci = stats.bootstrap_ci(np.full(8, 3.0), seed=0)
        self.assertEqual((ci.lo, ci.point, ci.hi), (3.0, 3.0, 3.0))

    def test_bootstrap_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            stats.bootstrap_ci(np.array([]))

    def test_stratified_bootstrap_determinism_and_length_check(self) -> None:
        vals = np.array([10.0, 12.0, 40.0, 42.0, 70.0, 72.0])
        labels = np.array(["a", "a", "b", "b", "c", "c"])
        a = stats.stratified_bootstrap_ci(vals, labels, seed=3)
        b = stats.stratified_bootstrap_ci(vals, labels, seed=3)
        self.assertEqual(a.as_tuple(), b.as_tuple())
        self.assertAlmostEqual(a.point, float(np.mean(vals)))
        with self.assertRaises(ValueError):
            stats.stratified_bootstrap_ci(vals, labels[:-1])

    def test_paired_bootstrap_excludes_zero(self) -> None:
        # A strictly better than B by ~10 everywhere -> CI excludes 0, positive.
        diffs = np.array([9.0, 11.0, 10.0, 12.0, 8.0, 10.5, 9.5, 11.5])
        ci = stats.paired_bootstrap_ci(diffs, seed=0)
        self.assertTrue(ci.excludes(0.0))
        self.assertGreater(ci.lo, 0.0)

    def test_ci_excludes(self) -> None:
        ci = stats.ConfidenceInterval(point=5.0, lo=1.0, hi=9.0)
        self.assertFalse(ci.excludes(5.0))   # inside
        self.assertTrue(ci.excludes(0.0))    # below lo
        self.assertTrue(ci.excludes(10.0))   # above hi


class TestScoringParse(unittest.TestCase):
    """Parsing of the real ``scoring.yaml`` schema on synthetic fixtures."""

    def test_parse_dict_full(self) -> None:
        result = scoring.parse_scoring_dict(_fixture_scoring())
        self.assertAlmostEqual(result.total, 96.0)
        b = result.single_trial()
        self.assertEqual(b.trial_id, "trial_1")
        self.assertEqual(b.tier_1, 1.0)
        self.assertEqual(b.tier_3, 75.0)
        self.assertAlmostEqual(b.tier_2, 20.0)
        self.assertAlmostEqual(b.total, 96.0)
        self.assertEqual(b.duration_score, 9.0)
        self.assertEqual(b.smoothness_score, 5.0)
        self.assertEqual(b.efficiency_score, 6.0)
        self.assertTrue(b.inserted)

    def test_parse_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scoring.yaml"
            with p.open("w") as fh:
                yaml.safe_dump(_fixture_scoring(tier_3=42.0), fh)
            result = scoring.parse_scoring_file(p)
            self.assertEqual(result.path, p)
            self.assertEqual(result.single_trial().tier_3, 42.0)

    def test_missing_category_defaults_zero(self) -> None:
        doc = _fixture_scoring()
        del doc["trial_1"]["tier_2"]["categories"][scoring.CAT_CONTACTS]
        b = scoring.parse_scoring_dict(doc).single_trial()
        self.assertEqual(b.contacts_score, 0.0)

    def test_missing_total_raises(self) -> None:
        doc = _fixture_scoring()
        del doc["total"]
        with self.assertRaises(ValueError):
            scoring.parse_scoring_dict(doc)

    def test_malformed_tier_raises(self) -> None:
        doc = _fixture_scoring()
        doc["trial_1"]["tier_3"] = {"message": "no score"}
        with self.assertRaises(ValueError):
            scoring.parse_scoring_dict(doc)

    def test_single_trial_requires_one(self) -> None:
        doc = _fixture_scoring()
        doc["trial_2"] = doc["trial_1"]
        with self.assertRaises(ValueError):
            scoring.parse_scoring_dict(doc).single_trial()

    def test_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            scoring.parse_scoring_file("/no/such/scoring.yaml")


class TestOutcomeClassification(unittest.TestCase):
    """Outcome precedence and boundary behaviour."""

    def _classify(self, **kw) -> scoring.Outcome:
        return scoring.parse_scoring_dict(_fixture_scoring(**kw)).single_trial().outcome

    def test_full(self) -> None:
        self.assertEqual(self._classify(tier_3=75.0), scoring.Outcome.FULL)

    def test_full_wins_over_penalties(self) -> None:
        # A full insertion is still reported FULL even with force/contact penalty.
        self.assertEqual(
            self._classify(tier_3=75.0, force=-12.0, contacts=-24.0),
            scoring.Outcome.FULL,
        )

    def test_partial_range(self) -> None:
        self.assertEqual(self._classify(tier_3=38.0), scoring.Outcome.PARTIAL)
        self.assertEqual(self._classify(tier_3=50.0), scoring.Outcome.PARTIAL)
        self.assertEqual(self._classify(tier_3=44.0), scoring.Outcome.PARTIAL)

    def test_collision_over_proximity(self) -> None:
        self.assertEqual(
            self._classify(tier_3=10.0, contacts=-24.0), scoring.Outcome.COLLISION
        )

    def test_force_over_proximity(self) -> None:
        self.assertEqual(
            self._classify(tier_3=10.0, force=-12.0), scoring.Outcome.FORCE
        )

    def test_collision_precedes_force(self) -> None:
        self.assertEqual(
            self._classify(tier_3=0.0, force=-12.0, contacts=-24.0),
            scoring.Outcome.COLLISION,
        )

    def test_proximity(self) -> None:
        self.assertEqual(self._classify(tier_3=10.0), scoring.Outcome.PROXIMITY)
        self.assertEqual(self._classify(tier_3=37.9), scoring.Outcome.PROXIMITY)

    def test_miss_zero_and_wrong_port(self) -> None:
        self.assertEqual(self._classify(tier_3=0.0), scoring.Outcome.MISS)
        self.assertEqual(self._classify(tier_3=-12.0), scoring.Outcome.MISS)

    def test_flags(self) -> None:
        b = scoring.parse_scoring_dict(
            _fixture_scoring(tier_3=0.0, force=-12.0, contacts=-24.0)
        ).single_trial()
        self.assertTrue(b.has_force_penalty)
        self.assertTrue(b.has_collision_penalty)
        self.assertFalse(b.inserted)


class TestSuiteGeneration(unittest.TestCase):
    """Deterministic generation, strata coverage, and persisted artifacts."""

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_determinism(self) -> None:
        a, _ = suite.generate_suite(n=30, seed=99)
        b, _ = suite.generate_suite(n=30, seed=99)
        self.assertEqual(
            [yaml.safe_dump(c, sort_keys=False) for c, _ in a],
            [yaml.safe_dump(c, sort_keys=False) for c, _ in b],
        )
        self.assertEqual([m.config_id for _, m in a], [m.config_id for _, m in b])

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_seed_changes_output(self) -> None:
        a, _ = suite.generate_suite(n=10, seed=1)
        b, _ = suite.generate_suite(n=10, seed=2)
        self.assertNotEqual(a[0][1].board_x, b[0][1].board_x)

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_strata_coverage_and_official(self) -> None:
        members = [m for _, m in suite.generate_suite(n=50, seed=7)[0]]
        counts = suite.stratum_counts(members)
        self.assertEqual(len(suite.all_cells()), 20)
        # every one of the 20 cells is covered at least twice by 50 round-robin.
        stratified = [m for m in members if m.source == "stratified"]
        strat_counts = suite.stratum_counts(stratified)
        self.assertEqual(len(strat_counts), 20)
        self.assertGreaterEqual(min(strat_counts.values()), 2)
        official = [m for m in members if m.source == "official"]
        self.assertEqual(len(official), 3)
        self.assertTrue(all(m.config_id.startswith("official_") for m in official))
        self.assertEqual(len(counts), 20)

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_continuous_axes_in_range(self) -> None:
        for _, m in suite.generate_suite(n=60, seed=13)[0]:
            if m.source != "stratified":
                continue
            self.assertGreaterEqual(m.board_x, suite.BOARD_X_RANGE[0] - 1e-9)
            self.assertLessEqual(m.board_x, suite.BOARD_X_RANGE[1] + 1e-9)
            self.assertGreaterEqual(m.board_y, suite.BOARD_Y_RANGE[0] - 1e-9)
            self.assertLessEqual(m.board_y, suite.BOARD_Y_RANGE[1] + 1e-9)
            self.assertLessEqual(abs(m.board_yaw), math.pi + 1e-9)
            self.assertGreaterEqual(m.grasp_z, suite.GRASP_Z_RANGE[0] - 1e-9)
            self.assertLessEqual(m.grasp_z, suite.GRASP_Z_RANGE[1] + 1e-9)

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_configs_are_single_trial_and_complete(self) -> None:
        for cfg, m in suite.generate_suite(n=8, seed=5)[0]:
            self.assertEqual(list(cfg["trials"].keys()), ["trial_1"])
            self.assertIn("robot", cfg)
            self.assertIn("scoring", cfg)
            task = cfg["trials"]["trial_1"]["tasks"]["task_1"]
            for field in ("plug_type", "port_name", "target_module_name", "time_limit"):
                self.assertIn(field, task)
            # round-trips through YAML unchanged.
            self.assertEqual(yaml.safe_load(yaml.safe_dump(cfg)), cfg)

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_write_and_read_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = suite.write_suite(tmp, n=20, seed=3)
            self.assertTrue((Path(tmp) / "manifest.csv").is_file())
            self.assertTrue((Path(tmp) / "suite_meta.yaml").is_file())
            for m in written:
                self.assertTrue((Path(tmp) / m.config_file).is_file())
            read_back = suite.read_manifest(Path(tmp) / "manifest.csv")
            self.assertEqual(len(read_back), len(written))
            self.assertEqual(read_back[0].config_id, written[0].config_id)
            self.assertEqual(read_back[0].stratum, written[0].stratum)

    def test_assign_strata_validation(self) -> None:
        with self.assertRaises(ValueError):
            suite.assign_strata(0)
        self.assertEqual(len(suite.assign_strata(50)), 50)


class TestCompareVerdict(unittest.TestCase):
    """Paired compare-mode verdict logic."""

    def _rows(self, totals: list[float]) -> list[runner.TrialResult]:
        rows = []
        for i, t in enumerate(totals):
            rows.append(
                runner.TrialResult(
                    config_id=f"cfg_{i:03d}", source="stratified", rail=i % 5,
                    plug="SFP", port=i % 2, outcome="full", inserted=True,
                    insertion_success=True, total=t, tier_1=1.0, tier_2=20.0,
                    tier_3=t - 21.0, force_score=0.0, contacts_score=0.0,
                    duration_score=8.0, smoothness_score=6.0, efficiency_score=6.0,
                    duration_s=0.0, completed=True, timed_out=False,
                )
            )
        return rows

    def test_a_beats_b(self) -> None:
        a = self._rows([90, 92, 88, 91, 89, 93, 90, 91])
        b = self._rows([70, 72, 68, 71, 69, 73, 70, 71])
        cmp = report.compare(a, b, name_a="A", name_b="B")
        self.assertEqual(cmp.verdict, "A beats B")
        self.assertTrue(cmp.mean_diff.excludes(0.0))
        self.assertGreater(cmp.mean_diff.lo, 0.0)

    def test_b_beats_a(self) -> None:
        a = self._rows([70, 72, 68, 71, 69, 73, 70, 71])
        b = self._rows([90, 92, 88, 91, 89, 93, 90, 91])
        cmp = report.compare(a, b, name_a="A", name_b="B")
        self.assertEqual(cmp.verdict, "B beats A")

    def test_inconclusive(self) -> None:
        a = self._rows([80, 60, 90, 50, 85, 55])
        b = self._rows([60, 80, 50, 90, 55, 85])
        cmp = report.compare(a, b, name_a="A", name_b="B")
        self.assertEqual(cmp.verdict, "inconclusive")
        self.assertFalse(cmp.mean_diff.excludes(0.0))

    def test_mismatched_sets_raise(self) -> None:
        a = self._rows([80, 60, 90])
        b = self._rows([80, 60])
        with self.assertRaises(ValueError):
            report.compare(a, b)


class TestDryRunEndToEnd(unittest.TestCase):
    """Full suite -> dry-run -> report path without any sim."""

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_dry_run_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite.write_suite(root / "suite", n=20, seed=7)
            results = runner.run_suite(
                root / "suite", root / "res", policy="X.Y", dry_run=True
            )
            self.assertEqual(len(results), 23)  # 20 stratified + 3 official
            agg = report.write_report(root / "res", results, name="dry")
            self.assertEqual(agg.n, 23)
            self.assertTrue((root / "res" / report.REPORT_MD).is_file())
            self.assertTrue((root / "res" / report.RESULTS_CSV).is_file())
            self.assertTrue((root / "res" / report.SUMMARY_JSON).is_file())
            # per-trial scoring artifacts exist.
            self.assertTrue((root / "res" / "trials" / "cfg_000" / "scoring.yaml").is_file())
            # every outcome label is a valid enum value.
            valid = {o.value for o in scoring.Outcome}
            self.assertTrue(all(r.outcome in valid for r in results))

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_dry_run_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite.write_suite(root / "suite", n=20, seed=7)
            results = runner.run_suite(
                root / "suite", root / "res", policy="X.Y", dry_run=True, limit=5
            )
            self.assertEqual(len(results), 5)

    def test_dry_run_reproducible_scores(self) -> None:
        member = suite.SuiteMember(
            "cfg_000", "stratified", suite.Stratum(2, "SFP", 0),
            0.16, -0.1, 0.1, 0.042, "configs/cfg_000.yaml",
        )
        with tempfile.TemporaryDirectory() as tmp:
            r = runner.DryRunSimRunner()
            o1 = r.run_trial(member, Path("x"), "p", Path(tmp) / "a")
            o2 = r.run_trial(member, Path("x"), "p", Path(tmp) / "b")
            d1 = scoring.parse_scoring_file(o1.scoring_path).single_trial()
            d2 = scoring.parse_scoring_file(o2.scoring_path).single_trial()
            self.assertEqual(d1.total, d2.total)


class TestCliSmoke(unittest.TestCase):
    """The CLI parser wires subcommands correctly."""

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_cli_gen_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            rc = eval_suite.main(["gen", "--out", f"{tmp}/suite", "--n", "12"])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(tmp) / "suite" / "manifest.csv").is_file())
            rc = eval_suite.main(
                ["run", "--suite", f"{tmp}/suite", "--out", f"{tmp}/res",
                 "--dry-run", "--limit", "4"]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(tmp) / "res" / "report.md").is_file())


class TestPolicyLaunchWiring(unittest.TestCase):
    """Policy-interpreter override and checkpoint env-var delivery.

    These exercise pure string/env logic only (no ROS/Gazebo/GPU): the bringup
    script is built and inspected, and the checkpoint env export path is driven
    through the dry-run runner.
    """

    def test_default_launch_cmd(self) -> None:
        self.assertEqual(
            runner.SimEnv().policy_launch_cmd,
            "/home/kiwoos/venvs/aic-deploy/bin/python -u "
            "/home/kiwoos/ws_aic/install/lib/aic_model/aic_model",
        )

    def test_bringup_uses_override_interpreter(self) -> None:
        custom = "/home/kiwoos/venvs/aic-deploy/bin/python /ws/aic_model"
        sim = runner.SimRunner(env=runner.SimEnv(policy_launch_cmd=custom))
        script = sim._bringup_script(
            Path("/c/cfg.yaml"), "pkg.Cls", Path("/r"), Path("/r/run.log")
        )
        self.assertIn(f"{custom} --ros-args", script)
        self.assertNotIn("ros2 run aic_model aic_model --ros-args", script)
        self.assertIn("-p policy:=pkg.Cls", script)

    def test_bringup_default_uses_ros2_run(self) -> None:
        sim = runner.SimRunner()
        script = sim._bringup_script(
            Path("/c/cfg.yaml"), "pkg.Cls", Path("/r"), Path("/r/run.log")
        )
        self.assertIn("/home/kiwoos/venvs/aic-deploy/bin/python -u", script)

    def test_bringup_enables_nounset_after_sourcing(self) -> None:
        # `set -u` must come AFTER sourcing ROS/ws setup.bash: those scripts
        # reference unbound vars and would abort a nounset shell before launch.
        sim = runner.SimRunner()
        script = sim._bringup_script(
            Path("/c/cfg.yaml"), "pkg.Cls", Path("/r"), Path("/r/run.log")
        )
        src_idx = script.index(f"source {runner.SimEnv().ros_setup}")
        nounset_idx = script.index("\nset -u")
        self.assertLess(src_idx, nounset_idx)

    def test_bringup_cleanup_excludes_harness_ancestors(self) -> None:
        # cleanup() greps for "aic_model" to kill the policy node; the harness
        # (and any wrapper shell above it) also carries that path in argv via
        # --policy-cmd, so cleanup must protect the whole ancestor chain of the
        # bringup script or it would kill -9 the harness itself.
        sim = runner.SimRunner()
        script = sim._bringup_script(
            Path("/c/cfg.yaml"), "pkg.Cls", Path("/r"), Path("/r/run.log")
        )
        self.assertIn("ANC=$$", script)
        self.assertIn("/proc/$ANC/stat", script)
        self.assertIn('case "$KEEP" in', script)

    @unittest.skipUnless(EVAL_CONFIG.is_file(), "eval_config.yaml not present")
    def test_run_suite_exports_checkpoint_env(self) -> None:
        saved = {k: os.environ.get(k) for k in ("AIC_CKPT", "AIC_CHECKPOINT")}
        for k in saved:
            os.environ.pop(k, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                suite.write_suite(root / "suite", n=8, seed=7)
                runner.run_suite(
                    root / "suite", root / "res", policy="X.Y",
                    checkpoint="/tmp/fake_ckpt.pt", dry_run=True, limit=2,
                )
                self.assertEqual(os.environ["AIC_CKPT"], "/tmp/fake_ckpt.pt")
                self.assertEqual(os.environ["AIC_CHECKPOINT"], "/tmp/fake_ckpt.pt")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
