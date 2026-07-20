"""Unit tests for gen_dagger_ab5_manifest (CPU-only, no ROS/GPU).

Verifies the pure row-generation logic and -- crucially -- that the generated
manifest + per-rep symlink filenames are exactly what ``collect_dagger.sh``'s task
loader (``campaign_lib.load_tasks`` / ``parse_rep``) expects to consume.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import campaign_lib
import gen_dagger_ab5_manifest as gen


class TestRepFilename(unittest.TestCase):
    """The per-rep symlink filename round-trips through campaign_lib.parse_rep."""

    def test_filename_format(self) -> None:
        self.assertEqual(gen.rep_config_filename("official_1", 0), "official_1_r0.yaml")
        self.assertEqual(gen.rep_config_filename("cfg_005", 2), "cfg_005_r2.yaml")

    def test_parse_rep_round_trips(self) -> None:
        for base in ("official_1", "official_3", "cfg_001", "cfg_005"):
            for rep in (0, 1, 2, 7):
                name = gen.rep_config_filename(base, rep)
                self.assertEqual(campaign_lib.parse_rep(name), rep)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            gen.rep_config_filename("", 0)
        with self.assertRaises(ValueError):
            gen.rep_config_filename("official_1", -1)


class TestBuildRows(unittest.TestCase):
    """Row generation (config x rep ordering + plug/stratum wiring)."""

    def test_row_count_and_order(self) -> None:
        link_dir = pathlib.Path("/links")
        cfg_dir = pathlib.Path("/cfgs")
        rows = gen.build_rows(gen.AB5_CONFIGS, 3, link_dir, cfg_dir)
        self.assertEqual(len(rows), 5 * 3)
        # config-then-rep order: first three rows are official_1 r0..r2.
        self.assertEqual(
            [pathlib.Path(r.config).name for r in rows[:3]],
            ["official_1_r0.yaml", "official_1_r1.yaml", "official_1_r2.yaml"],
        )
        # official_3 is the SC plug; the rest are SFP.
        by_stratum = {r.stratum: r for r in rows}
        self.assertEqual(by_stratum["ab5_official_3"].plug, "sc")
        self.assertEqual(by_stratum["ab5_official_1"].plug, "sfp")
        self.assertEqual(by_stratum["ab5_cfg_005"].plug, "sfp")

    def test_link_targets_point_at_real_configs(self) -> None:
        rows = gen.build_rows(gen.AB5_CONFIGS, 1, pathlib.Path("/l"), pathlib.Path("/c"))
        self.assertEqual(rows[0].link_target, "/c/official_1.yaml")

    def test_reps_validation(self) -> None:
        with self.assertRaises(ValueError):
            gen.build_rows(gen.AB5_CONFIGS, 0, pathlib.Path("/l"), pathlib.Path("/c"))


class TestManifestConsumedByCampaignLib(unittest.TestCase):
    """End-to-end: write manifest + symlinks, load through campaign_lib.load_tasks."""

    def test_load_tasks_matches_generated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            # Fake real configs (bodies unused by campaign_lib; only need to exist
            # so create_symlinks' target check passes).
            for cfg in gen.AB5_CONFIGS:
                (cfg_dir / f"{cfg.base}.yaml").write_text("task_1: {}\n")
            link_dir = root / "configs_dagger_ab5"
            rows = gen.build_rows(gen.AB5_CONFIGS, 2, link_dir, cfg_dir)
            gen.create_symlinks(rows)
            manifest = link_dir / "manifest.csv"
            gen.write_manifest(rows, manifest)

            tasks = campaign_lib.load_tasks(str(manifest))
            self.assertEqual(len(tasks), len(rows))
            # Reps parse correctly and episode dirs are unique per (stratum, rep).
            ep_dirs = {t.episode_dir for t in tasks}
            self.assertEqual(len(ep_dirs), len(rows))
            first = tasks[0]
            self.assertEqual(first.stratum, "ab5_official_1")
            self.assertEqual(first.rep, 0)
            self.assertEqual(first.episode_dir, "ep_ab5_official_1_r0")
            # The symlink the manifest points at resolves to the real config.
            self.assertTrue(pathlib.Path(first.config).is_symlink())
            self.assertTrue(pathlib.Path(first.config).resolve().exists())

            # The --plug filter selects only the SC config's rows.
            sc_tasks = campaign_lib.load_tasks(str(manifest), plug="sc")
            self.assertEqual({t.stratum for t in sc_tasks}, {"ab5_official_3"})
            self.assertEqual(len(sc_tasks), 2)

    def test_create_symlinks_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            rows = gen.build_rows(
                gen.AB5_CONFIGS, 1, root / "links", root / "missing"
            )
            with self.assertRaises(FileNotFoundError):
                gen.create_symlinks(rows)


if __name__ == "__main__":
    unittest.main()
