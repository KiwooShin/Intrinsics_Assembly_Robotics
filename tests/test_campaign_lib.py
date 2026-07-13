"""Unit tests for campaign_lib.py (Phase-0 campaign manifest/naming helpers).

Runs without ROS/Gazebo/GPU (pure CSV + string logic).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import campaign_lib

_MANIFEST_HEADER = "config,stratum,plug,rep\n"


def _write_manifest(rows: list[tuple[str, str, str]]) -> str:
    """Write a minimal manifest CSV (config,stratum,plug) and return its path."""
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    fh.write("config,stratum,plug\n")
    for config, stratum, plug in rows:
        fh.write(f"{config},{stratum},{plug}\n")
    fh.close()
    return fh.name


class TestParseRep(unittest.TestCase):
    def test_parses_trailing_rep(self) -> None:
        self.assertEqual(
            campaign_lib.parse_rep("/d/p0_sfp_rail0_sfp_port_0_r3.yaml"), 3
        )
        self.assertEqual(campaign_lib.parse_rep("p0_sc_rail1_r0.yaml"), 0)
        self.assertEqual(campaign_lib.parse_rep("p0_sfp_rail4_sfp_port_1_r11.yaml"), 11)

    def test_missing_rep_raises(self) -> None:
        with self.assertRaises(ValueError):
            campaign_lib.parse_rep("/d/p0_sfp_rail0_sfp_port_0.yaml")


class TestEpisodeDirName(unittest.TestCase):
    def test_format(self) -> None:
        self.assertEqual(
            campaign_lib.episode_dir_name("sfp_rail2_sfp_port_0", 0),
            "ep_sfp_rail2_sfp_port_0_r0",
        )
        self.assertEqual(
            campaign_lib.episode_dir_name("sc_rail1", 3), "ep_sc_rail1_r3"
        )

    def test_unique_per_stratum_rep(self) -> None:
        names = {
            campaign_lib.episode_dir_name(s, r)
            for s in ("sfp_rail0_sfp_port_0", "sc_rail0")
            for r in range(4)
        }
        self.assertEqual(len(names), 8)

    def test_empty_stratum_raises(self) -> None:
        with self.assertRaises(ValueError):
            campaign_lib.episode_dir_name("", 0)

    def test_negative_rep_raises(self) -> None:
        with self.assertRaises(ValueError):
            campaign_lib.episode_dir_name("sc_rail0", -1)


class TestLoadTasks(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _write_manifest(
            [
                ("/d/p0_sfp_rail0_sfp_port_0_r0.yaml", "sfp_rail0_sfp_port_0", "sfp"),
                ("/d/p0_sc_rail0_r0.yaml", "sc_rail0", "sc"),
                ("/d/p0_sfp_rail1_sfp_port_1_r2.yaml", "sfp_rail1_sfp_port_1", "sfp"),
            ]
        )

    def tearDown(self) -> None:
        Path(self.path).unlink(missing_ok=True)

    def test_order_and_fields(self) -> None:
        tasks = campaign_lib.load_tasks(self.path)
        self.assertEqual([t.stratum for t in tasks],
                         ["sfp_rail0_sfp_port_0", "sc_rail0", "sfp_rail1_sfp_port_1"])
        self.assertEqual(tasks[0].episode_dir, "ep_sfp_rail0_sfp_port_0_r0")
        self.assertEqual(tasks[1].episode_dir, "ep_sc_rail0_r0")
        self.assertEqual(tasks[2].rep, 2)
        self.assertEqual(tasks[2].episode_dir, "ep_sfp_rail1_sfp_port_1_r2")

    def test_plug_filter_sfp(self) -> None:
        tasks = campaign_lib.load_tasks(self.path, plug="sfp")
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(t.plug == "sfp" for t in tasks))

    def test_plug_filter_sc(self) -> None:
        tasks = campaign_lib.load_tasks(self.path, plug="sc")
        self.assertEqual([t.stratum for t in tasks], ["sc_rail0"])

    def test_bad_plug_raises(self) -> None:
        with self.assertRaises(ValueError):
            campaign_lib.load_tasks(self.path, plug="lc")

    def test_missing_manifest_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            campaign_lib.load_tasks("/no/such/manifest.csv")


class TestRealManifest(unittest.TestCase):
    """If the real Phase-0 manifest exists, it parses to 48 tasks (40 SFP + 8 SC)."""

    MANIFEST = Path("/home/kiwoos/data/configs_phase0/manifest.csv")

    def setUp(self) -> None:
        if not self.MANIFEST.exists():
            self.skipTest(f"real manifest not present at {self.MANIFEST}")

    def test_counts_and_unique_episode_dirs(self) -> None:
        tasks = campaign_lib.load_tasks(str(self.MANIFEST))
        self.assertEqual(len(tasks), 48)
        self.assertEqual(sum(t.plug == "sfp" for t in tasks), 40)
        self.assertEqual(sum(t.plug == "sc" for t in tasks), 8)
        self.assertEqual(len({t.episode_dir for t in tasks}), 48)
        for t in tasks:
            self.assertTrue(Path(t.config).exists(), f"missing config {t.config}")


if __name__ == "__main__":
    unittest.main()
