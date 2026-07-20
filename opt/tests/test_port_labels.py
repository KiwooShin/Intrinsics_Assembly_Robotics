"""Unit tests for opt.port_labels (hindsight port-offset labels, CPU-only)."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from opt.tests import _pathfix  # noqa: F401  (path side effect)
from opt import port_labels


_LOG_HEADER = (
    "timestamp,config,stratum,plug,rep,episode_dir,score,frames,"
    "insertion_events,wall_clock_s,status\n"
)


def _write_log(root: pathlib.Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Write a minimal campaign_log.csv.

    Args:
        root: Dataset root the log is written into.
        rows: ``(episode_dir, score, insertion_events, status)`` tuples.
    """
    lines = [_LOG_HEADER]
    for ep, score, ins, status in rows:
        lines.append(
            f"20260719_000000,/cfg/{ep}.yaml,strat,sfp,0,{ep},{score},100,"
            f"{ins},400,{status}\n"
        )
    (root / port_labels.CAMPAIGN_LOG_NAME).write_text("".join(lines))


def _make_episode(
    root: pathlib.Path, name: str, poses: np.ndarray
) -> str:
    """Create an episode dir with a ``tcp_poses.npy`` and return its path."""
    ep = root / name
    ep.mkdir(parents=True, exist_ok=True)
    np.save(ep / "tcp_poses.npy", poses)
    return str(ep)


def _descend_poses(n: int, target: np.ndarray) -> np.ndarray:
    """Return identity-orientation poses that descend to ``target`` and seat."""
    poses = np.zeros((n, 7))
    poses[:, 6] = 1.0
    # Approach linearly from an offset start, then hold at target for the tail.
    n_move = n - 5
    start = target + np.array([0.0, 0.0, 0.20])
    poses[:n_move, :3] = np.linspace(start, target, n_move)
    poses[n_move:, :3] = target
    return poses


class CampaignLogTest(unittest.TestCase):
    """Parsing and the KEEP + insertion validity rule."""

    def test_parse_and_validity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_log(
                root,
                [
                    ("ep_a", "92.0", "1", "KEEP"),
                    ("ep_b", "19.0", "", "DROP_SCORE"),
                    ("ep_c", "88.0", "0", "KEEP"),  # KEEP but no insertion
                ],
            )
            status = port_labels.parse_campaign_log(root / port_labels.CAMPAIGN_LOG_NAME)
            self.assertTrue(status["ep_a"].valid)
            self.assertFalse(status["ep_b"].valid)  # dropped
            self.assertFalse(status["ep_c"].valid)  # KEEP but 0 insertions
            self.assertIsNone(status["ep_b"].insertion_events)
            self.assertEqual(status["ep_a"].insertion_events, 1)

    def test_missing_log_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                port_labels.parse_campaign_log(pathlib.Path(d) / "nope.csv")

    def test_skip_exists_rows_ignored(self) -> None:
        """A KEEP followed by resume SKIP_EXISTS rows stays valid (effective KEEP)."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_log(
                root,
                [
                    ("ep_a", "92.0", "1", "KEEP"),
                    ("ep_a", "", "", "SKIP_EXISTS"),
                    ("ep_a", "", "", "SKIP_EXISTS"),
                ],
            )
            status = port_labels.parse_campaign_log(root / port_labels.CAMPAIGN_LOG_NAME)
            self.assertTrue(status["ep_a"].valid)  # skips do not override the KEEP
            self.assertEqual(status["ep_a"].status, "KEEP")

    def test_recollection_uses_latest_verdict(self) -> None:
        """A DROP re-collected into a KEEP takes the later KEEP verdict."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_log(
                root,
                [
                    ("ep_a", "19.0", "", "DROP_SCORE"),
                    ("ep_a", "94.0", "1", "KEEP"),
                ],
            )
            status = port_labels.parse_campaign_log(root / port_labels.CAMPAIGN_LOG_NAME)
            self.assertTrue(status["ep_a"].valid)


class BuildLabelsTest(unittest.TestCase):
    """Per-frame labels, validity mask, ordering, and normalization."""

    def test_offset_zero_at_terminal_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            tgt_a = np.array([0.1, -0.2, 0.15])
            tgt_b = np.array([0.3, 0.1, 0.12])
            ep_a = _make_episode(root, "ep_a", _descend_poses(30, tgt_a))
            ep_b = _make_episode(root, "ep_b", _descend_poses(20, tgt_b))
            _write_log(
                root,
                [("ep_a", "92.0", "1", "KEEP"), ("ep_b", "19.0", "", "DROP_SCORE")],
            )
            ls = port_labels.build_labels([ep_a, ep_b], aux_dim=3, aux_frame="tcp")
            # Concatenated in episode order; frame counts recorded.
            self.assertEqual(ls.frame_counts, (30, 20))
            self.assertEqual(ls.offsets.shape, (50, 3))
            # ep_a valid, ep_b invalid (dropped).
            self.assertTrue(ls.valid[:30].all())
            self.assertFalse(ls.valid[30:].any())
            # Offset ~0 at each episode's seated terminal frame.
            np.testing.assert_allclose(ls.offsets[29], [0.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(ls.offsets[49], [0.0, 0.0, 0.0], atol=1e-6)
            # A mid-approach frame has a plausible several-cm magnitude.
            self.assertGreater(float(np.linalg.norm(ls.offsets[0])), 0.1)

    def test_label_before_trim_ordering(self) -> None:
        """Trimming the seated tail must not change the surviving frames' labels."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            tgt = np.array([0.2, 0.0, 0.1])
            ep = _make_episode(root, "ep_a", _descend_poses(40, tgt))
            _write_log(root, [("ep_a", "92.0", "1", "KEEP")])
            ls = port_labels.build_labels([ep], aux_dim=3, aux_frame="tcp")
            # Simulate a downstream tail-trim that drops the last 5 seated frames:
            # the labels for kept frames are identical (target is read pre-trim).
            trimmed = ls.offsets[:35]
            np.testing.assert_allclose(trimmed, ls.offsets[:35])
            # And the target is the median tail, unaffected by how many tail
            # frames a later stage keeps.
            self.assertEqual(len(ls.episodes), 1)
            np.testing.assert_allclose(
                ls.episodes[0].target_position, tgt, atol=1e-9
            )

    def test_six_dim_has_axis_block(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            tgt = np.array([0.0, 0.0, 0.1])
            ep = _make_episode(root, "ep_a", _descend_poses(25, tgt))
            _write_log(root, [("ep_a", "92.0", "1", "KEEP")])
            ls = port_labels.build_labels([ep], aux_dim=6, aux_frame="base")
            self.assertEqual(ls.offsets.shape, (25, 6))
            # Approach was pure -Z; the base-frame axis block is ~[0,0,-1].
            np.testing.assert_allclose(ls.offsets[0, 3:], [0.0, 0.0, -1.0], atol=1e-6)

    def test_missing_row_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            tgt = np.array([0.1, 0.1, 0.1])
            ep = _make_episode(root, "ep_orphan", _descend_poses(15, tgt))
            _write_log(root, [("ep_other", "92.0", "1", "KEEP")])
            ls = port_labels.build_labels([ep], aux_dim=3, aux_frame="tcp")
            self.assertFalse(ls.valid.any())  # no matching campaign row

    def test_normalization_stats_over_valid_only(self) -> None:
        offsets = np.array(
            [[1.0, 2.0, 3.0], [100.0, 100.0, 100.0], [3.0, 4.0, 5.0]],
            dtype=np.float32,
        )
        valid = np.array([True, False, True])
        omean, ostd = port_labels.normalization_stats(offsets, valid)
        # Mean over valid rows only (the huge invalid row is excluded).
        np.testing.assert_allclose(omean, [2.0, 3.0, 4.0], atol=1e-5)
        self.assertTrue((ostd > 0).all())

    def test_normalization_requires_valid(self) -> None:
        with self.assertRaises(ValueError):
            port_labels.normalization_stats(
                np.zeros((3, 3), dtype=np.float32), np.zeros(3, dtype=bool)
            )

    def test_bad_aux_dim_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ep = _make_episode(root, "ep_a", _descend_poses(10, np.zeros(3)))
            _write_log(root, [("ep_a", "92.0", "1", "KEEP")])
            status = port_labels.load_status_map([ep])
            with self.assertRaises(ValueError):
                port_labels.build_episode_label(ep, status, aux_dim=4)


class AutoDeriveLogTest(unittest.TestCase):
    """Auto-deriving campaign_log.csv per episode across multiple datasets."""

    def test_merge_two_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            ds0, ds2 = root / "ds0", root / "ds2"
            ds0.mkdir()
            ds2.mkdir()
            ep0 = _make_episode(ds0, "ep_a", _descend_poses(12, np.array([0.1, 0, 0.1])))
            ep2 = _make_episode(ds2, "ep_b", _descend_poses(12, np.array([0.2, 0, 0.1])))
            _write_log(ds0, [("ep_a", "92.0", "1", "KEEP")])
            _write_log(ds2, [("ep_b", "80.0", "2", "KEEP")])
            ls = port_labels.build_labels([ep0, ep2], aux_dim=3, aux_frame="tcp")
            self.assertTrue(ls.valid.all())  # both KEEP + inserted, merged logs


class PortTargetOverrideTest(unittest.TestCase):
    """The privileged-DAgger port_target.npy override (dagger_relabel path)."""

    def test_load_port_target_absent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ep = _make_episode(
                pathlib.Path(d), "ep_x", _descend_poses(10, np.array([0.1, 0, 0.1]))
            )
            self.assertIsNone(port_labels.load_port_target(ep))

    def test_load_port_target_bad_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ep = _make_episode(
                pathlib.Path(d), "ep_x", _descend_poses(10, np.array([0.1, 0, 0.1]))
            )
            np.save(pathlib.Path(ep) / port_labels.PORT_TARGET_NAME, np.zeros(2))
            with self.assertRaises(ValueError):
                port_labels.load_port_target(ep)

    def test_override_replaces_target_and_forces_valid(self) -> None:
        # A non-seating stall: TCP never reaches the port, but port_target.npy
        # supplies the true target and the episode is valid despite no insertion.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            # Poses stall 5 cm short of the port on x.
            poses = np.zeros((8, 7))
            poses[:, 6] = 1.0
            poses[:, 0] = np.linspace(0.0, 0.15, 8)  # ends at x=0.15
            ep = _make_episode(root, "ep_dagger", poses)
            port = np.array([0.20, 0.0, 0.0])  # true port 5 cm beyond the stall
            np.save(pathlib.Path(ep) / port_labels.PORT_TARGET_NAME, port)
            # A DROP_SCORE / no-insertion log row would normally be invalid...
            _write_log(root, [("ep_dagger", "", "", "DROP_SCORE")])
            ls = port_labels.build_labels([ep], aux_dim=3, aux_frame="tcp")
            self.assertTrue(ls.valid.all())  # override forces validity
            # Target is the stored port, NOT robust_terminal (which would be 0.15).
            np.testing.assert_allclose(ls.episodes[0].target_position, port, atol=1e-9)
            # Terminal-frame label = true-port offset in TCP frame (identity quat).
            np.testing.assert_allclose(ls.offsets[-1], [0.05, 0.0, 0.0], atol=1e-6)

    def test_hindsight_path_unchanged_without_override(self) -> None:
        # Regression guard: absent port_target.npy -> byte-identical legacy target.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            poses = _descend_poses(12, np.array([0.1, 0.0, 0.1]))
            ep = _make_episode(root, "ep_h", poses)
            _write_log(root, [("ep_h", "92.0", "1", "KEEP")])
            ls = port_labels.build_labels([ep], aux_dim=3, aux_frame="tcp")
            tgt, _ = port_labels.port_offset.robust_terminal(poses)
            np.testing.assert_allclose(ls.episodes[0].target_position, tgt, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
