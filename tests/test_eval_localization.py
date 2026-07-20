"""Unit tests for eval_localization pure logic (CPU-only, no torch/GPU).

Covers the four pure seams the eval decision metric rests on: the mm error
aggregation (mean/median/p90 lateral & 3-D), the near-port masking, the
deploy-target resolution / true-offset recomputation (cross-checked against the
shared ``port_offset`` geometry), and the episode glob/split selection. The
torch-dependent checkpoint/inference path is exercised only against a real
checkpoint on GPU (out of scope here) and is not imported by these tests.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

import eval_localization as el
from aic_example_policies.ros import port_offset


def _yaw_quat(theta: float) -> np.ndarray:
    """Return the ``[x, y, z, w]`` quaternion of a rotation ``theta`` about +z."""
    return np.array([0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)])


class TestOffsetErrorStats(unittest.TestCase):
    """The mm aggregation math (predicted vs. true -> mean/median/p90)."""

    def _known_errors(self) -> tuple[np.ndarray, np.ndarray]:
        # pred - true = these error vectors (meters):
        #   [3,4,0]mm -> 3D 5mm, lat 5mm
        #   [0,0,12]mm -> 3D 12mm, lat 0mm
        #   [6,8,0]mm -> 3D 10mm, lat 10mm
        err = np.array(
            [[0.003, 0.004, 0.0], [0.0, 0.0, 0.012], [0.006, 0.008, 0.0]]
        )
        return err, np.zeros_like(err)

    def test_full_3d_and_lateral_stats(self) -> None:
        pred, true = self._known_errors()
        s = el.offset_error_stats(pred, true)
        self.assertEqual(s.n_frames, 3)
        # 3D distances sorted [5, 10, 12] mm.
        self.assertAlmostEqual(s.mean_3d_mm, 9.0, places=4)
        self.assertAlmostEqual(s.median_3d_mm, 10.0, places=4)
        self.assertAlmostEqual(s.p90_3d_mm, 11.6, places=4)  # linear interp
        # Lateral (xy) distances sorted [0, 5, 10] mm.
        self.assertAlmostEqual(s.mean_lat_mm, 5.0, places=4)
        self.assertAlmostEqual(s.median_lat_mm, 5.0, places=4)
        self.assertAlmostEqual(s.p90_lat_mm, 9.0, places=4)

    def test_mask_selects_subset(self) -> None:
        pred, true = self._known_errors()
        mask = np.array([True, False, True])
        s = el.offset_error_stats(pred, true, mask)
        self.assertEqual(s.n_frames, 2)
        # 3D [5, 10] -> mean 7.5, median 7.5, p90 9.5.
        self.assertAlmostEqual(s.mean_3d_mm, 7.5, places=4)
        self.assertAlmostEqual(s.median_3d_mm, 7.5, places=4)
        self.assertAlmostEqual(s.p90_3d_mm, 9.5, places=4)

    def test_error_is_frame_invariant_in_3d(self) -> None:
        # Rotating BOTH pred and true by the same per-frame quaternion leaves the
        # 3-D error unchanged (what deployment sees in base_link).
        rng = np.random.default_rng(0)
        pred = rng.normal(scale=0.02, size=(20, 3))
        true = rng.normal(scale=0.02, size=(20, 3))
        quats = np.tile(_yaw_quat(0.7), (20, 1))
        pred_b = port_offset.rotate_vectors_by_quats(quats, pred)
        true_b = port_offset.rotate_vectors_by_quats(quats, true)
        self.assertAlmostEqual(
            el.offset_error_stats(pred, true).median_3d_mm,
            el.offset_error_stats(pred_b, true_b).median_3d_mm,
            places=6,
        )

    def test_zero_error(self) -> None:
        a = np.array([[0.01, -0.02, 0.03], [0.0, 0.0, 0.0]])
        s = el.offset_error_stats(a, a)
        self.assertEqual(s.n_frames, 2)
        for v in (s.mean_3d_mm, s.median_3d_mm, s.p90_3d_mm, s.mean_lat_mm):
            self.assertAlmostEqual(v, 0.0, places=9)


class TestOffsetErrorStatsValidation(unittest.TestCase):
    """Input validation at the aggregation boundary."""

    def test_wrong_predicted_shape(self) -> None:
        with self.assertRaises(ValueError):
            el.offset_error_stats(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            el.offset_error_stats(np.zeros((4, 3)), np.zeros((5, 3)))

    def test_empty(self) -> None:
        with self.assertRaises(ValueError):
            el.offset_error_stats(np.zeros((0, 3)), np.zeros((0, 3)))

    def test_non_finite(self) -> None:
        bad = np.array([[np.nan, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            el.offset_error_stats(bad, np.zeros((1, 3)))

    def test_wrong_mask_shape(self) -> None:
        with self.assertRaises(ValueError):
            el.offset_error_stats(
                np.zeros((3, 3)), np.zeros((3, 3)), np.array([True, False])
            )

    def test_mask_selects_nothing(self) -> None:
        with self.assertRaises(ValueError):
            el.offset_error_stats(
                np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3, dtype=bool)
            )


class TestNearPortMask(unittest.TestCase):
    """The near-port operating-point subset selector."""

    def test_boundary_inclusive(self) -> None:
        true = np.array([[0.02, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.03, 0.0]])
        mask = el.near_port_mask(true, 0.03)
        np.testing.assert_array_equal(mask, np.array([True, False, True]))

    def test_bad_radius(self) -> None:
        with self.assertRaises(ValueError):
            el.near_port_mask(np.zeros((2, 3)), 0.0)

    def test_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            el.near_port_mask(np.zeros((2, 4)), 0.03)


class TestDeployTargetResolution(unittest.TestCase):
    """The deploy conversion + true-offset recomputation vs. shared geometry."""

    def _poses(self) -> np.ndarray:
        # 4 frames, varying position, non-trivial yaw orientations.
        pos = np.array(
            [[-0.30, 0.10, 0.20], [-0.34, 0.11, 0.19],
             [-0.37, 0.115, 0.185], [-0.39, 0.117, 0.181]]
        )
        quats = np.stack([_yaw_quat(t) for t in (0.0, 0.3, 0.6, 0.9)])
        return np.concatenate([pos, quats], axis=1)

    def test_tcp_offset_resolves_to_true_target(self) -> None:
        poses = self._poses()
        port = np.array([-0.401, 0.116, 0.179])
        true_off_tcp = port_offset.per_frame_tcp_offsets(poses, port, frame="tcp")
        off_base, tgt_base = el.resolve_deploy_targets(poses, true_off_tcp, "tcp")
        # Every frame's resolved target is the (static) true port position.
        np.testing.assert_allclose(tgt_base, np.tile(port, (4, 1)), atol=1e-9)
        np.testing.assert_allclose(off_base, port[None, :] - poses[:, :3], atol=1e-9)

    def test_base_frame_passthrough(self) -> None:
        poses = self._poses()
        port = np.array([-0.401, 0.116, 0.179])
        off_base_in = port[None, :] - poses[:, :3]
        off_base, tgt_base = el.resolve_deploy_targets(poses, off_base_in, "base")
        np.testing.assert_allclose(off_base, off_base_in, atol=1e-12)
        np.testing.assert_allclose(tgt_base, np.tile(port, (4, 1)), atol=1e-9)

    def test_true_offsets_from_target_matches_port_offset(self) -> None:
        poses = self._poses()
        port = np.array([-0.401, 0.116, 0.179])
        got = el.true_offsets_from_target(poses, port, "tcp")
        for i in range(poses.shape[0]):
            expect = port_offset.tcp_frame_offset(poses[i, :3], poses[i, 3:7], port)
            np.testing.assert_allclose(got[i], expect, atol=1e-9)

    def test_resolve_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            el.resolve_deploy_targets(np.zeros((3, 6)), np.zeros((3, 3)), "tcp")
        with self.assertRaises(ValueError):
            el.resolve_deploy_targets(np.zeros((3, 7)), np.zeros((2, 3)), "tcp")
        with self.assertRaises(ValueError):
            el.resolve_deploy_targets(np.zeros((3, 7)), np.zeros((3, 3)), "world")


class TestEpisodeSelection(unittest.TestCase):
    """Glob expansion + held-out split logic."""

    def test_expand_globs_sorted_and_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for name in ("ep_c", "ep_a", "ep_b"):
                (root / name).mkdir()
            got = el.expand_episode_globs(f"{root}/ep_*")
            self.assertEqual(
                [pathlib.Path(p).name for p in got], ["ep_a", "ep_b", "ep_c"]
            )
            # Explicit path first, then the glob; the duplicate is dropped.
            got2 = el.expand_episode_globs(f"{root}/ep_a,{root}/ep_*")
            names2 = [pathlib.Path(p).name for p in got2]
            self.assertEqual(names2, ["ep_a", "ep_b", "ep_c"])

    def test_held_out_split_tail(self) -> None:
        dirs = [f"/x/ep_{i:02d}" for i in range(10)]
        # ceil(0.2 * 10) = 2 -> last two, sorted.
        self.assertEqual(el.held_out_split(dirs, 0.2), ["/x/ep_08", "/x/ep_09"])
        # ceil(0.15 * 10) = 2.
        self.assertEqual(len(el.held_out_split(dirs, 0.15)), 2)
        # Tiny fraction still yields at least one.
        self.assertEqual(el.held_out_split(dirs, 0.001), ["/x/ep_09"])
        # Whole set.
        self.assertEqual(len(el.held_out_split(dirs, 1.0)), 10)

    def test_held_out_split_sorts_input(self) -> None:
        dirs = ["/x/ep_02", "/x/ep_00", "/x/ep_01"]
        # ceil(0.3 * 3) = 1 -> the single largest after sorting.
        self.assertEqual(el.held_out_split(dirs, 0.3), ["/x/ep_02"])

    def test_held_out_split_validation(self) -> None:
        with self.assertRaises(ValueError):
            el.held_out_split([], 0.2)
        with self.assertRaises(ValueError):
            el.held_out_split(["/x/ep_0"], 0.0)
        with self.assertRaises(ValueError):
            el.held_out_split(["/x/ep_0"], 1.5)


if __name__ == "__main__":
    unittest.main()
