"""Unit tests for dagger_relabel (privileged-DAgger port relabeling, CPU-only).

All tests use small in-memory synthetic arrays / TF forests and run without ROS,
Gazebo, or a GPU. They cover the four pure seams the task calls out: the
port-offset-in-TCP-frame computation (cross-checked against
``port_offset.tcp_frame_offset``), the TF-forest resolution, the stall-window
selection, and the emitted episode + label-file schema (including that the
extended ``opt.port_labels`` consumes it identically to what ``train_v3`` feeds
the aux head).
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

import dagger_relabel as dr
from aic_example_policies.ros import port_offset
from opt import port_labels


def _yaw_quat(theta: float) -> np.ndarray:
    """Return the ``[x, y, z, w]`` quaternion of a rotation ``theta`` about +z."""
    return np.array([0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)])


class TestQuaternionAndTransformMath(unittest.TestCase):
    def test_rotation_matrix_matches_port_offset(self) -> None:
        # quat_to_rotation_matrix must agree with the deploy/train rotation math.
        for theta in (0.0, 0.3, -1.2, np.pi / 2):
            q = _yaw_quat(theta)
            rot = dr.quat_to_rotation_matrix(q)
            for v in (np.array([1.0, 0.0, 0.0]), np.array([0.3, -0.7, 1.1])):
                np.testing.assert_allclose(
                    rot @ v, port_offset.rotate_vector_by_quat(q, v), atol=1e-12
                )

    def test_identity_quat_is_identity_matrix(self) -> None:
        np.testing.assert_allclose(
            dr.quat_to_rotation_matrix([0, 0, 0, 1]), np.eye(3), atol=1e-12
        )

    def test_zero_norm_quat_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.quat_to_rotation_matrix([0, 0, 0, 0])

    def test_homogeneous_invert_round_trip(self) -> None:
        mat = dr.homogeneous_transform([1.0, -2.0, 0.5], _yaw_quat(0.9))
        np.testing.assert_allclose(
            dr.invert_homogeneous(mat) @ mat, np.eye(4), atol=1e-12
        )

    def test_invert_bad_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.invert_homogeneous(np.eye(3))


class TestTransformForest(unittest.TestCase):
    def _forest(self, base_t: np.ndarray, base_q: np.ndarray) -> dr.TransformForest:
        # world -> base_link and world -> board -> port_entrance.
        f = dr.TransformForest()
        f.add_transform("world", "base_link", base_t, base_q)
        f.add_transform("world", "board", [0.4, 0.1, 1.14], [0, 0, 0, 1])
        f.add_transform("board", "port_link_entrance", [0.05, -0.02, 0.0], [0, 0, 0, 1])
        return f

    def test_identity_base_position(self) -> None:
        # base_link == world -> port position in base == world port position.
        f = self._forest(np.zeros(3), np.array([0, 0, 0, 1.0]))
        pos = f.position_of("base_link", "port_link_entrance")
        np.testing.assert_allclose(pos, [0.45, 0.08, 1.14], atol=1e-12)

    def test_translated_base_position(self) -> None:
        f = self._forest(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0, 1.0]))
        pos = f.position_of("base_link", "port_link_entrance")
        np.testing.assert_allclose(pos, [0.35, -0.12, 0.84], atol=1e-12)

    def test_yawed_base_position(self) -> None:
        # Cross-check the resolver against an explicit inverse-rotation formula.
        base_t = np.array([0.1, 0.2, 0.3])
        base_q = _yaw_quat(0.7)
        f = self._forest(base_t, base_q)
        pos = f.position_of("base_link", "port_link_entrance")
        world_port = np.array([0.45, 0.08, 1.14])
        expected = port_offset.rotate_vector_by_quat_inverse(base_q, world_port - base_t)
        np.testing.assert_allclose(pos, expected, atol=1e-12)

    def test_resolve_round_trip_inverse(self) -> None:
        f = self._forest(np.array([0.1, 0.2, 0.3]), _yaw_quat(0.4))
        a = f.resolve("base_link", "port_link_entrance")
        b = f.resolve("port_link_entrance", "base_link")
        np.testing.assert_allclose(a @ b, np.eye(4), atol=1e-10)

    def test_unknown_frame_raises(self) -> None:
        f = self._forest(np.zeros(3), np.array([0, 0, 0, 1.0]))
        with self.assertRaises(ValueError):
            f.resolve("base_link", "no_such_frame")

    def test_disconnected_trees_raise(self) -> None:
        f = dr.TransformForest()
        f.add_transform("world_a", "base_link", np.zeros(3), [0, 0, 0, 1])
        f.add_transform("world_b", "port_link_entrance", np.zeros(3), [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            f.resolve("base_link", "port_link_entrance")

    def test_cycle_detected(self) -> None:
        f = dr.TransformForest()
        f.add_transform("a", "b", np.zeros(3), [0, 0, 0, 1])
        f.add_transform("b", "a", np.zeros(3), [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            f.resolve("a", "b")

    def test_last_transform_wins(self) -> None:
        f = dr.TransformForest()
        f.add_transform("world", "base_link", [9, 9, 9], [0, 0, 0, 1])
        f.add_transform("world", "base_link", [0, 0, 0], [0, 0, 0, 1])
        f.add_transform("world", "port_link_entrance", [1, 2, 3], [0, 0, 0, 1])
        np.testing.assert_allclose(
            f.position_of("base_link", "port_link_entrance"), [1, 2, 3], atol=1e-12
        )


class TestSelectEntranceFrame(unittest.TestCase):
    def test_unique_entrance(self) -> None:
        frames = {"world", "base_link", "board/mod/sfp_port_0_link_entrance"}
        self.assertEqual(
            dr.select_entrance_frame(frames), "board/mod/sfp_port_0_link_entrance"
        )

    def test_port_name_disambiguates(self) -> None:
        frames = {
            "board/mod0/sfp_port_0_link_entrance",
            "board/mod1/sfp_port_1_link_entrance",
        }
        self.assertEqual(
            dr.select_entrance_frame(frames, port_name="sfp_port_1"),
            "board/mod1/sfp_port_1_link_entrance",
        )

    def test_explicit_overrides(self) -> None:
        frames = {"a_link_entrance", "b_link_entrance"}
        self.assertEqual(
            dr.select_entrance_frame(frames, explicit="b_link_entrance"),
            "b_link_entrance",
        )

    def test_explicit_missing_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.select_entrance_frame({"a_link_entrance"}, explicit="z_link_entrance")

    def test_ambiguous_raises(self) -> None:
        frames = {"a_link_entrance", "b_link_entrance"}
        with self.assertRaises(ValueError):
            dr.select_entrance_frame(frames)

    def test_prefer_narrows_ambiguity(self) -> None:
        # Two entrance frames on /tf (target + distractor); /scoring/tf scopes one.
        frames = {
            "board/mod0/sfp_port_0_link_entrance",
            "board/mod1/sfp_port_1_link_entrance",
        }
        self.assertEqual(
            dr.select_entrance_frame(
                frames, prefer={"board/mod1/sfp_port_1_link_entrance"}
            ),
            "board/mod1/sfp_port_1_link_entrance",
        )

    def test_prefer_still_ambiguous_raises(self) -> None:
        frames = {"a_link_entrance", "b_link_entrance"}
        with self.assertRaises(ValueError):
            dr.select_entrance_frame(frames, prefer=frames)

    def test_none_present_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.select_entrance_frame({"world", "base_link"})

    def test_port_name_no_match_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.select_entrance_frame({"sfp_port_0_link_entrance"}, port_name="sc_port_1")


class TestPortOffsetLabels(unittest.TestCase):
    """The label must equal port_offset.tcp_frame_offset per frame (the invariant
    train_v3 --port-aux --aux-frame tcp relies on)."""

    def _poses(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # identity
                [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],  # translated
                [0.1, -0.2, 0.3, *_yaw_quat(0.6)],  # translated + yawed
            ]
        )

    def test_matches_tcp_frame_offset_per_frame(self) -> None:
        poses = self._poses()
        port = np.array([0.5, 0.4, 0.9])
        labels = dr.port_offset_labels(poses, port, frame="tcp")
        self.assertEqual(labels.shape, (3, 3))
        for i, pose in enumerate(poses):
            expected = port_offset.tcp_frame_offset(pose[:3], pose[3:], port)
            np.testing.assert_allclose(labels[i], expected, atol=1e-12)

    def test_base_frame_is_plain_delta(self) -> None:
        poses = self._poses()
        port = np.array([0.5, 0.4, 0.9])
        labels = dr.port_offset_labels(poses, port, frame="base")
        np.testing.assert_allclose(labels, port[None, :] - poses[:, :3], atol=1e-12)


class TestStallWindow(unittest.TestCase):
    def _ts(self, n: int, dt: float = 0.275) -> np.ndarray:
        return np.arange(n, dtype=np.float64) * dt

    def test_detects_sustained_stall_after_grace(self) -> None:
        n = 120
        ts = self._ts(n)  # ~32.7 s total
        speed = np.full(n, 0.05)
        speed[80:] = 0.001  # stalls at frame 80 (t~22 s), well past 15 s grace
        win = dr.stall_window(speed, ts, lookback_s=2.0, min_frames=8)
        self.assertIsNotNone(win)
        start, stop = win
        self.assertEqual(stop, n)
        self.assertLess(start, 80)  # includes the pre-stall lookback
        # lookback of 2 s at dt=0.275 is ~7 frames.
        self.assertGreaterEqual(start, 80 - 9)

    def test_no_stall_when_always_moving(self) -> None:
        n = 120
        self.assertIsNone(
            dr.stall_window(np.full(n, 0.05), self._ts(n))
        )

    def test_brief_dip_does_not_trigger(self) -> None:
        n = 120
        speed = np.full(n, 0.05)
        speed[80:85] = 0.0  # ~1.4 s dip, < 3 s window
        self.assertIsNone(dr.stall_window(speed, self._ts(n)))

    def test_grace_period_blocks_early_stall(self) -> None:
        # Low from the very start: must NOT latch before min_runtime_s, and the
        # onset is frame 0 so lookback start clamps to 0.
        n = 120
        ts = self._ts(n)
        speed = np.full(n, 0.0)
        win = dr.stall_window(speed, ts, min_runtime_s=15.0, min_frames=8)
        self.assertIsNotNone(win)
        start, stop = win
        self.assertEqual(start, 0)
        self.assertEqual(stop, n)

    def test_min_frames_floor(self) -> None:
        # Stall only in the final 3 frames but min_frames=8 -> window widened.
        n = 100
        ts = self._ts(n)
        speed = np.full(n, 0.05)
        speed[97:] = 0.0
        # Force the onset near the end by requiring only a tiny window_s.
        win = dr.stall_window(
            speed, ts, stall_window_s=0.2, lookback_s=0.0, min_frames=8
        )
        self.assertIsNotNone(win)
        start, stop = win
        self.assertEqual(stop, n)
        self.assertGreaterEqual(stop - start, 8)

    def test_bad_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            dr.stall_window(np.zeros((2, 2)), np.zeros(2))
        with self.assertRaises(ValueError):
            dr.stall_window(np.zeros(3), np.zeros(2))
        with self.assertRaises(ValueError):
            dr.stall_window(np.zeros(3), np.zeros(3), speed_threshold=0.0)


class TestSelectWindowFallback(unittest.TestCase):
    def test_terminal_window(self) -> None:
        self.assertEqual(dr.terminal_window(50, 24), (26, 50))
        self.assertEqual(dr.terminal_window(10, 24), (0, 10))

    def test_select_window_uses_stall(self) -> None:
        n = 120
        ts = np.arange(n, dtype=np.float64) * 0.275
        vel = np.zeros((n, 6))
        vel[:80, 0] = 0.05  # moving, then stalled
        (start, stop), stalled = dr.select_window(vel, ts)
        self.assertTrue(stalled)
        self.assertEqual(stop, n)

    def test_select_window_falls_back(self) -> None:
        n = 60
        ts = np.arange(n, dtype=np.float64) * 0.275
        vel = np.zeros((n, 6))
        vel[:, 0] = 0.05  # always moving -> no stall
        (start, stop), stalled = dr.select_window(vel, ts, fallback_frames=24)
        self.assertFalse(stalled)
        self.assertEqual((start, stop), (36, 60))


class TestSliceFrameArrays(unittest.TestCase):
    def test_slices_all(self) -> None:
        arrays = {
            "tcp_poses.npy": np.arange(70).reshape(10, 7).astype(float),
            "timestamps.npy": np.arange(10).astype(float),
        }
        out = dr.slice_frame_arrays(arrays, 3, 8)
        self.assertEqual(out["tcp_poses.npy"].shape, (5, 7))
        np.testing.assert_array_equal(out["timestamps.npy"], np.arange(3, 8))

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            dr.slice_frame_arrays(
                {"a": np.zeros((4, 2)), "b": np.zeros((5, 2))}, 0, 3
            )

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.slice_frame_arrays({"a": np.zeros((4, 2))}, 2, 2)
        with self.assertRaises(ValueError):
            dr.slice_frame_arrays({"a": np.zeros((4, 2))}, 0, 5)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            dr.slice_frame_arrays({}, 0, 1)


def _synthetic_frame_arrays(n: int) -> dict[str, np.ndarray]:
    """Build a minimal set of per-frame arrays with valid TCP poses."""
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, 3] = 0.0
    poses[:, 6] = 1.0  # identity quats
    poses[:, 0] = np.linspace(0.0, 0.1, n)  # some x motion
    return {
        "tcp_poses.npy": poses,
        "tcp_velocities.npy": np.zeros((n, 6), dtype=np.float32),
        "timestamps.npy": np.arange(n, dtype=np.float64) * 0.275,
        "wrenches.npy": np.zeros((n, 6), dtype=np.float32),
        "joint_positions.npy": np.zeros((n, 7), dtype=np.float32),
        "center_images.npy": np.zeros((n, 2, 2, 3), dtype=np.uint8),
        "left_images.npy": np.zeros((n, 2, 2, 3), dtype=np.uint8),
        "right_images.npy": np.zeros((n, 2, 2, 3), dtype=np.uint8),
    }


class TestWriteRelabeledEpisode(unittest.TestCase):
    def test_writes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            arrays = _synthetic_frame_arrays(12)
            port = np.array([0.5, 0.4, 0.9])
            res = dr.write_relabeled_episode(
                pathlib.Path(td) / "ep", arrays, port,
                entrance_frame="p_link_entrance", window=(0, 12),
            )
            ep = pathlib.Path(res.episode_dir)
            # All standard per-frame arrays present.
            for name in dr.FRAME_ARRAY_NAMES:
                self.assertTrue((ep / name).exists(), name)
            # port_target.npy schema: length-3 float.
            target = np.load(ep / dr.PORT_TARGET_NAME)
            self.assertEqual(target.shape, (3,))
            np.testing.assert_allclose(target, port)
            # port_offsets.npy schema: (N, 3), equals the reused offset math.
            offsets = np.load(ep / dr.PORT_OFFSETS_NAME)
            self.assertEqual(offsets.shape, (12, 3))
            np.testing.assert_allclose(
                offsets, dr.port_offset_labels(arrays["tcp_poses.npy"], port), atol=1e-6
            )
            # Non-seating stall marker.
            self.assertEqual(int(np.load(ep / "insertion_frame.npy")), -1)
            self.assertEqual(res.n_frames, 12)

    def test_missing_poses_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                dr.write_relabeled_episode(td, {"timestamps.npy": np.zeros(3)}, [0, 0, 0])

    def test_bad_port_length_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                dr.write_relabeled_episode(
                    td, {"tcp_poses.npy": np.zeros((3, 7))}, [0, 0]
                )


class TestPortLabelsConsumption(unittest.TestCase):
    """End-to-end (no ROS/GPU): an emitted episode + campaign_log is consumed by
    opt.port_labels.build_labels EXACTLY as train_v3 --port-aux would, yielding
    all-valid labels equal to the true-port offset."""

    def test_build_labels_reads_port_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            arrays = _synthetic_frame_arrays(10)
            port = np.array([0.55, 0.42, 0.88])
            res = dr.write_relabeled_episode(root / "ep_dagger_0", arrays, port)
            log = root / port_labels.CAMPAIGN_LOG_NAME
            dr.append_campaign_row(
                log, episode_dir="ep_dagger_0", config="/c.yaml",
                stratum="dagger", plug="sfp", rep="0", frames=res.n_frames,
            )
            label_set = port_labels.build_labels(
                [res.episode_dir], aux_dim=3, aux_frame="tcp", campaign_log=str(log)
            )
            # All frames valid (port_target override forces validity).
            self.assertEqual(label_set.offsets.shape, (10, 3))
            self.assertTrue(bool(label_set.valid.all()))
            # Labels equal the true-port offsets the aux head will regress.
            np.testing.assert_allclose(
                label_set.offsets,
                dr.port_offset_labels(arrays["tcp_poses.npy"], port).astype(np.float32),
                atol=1e-5,
            )

    def test_campaign_row_parses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = pathlib.Path(td) / port_labels.CAMPAIGN_LOG_NAME
            dr.append_campaign_row(
                log, episode_dir="ep0", config="/c.yaml", stratum="s",
                plug="sfp", rep="0", frames=15,
            )
            status = port_labels.parse_campaign_log(log)
            self.assertIn("ep0", status)
            self.assertTrue(status["ep0"].valid)


if __name__ == "__main__":
    unittest.main()
