"""Unit tests for prepare_dataset.py trim/sync/serialisation logic.

All tests use small in-memory synthetic arrays and run without ROS/Gazebo/GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import prepare_dataset as pd


def _img(fill: int, h: int = 2, w: int = 3) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


class TestComputeTaskWindow(unittest.TestCase):
    def test_basic_window_with_margin(self) -> None:
        t0, t1 = pd.compute_task_window([10.0, 12.0, 11.0])
        self.assertAlmostEqual(t0, 10.0)
        self.assertAlmostEqual(t1, 12.3)

    def test_single_command(self) -> None:
        t0, t1 = pd.compute_task_window([5.0])
        self.assertAlmostEqual(t0, 5.0)
        self.assertAlmostEqual(t1, 5.3)

    def test_empty_returns_unbounded(self) -> None:
        t0, t1 = pd.compute_task_window([])
        self.assertEqual(t0, float("-inf"))
        self.assertEqual(t1, float("inf"))

    def test_custom_margin(self) -> None:
        _, t1 = pd.compute_task_window([1.0, 2.0], margin=1.5)
        self.assertAlmostEqual(t1, 3.5)

    def test_negative_margin_raises(self) -> None:
        with self.assertRaises(ValueError):
            pd.compute_task_window([1.0], margin=-0.1)


class TestReorderJointPositions(unittest.TestCase):
    def test_reorders_into_canonical_order(self) -> None:
        # Deliberately shuffled publication order.
        names = [
            "elbow_joint", "shoulder_pan_joint", "gripper/left_finger_joint",
            "wrist_2_joint", "shoulder_lift_joint", "wrist_3_joint", "wrist_1_joint",
        ]
        positions = [2.0, 0.0, 6.0, 4.0, 1.0, 5.0, 3.0]  # value == canonical index
        out = pd.reorder_joint_positions(names, positions)
        self.assertEqual(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_missing_name_returns_none(self) -> None:
        names = ["shoulder_pan_joint"]  # incomplete
        self.assertIsNone(pd.reorder_joint_positions(names, [0.0]))

    def test_length_mismatch_returns_none(self) -> None:
        self.assertIsNone(
            pd.reorder_joint_positions(["shoulder_pan_joint"], [0.0, 1.0]))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(pd.reorder_joint_positions([], []))


class TestSynchronizeFrames(unittest.TestCase):
    def setUp(self) -> None:
        self.center = [(1.0, _img(10)), (2.0, _img(20)), (3.0, _img(30))]
        self.left = [(0.9, _img(1)), (1.9, _img(2)), (2.9, _img(3))]
        self.right = [(1.1, _img(4)), (2.1, _img(5)), (3.1, _img(6))]
        self.controller = [
            (1.0, [0.0] * 7, [0.1] * 6, [0.0] * 6),
            (2.0, [1.0] * 7, [0.2] * 6, [0.0] * 6),
            (3.0, [2.0] * 7, [0.3] * 6, [0.0] * 6),
        ]
        self.wrench = [
            (0.95, [11.0] * 6), (2.05, [22.0] * 6), (3.02, [33.0] * 6),
        ]
        self.joints = [
            (1.02, [0.1] * 7), (1.98, [0.2] * 7), (3.05, [0.3] * 7),
        ]

    def test_wrench_joints_default_zeros_when_absent(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 1.0, 1.0)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].wrench, [0.0] * 6)
        self.assertEqual(frames[0].joint_pos, [0.0] * 7)

    def test_nearest_wrench_and_joints_selected(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 2.0, 2.0,
            wrench=self.wrench, joints=self.joints)
        self.assertEqual(len(frames), 1)
        # nearest wrench to t=2.0 is (2.05, 22.0); nearest joints is (1.98, 0.2)
        self.assertEqual(frames[0].wrench, [22.0] * 6)
        self.assertEqual(frames[0].joint_pos, [0.2] * 7)

    def test_wrench_never_drops_frames(self) -> None:
        # A far-away single wrench sample must NOT reduce the kept-frame count.
        far_wrench = [(1000.0, [9.0] * 6)]
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 1.0, 3.0,
            wrench=far_wrench)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(f.wrench == [9.0] * 6 for f in frames))

    def test_trims_outside_window(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 1.0, 2.5)
        self.assertEqual([f.timestamp for f in frames], [1.0, 2.0])

    def test_nearest_side_camera_selected(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 1.0, 1.0)
        self.assertEqual(len(frames), 1)
        # nearest left to t=1.0 is (0.9, fill=1); nearest right is (1.1, fill=4)
        self.assertEqual(int(frames[0].left_image[0, 0, 0]), 1)
        self.assertEqual(int(frames[0].right_image[0, 0, 0]), 4)
        self.assertEqual(int(frames[0].center_image[0, 0, 0]), 10)

    def test_nearest_controller_state_selected(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, self.controller, 2.0, 2.0)
        self.assertEqual(frames[0].tcp_pose, [1.0] * 7)
        self.assertEqual(frames[0].tcp_velocity, [0.2] * 6)

    def test_drops_far_controller(self) -> None:
        controller = [(100.0, [0.0] * 7, [0.0] * 6, [0.0] * 6)]
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, controller, 1.0, 3.0, max_dt=0.5)
        self.assertEqual(frames, [])

    def test_empty_side_camera_returns_empty(self) -> None:
        frames = pd.synchronize_frames(
            self.center, [], self.right, self.controller, 1.0, 3.0)
        self.assertEqual(frames, [])

    def test_empty_controller_returns_empty(self) -> None:
        frames = pd.synchronize_frames(
            self.center, self.left, self.right, [], 1.0, 3.0)
        self.assertEqual(frames, [])


class TestSaveEpisodes(unittest.TestCase):
    def _make_frames(self, n: int) -> list[pd.EpisodeFrame]:
        return [
            pd.EpisodeFrame(
                timestamp=float(i),
                left_image=_img(i, 4, 5),
                center_image=_img(i + 100, 4, 5),
                right_image=_img(i + 200, 4, 5),
                tcp_pose=[float(i)] * 7,
                tcp_velocity=[float(i) * 0.1] * 6,
                tcp_error=[0.0] * 6,
                wrench=[float(i) * 0.5] * 6,
                joint_pos=[float(i) * 0.2] * 7,
            )
            for i in range(n)
        ]

    def test_shapes_and_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            n = pd.save_episodes(self._make_frames(3), d)
            self.assertEqual(n, 3)
            p = Path(d)
            center = np.load(p / "center_images.npy")
            poses = np.load(p / "tcp_poses.npy")
            vels = np.load(p / "tcp_velocities.npy")
            ts = np.load(p / "timestamps.npy")
            wrenches = np.load(p / "wrenches.npy")
            joints = np.load(p / "joint_positions.npy")
            self.assertEqual(center.shape, (3, 4, 5, 3))
            self.assertEqual(center.dtype, np.uint8)
            self.assertEqual(poses.shape, (3, 7))
            self.assertEqual(vels.shape, (3, 6))
            self.assertEqual(ts.shape, (3,))
            self.assertEqual(poses.dtype.kind, "f")
            # new proprioceptive fields: (N, 6) and (N, 7) float32
            self.assertEqual(wrenches.shape, (3, 6))
            self.assertEqual(wrenches.dtype, np.float32)
            self.assertEqual(joints.shape, (3, 7))
            self.assertEqual(joints.dtype, np.float32)
            np.testing.assert_allclose(wrenches[2], [1.0] * 6)
            np.testing.assert_allclose(joints[2], [0.4] * 7)
            # center images carry the +100 offset fill values
            self.assertEqual(int(center[2, 0, 0, 0]), 102)

    def test_default_frame_writes_zero_proprioception(self) -> None:
        # EpisodeFrame constructed without wrench/joint_pos zero-fills (backward compat).
        frame = pd.EpisodeFrame(
            timestamp=0.0,
            left_image=_img(0, 4, 5),
            center_image=_img(0, 4, 5),
            right_image=_img(0, 4, 5),
            tcp_pose=[0.0] * 7,
            tcp_velocity=[0.0] * 6,
            tcp_error=[0.0] * 6,
        )
        self.assertEqual(frame.wrench, [0.0] * 6)
        self.assertEqual(frame.joint_pos, [0.0] * 7)
        with tempfile.TemporaryDirectory() as d:
            pd.save_episodes([frame], d)
            self.assertEqual(np.load(Path(d) / "wrenches.npy").shape, (1, 6))
            self.assertEqual(np.load(Path(d) / "joint_positions.npy").shape, (1, 7))

    def test_empty_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            n = pd.save_episodes([], d)
            self.assertEqual(n, 0)
            self.assertFalse((Path(d) / "center_images.npy").exists())
            self.assertFalse((Path(d) / "wrenches.npy").exists())
            self.assertFalse((Path(d) / "joint_positions.npy").exists())


class TestSeatFrameIndex(unittest.TestCase):
    """Pure seat-event -> nearest-frame mapping (no rosbag I/O)."""

    def test_maps_to_nearest_frame(self) -> None:
        frame_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # synthetic timestamps
        # Event at 2.6 is nearest frame 3 (|3-2.6|=0.4 < |2-2.6|=0.6).
        self.assertEqual(pd.seat_frame_index(frame_times, [2.6]), 3)

    def test_earliest_event_is_used(self) -> None:
        frame_times = [0.0, 1.0, 2.0, 3.0, 4.0]
        # Two events; the earliest (1.1) wins -> nearest frame 1.
        self.assertEqual(pd.seat_frame_index(frame_times, [3.9, 1.1]), 1)

    def test_exact_match(self) -> None:
        self.assertEqual(pd.seat_frame_index([10.0, 10.5, 11.0], [10.5]), 1)

    def test_no_events_returns_minus_one(self) -> None:
        self.assertEqual(pd.seat_frame_index([0.0, 1.0, 2.0], []), -1)

    def test_no_frames_returns_minus_one(self) -> None:
        self.assertEqual(pd.seat_frame_index([], [1.0]), -1)

    def test_event_before_first_frame_clamps_to_zero(self) -> None:
        self.assertEqual(pd.seat_frame_index([5.0, 6.0, 7.0], [1.0]), 0)


class TestSaveSeatMarker(unittest.TestCase):
    """save_seat_marker persistence (additive .npy artifacts)."""

    def _frames_at(self, times: list[float]) -> list[pd.EpisodeFrame]:
        return [
            pd.EpisodeFrame(
                timestamp=t,
                left_image=_img(0),
                center_image=_img(0),
                right_image=_img(0),
                tcp_pose=[0.0] * 7,
                tcp_velocity=[0.0] * 6,
                tcp_error=[0.0] * 6,
            )
            for t in times
        ]

    def test_writes_index_and_times(self) -> None:
        frames = self._frames_at([0.0, 1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as d:
            idx = pd.save_seat_marker(frames, [2.1], d)
            self.assertEqual(idx, 2)  # 2.1 nearest frame at t=2.0
            saved_idx = np.load(Path(d) / "insertion_frame.npy")
            saved_t = np.load(Path(d) / "seat_time.npy")
            self.assertEqual(int(saved_idx), 2)
            self.assertEqual(saved_idx.dtype, np.int64)
            np.testing.assert_allclose(saved_t, [2.1])

    def test_no_events_writes_minus_one_and_empty_times(self) -> None:
        frames = self._frames_at([0.0, 1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            idx = pd.save_seat_marker(frames, [], d)
            self.assertEqual(idx, -1)
            self.assertEqual(int(np.load(Path(d) / "insertion_frame.npy")), -1)
            self.assertEqual(np.load(Path(d) / "seat_time.npy").shape, (0,))

    def test_empty_frames_write_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            idx = pd.save_seat_marker([], [1.0], d)
            self.assertEqual(idx, -1)
            self.assertFalse((Path(d) / "insertion_frame.npy").exists())
            self.assertFalse((Path(d) / "seat_time.npy").exists())

    def test_marker_index_slices_saved_timestamps(self) -> None:
        # The persisted index must address the same array save_episodes writes.
        frames = self._frames_at([100.0, 100.5, 101.0, 101.5, 102.0])
        with tempfile.TemporaryDirectory() as d:
            pd.save_episodes(frames, d)
            idx = pd.save_seat_marker(frames, [101.4], d)
            ts = np.load(Path(d) / "timestamps.npy")
            self.assertEqual(idx, 3)  # 101.4 nearest 101.5 (frame 3)
            self.assertAlmostEqual(float(ts[idx]), 101.5)


if __name__ == "__main__":
    unittest.main()
