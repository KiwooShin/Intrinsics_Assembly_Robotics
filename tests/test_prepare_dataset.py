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
            self.assertEqual(center.shape, (3, 4, 5, 3))
            self.assertEqual(center.dtype, np.uint8)
            self.assertEqual(poses.shape, (3, 7))
            self.assertEqual(vels.shape, (3, 6))
            self.assertEqual(ts.shape, (3,))
            self.assertEqual(poses.dtype.kind, "f")
            # center images carry the +100 offset fill values
            self.assertEqual(int(center[2, 0, 0, 0]), 102)

    def test_empty_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            n = pd.save_episodes([], d)
            self.assertEqual(n, 0)
            self.assertFalse((Path(d) / "center_images.npy").exists())


if __name__ == "__main__":
    unittest.main()
