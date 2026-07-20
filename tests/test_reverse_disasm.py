#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the disassembly -> insertion time-reversal post-process.

Exercises :mod:`reverse_disasm` -- the pure reversal math (reverse+negate the
action, reverse observation order without changing values, rebuild monotonic
timestamps, seat marker = N-1) and the thin ``.npy`` I/O wrapper. Requires neither
ROS, Gazebo, torch, nor a GPU.

Run with::

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

import reverse_disasm as rd


class ReverseActionTest(unittest.TestCase):
    """``a_insert[j] = -velocities[N-1-j]`` (reverse order + negate all six)."""

    def test_reverse_and_negate_explicit(self) -> None:
        vel = np.array(
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],   # frame 0 (at seat depth)
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],   # frame 1
                [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],   # frame 2 (fully retracted)
            ]
        )
        got = rd.reverse_action(vel)
        want = np.array(
            [
                [-7.0, -8.0, -9.0, -0.7, -0.8, -0.9],  # reversed frame 0 = -vel[2]
                [-4.0, -5.0, -6.0, -0.4, -0.5, -0.6],  # reversed frame 1 = -vel[1]
                [-1.0, -2.0, -3.0, -0.1, -0.2, -0.3],  # reversed frame 2 = -vel[0]
            ]
        )
        np.testing.assert_allclose(got, want)

    def test_indexwise_identity(self) -> None:
        rng = np.random.default_rng(3)
        vel = rng.standard_normal((17, 6))
        got = rd.reverse_action(vel)
        n = vel.shape[0]
        for j in range(n):
            np.testing.assert_allclose(got[j], -vel[n - 1 - j])

    def test_double_reversal_returns_original(self) -> None:
        # Reversing an insertion back to a retract and again recovers the input.
        rng = np.random.default_rng(4)
        vel = rng.standard_normal((11, 6))
        np.testing.assert_allclose(rd.reverse_action(rd.reverse_action(vel)), vel)

    def test_all_six_components_flip_sign(self) -> None:
        vel = np.ones((5, 6))
        self.assertTrue(np.all(rd.reverse_action(vel) == -1.0))

    def test_bad_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            rd.reverse_action(np.zeros((4, 3)))  # wrong width
        with self.assertRaises(ValueError):
            rd.reverse_action(np.zeros((0, 6)))  # empty
        with self.assertRaises(ValueError):
            rd.reverse_action(np.zeros((6,)))    # not 2-D


class ReverseObservationTest(unittest.TestCase):
    """Observation arrays reverse in ORDER only; values are untouched."""

    def test_order_reversed_values_intact(self) -> None:
        poses = np.arange(7 * 4, dtype=np.float64).reshape(4, 7)
        got = rd.reverse_observation(poses)
        np.testing.assert_allclose(got, poses[::-1])
        # Value at reversed frame j equals recorded frame N-1-j (unchanged values).
        for j in range(4):
            np.testing.assert_allclose(got[j], poses[3 - j])

    def test_works_for_image_stack(self) -> None:
        imgs = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
        got = rd.reverse_observation(imgs)
        np.testing.assert_array_equal(got, imgs[::-1])
        self.assertEqual(got.dtype, np.uint8)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            rd.reverse_observation(np.zeros((0, 7)))


class RebuildTimestampsTest(unittest.TestCase):
    """Rebuilt timestamps are monotonic and preserve reversed inter-frame gaps."""

    def test_monotonic_and_endpoints(self) -> None:
        t = np.array([10.0, 10.5, 10.9, 11.6, 12.0])
        new = rd.rebuild_timestamps(t)
        self.assertTrue(np.all(np.diff(new) > 0.0))          # strictly increasing
        self.assertAlmostEqual(new[0], t[0])
        self.assertAlmostEqual(new[-1], t[-1])

    def test_gaps_are_reversed(self) -> None:
        t = np.array([0.0, 0.5, 0.9, 1.6, 2.0])
        new = rd.rebuild_timestamps(t)
        orig_gaps = np.diff(t)              # [0.5, 0.4, 0.7, 0.4]
        new_gaps = np.diff(new)
        np.testing.assert_allclose(new_gaps, orig_gaps[::-1])

    def test_uniform_timestamps_unchanged(self) -> None:
        t = np.array([0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(rd.rebuild_timestamps(t), t)

    def test_bad_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            rd.rebuild_timestamps(np.zeros((0,)))
        with self.assertRaises(ValueError):
            rd.rebuild_timestamps(np.zeros((3, 2)))


class ReversedSeatIndexTest(unittest.TestCase):
    """The seat is the last frame of the reversed episode."""

    def test_value(self) -> None:
        self.assertEqual(rd.reversed_seat_index(25), 24)
        self.assertEqual(rd.reversed_seat_index(1), 0)

    def test_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            rd.reversed_seat_index(0)


class ReverseEpisodeIOTest(unittest.TestCase):
    """End-to-end ``.npy`` round trip through :func:`reverse_episode`."""

    def _write_episode(self, d: pathlib.Path, n: int, with_extra: bool) -> dict:
        rng = np.random.default_rng(7)
        arrays = {
            "center_images.npy": rng.integers(0, 255, (n, 4, 4, 3), dtype=np.uint8),
            "left_images.npy": rng.integers(0, 255, (n, 4, 4, 3), dtype=np.uint8),
            "right_images.npy": rng.integers(0, 255, (n, 4, 4, 3), dtype=np.uint8),
            "tcp_poses.npy": rng.standard_normal((n, 7)).astype(np.float32),
            "tcp_velocities.npy": rng.standard_normal((n, 6)).astype(np.float32),
            "timestamps.npy": np.cumsum(rng.uniform(0.05, 0.1, n)).astype(np.float64),
        }
        if with_extra:
            arrays["wrenches.npy"] = rng.standard_normal((n, 6)).astype(np.float32)
            arrays["joint_positions.npy"] = rng.standard_normal((n, 7)).astype(np.float32)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in arrays.items():
            np.save(d / name, arr)
        return arrays

    def test_full_round_trip(self) -> None:
        n = 12
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = pathlib.Path(tmp) / "raw"
            out_dir = pathlib.Path(tmp) / "ep"
            src = self._write_episode(in_dir, n, with_extra=True)

            written = rd.reverse_episode(in_dir, out_dir)
            self.assertEqual(written, n)

            # Action: reversed + negated.
            out_vel = np.load(out_dir / "tcp_velocities.npy")
            np.testing.assert_allclose(
                out_vel, -src["tcp_velocities.npy"][::-1], rtol=1e-6, atol=1e-6
            )
            # Observations: reversed order, values intact.
            for name in ("center_images.npy", "tcp_poses.npy", "wrenches.npy",
                         "joint_positions.npy"):
                np.testing.assert_array_equal(
                    np.load(out_dir / name), src[name][::-1]
                )
            # Wrench sign is NOT flipped (it is a proxy kept as-is).
            np.testing.assert_array_equal(
                np.load(out_dir / "wrenches.npy"), src["wrenches.npy"][::-1]
            )
            # Timestamps: monotonic.
            out_ts = np.load(out_dir / "timestamps.npy")
            self.assertTrue(np.all(np.diff(out_ts) > 0.0))
            # Seat marker = N-1.
            self.assertEqual(int(np.load(out_dir / "insertion_frame.npy")), n - 1)
            seat_time = np.load(out_dir / "seat_time.npy")
            self.assertEqual(seat_time.shape, (1,))
            self.assertAlmostEqual(float(seat_time[0]), float(out_ts[-1]))

    def test_missing_optional_extras_ok(self) -> None:
        n = 6
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = pathlib.Path(tmp) / "raw"
            out_dir = pathlib.Path(tmp) / "ep"
            self._write_episode(in_dir, n, with_extra=False)
            written = rd.reverse_episode(in_dir, out_dir)
            self.assertEqual(written, n)
            self.assertTrue((out_dir / "tcp_velocities.npy").is_file())
            self.assertFalse((out_dir / "wrenches.npy").is_file())

    def test_missing_required_array_raises(self) -> None:
        n = 5
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = pathlib.Path(tmp) / "raw"
            out_dir = pathlib.Path(tmp) / "ep"
            self._write_episode(in_dir, n, with_extra=False)
            (in_dir / "tcp_poses.npy").unlink()
            with self.assertRaises(FileNotFoundError):
                rd.reverse_episode(in_dir, out_dir)

    def test_frame_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = pathlib.Path(tmp) / "raw"
            out_dir = pathlib.Path(tmp) / "ep"
            self._write_episode(in_dir, 8, with_extra=True)
            np.save(in_dir / "timestamps.npy", np.arange(7, dtype=np.float64))
            with self.assertRaises(ValueError):
                rd.reverse_episode(in_dir, out_dir)

    def test_missing_input_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                rd.reverse_episode(pathlib.Path(tmp) / "nope", pathlib.Path(tmp) / "o")


if __name__ == "__main__":
    unittest.main()
