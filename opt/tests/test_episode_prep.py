"""Unit tests for opt.episode_prep pure numpy helpers (CPU-only)."""

from __future__ import annotations

import unittest

import numpy as np

from opt.tests import _pathfix  # noqa: F401  (path side effect)
from opt import episode_prep


class FrameSpeedTest(unittest.TestCase):
    def test_linear_norm_ignores_angular(self) -> None:
        # vx=3, vy=4 -> speed 5; angular columns must not affect the norm.
        vel = np.array([[3.0, 4.0, 0.0, 9.0, 9.0, 9.0]], dtype=np.float32)
        np.testing.assert_allclose(episode_prep.frame_speed(vel), [5.0])

    def test_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            episode_prep.frame_speed(np.zeros((3,)))
        with self.assertRaises(ValueError):
            episode_prep.frame_speed(np.zeros((3, 2)))  # narrower than 3


class TerminalTailTrimTest(unittest.TestCase):
    def test_trims_seated_tail_with_margin(self) -> None:
        # Frames 0..4 move, 5..9 are a seated zero-velocity tail.
        speed = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        # Last moving frame is index 4; keep 4+1 + margin(2) = 7 frames.
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.5, margin_frames=2), 7
        )

    def test_margin_clamps_to_length(self) -> None:
        speed = np.array([1, 1, 0, 0], dtype=np.float32)
        # last moving index 1 -> 1+1+10 = 12, clamped to n=4.
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.5, margin_frames=10), 4
        )

    def test_zero_margin_keeps_up_to_last_moving(self) -> None:
        speed = np.array([1, 1, 1, 0, 0], dtype=np.float32)
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.5, margin_frames=0), 3
        )

    def test_all_below_threshold_keeps_all(self) -> None:
        # Degenerate near-stationary demo: no anchor, keep everything (not 0).
        speed = np.array([0.001, 0.002, 0.0], dtype=np.float32)
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.003, margin_frames=1), 3
        )

    def test_none_below_threshold_keeps_all(self) -> None:
        # Every frame moves (no seated tail): last moving is the final frame.
        speed = np.array([1, 2, 3, 4], dtype=np.float32)
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.5, margin_frames=1), 4
        )

    def test_threshold_is_inclusive(self) -> None:
        # A frame exactly at threshold counts as moving.
        speed = np.array([1.0, 0.003, 0.0], dtype=np.float32)
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(speed, 0.003, margin_frames=0), 2
        )

    def test_empty_episode(self) -> None:
        self.assertEqual(
            episode_prep.terminal_tail_trim_length(np.zeros((0,)), 0.003, 1), 0
        )

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            episode_prep.terminal_tail_trim_length(np.zeros((2, 2)), 0.1, 1)
        with self.assertRaises(ValueError):
            episode_prep.terminal_tail_trim_length(np.zeros((3,)), -0.1, 1)
        with self.assertRaises(ValueError):
            episode_prep.terminal_tail_trim_length(np.zeros((3,)), 0.1, -1)


class PushinWeightsTest(unittest.TestCase):
    def test_ramp_values_and_boundaries(self) -> None:
        # n=6, ramp over last 3 frames to W=4: first 3 frames weight 1.0, then the
        # ramp linspace(1, 4, 3) = [1, 2.5, 4] on the final 3 frames.
        w = episode_prep.pushin_weights(6, ramp_frames=3, w_max=4.0)
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 1.0, 2.5, 4.0], rtol=1e-6)
        self.assertEqual(w.dtype, np.float32)

    def test_last_frame_is_w_max(self) -> None:
        w = episode_prep.pushin_weights(20, ramp_frames=7, w_max=4.0)
        self.assertAlmostEqual(float(w[-1]), 4.0, places=5)
        # Frame right before the ramp window is still 1.0.
        self.assertAlmostEqual(float(w[20 - 7 - 1]), 1.0, places=5)
        self.assertAlmostEqual(float(w[20 - 7]), 1.0, places=5)  # ramp start == 1.0

    def test_ramp_longer_than_episode_clamps(self) -> None:
        # ramp_frames > n: the whole episode ramps 1 -> W.
        w = episode_prep.pushin_weights(3, ramp_frames=10, w_max=3.0)
        np.testing.assert_allclose(w, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_weight_one_is_all_ones(self) -> None:
        w = episode_prep.pushin_weights(5, ramp_frames=3, w_max=1.0)
        np.testing.assert_array_equal(w, np.ones(5, dtype=np.float32))

    def test_zero_ramp_is_all_ones(self) -> None:
        w = episode_prep.pushin_weights(5, ramp_frames=0, w_max=4.0)
        np.testing.assert_array_equal(w, np.ones(5, dtype=np.float32))

    def test_single_frame_episode(self) -> None:
        w = episode_prep.pushin_weights(1, ramp_frames=7, w_max=4.0)
        np.testing.assert_allclose(w, [4.0])

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            episode_prep.pushin_weights(0, 3, 4.0)
        with self.assertRaises(ValueError):
            episode_prep.pushin_weights(5, 3, 0.5)


class EpisodeBoundsTest(unittest.TestCase):
    def test_contiguous_blocks(self) -> None:
        epid = np.array([0, 0, 0, 1, 1, 2])
        self.assertEqual(
            episode_prep.episode_bounds(epid), [(0, 3), (3, 5), (5, 6)]
        )

    def test_non_dense_ids(self) -> None:
        # Ids need not be dense/sorted, only contiguous per block.
        epid = np.array([5, 5, 2, 2, 2, 9])
        self.assertEqual(
            episode_prep.episode_bounds(epid), [(0, 2), (2, 5), (5, 6)]
        )

    def test_empty(self) -> None:
        self.assertEqual(episode_prep.episode_bounds(np.zeros((0,), dtype=int)), [])


class BuildKeepAndWeightsTest(unittest.TestCase):
    def _two_episode_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Episode 0: 5 frames, moving then seated tail (frames 3,4 stopped).
        # Episode 1: 4 frames, all moving.
        ep0 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        ep1 = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]])
        vel = np.concatenate([ep0, ep1]).astype(np.float32)
        # Pad velocities to 6-D twist (angular zeros).
        vel = np.concatenate([vel, np.zeros((len(vel), 3), np.float32)], axis=1)
        epid = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
        return epid, vel

    def test_trim_drops_seated_tail_per_episode(self) -> None:
        epid, vel = self._two_episode_data()
        keep, weights = episode_prep.build_keep_and_weights(
            epid, vel, tail_trim=True, trim_threshold=0.5, trim_margin_frames=0,
            pushin_ramp_frames=0, pushin_weight=1.0,
        )
        # Ep0 last moving index 2 -> keep 3; ep1 all moving -> keep 4.
        np.testing.assert_array_equal(
            keep, [True, True, True, False, False, True, True, True, True]
        )
        # No push-in weighting -> kept frames weight 1, dropped frames 0.
        np.testing.assert_array_equal(
            weights, [1, 1, 1, 0, 0, 1, 1, 1, 1]
        )

    def test_weights_ramp_within_trimmed_length(self) -> None:
        epid, vel = self._two_episode_data()
        keep, weights = episode_prep.build_keep_and_weights(
            epid, vel, tail_trim=True, trim_threshold=0.5, trim_margin_frames=0,
            pushin_ramp_frames=2, pushin_weight=3.0,
        )
        # Ep0 kept length 3, ramp over last 2 -> [1, 1, 3]; ep1 length 4 -> [1,1,1,3].
        np.testing.assert_allclose(
            weights[keep], [1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], rtol=1e-6
        )

    def test_no_trim_keeps_all(self) -> None:
        epid, vel = self._two_episode_data()
        keep, weights = episode_prep.build_keep_and_weights(
            epid, vel, tail_trim=False, trim_threshold=0.5, trim_margin_frames=0,
            pushin_ramp_frames=0, pushin_weight=1.0,
        )
        self.assertTrue(keep.all())
        np.testing.assert_array_equal(weights, np.ones(len(epid), np.float32))

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            episode_prep.build_keep_and_weights(
                np.zeros(3, int), np.zeros((4, 6), np.float32),
                tail_trim=True, trim_threshold=0.1, trim_margin_frames=0,
                pushin_ramp_frames=0, pushin_weight=1.0,
            )


class SecondsToFramesTest(unittest.TestCase):
    def test_rounds_to_nearest_frame(self) -> None:
        # 0.3 s at 0.275 s/frame -> round(1.09) = 1.
        self.assertEqual(episode_prep.seconds_to_frames(0.3, 0.275), 1)
        # 2.0 s at 0.275 s/frame -> round(7.27) = 7.
        self.assertEqual(episode_prep.seconds_to_frames(2.0, 0.275), 7)

    def test_minimum_floor(self) -> None:
        self.assertEqual(episode_prep.seconds_to_frames(0.01, 0.275, minimum=1), 1)
        self.assertEqual(episode_prep.seconds_to_frames(0.0, 0.275, minimum=0), 0)

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            episode_prep.seconds_to_frames(-1.0, 0.275)
        with self.assertRaises(ValueError):
            episode_prep.seconds_to_frames(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
