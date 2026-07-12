"""Unit tests for opt.dataset pure numpy helpers (CPU-only)."""

from __future__ import annotations

import unittest

import numpy as np

from opt.tests import _pathfix  # noqa: F401
from opt import dataset


class BuildActionChunksTest(unittest.TestCase):
    def test_shape_and_clamping(self) -> None:
        vel = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
        chunks = dataset.build_action_chunks(vel, k=3)
        self.assertEqual(chunks.shape, (5, 3, 6))
        # First frame: rows 0,1,2.
        np.testing.assert_array_equal(chunks[0], vel[[0, 1, 2]])
        # Last frame clamps: rows 4,4,4.
        np.testing.assert_array_equal(chunks[4], vel[[4, 4, 4]])

    def test_k_one_is_identity(self) -> None:
        vel = np.random.default_rng(0).normal(size=(7, 6))
        chunks = dataset.build_action_chunks(vel, k=1)
        np.testing.assert_array_equal(chunks[:, 0], vel)

    def test_matches_train_v2_indexing(self) -> None:
        # Replicate the exact expression used in train_v2.load_all.
        n, k = 8, 4
        vel = np.random.default_rng(1).normal(size=(n, 6))
        idx = np.clip(np.arange(n)[:, None] + np.arange(k)[None, :], 0, n - 1)
        np.testing.assert_array_equal(dataset.build_action_chunks(vel, k), vel[idx])

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            dataset.build_action_chunks(np.zeros((3, 6)), k=0)
        with self.assertRaises(ValueError):
            dataset.build_action_chunks(np.zeros((3,)), k=2)


class NormalizationTest(unittest.TestCase):
    def test_stats_and_roundtrip(self) -> None:
        x = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])
        mean, std = dataset.normalization_stats(x)
        np.testing.assert_allclose(mean, [2.0, 14.0])
        # std floored by +1e-6.
        self.assertTrue(np.all(std > 0))
        z = dataset.normalize(x, mean, std)
        np.testing.assert_allclose(z.mean(0), [0.0, 0.0], atol=1e-6)

    def test_constant_column_no_div0(self) -> None:
        x = np.ones((5, 3))
        mean, std = dataset.normalization_stats(x)
        z = dataset.normalize(x, mean, std)  # would be 0/1e-6 -> 0, finite.
        self.assertTrue(np.all(np.isfinite(z)))


class EpisodeSplitTest(unittest.TestCase):
    def test_masks_partition(self) -> None:
        ep = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        tr, va = dataset.episode_split_masks(ep, {2, 3})
        np.testing.assert_array_equal(va, [False, False, False, False, True, True, True, True])
        np.testing.assert_array_equal(tr, ~va)

    def test_empty_val_is_overfit(self) -> None:
        ep = np.array([0, 0, 1, 1])
        tr, va = dataset.episode_split_masks(ep, set())
        np.testing.assert_array_equal(tr, va)
        self.assertTrue(np.all(tr))


class FirstActionMetricTest(unittest.TestCase):
    def test_denormalized_first_action_error(self) -> None:
        # Two frames, K=2, D=2. Only first chunk step matters.
        pred = np.zeros((2, 2, 2))
        target = np.ones((2, 2, 2))
        act_std = np.array([0.5, 2.0])
        # |0-1| * std = [0.5, 2.0], mean over dims = 1.25, mean over frames = 1.25.
        self.assertAlmostEqual(dataset.first_action_l1_meters(pred, target, act_std), 1.25)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            dataset.first_action_l1_meters(np.zeros((2, 2, 2)), np.zeros((2, 3, 2)), np.ones(2))


if __name__ == "__main__":
    unittest.main()
