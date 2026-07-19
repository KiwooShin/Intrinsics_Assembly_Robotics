#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for ROS-free ACT temporal ensembling used by ``DeployACT``.

These exercise :mod:`aic_example_policies.ros.chunk_ensemble` -- the pure
exp-weighted temporal ensemble over overlapping receding-horizon twist chunks
and the ``select_exec_twists`` seam ``DeployACT`` calls -- and require neither
ROS, Gazebo, torch, nor a GPU.

Run with::

    python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aic_example_policies.ros.chunk_ensemble import (
    DEFAULT_DECAY,
    BufferedChunk,
    ChunkEnsemble,
    select_exec_twists,
)


def _twist_chunk(value: float, k: int = 16, dim: int = 6) -> np.ndarray:
    """Return a ``(k, dim)`` chunk whose first component is ``value`` everywhere.

    Args:
        value: Value placed in component 0 (the ``vx`` closing velocity) of every
            step; other components are zero.
        k: Chunk length.
        dim: Action dimension.

    Returns:
        The constant ``(k, dim)`` chunk.
    """
    arr = np.zeros((k, dim), dtype=np.float64)
    arr[:, 0] = value
    return arr


class BufferedChunkTest(unittest.TestCase):
    """Tests for the :class:`BufferedChunk` record."""

    def test_coverage_window_inclusive(self) -> None:
        """A length-K chunk covers exactly [start, start+K-1] inclusive."""
        bc = BufferedChunk(10, np.zeros((4, 6)))
        self.assertEqual(bc.length, 4)
        self.assertEqual(bc.end_step, 13)
        self.assertFalse(bc.covers(9))
        self.assertTrue(bc.covers(10))
        self.assertTrue(bc.covers(13))
        self.assertFalse(bc.covers(14))

    def test_at_indexes_by_offset(self) -> None:
        """``at`` returns the row at ``step - start_step``."""
        chunk = np.arange(12, dtype=np.float64).reshape(4, 3)
        bc = BufferedChunk(100, chunk)
        np.testing.assert_array_equal(bc.at(100), chunk[0])
        np.testing.assert_array_equal(bc.at(102), chunk[2])

    def test_at_out_of_range_raises(self) -> None:
        """Indexing a step outside coverage raises IndexError."""
        bc = BufferedChunk(5, np.zeros((2, 6)))
        with self.assertRaises(IndexError):
            bc.at(7)


class WeightsTest(unittest.TestCase):
    """Tests for the exponential weighting math."""

    def test_weights_sum_to_one(self) -> None:
        """Weights are normalized regardless of count and decay."""
        for count in (1, 2, 4, 7):
            w = ChunkEnsemble.weights(count, 0.01)
            self.assertEqual(w.shape, (count,))
            self.assertAlmostEqual(float(w.sum()), 1.0, places=12)

    def test_weights_oldest_largest(self) -> None:
        """With positive decay the oldest (i=0) weight is the largest, decreasing."""
        w = ChunkEnsemble.weights(4, 0.01)
        self.assertTrue(np.all(np.diff(w) < 0.0), f"weights not decreasing: {w}")
        self.assertEqual(int(np.argmax(w)), 0)

    def test_weights_hand_computed_ln2(self) -> None:
        """decay=ln2 gives unnormalized [1, 1/2, 1/4] -> normalized [4/7, 2/7, 1/7]."""
        w = ChunkEnsemble.weights(3, math.log(2.0))
        np.testing.assert_allclose(w, [4 / 7, 2 / 7, 1 / 7], atol=1e-12)

    def test_weights_uniform_when_zero_decay(self) -> None:
        """decay=0 yields a plain uniform average over covering chunks."""
        w = ChunkEnsemble.weights(5, 0.0)
        np.testing.assert_allclose(w, np.full(5, 1 / 5), atol=1e-12)

    def test_weights_bad_count_raises(self) -> None:
        """A count below 1 raises ValueError."""
        with self.assertRaises(ValueError):
            ChunkEnsemble.weights(0, 0.01)

    def test_weights_negative_decay_raises(self) -> None:
        """A negative decay raises ValueError."""
        with self.assertRaises(ValueError):
            ChunkEnsemble.weights(3, -0.01)


class ChunkEnsembleQueryTest(unittest.TestCase):
    """Tests for buffering, querying, and pruning."""

    def test_single_chunk_is_passthrough(self) -> None:
        """One covering chunk returns its own prediction exactly (weight 1)."""
        ens = ChunkEnsemble(decay=0.01)
        chunk = np.arange(24, dtype=np.float64).reshape(4, 6)
        ens.add(0, chunk)
        for step in range(4):
            np.testing.assert_array_equal(ens.query(step), chunk[step])

    def test_two_chunk_average_hand_computed(self) -> None:
        """Two covering chunks combine with oldest-first weights [2/3, 1/3]."""
        ens = ChunkEnsemble(decay=math.log(2.0))
        # Oldest covers [0..3] predicting vx=1.0; newest covers [2..5] vx=4.0.
        ens.add(0, _twist_chunk(1.0, k=4))
        ens.add(2, _twist_chunk(4.0, k=4))
        # Step 2 is covered by both; weights [2/3 (oldest), 1/3 (newest)].
        got = ens.query(2)
        self.assertAlmostEqual(got[0], (2 / 3) * 1.0 + (1 / 3) * 4.0, places=12)

    def test_oldest_prediction_dominates(self) -> None:
        """With positive decay the ensemble sits nearer the oldest prediction."""
        ens = ChunkEnsemble(decay=0.5)
        ens.add(0, _twist_chunk(-0.05, k=4))  # oldest: strong closing velocity
        ens.add(2, _twist_chunk(0.0, k=4))    # newest: attractor (zero)
        got = ens.query(2)[0]
        self.assertLess(got, -0.025)  # closer to -0.05 than to 0.0
        self.assertGreater(got, -0.05)

    def test_zero_decay_is_plain_mean(self) -> None:
        """decay=0 averages covering chunks uniformly."""
        ens = ChunkEnsemble(decay=0.0)
        ens.add(0, _twist_chunk(-0.06, k=4))
        ens.add(2, _twist_chunk(0.0, k=4))
        self.assertAlmostEqual(ens.query(2)[0], -0.03, places=12)

    def test_covering_is_oldest_first(self) -> None:
        """``covering`` returns hits sorted by ascending start_step."""
        ens = ChunkEnsemble(decay=0.01)
        ens.add(4, _twist_chunk(2.0, k=8))
        ens.add(0, _twist_chunk(1.0, k=8))  # added later but older start_step
        hits = ens.covering(5)
        self.assertEqual([h.start_step for h in hits], [0, 4])

    def test_query_no_coverage_raises(self) -> None:
        """Querying a step no buffered chunk covers raises KeyError."""
        ens = ChunkEnsemble()
        ens.add(0, _twist_chunk(1.0, k=4))
        with self.assertRaises(KeyError):
            ens.query(99)

    def test_add_bad_shape_raises(self) -> None:
        """A non-(K, D) chunk raises ValueError."""
        ens = ChunkEnsemble()
        with self.assertRaises(ValueError):
            ens.add(0, np.zeros(6))  # 1-D, not (K, 6)

    def test_negative_decay_raises(self) -> None:
        """Constructing with negative decay raises ValueError."""
        with self.assertRaises(ValueError):
            ChunkEnsemble(decay=-1.0)

    def test_add_copies_input(self) -> None:
        """Mutating the caller's array after add does not corrupt the buffer."""
        ens = ChunkEnsemble()
        chunk = _twist_chunk(1.0, k=4)
        ens.add(0, chunk)
        chunk[:, 0] = 999.0
        np.testing.assert_array_equal(ens.query(0), [1.0, 0, 0, 0, 0, 0])


class PruneTest(unittest.TestCase):
    """Tests for buffer pruning."""

    def test_prune_drops_stale_keeps_covering(self) -> None:
        """Chunks whose coverage ends before keep_from_step are removed."""
        ens = ChunkEnsemble(decay=0.01)
        ens.add(0, _twist_chunk(1.0, k=4))   # covers [0, 3]
        ens.add(4, _twist_chunk(2.0, k=4))   # covers [4, 7]
        ens.add(8, _twist_chunk(3.0, k=4))   # covers [8, 11]
        removed = ens.prune(keep_from_step=8)
        self.assertEqual(removed, 2)
        self.assertEqual(len(ens), 1)
        # Only the chunk covering >= 8 survives.
        np.testing.assert_array_equal(ens.query(8), [3.0, 0, 0, 0, 0, 0])

    def test_prune_keeps_boundary_chunk(self) -> None:
        """A chunk whose end_step == keep_from_step is retained."""
        ens = ChunkEnsemble()
        ens.add(0, _twist_chunk(1.0, k=4))  # covers [0, 3]
        removed = ens.prune(keep_from_step=3)
        self.assertEqual(removed, 0)
        self.assertEqual(len(ens), 1)

    def test_clear_empties_buffer(self) -> None:
        """clear() drops every buffered chunk."""
        ens = ChunkEnsemble()
        ens.add(0, _twist_chunk(1.0, k=4))
        ens.clear()
        self.assertEqual(len(ens), 0)


class SelectExecTwistsTest(unittest.TestCase):
    """Tests for the ``DeployACT`` execution seam (disabled + enabled paths)."""

    def test_disabled_path_is_byte_identical_passthrough(self) -> None:
        """ensemble=None returns chunk[:exec_steps] unchanged (values and dtype)."""
        rng = np.random.default_rng(0)
        chunk = rng.normal(size=(16, 6)).astype(np.float32)  # DeployACT dtype
        out = select_exec_twists(None, chunk, start_step=40, exec_steps=4)
        np.testing.assert_array_equal(out, chunk[:4])
        self.assertEqual(out.dtype, chunk.dtype)

    def test_disabled_path_does_not_mutate_input(self) -> None:
        """The disabled passthrough leaves the input chunk untouched."""
        chunk = _twist_chunk(0.03, k=16)
        before = chunk.copy()
        select_exec_twists(None, chunk, start_step=0, exec_steps=4)
        np.testing.assert_array_equal(chunk, before)

    def test_bad_exec_steps_raises(self) -> None:
        """exec_steps < 1 raises ValueError on both paths."""
        chunk = _twist_chunk(1.0, k=8)
        with self.assertRaises(ValueError):
            select_exec_twists(None, chunk, 0, 0)
        with self.assertRaises(ValueError):
            select_exec_twists(ChunkEnsemble(), chunk, 0, 0)

    def test_first_cycle_equals_passthrough(self) -> None:
        """With a warm-start buffer, cycle 0 ensembling equals the raw chunk."""
        ens = ChunkEnsemble(decay=0.01)
        chunk = _twist_chunk(-0.05, k=16)
        out = select_exec_twists(ens, chunk, start_step=0, exec_steps=4)
        np.testing.assert_allclose(out, chunk[:4], atol=1e-12)

    def test_enabled_path_buffers_and_prunes(self) -> None:
        """During-cycle overlap reaches ceil(K/exec_steps); the buffer stays bounded.

        The receding horizon adds one chunk, queries this cycle's steps (when the
        overlap peaks at ceil(K/exec_steps)), then prunes chunks that cannot cover
        the next cycle -- so the post-prune buffer never grows unbounded.
        """
        ens = ChunkEnsemble(decay=0.01)
        k, exec_steps = 16, 4
        cap = math.ceil(k / exec_steps)
        max_overlap = 0
        for c in range(12):
            ens.add(c * exec_steps, _twist_chunk(-0.05, k=k))
            max_overlap = max(max_overlap, len(ens.covering(c * exec_steps)))
            ens.prune(c * exec_steps + exec_steps)
            self.assertLessEqual(len(ens), cap)
        self.assertEqual(max_overlap, cap)


class AttractorScenarioTest(unittest.TestCase):
    """The core mechanistic claim: older approaching chunks break the stall."""

    def test_older_chunk_injects_closing_velocity_that_decays(self) -> None:
        """Older approach chunks yield non-zero closing vel that decays as they expire.

        Models the last-inch attractor: up to cycle ``stall`` every chunk
        commands a constant closing velocity (vx=-0.05); from ``stall`` on the
        freshest chunk mode-averages to zero. The ensembled first executed twist
        must (a) be non-zero closing at stall onset -- the nudge the raw newest
        chunk (0) cannot provide -- and (b) decay monotonically to ~0 as the
        approaching chunks are pruned out of the buffer.
        """
        k, exec_steps, decay = 16, 4, DEFAULT_DECAY
        stall = 6  # buffer is fully warm well before this
        span = math.ceil(k / exec_steps)  # cycles an approach chunk survives

        ens = ChunkEnsemble(decay=decay)
        exec_vx: list[float] = []
        raw_vx: list[float] = []
        for c in range(stall + span + 2):
            vx = -0.05 if c < stall else 0.0
            raw_vx.append(vx)
            out = select_exec_twists(ens, _twist_chunk(vx, k=k), c * exec_steps, exec_steps)
            exec_vx.append(float(out[0, 0]))  # first executed step's vx

        # Contrast: the raw newest chunk offers zero closing velocity at stall.
        self.assertEqual(raw_vx[stall], 0.0)

        # (a) The ensemble injects a real closing nudge at stall onset.
        self.assertLess(exec_vx[stall], -1e-3)

        # (b) It decays monotonically toward zero over the next `span` cycles as
        #     the approaching chunks expire.
        tail = exec_vx[stall : stall + span + 1]
        self.assertTrue(
            all(a < b + 1e-12 for a, b in zip(tail, tail[1:])),
            f"closing velocity did not decay monotonically: {tail}",
        )
        self.assertAlmostEqual(exec_vx[stall + span], 0.0, places=9)

        # Cross-check the stall-onset value against an independent hand
        # computation: chunks from cycles [stall-3..stall] cover the first
        # executed step; the three oldest command -0.05, the newest 0.
        w = np.exp(-decay * np.arange(4)) / np.exp(-decay * np.arange(4)).sum()
        expected = -0.05 * (w[0] + w[1] + w[2])  # newest (w[3]) multiplies 0.0
        self.assertAlmostEqual(exec_vx[stall], expected, places=9)


if __name__ == "__main__":
    unittest.main()
