#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""ACT-style temporal ensembling over overlapping receding-horizon twist chunks.

``DeployACT`` runs a receding-horizon loop: at inference cycle ``c`` it predicts
a ``K``-step twist chunk from a fresh observation, executes only the first
``exec_steps`` twists (integrated into absolute pose targets), then re-infers.
Because ``K > exec_steps`` the chunks *overlap* in absolute time: the chunk from
cycle ``c`` covers absolute frame steps ``[c*exec_steps, c*exec_steps + K - 1]``,
so any single absolute step is predicted by up to ``ceil(K / exec_steps)`` past
chunks.

Near the insertion port the deterministic ACT head mode-averages the demos'
bimodal "keep pushing in / stopped, done" action distribution toward *zero*
velocity, so the freshest chunk stalls the arm 5-8 cm short (the last-inch
fixed-point attractor documented in ``SESSION_REPORT.md`` 2026-07-19 FINAL).
Chunks predicted one or two inferences earlier -- from farther-back views that
still commanded a closing approach -- carry non-zero closing velocity for those
same near-port absolute steps. Temporal ensembling (ACT / ALOHA, Zhao et al.,
"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
arXiv:2304.13705) averages, at each absolute step, every buffered chunk's
prediction for that step, so the older approaching predictions inject the closing
velocity the freshest (zero) prediction has lost -- a mechanistic attempt to
break out of the attractor.

Weighting convention (matches the paper and Tony Zhao's reference
implementation): the predictions covering a given absolute step are ordered
*oldest-first*, index ``i = 0`` is the OLDEST prediction, and the weight is
``w_i = exp(-m * i)`` normalized to sum to 1. Thus the oldest prediction (made
from the farthest-back view, ``i = 0``) receives the *largest* weight, and a
*smaller* ``m`` incorporates new observations faster (all weights approach
uniform as ``m -> 0``); a *larger* ``m`` down-weights newer predictions and
leans harder on the older approaching ones. The ACT paper uses ``m = 0.01``.

This module is deliberately free of ROS, Gazebo, and torch imports so it can be
unit-tested on any machine. It operates on plain ``numpy`` arrays and integer
step indices; the action dimension ``D`` is generic (``6`` for base_link twists
in ``DeployACT``, but the math is dimension-agnostic).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# Default exponential decay rate ``m``. Matches the ACT paper (arXiv:2304.13705)
# and Tony Zhao's reference implementation. Overridable via ``AIC_ENSEMBLE_M``.
DEFAULT_DECAY = 0.01


@dataclasses.dataclass(frozen=True)
class BufferedChunk:
    """One buffered action-chunk prediction tagged with its absolute start step.

    Attributes:
        start_step: Absolute frame index of the chunk's first entry. A chunk of
            length ``K`` covers absolute steps ``[start_step, start_step+K-1]``.
        chunk: Predicted action chunk shaped ``(K, D)`` (e.g. ``(K, 6)`` twists),
            already denormalized to physical units.
    """

    start_step: int
    chunk: np.ndarray

    @property
    def length(self) -> int:
        """Number of steps ``K`` this chunk predicts."""
        return int(self.chunk.shape[0])

    @property
    def end_step(self) -> int:
        """Absolute index of the last step this chunk covers (inclusive)."""
        return self.start_step + self.length - 1

    def covers(self, step: int) -> bool:
        """Return whether this chunk has a prediction for absolute ``step``.

        Args:
            step: Absolute frame index to test.

        Returns:
            True if ``start_step <= step <= end_step``.
        """
        return self.start_step <= step <= self.end_step

    def at(self, step: int) -> np.ndarray:
        """Return this chunk's predicted action for absolute ``step``.

        Args:
            step: Absolute frame index; must satisfy :meth:`covers`.

        Returns:
            The ``(D,)`` action-vector prediction for ``step``.

        Raises:
            IndexError: If ``step`` is outside the chunk's coverage window.
        """
        if not self.covers(step):
            raise IndexError(
                f"step {step} outside chunk coverage "
                f"[{self.start_step}, {self.end_step}]"
            )
        return self.chunk[step - self.start_step]


class ChunkEnsemble:
    """Exponentially-weighted temporal ensemble over overlapping action chunks.

    Buffers recent ``(start_step, chunk)`` predictions and, for any queried
    absolute step, returns the exponentially-weighted average of every buffered
    chunk that covers that step. See the module docstring for the weighting
    convention (``w_i = exp(-m*i)``, ``i=0`` oldest, normalized).

    The buffer is a plain list appended in inference order (so it is naturally
    sorted oldest-first). It is expected to hold only ``ceil(K/exec_steps)``
    entries in steady state once :meth:`prune` is called each cycle, so the
    linear scans are negligible.
    """

    def __init__(self, decay: float = DEFAULT_DECAY) -> None:
        """Initialize an empty ensemble.

        Args:
            decay: Exponential weight decay rate ``m`` (>= 0). ``0`` yields a
                uniform average over covering chunks; larger values concentrate
                weight on the oldest covering chunk. Defaults to
                :data:`DEFAULT_DECAY`.

        Raises:
            ValueError: If ``decay`` is negative or not finite.
        """
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError(f"decay must be a finite float >= 0, got {decay!r}")
        self._decay = float(decay)
        self._buffer: list[BufferedChunk] = []

    @property
    def decay(self) -> float:
        """The exponential decay rate ``m``."""
        return self._decay

    def __len__(self) -> int:
        """Number of chunks currently buffered."""
        return len(self._buffer)

    @staticmethod
    def weights(count: int, decay: float) -> np.ndarray:
        """Return the normalized oldest-first exponential weights.

        Args:
            count: Number of covering predictions (>= 1).
            decay: Exponential decay rate ``m`` (>= 0).

        Returns:
            A ``(count,)`` array summing to 1 where entry ``i`` is
            ``exp(-decay*i) / sum_j exp(-decay*j)`` and ``i=0`` is the oldest
            prediction. With ``decay == 0`` every weight equals ``1/count``.

        Raises:
            ValueError: If ``count < 1`` or ``decay`` is negative/non-finite.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError(f"decay must be a finite float >= 0, got {decay!r}")
        idx = np.arange(count, dtype=np.float64)  # 0 == oldest prediction
        raw = np.exp(-decay * idx)
        return raw / raw.sum()

    def add(self, start_step: int, chunk: np.ndarray) -> None:
        """Buffer a freshly predicted chunk at its absolute start step.

        Args:
            start_step: Absolute frame index of the chunk's first entry.
            chunk: Predicted chunk shaped ``(K, D)`` with ``K >= 1``. Copied to a
                ``float64`` array so later external mutation cannot corrupt the
                buffer.

        Raises:
            ValueError: If ``chunk`` is not a 2-D ``(K, D)`` array with ``K>=1``.
        """
        arr = np.array(chunk, dtype=np.float64, copy=True)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError(
                f"chunk must have shape (K, D) with K>=1, got {arr.shape}"
            )
        self._buffer.append(BufferedChunk(int(start_step), arr))

    def covering(self, step: int) -> list[BufferedChunk]:
        """Return the buffered chunks covering ``step``, oldest-first.

        Args:
            step: Absolute frame index.

        Returns:
            Buffered chunks whose coverage window includes ``step``, ordered by
            ascending ``start_step`` (oldest prediction first). Empty if none.
        """
        hits = [bc for bc in self._buffer if bc.covers(step)]
        hits.sort(key=lambda bc: bc.start_step)
        return hits

    def query(self, step: int) -> np.ndarray:
        """Return the exp-weighted average prediction for absolute ``step``.

        Averages every buffered chunk that covers ``step`` using oldest-first
        exponential weights (see :meth:`weights`). A single covering chunk yields
        an exact passthrough of that chunk's prediction (weight 1).

        Args:
            step: Absolute frame index to resolve. Must be covered by at least
                one buffered chunk (the caller normally :meth:`add`\\ ed the fresh
                chunk covering it this cycle).

        Returns:
            The ``(D,)`` ensembled action vector for ``step``.

        Raises:
            KeyError: If no buffered chunk covers ``step``.
        """
        hits = self.covering(step)
        if not hits:
            raise KeyError(f"no buffered chunk covers step {step}")
        preds = np.stack([bc.at(step) for bc in hits])  # (n, D), oldest first
        w = self.weights(len(hits), self._decay)  # (n,)
        return (preds * w[:, None]).sum(axis=0)

    def prune(self, keep_from_step: int) -> int:
        """Drop chunks that can no longer cover any step ``>= keep_from_step``.

        A chunk whose ``end_step < keep_from_step`` will never be queried again
        (queries only move forward), so it is removed to bound the buffer.

        Args:
            keep_from_step: The lowest absolute step that may still be queried.

        Returns:
            The number of chunks removed.
        """
        before = len(self._buffer)
        self._buffer = [bc for bc in self._buffer if bc.end_step >= keep_from_step]
        return before - len(self._buffer)

    def clear(self) -> None:
        """Drop all buffered chunks (e.g. at the start of a new episode)."""
        self._buffer.clear()


def select_exec_twists(
    ensemble: ChunkEnsemble | None,
    chunk: np.ndarray,
    start_step: int,
    exec_steps: int,
) -> np.ndarray:
    """Return the ``exec_steps`` twists to execute this receding-horizon cycle.

    This is the single seam ``DeployACT`` calls each cycle, so the enabled and
    disabled behaviors live in one tested place.

    * When ``ensemble is None`` (temporal ensembling disabled) it is a pure
      passthrough returning ``chunk[:exec_steps]`` unchanged -- byte-identical to
      the non-ensembled receding-horizon path.
    * When an ``ensemble`` is supplied, the fresh ``chunk`` is buffered at
      ``start_step`` and each of the next ``exec_steps`` absolute steps
      (``start_step .. start_step+exec_steps-1``) is resolved to the exp-weighted
      average over every buffered chunk that covers it; chunks that can no longer
      cover the next cycle's steps are then pruned.

    Args:
        ensemble: The temporal ensemble, or ``None`` to disable ensembling.
        chunk: The freshly predicted chunk shaped ``(K, D)`` with ``K >=
            exec_steps``.
        start_step: Absolute frame index of ``chunk``'s first entry
            (``cycle * exec_steps`` in ``DeployACT``).
        exec_steps: Number of leading steps executed this cycle (>= 1).

    Returns:
        An ``(exec_steps, D)`` array of the twists to integrate and command.

    Raises:
        ValueError: If ``exec_steps < 1``.
    """
    if exec_steps < 1:
        raise ValueError(f"exec_steps must be >= 1, got {exec_steps}")
    if ensemble is None:
        return np.asarray(chunk)[:exec_steps]
    ensemble.add(start_step, chunk)
    out = np.stack(
        [ensemble.query(start_step + j) for j in range(exec_steps)]
    )
    # After executing steps [start_step, start_step+exec_steps-1], the next query
    # begins at start_step+exec_steps; anything not covering that is now stale.
    ensemble.prune(start_step + exec_steps)
    return out
