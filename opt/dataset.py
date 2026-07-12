"""Pure numpy data helpers shared by the trainer and sweep harness.

These functions reproduce train_v2.py's action-chunk indexing, per-dimension
normalization, and episode-level train/val split as pure array math so they can
be unit-tested CPU-only (no torch, ROS, or GPU). The heavy GPU tensor loading
itself stays in train_v2.load_all, which these helpers mirror.
"""

from __future__ import annotations

import numpy as np


def build_action_chunks(vel: np.ndarray, k: int) -> np.ndarray:
    """Build overlapping future-action chunks with edge clamping.

    For frame ``i`` the chunk is ``vel[i], vel[i+1], ..., vel[i+k-1]`` with
    indices clamped to the last frame (matching train_v2.load_all).

    Args:
        vel: Per-frame action array of shape (n, d).
        k: Chunk length (>= 1).

    Returns:
        Array of shape (n, k, d).

    Raises:
        ValueError: If ``k`` < 1 or ``vel`` is not 2-D.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if vel.ndim != 2:
        raise ValueError(f"vel must be 2-D (n, d), got shape {vel.shape}")
    n = vel.shape[0]
    idx = np.clip(np.arange(n)[:, None] + np.arange(k)[None, :], 0, n - 1)
    return vel[idx]


def normalization_stats(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and (eps-floored) std for normalization.

    Args:
        x: Array of shape (n, d).
        eps: Value added to the std to avoid divide-by-zero, matching train_v2.

    Returns:
        Tuple ``(mean, std)`` each of shape (d,).

    Raises:
        ValueError: If ``x`` is not 2-D.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d), got shape {x.shape}")
    return x.mean(0), x.std(0) + eps


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply ``(x - mean) / std`` broadcasting over the last axis."""
    return (x - mean) / std


def episode_split_masks(
    ep_ids: np.ndarray, val_ids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean train/val masks from a per-frame episode-id array.

    Args:
        ep_ids: Per-frame episode-id array of shape (n,).
        val_ids: Episode ids to hold out for validation.

    Returns:
        Tuple ``(train_mask, val_mask)`` boolean arrays of shape (n,). When
        ``val_ids`` is empty the val mask equals the train mask (overfit mode).

    Raises:
        ValueError: If ``ep_ids`` is not 1-D.
    """
    if ep_ids.ndim != 1:
        raise ValueError(f"ep_ids must be 1-D, got shape {ep_ids.shape}")
    if not val_ids:
        allm = np.ones_like(ep_ids, dtype=bool)
        return allm, allm
    val_mask = np.isin(ep_ids, np.array(sorted(val_ids)))
    return ~val_mask, val_mask


def first_action_l1_meters(
    pred_norm: np.ndarray, target_norm: np.ndarray, act_std: np.ndarray
) -> float:
    """Mean un-normalized L1 error of the first predicted action (m/s).

    Reproduces train_v2's reported metric: take the first chunk step, undo the
    action normalization by multiplying the absolute error by ``act_std``, then
    average over action dims and frames.

    Args:
        pred_norm: Normalized predictions of shape (n, k, d).
        target_norm: Normalized targets of shape (n, k, d).
        act_std: Per-dim action std of shape (d,) used to de-normalize.

    Returns:
        Mean first-action absolute error in physical units (m/s for twists).

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if pred_norm.shape != target_norm.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_norm.shape} vs {target_norm.shape}"
        )
    if pred_norm.ndim != 3:
        raise ValueError(f"expected (n, k, d), got {pred_norm.shape}")
    err = np.abs(pred_norm[:, 0] - target_norm[:, 0]) * act_std
    return float(err.mean(1).mean())
