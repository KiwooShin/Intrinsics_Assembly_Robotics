#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""ROS-free assembly + normalization of the ACT-lite policy's proprioceptive state.

``DeployACT`` builds the policy's state vector from the current ``Observation``:
the 7-D TCP pose ``[x, y, z, qx, qy, qz, qw]`` and -- for a wrench-augmented
(13-D) checkpoint -- the live 6-D wrist wrench ``[fx, fy, fz, tx, ty, tz]``
appended in the exact order the training data recorded it
(``prepare_dataset.py`` samples ``/fts_broadcaster/wrench`` as force-then-torque).

The checkpoint records ``state_dim`` (7 or 13). A 7-D checkpoint predates the
wrench upgrade; its ``state_dim`` key may be absent, so callers default to 7 and
this module's ``assemble_state`` returns the pose unchanged -- byte-identical to
the legacy behavior. A 13-D checkpoint requires the wrench to be supplied.

This module deliberately imports only numpy (no ROS, torch, or GPU) so it is
unit-testable on any machine, mirroring ``chunk_ensemble.py``. ``DeployACT``
extracts the raw pose/wrench numbers from the ROS message and calls these
functions; the torch/normalization tensor plumbing stays in the node.
"""

from __future__ import annotations

import numpy as np

# State-width constants. The pose is a 3-D position + 4-D quaternion; the wrench
# is a 3-D force + 3-D torque appended after the pose for the 13-D variant.
POSE_DIM: int = 7
WRENCH_DIM: int = 6
POSE_WRENCH_DIM: int = POSE_DIM + WRENCH_DIM  # 13

# The two supported policy state dimensionalities.
SUPPORTED_STATE_DIMS: tuple[int, ...] = (POSE_DIM, POSE_WRENCH_DIM)


def assemble_state(
    pose: np.ndarray, wrench: np.ndarray | None, state_dim: int
) -> np.ndarray:
    """Assemble the raw (un-normalized) policy state for ``state_dim``.

    Args:
        pose: TCP pose shaped ``(7,)`` as ``[x, y, z, qx, qy, qz, qw]``.
        wrench: Wrist wrench shaped ``(6,)`` as ``[fx, fy, fz, tx, ty, tz]``, or
            ``None`` for a 7-D policy. Required when ``state_dim == 13``.
        state_dim: Target state width, ``7`` (pose only) or ``13`` (pose+wrench).

    Returns:
        A ``float32`` array shaped ``(state_dim,)``: the pose for ``state_dim==7``
        (byte-identical to the legacy path), or the pose with the wrench appended
        for ``state_dim==13``.

    Raises:
        ValueError: If ``pose`` is not ``(7,)``, ``state_dim`` is unsupported, or
            ``state_dim == 13`` without a valid ``(6,)`` ``wrench``.
    """
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape != (POSE_DIM,):
        raise ValueError(f"pose must have shape ({POSE_DIM},), got {pose.shape}")
    if state_dim == POSE_DIM:
        return pose
    if state_dim == POSE_WRENCH_DIM:
        if wrench is None:
            raise ValueError(
                "state_dim=13 requires a 6-D wrench; the observation's "
                "wrist_wrench was not supplied."
            )
        wrench = np.asarray(wrench, dtype=np.float32).reshape(-1)
        if wrench.shape != (WRENCH_DIM,):
            raise ValueError(
                f"wrench must have shape ({WRENCH_DIM},), got {wrench.shape}"
            )
        return np.concatenate([pose, wrench]).astype(np.float32)
    raise ValueError(
        f"unsupported state_dim {state_dim}; expected one of {SUPPORTED_STATE_DIMS}"
    )


def normalize_state(
    state: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return ``(state - mean) / std`` (matches the trainer's normalization).

    Args:
        state: Raw state shaped ``(state_dim,)``.
        mean: Per-dim mean shaped ``(state_dim,)`` (checkpoint ``smean``).
        std: Per-dim std shaped ``(state_dim,)`` (checkpoint ``sstd``, eps-floored).

    Returns:
        The normalized state, same shape as ``state``.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    state = np.asarray(state, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if not (state.shape == mean.shape == std.shape):
        raise ValueError(
            f"state/mean/std shape mismatch: {state.shape}, {mean.shape}, {std.shape}"
        )
    return (state - mean) / std


def denormalize_state(
    state_n: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Invert :func:`normalize_state`: ``state_n * std + mean``.

    Provided for round-trip verification and symmetry with the action
    de-normalization used at inference.

    Args:
        state_n: Normalized state shaped ``(state_dim,)``.
        mean: Per-dim mean shaped ``(state_dim,)``.
        std: Per-dim std shaped ``(state_dim,)``.

    Returns:
        The reconstructed raw state, same shape as ``state_n``.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    state_n = np.asarray(state_n, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if not (state_n.shape == mean.shape == std.shape):
        raise ValueError(
            f"state_n/mean/std shape mismatch: {state_n.shape}, {mean.shape}, "
            f"{std.shape}"
        )
    return state_n * std + mean
