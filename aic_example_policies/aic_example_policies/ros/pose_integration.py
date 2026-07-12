#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Pure pose-integration math for velocity-chunk -> absolute-pose deployment.

``DeployACT`` runs a policy that emits a chunk of Cartesian *twists* (6-D
linear+angular velocity, expressed in the ``base_link`` frame, spaced one
training frame apart). To drive the arm under the impedance controller's
``MODE_POSITION`` path -- the same path the ``CheatCode`` oracle used to
generate the training data -- those twists must be integrated into a sequence
of absolute ``base_link`` pose targets, re-anchored to the robot's measured
pose at every inference (receding horizon).

This module holds only the numerical integration, deliberately free of ROS,
Gazebo, and torch imports so it can be unit-tested on any machine. Quaternions
use the ``[x, y, z, w]`` (Hamilton) convention throughout, matching
``geometry_msgs/Quaternion`` field order.
"""

from __future__ import annotations

import numpy as np


def quaternion_from_angular_velocity(omega: np.ndarray, dt: float) -> np.ndarray:
    """Return the incremental rotation quaternion for a constant angular velocity.

    Integrates a constant angular velocity ``omega`` (rad/s, expressed in the
    fixed/world frame) over ``dt`` seconds into a unit quaternion via the SO(3)
    exponential map.

    Args:
        omega: Angular velocity vector shaped ``(3,)`` in rad/s.
        dt: Integration timestep in seconds.

    Returns:
        The unit quaternion ``[x, y, z, w]`` representing the rotation over
        ``dt``. Reduces to the identity ``[0, 0, 0, 1]`` for a near-zero rotation.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega) * dt)
    if theta < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / np.linalg.norm(omega)
    half = 0.5 * theta
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``q1 * q2`` of two ``[x, y, z, w]`` quaternions.

    Args:
        q1: Left quaternion ``[x, y, z, w]``.
        q2: Right quaternion ``[x, y, z, w]``.

    Returns:
        The product quaternion ``[x, y, z, w]`` (not renormalized).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def integrate_twist_chunk(
    position: np.ndarray,
    quaternion: np.ndarray,
    twists: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate a chunk of base-frame twists into absolute pose targets.

    Starting from the current TCP pose, each twist ``[vx, vy, vz, wx, wy, wz]``
    is treated as constant over ``dt`` and applied in the fixed ``base_link``
    frame: the translation advances by ``v * dt`` (Euler) and the orientation is
    left-multiplied by the incremental rotation ``exp(omega * dt)`` (world-frame
    angular velocity), then renormalized. The pose after step ``i`` becomes the
    starting pose for step ``i + 1``, yielding one absolute target per twist.

    Args:
        position: Starting TCP position ``[x, y, z]`` in ``base_link`` (meters).
        quaternion: Starting TCP orientation ``[x, y, z, w]`` in ``base_link``.
        twists: Twist chunk shaped ``(n, 6)`` -- linear (0:3) and angular (3:6)
            velocity in ``base_link`` frame, m/s and rad/s.
        dt: Timestep between consecutive twists in seconds (the training frame
            period). Must be positive.

    Returns:
        An array shaped ``(n, 7)`` of absolute pose targets, each row
        ``[x, y, z, qx, qy, qz, qw]`` in ``base_link``.

    Raises:
        ValueError: If ``dt`` is not positive or ``twists`` is not ``(n, 6)``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    twists = np.asarray(twists, dtype=np.float64)
    if twists.ndim != 2 or twists.shape[1] != 6:
        raise ValueError(f"twists must have shape (n, 6), got {twists.shape}")

    pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        raise ValueError("quaternion has near-zero norm")
    quat = quat / norm

    out = np.empty((len(twists), 7), dtype=np.float64)
    for i, twist in enumerate(twists):
        pos = pos + twist[:3] * dt
        dq = quaternion_from_angular_velocity(twist[3:], dt)
        quat = quaternion_multiply(dq, quat)
        quat = quat / np.linalg.norm(quat)
        out[i, :3] = pos
        out[i, 3:] = quat
    return out


def expand_twists(twists: np.ndarray, substeps: int) -> np.ndarray:
    """Repeat each twist ``substeps`` times for finer-grained integration.

    Commanding position targets at the raw training frame period (~0.275 s)
    snaps the impedance reference in coarse jumps. Repeating each constant twist
    ``substeps`` times and integrating at ``dt / substeps`` produces a smoother,
    higher-rate pose stream (closer to the oracle's 20 Hz command cadence)
    without changing the trajectory the twists describe.

    Args:
        twists: Twist chunk shaped ``(n, 6)``.
        substeps: Number of sub-steps per twist. Must be >= 1.

    Returns:
        A twist array shaped ``(n * substeps, 6)``.

    Raises:
        ValueError: If ``substeps`` < 1.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    twists = np.asarray(twists, dtype=np.float64)
    return np.repeat(twists, substeps, axis=0)
