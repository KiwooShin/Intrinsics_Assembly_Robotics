#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Pure-numpy TCP<->port offset math for the port-bearing auxiliary head.

The ``port_aux`` auxiliary head predicts, at every frame, the vector from the
current tool-center-point (TCP) to the target port entrance -- the "remaining
vector to insertion". Labels are produced by *hindsight terminal-TCP
relabeling*: the seated terminal TCP pose of each successful demo is the static
target, and the per-frame label is the offset from that frame's TCP to the
target, expressed (by default) in the frame's own TCP frame. See
``docs/design_port_aux_head.md`` sections 1-2.

This module holds only the numerical geometry -- quaternion vector rotation, the
TCP-frame <-> base_link round trip, the robust terminal target, per-frame offset
labels, the approach-axis label, and the deploy-time prediction record with its
plausibility gate. It is deliberately free of ROS, Gazebo, and torch imports so
it unit-tests on any machine (CLAUDE.md section 2), reusing only
``pose_integration.quaternion_multiply`` for the Hamilton product.

Quaternions use the ``[x, y, z, w]`` (Hamilton) convention throughout, matching
``geometry_msgs/Quaternion`` field order, ``tcp_poses.npy`` column order
(``[x, y, z, qx, qy, qz, qw]``), and ``pose_integration.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from aic_example_policies.ros.pose_integration import quaternion_multiply

# Default number of trailing frames whose component-wise median is the robust
# terminal target position (design section 1.1, ``M = 5``).
DEFAULT_TERMINAL_FRAMES: int = 5
# Default number of trailing above-threshold displacements averaged for the
# base-frame approach-axis label (design section 2.4 / 1.1).
DEFAULT_AXIS_FRAMES: int = 5
# Default minimum per-step displacement magnitude (m) for an approach-axis
# sample; smaller near-seating moves are ignored so a stalled tail does not
# dilute the direction estimate.
DEFAULT_AXIS_MIN_DISPLACEMENT: float = 1e-4

_QUAT_MIN_NORM: float = 1e-9
_VEC_MIN_NORM: float = 1e-9


def quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate ``[-x, -y, -z, w]`` of a ``[x, y, z, w]`` quaternion.

    For a unit quaternion the conjugate is its inverse (the inverse rotation).

    Args:
        quat: Quaternion ``[x, y, z, w]``.

    Returns:
        The conjugate quaternion ``[-x, -y, -z, w]`` as a length-4 array.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def rotate_vector_by_quat(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Actively rotate a 3-vector by a quaternion: ``v' = R(q) v``.

    Computes ``q (0, v) q^-1`` via the Hamilton product, so ``v`` and the result
    are expressed in the same fixed frame and ``q`` maps the TCP frame's axes
    into that frame. Equivalently this maps a vector *from* the TCP frame *into*
    base_link when ``q`` is the TCP orientation in base_link.

    Args:
        quat: Rotation quaternion ``[x, y, z, w]`` (renormalized internally).
        vec: Vector ``[x, y, z]`` to rotate.

    Returns:
        The rotated vector ``[x, y, z]`` as a length-3 array.

    Raises:
        ValueError: If ``quat`` has near-zero norm.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < _QUAT_MIN_NORM:
        raise ValueError("quat has near-zero norm")
    q = q / norm
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    qv = np.array([v[0], v[1], v[2], 0.0])
    out = quaternion_multiply(quaternion_multiply(q, qv), quaternion_conjugate(q))
    return out[:3]


def rotate_vector_by_quat_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by the inverse of a quaternion: ``v' = R(q)^T v``.

    Maps a vector expressed in base_link *into* the TCP frame when ``q`` is the
    TCP orientation in base_link (the inverse of :func:`rotate_vector_by_quat`).

    Args:
        quat: Rotation quaternion ``[x, y, z, w]`` (renormalized internally).
        vec: Vector ``[x, y, z]`` to rotate by the inverse rotation.

    Returns:
        The inverse-rotated vector ``[x, y, z]`` as a length-3 array.

    Raises:
        ValueError: If ``quat`` has near-zero norm.
    """
    return rotate_vector_by_quat(quaternion_conjugate(quat), vec)


def rotate_vectors_by_quats(
    quats: np.ndarray, vecs: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """Batch-rotate vectors by per-row quaternions (vectorized ``R(q) v``).

    Uses the algebraically-equivalent ``v' = v + w*t + u x t`` with
    ``t = 2 (u x v)`` (``u`` the quaternion vector part, ``w`` the scalar part),
    matching :func:`rotate_vector_by_quat` row-for-row while avoiding a Python
    loop over the ~40k dataset frames.

    Args:
        quats: Quaternions shaped ``(n, 4)`` ``[x, y, z, w]`` (renormalized).
        vecs: Vectors shaped ``(n, 3)`` to rotate.
        inverse: When True apply each quaternion's inverse rotation (conjugate),
            i.e. ``R(q)^T v``.

    Returns:
        An ``(n, 3)`` array of rotated vectors.

    Raises:
        ValueError: If shapes are not ``(n, 4)`` / ``(n, 3)`` with equal ``n``.
    """
    q = np.asarray(quats, dtype=np.float64)
    v = np.asarray(vecs, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (n, 4), got {q.shape}")
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vecs must have shape (n, 3), got {v.shape}")
    if q.shape[0] != v.shape[0]:
        raise ValueError(
            f"quats ({q.shape[0]}) and vecs ({v.shape[0]}) row counts differ"
        )
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms < _QUAT_MIN_NORM, 1.0, norms)
    q = q / norms
    u = q[:, :3].copy()
    if inverse:
        u = -u
    w = q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def tcp_frame_offset(
    tcp_position: np.ndarray, tcp_quaternion: np.ndarray, target_position_base: np.ndarray
) -> np.ndarray:
    """Return the TCP-frame offset from a TCP pose to a base_link target point.

    ``offset_tcp = R(q)^T (target_base - tcp_pos)`` -- the vector from the TCP to
    the target, expressed in the TCP's own frame (design section 1.1).

    Args:
        tcp_position: TCP position ``[x, y, z]`` in base_link (m).
        tcp_quaternion: TCP orientation ``[x, y, z, w]`` in base_link.
        target_position_base: Target point ``[x, y, z]`` in base_link (m).

    Returns:
        The offset ``[x, y, z]`` in the TCP frame as a length-3 array.
    """
    pos = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    target = np.asarray(target_position_base, dtype=np.float64).reshape(3)
    return rotate_vector_by_quat_inverse(tcp_quaternion, target - pos)


def base_target_from_tcp_offset(
    tcp_position: np.ndarray, tcp_quaternion: np.ndarray, offset_tcp: np.ndarray
) -> np.ndarray:
    """Return the base_link target point implied by a TCP-frame offset.

    Inverse of :func:`tcp_frame_offset`:
    ``target_base = tcp_pos + R(q) offset_tcp``. This is the single rotation
    ``DeployACT`` applies at deploy time to turn a TCP-frame aux prediction into
    a base_link port target.

    Args:
        tcp_position: TCP position ``[x, y, z]`` in base_link (m).
        tcp_quaternion: TCP orientation ``[x, y, z, w]`` in base_link.
        offset_tcp: Offset ``[x, y, z]`` in the TCP frame (m).

    Returns:
        The target point ``[x, y, z]`` in base_link as a length-3 array.
    """
    pos = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    return pos + rotate_vector_by_quat(tcp_quaternion, offset_tcp)


def robust_terminal(
    poses: np.ndarray, terminal_frames: int = DEFAULT_TERMINAL_FRAMES
) -> tuple[np.ndarray, np.ndarray]:
    """Return the robust terminal target (median-of-tail position + end quat).

    The target position is the component-wise median of the last
    ``terminal_frames`` positions (robust to the last-inch tracking jitter); the
    orientation is the final frame's quaternion. Computed on the *full,
    untrimmed* pose array (design section 1.1 / 3.2 -- the seated tail is the
    target and must not be trimmed away before the target is read off).

    Args:
        poses: Pose array shaped ``(n, 7)`` ``[x, y, z, qx, qy, qz, qw]`` in
            base_link, in time order (``n >= 1``).
        terminal_frames: Number of trailing frames to median over; clamped to
            ``n``. Must be >= 1.

    Returns:
        A tuple ``(target_position, target_quaternion)`` of a length-3 base_link
        position and a length-4 ``[x, y, z, w]`` quaternion.

    Raises:
        ValueError: If ``poses`` is not ``(n, 7)`` with ``n >= 1`` or
            ``terminal_frames`` < 1.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (n, 7), got {p.shape}")
    if p.shape[0] < 1:
        raise ValueError("poses must have at least one frame")
    if terminal_frames < 1:
        raise ValueError(f"terminal_frames must be >= 1, got {terminal_frames}")
    m = min(int(terminal_frames), p.shape[0])
    target_position = np.median(p[-m:, :3], axis=0)
    target_quaternion = p[-1, 3:].copy()
    return target_position, target_quaternion


def per_frame_tcp_offsets(
    poses: np.ndarray, target_position_base: np.ndarray, frame: str = "tcp"
) -> np.ndarray:
    """Return per-frame offset labels from each TCP pose to the static target.

    For each frame ``t`` the label is the vector from that frame's TCP to the
    (episode-static) target, expressed either in the frame's own TCP frame
    (``frame='tcp'``, ``R(q_t)^T (target - pos_t)``) or directly in base_link
    (``frame='base'``, ``target - pos_t``). At the terminal frame the label is
    ~0; along the approach it shrinks toward 0 -- the "remaining vector to
    insertion" at every frame.

    Args:
        poses: Pose array shaped ``(n, 7)`` ``[x, y, z, qx, qy, qz, qw]``.
        target_position_base: Static target point ``[x, y, z]`` in base_link (m).
        frame: ``'tcp'`` (default) or ``'base'`` -- the label frame.

    Returns:
        An ``(n, 3)`` array of per-frame offset labels (m).

    Raises:
        ValueError: If ``poses`` is not ``(n, 7)`` or ``frame`` is unknown.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (n, 7), got {p.shape}")
    if frame not in ("tcp", "base"):
        raise ValueError(f"frame must be 'tcp' or 'base', got {frame!r}")
    target = np.asarray(target_position_base, dtype=np.float64).reshape(3)
    delta = target[None, :] - p[:, :3]  # (n, 3) base_link
    if frame == "base":
        return delta
    return rotate_vectors_by_quats(p[:, 3:], delta, inverse=True)


def base_approach_axis(
    poses: np.ndarray,
    axis_frames: int = DEFAULT_AXIS_FRAMES,
    min_displacement: float = DEFAULT_AXIS_MIN_DISPLACEMENT,
) -> np.ndarray | None:
    """Return the base_link approach-axis unit vector, or None if undefined.

    ``a_hat = normalize(mean of the last ``axis_frames`` above-threshold TCP
    displacements)`` -- the direction the TCP was travelling into the port just
    before seating (design section 1.1 / 2.4). Sub-threshold near-seating moves
    are excluded so the stalled tail does not dilute the direction.

    Args:
        poses: Pose array shaped ``(n, 7)``.
        axis_frames: Number of trailing above-threshold displacements to average.
            Must be >= 1.
        min_displacement: Minimum per-step displacement magnitude (m) recorded.
            Must be >= 0.

    Returns:
        The unit approach axis ``[x, y, z]`` in base_link, or None when no
        above-threshold displacement exists or their mean cancels to ~0.

    Raises:
        ValueError: If ``poses`` is not ``(n, 7)``, ``axis_frames`` < 1, or
            ``min_displacement`` < 0.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (n, 7), got {p.shape}")
    if axis_frames < 1:
        raise ValueError(f"axis_frames must be >= 1, got {axis_frames}")
    if min_displacement < 0.0:
        raise ValueError(f"min_displacement must be >= 0, got {min_displacement}")
    if p.shape[0] < 2:
        return None
    disp = np.diff(p[:, :3], axis=0)  # (n-1, 3)
    mags = np.linalg.norm(disp, axis=1)
    moving = disp[mags >= min_displacement]
    if moving.shape[0] == 0:
        return None
    mean = np.mean(moving[-int(axis_frames):], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < _VEC_MIN_NORM:
        return None
    return mean / norm


def per_frame_axis_labels(
    poses: np.ndarray, axis_base: np.ndarray | None, frame: str = "tcp"
) -> np.ndarray:
    """Return per-frame approach-axis labels in the requested frame.

    The base_link approach axis is a single episode-static unit vector; this
    expresses it per-frame either in each frame's TCP frame (``frame='tcp'``,
    ``R(q_t)^T a_hat``) or directly in base_link (``frame='base'``, broadcast).
    When ``axis_base`` is None (undefined axis) all-zero labels are returned so
    the caller can still stack a fixed-width block (the frames stay valid on the
    offset channels; the axis channels simply carry no signal).

    Args:
        poses: Pose array shaped ``(n, 7)``.
        axis_base: Unit approach axis ``[x, y, z]`` in base_link, or None.
        frame: ``'tcp'`` (default) or ``'base'``.

    Returns:
        An ``(n, 3)`` array of per-frame axis labels.

    Raises:
        ValueError: If ``poses`` is not ``(n, 7)`` or ``frame`` is unknown.
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must have shape (n, 7), got {p.shape}")
    if frame not in ("tcp", "base"):
        raise ValueError(f"frame must be 'tcp' or 'base', got {frame!r}")
    n = p.shape[0]
    if axis_base is None:
        return np.zeros((n, 3), dtype=np.float64)
    a = np.asarray(axis_base, dtype=np.float64).reshape(3)
    if frame == "base":
        return np.tile(a, (n, 1))
    return rotate_vectors_by_quats(p[:, 3:], np.tile(a, (n, 1)), inverse=True)


@dataclasses.dataclass(frozen=True)
class PortOffsetPrediction:
    """A deploy-time port-bearing prediction resolved into base_link.

    Attributes:
        offset_base: TCP->target offset ``[x, y, z]`` in base_link (m).
        target_base: The implied target point ``[x, y, z]`` in base_link (m),
            i.e. ``tcp_pos + offset_base``.
        magnitude: ``|offset_base|`` (m) -- the remaining distance to the port.
        axis_base: Unit approach axis in base_link (only when the head predicts a
            6-D offset+axis), else None.
    """

    offset_base: np.ndarray
    target_base: np.ndarray
    magnitude: float
    axis_base: np.ndarray | None = None

    def plausible(self, min_magnitude: float, max_magnitude: float) -> bool:
        """Whether the prediction passes the magnitude plausibility gate.

        Args:
            min_magnitude: Lower bound (m); a smaller offset is rejected (already
                at/inside the port or a degenerate ~0 prediction).
            max_magnitude: Upper bound (m); a larger offset is rejected (a
                wrong-port lock or an out-of-range regression).

        Returns:
            True iff ``offset_base``/``magnitude`` are finite and
            ``min_magnitude <= magnitude <= max_magnitude``.
        """
        if not np.all(np.isfinite(self.offset_base)):
            return False
        if not np.isfinite(self.magnitude):
            return False
        return float(min_magnitude) <= self.magnitude <= float(max_magnitude)


def predict_from_aux(
    tcp_position: np.ndarray,
    tcp_quaternion: np.ndarray,
    aux_vector: np.ndarray,
    frame: str = "tcp",
) -> PortOffsetPrediction:
    """Resolve a de-normalized aux-head output into a base_link prediction.

    The aux head emits a 3-D offset (or 6-D offset+axis) in ``frame``; this
    rotates it into base_link (a single live-quaternion rotation when
    ``frame='tcp'``) and forms the target point and magnitude.

    Args:
        tcp_position: Live TCP position ``[x, y, z]`` in base_link (m).
        tcp_quaternion: Live TCP orientation ``[x, y, z, w]`` in base_link.
        aux_vector: De-normalized aux output, length 3 (offset) or 6
            (offset + axis), in ``frame``.
        frame: The frame the aux output is expressed in: ``'tcp'`` or ``'base'``.

    Returns:
        The resolved :class:`PortOffsetPrediction`.

    Raises:
        ValueError: If ``aux_vector`` is not length 3 or 6, or ``frame`` is
            unknown.
    """
    if frame not in ("tcp", "base"):
        raise ValueError(f"frame must be 'tcp' or 'base', got {frame!r}")
    aux = np.asarray(aux_vector, dtype=np.float64).reshape(-1)
    if aux.shape[0] not in (3, 6):
        raise ValueError(f"aux_vector must have length 3 or 6, got {aux.shape[0]}")
    pos = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    offset = aux[:3]
    if frame == "tcp":
        offset_base = rotate_vector_by_quat(tcp_quaternion, offset)
    else:
        offset_base = offset
    axis_base: np.ndarray | None = None
    if aux.shape[0] == 6:
        axis = aux[3:6]
        axis_b = (
            rotate_vector_by_quat(tcp_quaternion, axis) if frame == "tcp" else axis
        )
        norm = float(np.linalg.norm(axis_b))
        axis_base = axis_b / norm if norm >= _VEC_MIN_NORM else axis_b
    magnitude = float(np.linalg.norm(offset_base))
    return PortOffsetPrediction(
        offset_base=offset_base,
        target_base=pos + offset_base,
        magnitude=magnitude,
        axis_base=axis_base,
    )
