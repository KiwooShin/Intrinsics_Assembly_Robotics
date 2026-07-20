"""Privileged-DAgger port-localization relabeling of DEPLOY-policy rollouts.

The deployed insertion policy reaches the port vicinity then *stalls* 13-61 mm
to the side of a 1-2.5 mm bore (``SESSION_REPORT.md`` Cycle 6). The port-bearing
auxiliary head that should re-center it was trained on the *oracle's* approach
views and does not transfer to the policy's own stall distribution (aux val
0.86 cm -> 15-61 mm at deploy: a covariate shift). This module closes that shift
offline: it takes a bag recorded while the DEPLOY policy ran in a
``ground_truth:=true`` sim, reads the TRUE port entrance transform from
``/scoring/tf`` (+ ``/tf`` / ``/tf_static``), and relabels the policy's own stall
states with the true TCP->port offset -- producing training episodes for
retraining the aux head on the states it actually visits.

The label is compliant with the eval rules exactly as the CheatCode teacher is:
the port TF is a *training-time* privileged label (available only under
``ground_truth:=true``); the retrained head still reads only eval-legal
RGB/TCP/wrench at deploy.

Design (CLAUDE.md section 2 -- pure logic factored from I/O):

  * The numerical core -- quaternion/homogeneous-transform math, the TF-forest
    resolver that composes ``base_link -> {port}_link_entrance``, the stall-window
    selector matching ``guarded_descent.StallDetector``, the array slicer, and the
    label schema -- is pure numpy and unit-tests with no ROS/Gazebo/GPU.
  * The bag-reading shell (:func:`read_port_target_from_bag`, :func:`relabel_bag`)
    lazily imports ``rosbags`` and reuses ``prepare_dataset.process_bag`` to build
    the standard episode arrays, so the two never diverge.

The per-frame offset is computed with the SAME
``aic_example_policies.ros.port_offset.per_frame_tcp_offsets`` that
``opt.port_labels`` / ``DeployACT`` use, so the emitted labels match what
``opt/train_v3.py --port-aux --aux-frame tcp`` expects. Each emitted episode also
carries a ``port_target.npy`` (the true port position in base_link); the small
additive hook in ``opt.port_labels.build_episode_label`` reads it as the target
instead of the (invalid, non-seating) ``robust_terminal`` pose. See the module
docstring of ``opt/port_labels.py`` and ``docs/design_port_aux_head.md`` section
1.2 (the "true privileged port-entrance TF" fast-follow this implements).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

import numpy as np

# Make the ROS-free offset geometry and the opt helpers importable when this
# module is run/imported from the repo root (mirrors opt/port_labels.py). The
# package root is ``<repo>/aic_example_policies``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _REPO_ROOT / "aic_example_policies"
for _p in (str(_PKG_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aic_example_policies.ros import port_offset  # noqa: E402
from opt import episode_prep  # noqa: E402

_LOG = logging.getLogger("dagger_relabel")

# --- Stall definition (mirrors guarded_descent.StallDetector defaults) ---------
# A stall fires when the TCP linear speed stays below DEFAULT_SPEED_THRESHOLD for
# a continuous DEFAULT_STALL_WINDOW_S seconds, after a DEFAULT_MIN_RUNTIME_S grace
# period (episode-relative here: the converted episode is already trimmed to the
# task window, so its first frame ~ policy activation).
DEFAULT_SPEED_THRESHOLD: float = 0.01  # m/s
DEFAULT_STALL_WINDOW_S: float = 3.0  # s of sustained low speed
DEFAULT_MIN_RUNTIME_S: float = 15.0  # s startup grace
# Seconds of the deceleration-into-stall approach kept before the stall onset, so
# the head also sees the low-speed pre-stall frames the deploy handoff queries.
DEFAULT_STALL_LOOKBACK_S: float = 2.0
# Minimum frames an emitted stall window must contain.
DEFAULT_MIN_WINDOW_FRAMES: int = 8
# When no stall is detected, fall back to keeping this many terminal frames.
DEFAULT_FALLBACK_FRAMES: int = 24

# Suffix of a port-entrance TF child frame (ScoringTier2::PortEntranceTfName()).
ENTRANCE_SUFFIX: str = "_link_entrance"
# Default base frame the label is expressed relative to (matches tcp_poses.npy).
DEFAULT_BASE_FRAME: str = "base_link"

# On-disk artifacts written into each emitted episode dir.
PORT_TARGET_NAME: str = "port_target.npy"  # (3,) true port position in base_link
PORT_OFFSETS_NAME: str = "port_offsets.npy"  # (N, aux_dim) per-frame labels (m)
# Per-frame ``.npy`` arrays produced by prepare_dataset that are sliced to the
# stall window (all share the frame axis 0).
FRAME_ARRAY_NAMES: tuple[str, ...] = (
    "center_images.npy",
    "left_images.npy",
    "right_images.npy",
    "tcp_poses.npy",
    "tcp_velocities.npy",
    "timestamps.npy",
    "wrenches.npy",
    "joint_positions.npy",
)

_QUAT_MIN_NORM: float = 1e-9

# Topics carrying transforms in a deploy-rollout bag (union feeds the forest).
TF_TOPICS: tuple[str, ...] = (
    "/tf",
    "/tf_static",
    "/scoring/tf",
    "/scoring/tf_static",
)


# =============================================================================
# Pure transform math (unit-tested, no ROS)
# =============================================================================
def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Return the 3x3 active rotation matrix of a ``[x, y, z, w]`` quaternion.

    Matches :func:`aic_example_policies.ros.port_offset.rotate_vector_by_quat`
    (i.e. ``R(q) v`` maps a vector from the quaternion's frame into the parent
    frame). The quaternion is renormalized internally.

    Args:
        quat: Quaternion ``[x, y, z, w]``.

    Returns:
        The ``(3, 3)`` rotation matrix.

    Raises:
        ValueError: If ``quat`` has near-zero norm.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < _QUAT_MIN_NORM:
        raise ValueError("quat has near-zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def homogeneous_transform(translation: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build the ``4x4`` homogeneous transform of a translation + quaternion.

    The result is ``T_parent_from_child`` for a TF whose ``translation``/``quat``
    give the pose of ``child`` in ``parent`` (a point ``p_child`` maps to
    ``p_parent = R q . p_child + t``).

    Args:
        translation: Translation ``[x, y, z]`` (m).
        quat: Rotation quaternion ``[x, y, z, w]``.

    Returns:
        The ``(4, 4)`` homogeneous transform.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rotation_matrix(quat)
    mat[:3, 3] = t
    return mat


def invert_homogeneous(mat: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid ``4x4`` homogeneous transform.

    Uses the closed form ``[[R^T, -R^T t], [0, 1]]`` (valid because ``R`` is
    orthonormal), avoiding a general matrix inverse.

    Args:
        mat: A ``(4, 4)`` rigid homogeneous transform.

    Returns:
        The inverse ``(4, 4)`` transform.

    Raises:
        ValueError: If ``mat`` is not ``(4, 4)``.
    """
    m = np.asarray(mat, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"mat must be (4, 4), got {m.shape}")
    rot = m[:3, :3]
    trans = m[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


@dataclasses.dataclass
class TransformForest:
    """A static TF tree: ``child_frame -> (parent_frame, T_parent_from_child)``.

    Transforms are accumulated by :meth:`add_transform`; the last one wins for a
    given child (base_link, the task board, and the port frames are all static in
    the sim, so a single representative snapshot resolves them exactly). The
    resolver composes edge chains and finds the lowest common ancestor, so it does
    not require every frame to share one global root.

    Attributes:
        edges: Map from a child frame name to ``(parent, T_parent_from_child)``.
    """

    edges: dict[str, tuple[str, np.ndarray]] = dataclasses.field(default_factory=dict)

    def add_transform(
        self, parent: str, child: str, translation: np.ndarray, quat: np.ndarray
    ) -> None:
        """Insert (or overwrite) the ``parent -> child`` edge.

        Args:
            parent: Parent frame id (``header.frame_id``).
            child: Child frame id (``child_frame_id``).
            translation: Child origin in parent ``[x, y, z]`` (m).
            quat: Child orientation in parent ``[x, y, z, w]``.
        """
        self.edges[child] = (parent, homogeneous_transform(translation, quat))

    def frame_ids(self) -> set[str]:
        """Return every frame name that appears as a parent or a child.

        Returns:
            The set of all frame ids known to the forest.
        """
        names: set[str] = set()
        for child, (parent, _) in self.edges.items():
            names.add(child)
            names.add(parent)
        return names

    def _chain_to_root(self, frame: str) -> list[str]:
        """Return ``[frame, parent, ..., root]`` walking parent edges upward.

        Args:
            frame: Frame to walk up from.

        Returns:
            The ordered list of frames from ``frame`` to its root (a frame with no
            parent edge).

        Raises:
            ValueError: If a cycle is detected among the parent edges.
        """
        chain: list[str] = [frame]
        seen: set[str] = {frame}
        cur = frame
        while cur in self.edges:
            parent = self.edges[cur][0]
            if parent in seen:
                raise ValueError(f"cycle in TF tree at frame {parent!r}")
            chain.append(parent)
            seen.add(parent)
            cur = parent
        return chain

    def _transform_up_to(self, frame: str, ancestor: str) -> np.ndarray:
        """Return ``T_ancestor_from_frame`` composing edges ``frame -> ancestor``.

        Args:
            frame: Descendant frame.
            ancestor: An ancestor of ``frame`` (may equal ``frame``).

        Returns:
            The ``(4, 4)`` transform mapping a point in ``frame`` into ``ancestor``.

        Raises:
            ValueError: If ``ancestor`` is not reached walking up from ``frame``.
        """
        mat = np.eye(4, dtype=np.float64)
        cur = frame
        while cur != ancestor:
            if cur not in self.edges:
                raise ValueError(f"{ancestor!r} is not an ancestor of {frame!r}")
            parent, t_parent_from_cur = self.edges[cur]
            mat = t_parent_from_cur @ mat
            cur = parent
        return mat

    def resolve(self, target_frame: str, source_frame: str) -> np.ndarray:
        """Return ``T_target_from_source`` (pose of ``source`` in ``target``).

        Args:
            target_frame: Frame the result is expressed in.
            source_frame: Frame whose pose is resolved.

        Returns:
            The ``(4, 4)`` homogeneous transform mapping a point in
            ``source_frame`` into ``target_frame``.

        Raises:
            ValueError: If either frame is unknown or they share no common
                ancestor (disconnected trees).
        """
        known = self.frame_ids()
        for name in (target_frame, source_frame):
            if name not in known:
                raise ValueError(f"frame {name!r} not present in the TF forest")
        target_chain = self._chain_to_root(target_frame)
        source_chain = self._chain_to_root(source_frame)
        target_set = set(target_chain)
        lca: str | None = next((f for f in source_chain if f in target_set), None)
        if lca is None:
            raise ValueError(
                f"{target_frame!r} and {source_frame!r} have no common ancestor "
                f"(roots {target_chain[-1]!r} vs {source_chain[-1]!r})"
            )
        t_lca_from_source = self._transform_up_to(source_frame, lca)
        t_lca_from_target = self._transform_up_to(target_frame, lca)
        return invert_homogeneous(t_lca_from_target) @ t_lca_from_source

    def position_of(
        self, target_frame: str, source_frame: str
    ) -> np.ndarray:
        """Return the origin of ``source_frame`` expressed in ``target_frame``.

        Args:
            target_frame: Frame the position is expressed in (e.g. ``base_link``).
            source_frame: Frame whose origin is located (e.g. the port entrance).

        Returns:
            The origin ``[x, y, z]`` (m) as a length-3 array.
        """
        return self.resolve(target_frame, source_frame)[:3, 3]


def select_entrance_frame(
    frame_ids: set[str] | list[str],
    *,
    port_name: str | None = None,
    explicit: str | None = None,
    prefer: set[str] | None = None,
) -> str:
    """Choose the target port-entrance frame from the frames present in a bag.

    Resolution order: an ``explicit`` frame (validated to exist) wins; else the
    unique frame ending in ``{port_name}{ENTRANCE_SUFFIX}`` when ``port_name`` is
    given; else the unique frame ending in :data:`ENTRANCE_SUFFIX`. When several
    entrance frames exist (distractor NIC/SC ports publish their own on ``/tf``),
    the candidates are first narrowed to ``prefer`` -- the frames the scoring node
    scopes to the TASK's connection on ``/scoring/tf`` -- which normally isolates
    the single target port. Genuine ambiguity raises so the operator passes
    ``--port-frame`` / ``--port-name`` rather than silently mislabeling.

    Args:
        frame_ids: All frame names present in the TF forest.
        port_name: Optional port name (e.g. ``sfp_port_0``) to disambiguate when
            several entrance frames are present.
        explicit: Optional fully-qualified entrance frame that overrides detection.
        prefer: Optional frames to prefer when auto-detection is ambiguous (the
            target-scoped ``/scoring/tf`` child frames).

    Returns:
        The chosen entrance frame name.

    Raises:
        ValueError: If no candidate exists, an ``explicit``/``port_name`` request
            matches nothing, or detection is ambiguous.
    """
    ids = set(frame_ids)
    if explicit:
        if explicit not in ids:
            raise ValueError(f"explicit port frame {explicit!r} not present in bag")
        return explicit
    entrances = sorted(f for f in ids if f.endswith(ENTRANCE_SUFFIX))
    if port_name:
        suffix = f"{port_name}{ENTRANCE_SUFFIX}"
        matches = [f for f in entrances if f.endswith(suffix) or f == suffix]
        if not matches:
            raise ValueError(
                f"no entrance frame for port_name {port_name!r}; "
                f"candidates: {entrances}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"port_name {port_name!r} is ambiguous among {matches}; "
                "pass --port-frame"
            )
        return matches[0]
    if not entrances:
        raise ValueError(
            f"no '*{ENTRANCE_SUFFIX}' frame found; is this a ground_truth bag?"
        )
    if len(entrances) > 1 and prefer:
        scoped = [f for f in entrances if f in prefer]
        if len(scoped) == 1:
            return scoped[0]
        if len(scoped) > 1:
            entrances = scoped  # still ambiguous -> report the scoped subset
    if len(entrances) > 1:
        raise ValueError(
            f"multiple entrance frames {entrances}; pass --port-name or --port-frame"
        )
    return entrances[0]


# =============================================================================
# Offset labels (reuse the deploy/train offset geometry verbatim)
# =============================================================================
def port_offset_labels(
    tcp_poses: np.ndarray, port_position_base: np.ndarray, frame: str = "tcp"
) -> np.ndarray:
    """Return per-frame TCP->port offset labels against a true port position.

    Thin wrapper over
    :func:`aic_example_policies.ros.port_offset.per_frame_tcp_offsets` so the
    DAgger labels are computed by the exact function ``opt.port_labels`` and
    ``DeployACT`` use -- the label frame convention can never drift. For a single
    frame this equals ``port_offset.tcp_frame_offset(pos, quat, port)``.

    Args:
        tcp_poses: Pose array ``(n, 7)`` ``[x, y, z, qx, qy, qz, qw]`` in base_link.
        port_position_base: True port position ``[x, y, z]`` in base_link (m).
        frame: ``'tcp'`` (default) or ``'base'`` -- the label frame.

    Returns:
        An ``(n, 3)`` array of per-frame offset labels (m).
    """
    return port_offset.per_frame_tcp_offsets(
        tcp_poses, port_position_base, frame=frame
    )


# =============================================================================
# Stall-window selection (matches StallDetector semantics)
# =============================================================================
def stall_window(
    speed: np.ndarray,
    timestamps: np.ndarray,
    *,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    stall_window_s: float = DEFAULT_STALL_WINDOW_S,
    min_runtime_s: float = DEFAULT_MIN_RUNTIME_S,
    lookback_s: float = DEFAULT_STALL_LOOKBACK_S,
    min_frames: int = DEFAULT_MIN_WINDOW_FRAMES,
) -> tuple[int, int] | None:
    """Return the ``[start, stop)`` stall window, or ``None`` if no stall fired.

    Reproduces ``guarded_descent.StallDetector`` on the recorded trace: it finds
    the first continuous low-speed span (``speed < speed_threshold``) that has
    lasted at least ``stall_window_s`` seconds at a point at least
    ``min_runtime_s`` seconds into the episode. The window starts a ``lookback_s``
    approach before that span began (so the deceleration-into-stall frames the
    deploy handoff queries are included) and runs to the end of the episode (the
    stalled dwell + any guarded-descent re-tries -- all deploy states we want the
    aux head to learn). Timestamps are used for the durations exactly as the live
    detector uses wall-clock time; ``min_runtime_s`` is measured from the first
    frame (the converted episode is already trimmed to the task window).

    Args:
        speed: Per-frame non-negative linear speed ``(n,)`` (m/s), e.g. from
            :func:`opt.episode_prep.frame_speed`.
        timestamps: Per-frame capture times ``(n,)`` (s), same order as ``speed``.
        speed_threshold: Speed strictly below which a frame is "stalled". > 0.
        stall_window_s: Continuous low-speed seconds required to latch. > 0.
        min_runtime_s: Grace seconds from the first frame before a stall can
            latch. >= 0.
        lookback_s: Approach seconds kept before the stall span begins. >= 0.
        min_frames: Minimum window length in frames; the start is pulled back to
            reach it where the episode allows. >= 1.

    Returns:
        A ``(start, stop)`` half-open range with ``0 <= start < stop <= n``, or
        ``None`` when the trace never satisfies the stall condition.

    Raises:
        ValueError: On shape/value violations (see argument bounds).
    """
    spd = np.asarray(speed, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if spd.ndim != 1:
        raise ValueError(f"speed must be 1-D (n,), got {spd.shape}")
    if ts.shape != spd.shape:
        raise ValueError(f"timestamps {ts.shape} must match speed {spd.shape}")
    if speed_threshold <= 0.0:
        raise ValueError(f"speed_threshold must be > 0, got {speed_threshold}")
    if stall_window_s <= 0.0:
        raise ValueError(f"stall_window_s must be > 0, got {stall_window_s}")
    if min_runtime_s < 0.0:
        raise ValueError(f"min_runtime_s must be >= 0, got {min_runtime_s}")
    if lookback_s < 0.0:
        raise ValueError(f"lookback_s must be >= 0, got {lookback_s}")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")
    n = int(spd.shape[0])
    if n == 0:
        return None
    t0 = ts[0]
    low_since_idx: int | None = None
    for i in range(n):
        if spd[i] < speed_threshold:
            if low_since_idx is None:
                low_since_idx = i
            low_duration = ts[i] - ts[low_since_idx]
            runtime = ts[i] - t0
            if runtime >= min_runtime_s and low_duration >= stall_window_s:
                onset = low_since_idx
                start = _lookback_start(ts, onset, lookback_s)
                stop = n
                if stop - start < min_frames:
                    start = max(0, stop - min_frames)
                return start, stop
        else:
            low_since_idx = None
    return None


def _lookback_start(timestamps: np.ndarray, onset: int, lookback_s: float) -> int:
    """Return the first index within ``lookback_s`` seconds before ``onset``.

    Args:
        timestamps: Per-frame times ``(n,)`` (s), non-decreasing.
        onset: Index the stall span begins at.
        lookback_s: Seconds of approach to keep before ``onset``.

    Returns:
        The smallest index ``j`` with ``timestamps[onset] - timestamps[j] <=
        lookback_s`` (clamped to ``[0, onset]``).
    """
    target = timestamps[onset] - lookback_s
    j = onset
    while j > 0 and timestamps[j - 1] >= target:
        j -= 1
    return j


def terminal_window(n_frames: int, fallback_frames: int) -> tuple[int, int]:
    """Return the ``[start, n)`` window of the last ``fallback_frames`` frames.

    Used when no stall is detected: the terminal frames are still the deploy
    policy's near-end states, a reasonable relabeling target.

    Args:
        n_frames: Total frames in the episode (>= 1).
        fallback_frames: Number of trailing frames to keep (clamped to
            ``n_frames``; >= 1).

    Returns:
        A ``(start, stop)`` half-open range.

    Raises:
        ValueError: If ``n_frames`` < 1 or ``fallback_frames`` < 1.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if fallback_frames < 1:
        raise ValueError(f"fallback_frames must be >= 1, got {fallback_frames}")
    start = max(0, n_frames - int(fallback_frames))
    return start, n_frames


def select_window(
    tcp_velocities: np.ndarray,
    timestamps: np.ndarray,
    *,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    stall_window_s: float = DEFAULT_STALL_WINDOW_S,
    min_runtime_s: float = DEFAULT_MIN_RUNTIME_S,
    lookback_s: float = DEFAULT_STALL_LOOKBACK_S,
    min_frames: int = DEFAULT_MIN_WINDOW_FRAMES,
    fallback_frames: int = DEFAULT_FALLBACK_FRAMES,
) -> tuple[tuple[int, int], bool]:
    """Return the training window and whether a stall was detected.

    Combines :func:`opt.episode_prep.frame_speed` + :func:`stall_window` with a
    terminal-window fallback so a rollout that never latches a stall (e.g. crept
    to the time limit) still yields deploy states to relabel.

    Args:
        tcp_velocities: Per-frame twist array ``(n, d)`` (d >= 3).
        timestamps: Per-frame times ``(n,)`` (s).
        speed_threshold: See :func:`stall_window`.
        stall_window_s: See :func:`stall_window`.
        min_runtime_s: See :func:`stall_window`.
        lookback_s: See :func:`stall_window`.
        min_frames: See :func:`stall_window`.
        fallback_frames: Trailing frames kept when no stall is detected.

    Returns:
        A tuple ``((start, stop), stalled)`` where ``stalled`` is True iff the
        stall condition fired (False means the terminal-window fallback was used).
    """
    speed = episode_prep.frame_speed(np.asarray(tcp_velocities, dtype=np.float64))
    win = stall_window(
        speed,
        np.asarray(timestamps, dtype=np.float64),
        speed_threshold=speed_threshold,
        stall_window_s=stall_window_s,
        min_runtime_s=min_runtime_s,
        lookback_s=lookback_s,
        min_frames=min_frames,
    )
    if win is not None:
        return win, True
    return terminal_window(int(speed.shape[0]), fallback_frames), False


# =============================================================================
# Episode emission (pure: numpy arrays in, .npy files out -- no ROS)
# =============================================================================
def slice_frame_arrays(
    arrays: dict[str, np.ndarray], start: int, stop: int
) -> dict[str, np.ndarray]:
    """Slice every per-frame array to ``[start, stop)`` on axis 0.

    Args:
        arrays: Map of array-name -> ``(n, ...)`` array; every value must share
            the same length ``n`` on axis 0.
        start: Inclusive window start.
        stop: Exclusive window stop.

    Returns:
        A new dict with each array sliced to the window.

    Raises:
        ValueError: If ``arrays`` is empty, lengths disagree, or the window is
            invalid.
    """
    if not arrays:
        raise ValueError("arrays must be non-empty")
    lengths = {name: int(a.shape[0]) for name, a in arrays.items()}
    n = next(iter(lengths.values()))
    if any(v != n for v in lengths.values()):
        raise ValueError(f"frame arrays have mismatched lengths: {lengths}")
    if not (0 <= start < stop <= n):
        raise ValueError(f"invalid window [{start}, {stop}) for n={n}")
    return {name: a[start:stop] for name, a in arrays.items()}


@dataclasses.dataclass(frozen=True)
class RelabelResult:
    """Outcome of relabeling one deploy rollout.

    Attributes:
        episode_dir: Directory the relabeled episode was written to.
        n_frames: Frames written (the window length).
        window: The ``(start, stop)`` window selected on the full rollout.
        stalled: Whether a stall latched (False -> terminal-window fallback).
        port_position_base: True port entrance position ``[x, y, z]`` in base_link.
        entrance_frame: The port-entrance TF frame the target was read from.
        max_offset_cm: Largest per-frame |offset| in the window (cm), a sanity
            magnitude (a stall should be within a few cm of the port).
    """

    episode_dir: str
    n_frames: int
    window: tuple[int, int]
    stalled: bool
    port_position_base: np.ndarray
    entrance_frame: str
    max_offset_cm: float


def write_relabeled_episode(
    out_dir: str | pathlib.Path,
    frame_arrays: dict[str, np.ndarray],
    port_position_base: np.ndarray,
    *,
    aux_frame: str = "tcp",
    entrance_frame: str = "",
    window: tuple[int, int] = (0, 0),
    stalled: bool = True,
) -> RelabelResult:
    """Write a relabeled stall-window episode + its port target/offset labels.

    Emits the standard ``prepare_dataset`` per-frame arrays (already sliced to the
    window), plus ``port_target.npy`` (the base_link port position the extended
    ``opt.port_labels`` reads as the target) and ``port_offsets.npy`` (the
    precomputed per-frame labels, provenance / for inspection). An
    ``insertion_frame.npy`` of ``-1`` is written so the last-inch selector treats
    these non-seating episodes as "no seat".

    Args:
        out_dir: Episode output directory (created if missing).
        frame_arrays: The sliced per-frame arrays, keyed by
            :data:`FRAME_ARRAY_NAMES` (``tcp_poses.npy`` is required).
        port_position_base: True port position ``[x, y, z]`` in base_link (m).
        aux_frame: Label frame for ``port_offsets.npy`` (``'tcp'`` / ``'base'``).
        entrance_frame: The entrance frame name (recorded in the result).
        window: The ``(start, stop)`` window (recorded in the result).
        stalled: Whether a stall latched (recorded in the result).

    Returns:
        The populated :class:`RelabelResult`.

    Raises:
        ValueError: If ``tcp_poses.npy`` is missing from ``frame_arrays`` or the
            port position is not length 3.
    """
    if "tcp_poses.npy" not in frame_arrays:
        raise ValueError("frame_arrays must contain 'tcp_poses.npy'")
    port = np.asarray(port_position_base, dtype=np.float64).reshape(-1)
    if port.shape[0] != 3:
        raise ValueError(f"port_position_base must be length 3, got {port.shape}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(frame_arrays["tcp_poses.npy"], dtype=np.float64)
    n = int(poses.shape[0])
    for name, arr in frame_arrays.items():
        np.save(out / name, arr)
    np.save(out / PORT_TARGET_NAME, port.astype(np.float64))
    offsets = port_offset_labels(poses, port, frame=aux_frame).astype(np.float32)
    np.save(out / PORT_OFFSETS_NAME, offsets)
    # Non-seating deploy stalls: no seat marker.
    np.save(out / "insertion_frame.npy", np.asarray(-1, dtype=np.int64))
    np.save(out / "seat_time.npy", np.asarray([], dtype=np.float64))
    # Sanity magnitude in base_link (frame-independent) for logging/QA.
    base_off = port_offset_labels(poses, port, frame="base")
    max_offset_cm = float(np.max(np.linalg.norm(base_off, axis=1)) * 100.0) if n else 0.0
    return RelabelResult(
        episode_dir=str(out),
        n_frames=n,
        window=window,
        stalled=stalled,
        port_position_base=port,
        entrance_frame=entrance_frame,
        max_offset_cm=max_offset_cm,
    )


# =============================================================================
# campaign_log.csv (validity join reused by opt.port_labels)
# =============================================================================
CAMPAIGN_LOG_HEADER: str = (
    "timestamp,config,stratum,plug,rep,episode_dir,score,frames,"
    "insertion_events,wall_clock_s,status\n"
)


def append_campaign_row(
    log_path: str | pathlib.Path,
    *,
    episode_dir: str,
    config: str,
    stratum: str,
    plug: str,
    rep: str,
    frames: int,
    timestamp: str = "",
) -> None:
    """Append a KEEP row for a relabeled episode to the DAgger campaign log.

    Rows are marked ``KEEP`` with ``insertion_events=1`` so the existing
    ``opt.port_labels`` validity join accepts every relabeled episode as a label
    source; the true target still comes from ``port_target.npy`` (the additive
    hook), not from the (non-seating) TCP tail. The header is written on first use.

    Args:
        log_path: Path to the ``campaign_log.csv`` (created with a header if new).
        episode_dir: Episode directory basename recorded in the row.
        config: Config yaml path for provenance.
        stratum: Stratum label for provenance.
        plug: Plug type (``sfp`` / ``sc``).
        rep: Repetition index.
        frames: Frames written for this episode.
        timestamp: Optional collection timestamp string.
    """
    path = pathlib.Path(log_path)
    if not path.exists():
        path.write_text(CAMPAIGN_LOG_HEADER)
    row = (
        f"{timestamp},{config},{stratum},{plug},{rep},{episode_dir},"
        f",{frames},1,,KEEP\n"
    )
    with path.open("a") as fh:
        fh.write(row)


# =============================================================================
# Bag-reading shell (lazy ROS imports -- never exercised by unit tests)
# =============================================================================
def read_port_target_from_bag(
    bag_path: str,
    *,
    port_name: str | None = None,
    port_frame: str | None = None,
    base_frame: str = DEFAULT_BASE_FRAME,
) -> tuple[np.ndarray, str]:
    """Read the true port entrance position in base_link from a rollout bag.

    Reads every transform on :data:`TF_TOPICS`, builds a :class:`TransformForest`
    (base_link, task board, and port frames are static, so the last transform per
    child is exact), selects the target entrance frame, and resolves its origin in
    ``base_frame``.

    Args:
        bag_path: Path to the rosbag2/MCAP directory.
        port_name: Optional port name to disambiguate the entrance frame.
        port_frame: Optional fully-qualified entrance frame (overrides detection).
        base_frame: Frame the port position is expressed in (default ``base_link``).

    Returns:
        A tuple ``(port_position_base, entrance_frame)``.

    Raises:
        FileNotFoundError: If the bag does not exist.
        ValueError: If no transforms / no entrance frame are found, or the target
            cannot be resolved into ``base_frame``.
    """
    from rosbags.rosbag2 import Reader  # lazy: ROS-only path

    import prepare_dataset  # reuse the custom-msg typestore builder

    if not pathlib.Path(bag_path).exists():
        raise FileNotFoundError(f"bag not found: {bag_path}")
    ts_store = prepare_dataset._build_typestore()
    forest = TransformForest()
    scoring_frames: set[str] = set()  # child frames scoped to the task connection
    n_tf = 0
    with Reader(bag_path) as reader:
        for conn, _t, raw in reader.messages():
            if conn.topic not in TF_TOPICS:
                continue
            try:
                msg = ts_store.deserialize_cdr(raw, conn.msgtype)
            except Exception:  # noqa: BLE001 - skip undecodable messages
                continue
            is_scoring = conn.topic.startswith("/scoring/")
            for tf in msg.transforms:
                tr = tf.transform.translation
                rot = tf.transform.rotation
                forest.add_transform(
                    tf.header.frame_id,
                    tf.child_frame_id,
                    np.array([tr.x, tr.y, tr.z], dtype=np.float64),
                    np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64),
                )
                if is_scoring:
                    scoring_frames.add(tf.child_frame_id)
                n_tf += 1
    if n_tf == 0:
        raise ValueError(
            f"no transforms on {TF_TOPICS} in {bag_path}; "
            "was the bag recorded with ground_truth:=true and /scoring/tf?"
        )
    entrance = select_entrance_frame(
        forest.frame_ids(),
        port_name=port_name,
        explicit=port_frame,
        prefer=scoring_frames or None,
    )
    port = forest.position_of(base_frame, entrance)
    _LOG.info(
        "port target: %s @ base_link=%s (from %d transforms)",
        entrance, np.round(port, 4).tolist(), n_tf,
    )
    return port, entrance


def _load_frame_arrays(episode_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Load the per-frame arrays prepare_dataset wrote, keyed by file name.

    Args:
        episode_dir: Directory holding the converted per-frame ``.npy`` arrays.

    Returns:
        A dict of ``name -> array`` for every present :data:`FRAME_ARRAY_NAMES`
        entry (``tcp_poses.npy`` and ``tcp_velocities.npy`` are required).

    Raises:
        FileNotFoundError: If a required array is missing.
    """
    arrays: dict[str, np.ndarray] = {}
    for name in FRAME_ARRAY_NAMES:
        path = episode_dir / name
        if path.exists():
            arrays[name] = np.load(path)
    for required in ("tcp_poses.npy", "tcp_velocities.npy", "timestamps.npy"):
        if required not in arrays:
            raise FileNotFoundError(f"{required} missing in {episode_dir}")
    return arrays


def relabel_bag(
    bag_path: str,
    out_dir: str,
    *,
    port_name: str | None = None,
    port_frame: str | None = None,
    aux_frame: str = "tcp",
    base_frame: str = DEFAULT_BASE_FRAME,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    stall_window_s: float = DEFAULT_STALL_WINDOW_S,
    min_runtime_s: float = DEFAULT_MIN_RUNTIME_S,
    lookback_s: float = DEFAULT_STALL_LOOKBACK_S,
    min_frames: int = DEFAULT_MIN_WINDOW_FRAMES,
    fallback_frames: int = DEFAULT_FALLBACK_FRAMES,
) -> RelabelResult:
    """Convert + relabel one deploy-rollout bag into a training episode.

    Pipeline: reuse :func:`prepare_dataset.process_bag` to build the standard
    episode arrays -> read the true port target from the bag's TF -> select the
    stall window -> slice -> write the relabeled episode + labels. A temporary
    full-conversion directory is used and removed; the caller deletes the raw bag.

    Args:
        bag_path: Path to the rosbag2/MCAP directory.
        out_dir: Final relabeled-episode output directory.
        port_name: Optional port name to disambiguate the entrance frame.
        port_frame: Optional fully-qualified entrance frame (overrides detection).
        aux_frame: Label frame (``'tcp'`` / ``'base'``).
        base_frame: Frame the port position is expressed in.
        speed_threshold: Stall speed threshold (m/s).
        stall_window_s: Continuous low-speed seconds to latch a stall.
        min_runtime_s: Startup grace seconds before a stall can latch.
        lookback_s: Approach seconds kept before the stall onset.
        min_frames: Minimum window frames.
        fallback_frames: Terminal frames kept when no stall latches.

    Returns:
        The :class:`RelabelResult`.

    Raises:
        FileNotFoundError: If the bag or a required converted array is missing.
        ValueError: If no frames convert or the port target cannot be read.
    """
    import shutil
    import tempfile

    import prepare_dataset

    out_path = pathlib.Path(out_dir)
    port, entrance = read_port_target_from_bag(
        bag_path, port_name=port_name, port_frame=port_frame, base_frame=base_frame
    )
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="dagger_full_"))
    try:
        n = prepare_dataset.process_bag(bag_path, str(tmp_dir))
        if n == 0:
            raise ValueError(f"prepare_dataset produced 0 frames for {bag_path}")
        arrays = _load_frame_arrays(tmp_dir)
        (win_start, win_stop), stalled = select_window(
            arrays["tcp_velocities.npy"],
            arrays["timestamps.npy"],
            speed_threshold=speed_threshold,
            stall_window_s=stall_window_s,
            min_runtime_s=min_runtime_s,
            lookback_s=lookback_s,
            min_frames=min_frames,
            fallback_frames=fallback_frames,
        )
        sliced = slice_frame_arrays(arrays, win_start, win_stop)
        result = write_relabeled_episode(
            out_path,
            sliced,
            port,
            aux_frame=aux_frame,
            entrance_frame=entrance,
            window=(win_start, win_stop),
            stalled=stalled,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    _LOG.info(
        "relabeled %s -> %s: %d frames, window=%s stalled=%s max|off|=%.1fcm",
        bag_path, out_path, result.n_frames, result.window, result.stalled,
        result.max_offset_cm,
    )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    ap = argparse.ArgumentParser(
        description="Relabel a DEPLOY-policy rollout bag with the true port offset."
    )
    ap.add_argument("bag", help="rosbag2/MCAP directory recorded under ground_truth:=true")
    ap.add_argument("out", help="output episode directory")
    ap.add_argument("--port-name", default=None, help="port name to disambiguate the entrance frame")
    ap.add_argument("--port-frame", default=None, help="explicit entrance frame (overrides detection)")
    ap.add_argument("--aux-frame", default="tcp", choices=("tcp", "base"))
    ap.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    ap.add_argument("--speed-threshold", type=float, default=DEFAULT_SPEED_THRESHOLD)
    ap.add_argument("--stall-window-s", type=float, default=DEFAULT_STALL_WINDOW_S)
    ap.add_argument("--min-runtime-s", type=float, default=DEFAULT_MIN_RUNTIME_S)
    ap.add_argument("--lookback-s", type=float, default=DEFAULT_STALL_LOOKBACK_S)
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_WINDOW_FRAMES)
    ap.add_argument("--fallback-frames", type=int, default=DEFAULT_FALLBACK_FRAMES)
    ap.add_argument(
        "--campaign-log", default="",
        help="append a KEEP row to this campaign_log.csv (validity join)",
    )
    ap.add_argument("--config", default="", help="config yaml (campaign_log provenance)")
    ap.add_argument("--stratum", default="dagger", help="stratum label (campaign_log)")
    ap.add_argument("--plug", default="", help="plug type (campaign_log)")
    ap.add_argument("--rep", default="0", help="rep index (campaign_log)")
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: relabel one bag and (optionally) log a campaign row.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="[dagger_relabel] %(message)s")
    args = _build_arg_parser().parse_args(argv)
    result = relabel_bag(
        args.bag,
        args.out,
        port_name=args.port_name,
        port_frame=args.port_frame,
        aux_frame=args.aux_frame,
        base_frame=args.base_frame,
        speed_threshold=args.speed_threshold,
        stall_window_s=args.stall_window_s,
        min_runtime_s=args.min_runtime_s,
        lookback_s=args.lookback_s,
        min_frames=args.min_frames,
        fallback_frames=args.fallback_frames,
    )
    if args.campaign_log:
        append_campaign_row(
            args.campaign_log,
            episode_dir=pathlib.Path(result.episode_dir).name,
            config=args.config,
            stratum=args.stratum,
            plug=args.plug,
            rep=args.rep,
            frames=result.n_frames,
        )
    print(
        f"Done: {result.n_frames} frames -> {result.episode_dir} "
        f"(stalled={result.stalled}, max|off|={result.max_offset_cm:.1f}cm, "
        f"port_frame={result.entrance_frame})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
