"""Pure numpy episode-preparation helpers for the ACT-lite trainer.

These implement the three demo-side fixes for the "last-inch zero-velocity
attractor" documented in ``SESSION_REPORT.md`` (2026-07-19), factored as pure
array math so they unit-test CPU-only (no torch, ROS, or GPU):

  * :func:`terminal_tail_trim_length` -- drop the long *seated* zero-velocity
    tail each demo ends with (the frames that alias the near-port approach and
    teach the deterministic head to regress toward zero velocity near the port),
    keeping frames up to the last moving frame plus a small margin.
  * :func:`pushin_weights` -- per-frame loss weights that up-weight the final
    push-in phase of each (trimmed) episode so the small fraction of push-in
    frames is not drowned out by the many approach frames.
  * :func:`build_keep_and_weights` -- combine the two per episode over a whole
    concatenated dataset, given the per-frame episode-id and velocity arrays,
    returning a global keep-mask and per-frame weight vector.

The heavy GPU tensor loading stays in ``train_v2.load_all``; ``opt.train_v3``
calls :func:`build_keep_and_weights` on the (cheap) per-frame episode-id and
first-action arrays and applies the returned mask/weights to the resident GPU
tensors.
"""

from __future__ import annotations

import numpy as np

# Number of leading (linear, xyz) twist components whose L2 norm is the frame
# "speed" in m/s. The trailing 3 are angular (rad/s) and are excluded so the
# threshold is a clean m/s quantity, matching the task's ">= 0.003 m/s" spec.
LINEAR_DIMS: int = 3


def frame_speed(velocities: np.ndarray, n_linear: int = LINEAR_DIMS) -> np.ndarray:
    """Return the per-frame linear speed ``|v_xyz|`` in m/s.

    Args:
        velocities: Per-frame twist array shaped ``(n, d)`` with ``d >= n_linear``
            ordered ``[vx, vy, vz, wx, wy, wz]`` (the collection convention).
        n_linear: Number of leading linear components to include in the norm.

    Returns:
        A ``(n,)`` array of L2 norms over the first ``n_linear`` components.

    Raises:
        ValueError: If ``velocities`` is not 2-D or narrower than ``n_linear``.
    """
    if velocities.ndim != 2:
        raise ValueError(
            f"velocities must be 2-D (n, d), got shape {velocities.shape}"
        )
    if velocities.shape[1] < n_linear:
        raise ValueError(
            f"velocities has {velocities.shape[1]} cols, need >= {n_linear}"
        )
    return np.linalg.norm(velocities[:, :n_linear], axis=1)


def terminal_tail_trim_length(
    speed: np.ndarray, threshold: float, margin_frames: int
) -> int:
    """Return how many leading frames to keep after trimming the seated tail.

    Finds the last frame whose ``speed >= threshold`` and keeps everything up to
    and including that frame plus ``margin_frames`` extra frames (clamped to the
    episode length). The margin retains the brief settle right after motion stops
    so the policy still sees a short "arrived" cue without the long seated tail.

    Edge cases (both keep the whole episode, never returning 0):
      * No frame is below ``threshold`` (the demo never stalls): the last moving
        frame is the final frame, so ``len + margin`` clamps to ``n``.
      * Every frame is below ``threshold`` (a degenerate near-stationary demo):
        there is nothing to anchor a trim to, so all ``n`` frames are kept.

    Args:
        speed: Per-frame non-negative speed array shaped ``(n,)``.
        threshold: Speed (same units as ``speed``) at/above which a frame counts
            as "moving". Must be finite and >= 0.
        margin_frames: Extra frames kept after the last moving frame. Must be >= 0.

    Returns:
        The number of leading frames to keep, in ``[1, n]`` (``0`` only if the
        episode itself is empty).

    Raises:
        ValueError: If ``speed`` is not 1-D, ``threshold`` < 0, or
            ``margin_frames`` < 0.
    """
    if speed.ndim != 1:
        raise ValueError(f"speed must be 1-D (n,), got shape {speed.shape}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if margin_frames < 0:
        raise ValueError(f"margin_frames must be >= 0, got {margin_frames}")
    n = int(speed.shape[0])
    if n == 0:
        return 0
    moving = np.nonzero(speed >= threshold)[0]
    if moving.size == 0:
        # Nothing moves anywhere -> no reliable anchor; keep the whole episode.
        return n
    last_moving = int(moving[-1])
    return min(n, last_moving + 1 + int(margin_frames))


def pushin_weights(n: int, ramp_frames: int, w_max: float) -> np.ndarray:
    """Return per-frame loss weights that ramp up over the final push-in phase.

    Weights are ``1.0`` for the bulk of the episode and rise linearly to
    ``w_max`` across the last ``ramp_frames`` frames, so the final push-in frame
    carries the largest weight. This counters the approach frames (the vast
    majority) drowning the sparse push-in frames in the mean L1 loss.

    Args:
        n: Number of frames in the (already trimmed) episode. Must be >= 1.
        ramp_frames: Length of the trailing linear ramp in frames. Values <= 0
            (or ``w_max <= 1``) disable ramping and return all-ones. A ramp
            longer than ``n`` is clamped to the whole episode.
        w_max: Peak weight at the final frame. Must be >= 1.

    Returns:
        A ``(n,)`` ``float32`` array of per-frame weights in ``[1, w_max]``.

    Raises:
        ValueError: If ``n`` < 1 or ``w_max`` < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if w_max < 1.0:
        raise ValueError(f"w_max must be >= 1, got {w_max}")
    w = np.ones(n, dtype=np.float32)
    if w_max == 1.0 or ramp_frames <= 0:
        return w
    m = min(int(ramp_frames), n)
    if m > 1:
        ramp = np.linspace(1.0, w_max, m, dtype=np.float32)
    else:  # a single-frame ramp (or n==1): only the final frame is up-weighted.
        ramp = np.array([w_max], dtype=np.float32)
    w[n - m:] = ramp
    return w


def episode_bounds(epid: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[start, stop)`` index ranges for each contiguous episode block.

    ``train_v2.load_all`` concatenates episodes in order, so every episode is a
    contiguous run of equal ids. This returns those runs; it does not assume the
    ids are sorted or dense, only that equal ids are contiguous.

    Args:
        epid: Per-frame episode-id array shaped ``(n,)``.

    Returns:
        A list of ``(start, stop)`` half-open index ranges covering ``[0, n)``.

    Raises:
        ValueError: If ``epid`` is not 1-D.
    """
    if epid.ndim != 1:
        raise ValueError(f"epid must be 1-D, got shape {epid.shape}")
    n = int(epid.shape[0])
    if n == 0:
        return []
    # Boundaries where the id changes from the previous frame.
    change = np.nonzero(epid[1:] != epid[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def build_keep_and_weights(
    epid: np.ndarray,
    velocities: np.ndarray,
    *,
    tail_trim: bool,
    trim_threshold: float,
    trim_margin_frames: int,
    pushin_ramp_frames: int,
    pushin_weight: float,
    n_linear: int = LINEAR_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a global keep-mask and per-frame loss weights over all episodes.

    For each contiguous episode (see :func:`episode_bounds`):
      1. If ``tail_trim`` is set, compute the kept length with
         :func:`terminal_tail_trim_length` on the episode's linear speed;
         otherwise keep the whole episode.
      2. Assign :func:`pushin_weights` over the kept frames.

    Frames dropped by trimming get ``keep=False`` and weight ``0`` (they are
    masked out downstream). With ``tail_trim=False`` and ``pushin_weight<=1`` the
    result is "keep everything, weight 1" -- but callers should skip this call in
    that case to preserve the exact legacy training path.

    Args:
        epid: Per-frame episode-id array shaped ``(n,)`` (contiguous blocks).
        velocities: Per-frame twist array shaped ``(n, d)`` (the first action of
            each frame's chunk, ``d >= n_linear``).
        tail_trim: Whether to trim each episode's seated zero-velocity tail.
        trim_threshold: Moving-speed threshold in m/s for the trim.
        trim_margin_frames: Frames kept after the last moving frame.
        pushin_ramp_frames: Length of the trailing loss-weight ramp in frames.
        pushin_weight: Peak push-in loss weight (``1.0`` disables ramping).
        n_linear: Leading linear components used for the speed norm.

    Returns:
        A tuple ``(keep_mask, weights)`` of ``(n,)`` arrays: ``keep_mask`` is
        ``bool`` (kept frames), ``weights`` is ``float32`` (ramp value on kept
        frames, ``0`` on dropped frames). Apply both by indexing with
        ``keep_mask``.

    Raises:
        ValueError: If ``epid`` is not 1-D or ``velocities`` is not 2-D with a
            matching first dimension.
    """
    if epid.ndim != 1:
        raise ValueError(f"epid must be 1-D, got shape {epid.shape}")
    if velocities.ndim != 2:
        raise ValueError(
            f"velocities must be 2-D (n, d), got shape {velocities.shape}"
        )
    if epid.shape[0] != velocities.shape[0]:
        raise ValueError(
            f"epid ({epid.shape[0]}) and velocities ({velocities.shape[0]}) "
            "must have the same number of frames"
        )
    n = int(epid.shape[0])
    keep = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float32)
    for start, stop in episode_bounds(epid):
        length = stop - start
        if tail_trim:
            speed = frame_speed(velocities[start:stop], n_linear)
            keep_len = terminal_tail_trim_length(
                speed, trim_threshold, trim_margin_frames
            )
        else:
            keep_len = length
        keep[start:start + keep_len] = True
        weights[start:start + keep_len] = pushin_weights(
            keep_len, pushin_ramp_frames, pushin_weight
        )
    return keep, weights


def seconds_to_frames(seconds: float, dt_frame: float, minimum: int = 1) -> int:
    """Convert a duration in seconds to a frame count at period ``dt_frame``.

    Args:
        seconds: Duration in seconds. Must be >= 0.
        dt_frame: Frame period in seconds (e.g. ``0.275`` for the ~3.64 Hz
            recording). Must be > 0.
        minimum: Floor applied to the result so a positive duration never rounds
            to fewer than ``minimum`` frames.

    Returns:
        ``max(minimum, round(seconds / dt_frame))`` as an int.

    Raises:
        ValueError: If ``seconds`` < 0 or ``dt_frame`` <= 0.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    if dt_frame <= 0:
        raise ValueError(f"dt_frame must be > 0, got {dt_frame}")
    return max(int(minimum), int(round(seconds / dt_frame)))
