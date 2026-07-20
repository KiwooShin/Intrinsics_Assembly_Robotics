"""Pure numpy episode-preparation helpers for the ACT-lite trainer.

These implement the three demo-side fixes for the "last-inch zero-velocity
attractor" documented in ``SESSION_REPORT.md`` (2026-07-19), factored as pure
array math so they unit-test CPU-only (no torch, ROS, or GPU):

  * :func:`terminal_tail_trim_length` -- drop the long *seated* zero-velocity
    tail each demo ends with (the frames that alias the near-port approach and
    teach the deterministic head to regress toward zero velocity near the port),
    keeping frames up to the last moving frame plus a small margin.
  * :func:`last_inch_window` -- the *inverse* selector for an insertion
    specialist: keep only the terminal approach->seat window and drop the long
    lead-in, so the policy trains on the last inch alone (INSERTION_PLAN.md
    P-INSERT-1 code change #1).
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


def last_inch_window(
    speed: np.ndarray,
    *,
    thr: float = 0.003,
    lookback_s: float | None = None,
    lookback_frames: int | None = None,
    dt: float | None = None,
    min_frames: int = 8,
    seat_index: int | None = None,
) -> tuple[int, int]:
    """Return the terminal ``[start, stop)`` "last-inch" window of an episode.

    The inverse of :func:`terminal_tail_trim_length`: instead of dropping the
    seated zero-velocity tail and keeping the long approach, this keeps only the
    terminal approach->seat window and drops the long lead-in, so an insertion
    specialist trains on the last inch alone.

    The window *end* is the seat (one past the seat frame). When ``seat_index`` is
    supplied (the exact marker persisted by ``prepare_dataset`` code change #3) it
    is used directly; otherwise the end falls back to the speed-derived last
    moving frame -- the same anchor :func:`terminal_tail_trim_length` uses, with
    zero margin. The window *start* is ``end`` minus a lookback, given either in
    seconds (``lookback_s`` together with ``dt``) or in frames
    (``lookback_frames``), then floored so at least ``min_frames`` frames survive
    where the episode is long enough, and clamped so ``start >= 0``.

    Args:
        speed: Per-frame non-negative linear speed array shaped ``(n,)`` (m/s),
            e.g. the output of :func:`frame_speed`.
        thr: Speed (same units as ``speed``) at/above which a frame counts as
            "moving", used only for the speed-derived end. Must be finite and
            >= 0.
        lookback_s: Lookback length in seconds. Mutually exclusive with
            ``lookback_frames`` and requires a positive ``dt``. Must be >= 0.
        lookback_frames: Lookback length in frames. Mutually exclusive with
            ``lookback_s``. Must be >= 0.
        dt: Frame period in seconds, required iff ``lookback_s`` is given (> 0).
        min_frames: Minimum window length in frames (>= 1); the start is pulled
            back so ``stop - start >= min_frames`` where the episode allows it,
            otherwise the whole ``[0, stop)`` prefix is kept.
        seat_index: Optional exact seat frame index into ``speed``. When >= 0 the
            window ends one past it; ``None`` or a negative value falls back to
            the speed-derived end (episodes lacking the persisted marker).

    Returns:
        A ``(start, stop)`` half-open index range with ``0 <= start < stop <= n``.

    Raises:
        ValueError: If ``speed`` is not 1-D or is empty, ``thr`` < 0,
            ``min_frames`` < 1, no lookback is given, both lookbacks are given, a
            lookback is negative, or ``lookback_s`` is given without a positive
            ``dt``.
    """
    if speed.ndim != 1:
        raise ValueError(f"speed must be 1-D (n,), got shape {speed.shape}")
    n = int(speed.shape[0])
    if n == 0:
        raise ValueError("speed must be non-empty")
    if thr < 0:
        raise ValueError(f"thr must be >= 0, got {thr}")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")
    if lookback_s is None and lookback_frames is None:
        raise ValueError("a lookback is required: pass lookback_s or lookback_frames")
    if lookback_s is not None and lookback_frames is not None:
        raise ValueError("pass exactly one of lookback_s or lookback_frames, not both")
    if lookback_frames is not None:
        if lookback_frames < 0:
            raise ValueError(f"lookback_frames must be >= 0, got {lookback_frames}")
        lookback = int(lookback_frames)
    else:
        if lookback_s < 0:
            raise ValueError(f"lookback_s must be >= 0, got {lookback_s}")
        if dt is None or dt <= 0:
            raise ValueError(f"lookback_s requires dt > 0, got dt={dt}")
        lookback = seconds_to_frames(lookback_s, dt, minimum=0)

    # Window end: the exact seat marker when present, else the speed-derived last
    # moving frame (the zero-margin tail anchor).
    if seat_index is not None and seat_index >= 0:
        stop = min(int(seat_index) + 1, n)
    else:
        stop = terminal_tail_trim_length(speed, thr, 0)

    start = stop - lookback
    # Floor the window to min_frames where the episode allows it, then clamp to 0.
    if stop - start < min_frames:
        start = stop - min_frames
    start = max(0, start)
    return start, stop


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


def build_last_inch_keep_and_weights(
    epid: np.ndarray,
    velocities: np.ndarray,
    *,
    thr: float,
    min_frames: int,
    lookback_s: float | None = None,
    lookback_frames: int | None = None,
    dt: float | None = None,
    pushin_ramp_frames: int = 0,
    pushin_weight: float = 1.0,
    seat_indices: list[int] | None = None,
    n_linear: int = LINEAR_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a global keep-mask + loss weights keeping only each last-inch window.

    The last-inch counterpart of :func:`build_keep_and_weights`: for each
    contiguous episode (see :func:`episode_bounds`) it keeps only the terminal
    :func:`last_inch_window` (approach->seat) and drops the long lead-in, then
    assigns :func:`pushin_weights` over the kept window. Frames outside the window
    get ``keep=False`` and weight ``0`` (masked out downstream).

    Args:
        epid: Per-frame episode-id array shaped ``(n,)`` (contiguous blocks).
        velocities: Per-frame twist array shaped ``(n, d)`` (the first action of
            each frame's chunk, ``d >= n_linear``).
        thr: Moving-speed threshold in m/s for the speed-derived window end.
        min_frames: Minimum frames kept per episode window (>= 1).
        lookback_s: Lookback length in seconds (needs ``dt``); mutually exclusive
            with ``lookback_frames``.
        lookback_frames: Lookback length in frames; mutually exclusive with
            ``lookback_s``.
        dt: Frame period in seconds, required iff ``lookback_s`` is given.
        pushin_ramp_frames: Length of the trailing loss-weight ramp in frames.
        pushin_weight: Peak push-in loss weight (``1.0`` disables ramping).
        seat_indices: Optional per-episode exact seat frame indices, one entry per
            :func:`episode_bounds` block in block order; a negative entry (e.g.
            ``-1``) falls back to the speed-derived end for that episode. ``None``
            uses the speed-derived end for every episode.
        n_linear: Leading linear components used for the speed norm.

    Returns:
        A tuple ``(keep_mask, weights)`` of ``(n,)`` arrays: ``keep_mask`` is
        ``bool`` (kept frames), ``weights`` is ``float32`` (ramp value on kept
        frames, ``0`` on dropped frames). Apply both by indexing with
        ``keep_mask``.

    Raises:
        ValueError: If ``epid`` is not 1-D, ``velocities`` is not 2-D with a
            matching first dimension, or ``seat_indices`` (when given) does not
            have exactly one entry per episode block.
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
    bounds = episode_bounds(epid)
    if seat_indices is not None and len(seat_indices) != len(bounds):
        raise ValueError(
            f"seat_indices ({len(seat_indices)}) must have one entry per episode "
            f"block ({len(bounds)})"
        )
    n = int(epid.shape[0])
    keep = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float32)
    for block_i, (start, stop) in enumerate(bounds):
        speed = frame_speed(velocities[start:stop], n_linear)
        seat: int | None = None
        if seat_indices is not None:
            s = int(seat_indices[block_i])
            seat = s if s >= 0 else None
        w_start, w_stop = last_inch_window(
            speed,
            thr=thr,
            lookback_s=lookback_s,
            lookback_frames=lookback_frames,
            dt=dt,
            min_frames=min_frames,
            seat_index=seat,
        )
        keep[start + w_start:start + w_stop] = True
        weights[start + w_start:start + w_stop] = pushin_weights(
            w_stop - w_start, pushin_ramp_frames, pushin_weight
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
