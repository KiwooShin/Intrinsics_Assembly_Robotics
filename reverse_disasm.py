#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Time-reverse a recorded perturbed-retract episode into an insertion demo.

``DisassembleCode`` records a *disassembly* episode: the plug starts at (near) seat
depth (frame 0) and is slowly, perturbedly retracted out of the port (frame N-1),
without ever dwelling long enough to fire the 1 s touch latch (so the seat stays
reversible). This module turns that recording into a standard ``prepare_dataset``
insertion episode by time-reversing it, so ``opt/train_v3.py --wrench`` consumes it
unchanged.

The transform (design-verified):

* **Observations** (``{center,left,right}_images``, ``tcp_poses``, ``wrenches``,
  ``joint_positions``) are kept *forward-recorded* -- their recorded values are the
  genuine sensor readings -- and only their frame ORDER is reversed, so reversed
  frame ``j`` shows exactly what was recorded at frame ``N-1-j``. Reversed frame 0
  is the fully-retracted (far-out) pose; reversed frame ``N-1`` is the seat.
* **Action** (``tcp_velocities``) is reversed in order AND negated::

      a_insert[j] = -tcp_velocity[N-1-j]

  The action is a **base_link (spatial / fixed-frame) twist** -- the exact space
  ``train_v2.Policy`` predicts and ``DeployACT`` integrates. Time-reversal of a
  fixed-frame twist negates all six components (linear AND angular) with **no
  body-frame coupling**: the twist's reference frame (``base_link``) does not move
  under time reversal, so unlike a body/tool-frame twist there is no orientation-
  dependent term to correct. Reversing the order and negating therefore yields the
  exact velocity that drives the plug inward at each reversed frame.
* **Timestamps** are rebuilt monotonic from the reversed inter-frame gaps
  (``t[0] + (t[-1] - t[::-1])``), preserving the recording's per-step durations.
* ``insertion_frame.npy`` is set to ``N-1`` (the seat is the LAST frame of the
  reversed episode), which is exactly the seat marker the last-inch selector
  (``episode_prep.last_inch_window``) reads.

The recorded **wrench is a PROXY** and is kept AS-IS (order-reversed, sign
unchanged). Friction shear flips direction on time reversal and there is no seating
force spike in a retract, so the wrench does not match a true insertion's forces;
but its recorded values still capture the contact geometry (which surfaces touch,
normal-force magnitude), which is the useful signal for the wrench-augmented head.

Pure array helpers (:func:`reverse_action`, :func:`reverse_observation`,
:func:`rebuild_timestamps`) are ROS/torch/GPU-free and unit-tested; the thin I/O
wrapper :func:`reverse_episode` loads/saves the ``.npy`` layout.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# The action twist width (base_link [vx, vy, vz, wx, wy, wz]).
ACTION_DIM: int = 6

# Per-episode observation arrays that are reversed in ORDER but not in value.
_OBSERVATION_FILES: tuple[str, ...] = (
    "center_images.npy",
    "left_images.npy",
    "right_images.npy",
    "tcp_poses.npy",
    "wrenches.npy",
    "joint_positions.npy",
)


def reverse_action(velocities: np.ndarray) -> np.ndarray:
    """Return the time-reversed, negated insertion action for a retract episode.

    Implements ``a_insert[j] = -velocities[N-1-j]`` for every frame ``j`` -- i.e.
    reverse the frame order and negate all six twist components. Correct because the
    action is a base_link (fixed-frame) twist: time reversal negates a fixed-frame
    twist exactly, with no body-frame coupling.

    Args:
        velocities: Recorded retract twist array shaped ``(N, 6)`` as
            ``[vx, vy, vz, wx, wy, wz]`` per frame.

    Returns:
        A ``(N, 6)`` array (same dtype as the input after float cast) of the
        reversed, negated insertion actions.

    Raises:
        ValueError: If ``velocities`` is not 2-D with exactly ``ACTION_DIM``
            columns, or is empty.
    """
    v = np.asarray(velocities)
    if v.ndim != 2 or v.shape[1] != ACTION_DIM:
        raise ValueError(
            f"velocities must be 2-D (N, {ACTION_DIM}), got shape {v.shape}"
        )
    if v.shape[0] == 0:
        raise ValueError("velocities must be non-empty")
    return (-v[::-1]).copy()


def reverse_observation(array: np.ndarray) -> np.ndarray:
    """Return an observation array with its frame order reversed (values intact).

    Reverses only axis 0 (frames); the per-frame values (pixels, pose, wrench,
    joints) are the genuine recorded readings and are left unchanged.

    Args:
        array: An observation array shaped ``(N, ...)`` (any trailing shape).

    Returns:
        A ``(N, ...)`` array with axis 0 reversed.

    Raises:
        ValueError: If ``array`` is empty on axis 0.
    """
    a = np.asarray(array)
    if a.shape[0] == 0:
        raise ValueError("observation array must be non-empty on axis 0")
    return a[::-1].copy()


def rebuild_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Rebuild strictly monotonic timestamps for the reversed episode.

    Naively reversing timestamps yields a decreasing sequence; this instead
    reverses the inter-frame GAPS so the reversed episode advances forward in time
    with the same per-step durations::

        new_ts[j] = t[0] + (t[-1] - t[N-1-j])

    which is ``t[0]`` at ``j = 0`` and ``t[-1]`` at ``j = N-1``, monotonically
    non-decreasing whenever the input was.

    Args:
        timestamps: Recorded per-frame capture times ``(N,)`` (seconds),
            non-decreasing.

    Returns:
        A ``(N,)`` float64 array of rebuilt timestamps.

    Raises:
        ValueError: If ``timestamps`` is not 1-D or is empty.
    """
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or t.shape[0] == 0:
        raise ValueError(f"timestamps must be a non-empty 1-D array, got {t.shape}")
    return t[0] + (t[-1] - t[::-1])


def reversed_seat_index(n_frames: int) -> int:
    """Return the seat frame index of a reversed episode: ``n_frames - 1``.

    Args:
        n_frames: Number of frames in the episode.

    Returns:
        ``n_frames - 1`` (the seat is the last frame of the reversed episode).

    Raises:
        ValueError: If ``n_frames`` < 1.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    return n_frames - 1


def reverse_episode(input_dir: str | pathlib.Path, output_dir: str | pathlib.Path) -> int:
    """Load a recorded retract episode, time-reverse it, and save an insertion demo.

    Reads the ``prepare_dataset`` ``.npy`` layout from ``input_dir``, applies the
    reversal (observations order-reversed, ``tcp_velocities`` reversed + negated,
    timestamps rebuilt), and writes the same layout to ``output_dir`` plus
    ``insertion_frame.npy = N-1`` and a single-element ``seat_time.npy``. Optional
    proprioceptive arrays (``wrenches.npy`` / ``joint_positions.npy``) are reversed
    when present and skipped when absent (older collection runs).

    Args:
        input_dir: Directory holding the recorded retract episode's ``.npy`` arrays.
            Must contain ``center_images.npy``, ``left_images.npy``,
            ``right_images.npy``, ``tcp_poses.npy``, ``tcp_velocities.npy``, and
            ``timestamps.npy``.
        output_dir: Directory to write the reversed insertion episode into (created
            if missing).

    Returns:
        The number of frames written.

    Raises:
        FileNotFoundError: If a required array is missing from ``input_dir``.
        ValueError: If the arrays disagree on frame count or an array is malformed.
    """
    in_dir = pathlib.Path(input_dir)
    out_dir = pathlib.Path(output_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"input episode dir not found: {in_dir}")

    required = (
        "center_images.npy",
        "left_images.npy",
        "right_images.npy",
        "tcp_poses.npy",
        "tcp_velocities.npy",
        "timestamps.npy",
    )
    for name in required:
        if not (in_dir / name).is_file():
            raise FileNotFoundError(f"missing required array {name} in {in_dir}")

    velocities = np.load(in_dir / "tcp_velocities.npy")
    n = int(velocities.shape[0])
    a_insert = reverse_action(velocities)

    timestamps = np.load(in_dir / "timestamps.npy")
    if timestamps.shape[0] != n:
        raise ValueError(
            f"timestamps has {timestamps.shape[0]} frames, expected {n}"
        )
    new_ts = rebuild_timestamps(timestamps)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Observations: order-reversed, values unchanged. Skip absent optional arrays.
    for name in _OBSERVATION_FILES:
        path = in_dir / name
        if not path.is_file():
            continue
        arr = np.load(path)
        if arr.shape[0] != n:
            raise ValueError(
                f"{name} has {arr.shape[0]} frames, expected {n}"
            )
        np.save(out_dir / name, reverse_observation(arr))

    np.save(out_dir / "tcp_velocities.npy", a_insert.astype(np.float32))
    np.save(out_dir / "timestamps.npy", new_ts.astype(np.float64))

    seat_idx = reversed_seat_index(n)
    np.save(out_dir / "insertion_frame.npy", np.asarray(seat_idx, dtype=np.int64))
    # Seat time = the reversed episode's final timestamp (provenance for the marker).
    np.save(out_dir / "seat_time.npy", np.asarray([new_ts[-1]], dtype=np.float64))
    return n


def main(argv: list[str] | None = None) -> None:
    """CLI: ``reverse_disasm.py <input_episode_dir> <output_episode_dir>``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(
        description="Time-reverse a recorded perturbed-retract episode into an "
        "insertion demo (prepare_dataset layout)."
    )
    ap.add_argument("input_dir", help="recorded retract episode dir (.npy layout)")
    ap.add_argument("output_dir", help="output insertion episode dir")
    args = ap.parse_args(argv)
    n = reverse_episode(args.input_dir, args.output_dir)
    print(f"Done: {n} frames -> {args.output_dir} (insertion_frame={n - 1})")


if __name__ == "__main__":
    main(sys.argv[1:])
