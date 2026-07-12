"""Verify a converted single-episode dataset is well-formed.

Exposes pure, importable helpers (:func:`load_arrays`, :func:`verify_dataset`) that
run without ROS/Gazebo/GPU. Optional montage rendering uses cv2 and is guarded.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Optional

import numpy as np

FILES = [
    'center_images', 'left_images', 'right_images',
    'tcp_velocities', 'tcp_poses', 'timestamps',
]
CAMS = ['center_images', 'left_images', 'right_images']


@dataclasses.dataclass
class DatasetReport:
    """Summary of the structural checks on a converted episode.

    Attributes:
        n_frames: Number of frames (rows in ``center_images``).
        lengths: Per-field first-axis length.
        length_match: Whether every present field has the same length.
        duration: Episode duration in seconds, or ``None`` if unavailable.
        rate_hz: Effective frame rate, or ``None`` when duration is non-positive.
        monotonic: Whether timestamps are non-decreasing, or ``None`` if unavailable.
    """

    n_frames: int
    lengths: dict[str, int]
    length_match: bool
    duration: Optional[float]
    rate_hz: Optional[float]
    monotonic: Optional[bool]


def load_arrays(data_dir: str) -> dict[str, np.ndarray]:
    """Load whichever of the expected ``.npy`` files are present.

    Args:
        data_dir: Directory holding the converted ``*.npy`` arrays.

    Returns:
        A mapping of field name to loaded array, for the files that exist.
    """
    arr: dict[str, np.ndarray] = {}
    for f in FILES:
        p = os.path.join(data_dir, f"{f}.npy")
        if os.path.exists(p):
            arr[f] = np.load(p)
    return arr


def verify_dataset(data_dir: str) -> DatasetReport:
    """Run structural checks on a converted episode directory.

    Args:
        data_dir: Directory holding the converted ``*.npy`` arrays.

    Returns:
        A :class:`DatasetReport` describing frame count, length consistency, and
        (when timestamps are present) duration/rate/monotonicity.

    Raises:
        FileNotFoundError: If ``center_images.npy`` is missing (nothing to verify).
    """
    arr = load_arrays(data_dir)
    if 'center_images' not in arr:
        raise FileNotFoundError(f"no center_images.npy in {data_dir}")

    n_frames = int(arr['center_images'].shape[0])
    lengths = {k: int(v.shape[0]) for k, v in arr.items()}
    length_match = len(set(lengths.values())) == 1

    duration: Optional[float] = None
    rate_hz: Optional[float] = None
    monotonic: Optional[bool] = None
    if 'timestamps' in arr and n_frames > 1:
        t = arr['timestamps']
        duration = float(t[-1] - t[0])
        # Guard against zero/negative duration (e.g. all-equal or unsorted stamps),
        # which would otherwise divide by zero.
        rate_hz = float(n_frames / duration) if duration > 0 else None
        monotonic = bool(np.all(np.diff(t) >= 0))

    return DatasetReport(
        n_frames=n_frames,
        lengths=lengths,
        length_match=length_match,
        duration=duration,
        rate_hz=rate_hz,
        monotonic=monotonic,
    )


def _write_montage(arr: dict[str, np.ndarray], data_dir: str) -> Optional[str]:
    """Write a small L/C/R frame montage for visual inspection (best effort).

    Args:
        arr: Loaded arrays (as returned by :func:`load_arrays`).
        data_dir: Directory to write ``_verify_montage.png`` into.

    Returns:
        The output path on success, or ``None`` if cv2 is unavailable or it fails.
    """
    try:
        import cv2
    except ImportError:
        return None
    try:
        n = arr['center_images'].shape[0]
        idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        rows = [
            np.hstack([arr[cam][i] for i in idxs]) for cam in CAMS if cam in arr
        ]
        if not rows:
            return None
        montage = np.vstack(rows)
        out = os.path.join(data_dir, "_verify_montage.png")
        cv2.imwrite(out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        return out
    except Exception as e:  # noqa: BLE001 - montage is a best-effort diagnostic
        print(f"montage failed: {e}")
        return None


def main() -> None:
    """CLI: print the verification report for the dataset dir in ``sys.argv[1]``."""
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/training/one_episode")
    print(f"=== Verifying {d} ===")
    arr = load_arrays(d)
    for f in FILES:
        if f in arr:
            print(f"  {f:16s} shape={str(arr[f].shape):22s} dtype={arr[f].dtype}")
        else:
            print(f"  MISSING: {f}.npy")

    if 'center_images' not in arr:
        print("FAIL: no images")
        sys.exit(1)

    report = verify_dataset(d)
    print(f"\nFrames: {report.n_frames}")
    print(f"length match: {report.length_match}  ({report.lengths})")
    if report.duration is not None:
        rate = f"{report.rate_hz:.1f}" if report.rate_hz is not None else "n/a"
        print(
            f"duration: {report.duration:.2f}s  effective rate: {rate} Hz  "
            f"monotonic: {report.monotonic}"
        )

    for cam in CAMS:
        if cam in arr:
            im = arr[cam]
            print(
                f"{cam}: min={im.min()} max={im.max()} mean={im.mean():.1f} "
                f"std={im.std():.1f}  frame0!=frameLast: "
                f"{not np.array_equal(im[0], im[-1])}"
            )

    if 'tcp_velocities' in arr:
        v = arr['tcp_velocities']
        sp = np.linalg.norm(v[:, :3], axis=1)
        print(
            f"\ntcp_velocity linear speed: mean={sp.mean():.4f} max={sp.max():.4f} "
            f"m/s  nonzero frames: {(sp > 1e-3).sum()}/{report.n_frames}"
        )

    if 'tcp_poses' in arr:
        p = arr['tcp_poses']
        travel = np.linalg.norm(p[-1, :3] - p[0, :3])
        print(f"TCP net travel: {travel * 1000:.1f} mm  (start->end position)")

    out = _write_montage(arr, d)
    if out:
        print(f"\nMontage (rows=L/C/R, cols=t0..tEnd) -> {out}")
    print("=== done ===")


if __name__ == '__main__':
    main()
