"""Render a side-by-side (L/C/R) video of the CheatCode SFP insertion from a bag.

Trims to the task window (first..last pose_command) and encodes h264 via ffmpeg.
The ROS/cv2/ffmpeg-heavy work lives in :func:`render_video`; the pure
:func:`nearest` and :func:`select_window` helpers are importable and testable
without those dependencies.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence, TypeVar

FPS = 12
CW, CH = 432, 384          # per-camera (aspect 1152/1024 = 1.125)
HEADER = 40
CMD_MARGIN = 0.3

T = TypeVar("T")


def nearest(lst: Sequence[tuple[float, T]], t: float) -> T:
    """Return the payload of the ``(timestamp, payload)`` tuple nearest to ``t``.

    Args:
        lst: Non-empty sequence of ``(timestamp, payload)`` tuples.
        t: Target timestamp.

    Returns:
        The ``payload`` whose timestamp is closest to ``t``.

    Raises:
        ValueError: If ``lst`` is empty.
    """
    if not lst:
        raise ValueError("nearest() requires a non-empty sequence")
    return min(lst, key=lambda x: abs(x[0] - t))[1]


def select_window(
    frames: Sequence[tuple[float, T]], t0: float, t1: float
) -> list[tuple[float, T]]:
    """Keep only ``(timestamp, payload)`` frames whose timestamp is in ``[t0, t1]``.

    Args:
        frames: Sequence of ``(timestamp, payload)`` tuples.
        t0: Inclusive lower bound.
        t1: Inclusive upper bound.

    Returns:
        The filtered list, preserving order.
    """
    return [(t, a) for (t, a) in frames if t0 <= t <= t1]


def render_video(bag: str, out: str, fps: int = FPS) -> int:
    """Render the trimmed L/C/R insertion video from a bag to ``out``.

    Args:
        bag: Path to the rosbag2/MCAP directory.
        out: Output ``.mp4`` path.
        fps: Output frame rate.

    Returns:
        The number of frames written.

    Raises:
        ValueError: If the bag contains no ``pose_commands`` (no task window).
    """
    import cv2
    import numpy as np
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_KILTED)
    imgs: dict[str, list[tuple[float, np.ndarray]]] = {
        'left': [], 'center': [], 'right': []
    }
    cmd_times: list[float] = []
    print(f"Reading {bag} ...")
    with Reader(bag) as reader:
        for conn, t_ns, raw in reader.messages():
            t = t_ns * 1e-9
            if conn.topic in (
                '/left_camera/image', '/center_camera/image', '/right_camera/image'
            ):
                cam = conn.topic.split('/')[1].replace('_camera', '')
                m = ts.deserialize_cdr(raw, conn.msgtype)
                a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                imgs[cam].append((t, a))
            elif conn.topic == '/aic_controller/pose_commands':
                cmd_times.append(t)

    if not cmd_times:
        raise ValueError(f"no pose_commands in {bag}; cannot determine task window")

    t0, t1 = min(cmd_times), max(cmd_times) + CMD_MARGIN
    center = select_window(imgs['center'], t0, t1)
    print(f"task window {t1 - t0:.1f}s | center frames in window: {len(center)}")

    W = CW * 3
    H = CH + HEADER
    _fflog = open('/tmp/ffmpeg_video.log', 'wb')
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
         '-r', str(fps), '-i', '-', '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
         '-b:v', '5M', out],
        stdin=subprocess.PIPE, stdout=_fflog, stderr=_fflog)

    n = len(center)
    for i, (t, c) in enumerate(center):
        l = nearest(imgs['left'], t); r = nearest(imgs['right'], t)
        tiles = []
        for name, im in (('LEFT', l), ('CENTER', c), ('RIGHT', r)):
            im = cv2.cvtColor(cv2.resize(im, (CW, CH)), cv2.COLOR_RGB2BGR)
            cv2.putText(im, name, (10, CH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(im)
        row = np.hstack(tiles)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[HEADER:] = row
        cv2.putText(canvas, "AIC SFP insertion  -  CheatCode (score 93.2)", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"t={t - t0:5.1f}s  {i + 1}/{n}", (W - 230, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        # progress bar
        cv2.rectangle(canvas, (10, 33), (W - 240, 37), (60, 60, 60), -1)
        cv2.rectangle(canvas, (10, 33), (10 + int((W - 250) * (i + 1) / n), 37), (0, 220, 255), -1)
        ff.stdin.write(canvas.tobytes())

    ff.stdin.close(); ff.wait()
    sz = os.path.getsize(out) / 1e6
    print(f"Wrote {out}  ({n} frames @ {fps}fps = {n / fps:.1f}s, {sz:.1f} MB)")
    return n


def main() -> None:
    """CLI entry point: render a video from ``argv[1]`` (bag) to ``argv[2]`` (out)."""
    bag = (
        sys.argv[1] if len(sys.argv) > 1
        else os.path.expanduser("~/data/demos/one_20260617_233031")
    )
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/kiwoos/work/insertion_demo.mp4'
    render_video(bag, out)


if __name__ == '__main__':
    main()
