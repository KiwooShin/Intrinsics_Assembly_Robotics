"""Convert collected rosbag demos to a LeRobot-compatible dataset for ACT fine-tuning.

Input: MCAP bags with camera images, controller_state (TCP velocity = actions),
joint_states.
Output: per-episode ``.npy`` arrays (images + TCP pose/velocity + timestamps)
compatible with the ACT trainers in this repo.

The heavy ROS/cv2 dependencies are imported lazily inside :func:`process_bag`, so
the pure trimming/synchronisation/serialisation helpers can be imported and unit
tested without ROS, Gazebo, or a GPU.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

import numpy as np


@dataclasses.dataclass
class EpisodeFrame:
    """A single synchronised (image, state, action) sample.

    Attributes:
        timestamp: Capture time of the center image, in seconds.
        left_image: Left camera frame, ``uint8`` array shaped ``(H, W, 3)``.
        center_image: Center camera frame, ``uint8`` array shaped ``(H, W, 3)``.
        right_image: Right camera frame, ``uint8`` array shaped ``(H, W, 3)``.
        tcp_pose: TCP pose as ``[x, y, z, qx, qy, qz, qw]`` (7 values).
        tcp_velocity: TCP velocity as ``[vx, vy, vz, wx, wy, wz]`` (the action label).
        tcp_error: TCP tracking error (6 values).
    """

    timestamp: float
    left_image: np.ndarray
    center_image: np.ndarray
    right_image: np.ndarray
    tcp_pose: list[float]
    tcp_velocity: list[float]
    tcp_error: list[float]


def compute_task_window(
    cmd_times: list[float], margin: float = 0.3
) -> tuple[float, float]:
    """Compute the [start, end] time window of the task-execution phase.

    The model publishes ``pose_commands`` only while executing the insertion, so
    ``[first_cmd, last_cmd + margin]`` brackets the real demonstration and excludes
    the pre-task idle and the post-task reset/teleport back to home. When no command
    timestamps are present the window is unbounded (keep everything) as a fallback.

    Args:
        cmd_times: Timestamps (seconds) of ``/aic_controller/pose_commands`` messages.
        margin: Extra seconds kept after the last command to retain the final
            seated frame. Must be non-negative.

    Returns:
        A ``(t_start, t_end)`` tuple in seconds. Returns ``(-inf, +inf)`` when
        ``cmd_times`` is empty.

    Raises:
        ValueError: If ``margin`` is negative.
    """
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    if not cmd_times:
        return float("-inf"), float("inf")
    return min(cmd_times), max(cmd_times) + margin


def synchronize_frames(
    center: list[tuple[float, np.ndarray]],
    left: list[tuple[float, np.ndarray]],
    right: list[tuple[float, np.ndarray]],
    controller: list[tuple[float, list[float], list[float], list[float]]],
    t_start: float,
    t_end: float,
    max_dt: float = 0.5,
) -> list[EpisodeFrame]:
    """Synchronise camera frames with the nearest controller state, within the window.

    For every center frame inside ``[t_start, t_end]`` the nearest (in time) left
    frame, right frame, and controller state are selected. Frames whose nearest
    controller state is more than ``max_dt`` seconds away are dropped.

    Args:
        center: ``(timestamp, image)`` tuples for the center camera.
        left: ``(timestamp, image)`` tuples for the left camera.
        right: ``(timestamp, image)`` tuples for the right camera.
        controller: ``(timestamp, pose, velocity, error)`` tuples.
        t_start: Inclusive lower bound of the task window (seconds).
        t_end: Inclusive upper bound of the task window (seconds).
        max_dt: Maximum allowed |image - controller| time gap (seconds).

    Returns:
        The list of synchronised :class:`EpisodeFrame` records, in center-frame order.
        Empty if either side camera or the controller stream is empty.
    """
    frames: list[EpisodeFrame] = []
    if not left or not right or not controller:
        return frames
    for t_img, img_c in center:
        if not (t_start <= t_img <= t_end):
            continue
        t_left, img_left = min(left, key=lambda x: abs(x[0] - t_img))
        t_right, img_right = min(right, key=lambda x: abs(x[0] - t_img))
        t_ctrl, pose, vel, err = min(controller, key=lambda x: abs(x[0] - t_img))
        if abs(t_img - t_ctrl) > max_dt:
            continue
        frames.append(
            EpisodeFrame(
                timestamp=t_img,
                left_image=img_left,
                center_image=img_c,
                right_image=img_right,
                tcp_pose=pose,
                tcp_velocity=vel,
                tcp_error=err,
            )
        )
    return frames


def save_episodes(frames: list[EpisodeFrame], output_dir: str) -> int:
    """Serialise synchronised frames to per-field ``.npy`` arrays.

    Args:
        frames: Synchronised episode frames to save.
        output_dir: Directory to write the ``.npy`` files into (created if missing).

    Returns:
        The number of frames written (0 when ``frames`` is empty).
    """
    if not frames:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/center_images.npy", np.array([f.center_image for f in frames]))
    np.save(f"{output_dir}/left_images.npy", np.array([f.left_image for f in frames]))
    np.save(f"{output_dir}/right_images.npy", np.array([f.right_image for f in frames]))
    np.save(f"{output_dir}/tcp_velocities.npy", np.array([f.tcp_velocity for f in frames]))
    np.save(f"{output_dir}/tcp_poses.npy", np.array([f.tcp_pose for f in frames]))
    np.save(f"{output_dir}/timestamps.npy", np.array([f.timestamp for f in frames]))
    return len(frames)


def _build_typestore() -> Any:
    """Build a ROS 2 typestore with the project's custom message definitions.

    Returns:
        A ``rosbags`` typestore able to deserialise ``aic_control_interfaces`` messages.
    """
    from pathlib import Path

    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts_store = get_typestore(Stores.ROS2_KILTED)
    msg_dir = Path("/home/kiwoos/ws_aic/install/share/aic_control_interfaces/msg/")
    if msg_dir.exists():
        for mf in msg_dir.glob("*.msg"):
            try:
                ts_store.register(
                    get_types_from_msg(
                        mf.read_text(), f"aic_control_interfaces/msg/{mf.stem}"
                    )
                )
            except Exception:  # noqa: BLE001 - a single bad .msg must not abort the run
                pass
    return ts_store


def process_bag(bag_path: str, output_dir: str) -> int:
    """Extract synchronised (state, action, image) tuples from a bag and save them.

    Args:
        bag_path: Path to the rosbag2/MCAP directory to read.
        output_dir: Directory to write the converted ``.npy`` dataset into.

    Returns:
        The number of synchronised frames written.
    """
    import cv2
    from rosbags.rosbag2 import Reader

    os.makedirs(output_dir, exist_ok=True)
    ts_store = _build_typestore()

    controller_data: list[tuple[float, list[float], list[float], list[float]]] = []
    images: dict[str, list[tuple[float, np.ndarray]]] = {
        "left": [],
        "center": [],
        "right": [],
    }
    wrench_data: list[tuple[float, list[float]]] = []
    cmd_times: list[float] = []
    insertion_times: list[float] = []

    print(f"Reading {bag_path}...")
    with Reader(bag_path) as reader:
        for conn, ts, raw in reader.messages():
            t = ts * 1e-9  # nanoseconds to seconds

            if conn.topic == "/aic_controller/controller_state":
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    p = msg.tcp_pose.position
                    q = msg.tcp_pose.orientation
                    v = msg.tcp_velocity
                    pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                    vel = [
                        v.linear.x, v.linear.y, v.linear.z,
                        v.angular.x, v.angular.y, v.angular.z,
                    ]
                    err = list(msg.tcp_error)[:6]
                    controller_data.append((t, pose, vel, err))
                except Exception:  # noqa: BLE001 - skip undecodable messages
                    pass

            elif conn.topic in (
                "/left_camera/image", "/center_camera/image", "/right_camera/image"
            ):
                cam = conn.topic.split("/")[1].replace("_camera", "")
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                    # Scale to 25% (288x256) matching RunACT - proper bilinear resize.
                    scaled = cv2.resize(arr, (288, 256), interpolation=cv2.INTER_AREA)
                    images[cam].append((t, scaled))
                except Exception:  # noqa: BLE001 - skip undecodable messages
                    pass

            elif conn.topic == "/fts_broadcaster/wrench":
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    f = msg.wrench.force
                    torq = msg.wrench.torque
                    wrench_data.append(
                        (t, [f.x, f.y, f.z, torq.x, torq.y, torq.z])
                    )
                except Exception:  # noqa: BLE001 - skip undecodable messages
                    pass

            elif conn.topic == "/aic_controller/pose_commands":
                cmd_times.append(t)  # only need the timing of model activity

            elif conn.topic == "/scoring/insertion_event":
                insertion_times.append(t)

    print(
        f"  controller: {len(controller_data)} | images per cam: "
        f"{len(images['center'])} | wrench: {len(wrench_data)}"
    )

    t_start, t_end = compute_task_window(cmd_times)
    span = (t_end - t_start) if cmd_times else 0.0
    print(
        f"  pose_commands: {len(cmd_times)} | insertion_events: "
        f"{len(insertion_times)} | task window: {span:.1f}s"
    )

    n_total = len(images["center"])
    frames = synchronize_frames(
        images["center"], images["left"], images["right"],
        controller_data, t_start, t_end,
    )
    print(f"  Synchronized frames: {len(frames)} (kept from {n_total} total after trim)")

    n_saved = save_episodes(frames, output_dir)
    if n_saved:
        print(f"  Saved {n_saved} frames to {output_dir}")
    return n_saved


if __name__ == "__main__":
    bag = (
        sys.argv[1] if len(sys.argv) > 1
        else os.path.expanduser("~/data/demos/sample_0_20260531_205719")
    )
    out = (
        sys.argv[2] if len(sys.argv) > 2
        else os.path.expanduser("~/training/episode_0")
    )
    n = process_bag(bag, out)
    print(f"Done: {n} frames")
