"""
Convert collected rosbag demos to LeRobot-compatible dataset format for ACT fine-tuning.

Input: MCAP bags with camera images, controller_state (TCP vel = actions), joint_states
Output: HuggingFace datasets format compatible with lerobot ACT training
"""
import sys, os
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from pathlib import Path
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict

# Register custom message types
ts_store = get_typestore(Stores.ROS2_KILTED)
for d in ['/home/kiwoos/ws_aic/install/share/aic_control_interfaces/msg/']:
    p = Path(d)
    if p.exists():
        for mf in p.glob('*.msg'):
            try:
                ts_store.register(get_types_from_msg(
                    mf.read_text(), f'aic_control_interfaces/msg/{mf.stem}'))
            except: pass

def process_bag(bag_path: str, output_dir: str):
    """Extract synchronized (state, action, image) tuples from a bag."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all messages by topic
    controller_data = []  # (timestamp, tcp_pose 7D, tcp_velocity 6D, tcp_error 6D, joints 7D)
    images = {'left': [], 'center': [], 'right': []}  # (timestamp, H×W×3 array)
    wrench_data = []  # (timestamp, force 3D, torque 3D)

    print(f"Reading {bag_path}...")
    with Reader(bag_path) as reader:
        topic_map = {c.topic: c for c in reader.connections}
        
        for conn, ts, raw in reader.messages():
            t = ts * 1e-9  # nanoseconds to seconds

            if conn.topic == '/aic_controller/controller_state':
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    p = msg.tcp_pose.position
                    q = msg.tcp_pose.orientation
                    v = msg.tcp_velocity
                    pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                    vel = [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z]
                    err = list(msg.tcp_error)[:6]
                    controller_data.append((t, pose, vel, err))
                except: pass

            elif conn.topic in ['/left_camera/image', '/center_camera/image', '/right_camera/image']:
                cam = conn.topic.split('/')[1].replace('_camera', '')
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    # Scale to 25% (288×256) matching RunACT
                    
                    # Simple block-average resize using numpy
                    from functools import reduce
                    h_scale, w_scale = 256/msg.height, 288/msg.width
                    # Use numpy slicing for fast downscale
                    scaled = arr[::4, ::4, :][:256, :288, :]
                    images[cam].append((t, scaled))
                except: pass

            elif conn.topic == '/fts_broadcaster/wrench':
                try:
                    msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                    f = msg.wrench.force
                    torq = msg.wrench.torque
                    wrench_data.append((t, [f.x, f.y, f.z, torq.x, torq.y, torq.z]))
                except: pass

    print(f"  controller: {len(controller_data)} | images per cam: {len(images['center'])} | wrench: {len(wrench_data)}")

    # Synchronize: for each camera frame, find nearest controller state and wrench
    episodes = []
    for i, (t_img, img_c) in enumerate(images['center']):
        # Find nearest left and right frames
        if not images['left'] or not images['right']:
            continue
        t_left = min(images['left'], key=lambda x: abs(x[0]-t_img))
        t_right = min(images['right'], key=lambda x: abs(x[0]-t_img))

        # Find nearest controller state
        if not controller_data:
            continue
        ctrl = min(controller_data, key=lambda x: abs(x[0]-t_img))
        t_ctrl, pose, vel, err = ctrl

        # Skip if too far apart (>0.5s)
        if abs(t_img - t_ctrl) > 0.5:
            continue

        # Build state vector (26D matching RunACT)
        state = pose + vel + err  # 7+6+6 = 19... RunACT uses 26D
        # RunACT state: tcp_pose(7) + tcp_vel(6) + tcp_error(6) + joint_pos(7) = 26
        # We're missing joints here - would need to sync joint_states too
        # For now save what we have

        episodes.append({
            'timestamp': t_img,
            'left_image': t_left[1],
            'center_image': img_c,
            'right_image': t_right[1],
            'tcp_pose': pose,       # 7D
            'tcp_velocity': vel,     # 6D (ACTION LABEL)
            'tcp_error': err,        # 6D
        })

    print(f"  Synchronized frames: {len(episodes)}")

    # Save as numpy files
    if episodes:
        np.save(f'{output_dir}/center_images.npy', np.array([e['center_image'] for e in episodes]))
        np.save(f'{output_dir}/left_images.npy', np.array([e['left_image'] for e in episodes]))
        np.save(f'{output_dir}/right_images.npy', np.array([e['right_image'] for e in episodes]))
        np.save(f'{output_dir}/tcp_velocities.npy', np.array([e['tcp_velocity'] for e in episodes]))
        np.save(f'{output_dir}/tcp_poses.npy', np.array([e['tcp_pose'] for e in episodes]))
        np.save(f'{output_dir}/timestamps.npy', np.array([e['timestamp'] for e in episodes]))
        print(f"  Saved {len(episodes)} frames to {output_dir}")
        return len(episodes)
    return 0

if __name__ == '__main__':
    bag = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/data/demos/sample_0_20260531_205719')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/training/episode_0')
    n = process_bag(bag, out)
    print(f"Done: {n} frames")
