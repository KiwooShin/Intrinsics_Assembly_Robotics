#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
# Deploy a standalone ACT-style policy (trained by ~/training/train_v2.py) into the
# aic_model framework. Self-contained: embeds the model arch + matches train_v2
# preprocessing (3 cams resized to 128, TCP-pose state, 6-D twist action chunk).
# Checkpoint via env AIC_CKPT (default: the latest v2 checkpoint).
#
# Execution model: the policy predicts a chunk of base_link-frame twists. Rather
# than streaming them as MODE_VELOCITY setpoints (which the controller integrates
# fully open-loop from its own last reference and never re-anchors to the measured
# pose -- so per-config model bias drifts the arm away or stalls it), we integrate
# the predicted twist chunk into absolute base_link pose targets, re-anchored to
# the measured TCP pose at every inference (receding horizon), and command them via
# MODE_POSITION with the same stiffness/damping the CheatCode oracle used to
# generate the training data. See pose_integration.py for the (ROS-free) math.
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from aic_example_policies.ros.pose_integration import (
    expand_twists,
    integrate_twist_chunk,
)

IMG = 128

# Training frame period (~3.64 Hz recording); the predicted twist chunk is spaced
# one such frame apart. Measured from ds_wide timestamps.npy (median dt 0.275 s).
DT_FRAME = 0.275
# Receding-horizon length: chunk steps executed as absolute pose targets before
# re-inferring against a fresh observation. Smaller = more closed-loop / less drift.
EXEC_STEPS = 4
# Sub-steps per chunk step, for a smoother (~18 Hz) pose-target stream.
SUBSTEPS = 5
# Stiffness/damping matching CheatCode's set_pose_target defaults (the oracle that
# generated the training data), so the deployed controller behavior is consistent.
POSE_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
POSE_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]


class _Encoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, out),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x): return self.net(x)

class _Policy(nn.Module):
    def __init__(self, K, state_dim=7, feat=128):
        super().__init__(); self.K = K
        self.enc = _Encoder(feat)
        self.head = nn.Sequential(nn.Linear(feat * 3 + state_dim, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, K * 6))
    def forward(self, imgs, state):
        B = imgs.shape[0]
        f = self.enc(imgs.view(B * 3, *imgs.shape[2:])).view(B, -1)
        return self.head(torch.cat([f, state], 1)).view(B, self.K, 6)


class DeployACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.environ.get("AIC_CKPT", "/home/kiwoos/training/ckpt/v2_wide.pt")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.K = ck["K"]
        self.model = _Policy(self.K).to(self.device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.amean = ck["amean"].to(self.device); self.astd = ck["astd"].to(self.device)
        self.smean = ck["smean"].to(self.device); self.sstd = ck["sstd"].to(self.device)
        self.exec_steps = min(EXEC_STEPS, self.K)
        self.get_logger().info(
            f"DeployACT loaded {ckpt_path} (K={self.K}) on {self.device}; "
            f"MODE_POSITION receding horizon exec_steps={self.exec_steps} substeps={SUBSTEPS}"
        )

    def _img(self, raw):
        a = np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)
        a = cv2.resize(a, (288, 256), interpolation=cv2.INTER_AREA)   # match prepare_dataset
        t = torch.from_numpy(a).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", align_corners=False)  # train_v2
        return (t - 0.5) / 0.5

    def _predict_chunk(self, obs) -> np.ndarray:
        """Run inference and return the denormalized twist chunk.

        Args:
            obs: The current ``Observation`` message.

        Returns:
            A ``(K, 6)`` numpy array of base_link-frame twists (m/s, rad/s).
        """
        imgs = torch.stack([self._img(obs.left_image), self._img(obs.center_image),
                            self._img(obs.right_image)], 1)      # (1,3,3,H,W)
        p = obs.controller_state.tcp_pose
        state = torch.tensor([[p.position.x, p.position.y, p.position.z,
                               p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]],
                             device=self.device, dtype=torch.float32)
        state = (state - self.smean) / self.sstd
        with torch.inference_mode():
            pred = self.model(imgs, state)[0]                      # (K, 6) normalized
        return (pred * self.astd + self.amean).cpu().numpy()

    def insert_cable(self, task: Task, get_observation: GetObservationCallback,
                     move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback, **kwargs):
        """Drive the arm via receding-horizon MODE_POSITION targets for ~30 s.

        Each cycle: read the current observation, predict a twist chunk, integrate
        the first ``exec_steps`` twists (sub-stepped) into absolute base_link pose
        targets anchored to the *measured* TCP pose, and command them via
        ``set_pose_target`` (MODE_POSITION, CheatCode gains), then re-infer.

        Args:
            task: The insert-cable task (names the target port).
            get_observation: Callback returning the latest ``Observation``.
            move_robot: Callback that forwards a ``MotionUpdate`` to the controller.
            send_feedback: Callback for progress strings to the engine.
            **kwargs: Ignored; accepted for forward compatibility.

        Returns:
            ``True`` when the fixed time budget elapses.
        """
        self.get_logger().info(f"DeployACT.insert_cable enter. Task: {task.port_name}")
        dt_fine = DT_FRAME / SUBSTEPS
        start = self.time_now()
        budget = Duration(seconds=30.0)
        last_log = 0.0
        while (self.time_now() - start) < budget:
            obs = get_observation()
            if obs is None:
                continue
            chunk = self._predict_chunk(obs)                       # (K, 6)
            p = obs.controller_state.tcp_pose
            pos0 = np.array([p.position.x, p.position.y, p.position.z])
            quat0 = np.array([p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w])
            # Integrate the first exec_steps chunk twists (sub-stepped for
            # smoothness) into absolute base_link pose targets, re-anchored to
            # the measured pose pos0/quat0.
            fine = expand_twists(chunk[: self.exec_steps], SUBSTEPS)
            targets = integrate_twist_chunk(pos0, quat0, fine, dt_fine)
            for tgt in targets:
                pose = Pose(
                    position=Point(x=float(tgt[0]), y=float(tgt[1]), z=float(tgt[2])),
                    orientation=Quaternion(x=float(tgt[3]), y=float(tgt[4]),
                                           z=float(tgt[5]), w=float(tgt[6])),
                )
                self.set_pose_target(move_robot=move_robot, pose=pose,
                                     stiffness=POSE_STIFFNESS, damping=POSE_DAMPING)
                self.sleep_for(dt_fine)
            send_feedback("DeployACT in progress")
            elapsed = (self.time_now() - start).nanoseconds * 1e-9
            if elapsed - last_log >= 2.0:
                last_log = elapsed
                lin = float(np.linalg.norm(chunk[0, :3]))
                self.get_logger().info(
                    f"t={elapsed:4.1f}s pred|lin|={lin:.4f} m/s "
                    f"tcp=({pos0[0]:+.3f},{pos0[1]:+.3f},{pos0[2]:+.3f}) "
                    f"target=({targets[-1, 0]:+.3f},{targets[-1, 1]:+.3f},{targets[-1, 2]:+.3f})"
                )
        self.get_logger().info("DeployACT.insert_cable exiting")
        return True
