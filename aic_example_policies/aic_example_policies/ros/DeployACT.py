#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
# Deploy a standalone ACT-style policy (trained by ~/training/train_v2.py) into the
# aic_model framework. Self-contained: embeds the model arch + matches train_v2
# preprocessing (3 cams resized to 128, TCP-pose state, 6-D twist action chunk).
# Checkpoint via env AIC_CKPT (default: the latest v2 checkpoint).
import os, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Wrench
from aic_model.policy import (GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback)
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode

IMG = 128

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
        self.get_logger().info(f"DeployACT loaded {ckpt_path} (K={self.K}) on {self.device}")

    def _img(self, raw):
        a = np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)
        a = cv2.resize(a, (288, 256), interpolation=cv2.INTER_AREA)   # match prepare_dataset
        t = torch.from_numpy(a).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", align_corners=False)  # match train_v2
        return (t - 0.5) / 0.5

    def insert_cable(self, task: Task, get_observation: GetObservationCallback,
                     move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback, **kwargs):
        self.get_logger().info(f"DeployACT.insert_cable enter. Task: {task.port_name}")
        start = time.time()
        while time.time() - start < 30.0:
            loop = time.time()
            obs = get_observation()
            if obs is None:
                continue
            imgs = torch.stack([self._img(obs.left_image), self._img(obs.center_image),
                                self._img(obs.right_image)], 1)      # (1,3,3,H,W)
            p = obs.controller_state.tcp_pose
            state = torch.tensor([[p.position.x, p.position.y, p.position.z,
                                   p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]],
                                 device=self.device, dtype=torch.float32)
            state = (state - self.smean) / self.sstd
            with torch.inference_mode():
                pred = self.model(imgs, state)[0, 0]                  # first action of chunk (6,)
            act = (pred * self.astd + self.amean).cpu().numpy()
            twist = Twist(linear=Vector3(x=float(act[0]), y=float(act[1]), z=float(act[2])),
                          angular=Vector3(x=float(act[3]), y=float(act[4]), z=float(act[5])))
            move_robot(motion_update=self._twist_cmd(twist))
            send_feedback("DeployACT in progress")
            time.sleep(max(0, 0.25 - (time.time() - loop)))
        self.get_logger().info("DeployACT.insert_cable exiting")
        return True

    def _twist_cmd(self, twist: Twist, frame_id: str = "base_link"):
        m = MotionUpdate()
        m.velocity = twist
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        m.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
        m.feedforward_wrench_at_tip = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0),
                                             torque=Vector3(x=0.0, y=0.0, z=0.0))
        m.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        m.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return m
