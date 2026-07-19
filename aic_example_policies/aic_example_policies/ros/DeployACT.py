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
from aic_example_policies.ros.chunk_ensemble import (
    DEFAULT_DECAY,
    ChunkEnsemble,
    select_exec_twists,
)
from aic_example_policies.ros import state_assembly
from aic_example_policies.ros.guarded_descent import (
    GuardedDescentConfig,
    GuardedDescentController,
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
        # State dimensionality: 7-D (TCP pose) checkpoints predate the wrench
        # upgrade and lack this key, so default to 7 for byte-identical behavior;
        # 13-D checkpoints append the live wrist wrench to the state. Same
        # opt-in pattern as AIC_ENSEMBLE: the checkpoint decides, no env needed.
        self.state_dim = int(ck.get("state_dim", state_assembly.POSE_DIM))
        self.model = _Policy(self.K, state_dim=self.state_dim).to(self.device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.amean = ck["amean"].to(self.device); self.astd = ck["astd"].to(self.device)
        self.smean = ck["smean"].to(self.device); self.sstd = ck["sstd"].to(self.device)
        self.exec_steps = min(EXEC_STEPS, self.K)
        # ACT temporal ensembling (arXiv:2304.13705), opt-in via AIC_ENSEMBLE=1.
        # Read once here at node start (same lifecycle as AIC_CKPT). When unset
        # the ensemble is None and the execution path is byte-identical to the
        # plain receding-horizon behavior. Weight decay m via AIC_ENSEMBLE_M.
        self._ensemble: ChunkEnsemble | None = None
        if os.environ.get("AIC_ENSEMBLE", "0").strip() == "1":
            m = float(os.environ.get("AIC_ENSEMBLE_M", str(DEFAULT_DECAY)))
            self._ensemble = ChunkEnsemble(decay=m)
        # Guarded-descent probe (SESSION_REPORT.md 2026-07-19 stall analysis),
        # opt-in via AIC_GUARDED=1 (+ AIC_GUARDED_* overrides). Same lifecycle as
        # AIC_ENSEMBLE: the config is parsed once here; when disabled it is None
        # and the execution path below is byte-identical to the plain policy. A
        # fresh GuardedDescentController is built per episode in insert_cable.
        self._guarded_config: GuardedDescentConfig | None = None
        _gcfg = GuardedDescentConfig.from_env(os.environ)
        if _gcfg.enabled:
            self._guarded_config = _gcfg
        guarded_banner = "OFF"
        if self._guarded_config is not None:
            gc = self._guarded_config
            guarded_banner = (
                f"ON (speed<{gc.speed_threshold:g} m/s for {gc.stall_window_s:g}s after "
                f"{gc.min_runtime_s:g}s, step={gc.step_size * 1e3:g}mm "
                f"cap={gc.travel_cap * 1e3:g}mm force_thr={gc.contact_force_threshold:g}N, "
                f"z_stiff={gc.z_stiffness})"
            )
        ensemble_banner = (
            "ON (m=%g, w_i=exp(-m*i), i=0 oldest)" % self._ensemble.decay
            if self._ensemble is not None
            else "OFF"
        )
        self.get_logger().info(
            f"DeployACT loaded {ckpt_path} (K={self.K}, state_dim={self.state_dim}"
            f"{', +wrist_wrench' if self.state_dim == state_assembly.POSE_WRENCH_DIM else ''}) "
            f"on {self.device}; "
            f"MODE_POSITION receding horizon exec_steps={self.exec_steps} substeps={SUBSTEPS}; "
            f"temporal_ensembling={ensemble_banner}; guarded_descent={guarded_banner}"
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
        pose = np.array([p.position.x, p.position.y, p.position.z,
                         p.orientation.x, p.orientation.y, p.orientation.z,
                         p.orientation.w], dtype=np.float32)
        # For a 13-D (wrench-augmented) checkpoint, append the live wrist wrench in
        # the training order [fx, fy, fz, tx, ty, tz]; a 7-D checkpoint ignores it.
        wrench = None
        if self.state_dim == state_assembly.POSE_WRENCH_DIM:
            w = obs.wrist_wrench.wrench
            wrench = np.array([w.force.x, w.force.y, w.force.z,
                               w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)
        raw_state = state_assembly.assemble_state(pose, wrench, self.state_dim)
        state = torch.from_numpy(raw_state).to(self.device).unsqueeze(0)
        state = (state - self.smean) / self.sstd
        with torch.inference_mode():
            pred = self.model(imgs, state)[0]                      # (K, 6) normalized
        return (pred * self.astd + self.amean).cpu().numpy()

    @staticmethod
    def _wrench_force(obs) -> np.ndarray:
        """Return the wrist force ``[fx, fy, fz]`` (N) from an observation.

        Args:
            obs: The current ``Observation`` message.

        Returns:
            A ``(3,)`` numpy array of the wrist-wrench force components.
        """
        f = obs.wrist_wrench.wrench.force
        return np.array([f.x, f.y, f.z], dtype=np.float64)

    def _command_targets(self, move_robot: MoveRobotCallback, targets: np.ndarray,
                         stiffness: list, dt_fine: float) -> None:
        """Command a sequence of absolute pose targets via ``set_pose_target``.

        Args:
            move_robot: Callback that forwards a ``MotionUpdate`` to the controller.
            targets: An ``(n, 7)`` array of ``[x, y, z, qx, qy, qz, qw]`` poses.
            stiffness: The 6-D MODE_POSITION stiffness to command.
            dt_fine: Sleep between consecutive sub-pose commands (s).
        """
        for tgt in targets:
            pose = Pose(
                position=Point(x=float(tgt[0]), y=float(tgt[1]), z=float(tgt[2])),
                orientation=Quaternion(x=float(tgt[3]), y=float(tgt[4]),
                                       z=float(tgt[5]), w=float(tgt[6])),
            )
            self.set_pose_target(move_robot=move_robot, pose=pose,
                                 stiffness=stiffness, damping=POSE_DAMPING)
            self.sleep_for(dt_fine)

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
        # Absolute frame-step counter for temporal ensembling: chunk from cycle c
        # starts at abs_step = c*exec_steps and covers [abs_step, abs_step+K-1].
        # Advanced only when a cycle actually executes (not on a dropped obs).
        abs_step = 0
        if self._ensemble is not None:
            self._ensemble.clear()
        # Fresh guarded-descent controller per episode (None when AIC_GUARDED off,
        # keeping the loop below byte-identical to the plain policy).
        guarded = (
            GuardedDescentController(self._guarded_config)
            if self._guarded_config is not None
            else None
        )
        while (self.time_now() - start) < budget:
            obs = get_observation()
            if obs is None:
                continue
            # Guarded-descent probe (opt-in). While approaching this only observes
            # and returns targets=None so the learned path below runs unchanged;
            # once the stall fires it returns scripted descent targets to command
            # in place of the learned chunk. The whole block is skipped when the
            # probe is disabled, so DeployACT is byte-identical with AIC_GUARDED
            # unset.
            if guarded is not None:
                p_g = obs.controller_state.tcp_pose
                pos_g = np.array([p_g.position.x, p_g.position.y, p_g.position.z])
                quat_g = np.array([p_g.orientation.x, p_g.orientation.y,
                                   p_g.orientation.z, p_g.orientation.w])
                elapsed_g = (self.time_now() - start).nanoseconds * 1e-9
                gstep = guarded.cycle(elapsed_g, pos_g, quat_g,
                                      self._wrench_force(obs), substeps=SUBSTEPS)
                for line in gstep.log_lines:
                    self.get_logger().info(line)
                if gstep.targets is not None:
                    stiffness = gstep.stiffness if gstep.stiffness is not None else POSE_STIFFNESS
                    self._command_targets(move_robot, gstep.targets, stiffness, dt_fine)
                    send_feedback("DeployACT guarded descent")
                    if elapsed_g - last_log >= 2.0:
                        last_log = elapsed_g
                        self.get_logger().info(
                            f"t={elapsed_g:4.1f}s GUARDED phase={gstep.phase} "
                            f"steps={gstep.steps} travel={gstep.travel*1e3:.1f}mm "
                            f"|F-base|={gstep.force_delta:.2f}N contacts={gstep.contacts} "
                            f"tcp=({pos_g[0]:+.3f},{pos_g[1]:+.3f},{pos_g[2]:+.3f})"
                        )
                    continue
            chunk = self._predict_chunk(obs)                       # (K, 6)
            p = obs.controller_state.tcp_pose
            pos0 = np.array([p.position.x, p.position.y, p.position.z])
            quat0 = np.array([p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w])
            # Select the first exec_steps twists to execute. With ensembling OFF
            # this is chunk[:exec_steps] unchanged; with it ON each executed step
            # is the exp-weighted average of all buffered chunks covering it
            # (breaking the near-port zero-velocity attractor). Then integrate
            # (sub-stepped) into absolute base_link pose targets re-anchored to
            # the measured pose pos0/quat0.
            exec_twists = select_exec_twists(
                self._ensemble, chunk, abs_step, self.exec_steps
            )
            abs_step += self.exec_steps
            fine = expand_twists(exec_twists, SUBSTEPS)
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
