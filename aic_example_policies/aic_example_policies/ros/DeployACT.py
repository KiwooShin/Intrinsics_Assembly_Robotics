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
from aic_example_policies.ros.port_offset import (
    PortOffsetPrediction,
    predict_from_aux,
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
    """Inference mirror of ``train_v2.Policy`` (kept byte-identical in lockstep).

    When ``aux_dim > 0`` the port-bearing auxiliary head is added and ``forward``
    returns ``(action, aux)``; with ``aux_dim == 0`` (old checkpoints) the module
    and forward output are identical to the legacy policy.
    """

    def __init__(self, K, state_dim=7, feat=128, aux_dim=0):
        super().__init__(); self.K = K
        self.aux_dim = aux_dim
        self.enc = _Encoder(feat)
        self.head = nn.Sequential(nn.Linear(feat * 3 + state_dim, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, K * 6))
        if aux_dim > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(feat * 3 + state_dim, 256), nn.ReLU(),
                nn.Linear(256, aux_dim))
    def forward(self, imgs, state):
        B = imgs.shape[0]
        f = self.enc(imgs.view(B * 3, *imgs.shape[2:])).view(B, -1)
        h = torch.cat([f, state], 1)
        act = self.head(h).view(B, self.K, 6)
        if self.aux_dim > 0:
            return act, self.aux_head(h)
        return act


class _AuxBearingProvider:
    """Adapts ``DeployACT._predict_offset`` to the guarded ``PortBearingProvider``.

    Holds the latest observation (set by the deploy loop each cycle) and, when
    queried by ``GuardedDescentController``, runs the aux head on it and returns
    the base_link port target with a plausibility ``ok`` flag. Kept tiny so the
    guarded-descent module stays torch/ROS-free.
    """

    def __init__(self, deploy: "DeployACT", min_mag: float, max_mag: float) -> None:
        """Initialize the provider.

        Args:
            deploy: The owning :class:`DeployACT` (runs the aux head).
            min_mag: Lower plausible offset magnitude (m).
            max_mag: Upper plausible offset magnitude (m).
        """
        self._deploy = deploy
        self._min_mag = float(min_mag)
        self._max_mag = float(max_mag)
        self.obs = None

    def predict(self, position: np.ndarray, quaternion: np.ndarray):
        """Return ``(target_base, magnitude, ok)`` for the latest observation.

        Args:
            position: Current TCP position ``(3,)`` (unused; the offset is
                resolved from ``obs`` for image+pose consistency).
            quaternion: Current TCP orientation ``(4,)`` (unused; see above).

        Returns:
            ``(target_base, magnitude, ok)`` per the ``PortBearingProvider``
            protocol; ``(None, 0.0, False)`` when no observation/aux head.
        """
        if self.obs is None:
            return None, 0.0, False
        pred = self._deploy._predict_offset(self.obs)
        if pred is None:
            return None, 0.0, False
        return pred.target_base, pred.magnitude, pred.plausible(self._min_mag, self._max_mag)


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
        # Port-bearing auxiliary head (docs/design_port_aux_head.md), opt-in by
        # the checkpoint: old checkpoints lack ``has_aux`` -> aux_dim 0 -> a plain
        # _Policy, byte-identical. ``omean/ostd/aux_frame`` de-normalize the aux
        # output and resolve it into base_link at deploy time.
        self.has_aux = bool(ck.get("has_aux", False))
        self.aux_dim = int(ck.get("aux_dim", 0)) if self.has_aux else 0
        self.aux_frame = str(ck.get("aux_frame", "tcp"))
        self.omean = ck["omean"].to(self.device) if self.aux_dim > 0 else None
        self.ostd = ck["ostd"].to(self.device) if self.aux_dim > 0 else None
        self.model = _Policy(self.K, state_dim=self.state_dim, aux_dim=self.aux_dim).to(self.device)
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
        # Learned port-bearing handoff (AIC_GUARDED_AUX): only when the guarded
        # probe is on AND this checkpoint carries an aux head. Otherwise the
        # guarded controller uses the motion-axis estimate unchanged.
        self._aux_bearing_enabled = bool(
            self.has_aux
            and self._guarded_config is not None
            and self._guarded_config.use_aux_bearing
        )
        guarded_banner = "OFF"
        if self._guarded_config is not None:
            gc = self._guarded_config
            aux_tag = (
                f", aux_bearing=ON(frame={self.aux_frame}, "
                f"mag[{gc.aux_min_mag * 1e3:g},{gc.aux_max_mag * 1e3:g}]mm, "
                f"reaim={gc.reaim})"
                if self._aux_bearing_enabled
                else ""
            )
            guarded_banner = (
                f"ON (speed<{gc.speed_threshold:g} m/s for {gc.stall_window_s:g}s after "
                f"{gc.min_runtime_s:g}s, step={gc.step_size * 1e3:g}mm "
                f"cap={gc.travel_cap * 1e3:g}mm force_thr={gc.contact_force_threshold:g}N, "
                f"z_stiff={gc.z_stiffness}{aux_tag})"
            )
        ensemble_banner = (
            "ON (m=%g, w_i=exp(-m*i), i=0 oldest)" % self._ensemble.decay
            if self._ensemble is not None
            else "OFF"
        )
        aux_tag = (
            f", +port_aux(dim={self.aux_dim},frame={self.aux_frame})"
            if self.aux_dim > 0 else ""
        )
        self.get_logger().info(
            f"DeployACT loaded {ckpt_path} (K={self.K}, state_dim={self.state_dim}"
            f"{', +wrist_wrench' if self.state_dim == state_assembly.POSE_WRENCH_DIM else ''}"
            f"{aux_tag}) "
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

    def _forward(self, imgs, state):
        """Run the policy, returning ``(action, aux_or_None)`` uniformly.

        Args:
            imgs: Image tensor ``(1, 3, 3, H, W)``.
            state: Normalized state tensor ``(1, state_dim)``.

        Returns:
            ``(action, aux)`` where ``action`` is ``(1, K, 6)`` and ``aux`` is
            ``(1, aux_dim)`` when the aux head is present, else ``None``.
        """
        out = self.model(imgs, state)
        if self.aux_dim > 0:
            return out
        return out, None

    def _prepare_inputs(self, obs):
        """Build the normalized network inputs and the raw TCP pose from an obs.

        Args:
            obs: The current ``Observation`` message.

        Returns:
            ``(imgs, state, pose)`` where ``imgs`` is ``(1, 3, 3, H, W)``,
            ``state`` is the normalized ``(1, state_dim)`` tensor, and ``pose`` is
            the raw ``[x, y, z, qx, qy, qz, qw]`` base_link TCP pose (numpy).
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
        return imgs, state, pose

    def _predict_chunk(self, obs) -> np.ndarray:
        """Run inference and return the denormalized twist chunk.

        Args:
            obs: The current ``Observation`` message.

        Returns:
            A ``(K, 6)`` numpy array of base_link-frame twists (m/s, rad/s).
        """
        imgs, state, _ = self._prepare_inputs(obs)
        with torch.inference_mode():
            act, _ = self._forward(imgs, state)                    # act (1,K,6) norm
        return (act[0] * self.astd + self.amean).cpu().numpy()

    def _predict_offset(self, obs) -> PortOffsetPrediction | None:
        """Run the aux head and resolve a base_link TCP->port prediction.

        Args:
            obs: The current ``Observation`` message.

        Returns:
            The :class:`PortOffsetPrediction` (offset/target in base_link), or
            None when this checkpoint has no aux head.
        """
        if self.aux_dim == 0:
            return None
        imgs, state, pose = self._prepare_inputs(obs)
        with torch.inference_mode():
            _, aux = self._forward(imgs, state)                    # aux (1,aux_dim) norm
        aux_vec = (aux[0] * self.ostd + self.omean).cpu().numpy()  # de-normalized
        return predict_from_aux(pose[:3], pose[3:7], aux_vec, frame=self.aux_frame)

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
        # keeping the loop below byte-identical to the plain policy). When the
        # learned port bearing is enabled, wire a per-episode provider whose
        # ``obs`` the loop refreshes each cycle so the aux head aims the handoff.
        aux_provider = (
            _AuxBearingProvider(
                self, self._guarded_config.aux_min_mag, self._guarded_config.aux_max_mag
            )
            if self._guarded_config is not None and self._aux_bearing_enabled
            else None
        )
        guarded = (
            GuardedDescentController(self._guarded_config, bearing_provider=aux_provider)
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
                if aux_provider is not None:
                    aux_provider.obs = obs  # refresh the aux head's live obs
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
