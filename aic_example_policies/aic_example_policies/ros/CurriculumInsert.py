#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Curriculum evaluation policy: privileged staging, then a learned insertion.

RUN #4 milestone harness (2026-07-21). The curriculum's easiest case starts the
plug already located above the aligned port ("locate the end point on top of the
port" — user directive), so the learned specialist only has to perform the
*insertion motion* (descend + seat). This policy:

1. **Stage (privileged)**: like :class:`CheatCode`, reads the true port TF
   (``ground_truth:=true``, training/curriculum-legal) and drives the plug tip to
   ``CURR_STANDOFF_M`` above the port mouth — optionally displaced laterally by
   ``CURR_LAT_OFFSET_MM`` along ``CURR_LAT_AZIMUTH_DEG`` for the capture-radius
   curriculum (milestone 2+).
2. **Hand off (learned)**: runs the specialist checkpoint ``CURR_CKPT`` (a
   ``train_v3`` last-inch model; 7-D or 13-D wrench-augmented state) in the same
   receding-horizon MODE_POSITION loop as :class:`DeployACT`: predict a base_link
   twist chunk from the live 3xRGB (+ TCP pose + wrench) observation, integrate
   ``EXEC_STEPS`` steps into absolute pose targets anchored at the *measured* TCP
   pose, command them, re-infer. The seat weld (TouchPlugin) fires on its own when
   the plug seats; the trial then scores an ``insertion_event``.

Env knobs (all optional except the checkpoint):
    CURR_CKPT             specialist checkpoint path (required).
    CURR_STANDOFF_M       staging height above the port mouth (default 0.02).
    CURR_LAT_OFFSET_MM    commanded lateral offset magnitude at staging (default 0).
    CURR_LAT_AZIMUTH_DEG  offset bearing in the lateral plane (default 0).
    CURR_BUDGET_S         learned-phase time budget in sim seconds (default 45).

The pure geometry (offset vector, chunk -> pose-target integration) is factored
into module-level numpy functions so it unit-tests without ROS, torch, or Gazebo
(``tests/test_curriculum_insert.py``).
"""
from __future__ import annotations

import os

import numpy as np

# Torch and the model mirror import lazily inside _load_specialist so the pure
# helpers stay importable in ROS/torch-free unit tests.

# Staging pace: interpolation steps for the high approach, and the gradual free-
# space descent step/period from the approach standoff down to the handoff height
# (mirrors the DisassembleCode staging pattern; all in free space, no contact).
_STAGE_STEPS: int = 100
_STAGE_STANDOFF_M: float = 0.2
_DESCENT_STEP_M: float = 0.002
_DESCENT_DT_S: float = 0.05


def lateral_offset_vector(offset_mm: float, azimuth_deg: float) -> np.ndarray:
    """Return the world-frame lateral (x, y) offset for the curriculum stage.

    Args:
        offset_mm: Offset magnitude in millimeters (>= 0).
        azimuth_deg: Bearing in the lateral plane, degrees; 0 = +x, 90 = +y.

    Returns:
        A ``(2,)`` float64 array ``[dx, dy]`` in meters.

    Raises:
        ValueError: If ``offset_mm`` is negative.
    """
    if offset_mm < 0.0:
        raise ValueError(f"offset_mm must be >= 0, got {offset_mm}")
    r = offset_mm * 1e-3
    a = np.deg2rad(azimuth_deg)
    return np.array([r * np.cos(a), r * np.sin(a)], dtype=np.float64)


def integrate_chunk_targets(
    position: np.ndarray,
    quaternion: np.ndarray,
    chunk: np.ndarray,
    exec_steps: int,
    dt: float,
    substeps: int,
) -> np.ndarray:
    """Integrate a twist chunk into an interpolated absolute pose-target stream.

    Mirrors ``DeployACT``'s receding-horizon execution: the first ``exec_steps``
    chunk twists integrate cumulatively from the measured TCP ``position`` (linear
    part only; the staged orientation is held), and each consecutive pair of
    integrated positions is linearly interpolated with ``substeps`` sub-poses for
    a smooth command stream.

    Args:
        position: Measured TCP position ``(3,)`` in base_link (m).
        quaternion: Orientation ``(4,)`` ``[x, y, z, w]`` to hold on every target.
        chunk: Predicted twist chunk ``(K, 6)`` (m/s, rad/s), base_link frame.
        exec_steps: Chunk steps to execute before re-inference (1..K).
        dt: Seconds between consecutive chunk steps.
        substeps: Interpolated sub-poses per chunk step (>= 1).

    Returns:
        An ``(exec_steps * substeps, 7)`` float64 array of ``[x, y, z, qx, qy,
        qz, qw]`` absolute pose targets.

    Raises:
        ValueError: If shapes/ranges are invalid.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    ch = np.asarray(chunk, dtype=np.float64)
    if ch.ndim != 2 or ch.shape[1] != 6:
        raise ValueError(f"chunk must be (K, 6), got {ch.shape}")
    if not 1 <= exec_steps <= ch.shape[0]:
        raise ValueError(f"exec_steps must be in 1..{ch.shape[0]}, got {exec_steps}")
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    knots = [pos]
    p = pos.copy()
    for k in range(exec_steps):
        p = p + ch[k, :3] * dt
        knots.append(p.copy())

    out = np.empty((exec_steps * substeps, 7), dtype=np.float64)
    row = 0
    for k in range(exec_steps):
        a, b = knots[k], knots[k + 1]
        for s in range(1, substeps + 1):
            out[row, :3] = a + (b - a) * (s / substeps)
            out[row, 3:] = quat
            row += 1
    return out


class _SpecialistModel:
    """Loaded ``train_v3`` checkpoint + the exact deploy-side preprocessing.

    Mirrors ``DeployACT`` inference (image resize 256x288 -> IMG bilinear, the
    ``(x - 0.5) / 0.5`` normalization, state z-score, action de-normalization) so
    the harness evaluates the checkpoint byte-identically to a deployment.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda") -> None:
        """Load the checkpoint.

        Args:
            ckpt_path: Path to a ``train_v3`` checkpoint.
            device: Torch device string.

        Raises:
            FileNotFoundError: If ``ckpt_path`` does not exist.
        """
        import torch

        from aic_example_policies.ros.DeployACT import IMG, _Policy

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CURR_CKPT not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self._torch = torch
        self.device = device
        self.img = int(ck.get("img", IMG))
        self.K = int(ck["K"])
        self.state_dim = int(ck.get("state_dim", 7))
        self.model = _Policy(self.K, state_dim=self.state_dim).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.smean = ck["smean"].to(device)
        self.sstd = ck["sstd"].to(device)
        self.amean = ck["amean"].to(device)
        self.astd = ck["astd"].to(device)

    def _img(self, raw) -> "object":
        """Preprocess one camera Image msg to a normalized ``(1,3,H,W)`` tensor."""
        import cv2
        import torch.nn.functional as F

        a = np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)
        a = cv2.resize(a, (288, 256), interpolation=cv2.INTER_AREA)
        t = (
            self._torch.from_numpy(a)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        t = F.interpolate(t, size=(self.img, self.img), mode="bilinear",
                          align_corners=False)
        return (t - 0.5) / 0.5

    def predict_chunk(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the de-normalized twist chunk for a live observation.

        Args:
            obs: The framework ``Observation`` message.

        Returns:
            ``(chunk, position, quaternion)`` — the ``(K, 6)`` twist chunk plus
            the measured TCP position ``(3,)`` and orientation ``(4,)`` the chunk
            should integrate from.
        """
        torch = self._torch
        imgs = torch.stack(
            [self._img(obs.left_image), self._img(obs.center_image),
             self._img(obs.right_image)], 1)
        p = obs.controller_state.tcp_pose
        pose = np.array([p.position.x, p.position.y, p.position.z,
                         p.orientation.x, p.orientation.y, p.orientation.z,
                         p.orientation.w], dtype=np.float32)
        state = pose
        if self.state_dim == 13:
            w = obs.wrist_wrench.wrench
            wrench = np.array([w.force.x, w.force.y, w.force.z,
                               w.torque.x, w.torque.y, w.torque.z],
                              dtype=np.float32)
            state = np.concatenate([pose, wrench])
        st = torch.from_numpy(state).to(self.device).unsqueeze(0)
        st = (st - self.smean) / self.sstd
        with torch.inference_mode():
            out = self.model(imgs, st)
        act = out[0] if isinstance(out, tuple) else out
        chunk = (act[0] * self.astd + self.amean).cpu().numpy()
        return chunk, pose[:3].astype(np.float64), pose[3:].astype(np.float64)


def _f(name: str, default: float) -> float:
    """Read a float env knob with a default."""
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


# The ROS policy class imports lazily so the pure helpers above stay testable.
try:  # pragma: no cover - exercised only inside the aic_model runtime.
    from aic_example_policies.ros.CheatCode import CheatCode
    from aic_example_policies.ros.DeployACT import (
        DT_FRAME,
        EXEC_STEPS,
        POSE_DAMPING,
        POSE_STIFFNESS,
        SUBSTEPS,
    )
    from aic_model.policy import (
        GetObservationCallback,
        MoveRobotCallback,
        SendFeedbackCallback,
    )
    from aic_task_interfaces.msg import Task
    from geometry_msgs.msg import Point, Pose, Quaternion
    from rclpy.time import Time
    from tf2_ros import TransformException

    from aic_example_policies.ros.cheatcode_targeting import resolve_port_approach

    class CurriculumInsert(CheatCode):
        """Privileged staging above the aligned port, then a learned insertion."""

        def insert_cable(
            self,
            task: Task,
            get_observation: GetObservationCallback,
            move_robot: MoveRobotCallback,
            send_feedback: SendFeedbackCallback,
        ) -> bool:
            """Stage above the port (privileged), then run the specialist.

            Args:
                task: The engine task (module/port/plug names).
                get_observation: Callback returning the latest ``Observation``.
                move_robot: Callback forwarding motion commands.
                send_feedback: Progress feedback callback (unused).

            Returns:
                True when the learned-phase budget elapses (the seat weld and
                scoring fire on their own if the plug seats).
            """
            ckpt = os.environ.get("CURR_CKPT", "").strip()
            standoff = _f("CURR_STANDOFF_M", 0.02)
            lat_mm = _f("CURR_LAT_OFFSET_MM", 0.0)
            lat_az = _f("CURR_LAT_AZIMUTH_DEG", 0.0)
            budget_s = _f("CURR_BUDGET_S", 45.0)
            self.get_logger().info(
                f"CurriculumInsert: ckpt={ckpt} standoff={standoff} "
                f"lat={lat_mm}mm@{lat_az}deg budget={budget_s}s"
            )
            self._task = task

            approach = resolve_port_approach(
                task.target_module_name, task.port_name, task.port_type
            )
            port_frame = approach.frame
            cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
            for frame in (port_frame, cable_tip_frame):
                if not self._wait_for_tf("base_link", frame):
                    return False
            try:
                port_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", port_frame, Time()
                ).transform
            except TransformException as ex:
                self.get_logger().error(f"port TF lookup failed: {ex}")
                return False

            dxy = lateral_offset_vector(lat_mm, lat_az)
            hover = np.array(
                [
                    port_tf.translation.x + dxy[0],
                    port_tf.translation.y + dxy[1],
                    port_tf.translation.z + standoff,
                ]
            )

            # --- Phase 1: privileged staging. High interpolated approach, then a
            # gradual free-space descent to the handoff hover point. ---
            for t in range(_STAGE_STEPS):
                frac = t / float(_STAGE_STEPS)
                try:
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_tf,
                            slerp_fraction=frac,
                            position_fraction=frac,
                            z_offset=_STAGE_STANDOFF_M,
                            reset_xy_integrator=True,
                        ),
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"staging TF failure: {ex}")
                self.sleep_for(_DESCENT_DT_S)

            z = _STAGE_STANDOFF_M
            while z > standoff:
                z = max(standoff, z - _DESCENT_STEP_M)
                tgt = np.array([hover[0], hover[1], port_tf.translation.z + z])
                try:
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self.calc_gripper_pose(
                            port_tf, target_position_base=tgt
                        ),
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"descent TF failure: {ex}")
                self.sleep_for(_DESCENT_DT_S)
            self.get_logger().info(
                f"CurriculumInsert: staged at {standoff * 1000:.0f}mm above port "
                f"(+{lat_mm:.1f}mm lateral) — handing off to "
                f"{os.environ.get('CURR_MODE', 'specialist')}"
            )

            # --- Phase 2 (oracle mode): scripted CheatCode-style descent ladder to
            # the TRUE port from the OFFSET hover. The first ladder target snaps
            # the xy back to the port center, so the recorded episode contains the
            # genuine lateral-correction motion the pure-vertical oracle corpus
            # lacks (M3 offset-staged demo collection). Ends with the natural seat
            # weld + a stabilization dwell, exactly like the training corpus. ---
            if os.environ.get("CURR_MODE", "specialist").strip() == "oracle":
                z = standoff
                floor_z = approach.descent_floor_z
                while z > floor_z:
                    z -= 0.0005
                    try:
                        self.set_pose_target(
                            move_robot=move_robot,
                            pose=self.calc_gripper_pose(port_tf, z_offset=z),
                        )
                    except TransformException as ex:
                        self.get_logger().warn(f"oracle descent TF failure: {ex}")
                    self.sleep_for(0.05)
                self.get_logger().info("CurriculumInsert(oracle): dwell at seat")
                self.sleep_for(5.0)
                return True

            # --- Phase 2: learned insertion (receding horizon). The specialist's
            # twists integrate into a VIRTUAL plug-tip target (initialized at the
            # hover point) commanded through the same ``calc_gripper_pose``
            # machinery staging used. Anchoring at the virtual target rather than
            # the raw ``controller_state.tcp_pose`` avoids the observed frame
            # mismatch where obs-anchored MotionUpdate targets landed on the
            # arm's current pose and produced zero motion (M1 eval, 2026-07-21:
            # tcp_z frozen 40 cycles despite v0z=-11 mm/s predictions). ---
            spec = _SpecialistModel(ckpt)
            exec_steps = min(EXEC_STEPS, spec.K)
            dt_fine = DT_FRAME / SUBSTEPS
            virt = hover.copy()
            floor_z = port_tf.translation.z - 0.020  # hard floor: 20mm below mouth
            start = self.time_now()
            cycles = 0
            while (self.time_now() - start).nanoseconds * 1e-9 < budget_s:
                obs = get_observation()
                if obs is None:
                    self.sleep_for(0.1)
                    continue
                chunk, pos, _quat = spec.predict_chunk(obs)
                for k in range(exec_steps):
                    step = chunk[k, :3] * DT_FRAME
                    for s in range(1, SUBSTEPS + 1):
                        point = virt + step * (s / SUBSTEPS)
                        point[2] = max(point[2], floor_z)
                        try:
                            self.set_pose_target(
                                move_robot=move_robot,
                                pose=self.calc_gripper_pose(
                                    port_tf, target_position_base=point
                                ),
                                stiffness=POSE_STIFFNESS, damping=POSE_DAMPING,
                            )
                        except TransformException as ex:
                            self.get_logger().warn(f"phase-2 TF failure: {ex}")
                        self.sleep_for(dt_fine)
                    virt = virt + step
                    virt[2] = max(virt[2], floor_z)
                cycles += 1
                if cycles % 8 == 0:
                    self.get_logger().info(
                        f"CurriculumInsert: cycle {cycles} tcp_z={pos[2]:.3f} "
                        f"virt_z={virt[2]:.3f} v0z={chunk[0, 2] * 1000:+.1f}mm/s"
                    )
            self.get_logger().info(
                f"CurriculumInsert: budget elapsed after {cycles} cycles "
                f"(virt_z={virt[2]:.3f})."
            )
            return True

except ImportError:  # pragma: no cover - pure-helper import path (unit tests).
    pass
