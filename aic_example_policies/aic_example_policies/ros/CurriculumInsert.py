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

# cheatcode_targeting is pure (numpy only), so these SC-geometry helpers import at
# module top -- the pose-conditioned SC staging axis is needed by the pure helper
# ``sc_axis_frame`` below (unit-tested without ROS). Mirrors DisassembleCode.
from aic_example_policies.ros.cheatcode_targeting import (
    SC_APPROACH_STANDOFF_M,
    SC_PORT_TYPE,
    sc_entrance_waypoint,
)

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


def _lateral_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane orthogonal to ``axis``.

    A right-handed ``(u, v)`` basis (both unit, mutually orthogonal, and orthogonal
    to the unit ``axis``) -- the port-face plane a commanded curriculum lateral
    offset is mapped into for a rotated SC port. Mirrors
    ``DisassembleCode._lateral_basis`` / ``guarded_descent._lateral_basis`` (kept
    local so the pure helpers stay ROS-free and unit-testable).

    Args:
        axis: Unit axis ``(3,)`` (the SC insertion axis).

    Returns:
        The ``(u, v)`` pair of ``(3,)`` unit vectors orthogonal to ``axis``.
    """
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    ref = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(a[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = ref - a * float(np.dot(ref, a))
    u = u / float(np.linalg.norm(u))
    v = np.cross(a, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def sc_axis_frame(
    entrance_pos: np.ndarray, entrance_quat: np.ndarray, standoff_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the pose-conditioned SC staging frame for the curriculum insert.

    Mirrors the proven ``CheatCode`` / ``DisassembleCode`` SC geometry: the SC
    entrance frame's local +z points *into* the port, so
    :func:`sc_entrance_waypoint` rotates it into the base frame to get the unit
    insertion axis, and the pre-insertion hover sits ``standoff_m`` back out of the
    mouth along that axis (``entrance_pos - standoff_m * insertion_axis``). A
    right-handed lateral basis ``(u, v)`` spans the plane orthogonal to the axis so
    a commanded lateral offset can be mapped off the world xy-plane onto the true
    (rotated) port face. Pure (pose in -> frame out): unit-tests without ROS.

    When the resolved axis is world-vertical (canonical mount / identity-ish pose)
    the hover displacement is a pure world-z step and the lateral basis is world
    ``(x, y)``, so the geometry reduces to the legacy world-z staging line.

    Args:
        entrance_pos: SC entrance-frame origin ``[x, y, z]`` in base_link (m).
        entrance_quat: SC entrance-frame orientation ``[x, y, z, w]`` in base_link
            (renormalized inside :func:`sc_entrance_waypoint`).
        standoff_m: Pre-insertion standoff distance (m) along the port axis; must be
            finite and >= 0.

    Returns:
        A tuple ``(hover_point, insertion_axis, u_lat, v_lat)`` of base-frame
        ``(3,)`` float64 arrays: the on-axis pre-insertion hover point, the unit
        insertion axis (pointing into the port), and the two unit lateral-basis
        vectors (mutually orthogonal, both orthogonal to the axis).

    Raises:
        ValueError: Propagated from :func:`sc_entrance_waypoint` for a bad pose or
            standoff (non-length-3 / non-finite position, near-zero-norm quaternion,
            or negative / non-finite standoff).
    """
    wp = sc_entrance_waypoint(entrance_pos, entrance_quat, standoff_m)
    insertion_axis = np.asarray(wp.insertion_axis, dtype=np.float64).reshape(3)
    u_lat, v_lat = _lateral_basis(insertion_axis)
    hover_point = np.asarray(wp.approach_point, dtype=np.float64).reshape(3)
    return hover_point, insertion_axis, u_lat, v_lat


def wrench_force_mag(force_xyz: object) -> float:
    """Return the Euclidean magnitude of a 3-D force vector, in newtons.

    Used by the contact-gated early-re-inference option (``CURR_REACT_WRENCH_N``):
    a spike above the threshold means the plug has touched the rim, so the learned
    loop should re-infer with that contact observation instead of finishing a chunk
    that was planned in free space.

    Args:
        force_xyz: The ``(fx, fy, fz)`` force components in newtons (any 3-element
            array-like).

    Returns:
        ``sqrt(fx**2 + fy**2 + fz**2)`` as a float.

    Raises:
        ValueError: If ``force_xyz`` does not have exactly 3 elements.
    """
    f = np.asarray(force_xyz, dtype=np.float64).ravel()
    if f.shape[0] != 3:
        raise ValueError(f"force_xyz must have 3 elements, got shape {f.shape}")
    return float(np.linalg.norm(f))


def contact_spike(force_now: object, force_ref: object, threshold: float) -> bool:
    """True when the force magnitude has DEVIATED from a reference by > threshold.

    On this rig the wrist F/T is bias-dominated: |F| reads ~20 N in free-space descent
    (plug weight / sensor bias) and DROPS to ~7-8 N when the plug tip contacts the port
    and its weight is supported. So contact is a *change* in |F| (here a drop of ~13 N),
    not an absolute high reading — an absolute threshold would fire backwards. This
    detects the change in either direction relative to the per-chunk baseline captured
    at re-inference time.

    Args:
        force_now: Current ``(fx, fy, fz)`` force (N).
        force_ref: The chunk-start reference ``(fx, fy, fz)`` force (N).
        threshold: Deviation magnitude in newtons that counts as a contact event.

    Returns:
        ``abs(|force_now| - |force_ref|) > threshold``.
    """
    return abs(wrench_force_mag(force_now) - wrench_force_mag(force_ref)) > threshold


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
            # Resolve the insertion frame. SFP keeps the legacy world-z path: the
            # hover is the port xy (+ lateral offset) at ``standoff`` above the mouth
            # and the descent is straight down. SC is pose-conditioned: its entrance
            # link is physically rotated, so staging/descent follow the true insertion
            # axis (into the port) resolved from the privileged entrance pose --
            # mirrors the proven CheatCode/DisassembleCode SC branch. ``stage_standoff``
            # is the high-approach height (SC pinned to the proven
            # SC_APPROACH_STANDOFF_M; both 0.2 m so the SFP path stays byte-identical).
            is_sc = task.port_type.strip().lower() == SC_PORT_TYPE
            stage_standoff = SC_APPROACH_STANDOFF_M if is_sc else _STAGE_STANDOFF_M
            if is_sc:
                entrance_pos = np.array(
                    [
                        port_tf.translation.x,
                        port_tf.translation.y,
                        port_tf.translation.z,
                    ]
                )
                entrance_quat = np.array(
                    [
                        port_tf.rotation.x,
                        port_tf.rotation.y,
                        port_tf.rotation.z,
                        port_tf.rotation.w,
                    ]
                )
                sc_hover, insertion_axis, u_lat, v_lat = sc_axis_frame(
                    entrance_pos, entrance_quat, standoff
                )
                # Map the world-parameterized curriculum offset onto the port face
                # (the plane orthogonal to the insertion axis). Reduces to the world
                # xy offset when the axis is world-vertical (u_lat=+x, v_lat=+y).
                lat_vec = dxy[0] * u_lat + dxy[1] * v_lat
                hover = sc_hover + lat_vec

                def _sc_axis_point(z_off: float, lat_frac: float = 1.0) -> np.ndarray:
                    """Plug-tip target ``z_off`` m outside the entrance along the axis.

                    ``entrance_pos - z_off * insertion_axis`` places the tip ``z_off``
                    m outside the mouth and steps it into the port as ``z_off``
                    decreases through 0 (mirrors CheatCode ``sc_target``); ``lat_frac``
                    scales the curriculum lateral offset (1 = staged offset, 0 =
                    on-axis / centered).
                    """
                    return entrance_pos - z_off * insertion_axis + lat_frac * lat_vec

                self.get_logger().info(
                    f"CurriculumInsert(SC): insertion_axis={insertion_axis} "
                    f"hover={hover} lat_vec={lat_vec}"
                )
            else:
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
                            z_offset=stage_standoff,
                            reset_xy_integrator=True,
                            target_position_base=(
                                _sc_axis_point(stage_standoff, 0.0) if is_sc else None
                            ),
                        ),
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"staging TF failure: {ex}")
                self.sleep_for(_DESCENT_DT_S)

            z = stage_standoff
            while z > standoff:
                # SC needs a slow, continuous descent (CheatCode's ~10mm/s) so the
                # impedance controller converges precisely onto the tilted insertion
                # axis before the seat; the fast SFP step (40mm/s) leaves it ~3.5mm
                # off and it rams the rotated port. SFP step unchanged.
                z = max(standoff, z - (0.0005 if is_sc else _DESCENT_STEP_M))
                if is_sc:
                    tgt = _sc_axis_point(z, 1.0)
                else:
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
                # CURR_CORRECT_AT_MM > 0 = LATE-CORRECTION demo: descend AT THE
                # OFFSET xy down to that height above the mouth, then translate
                # laterally to the true port xy over ~10 steps, then seat. Teaches
                # near-contact lateral recovery — the free-space-correcting default
                # (correct_at=0 behaves like the original: first target snaps to
                # the true xy high above) never shows rim-contact recovery, which
                # capped the learned capture radius at [1,2)mm (m3/m3c evals).
                correct_at = _f("CURR_CORRECT_AT_MM", 0.0) * 1e-3
                z = standoff
                floor_z = approach.descent_floor_z

                def _tip(zh: float, frac: float) -> np.ndarray:
                    """Plug-tip target at standoff-eq height ``zh``, offset by ``frac``.

                    SFP: world-z (port xy + ``frac``-scaled lateral, port.z + ``zh``).
                    SC: ``zh`` m outside the entrance along the insertion axis with the
                    curriculum lateral offset scaled by ``frac`` (``frac`` -> 0 puts the
                    tip on the axis / centered).
                    """
                    if is_sc:
                        return _sc_axis_point(zh, frac)
                    return np.array([
                        port_tf.translation.x + dxy[0] * frac,
                        port_tf.translation.y + dxy[1] * frac,
                        port_tf.translation.z + zh,
                    ])

                if correct_at > 0.0:
                    while z > correct_at:            # offset descent (frac=1)
                        z -= 0.0005
                        try:
                            self.set_pose_target(
                                move_robot=move_robot,
                                pose=self.calc_gripper_pose(
                                    port_tf, target_position_base=_tip(z, 1.0)),
                            )
                        except TransformException as ex:
                            self.get_logger().warn(f"late-corr TF: {ex}")
                        self.sleep_for(0.05)
                    for s in range(10, -1, -1):      # lateral correction at height
                        try:
                            self.set_pose_target(
                                move_robot=move_robot,
                                pose=self.calc_gripper_pose(
                                    port_tf,
                                    target_position_base=_tip(z, s / 10.0)),
                            )
                        except TransformException as ex:
                            self.get_logger().warn(f"late-corr TF: {ex}")
                        self.sleep_for(0.1)
                while z > floor_z:                   # centered final descent
                    z -= 0.0005
                    try:
                        self.set_pose_target(
                            move_robot=move_robot,
                            pose=(
                                self.calc_gripper_pose(
                                    port_tf, target_position_base=_sc_axis_point(z, 0.0)
                                )
                                if is_sc
                                else self.calc_gripper_pose(port_tf, z_offset=z)
                            ),
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
            # SC hard floor: never penetrate deeper than 20mm PAST the entrance ALONG
            # the insertion axis (the world-z floor above is meaningless for a rotated
            # port). Mirrors the SFP 20mm-below-mouth safety floor, measured along the
            # true axis; lateral (port-face) motion stays unconstrained. JUDGMENT CALL
            # -- see report; validate in-sim.
            sc_max_depth = 0.020

            def _sc_floor(p: np.ndarray) -> np.ndarray:
                """Clamp axial penetration to ``sc_max_depth`` along the insertion axis.

                Projects only the excess axial component back out along the axis, so
                the lateral (port-face) components of ``p`` are preserved (the SC
                analogue of the SFP ``point[2] = max(point[2], floor_z)`` clamp).
                """
                depth = float(np.dot(p - entrance_pos, insertion_axis))
                if depth > sc_max_depth:
                    return p - (depth - sc_max_depth) * insertion_axis
                return p
            # Contact-gated early re-inference (CURR_REACT_WRENCH_N > 0 enables it):
            # if |F| DEVIATES from the per-chunk baseline by more than this many newtons
            # mid-chunk (rig contact shows as a ~13 N drop, not a spike — see
            # contact_spike), stop executing the free-space-planned chunk and re-infer
            # with the contact observation. 0 = off => baseline behaviour byte-identical.
            react_n = _f("CURR_REACT_WRENCH_N", 0.0)
            f_base = None  # free-space force baseline, captured once at phase-2 start
            start = self.time_now()
            cycles = 0
            while (self.time_now() - start).nanoseconds * 1e-9 < budget_s:
                obs = get_observation()
                if obs is None:
                    self.sleep_for(0.1)
                    continue
                if f_base is None:  # capture the free-space baseline once, at the hover
                    wfb = obs.wrist_wrench.wrench.force
                    f_base = (wfb.x, wfb.y, wfb.z)
                chunk, pos, _quat = spec.predict_chunk(obs)
                for k in range(exec_steps):
                    step = chunk[k, :3] * DT_FRAME
                    for s in range(1, SUBSTEPS + 1):
                        point = virt + step * (s / SUBSTEPS)
                        if is_sc:
                            point = _sc_floor(point)
                        else:
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
                    if is_sc:
                        virt = _sc_floor(virt)
                    else:
                        virt[2] = max(virt[2], floor_z)
                    if react_n > 0.0:
                        ob = get_observation()
                        if ob is not None:
                            wf = ob.wrist_wrench.wrench.force
                            if contact_spike((wf.x, wf.y, wf.z), f_base, react_n):
                                self.get_logger().info(
                                    "CurriculumInsert: wrench-gate re-infer "
                                    f"(|dF|>{react_n:.1f}N vs chunk baseline) at k={k}")
                                break
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
