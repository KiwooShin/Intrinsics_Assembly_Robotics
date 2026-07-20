#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""``DisassembleCode`` -- collect insertion demos *by disassembly* (InsertionNet-backward).

A real seat in this sim is TERMINAL and irreversible: the port's ``TouchPlugin``
(``<time>1</time>``) fires only after **1 s of continuous** plug-tip contact, and
when it fires ``CablePlugin`` welds the plug static, locks the arm, and releases the
gripper -- so you cannot pull out of a real seat (see ``aic_gazebo`` CablePlugin.cc
/ ``TouchPlugin.cc``: the contact timer resets to zero the instant contact breaks).

``DisassembleCode`` exploits that 1 s latch. It reuses the ``CheatCode`` oracle's
privileged (ground-truth) alignment + descent to drive the gripper-welded plug down
to *near* seat depth, but it **never dwells**: it stops the descent ~2 mm short of
the true floor (``retract_start_z``) and begins a slow, laterally-perturbed
**retract** on the very first cycle at depth. Because the tip is moving *outward*
from the first retract cycle, continuous tip contact never reaches 1 s, the touch
latch never fires, and the seat stays reversible.

The recorded perturbed retract is time-reversed offline (:mod:`reverse_disasm`)
into an insertion demo whose action label *contains lateral correction* -- which
the pure-vertical oracle last-inch lacks (lat/vert 0.014). The lateral/axial/roll/
pitch schedule is drawn per episode by the pure, unit-tested
:mod:`aic_example_policies.ros.disassembly_schedule` module.

TEACHER-SIDE ONLY. Like ``CheatCode``, this policy is an oracle: it reads the
privileged port TF published under ``ground_truth:=true`` and is used ONLY to
collect demos. It is NEVER the deployed eval policy; the deployed policy
(``DeployACT``) trains on the reversed demos and needs no port TF. Select it with::

    ros2 run aic_model aic_model -p policy:=aic_example_policies.ros.DisassembleCode
"""
from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

import numpy as np

from aic_example_policies.ros import disassembly_schedule as sched
from aic_example_policies.ros.cheatcode_targeting import (
    SC_APPROACH_STANDOFF_M,
    SC_PORT_TYPE,
    resolve_port_approach,
    sc_entrance_waypoint,
)
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Quaternion, Transform, Vector3
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply


def _lateral_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane orthogonal to ``axis``.

    A stable right-handed basis ``(u, v)`` (both unit, mutually orthogonal, and
    orthogonal to the unit ``axis``) -- the lateral plane the perturbation spiral
    traces in. Mirrors ``guarded_descent._lateral_basis`` (kept local so this file
    has no dependency on the heavy guarded-descent module).

    Args:
        axis: Unit retract (out-of-port) axis ``(3,)``.

    Returns:
        The ``(u, v)`` pair of ``(3,)`` unit vectors orthogonal to ``axis``.
    """
    ref = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(axis[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = ref - axis * float(np.dot(ref, axis))
    u = u / float(np.linalg.norm(u))
    v = np.cross(axis, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def _small_rp_quat_wxyz(droll: float, dpitch: float) -> tuple[float, float, float, float]:
    """Return a ``(w, x, y, z)`` quaternion for a small roll-then-pitch rotation.

    Args:
        droll: Roll angle about local x (rad).
        dpitch: Pitch angle about local y (rad).

    Returns:
        The composed quaternion ``(w, x, y, z)`` (``q_pitch * q_roll``) in the
        transforms3d convention ``CheatCode.calc_gripper_pose`` consumes.
    """
    hr, hp = 0.5 * droll, 0.5 * dpitch
    q_roll = (np.cos(hr), np.sin(hr), 0.0, 0.0)   # about x
    q_pitch = (np.cos(hp), 0.0, np.sin(hp), 0.0)  # about y
    q = quaternion_multiply(q_pitch, q_roll)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


@dataclasses.dataclass(frozen=True)
class DisassembleConfig:
    """Env-tunable knobs for the ``DisassembleCode`` retract.

    Attributes:
        retract_start_z: The ``z_offset`` (m, added to the port-frame z along the
            port axis) at which the descent stops and the retract begins. Kept ~2 mm
            SHALLOWER than the ``CheatCode`` floor so the tip never accrues 1 s of
            continuous contact (the touch-latch threshold). Negative = into the
            port.
        axial_span_m: Total outward axial retract travel (m).
        axial_step_m: Axial travel per retract waypoint (m); sets the waypoint
            count ``round(axial_span_m / axial_step_m)``.
        dt_s: Sleep between retract waypoints (s); small step * this dt keeps the
            retract slow (friction shear << normal force, InsertionNet snapshot).
        seed: Base RNG seed for the per-episode schedule draw.
        defaults: The schedule distribution (spiral band, roll/pitch band, lift
            fraction/magnitudes) passed to :func:`disassembly_schedule.sample_config`.
    """

    retract_start_z: float
    axial_span_m: float
    axial_step_m: float
    dt_s: float
    seed: int
    defaults: sched.ScheduleDefaults

    @property
    def n_steps(self) -> int:
        """Number of retract waypoints implied by span / step (>= 2)."""
        return max(2, int(round(self.axial_span_m / self.axial_step_m)))

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DisassembleConfig":
        """Build a config from ``DISASM_*`` environment variables.

        Recognised (all optional): ``DISASM_RETRACT_START_Z``,
        ``DISASM_AXIAL_SPAN_M``, ``DISASM_AXIAL_STEP_M``, ``DISASM_DT``,
        ``DISASM_SEED``, ``DISASM_TURNS``, ``DISASM_RADIUS_MIN_M``,
        ``DISASM_RADIUS_MAX_M``, ``DISASM_ROLL_PITCH_MIN_RAD``,
        ``DISASM_ROLL_PITCH_MAX_RAD``, ``DISASM_LIFT_FRAC``,
        ``DISASM_LIFT_AXIAL_MIN_M``, ``DISASM_LIFT_AXIAL_MAX_M``,
        ``DISASM_LIFT_LATERAL_MIN_M``, ``DISASM_LIFT_LATERAL_MAX_M``.

        Args:
            env: Environment mapping (e.g. ``os.environ``).

        Returns:
            The parsed :class:`DisassembleConfig`.

        Raises:
            ValueError: If a numeric variable is unparseable or a band is invalid
                (delegated to :class:`disassembly_schedule.ScheduleDefaults`).
        """

        def f(key: str, default: float) -> float:
            raw = env.get(key, "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as ex:
                raise ValueError(f"{key} must be a float, got {raw!r}") from ex

        defaults = sched.ScheduleDefaults(
            radius_min_m=f("DISASM_RADIUS_MIN_M", sched.RADIUS_MIN_M),
            radius_max_m=f("DISASM_RADIUS_MAX_M", sched.RADIUS_MAX_M),
            turns=f("DISASM_TURNS", sched.DEFAULT_TURNS),
            axial_span_m=f("DISASM_AXIAL_SPAN_M", sched.DEFAULT_AXIAL_SPAN_M),
            roll_pitch_min_rad=f("DISASM_ROLL_PITCH_MIN_RAD", sched.ROLL_PITCH_MIN_RAD),
            roll_pitch_max_rad=f("DISASM_ROLL_PITCH_MAX_RAD", sched.ROLL_PITCH_MAX_RAD),
            lift_frac=f("DISASM_LIFT_FRAC", sched.DEFAULT_LIFT_FRAC),
            lift_axial_min_m=f("DISASM_LIFT_AXIAL_MIN_M", sched.LIFT_AXIAL_MIN_M),
            lift_axial_max_m=f("DISASM_LIFT_AXIAL_MAX_M", sched.LIFT_AXIAL_MAX_M),
            lift_lateral_min_m=f("DISASM_LIFT_LATERAL_MIN_M", sched.LIFT_LATERAL_MIN_M),
            lift_lateral_max_m=f("DISASM_LIFT_LATERAL_MAX_M", sched.LIFT_LATERAL_MAX_M),
        )
        seed_raw = env.get("DISASM_SEED", "0").strip() or "0"
        try:
            seed = int(seed_raw)
        except ValueError as ex:
            raise ValueError(f"DISASM_SEED must be an int, got {seed_raw!r}") from ex
        return cls(
            retract_start_z=f("DISASM_RETRACT_START_Z", -0.013),
            axial_span_m=defaults.axial_span_m,
            axial_step_m=f("DISASM_AXIAL_STEP_M", 0.0005),
            dt_s=f("DISASM_DT", 0.06),
            seed=seed,
            defaults=defaults,
        )


class DisassembleCode(CheatCode):
    """Oracle that seats *without latching*, then records a perturbed retract.

    Subclasses :class:`CheatCode` to reuse its privileged targeting
    (``calc_gripper_pose``, ``_wait_for_tf``) and the shared ``set_pose_target``
    (MODE_POSITION) motion path, overriding only :meth:`insert_cable` to (a) descend
    to ``retract_start_z`` without dwelling and (b) execute the perturbed retract.
    """

    def __init__(self, parent_node) -> None:
        """Load the ``DISASM_*`` config and initialise the CheatCode base."""
        super().__init__(parent_node)
        self._cfg = DisassembleConfig.from_env(os.environ)
        self.get_logger().info(
            f"DisassembleCode config: retract_start_z={self._cfg.retract_start_z} "
            f"axial_span={self._cfg.axial_span_m} step={self._cfg.axial_step_m} "
            f"dt={self._cfg.dt_s} n_steps={self._cfg.n_steps} seed={self._cfg.seed}"
        )

    def _perturbed_port(
        self, port_transform: Transform, droll: float, dpitch: float
    ) -> Transform:
        """Return a copy of ``port_transform`` with a small roll/pitch applied.

        Args:
            port_transform: The privileged port (or entrance) transform in
                ``base_link``.
            droll: Roll perturbation about the port local x (rad).
            dpitch: Pitch perturbation about the port local y (rad).

        Returns:
            A new ``Transform`` with the same translation and the perturbed
            rotation ``q_port * q_roll_pitch`` (unchanged when both angles are 0).
        """
        if droll == 0.0 and dpitch == 0.0:
            return port_transform
        r = port_transform.rotation
        q_port = (r.w, r.x, r.y, r.z)  # transforms3d (w, x, y, z) convention
        qw, qx, qy, qz = quaternion_multiply(
            q_port, _small_rp_quat_wxyz(droll, dpitch)
        )
        out = Transform()
        out.translation = Vector3(
            x=port_transform.translation.x,
            y=port_transform.translation.y,
            z=port_transform.translation.z,
        )
        out.rotation = Quaternion(w=float(qw), x=float(qx), y=float(qy), z=float(qz))
        return out

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Descend to (near) seat depth without dwelling, then perturbed-retract.

        Args:
            task: The insertion task (privileged port identity used via
                ``ground_truth``).
            get_observation: Unused (scripted teacher policy).
            move_robot: The Cartesian motion callback used by ``set_pose_target``.
            send_feedback: Unused (scripted teacher policy).

        Returns:
            ``True`` when the perturbed retract completed; ``False`` if a required
            TF never became available.
        """
        self.get_logger().info(f"DisassembleCode.insert_cable() task: {task}")
        self._task = task
        cfg = self._cfg

        approach = resolve_port_approach(
            task.target_module_name, task.port_name, task.port_type
        )
        port_frame = approach.frame
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        self.get_logger().info(
            f"DisassembleCode target: port_type={task.port_type!r} "
            f"port_frame={port_frame} retract_start_z={cfg.retract_start_z}"
        )
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False
        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        # Resolve the port axis + anchor. SFP: world-vertical, the legacy world-z
        # path (target = port xy, port.z + z_offset), so out-of-port is +z and the
        # anchor is the port frame position. SC: the pose-conditioned insertion axis
        # (points INTO the port), so out-of-port is -insertion_axis.
        is_sc = task.port_type.strip().lower() == SC_PORT_TYPE
        if is_sc:
            entrance_pos = np.array(
                [
                    port_transform.translation.x,
                    port_transform.translation.y,
                    port_transform.translation.z,
                ]
            )
            entrance_quat = np.array(
                [
                    port_transform.rotation.x,
                    port_transform.rotation.y,
                    port_transform.rotation.z,
                    port_transform.rotation.w,
                ]
            )
            insertion_axis = sc_entrance_waypoint(entrance_pos, entrance_quat).insertion_axis
            axis_out = -insertion_axis
            # Plug-tip anchor at retract-start depth: retract_start_z into the port
            # along +insertion_axis (matches CheatCode's sc_target(z)).
            anchor = entrance_pos - cfg.retract_start_z * insertion_axis
            stage_standoff = SC_APPROACH_STANDOFF_M
        else:
            axis_out = np.array([0.0, 0.0, 1.0])
            anchor = np.array(
                [
                    port_transform.translation.x,
                    port_transform.translation.y,
                    port_transform.translation.z + cfg.retract_start_z,
                ]
            )
            stage_standoff = 0.2
        u_lat, v_lat = _lateral_basis(axis_out)

        def axis_target(axial_out: float, lat_x: float, lat_y: float) -> np.ndarray:
            """Plug-tip target = anchor + axial_out*axis_out + lateral offsets."""
            return anchor + axial_out * axis_out + lat_x * u_lat + lat_y * v_lat

        def sc_stage_target(z: float) -> np.ndarray | None:
            """Staging plug-tip target ``z`` m outside the entrance (SC), else None."""
            if not is_sc:
                return None
            return entrance_pos - z * insertion_axis

        # --- Phase 1: stage above the port, then descend GRADUALLY to retract_start_z
        # (tracking z_offset -- no constant-target dwell). The 1 s touch-latch is
        # disabled for collection (collect_disassembly.sh disables each port's
        # TouchPlugin via its gz enable service), so a full slow perturbed retract is
        # reversible; the reversed seat marker is geometric (reverse_disasm N-1). ---
        for t in range(0, 100):
            interp_fraction = t / 100.0
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=stage_standoff,
                        reset_xy_integrator=True,
                        target_position_base=sc_stage_target(stage_standoff),
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during staging: {ex}")
            self.sleep_for(0.05)

        z_offset = stage_standoff
        while z_offset > cfg.retract_start_z:
            z_offset = max(cfg.retract_start_z, z_offset - 0.0005)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        z_offset=z_offset,
                        target_position_base=(
                            sc_stage_target(z_offset)
                            if is_sc
                            # SFP: track z_offset so the tip descends GRADUALLY to
                            # retract_start_z instead of being commanded straight to the
                            # final depth and held there ~21 s (the old axis_target(0,0,0)
                            # constant target caused the mid-descent weld). anchor sits at
                            # retract_start_z and axis_out is +z, so axial_out = z_offset -
                            # retract_start_z places the tip at height z_offset.
                            else axis_target(z_offset - cfg.retract_start_z, 0.0, 0.0)
                        ),
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during descent: {ex}")
            self.sleep_for(0.05)

        # --- Phase 2: perturbed retract. Draw the per-episode schedule and command
        # each waypoint OUTWARD along the port axis. The TouchPlugin latch is disabled
        # for collection (see collect_disassembly.sh::run_disasm_trial), so the tip may
        # dwell/recede through the seat plate without welding and the seat stays
        # reversible. ---
        rng = np.random.default_rng(cfg.seed)
        cfg_ep = sched.sample_config(rng, cfg.n_steps, cfg.defaults)
        offsets = sched.build_schedule(cfg_ep)
        self.get_logger().info(
            f"DisassembleCode retract: variant="
            f"{'lift_translate_reseat' if cfg_ep.lift_translate else 'spiral'} "
            f"radius_max={cfg_ep.radius_max_m*1e3:.3f}mm turns={cfg_ep.turns} "
            f"azimuth0={cfg_ep.azimuth0_rad:.3f} n={len(offsets)} (NO DWELL)"
        )
        for axial_out, lat_x, lat_y, droll, dpitch in offsets:
            target = axis_target(float(axial_out), float(lat_x), float(lat_y))
            perturbed = self._perturbed_port(
                port_transform, float(droll), float(dpitch)
            )
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        perturbed,
                        reset_xy_integrator=True,
                        target_position_base=target,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during retract: {ex}")
            self.sleep_for(cfg.dt_s)

        self.get_logger().info("DisassembleCode.insert_cable() exiting (no seat).")
        return True
