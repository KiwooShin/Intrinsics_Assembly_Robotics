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


import numpy as np

from aic_example_policies.ros.cheatcode_targeting import (
    SC_APPROACH_STANDOFF_M,
    SC_PORT_TYPE,
    resolve_port_approach,
    sc_entrance_waypoint,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

QuaternionTuple = tuple[float, float, float, float]


class CheatCode(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        target_position_base: np.ndarray | None = None,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment.

        Args:
            port_transform: Port (or entrance) frame pose in ``base_link``; its
                orientation drives the plug-alignment slerp.
            slerp_fraction: Interpolation fraction for the orientation slerp.
            position_fraction: Interpolation fraction for the position blend.
            z_offset: Legacy world-z standoff above ``port_transform`` used only
                when ``target_position_base`` is None.
            reset_xy_integrator: When True, zero the xy error integrators.
            target_position_base: Optional explicit plug-tip target ``[x, y, z]``
                in ``base_link``. When None (SFP / legacy), the target is the port
                xy and ``port.z + z_offset`` (byte-identical to the historical
                primitive). When provided (SC pose-conditioned path), it is used
                directly, letting the caller stage/descend along the port's true
                insertion axis instead of world-z. Reduces to the legacy target
                exactly when the axis is world-vertical.
        """
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        # Plug-tip target in base_link. Legacy (SFP / target_position_base None):
        # the port xy and port.z + z_offset (world-z hover/descent). Pose-
        # conditioned (SC): an explicit point supplied by the caller along the
        # port's insertion axis.
        if target_position_base is None:
            target_base = (
                port_transform.translation.x,
                port_transform.translation.y,
                port_transform.translation.z + z_offset,
            )
        else:
            target_base = (
                float(target_position_base[0]),
                float(target_position_base[1]),
                float(target_position_base[2]),
            )
        port_xy = (target_base[0], target_base[1])
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = target_base[2] - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCode.insert_cable() task: {task}")
        self._task = task

        # Resolve the port frame + descent floor from the port type. SFP keeps the
        # historical target (`..._link`, floor -0.015); SC retargets to the
        # dedicated entrance frame with a shallower floor -0.007 (see
        # cheatcode_targeting). SC additionally stages + descends along a pose-
        # conditioned insertion axis (below) rather than a fixed world-z line. The
        # log lines print what actually resolved so the post-B revalidation
        # (4-6 SC-pose trials; gate success >=85, contacts 0) can confirm quickly.
        approach = resolve_port_approach(
            task.target_module_name, task.port_name, task.port_type
        )
        port_frame = approach.frame
        descent_floor_z = approach.descent_floor_z
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        self.get_logger().info(
            f"CheatCode target: port_type={task.port_type!r} port_frame={port_frame} "
            f"cable_tip_frame={cable_tip_frame} descent_floor_z={descent_floor_z}"
        )

        # Wait for both the port and cable tip TFs to become available.
        # These come via ground_truth and may not be immediate.
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        # SC uses a pose-conditioned staging waypoint + insertion axis derived from
        # the (privileged, teacher-side) entrance-frame pose, so staging and descent
        # follow the port's true axis instead of a fixed world-z line (fixes the
        # off-axis graze under eval-band yaw; see cheatcode_targeting). SFP keeps the
        # legacy world-z path byte-identical (target_position_base stays None).
        is_sc = task.port_type.strip().lower() == SC_PORT_TYPE
        sc_axis = None
        entrance_pos = None
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
            sc_approach = sc_entrance_waypoint(entrance_pos, entrance_quat)
            sc_axis = sc_approach.insertion_axis
            self.get_logger().info(
                f"SC pose-conditioned waypoint: approach={sc_approach.approach_point} "
                f"insertion_axis={sc_axis} standoff={sc_approach.standoff_m}"
            )

        def sc_target(z: float) -> np.ndarray | None:
            """Plug-tip target ``z`` metres outside the entrance along the port axis.

            Returns None for SFP so ``calc_gripper_pose`` keeps its legacy world-z
            target. For SC, ``entrance_pos - z * insertion_axis`` places the target
            ``z`` m outside the mouth (``z`` = standoff at staging) and steps it into
            the port as ``z`` -> descent floor; reduces to the legacy world-z target
            when the axis is world-vertical.
            """
            if not is_sc:
                return None
            return entrance_pos - z * sc_axis

        # Staging standoff: SC uses the named pose-conditioned constant; SFP the
        # historical 0.2 m world-z hover (identical value, kept explicit).
        z_offset = SC_APPROACH_STANDOFF_M if is_sc else 0.2

        # Over five seconds, smoothly interpolate from the current position to
        # the pre-insertion staging waypoint.
        for t in range(0, 100):
            interp_fraction = t / 100.0
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                        target_position_base=sc_target(z_offset),
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        # Descend until the cable is inserted into the port. The floor is
        # per-port-type (SFP -0.015; SC shallower -0.007 -- just past the entrance).
        while True:
            if z_offset < descent_floor_z:
                break

            z_offset -= 0.0005
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        z_offset=z_offset,
                        target_position_base=sc_target(z_offset),
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)

        self.get_logger().info("CheatCode.insert_cable() exiting...")
        return True
