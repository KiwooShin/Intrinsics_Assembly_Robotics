#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Pure (ROS-free) target-frame resolution for the ``CheatCode`` insertion oracle.

``CheatCode`` drives the plug straight down onto a port TF frame and then descends
by a fixed ``base_link`` world-z amount. That single primitive was tuned for SFP
ports, whose port link points up, so a vertical descent onto ``..._link`` seats the
plug (descent floor ``-0.015`` m past the port-frame z).

The SC (LC/SC fiber) port breaks both assumptions: its ``sc_port_base_link`` is
mounted rotated (``rpy 1.57 0 1.57`` in ``task_board.urdf.xacro``), and the model
publishes a dedicated pre-insertion frame ``sc_port_base_link_entrance`` offset
``-0.01564`` m into the port mouth (``aic_assets/models/SC Port/model.sdf``).
Aiming the generic primitive at the SC ``..._link`` makes the finger ram the port
body instead of inserting (observed: contact ~-24, no insertion, score ~19).

This module maps a task's ``port_type`` to the frame ``CheatCode`` should approach
and how deep it should descend. It is deliberately free of ROS / Gazebo imports so
it can be unit-tested on any machine (see CLAUDE.md test rules). Frame-name
conventions were mirrored from the model SDFs:

* SFP entrance: ``sfp_port_{0,1}_link_entrance`` (``NIC Card`` model, offset -0.0458)
* SC  entrance: ``sc_port_base_link_entrance``   (``SC Port`` model, offset -0.01564)

so an entrance frame is uniformly ``{port_name}_link_entrance``.
"""
from __future__ import annotations

import dataclasses

# ``Task.port_type`` value (aic_task_interfaces/Task.msg) that denotes an SC port.
SC_PORT_TYPE = "sc"

# Descent floor: the final ``z_offset`` (metres, added to the target frame's z in
# base_link) at which ``CheatCode`` stops driving the plug down.
#
# SFP keeps the historically-tuned value so SFP behaviour is byte-identical.
SFP_DESCENT_FLOOR_Z = -0.015
# SC targets the *entrance* frame (already offset into the port mouth), so the
# descent should be shallow -- stop at the entrance rather than -0.015 past the
# rotated port centre.
#
# TODO(YAWFIX): 0.0 stops the plug exactly at the SC entrance frame. If the
# post-campaign 15-min sim validation shows the plug does not fully seat
# (insertion_events == 0 / total < 85), lower this toward -0.005 / -0.010. Kept a
# named constant so tuning is a one-line change and requires no ROS.
SC_DESCENT_FLOOR_Z = 0.0


@dataclasses.dataclass(frozen=True)
class PortApproach:
    """Where ``CheatCode`` should aim for a port and how deep to descend.

    Attributes:
        frame: The ``task_board`` TF frame to approach and align the plug with.
        descent_floor_z: The final ``z_offset`` (m) at which the descent stops.
    """

    frame: str
    descent_floor_z: float


def resolve_port_approach(
    target_module_name: str, port_name: str, port_type: str
) -> PortApproach:
    """Map a task's port identity to the frame to approach and the descent floor.

    SFP (and any non-SC ``port_type``) keeps the historical behaviour: approach
    ``task_board/{target_module_name}/{port_name}_link`` and descend to
    :data:`SFP_DESCENT_FLOOR_Z`. SC ports instead approach the dedicated entrance
    frame ``task_board/{target_module_name}/{port_name}_link_entrance`` and use the
    shallower :data:`SC_DESCENT_FLOOR_Z`.

    Args:
        target_module_name: Module hosting the port, e.g. ``nic_card_mount_2``
            (SFP) or ``sc_port_0`` (SC).
        port_name: Port name from the task, e.g. ``sfp_port_0`` (SFP) or
            ``sc_port_base`` (SC).
        port_type: Port type from the task, e.g. ``sfp`` or ``sc``; matched
            case-insensitively after stripping surrounding whitespace.

    Returns:
        The :class:`PortApproach` describing the frame to approach and the descent
        floor to stop at.

    Raises:
        ValueError: If any identifier is empty.
    """
    if not target_module_name or not port_name or not port_type:
        raise ValueError(
            "target_module_name, port_name and port_type must all be non-empty; "
            f"got {target_module_name!r}, {port_name!r}, {port_type!r}"
        )
    base_frame = f"task_board/{target_module_name}/{port_name}_link"
    if port_type.strip().lower() == SC_PORT_TYPE:
        return PortApproach(
            frame=f"{base_frame}_entrance", descent_floor_z=SC_DESCENT_FLOOR_Z
        )
    return PortApproach(frame=base_frame, descent_floor_z=SFP_DESCENT_FLOOR_Z)
