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

Beyond *which* frame to approach, SC also needs a pose-conditioned **pre-insertion
waypoint**: the generic primitive staged the plug a fixed amount straight up in
``base_link`` world-z above the port frame and then descended straight down. That
ignores the port's actual orientation, so under the eval-band board yaw + the
small target roll/pitch jitter the staging point and descent drift off the port's
true insertion axis and the plug grazes a distractor mount (observed contact ~-24,
Phase-2 forensics ``docs/research_2026-07-19/01_failure_forensics.md`` section 4).
:func:`sc_entrance_waypoint` computes the staging point and the insertion axis
*from the port pose the oracle already reads* (privileged, teacher-side only): the
SC entrance frame's local +z points into the port (the entrance sits at local
``-z = -0.01564`` under ``sc_port_base_link``), so the pre-insertion waypoint is
the entrance position stepped :data:`SC_APPROACH_STANDOFF_M` back out along that
axis. The function is pure (pose in -> waypoint out) so it unit-tests without ROS.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from aic_example_policies.ros.port_offset import rotate_vector_by_quat

# ``Task.port_type`` value (aic_task_interfaces/Task.msg) that denotes an SC port.
SC_PORT_TYPE = "sc"

# Descent floor: the final ``z_offset`` (metres, added to the target frame's z in
# base_link) at which ``CheatCode`` stops driving the plug down.
#
# SFP keeps the historically-tuned value so SFP behaviour is byte-identical.
SFP_DESCENT_FLOOR_Z = -0.015
# SC targets the *entrance* frame (already offset into the port mouth), so the
# descent should be shallow -- stop just past the entrance rather than -0.015 past
# the rotated port centre (which rammed the port body).
#
# Phase-2 forensics (docs/research_2026-07-19/01_failure_forensics.md section 4)
# found -0.005 seated the *official* SC pose but left off-official poses partially
# inserted (insertion_events == 0, score 62.8/65.0): the descent stopped ~2 mm
# short of full seating. The forensic recommendation was to micro-tune the floor
# to ~-0.007 (2 mm deeper) and re-validate zero-contact. -0.007 stays well inside
# the -0.01564 entrance offset so it cannot reach the port body.
#
# Revalidation 2026-07-19 (results/sc_oracle_reval, 7 SC poses at -0.007): 2 full
# seats (official_3, cfg_007 -- both sc_port_1) and 4 partial insertions + 1
# near-miss all stopping exactly 0.01 m short, zero contacts in every trial.
# The sc_port_0-module poses under-reach uniformly, so the floor moves 3 mm
# deeper. -0.010 still stays inside the -0.01564 entrance offset. If any contact
# appears at -0.010, raise back toward -0.007; do not deepen further.
SC_DESCENT_FLOOR_Z = -0.010

# --- Pose-conditioned SC entrance-waypoint geometry (used by sc_entrance_waypoint) ---
# The SC entrance frame's local +z axis points *into* the port: the entrance link
# sits at local (0, 0, -0.01564) under ``sc_port_base_link`` (SC Port/model.sdf),
# so travelling from the entrance toward the seated pose is +z, and the direction
# out of the mouth (where the plug stages before insertion) is -z.
PORT_INSERTION_AXIS_LOCAL: tuple[float, float, float] = (0.0, 0.0, 1.0)
# Pre-insertion standoff (m): how far *outside* the entrance mouth, along the port
# axis, the pose-conditioned staging waypoint sits. Matches the legacy fixed hover
# height (0.2 m) so the working official SC pose is preserved; kept a named
# constant so the standoff is a one-line, ROS-free tuning knob.
SC_APPROACH_STANDOFF_M: float = 0.2


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


@dataclasses.dataclass(frozen=True)
class EntranceApproach:
    """A pose-conditioned pre-insertion staging point and its insertion axis.

    Attributes:
        approach_point: The pre-insertion staging point ``[x, y, z]`` in the same
            frame as the input port pose (typically ``base_link``), sitting
            :attr:`standoff_m` outside the entrance mouth along the port axis.
        insertion_axis: Unit vector ``[x, y, z]`` (same frame) pointing *into* the
            port -- the direction the plug should descend/insert along. Descend
            from :attr:`approach_point` along ``+insertion_axis`` to seat.
        standoff_m: The standoff distance (m) used, i.e.
            ``approach_point = entrance_position - standoff_m * insertion_axis``.
    """

    approach_point: np.ndarray
    insertion_axis: np.ndarray
    standoff_m: float


def sc_entrance_waypoint(
    entrance_position: np.ndarray,
    entrance_orientation: np.ndarray,
    standoff_m: float = SC_APPROACH_STANDOFF_M,
) -> EntranceApproach:
    """Compute a pose-conditioned SC pre-insertion waypoint from the port pose.

    Replaces the legacy fixed-frame waypoint (a fixed world-z hover straight above
    the port frame) with one derived from the port's actual pose. The SC entrance
    frame's local :data:`PORT_INSERTION_AXIS_LOCAL` (``+z``) points into the port,
    so this rotates that axis into the pose's frame to get the base-frame insertion
    direction, then steps ``standoff_m`` back out along it to place the staging
    point outside the mouth::

        insertion_axis = R(entrance_orientation) @ PORT_INSERTION_AXIS_LOCAL
        approach_point = entrance_position - standoff_m * insertion_axis

    The function is pure (pose in -> waypoint out) and free of ROS/Gazebo imports so
    it unit-tests on any machine. At the canonical SC mount the resolved axis is
    ~world-vertical, so this reduces to the legacy straight-down descent; under the
    eval-band board yaw and the target roll/pitch jitter it tracks the true axis.

    Args:
        entrance_position: The SC entrance frame origin ``[x, y, z]`` (m),
            expressed in ``base_link`` (or any frame; the result is in that frame).
        entrance_orientation: The SC entrance frame orientation as a quaternion
            ``[x, y, z, w]`` (``geometry_msgs/Quaternion`` field order) in the same
            frame; renormalized internally.
        standoff_m: Pre-insertion standoff distance (m) along the port axis;
            defaults to :data:`SC_APPROACH_STANDOFF_M`. Must be finite and >= 0.

    Returns:
        The :class:`EntranceApproach` with the staging point and the unit
        base-frame insertion axis.

    Raises:
        ValueError: If ``entrance_position`` is not 3 finite values, ``standoff_m``
            is negative or non-finite, or ``entrance_orientation`` has near-zero
            norm (delegated to :func:`port_offset.rotate_vector_by_quat`).
    """
    pos = np.asarray(entrance_position, dtype=np.float64).reshape(-1)
    if pos.shape[0] != 3:
        raise ValueError(
            f"entrance_position must have length 3, got {pos.shape[0]}"
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"entrance_position must be finite, got {entrance_position!r}")
    if not np.isfinite(standoff_m):
        raise ValueError(f"standoff_m must be finite, got {standoff_m!r}")
    if standoff_m < 0.0:
        raise ValueError(f"standoff_m must be >= 0, got {standoff_m!r}")
    insertion_axis = rotate_vector_by_quat(
        entrance_orientation, np.asarray(PORT_INSERTION_AXIS_LOCAL, dtype=np.float64)
    )
    approach_point = pos - float(standoff_m) * insertion_axis
    return EntranceApproach(
        approach_point=approach_point,
        insertion_axis=insertion_axis,
        standoff_m=float(standoff_m),
    )
