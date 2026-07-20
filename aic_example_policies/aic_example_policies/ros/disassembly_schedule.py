#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Pure (ROS-free) perturbed-retract schedule for the ``DisassembleCode`` policy.

``DisassembleCode`` collects insertion demos *by disassembly*: it drives the
gripper-welded plug down to seat depth, then -- without dwelling (so the port's
1 s continuous-contact touch latch never fires and the seat stays reversible) --
executes a slow, laterally-perturbed retract along the port axis. That retract is
later time-reversed (:mod:`reverse_disasm`) into an insertion demo whose action
label *contains lateral correction*, which the pure-vertical oracle last-inch
lacks (lat/vert 0.014).

This module owns only the *geometry* of that retract -- the per-waypoint offsets
relative to the retract-start pose -- as pure numpy so it unit-tests on any
machine (no ROS, Gazebo, torch, or GPU), mirroring
:mod:`aic_example_policies.ros.cheatcode_targeting`. ``DisassembleCode`` samples a
:class:`RetractScheduleConfig` per episode, calls :func:`build_schedule` to get an
``(n_steps, 5)`` array of ``[axial_out, lat_x, lat_y, droll, dpitch]`` offsets, and
maps each row to a ``set_pose_target`` pose along the port axis.

Two schedule shapes are produced (InsertionNet static-snapshot rationale --
arXiv:2104.14223 -- the retract is slow so friction shear << normal force, so each
waypoint is a quasi-static contact snapshot):

* **spiral-out** (default): while retracting axially from the mouth plane to
  ``axial_span_m`` out, trace a bounded Archimedean spiral in the lateral plane,
  radius growing ``0 -> radius_max_m`` over ``turns`` turns, with a small
  ``+/- roll/pitch`` wobble. ``radius_max_m`` is drawn log-spaced in 0.2-3.0 mm
  (mass 0.3-1.5 mm) and the azimuth is uniform across episodes so the ensemble
  covers all lateral bearings.
* **lift-translate-reseat** (a ~20 % variant, the only physically viable way to
  teach a > 3 mm recovery): first retract ``lift_axial_m`` (3-5 mm) axially to
  clear the bore, THEN translate ``lift_lateral_m`` (3-8 mm) laterally in near
  free space, THEN descend back. The axial clearance strictly precedes the
  lateral move (:func:`lift_translate_reseat_offsets`), so the reversed insertion
  demo teaches "descend, translate over the hole, seat" rather than an impossible
  straight-line lateral shove through the bore.

Sign/frame convention for the offsets (consumed by ``DisassembleCode``):

* ``axial_out`` (m, >= 0): outward displacement from the retract-start point along
  the port axis (the direction *out* of the port). For SFP the port axis is world
  ``+z``, so ``axial_out`` maps to an increasing ``z_offset``.
* ``lat_x, lat_y`` (m): lateral offset in an orthonormal basis of the plane
  orthogonal to the port axis (for SFP, world ``x``/``y``).
* ``droll, dpitch`` (rad): small orientation perturbations composed onto the
  aligned plug orientation.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# --- Lateral spiral radius band (m). radius_max is drawn LOG-spaced across the
# full band; a log-uniform over 0.2-3.0 mm concentrates its mass around the
# geometric centre ~0.3-1.5 mm, matching the InsertionNet perturbation scale. ---
RADIUS_MIN_M: float = 0.2e-3
RADIUS_MAX_M: float = 3.0e-3

# Archimedean turns to grow the spiral radius 0 -> radius_max.
DEFAULT_TURNS: float = 3.0

# Total outward axial travel of the retract (m), from the retract-start depth to
# well clear of the mouth. ~13 mm clears the SFP bore (retract starts ~-0.013)
# and the remainder is the mouth-plane -> ~10-15 mm-out span the demo needs.
DEFAULT_AXIAL_SPAN_M: float = 0.025

# Small roll/pitch wobble band (rad); the per-episode amplitude is drawn uniformly
# in [min, max] so the plug samples a spread of approach angles.
ROLL_PITCH_MIN_RAD: float = 0.02
ROLL_PITCH_MAX_RAD: float = 0.05

# Fraction of episodes that use the lift-translate-reseat recovery variant.
DEFAULT_LIFT_FRAC: float = 0.2

# Lift-translate-reseat magnitudes (m): axial lift to clear the bore, then the
# lateral translate in near-free-space.
LIFT_AXIAL_MIN_M: float = 0.003
LIFT_AXIAL_MAX_M: float = 0.005
LIFT_LATERAL_MIN_M: float = 0.003
LIFT_LATERAL_MAX_M: float = 0.008


@dataclasses.dataclass(frozen=True)
class ScheduleDefaults:
    """Env-tunable knobs for :func:`sample_config` (the per-run distribution).

    Attributes:
        radius_min_m: Lower bound of the log-spaced spiral ``radius_max`` draw (m).
        radius_max_m: Upper bound of the log-spaced spiral ``radius_max`` draw (m).
        turns: Archimedean turns to grow the spiral radius to its max.
        axial_span_m: Total outward axial retract travel (m).
        roll_pitch_min_rad: Lower bound of the per-episode roll/pitch amplitude.
        roll_pitch_max_rad: Upper bound of the per-episode roll/pitch amplitude.
        lift_frac: Probability an episode uses the lift-translate-reseat variant.
        lift_axial_min_m: Lower bound of the axial bore-clearing lift (m).
        lift_axial_max_m: Upper bound of the axial bore-clearing lift (m).
        lift_lateral_min_m: Lower bound of the free-space lateral translate (m).
        lift_lateral_max_m: Upper bound of the free-space lateral translate (m).
    """

    radius_min_m: float = RADIUS_MIN_M
    radius_max_m: float = RADIUS_MAX_M
    turns: float = DEFAULT_TURNS
    axial_span_m: float = DEFAULT_AXIAL_SPAN_M
    roll_pitch_min_rad: float = ROLL_PITCH_MIN_RAD
    roll_pitch_max_rad: float = ROLL_PITCH_MAX_RAD
    lift_frac: float = DEFAULT_LIFT_FRAC
    lift_axial_min_m: float = LIFT_AXIAL_MIN_M
    lift_axial_max_m: float = LIFT_AXIAL_MAX_M
    lift_lateral_min_m: float = LIFT_LATERAL_MIN_M
    lift_lateral_max_m: float = LIFT_LATERAL_MAX_M

    def __post_init__(self) -> None:
        """Validate the band bounds.

        Raises:
            ValueError: If any band is non-positive or inverted, or ``lift_frac``
                is outside ``[0, 1]``.
        """
        if not (0.0 < self.radius_min_m <= self.radius_max_m):
            raise ValueError(
                "require 0 < radius_min_m <= radius_max_m, got "
                f"{self.radius_min_m}, {self.radius_max_m}"
            )
        if self.turns <= 0.0:
            raise ValueError(f"turns must be > 0, got {self.turns}")
        if self.axial_span_m <= 0.0:
            raise ValueError(f"axial_span_m must be > 0, got {self.axial_span_m}")
        if not (0.0 <= self.roll_pitch_min_rad <= self.roll_pitch_max_rad):
            raise ValueError(
                "require 0 <= roll_pitch_min_rad <= roll_pitch_max_rad, got "
                f"{self.roll_pitch_min_rad}, {self.roll_pitch_max_rad}"
            )
        if not (0.0 <= self.lift_frac <= 1.0):
            raise ValueError(f"lift_frac must be in [0, 1], got {self.lift_frac}")
        if not (0.0 < self.lift_axial_min_m <= self.lift_axial_max_m):
            raise ValueError(
                "require 0 < lift_axial_min_m <= lift_axial_max_m, got "
                f"{self.lift_axial_min_m}, {self.lift_axial_max_m}"
            )
        if not (0.0 < self.lift_lateral_min_m <= self.lift_lateral_max_m):
            raise ValueError(
                "require 0 < lift_lateral_min_m <= lift_lateral_max_m, got "
                f"{self.lift_lateral_min_m}, {self.lift_lateral_max_m}"
            )


@dataclasses.dataclass(frozen=True)
class RetractScheduleConfig:
    """A single episode's fully-resolved perturbed-retract geometry.

    All lengths are metres and all angles radians. Produced by
    :func:`sample_config` and consumed by :func:`build_schedule`.

    Attributes:
        n_steps: Number of retract waypoints (>= 2).
        axial_span_m: Total outward axial travel from the retract-start (> 0).
        radius_max_m: Peak lateral spiral radius (>= 0).
        turns: Archimedean turns to grow the radius to ``radius_max_m`` (> 0).
        azimuth0_rad: Starting spiral azimuth (uniform across episodes).
        roll_amp_rad: Roll wobble amplitude (>= 0).
        pitch_amp_rad: Pitch wobble amplitude (>= 0).
        lift_translate: When True, use the lift-translate-reseat recovery variant
            instead of the spiral-out; ``radius_max_m``/``turns`` are then ignored.
        lift_axial_m: Axial bore-clearing lift for the recovery variant (> 0).
        lift_lateral_m: Free-space lateral translate for the recovery variant
            (> 0).
    """

    n_steps: int
    axial_span_m: float
    radius_max_m: float
    turns: float
    azimuth0_rad: float
    roll_amp_rad: float
    pitch_amp_rad: float
    lift_translate: bool = False
    lift_axial_m: float = LIFT_AXIAL_MIN_M
    lift_lateral_m: float = LIFT_LATERAL_MIN_M

    def __post_init__(self) -> None:
        """Validate the resolved geometry.

        Raises:
            ValueError: If any field is out of range.
        """
        if self.n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {self.n_steps}")
        if self.axial_span_m <= 0.0:
            raise ValueError(f"axial_span_m must be > 0, got {self.axial_span_m}")
        if self.radius_max_m < 0.0:
            raise ValueError(f"radius_max_m must be >= 0, got {self.radius_max_m}")
        if self.turns <= 0.0:
            raise ValueError(f"turns must be > 0, got {self.turns}")
        if self.roll_amp_rad < 0.0 or self.pitch_amp_rad < 0.0:
            raise ValueError(
                "roll/pitch amplitudes must be >= 0, got "
                f"{self.roll_amp_rad}, {self.pitch_amp_rad}"
            )
        if self.lift_axial_m <= 0.0:
            raise ValueError(f"lift_axial_m must be > 0, got {self.lift_axial_m}")
        if self.lift_lateral_m <= 0.0:
            raise ValueError(f"lift_lateral_m must be > 0, got {self.lift_lateral_m}")
        if self.lift_axial_m >= self.axial_span_m:
            raise ValueError(
                "lift_axial_m must be < axial_span_m so the reseat/continue-out "
                f"segment is non-empty, got {self.lift_axial_m} >= {self.axial_span_m}"
            )


def draw_radius_max(
    rng: np.random.Generator,
    radius_min_m: float = RADIUS_MIN_M,
    radius_max_m: float = RADIUS_MAX_M,
) -> float:
    """Draw a spiral ``radius_max`` log-uniformly in ``[radius_min_m, radius_max_m]``.

    Log-spacing (uniform in ``log(radius)``) concentrates the mass toward the
    smaller end of the band, so most episodes get a sub-millimetre perturbation
    (the InsertionNet static-snapshot scale) while a tail reaches the 3 mm rim.

    Args:
        rng: Seeded numpy generator (per-episode determinism).
        radius_min_m: Lower band bound (m), must be > 0.
        radius_max_m: Upper band bound (m), must be >= ``radius_min_m``.

    Returns:
        A radius (m) in ``[radius_min_m, radius_max_m]``.

    Raises:
        ValueError: If the band is non-positive or inverted.
    """
    if not (0.0 < radius_min_m <= radius_max_m):
        raise ValueError(
            f"require 0 < radius_min_m <= radius_max_m, got {radius_min_m}, {radius_max_m}"
        )
    log_lo = np.log(radius_min_m)
    log_hi = np.log(radius_max_m)
    return float(np.exp(rng.uniform(log_lo, log_hi)))


def sample_config(
    rng: np.random.Generator, n_steps: int, defaults: ScheduleDefaults | None = None
) -> RetractScheduleConfig:
    """Draw one episode's :class:`RetractScheduleConfig` from the defaults.

    Draws (all per-episode): the log-spaced spiral ``radius_max``, a uniform
    starting azimuth in ``[0, 2*pi)``, uniform roll/pitch amplitudes in the
    defaults' band, whether this is a lift-translate-reseat episode (Bernoulli
    ``lift_frac``), and -- for the recovery variant -- uniform lift-axial and
    lift-lateral magnitudes.

    Args:
        rng: Seeded numpy generator (per-episode determinism).
        n_steps: Number of retract waypoints (>= 2).
        defaults: The env-tuned distribution; :class:`ScheduleDefaults` when None.

    Returns:
        The fully-resolved :class:`RetractScheduleConfig`.

    Raises:
        ValueError: If ``n_steps`` < 2 (delegated to the config validator).
    """
    d = defaults if defaults is not None else ScheduleDefaults()
    lift = bool(rng.random() < d.lift_frac)
    return RetractScheduleConfig(
        n_steps=int(n_steps),
        axial_span_m=d.axial_span_m,
        radius_max_m=draw_radius_max(rng, d.radius_min_m, d.radius_max_m),
        turns=d.turns,
        azimuth0_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        roll_amp_rad=float(rng.uniform(d.roll_pitch_min_rad, d.roll_pitch_max_rad)),
        pitch_amp_rad=float(rng.uniform(d.roll_pitch_min_rad, d.roll_pitch_max_rad)),
        lift_translate=lift,
        lift_axial_m=float(rng.uniform(d.lift_axial_min_m, d.lift_axial_max_m)),
        lift_lateral_m=float(rng.uniform(d.lift_lateral_min_m, d.lift_lateral_max_m)),
    )


def spiral_offsets(cfg: RetractScheduleConfig) -> np.ndarray:
    """Return the Archimedean spiral-out retract offsets, ``(n_steps, 5)``.

    Over ``s = 0 -> 1`` (``n_steps`` samples): the axial coordinate travels
    ``0 -> axial_span_m`` (monotone outward), the spiral radius grows
    ``0 -> radius_max_m`` linearly (an Archimedean spiral advancing ``turns`` full
    turns), and the roll/pitch wobble follows the spiral azimuth at the configured
    amplitudes::

        theta   = azimuth0 + 2*pi*turns*s
        radius  = radius_max * s
        lat_x   = radius * cos(theta)
        lat_y   = radius * sin(theta)
        axial   = axial_span * s
        droll   = roll_amp  * sin(theta)
        dpitch  = pitch_amp * cos(theta)

    Args:
        cfg: The resolved schedule config.

    Returns:
        A ``(n_steps, 5)`` float64 array of ``[axial_out, lat_x, lat_y, droll,
        dpitch]``; ``axial_out`` is non-decreasing and the lateral radius per row
        never exceeds ``radius_max_m``.
    """
    s = np.linspace(0.0, 1.0, cfg.n_steps)
    theta = cfg.azimuth0_rad + 2.0 * np.pi * cfg.turns * s
    radius = cfg.radius_max_m * s
    out = np.zeros((cfg.n_steps, 5), dtype=np.float64)
    out[:, 0] = cfg.axial_span_m * s
    out[:, 1] = radius * np.cos(theta)
    out[:, 2] = radius * np.sin(theta)
    out[:, 3] = cfg.roll_amp_rad * np.sin(theta)
    out[:, 4] = cfg.pitch_amp_rad * np.cos(theta)
    return out


def lift_translate_reseat_offsets(cfg: RetractScheduleConfig) -> np.ndarray:
    """Return the lift-translate-reseat recovery offsets, ``(n_steps, 5)``.

    Three contiguous segments (roughly equal in waypoint count), in the recorded
    (disassembly) time order, so that when :mod:`reverse_disasm` time-reverses the
    episode the insertion demo reads "descend, translate over the hole, seat":

    1. **lift** -- axial ``0 -> lift_axial_m`` with the lateral offset held at 0
       (pull straight out to clear the bore before moving sideways).
    2. **translate** -- axial held at ``lift_axial_m`` while the lateral offset
       goes ``0 -> lift_lateral_m`` along ``azimuth0`` (move sideways in near-free
       space, above the mouth).
    3. **continue-out** -- axial ``lift_axial_m -> axial_span_m`` with the lateral
       offset held at ``lift_lateral_m`` (retract the rest of the way out).

    The axial bore-clearing motion of segment 1 strictly precedes any lateral
    motion, which is the property that makes the reversed recovery physically
    realisable (no lateral shove through the bore). Roll/pitch are held at 0.

    Args:
        cfg: The resolved schedule config (``lift_axial_m`` < ``axial_span_m``).

    Returns:
        A ``(n_steps, 5)`` float64 array of ``[axial_out, lat_x, lat_y, droll,
        dpitch]``; ``axial_out`` is non-decreasing.
    """
    n = cfg.n_steps
    n_lift = max(1, n // 3)
    n_trans = max(1, n // 3)
    # The three segments must sum to exactly n; the continue-out segment takes the
    # remainder. When n is small, borrow from translate so continue-out has >= 1.
    n_out = n - n_lift - n_trans
    if n_out < 1:
        n_trans = n - n_lift - 1
        n_out = 1
    az = cfg.azimuth0_rad
    cx, cy = float(np.cos(az)), float(np.sin(az))
    out = np.zeros((n, 5), dtype=np.float64)

    # Segment 1: axial 0 -> lift_axial, lateral 0.
    lift_axial = np.linspace(0.0, cfg.lift_axial_m, n_lift)
    out[:n_lift, 0] = lift_axial

    # Segment 2: axial held at lift_axial, lateral 0 -> lift_lateral along az.
    trans = np.linspace(0.0, cfg.lift_lateral_m, n_trans)
    sl = slice(n_lift, n_lift + n_trans)
    out[sl, 0] = cfg.lift_axial_m
    out[sl, 1] = trans * cx
    out[sl, 2] = trans * cy

    # Segment 3: axial lift_axial -> axial_span, lateral held at lift_lateral.
    cont = np.linspace(cfg.lift_axial_m, cfg.axial_span_m, n_out)
    so = slice(n_lift + n_trans, n)
    out[so, 0] = cont
    out[so, 1] = cfg.lift_lateral_m * cx
    out[so, 2] = cfg.lift_lateral_m * cy
    return out


def build_schedule(cfg: RetractScheduleConfig) -> np.ndarray:
    """Return the per-episode retract offsets, dispatching on the variant.

    Args:
        cfg: The resolved schedule config.

    Returns:
        A ``(n_steps, 5)`` array of ``[axial_out, lat_x, lat_y, droll, dpitch]``
        offsets: the lift-translate-reseat recovery path when
        ``cfg.lift_translate`` is set, otherwise the spiral-out path.
    """
    if cfg.lift_translate:
        return lift_translate_reseat_offsets(cfg)
    return spiral_offsets(cfg)
