#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""ROS-free guarded-descent probe for the stalled ACT-lite deploy policy.

``DeployACT`` approaches the port cleanly then *stalls* 0.05-0.08 m short: the
deterministic ACT head mode-averages the demos' bimodal "keep pushing / stopped"
action distribution toward *zero* base_link twist, and because every cycle
re-anchors the MODE_POSITION target to the *measured* TCP pose, a zero predicted
twist is a fixed-point attractor (no motion). Temporal ensembling
(``chunk_ensemble.py``) is one attempt to break out; this module is a
complementary, model-free fallback: once a stall is *detected*, hand control to a
scripted **guarded descent** that steps the pose target along the estimated
insertion axis under a wrench-triggered contact guard.

The probe is deliberately eval-legal: it consumes only RGB/TCP/wrench signals
already in the ``Observation`` (no ground-truth port pose, no privileged
topics). The insertion axis is *estimated* from the direction the arm was
actually moving just before it stalled.

Three cooperating pieces (all pure ``numpy``, unit-testable with no ROS/torch):

* :class:`StallDetector` -- flags a sustained low-TCP-speed condition, but only
  after a startup grace period so the initial settle does not trigger it.
* :class:`ApproachAxisEstimator` -- estimates the insertion axis as the
  normalized mean of the last ``N`` non-trivial TCP displacement vectors (the
  approach direction captured *before* speed dropped).
* :class:`GuardedDescent` -- the per-cycle target generator + state machine:
  steps a fixed step along the axis, backs off one step and holds on a
  baseline-relative wrench spike, and holds once a total-travel cap is reached.

:class:`GuardedDescentController` wires the three together behind the single
per-cycle seam ``DeployACT`` calls (mirroring ``select_exec_twists`` in
``chunk_ensemble.py``): while approaching it returns ``targets=None`` so the
learned path runs unchanged (byte-identical when disabled), and once the stall
fires it returns the guarded pose targets to command instead.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import math
import os
import pathlib
from typing import Iterable, Mapping, Protocol

import numpy as np

from aic_example_policies.ros import pose_integration

# ---------------------------------------------------------------------------
# Defaults (all overridable through ``GuardedDescentConfig`` / ``AIC_GUARDED_*``).
# ---------------------------------------------------------------------------
DEFAULT_SPEED_THRESHOLD: float = 0.01  # m/s; below this the TCP is "stalled".
DEFAULT_STALL_WINDOW_S: float = 3.0  # s of sustained low speed before firing.
DEFAULT_MIN_RUNTIME_S: float = 15.0  # s grace period so startup does not trigger.
DEFAULT_AXIS_WINDOW: int = 8  # number of recent displacements averaged for the axis.
DEFAULT_AXIS_MIN_DISPLACEMENT: float = 1e-4  # m; smaller per-cycle moves are ignored.
DEFAULT_STEP_SIZE: float = 0.004  # m advanced along the axis per descent cycle.
DEFAULT_TRAVEL_CAP: float = 0.12  # m total commanded travel before holding.
DEFAULT_CONTACT_FORCE_THRESHOLD: float = 12.0  # N of baseline-relative |F| spike.
# CheatCode / DeployACT MODE_POSITION gains; base for the optional reduced-Z axis.
DEFAULT_BASE_STIFFNESS: tuple[float, ...] = (90.0, 90.0, 90.0, 50.0, 50.0, 50.0)

# --- Learned port-bearing (``port_aux``) handoff defaults (design section 4). ---
# The aux head's TCP->port offset replaces the motion-axis guess at handoff when
# it is plausible AND consistent; otherwise the controller falls back to the
# ApproachAxisEstimator (today's behavior). All overridable via AIC_GUARDED_AUX*.
DEFAULT_AUX_MIN_MAG: float = 0.005  # m; a smaller predicted offset is rejected.
DEFAULT_AUX_MAX_MAG: float = 0.12  # m; a larger predicted offset is rejected.
DEFAULT_AUX_CONSISTENCY_STD: float = 0.01  # m; cross-frame target spread ceiling.
DEFAULT_AUX_TRAVEL_MARGIN: float = 0.02  # m added to |offset| for the travel cap.
DEFAULT_AUX_BUFFER: int = 6  # recent approach-frame predictions kept for the median.
DEFAULT_AUX_MIN_SAMPLES: int = 3  # plausible predictions required before a handoff.
DEFAULT_REAIM_RATE: float = 0.2  # per-cycle axis blend toward a re-queried target.
# Fixed deep-travel cap (m) used when the aux head supplies an EXPLICIT approach
# axis (6-D checkpoint): the descent steers by that axis and travels a fixed
# distance, decoupled from the unreliable predicted |offset| magnitude (forensics
# 2026-07-19: depth val error ~4.6 cm, lateral/axis ~0.86 cm).
DEFAULT_AUX_FIXED_TRAVEL: float = 0.10

# --- Learned insertion-specialist (``AIC_SPECIALIST``) handoff defaults (INSERTION
# _PLAN.md code change #2). Once the approach stalls, a separate specialist
# checkpoint's predicted TCP-velocity chunks replace the scripted GuardedDescent,
# integrated into pose targets under the SAME travel-cap and wrench back-off
# guards. All overridable via ``AIC_SPECIALIST_*``. ---
DEFAULT_SPECIALIST_TRAVEL_CAP: float = 0.12  # m total travel from the anchor cap.
DEFAULT_SPECIALIST_MAX_STEPS: int = 0  # cap on integrated chunk steps (0 = unlimited).
DEFAULT_SPECIALIST_EXEC_STEPS: int = 4  # chunk steps integrated per descent cycle.
DEFAULT_SPECIALIST_DT: float = 0.275  # s training frame period for velocity integration.

# --- Scripted spiral-search insertion primitive (``AIC_SEARCH``) defaults. Once
# the approach stalls at the port mouth, a straight push jams on the rim
# (lateral/angular misalignment); the model-free fix (InsertionNet arXiv:2104.14223;
# "From Reach to Insert" arXiv:2605.04649) is a bounded Archimedean spiral/wiggle
# in the plane orthogonal to the approach axis under a downward force-push -- the
# lateral search finds the hole a straight push misses. Fully deploy-legal: uses
# only the current TCP pose + wrist wrench + the stall anchor (NO port TF). All
# overridable via ``AIC_SEARCH_*``. ---
DEFAULT_SEARCH_RADIUS: float = 0.004  # m; max lateral spiral radius (~port half-width).
DEFAULT_SEARCH_TURNS: float = 3.0  # Archimedean turns to grow the radius 0 -> max.
DEFAULT_SEARCH_Z_STEP: float = 0.0008  # m; downward push along the axis per cycle.
DEFAULT_SEARCH_ENGAGE_FORCE: float = 4.0  # N; |F-baseline| at/above which the spiral grows.
DEFAULT_SEARCH_TRAVEL_CAP: float = 0.06  # m; max total Z push (search depth) before holding.
DEFAULT_SEARCH_MAX_STEPS: int = 0  # cap on push steps (0 = unlimited).
DEFAULT_SEARCH_YAW_AMP: float = 0.0  # rad; optional tiny yaw-dither amplitude (0 = frozen).
DEFAULT_SEARCH_OMEGA: float = 0.4  # rad; spiral angle advanced per in-contact cycle.
# World-down unit axis for the optional vertical-search override (``AIC_SEARCH_VERTICAL``).
# The SFP/SC ports open ~world-vertical (last-inch forensics: pure world -z descent,
# <1mm lateral), so when the flag is set the spiral descends straight -z with its
# lateral plane in world XY instead of following the (possibly diagonal) motion-axis
# estimate. ``_lateral_basis((0, 0, -1))`` already yields a world-XY ``(u, v)`` pair,
# so passing this axis to :class:`SearchDescent` puts the whole spiral in world X/Y.
_WORLD_DOWN_AXIS: np.ndarray = np.array([0.0, 0.0, -1.0])

# Per-trial guarded telemetry mirror written in the process CWD (the trial dir).
DEFAULT_TRACE_FILENAME: str = "guarded_trace.log"

_QUAT_MIN_NORM: float = 1e-9
_AXIS_MIN_NORM: float = 1e-9


class PortBearingProvider(Protocol):
    """Structural type for a learned TCP->port bearing source (the aux head).

    Implemented by ``DeployACT`` (which runs the aux head on the live
    observation); kept behind this protocol so ``guarded_descent`` stays
    ROS/torch-free and unit-testable. Called once per cycle with the measured TCP
    pose.
    """

    def predict(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray | None, float, bool, np.ndarray | None]:
        """Return the estimated port target (and optional axis) in base_link.

        Args:
            position: Current TCP position ``(3,)`` in base_link (m).
            quaternion: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.

        Returns:
            A tuple ``(target_base, magnitude, ok, axis_base)`` where
            ``target_base`` is the ``(3,)`` predicted port point in base_link (or
            None when unavailable), ``magnitude`` is ``|offset|`` (m), ``ok`` flags
            a usable prediction (finite, plausible magnitude), and ``axis_base`` is
            the ``(3,)`` explicit unit approach axis in base_link when the head
            predicts one (6-D checkpoint) else None. A legacy 3-tuple return
            ``(target_base, magnitude, ok)`` is also accepted (treated as
            ``axis_base = None``).
        """
        ...


class SpecialistActionProvider(Protocol):
    """Structural type for a learned insertion specialist's action source.

    Implemented by ``DeployACT`` (which runs a *separate* specialist checkpoint on
    the live RGB+TCP+wrench observation); kept behind this protocol so
    ``guarded_descent`` stays ROS/torch-free and unit-testable. Called once per
    descent cycle to fetch the next TCP-velocity chunk to integrate.
    """

    def predict_chunk(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> np.ndarray | None:
        """Return the next base_link TCP-velocity chunk for the live observation.

        Args:
            position: Current TCP position ``(3,)`` in base_link (m). The chunk is
                resolved from the held observation, not this pose, but it is passed
                for protocol symmetry with :class:`PortBearingProvider`.
            quaternion: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.

        Returns:
            A ``(K, 6)`` array of base_link-frame twists (m/s, rad/s), or None when
            no observation is available yet (the descent then holds this cycle).
        """
        ...


class GuardedPhase(str, enum.Enum):
    """Lifecycle phase of the guarded-descent probe.

    Attributes:
        APPROACH: The learned policy is still driving; the probe only observes.
        DESCEND: The stall fired; scripted steps are advancing along the axis.
        HOLD: Travel is frozen -- either a contact back-off or the travel cap.
    """

    APPROACH = "approach"
    DESCEND = "descend"
    HOLD = "hold"


class StallDetector:
    """Detects a sustained TCP-speed stall after a startup grace period.

    A stall fires only when the TCP speed has stayed below ``speed_threshold`` for
    a continuous span of at least ``stall_window_s`` seconds *and* the policy has
    been running for at least ``min_runtime_s`` seconds. The grace period stops
    the initial settle (when the arm has not yet accelerated toward the port) from
    triggering. The detector latches: it fires at most once until :meth:`reset`.
    """

    def __init__(
        self,
        speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
        stall_window_s: float = DEFAULT_STALL_WINDOW_S,
        min_runtime_s: float = DEFAULT_MIN_RUNTIME_S,
    ) -> None:
        """Initialize the detector.

        Args:
            speed_threshold: TCP speed (m/s) strictly below which a cycle counts
                as "low". Must be positive.
            stall_window_s: Continuous seconds of low speed required to fire. Must
                be positive.
            min_runtime_s: Seconds of runtime before any fire is allowed. Must be
                non-negative.

        Raises:
            ValueError: If any threshold is out of range.
        """
        if not math.isfinite(speed_threshold) or speed_threshold <= 0.0:
            raise ValueError(f"speed_threshold must be a positive float, got {speed_threshold!r}")
        if not math.isfinite(stall_window_s) or stall_window_s <= 0.0:
            raise ValueError(f"stall_window_s must be a positive float, got {stall_window_s!r}")
        if not math.isfinite(min_runtime_s) or min_runtime_s < 0.0:
            raise ValueError(f"min_runtime_s must be a non-negative float, got {min_runtime_s!r}")
        self.speed_threshold = float(speed_threshold)
        self.stall_window_s = float(stall_window_s)
        self.min_runtime_s = float(min_runtime_s)
        self._low_since: float | None = None
        self._triggered = False

    @property
    def triggered(self) -> bool:
        """Whether the detector has latched a stall."""
        return self._triggered

    @property
    def low_since(self) -> float | None:
        """Timestamp when the current continuous low-speed span began, or None."""
        return self._low_since

    def update(self, t: float, speed: float) -> bool:
        """Feed one cycle's timestamp and TCP speed and test for a stall.

        Args:
            t: Elapsed time since the policy started, in seconds (monotonic).
            speed: Instantaneous TCP speed in m/s (non-negative).

        Returns:
            True only on the single cycle where the stall first fires; False
            otherwise (including every cycle after it has latched).
        """
        if self._triggered:
            return False
        if speed < self.speed_threshold:
            if self._low_since is None:
                self._low_since = float(t)
            low_duration = float(t) - self._low_since
            if float(t) >= self.min_runtime_s and low_duration >= self.stall_window_s:
                self._triggered = True
                return True
        else:
            self._low_since = None
        return False

    def reset(self) -> None:
        """Clear the latch and low-speed span (e.g. at the start of an episode)."""
        self._low_since = None
        self._triggered = False


class ApproachAxisEstimator:
    """Estimates the insertion axis from recent TCP displacement vectors.

    Keeps a rolling buffer of the last ``window`` per-cycle TCP displacement
    vectors whose magnitude exceeds ``min_displacement`` (so the near-zero moves
    of a stall do not dilute the estimate -- the buffer retains the approach
    motion recorded *before* speed dropped). The axis estimate is the normalized
    mean of the buffered displacements.
    """

    def __init__(
        self,
        window: int = DEFAULT_AXIS_WINDOW,
        min_displacement: float = DEFAULT_AXIS_MIN_DISPLACEMENT,
    ) -> None:
        """Initialize the estimator.

        Args:
            window: Number of recent above-threshold displacements to average.
                Must be >= 1.
            min_displacement: Minimum per-cycle displacement magnitude (m) for a
                sample to be recorded. Must be non-negative.

        Raises:
            ValueError: If ``window`` < 1 or ``min_displacement`` is negative.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not math.isfinite(min_displacement) or min_displacement < 0.0:
            raise ValueError(
                f"min_displacement must be a non-negative float, got {min_displacement!r}"
            )
        self.window = int(window)
        self.min_displacement = float(min_displacement)
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self.window)

    def __len__(self) -> int:
        """Number of displacement samples currently buffered."""
        return len(self._buffer)

    def observe(self, displacement: np.ndarray) -> None:
        """Record one per-cycle TCP displacement vector.

        Args:
            displacement: The ``(3,)`` change in TCP position (m) over the last
                cycle. Samples with magnitude below ``min_displacement`` are
                ignored so stationary cycles do not pollute the estimate.

        Raises:
            ValueError: If ``displacement`` is not a length-3 vector.
        """
        d = np.asarray(displacement, dtype=np.float64).reshape(-1)
        if d.shape != (3,):
            raise ValueError(f"displacement must have shape (3,), got {d.shape}")
        if float(np.linalg.norm(d)) >= self.min_displacement:
            self._buffer.append(d)

    def estimate(self) -> np.ndarray | None:
        """Return the unit insertion axis, or None if it cannot be estimated.

        Returns:
            The normalized ``(3,)`` mean of the buffered displacements, or None if
            no samples have been recorded or their mean has near-zero norm (e.g.
            equal-and-opposite motion that cancels out).
        """
        if not self._buffer:
            return None
        mean = np.mean(np.stack(self._buffer), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < _AXIS_MIN_NORM:
            return None
        return mean / norm

    def reset(self) -> None:
        """Drop all buffered displacements."""
        self._buffer.clear()


class GuardedDescent:
    """Scripted per-cycle pose-target generator with contact and travel guards.

    Once constructed at handoff (with a fixed anchor pose, an estimated unit
    axis, and the approach-phase wrench baseline), each :meth:`advance` call
    returns the next absolute ``(7,)`` pose target ``[x, y, z, qx, qy, qz, qw]``:
    the position steps ``step_size`` metres along the axis from the *fixed*
    anchor (never re-anchored to the measured pose -- that is exactly the stall
    attractor), the orientation is held at the anchor. Two guards freeze travel:

    * **Contact back-off** -- if the baseline-relative force magnitude
      ``|F - baseline|`` exceeds ``contact_force_threshold``, retreat one step and
      transition to :attr:`GuardedPhase.HOLD`.
    * **Travel cap** -- once the next step would exceed ``travel_cap`` of total
      commanded travel, hold at the current step.
    """

    def __init__(
        self,
        axis: np.ndarray,
        anchor_position: np.ndarray,
        anchor_quaternion: np.ndarray,
        baseline_force: np.ndarray | None,
        step_size: float = DEFAULT_STEP_SIZE,
        travel_cap: float = DEFAULT_TRAVEL_CAP,
        contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    ) -> None:
        """Initialize the descent from the handoff pose and estimated axis.

        Args:
            axis: Estimated insertion direction ``(3,)`` (need not be unit; it is
                normalized). Must have non-negligible norm.
            anchor_position: TCP position ``(3,)`` at handoff (m); the fixed
                origin the steps accumulate from.
            anchor_quaternion: TCP orientation ``(4,)`` ``[x, y, z, w]`` at
                handoff; held constant through the descent.
            baseline_force: Approach-phase mean force ``(3,)`` (N) for the
                contact guard, or None to disable the contact guard.
            step_size: Metres advanced per :meth:`advance` call. Must be positive.
            travel_cap: Maximum total commanded travel (m) before holding. Must be
                positive.
            contact_force_threshold: Baseline-relative ``|F - baseline|`` (N) that
                triggers a one-step back-off. Must be non-negative.

        Raises:
            ValueError: If any argument is out of range or malformed.
        """
        axis_arr = np.asarray(axis, dtype=np.float64).reshape(-1)
        if axis_arr.shape != (3,):
            raise ValueError(f"axis must have shape (3,), got {axis_arr.shape}")
        axis_norm = float(np.linalg.norm(axis_arr))
        if not math.isfinite(axis_norm) or axis_norm < _AXIS_MIN_NORM:
            raise ValueError(f"axis must have non-negligible norm, got {axis!r}")
        pos = np.asarray(anchor_position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"anchor_position must have shape (3,), got {pos.shape}")
        quat = np.asarray(anchor_quaternion, dtype=np.float64).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"anchor_quaternion must have shape (4,), got {quat.shape}")
        if float(np.linalg.norm(quat)) < _QUAT_MIN_NORM:
            raise ValueError("anchor_quaternion has near-zero norm")
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError(f"step_size must be a positive float, got {step_size!r}")
        if not math.isfinite(travel_cap) or travel_cap <= 0.0:
            raise ValueError(f"travel_cap must be a positive float, got {travel_cap!r}")
        if not math.isfinite(contact_force_threshold) or contact_force_threshold < 0.0:
            raise ValueError(
                f"contact_force_threshold must be a non-negative float, "
                f"got {contact_force_threshold!r}"
            )

        self.axis = axis_arr / axis_norm
        self.anchor_position = pos
        self.anchor_quaternion = quat
        self.baseline_force = (
            np.asarray(baseline_force, dtype=np.float64).reshape(-1)
            if baseline_force is not None
            else None
        )
        if self.baseline_force is not None and self.baseline_force.shape != (3,):
            raise ValueError(
                f"baseline_force must have shape (3,), got {self.baseline_force.shape}"
            )
        self.step_size = float(step_size)
        self.travel_cap = float(travel_cap)
        self.contact_force_threshold = float(contact_force_threshold)

        self.steps = 0
        self.phase = GuardedPhase.DESCEND
        self.contacts = 0
        self.last_force_delta = 0.0

    @property
    def travel(self) -> float:
        """Total commanded travel from the anchor, in metres."""
        return self.step_size * self.steps

    def target(self, steps: int | None = None) -> np.ndarray:
        """Return the absolute pose target for a given step count.

        Args:
            steps: Step index to evaluate; defaults to the current :attr:`steps`.

        Returns:
            The ``(7,)`` pose target ``[x, y, z, qx, qy, qz, qw]``.
        """
        s = self.steps if steps is None else int(steps)
        pos = self.anchor_position + self.axis * (self.step_size * s)
        out = np.empty(7, dtype=np.float64)
        out[:3] = pos
        out[3:] = self.anchor_quaternion
        return out

    def reaim(self, new_axis: np.ndarray, rate: float) -> None:
        """Rate-limit-blend the descent axis toward a freshly-queried direction.

        Used only when ``reaim`` is enabled: nudges the axis toward ``new_axis``
        by fraction ``rate`` and renormalizes, leaving the fixed anchor and travel
        cap untouched so the descent still terminates. A degenerate blend (axes
        nearly opposed) is ignored to avoid a zero-norm axis.

        Args:
            new_axis: Fresh unit direction ``(3,)`` toward the re-queried target.
            rate: Blend fraction in ``[0, 1]`` toward ``new_axis`` this cycle.
        """
        a = np.asarray(new_axis, dtype=np.float64).reshape(-1)
        if a.shape != (3,):
            raise ValueError(f"new_axis must have shape (3,), got {a.shape}")
        norm = float(np.linalg.norm(a))
        if norm < _AXIS_MIN_NORM:
            return
        a = a / norm
        blended = (1.0 - float(rate)) * self.axis + float(rate) * a
        bnorm = float(np.linalg.norm(blended))
        if bnorm < _AXIS_MIN_NORM:
            return
        self.axis = blended / bnorm

    def _force_delta(self, force: np.ndarray | None) -> float:
        """Return ``|F - baseline|`` (N), or 0.0 if the contact guard is inactive.

        Args:
            force: Current wrist force ``(3,)`` (N), or None.

        Returns:
            The baseline-relative force magnitude, or 0.0 when either the force or
            the baseline is unavailable.
        """
        if force is None or self.baseline_force is None:
            return 0.0
        f = np.asarray(force, dtype=np.float64).reshape(-1)
        if f.shape != (3,):
            raise ValueError(f"force must have shape (3,), got {f.shape}")
        return float(np.linalg.norm(f - self.baseline_force))

    def advance(self, force: np.ndarray | None) -> np.ndarray:
        """Advance one descent cycle and return the next pose target.

        Order of guards each cycle: while already holding, re-command the held
        target; otherwise test the contact guard (back off + hold on a spike),
        then the travel cap (hold at the current step), else take one step.

        Args:
            force: Current wrist force ``(3,)`` (N) for the contact guard, or None
                to skip it this cycle.

        Returns:
            The ``(7,)`` pose target to command this cycle.
        """
        self.last_force_delta = self._force_delta(force)
        if self.phase == GuardedPhase.HOLD:
            return self.target()
        if self.baseline_force is not None and self.last_force_delta > self.contact_force_threshold:
            self.steps = max(0, self.steps - 1)
            self.contacts += 1
            self.phase = GuardedPhase.HOLD
            return self.target()
        if self.step_size * (self.steps + 1) > self.travel_cap:
            self.phase = GuardedPhase.HOLD
            return self.target()
        self.steps += 1
        return self.target()


class SpecialistDescent:
    """Learned-specialist descent: integrates a predicted velocity chunk to targets.

    The drop-in learned counterpart to :class:`GuardedDescent` for the specialist
    handoff (INSERTION_PLAN.md change #2). Instead of stepping a fixed distance
    along an estimated axis, each :meth:`advance` queries a
    :class:`SpecialistActionProvider` for the next base_link TCP-velocity chunk
    (from the live RGB+TCP+wrench observation, exactly the deploy-legal signals the
    main policy uses -- no port TF), integrates the first ``exec_steps`` twists
    (sub-stepped) into absolute pose targets anchored to the *measured* pose
    (receding horizon), and applies the SAME two safety guards as the scripted
    descent:

    * **Wrench back-off** -- if ``|F - baseline|`` exceeds
      ``contact_force_threshold`` the chunk is discarded, a single one-step retreat
      target is commanded, and the descent transitions to :attr:`GuardedPhase.HOLD`.
    * **Travel cap / max steps** -- targets are clamped to lie within
      ``travel_cap`` of the fixed anchor, and once the cap (or ``max_steps``) is
      hit the descent holds.

    Duck-types the attributes ``GuardedDescentController`` reads off the scripted
    descent (:attr:`steps`, :attr:`travel`, :attr:`contacts`,
    :attr:`last_force_delta`, :attr:`phase`), so the controller's result assembly
    is shared unchanged.
    """

    def __init__(
        self,
        provider: SpecialistActionProvider,
        anchor_position: np.ndarray,
        anchor_quaternion: np.ndarray,
        baseline_force: np.ndarray | None,
        travel_cap: float = DEFAULT_SPECIALIST_TRAVEL_CAP,
        max_steps: int = DEFAULT_SPECIALIST_MAX_STEPS,
        exec_steps: int = DEFAULT_SPECIALIST_EXEC_STEPS,
        dt_frame: float = DEFAULT_SPECIALIST_DT,
        step_size: float = DEFAULT_STEP_SIZE,
        contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    ) -> None:
        """Initialize the specialist descent from the handoff pose.

        Args:
            provider: Learned action source queried once per descent cycle.
            anchor_position: TCP position ``(3,)`` at handoff (m); the fixed origin
                the travel cap is measured from.
            anchor_quaternion: TCP orientation ``(4,)`` ``[x, y, z, w]`` at handoff
                (kept for telemetry/back-off; the specialist predicts its own
                orientation deltas).
            baseline_force: Approach-phase mean force ``(3,)`` (N) for the contact
                guard, or None to disable it.
            travel_cap: Maximum total travel (m) from the anchor before holding.
                Must be positive.
            max_steps: Cap on integrated chunk steps (0 = unlimited). Must be >= 0.
            exec_steps: Chunk steps integrated per cycle. Must be >= 1.
            dt_frame: Training frame period (s) for velocity integration. Must be
                positive.
            step_size: Retreat distance (m) commanded on a contact back-off. Must
                be positive.
            contact_force_threshold: Baseline-relative ``|F - baseline|`` (N) that
                triggers the back-off. Must be non-negative.

        Raises:
            ValueError: If any argument is out of range or malformed.
        """
        pos = np.asarray(anchor_position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"anchor_position must have shape (3,), got {pos.shape}")
        quat = np.asarray(anchor_quaternion, dtype=np.float64).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"anchor_quaternion must have shape (4,), got {quat.shape}")
        if float(np.linalg.norm(quat)) < _QUAT_MIN_NORM:
            raise ValueError("anchor_quaternion has near-zero norm")
        if not math.isfinite(travel_cap) or travel_cap <= 0.0:
            raise ValueError(f"travel_cap must be a positive float, got {travel_cap!r}")
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}")
        if exec_steps < 1:
            raise ValueError(f"exec_steps must be >= 1, got {exec_steps}")
        if not math.isfinite(dt_frame) or dt_frame <= 0.0:
            raise ValueError(f"dt_frame must be a positive float, got {dt_frame!r}")
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError(f"step_size must be a positive float, got {step_size!r}")
        if not math.isfinite(contact_force_threshold) or contact_force_threshold < 0.0:
            raise ValueError(
                f"contact_force_threshold must be a non-negative float, "
                f"got {contact_force_threshold!r}"
            )

        self.provider = provider
        self.anchor_position = pos
        self.anchor_quaternion = quat
        self.baseline_force = (
            np.asarray(baseline_force, dtype=np.float64).reshape(-1)
            if baseline_force is not None
            else None
        )
        if self.baseline_force is not None and self.baseline_force.shape != (3,):
            raise ValueError(
                f"baseline_force must have shape (3,), got {self.baseline_force.shape}"
            )
        self.travel_cap = float(travel_cap)
        self.max_steps = int(max_steps)
        self.exec_steps = int(exec_steps)
        self.dt_frame = float(dt_frame)
        self.step_size = float(step_size)
        self.contact_force_threshold = float(contact_force_threshold)

        self.steps = 0
        self.phase = GuardedPhase.DESCEND
        self.contacts = 0
        self.last_force_delta = 0.0
        self._travel = 0.0
        self._last_position = pos.copy()
        self._last_dir: np.ndarray | None = None

    @property
    def travel(self) -> float:
        """Total commanded travel from the anchor to the last target, in metres."""
        return self._travel

    def _force_delta(self, force: np.ndarray | None) -> float:
        """Return ``|F - baseline|`` (N), or 0.0 when the contact guard is inactive.

        Args:
            force: Current wrist force ``(3,)`` (N), or None.

        Returns:
            The baseline-relative force magnitude, or 0.0 when either the force or
            the baseline is unavailable.
        """
        if force is None or self.baseline_force is None:
            return 0.0
        f = np.asarray(force, dtype=np.float64).reshape(-1)
        if f.shape != (3,):
            raise ValueError(f"force must have shape (3,), got {f.shape}")
        return float(np.linalg.norm(f - self.baseline_force))

    def _hold_target(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Return a single ``(1, 7)`` target that freezes at the measured pose.

        Args:
            position: Current TCP position ``(3,)`` (m).
            quaternion: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.

        Returns:
            A ``(1, 7)`` pose-target array re-commanding the current pose.
        """
        out = np.empty((1, 7), dtype=np.float64)
        out[0, :3] = position
        out[0, 3:] = quaternion
        return out

    def _backoff_target(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> np.ndarray:
        """Return a ``(1, 7)`` target retreating one ``step_size`` from the pose.

        Retreats along the negation of the last commanded motion direction (or
        simply holds if no motion has been commanded yet), mirroring the scripted
        descent's one-step contact back-off.

        Args:
            position: Current TCP position ``(3,)`` (m).
            quaternion: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.

        Returns:
            A ``(1, 7)`` retreat pose-target array.
        """
        out = np.empty((1, 7), dtype=np.float64)
        if self._last_dir is not None:
            out[0, :3] = position - self._last_dir * self.step_size
        else:
            out[0, :3] = position
        out[0, 3:] = quaternion
        return out

    def _clamp_to_cap(self, targets: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp each target to lie within ``travel_cap`` of the fixed anchor.

        Args:
            targets: The ``(n, 7)`` integrated pose targets.

        Returns:
            A ``(clamped_targets, capped)`` pair; ``capped`` is True when any target
            exceeded the cap and was projected back onto the cap sphere.
        """
        out = np.asarray(targets, dtype=np.float64).copy()
        capped = False
        for i in range(len(out)):
            d = out[i, :3] - self.anchor_position
            dist = float(np.linalg.norm(d))
            if dist > self.travel_cap:
                capped = True
                if dist > _AXIS_MIN_NORM:
                    out[i, :3] = self.anchor_position + d * (self.travel_cap / dist)
        return out, capped

    def advance(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        force: np.ndarray | None,
        substeps: int = 1,
        logs: list[str] | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Advance one specialist-descent cycle and return the pose targets.

        Order of guards each cycle: while already holding, re-command the measured
        pose; otherwise test the wrench back-off (retreat + hold on a spike), then
        the max-steps cap, then query the specialist and integrate its chunk into
        pose targets, clamping to the travel cap (hold once reached).

        Args:
            position: Current (measured) TCP position ``(3,)`` (m).
            quaternion: Current (measured) TCP orientation ``(4,)`` ``[x, y, z, w]``.
            force: Current wrist force ``(3,)`` (N) for the contact guard, or None.
            substeps: Sub-poses emitted per chunk step (matches DeployACT SUBSTEPS).
                Must be >= 1.
            logs: Optional mutable log list; ``[specialist]`` lines are appended.
            t: Elapsed time (s), for log lines only.

        Returns:
            An ``(m, 7)`` array of absolute pose targets to command this cycle.

        Raises:
            ValueError: If ``substeps`` < 1 or the provider returns a malformed chunk.
        """
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {pos.shape}")
        quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
        force_vec = None if force is None else np.asarray(force, dtype=np.float64).reshape(-1)

        self.last_force_delta = self._force_delta(force_vec)

        if self.phase == GuardedPhase.HOLD:
            return self._hold_target(pos, quat)

        # Wrench back-off guard (force safety preserved from the scripted descent).
        if (
            self.baseline_force is not None
            and self.last_force_delta > self.contact_force_threshold
        ):
            self.contacts += 1
            self.phase = GuardedPhase.HOLD
            if logs is not None:
                logs.append(
                    f"[specialist] phase -> HOLD at t={t:.1f}s: contact "
                    f"(|F-baseline|={self.last_force_delta:.2f}N > "
                    f"{self.contact_force_threshold:.1f}N), backing off one step; "
                    f"steps={self.steps} travel={self._travel * 1e3:.1f}mm"
                )
            return self._backoff_target(pos, quat)

        # Max-steps safety bound.
        if self.max_steps > 0 and self.steps >= self.max_steps:
            self.phase = GuardedPhase.HOLD
            if logs is not None:
                logs.append(
                    f"[specialist] phase -> HOLD at t={t:.1f}s: max_steps "
                    f"{self.max_steps} reached; travel={self._travel * 1e3:.1f}mm"
                )
            return self._hold_target(pos, quat)

        chunk = self.provider.predict_chunk(pos, quat)
        if chunk is None:
            # No observation yet: hold at the current measured pose this cycle.
            return self._hold_target(pos, quat)
        chunk_arr = np.asarray(chunk, dtype=np.float64)
        if chunk_arr.ndim != 2 or chunk_arr.shape[1] != 6:
            raise ValueError(
                f"specialist chunk must have shape (n, 6), got {chunk_arr.shape}"
            )

        exec_chunk = chunk_arr[: self.exec_steps]
        fine = pose_integration.expand_twists(exec_chunk, substeps)
        dt_fine = self.dt_frame / substeps
        targets = pose_integration.integrate_twist_chunk(pos, quat, fine, dt_fine)
        targets, capped = self._clamp_to_cap(targets)

        self.steps += len(exec_chunk)
        self._last_position = targets[-1, :3].copy()
        self._travel = float(np.linalg.norm(self._last_position - self.anchor_position))
        step_vec = targets[-1, :3] - pos
        step_norm = float(np.linalg.norm(step_vec))
        if step_norm >= _AXIS_MIN_NORM:
            self._last_dir = step_vec / step_norm

        if capped:
            self.phase = GuardedPhase.HOLD
            if logs is not None:
                logs.append(
                    f"[specialist] phase -> HOLD at t={t:.1f}s: travel cap "
                    f"{self.travel_cap * 1e3:.0f}mm reached; steps={self.steps}"
                )
        if logs is not None:
            lin = float(np.linalg.norm(exec_chunk[0, :3])) if len(exec_chunk) else 0.0
            logs.append(
                f"[specialist] t={t:.1f}s step={self.steps} "
                f"travel={self._travel * 1e3:.1f}mm |v0|={lin:.4f}m/s "
                f"|F-baseline|={self.last_force_delta:.2f}N contacts={self.contacts}"
            )
        return targets


def _lateral_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane orthogonal to ``axis``.

    Builds a stable right-handed basis ``(u, v)`` with ``u, v`` unit, mutually
    orthogonal, and both orthogonal to the (unit) ``axis`` -- the lateral plane the
    spiral search traces in. The world reference vector least aligned with ``axis``
    seeds ``u`` (Gram-Schmidt), then ``v = axis x u``.

    Args:
        axis: Unit approach/descent axis ``(3,)``.

    Returns:
        The ``(u, v)`` pair of ``(3,)`` unit vectors orthogonal to ``axis``.
    """
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - axis * float(np.dot(ref, axis))
    u = u / float(np.linalg.norm(u))
    v = np.cross(axis, u)
    v = v / float(np.linalg.norm(v))
    return u, v


class SearchDescent:
    """Scripted spiral-search insertion primitive with contact and travel guards.

    The model-free counterpart to :class:`SpecialistDescent` for the stall handoff
    (INSERTION_PLAN.md v2 point 5, "cheap scripted spiral/wiggle insurance under
    the push"). A decisive straight push jams on the port rim when the plug is
    laterally/angularly misaligned; this primitive instead traces a bounded
    Archimedean spiral in the plane *orthogonal to the approach axis* while pushing
    down along that axis, so the lateral search finds the hole a straight push
    misses (InsertionNet arXiv:2104.14223; "From Reach to Insert" arXiv:2605.04649).

    Geometry, anchored at the *fixed* stall pose (never re-anchored to the measured
    pose -- that is exactly the stall attractor), with ``axis`` the estimated
    insertion direction and ``(u, v)`` an orthonormal basis of its orthogonal
    plane::

        theta(step)  = omega * spiral_steps       # advances only while in contact
        radius(step) = min(a * theta, radius_max)  # a = radius_max / (2*pi*turns)
        lateral      = radius * (cos(theta) * u + sin(theta) * v)   # _|_ axis
        z_travel     = z_step * steps              # bounded by travel_cap
        position     = anchor + axis * z_travel + lateral

    The spiral grows to ``radius_max`` over ``turns`` turns, then keeps circling at
    the rim (radius clamped) while the push continues. The orientation is held at
    the anchor unless a tiny ``yaw_amp`` dither about the axis is requested. Three
    guards freeze travel, mirroring :class:`GuardedDescent`:

    * **Hard wrench back-off** -- if ``|F - baseline|`` exceeds
      ``contact_force_threshold`` (the hard 12 N), retreat one push step and HOLD
      (do not ram). Moderate contact below this threshold is *expected* during
      search and is tolerated.
    * **Engage gate** -- the lateral spiral only *grows* while in contact
      (``|F - baseline| >= search_engage_force``); in free space the primitive just
      pushes straight down, and once it drops into the hole (contact falls away) it
      stops searching and keeps pushing in.
    * **Travel cap / max steps** -- the total Z push is capped at ``travel_cap`` and
      the step count at ``max_steps``; hitting either holds.

    Duck-types the attributes ``GuardedDescentController`` reads off a descent
    (:attr:`steps`, :attr:`travel`, :attr:`contacts`, :attr:`last_force_delta`,
    :attr:`phase`), so the controller's result assembly is shared unchanged.
    """

    def __init__(
        self,
        axis: np.ndarray,
        anchor_position: np.ndarray,
        anchor_quaternion: np.ndarray,
        baseline_force: np.ndarray | None,
        radius_max: float = DEFAULT_SEARCH_RADIUS,
        turns: float = DEFAULT_SEARCH_TURNS,
        z_step: float = DEFAULT_SEARCH_Z_STEP,
        omega: float = DEFAULT_SEARCH_OMEGA,
        engage_force: float = DEFAULT_SEARCH_ENGAGE_FORCE,
        travel_cap: float = DEFAULT_SEARCH_TRAVEL_CAP,
        max_steps: int = DEFAULT_SEARCH_MAX_STEPS,
        yaw_amp: float = DEFAULT_SEARCH_YAW_AMP,
        contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
    ) -> None:
        """Initialize the spiral search from the handoff pose and estimated axis.

        Args:
            axis: Estimated insertion direction ``(3,)`` (need not be unit; it is
                normalized). The push accumulates along it and the spiral traces
                its orthogonal plane. Must have non-negligible norm.
            anchor_position: TCP position ``(3,)`` at handoff (m); the fixed origin
                the spiral + push accumulate from.
            anchor_quaternion: TCP orientation ``(4,)`` ``[x, y, z, w]`` at handoff;
                held constant (or yaw-dithered about the axis when ``yaw_amp`` != 0).
            baseline_force: Approach-phase mean force ``(3,)`` (N) for the contact
                guard + engage gate, or None to disable both.
            radius_max: Maximum lateral spiral radius (m). Must be positive.
            turns: Archimedean turns to grow the radius from 0 to ``radius_max``.
                Must be positive.
            z_step: Downward push along the axis per cycle (m). Must be positive.
            omega: Spiral angle advanced per in-contact cycle (rad). Must be
                positive.
            engage_force: ``|F - baseline|`` (N) at/above which the spiral grows.
                Must be non-negative.
            travel_cap: Maximum total Z push (m) before holding. Must be positive.
            max_steps: Cap on push steps (0 = unlimited). Must be >= 0.
            yaw_amp: Amplitude (rad) of an optional yaw dither about the axis
                (0 = orientation frozen at the anchor). Must be finite.
            contact_force_threshold: Baseline-relative ``|F - baseline|`` (N) that
                triggers the hard one-step back-off. Must be non-negative.

        Raises:
            ValueError: If any argument is out of range or malformed.
        """
        axis_arr = np.asarray(axis, dtype=np.float64).reshape(-1)
        if axis_arr.shape != (3,):
            raise ValueError(f"axis must have shape (3,), got {axis_arr.shape}")
        axis_norm = float(np.linalg.norm(axis_arr))
        if not math.isfinite(axis_norm) or axis_norm < _AXIS_MIN_NORM:
            raise ValueError(f"axis must have non-negligible norm, got {axis!r}")
        pos = np.asarray(anchor_position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"anchor_position must have shape (3,), got {pos.shape}")
        quat = np.asarray(anchor_quaternion, dtype=np.float64).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"anchor_quaternion must have shape (4,), got {quat.shape}")
        if float(np.linalg.norm(quat)) < _QUAT_MIN_NORM:
            raise ValueError("anchor_quaternion has near-zero norm")
        if not math.isfinite(radius_max) or radius_max <= 0.0:
            raise ValueError(f"radius_max must be a positive float, got {radius_max!r}")
        if not math.isfinite(turns) or turns <= 0.0:
            raise ValueError(f"turns must be a positive float, got {turns!r}")
        if not math.isfinite(z_step) or z_step <= 0.0:
            raise ValueError(f"z_step must be a positive float, got {z_step!r}")
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError(f"omega must be a positive float, got {omega!r}")
        if not math.isfinite(engage_force) or engage_force < 0.0:
            raise ValueError(f"engage_force must be a non-negative float, got {engage_force!r}")
        if not math.isfinite(travel_cap) or travel_cap <= 0.0:
            raise ValueError(f"travel_cap must be a positive float, got {travel_cap!r}")
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}")
        if not math.isfinite(yaw_amp):
            raise ValueError(f"yaw_amp must be finite, got {yaw_amp!r}")
        if not math.isfinite(contact_force_threshold) or contact_force_threshold < 0.0:
            raise ValueError(
                f"contact_force_threshold must be a non-negative float, "
                f"got {contact_force_threshold!r}"
            )

        self.axis = axis_arr / axis_norm
        self.anchor_position = pos
        self.anchor_quaternion = quat
        self.baseline_force = (
            np.asarray(baseline_force, dtype=np.float64).reshape(-1)
            if baseline_force is not None
            else None
        )
        if self.baseline_force is not None and self.baseline_force.shape != (3,):
            raise ValueError(
                f"baseline_force must have shape (3,), got {self.baseline_force.shape}"
            )
        self.radius_max = float(radius_max)
        self.turns = float(turns)
        self.z_step = float(z_step)
        self.search_omega = float(omega)
        self.search_engage_force = float(engage_force)
        self.travel_cap = float(travel_cap)
        self.max_steps = int(max_steps)
        self.search_yaw_amp = float(yaw_amp)
        self.contact_force_threshold = float(contact_force_threshold)

        # Archimedean coefficient r = a * theta; a chosen so r = radius_max at
        # theta = 2*pi*turns.
        self._a = self.radius_max / (2.0 * math.pi * self.turns)
        self._u, self._v = _lateral_basis(self.axis)

        self.steps = 0  # push steps (drives z_travel = z_step * steps).
        self.theta = 0.0  # accumulated spiral angle (grows only while in contact).
        self.phase = GuardedPhase.DESCEND
        self.contacts = 0
        self.last_force_delta = 0.0

    @property
    def travel(self) -> float:
        """Total commanded Z push from the anchor along the axis, in metres."""
        return self.z_step * self.steps

    @property
    def radius(self) -> float:
        """Current lateral spiral radius (m), clamped at ``radius_max``."""
        return min(self._a * self.theta, self.radius_max)

    def _force_delta(self, force: np.ndarray | None) -> float:
        """Return ``|F - baseline|`` (N), or 0.0 when the contact guard is inactive.

        Args:
            force: Current wrist force ``(3,)`` (N), or None.

        Returns:
            The baseline-relative force magnitude, or 0.0 when either the force or
            the baseline is unavailable.
        """
        if force is None or self.baseline_force is None:
            return 0.0
        f = np.asarray(force, dtype=np.float64).reshape(-1)
        if f.shape != (3,):
            raise ValueError(f"force must have shape (3,), got {f.shape}")
        return float(np.linalg.norm(f - self.baseline_force))

    def target(self) -> np.ndarray:
        """Return the absolute pose target for the current spiral + push state.

        Returns:
            The ``(7,)`` pose target ``[x, y, z, qx, qy, qz, qw]``: the anchor plus
            the accumulated axial push plus the lateral spiral offset, with the
            anchor orientation (optionally yaw-dithered about the axis).
        """
        r = self.radius
        lateral = r * (math.cos(self.theta) * self._u + math.sin(self.theta) * self._v)
        pos = self.anchor_position + self.axis * (self.z_step * self.steps) + lateral
        out = np.empty(7, dtype=np.float64)
        out[:3] = pos
        if self.search_yaw_amp != 0.0:
            angle = self.search_yaw_amp * math.sin(self.theta)
            dq = pose_integration.quaternion_from_angular_velocity(self.axis * angle, 1.0)
            quat = pose_integration.quaternion_multiply(dq, self.anchor_quaternion)
            out[3:] = quat / float(np.linalg.norm(quat))
        else:
            out[3:] = self.anchor_quaternion
        return out

    def advance(
        self,
        force: np.ndarray | None,
        substeps: int = 1,
        logs: list[str] | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Advance one spiral-search cycle and return the sub-stepped pose targets.

        Order of guards each cycle: while already holding, re-command the held
        target; otherwise test the hard wrench back-off (retreat one push step +
        HOLD on a spike above ``contact_force_threshold``), then the ``max_steps``
        cap, then the ``travel_cap`` on the Z push, else take one push step (always)
        and advance the spiral angle (only while in contact). The new target is
        linearly interpolated from the previous one over ``substeps`` sub-poses to
        match ``DeployACT``'s command cadence.

        Args:
            force: Current wrist force ``(3,)`` (N) for the contact guard, or None.
            substeps: Sub-poses emitted per cycle (matches DeployACT SUBSTEPS). Must
                be >= 1.
            logs: Optional mutable log list; ``[search]`` lines are appended.
            t: Elapsed time (s), for log lines only.

        Returns:
            An ``(substeps, 7)`` array of absolute pose targets to command.

        Raises:
            ValueError: If ``substeps`` < 1 or ``force`` has the wrong shape.
        """
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        force_vec = None if force is None else np.asarray(force, dtype=np.float64).reshape(-1)
        self.last_force_delta = self._force_delta(force_vec)
        prev_target = self.target()

        if self.phase == GuardedPhase.HOLD:
            return _interpolate_targets(prev_target, prev_target, substeps)

        # Hard wrench back-off (force safety preserved from the scripted descent):
        # moderate contact is tolerated, only a spike above the hard threshold backs
        # off one push step and holds.
        if (
            self.baseline_force is not None
            and self.last_force_delta > self.contact_force_threshold
        ):
            self.steps = max(0, self.steps - 1)
            self.contacts += 1
            self.phase = GuardedPhase.HOLD
            new_target = self.target()
            if logs is not None:
                logs.append(
                    f"[search] phase -> HOLD at t={t:.1f}s: contact "
                    f"(|F-baseline|={self.last_force_delta:.2f}N > "
                    f"{self.contact_force_threshold:.1f}N), backing off one step; "
                    f"steps={self.steps} z_travel={self.travel * 1e3:.1f}mm "
                    f"radius={self.radius * 1e3:.2f}mm"
                )
            return _interpolate_targets(prev_target, new_target, substeps)

        # Max-steps safety bound.
        if self.max_steps > 0 and self.steps >= self.max_steps:
            self.phase = GuardedPhase.HOLD
            if logs is not None:
                logs.append(
                    f"[search] phase -> HOLD at t={t:.1f}s: max_steps "
                    f"{self.max_steps} reached; z_travel={self.travel * 1e3:.1f}mm"
                )
            return _interpolate_targets(prev_target, prev_target, substeps)

        # Travel cap on the total Z push.
        if self.z_step * (self.steps + 1) > self.travel_cap:
            self.phase = GuardedPhase.HOLD
            if logs is not None:
                logs.append(
                    f"[search] phase -> HOLD at t={t:.1f}s: travel cap "
                    f"{self.travel_cap * 1e3:.0f}mm reached; steps={self.steps} "
                    f"radius={self.radius * 1e3:.2f}mm"
                )
            return _interpolate_targets(prev_target, prev_target, substeps)

        # Push one step (always); grow the lateral spiral only while in contact.
        self.steps += 1
        in_contact = (
            self.baseline_force is not None
            and self.last_force_delta >= self.search_engage_force
        )
        if in_contact:
            self.theta += self.search_omega
        new_target = self.target()
        if logs is not None:
            logs.append(
                f"[search] t={t:.1f}s step={self.steps} theta={self.theta:.2f}rad "
                f"radius={self.radius * 1e3:.2f}mm z_travel={self.travel * 1e3:.1f}mm "
                f"|F-baseline|={self.last_force_delta:.2f}N contact={int(in_contact)} "
                f"contacts={self.contacts}"
            )
        return _interpolate_targets(prev_target, new_target, substeps)


@dataclasses.dataclass(frozen=True)
class GuardedDescentConfig:
    """Immutable configuration for the guarded-descent probe.

    Attributes:
        enabled: Whether the probe is active (``AIC_GUARDED=1``). When False the
            controller is never constructed and ``DeployACT`` is byte-identical.
        speed_threshold: Stall speed threshold (m/s).
        stall_window_s: Sustained low-speed window before firing (s).
        min_runtime_s: Startup grace period before any fire is allowed (s).
        axis_window: Recent displacements averaged for the axis estimate.
        axis_min_displacement: Minimum per-cycle displacement recorded (m).
        step_size: Metres advanced along the axis per descent cycle.
        travel_cap: Maximum total commanded travel (m).
        contact_force_threshold: Baseline-relative ``|F|`` back-off threshold (N).
        z_stiffness: Optional reduced Z-axis stiffness passed to
            ``set_pose_target``; None keeps the base stiffness on every axis.
        base_stiffness: The 6-D MODE_POSITION stiffness the descent starts from.
        use_aux_bearing: Use the learned port-bearing provider to aim the descent
            at handoff (``AIC_GUARDED_AUX=1``); falls back to the motion axis when
            the prediction is implausible/inconsistent. Off = motion-axis only.
        aux_min_mag: Minimum plausible predicted offset magnitude (m).
        aux_max_mag: Maximum plausible predicted offset magnitude (m).
        aux_consistency_std: Ceiling on the cross-frame spread (m) of the buffered
            target predictions; a noisier estimate falls back to the motion axis.
        aux_travel_margin: Metres added to ``|offset|`` for the aux travel cap so
            the descent reaches the port then stops (design section 4.3). Used
            only for a 3-D (offset-only) checkpoint.
        aux_fixed_travel: Fixed deep-travel cap (m) used instead of
            ``|offset| + aux_travel_margin`` when the head supplies an explicit
            approach axis (6-D checkpoint), decoupling steering from the
            unreliable predicted depth magnitude.
        aux_buffer: Number of recent approach-frame predictions kept for the
            median target (steadier than a single stall-frame query).
        aux_min_samples: Minimum buffered plausible predictions before the aux
            target is trusted at handoff.
        reaim: Re-query the provider each descent cycle and rate-limit-blend the
            axis toward the fresh target (default off; enable only after the
            static handoff validates).
        reaim_rate: Per-cycle blend fraction toward the re-queried axis.
        specialist_enabled: Whether the LEARNED insertion specialist replaces the
            scripted descent once the stall latches (``AIC_SPECIALIST=1``). When
            False the descent is byte-identical to today's scripted GuardedDescent.
            ``enabled`` is forced True whenever this is set, since the specialist
            still needs the stall-detection machinery to hand off.
        specialist_travel_cap: Maximum total travel (m) from the stall anchor the
            specialist may command before holding (a hard safety limit).
        specialist_max_steps: Cap on the number of integrated chunk steps before
            holding (0 = unlimited); an additional safety bound.
        specialist_exec_steps: Chunk steps integrated into pose targets per descent
            cycle (receding horizon) before re-querying the specialist.
        specialist_dt: Training frame period (s) used to integrate the specialist's
            TCP-velocity chunk into pose targets (matches ``DeployACT.DT_FRAME``).
        search_enabled: Whether the scripted spiral-search primitive replaces the
            straight scripted descent once the stall latches (``AIC_SEARCH=1``).
            When False the descent is byte-identical to today. ``enabled`` is forced
            True whenever this is set, since the search still needs the
            stall-detection machinery to hand off. Takes PRECEDENCE over the learned
            specialist when both are set (search is the self-contained, model-free
            standalone primitive).
        search_radius_max: Maximum lateral spiral radius (m) (~port half-width).
        search_turns: Archimedean turns to grow the radius from 0 to the max.
        search_z_step: Downward push along the axis per cycle (m).
        search_omega: Spiral angle advanced per in-contact cycle (rad).
        search_engage_force: ``|F - baseline|`` (N) at/above which the spiral grows
            (moderate search contact is tolerated; the hard back-off stays at
            ``contact_force_threshold``).
        search_travel_cap: Maximum total Z push (m) -- the search depth -- before
            holding.
        search_max_steps: Cap on push steps before holding (0 = unlimited).
        search_yaw_amp: Amplitude (rad) of an optional yaw dither about the axis
            (0 = orientation frozen at the anchor).
        search_vertical_axis: Force the spiral search to descend along the world
            vertical ``(0, 0, -1)`` with its lateral spiral in the world XY plane
            (``AIC_SEARCH_VERTICAL=1``), applying the vertical-port prior instead
            of following the (possibly diagonal) motion-axis estimate. Scopes to
            the scripted :class:`SearchDescent` ONLY; the aux-bearing
            :class:`GuardedDescent` and learned :class:`SpecialistDescent` paths
            are untouched. Orientation stays frozen at the anchor -- only the
            translation axis changes. Default False (byte-identical when off).
    """

    enabled: bool = False
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD
    stall_window_s: float = DEFAULT_STALL_WINDOW_S
    min_runtime_s: float = DEFAULT_MIN_RUNTIME_S
    axis_window: int = DEFAULT_AXIS_WINDOW
    axis_min_displacement: float = DEFAULT_AXIS_MIN_DISPLACEMENT
    step_size: float = DEFAULT_STEP_SIZE
    travel_cap: float = DEFAULT_TRAVEL_CAP
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD
    z_stiffness: float | None = None
    base_stiffness: tuple[float, ...] = DEFAULT_BASE_STIFFNESS
    use_aux_bearing: bool = False
    aux_min_mag: float = DEFAULT_AUX_MIN_MAG
    aux_max_mag: float = DEFAULT_AUX_MAX_MAG
    aux_consistency_std: float = DEFAULT_AUX_CONSISTENCY_STD
    aux_travel_margin: float = DEFAULT_AUX_TRAVEL_MARGIN
    aux_fixed_travel: float = DEFAULT_AUX_FIXED_TRAVEL
    aux_buffer: int = DEFAULT_AUX_BUFFER
    aux_min_samples: int = DEFAULT_AUX_MIN_SAMPLES
    reaim: bool = False
    reaim_rate: float = DEFAULT_REAIM_RATE
    specialist_enabled: bool = False
    specialist_travel_cap: float = DEFAULT_SPECIALIST_TRAVEL_CAP
    specialist_max_steps: int = DEFAULT_SPECIALIST_MAX_STEPS
    specialist_exec_steps: int = DEFAULT_SPECIALIST_EXEC_STEPS
    specialist_dt: float = DEFAULT_SPECIALIST_DT
    search_enabled: bool = False
    search_radius_max: float = DEFAULT_SEARCH_RADIUS
    search_turns: float = DEFAULT_SEARCH_TURNS
    search_z_step: float = DEFAULT_SEARCH_Z_STEP
    search_omega: float = DEFAULT_SEARCH_OMEGA
    search_engage_force: float = DEFAULT_SEARCH_ENGAGE_FORCE
    search_travel_cap: float = DEFAULT_SEARCH_TRAVEL_CAP
    search_max_steps: int = DEFAULT_SEARCH_MAX_STEPS
    search_yaw_amp: float = DEFAULT_SEARCH_YAW_AMP
    search_vertical_axis: bool = False

    def stiffness(self) -> list[float] | None:
        """Return the descent stiffness list, or None to keep the caller default.

        Returns:
            A 6-element stiffness list with the Z (index 2) entry replaced by
            ``z_stiffness`` when it is set; otherwise None so ``DeployACT`` uses
            its own ``POSE_STIFFNESS`` unchanged.
        """
        if self.z_stiffness is None:
            return None
        gains = list(self.base_stiffness)
        gains[2] = float(self.z_stiffness)
        return gains

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "GuardedDescentConfig":
        """Build a config from an environment mapping (``AIC_GUARDED*``).

        Recognized variables (all optional except the enable flag):
        ``AIC_GUARDED`` (``"1"`` enables), ``AIC_GUARDED_SPEED``,
        ``AIC_GUARDED_STALL_WINDOW``, ``AIC_GUARDED_MIN_RUNTIME``,
        ``AIC_GUARDED_AXIS_N``, ``AIC_GUARDED_AXIS_MIN_DISP``,
        ``AIC_GUARDED_STEP``, ``AIC_GUARDED_TRAVEL_CAP``, ``AIC_GUARDED_FORCE``,
        ``AIC_GUARDED_ZSTIFFNESS``. Learned-bearing (``port_aux``) handoff:
        ``AIC_GUARDED_AUX`` (``"1"`` uses the aux bearing), ``AIC_GUARDED_AUX_MINMAG``,
        ``AIC_GUARDED_AUX_MAXMAG``, ``AIC_GUARDED_AUX_STD``,
        ``AIC_GUARDED_AUX_MARGIN``, ``AIC_GUARDED_AUX_TRAVEL`` (fixed deep-travel
        cap for the 6-D explicit-axis descent), ``AIC_GUARDED_REAIM``
        (``"1"`` re-aims). Learned insertion specialist (INSERTION_PLAN.md change
        #2): ``AIC_SPECIALIST`` (``"1"`` replaces the scripted descent with the
        learned specialist AND forces ``enabled`` so the stall machinery runs),
        ``AIC_SPECIALIST_TRAVEL_CAP``, ``AIC_SPECIALIST_MAX_STEPS``,
        ``AIC_SPECIALIST_EXEC_STEPS``, ``AIC_SPECIALIST_DT`` (the specialist
        checkpoint path itself is read by ``DeployACT`` via ``AIC_SPECIALIST_CKPT``).
        Scripted spiral-search primitive (INSERTION_PLAN.md v2 point 5):
        ``AIC_SEARCH`` (``"1"`` replaces the scripted descent with the model-free
        spiral search AND forces ``enabled`` so the stall machinery runs; takes
        PRECEDENCE over ``AIC_SPECIALIST`` when both are set), ``AIC_SEARCH_RADIUS``,
        ``AIC_SEARCH_TURNS``, ``AIC_SEARCH_Z_STEP``, ``AIC_SEARCH_OMEGA``,
        ``AIC_SEARCH_ENGAGE_FORCE``, ``AIC_SEARCH_TRAVEL_CAP``,
        ``AIC_SEARCH_MAX_STEPS``, ``AIC_SEARCH_YAW_AMP``, ``AIC_SEARCH_VERTICAL``
        (``"1"`` forces the search to descend along world-vertical ``(0, 0, -1)``
        with its spiral in the world XY plane; default off = motion-axis descent).

        Args:
            env: Environment mapping (e.g. ``os.environ``).

        Returns:
            The parsed :class:`GuardedDescentConfig`. When none of ``AIC_GUARDED``,
            ``AIC_SPECIALIST``, or ``AIC_SEARCH`` is exactly ``"1"`` the returned
            config has ``enabled=False`` and default thresholds.

        Raises:
            ValueError: If a provided override cannot be parsed as its numeric
                type.
        """
        specialist_enabled = env.get("AIC_SPECIALIST", "0").strip() == "1"
        search_enabled = env.get("AIC_SEARCH", "0").strip() == "1"
        # The specialist / spiral search reuse the stall-detection/handoff
        # machinery, so enabling either also enables the guarded controller even
        # when AIC_GUARDED is unset.
        enabled = (
            env.get("AIC_GUARDED", "0").strip() == "1"
            or specialist_enabled
            or search_enabled
        )

        def _f(key: str, default: float) -> float:
            raw = env.get(key)
            if raw is None or raw.strip() == "":
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be a float, got {raw!r}") from exc

        def _i(key: str, default: int) -> int:
            raw = env.get(key)
            if raw is None or raw.strip() == "":
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be an int, got {raw!r}") from exc

        z_raw = env.get("AIC_GUARDED_ZSTIFFNESS")
        z_stiffness = None if (z_raw is None or z_raw.strip() == "") else _f(
            "AIC_GUARDED_ZSTIFFNESS", 0.0
        )
        return cls(
            enabled=enabled,
            speed_threshold=_f("AIC_GUARDED_SPEED", DEFAULT_SPEED_THRESHOLD),
            stall_window_s=_f("AIC_GUARDED_STALL_WINDOW", DEFAULT_STALL_WINDOW_S),
            min_runtime_s=_f("AIC_GUARDED_MIN_RUNTIME", DEFAULT_MIN_RUNTIME_S),
            axis_window=_i("AIC_GUARDED_AXIS_N", DEFAULT_AXIS_WINDOW),
            axis_min_displacement=_f("AIC_GUARDED_AXIS_MIN_DISP", DEFAULT_AXIS_MIN_DISPLACEMENT),
            step_size=_f("AIC_GUARDED_STEP", DEFAULT_STEP_SIZE),
            travel_cap=_f("AIC_GUARDED_TRAVEL_CAP", DEFAULT_TRAVEL_CAP),
            contact_force_threshold=_f("AIC_GUARDED_FORCE", DEFAULT_CONTACT_FORCE_THRESHOLD),
            z_stiffness=z_stiffness,
            use_aux_bearing=env.get("AIC_GUARDED_AUX", "0").strip() == "1",
            aux_min_mag=_f("AIC_GUARDED_AUX_MINMAG", DEFAULT_AUX_MIN_MAG),
            aux_max_mag=_f("AIC_GUARDED_AUX_MAXMAG", DEFAULT_AUX_MAX_MAG),
            aux_consistency_std=_f("AIC_GUARDED_AUX_STD", DEFAULT_AUX_CONSISTENCY_STD),
            aux_travel_margin=_f("AIC_GUARDED_AUX_MARGIN", DEFAULT_AUX_TRAVEL_MARGIN),
            aux_fixed_travel=_f("AIC_GUARDED_AUX_TRAVEL", DEFAULT_AUX_FIXED_TRAVEL),
            reaim=env.get("AIC_GUARDED_REAIM", "0").strip() == "1",
            specialist_enabled=specialist_enabled,
            specialist_travel_cap=_f(
                "AIC_SPECIALIST_TRAVEL_CAP", DEFAULT_SPECIALIST_TRAVEL_CAP
            ),
            specialist_max_steps=_i("AIC_SPECIALIST_MAX_STEPS", DEFAULT_SPECIALIST_MAX_STEPS),
            specialist_exec_steps=_i(
                "AIC_SPECIALIST_EXEC_STEPS", DEFAULT_SPECIALIST_EXEC_STEPS
            ),
            specialist_dt=_f("AIC_SPECIALIST_DT", DEFAULT_SPECIALIST_DT),
            search_enabled=search_enabled,
            search_radius_max=_f("AIC_SEARCH_RADIUS", DEFAULT_SEARCH_RADIUS),
            search_turns=_f("AIC_SEARCH_TURNS", DEFAULT_SEARCH_TURNS),
            search_z_step=_f("AIC_SEARCH_Z_STEP", DEFAULT_SEARCH_Z_STEP),
            search_omega=_f("AIC_SEARCH_OMEGA", DEFAULT_SEARCH_OMEGA),
            search_engage_force=_f("AIC_SEARCH_ENGAGE_FORCE", DEFAULT_SEARCH_ENGAGE_FORCE),
            search_travel_cap=_f("AIC_SEARCH_TRAVEL_CAP", DEFAULT_SEARCH_TRAVEL_CAP),
            search_max_steps=_i("AIC_SEARCH_MAX_STEPS", DEFAULT_SEARCH_MAX_STEPS),
            search_yaw_amp=_f("AIC_SEARCH_YAW_AMP", DEFAULT_SEARCH_YAW_AMP),
            search_vertical_axis=env.get("AIC_SEARCH_VERTICAL", "0").strip() == "1",
        )


@dataclasses.dataclass(frozen=True)
class GuardedStep:
    """One cycle's guarded-descent decision returned to ``DeployACT``.

    Attributes:
        targets: The ``(n, 7)`` pose targets to command this cycle, or None to
            keep running the learned policy path unchanged.
        stiffness: The 6-D stiffness list for ``set_pose_target``, or None to use
            the caller's default.
        log_lines: Human-readable log lines for state transitions this cycle.
        phase: The controller phase after this cycle.
        steps: Descent steps taken so far.
        travel: Total commanded travel this descent (m).
        force_delta: Latest baseline-relative force magnitude (N).
        contacts: Number of contact back-offs so far.
        triggered_this_cycle: True on the exact cycle the stall handoff occurred.
        bearing_source: Which bearing aimed the descent -- ``"aux"`` (learned
            port bearing), ``"motion-axis"`` (ApproachAxisEstimator fallback), or
            ``""`` while still approaching.
    """

    targets: np.ndarray | None
    stiffness: list[float] | None
    log_lines: tuple[str, ...]
    phase: str
    steps: int
    travel: float
    force_delta: float
    contacts: int
    triggered_this_cycle: bool
    bearing_source: str = ""


class GuardedDescentController:
    """Wires stall detection, axis estimation, and guarded descent for DeployACT.

    This is the single per-cycle seam ``DeployACT`` calls (see :meth:`cycle`),
    analogous to ``select_exec_twists`` in ``chunk_ensemble.py``. Each cycle it is
    fed the elapsed time, TCP pose, and wrist force; while approaching it returns
    ``targets=None`` (learned path runs unchanged) and accumulates the wrench
    baseline, approach axis, and stall statistics; when the stall fires it builds
    a :class:`GuardedDescent` anchored at the current pose and thereafter returns
    the scripted descent targets.
    """

    def __init__(
        self,
        config: GuardedDescentConfig,
        bearing_provider: PortBearingProvider | None = None,
        specialist_provider: SpecialistActionProvider | None = None,
    ) -> None:
        """Initialize the controller from a config.

        Args:
            config: The parsed :class:`GuardedDescentConfig`. ``enabled`` is not
                enforced here (``DeployACT`` decides whether to construct the
                controller at all), but callers normally construct it only when
                ``config.enabled`` is True.
            bearing_provider: Optional learned port-bearing source (the aux head).
                When None -- or when ``config.use_aux_bearing`` is False -- the
                controller is byte-identical to the motion-axis-only behavior.
            specialist_provider: Optional learned insertion-specialist action
                source. When None -- or when ``config.specialist_enabled`` is
                False -- the descent is the scripted :class:`GuardedDescent`,
                byte-identical to today. When both are set, the stall handoff
                engages a :class:`SpecialistDescent` instead.
        """
        self.config = config
        self.bearing_provider = bearing_provider
        self.specialist_provider = specialist_provider
        self.stall = StallDetector(
            speed_threshold=config.speed_threshold,
            stall_window_s=config.stall_window_s,
            min_runtime_s=config.min_runtime_s,
        )
        self.axis_estimator = ApproachAxisEstimator(
            window=config.axis_window,
            min_displacement=config.axis_min_displacement,
        )
        self.descent: GuardedDescent | SpecialistDescent | SearchDescent | None = None
        self.phase = GuardedPhase.APPROACH
        self._last_position: np.ndarray | None = None
        self._last_time: float | None = None
        self._baseline_sum = np.zeros(3, dtype=np.float64)
        self._baseline_count = 0
        self._axis_unavailable_logged = False
        self._bearing_source = ""
        # Rolling buffer of recent plausible aux target predictions (base_link),
        # median-reduced at handoff for a steadier aim than a single stall-frame
        # query. Only populated when a provider is wired and use_aux_bearing is on.
        self._aux_targets: collections.deque[np.ndarray] = collections.deque(
            maxlen=config.aux_buffer
        )
        # Paired explicit approach axes (unit, base_link) when the head is 6-D,
        # else None per frame; kept in lockstep with ``_aux_targets`` so the
        # handoff can steer by the aux axis channel and cap travel at a fixed
        # depth, decoupled from the unreliable predicted |offset| magnitude.
        self._aux_axes: collections.deque[np.ndarray | None] = collections.deque(
            maxlen=config.aux_buffer
        )

    @property
    def _aux_active(self) -> bool:
        """Whether the learned port bearing should be consulted at handoff."""
        return self.bearing_provider is not None and self.config.use_aux_bearing

    @property
    def _specialist_active(self) -> bool:
        """Whether the learned insertion specialist drives the descent at handoff."""
        return self.specialist_provider is not None and self.config.specialist_enabled

    @property
    def _search_active(self) -> bool:
        """Whether the scripted spiral search drives the descent at handoff.

        The spiral search is model-free (no provider needed), so this is gated on
        the config flag alone. It takes precedence over the learned specialist when
        both are enabled.
        """
        return self.config.search_enabled

    @property
    def active(self) -> bool:
        """Whether the guarded descent has been engaged (stall fired)."""
        return self.descent is not None

    def _baseline(self) -> np.ndarray | None:
        """Return the running mean approach-phase force, or None if unobserved."""
        if self._baseline_count == 0:
            return None
        return self._baseline_sum / self._baseline_count

    def _speed_and_displacement(
        self, t: float, position: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return the TCP speed and displacement since the previous cycle.

        Args:
            t: Elapsed time (s).
            position: Current TCP position ``(3,)`` (m).

        Returns:
            A ``(speed, displacement)`` pair; both are zero on the first cycle or
            when the timestamp did not advance.
        """
        if self._last_position is None or self._last_time is None:
            return 0.0, np.zeros(3, dtype=np.float64)
        displacement = position - self._last_position
        dt = t - self._last_time
        speed = float(np.linalg.norm(displacement) / dt) if dt > 0.0 else 0.0
        return speed, displacement

    def cycle(
        self,
        t: float,
        position: np.ndarray,
        quaternion: np.ndarray,
        force: np.ndarray | None,
        substeps: int = 1,
    ) -> GuardedStep:
        """Process one control cycle and return the guarded-descent decision.

        Args:
            t: Elapsed time since the policy started (s).
            position: Current TCP position ``(3,)`` (m).
            quaternion: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.
            force: Current wrist force ``(3,)`` (N), or None if unavailable.
            substeps: Number of interpolated sub-poses to emit per descent step,
                for a smoother command stream (matches ``DeployACT``'s SUBSTEPS).
                Must be >= 1.

        Returns:
            A :class:`GuardedStep`. ``targets`` is None while approaching (run the
            learned path) and an ``(substeps, 7)`` array once descending/holding.

        Raises:
            ValueError: If ``substeps`` < 1 or a vector has the wrong shape.
        """
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {pos.shape}")
        quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
        force_vec = None if force is None else np.asarray(force, dtype=np.float64).reshape(-1)

        speed, displacement = self._speed_and_displacement(t, pos)
        self._last_position = pos
        self._last_time = float(t)

        logs: list[str] = []
        triggered_this_cycle = False

        if not self.active:
            # Approach phase: accumulate baseline/axis and test for a stall.
            if force_vec is not None and force_vec.shape == (3,):
                self._baseline_sum += force_vec
                self._baseline_count += 1
            self.axis_estimator.observe(displacement)
            # Buffer learned port-bearing predictions during the approach so the
            # handoff can median a steady target (design section 4.3).
            if self._aux_active:
                self._buffer_aux_prediction(pos, quat)
            if self.stall.update(t, speed):
                baseline = self._baseline()
                if not self._engage_descent(t, pos, quat, baseline, logs):
                    return self._result(None, tuple(logs), False)
                triggered_this_cycle = True
            else:
                return self._result(None, tuple(logs), False)
        elif self._aux_active and self.config.reaim and self.descent is not None:
            # Re-aim an in-progress descent toward a freshly-queried target.
            self._maybe_reaim(pos, quat)

        # Descending or holding: advance the state machine and emit targets.
        assert self.descent is not None
        if isinstance(self.descent, SpecialistDescent):
            # Learned specialist: it emits the (already sub-stepped) integrated
            # pose targets directly and manages its own HOLD transitions/logs.
            targets = self.descent.advance(pos, quat, force_vec, substeps, logs, t)
            if self.descent.phase == GuardedPhase.HOLD:
                self.phase = GuardedPhase.HOLD
            return self._result(targets, tuple(logs), triggered_this_cycle)
        if isinstance(self.descent, SearchDescent):
            # Scripted spiral search: it emits the (already sub-stepped) spiral +
            # push pose targets directly and manages its own HOLD transitions/logs.
            targets = self.descent.advance(force_vec, substeps, logs, t)
            if self.descent.phase == GuardedPhase.HOLD:
                self.phase = GuardedPhase.HOLD
            return self._result(targets, tuple(logs), triggered_this_cycle)
        prev_target = self.descent.target()
        prev_phase = self.descent.phase
        new_target = self.descent.advance(force_vec)
        if self.descent.phase != prev_phase and self.descent.phase == GuardedPhase.HOLD:
            self.phase = GuardedPhase.HOLD
            reason = (
                f"contact (|F-baseline|={self.descent.last_force_delta:.2f}N > "
                f"{self.config.contact_force_threshold:.1f}N), backed off one step"
                if self.descent.contacts > 0
                else f"travel cap {self.config.travel_cap*1e3:.0f}mm reached"
            )
            logs.append(
                f"[guarded] phase -> HOLD at t={t:.1f}s: {reason}; "
                f"steps={self.descent.steps} travel={self.descent.travel*1e3:.1f}mm"
            )
        targets = _interpolate_targets(prev_target, new_target, substeps)
        return self._result(targets, tuple(logs), triggered_this_cycle)

    def _buffer_aux_prediction(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Query the provider this cycle and buffer a plausible target (+ axis).

        Buffers the predicted base_link target and, in lockstep, the paired unit
        approach axis when the head supplies one (6-D checkpoint) or None (3-D
        checkpoint) so the handoff can steer by the explicit axis channel.

        Args:
            pos: Current TCP position ``(3,)`` (m).
            quat: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.
        """
        assert self.bearing_provider is not None
        target, magnitude, ok, axis = _split_reading(
            self.bearing_provider.predict(pos, quat)
        )
        if not ok or target is None:
            return
        t = np.asarray(target, dtype=np.float64).reshape(-1)
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            return
        if self.config.aux_min_mag <= magnitude <= self.config.aux_max_mag:
            self._aux_targets.append(t)
            self._aux_axes.append(_unit_axis(axis))

    def _resolve_aux_target(
        self, pos: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """Reduce the buffered predictions to a trusted target + axis, or explain.

        Applies the plausibility + consistency gate (design section 4.4): enough
        samples, in-range magnitude, low cross-frame spread, and pointing along
        the established approach (not away, which flags a wrong-port lock). When
        every buffered prediction carried an explicit approach axis (6-D
        checkpoint), the mean of those unit axes is returned as ``axis_base`` so
        the handoff steers by the aux axis rather than the noisy offset direction;
        otherwise ``axis_base`` is None (3-D checkpoint, offset-derived axis).

        Args:
            pos: Anchor TCP position ``(3,)`` at handoff (m).

        Returns:
            ``(target_base, axis_base, "")`` when the aux target is trusted (with
            ``axis_base`` a unit vector for a 6-D head else None), else
            ``(None, None, reason)`` naming the failed check for the fallback log.
        """
        n = len(self._aux_targets)
        if n < self.config.aux_min_samples:
            return None, None, f"only {n} aux samples (<{self.config.aux_min_samples})"
        stack = np.stack(self._aux_targets)
        median = np.median(stack, axis=0)
        spread = float(np.linalg.norm(stack.std(axis=0)))
        d = median - pos
        mag = float(np.linalg.norm(d))
        if not (self.config.aux_min_mag <= mag <= self.config.aux_max_mag):
            return None, None, f"|offset|={mag * 1e3:.0f}mm out of range"
        if spread > self.config.aux_consistency_std:
            return None, None, (
                f"cross-frame std {spread * 1e3:.1f}mm>"
                f"{self.config.aux_consistency_std * 1e3:.0f}mm"
            )
        motion_axis = self.axis_estimator.estimate()
        if motion_axis is not None and float(np.dot(d / mag, motion_axis)) <= 0.0:
            return None, None, "aux points away from the approach"
        return median, self._reduce_aux_axis(), ""

    def _reduce_aux_axis(self) -> np.ndarray | None:
        """Return the mean unit approach axis over the buffer, or None.

        Only returns an axis when every buffered prediction carried an explicit
        axis (a 6-D checkpoint); a single missing axis (3-D checkpoint, or a
        degenerate prediction) yields None so the caller derives the axis from the
        offset direction instead.

        Returns:
            The ``(3,)`` mean unit approach axis in base_link, or None.
        """
        axes = list(self._aux_axes)
        if not axes or any(a is None for a in axes):
            return None
        mean = np.mean(np.stack(axes), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < _AXIS_MIN_NORM:
            return None
        return mean / norm

    def _engage_descent(
        self,
        t: float,
        pos: np.ndarray,
        quat: np.ndarray,
        baseline: np.ndarray | None,
        logs: list[str],
    ) -> bool:
        """Build the :class:`GuardedDescent` at handoff, aux-first with fallback.

        Prefers the learned port bearing. For a 6-D checkpoint the head's EXPLICIT
        approach axis steers the descent (sign-checked against the offset) and
        travel is capped at the FIXED ``aux_fixed_travel`` depth, decoupling
        steering from the unreliable predicted ``|offset|`` magnitude; for a 3-D
        checkpoint the axis points along the offset and travel is capped at
        ``|offset| + margin``. Falls back to the :class:`ApproachAxisEstimator`
        motion axis when the aux prediction is unavailable/implausible/inconsistent.
        Returns False (staying on the learned path) only when no bearing is
        available.

        Args:
            t: Elapsed time (s).
            pos: Anchor TCP position ``(3,)`` (m).
            quat: Anchor TCP orientation ``(4,)`` ``[x, y, z, w]``.
            baseline: Approach-phase mean force ``(3,)`` (N), or None.
            logs: Mutable log-line list appended to in place.

        Returns:
            True when a guarded descent was engaged; False to keep the learned
            path (no usable bearing).
        """
        if self._search_active:
            # Scripted spiral search replaces the straight descent (takes precedence
            # over the learned specialist when both are enabled).
            return self._engage_search(t, pos, quat, baseline, logs)
        if self._specialist_active:
            # Learned insertion specialist replaces the scripted descent entirely.
            return self._engage_specialist(t, pos, quat, baseline, logs)
        aux_target: np.ndarray | None = None
        aux_axis: np.ndarray | None = None
        aux_reason = ""
        if self._aux_active:
            aux_target, aux_axis, aux_reason = self._resolve_aux_target(pos)

        axis_mode = ""
        if aux_target is not None:
            d = aux_target - pos
            mag = float(np.linalg.norm(d))
            if aux_axis is not None:
                # 6-D checkpoint: steer by the head's EXPLICIT approach axis and
                # travel a FIXED deep distance, decoupling the descent from the
                # unreliable predicted |offset| magnitude. Sign-check the axis
                # against the offset (flip if it points away from the target).
                axis = -aux_axis if float(np.dot(aux_axis, d)) < 0.0 else aux_axis
                travel_cap = min(self.config.travel_cap, self.config.aux_fixed_travel)
                axis_mode = "explicit6D"
            else:
                # 3-D checkpoint: aim along the offset direction and cap travel at
                # |offset| + margin (byte-identical to the pre-6-D behavior).
                axis = d / mag
                travel_cap = min(
                    self.config.travel_cap, mag + self.config.aux_travel_margin
                )
                axis_mode = "offset3D"
            self._bearing_source = "aux"
        else:
            axis = self.axis_estimator.estimate()
            if axis is None:
                if not self._axis_unavailable_logged:
                    self._axis_unavailable_logged = True
                    logs.append(
                        f"[guarded] stall detected at t={t:.1f}s but approach axis "
                        f"could not be estimated ({len(self.axis_estimator)} samples); "
                        f"staying on the learned path"
                    )
                return False
            travel_cap = self.config.travel_cap
            self._bearing_source = "motion-axis"

        self.descent = GuardedDescent(
            axis=axis,
            anchor_position=pos,
            anchor_quaternion=quat,
            baseline_force=baseline,
            step_size=self.config.step_size,
            travel_cap=travel_cap,
            contact_force_threshold=self.config.contact_force_threshold,
        )
        self.phase = GuardedPhase.DESCEND
        baseline_norm = 0.0 if baseline is None else float(np.linalg.norm(baseline))
        if self._bearing_source == "aux":
            bearing_desc = (
                f"bearing=aux[{axis_mode}] target=({aux_target[0]:+.3f},"
                f"{aux_target[1]:+.3f},{aux_target[2]:+.3f}) "
                f"|offset|={float(np.linalg.norm(d)) * 1e3:.1f}mm"
            )
        else:
            reason = f" (aux fallback: {aux_reason})" if aux_reason else ""
            bearing_desc = f"bearing=motion-axis{reason}"
        logs.append(
            f"[guarded] HANDOFF at t={t:.1f}s: stall detected, engaging guarded "
            f"descent. {bearing_desc} "
            f"axis=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}) "
            f"anchor=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
            f"baseline|F|={baseline_norm:.2f}N step={self.config.step_size*1e3:.1f}mm "
            f"cap={travel_cap*1e3:.0f}mm force_thr="
            f"{self.config.contact_force_threshold:.1f}N"
        )
        return True

    def _engage_specialist(
        self,
        t: float,
        pos: np.ndarray,
        quat: np.ndarray,
        baseline: np.ndarray | None,
        logs: list[str],
    ) -> bool:
        """Build the :class:`SpecialistDescent` at the stall handoff.

        Anchors the learned specialist at the current (stall) pose and wires the
        approach-phase wrench baseline into its contact guard. Unlike the scripted
        descent this never falls back for lack of a bearing -- the specialist reads
        the live observation each cycle -- so it always engages once the stall
        latches.

        Args:
            t: Elapsed time (s).
            pos: Anchor TCP position ``(3,)`` (m).
            quat: Anchor TCP orientation ``(4,)`` ``[x, y, z, w]``.
            baseline: Approach-phase mean force ``(3,)`` (N), or None.
            logs: Mutable log-line list appended to in place.

        Returns:
            Always True (the specialist descent is engaged).
        """
        assert self.specialist_provider is not None
        self.descent = SpecialistDescent(
            provider=self.specialist_provider,
            anchor_position=pos,
            anchor_quaternion=quat,
            baseline_force=baseline,
            travel_cap=self.config.specialist_travel_cap,
            max_steps=self.config.specialist_max_steps,
            exec_steps=self.config.specialist_exec_steps,
            dt_frame=self.config.specialist_dt,
            step_size=self.config.step_size,
            contact_force_threshold=self.config.contact_force_threshold,
        )
        self.phase = GuardedPhase.DESCEND
        self._bearing_source = "specialist"
        baseline_norm = 0.0 if baseline is None else float(np.linalg.norm(baseline))
        max_steps_desc = (
            "unlimited" if self.config.specialist_max_steps == 0
            else str(self.config.specialist_max_steps)
        )
        logs.append(
            f"[specialist] HANDOFF at t={t:.1f}s: stall detected, engaging learned "
            f"insertion specialist. "
            f"anchor=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
            f"baseline|F|={baseline_norm:.2f}N exec_steps={self.config.specialist_exec_steps} "
            f"dt={self.config.specialist_dt:g}s travel_cap="
            f"{self.config.specialist_travel_cap*1e3:.0f}mm max_steps={max_steps_desc} "
            f"force_thr={self.config.contact_force_threshold:.1f}N"
        )
        return True

    def _engage_search(
        self,
        t: float,
        pos: np.ndarray,
        quat: np.ndarray,
        baseline: np.ndarray | None,
        logs: list[str],
    ) -> bool:
        """Build the :class:`SearchDescent` at the stall handoff.

        Anchors the model-free spiral search at the current (stall) pose, using the
        :class:`ApproachAxisEstimator` motion axis as the push direction (and the
        seed for the orthogonal spiral plane) and wiring the approach-phase wrench
        baseline into its contact guard + engage gate. Like the scripted descent it
        needs a motion axis: when none can be estimated it stays on the learned path
        (returns False) rather than searching in an unknown direction.

        When :attr:`GuardedDescentConfig.search_vertical_axis` is set the push axis
        is instead forced to world-vertical ``(0, 0, -1)`` (the vertical-port prior),
        with the spiral lateral basis in the world XY plane. That override is
        self-directed -- it does not consult the motion-axis estimate -- so the
        search engages even when no approach motion was recorded. Every
        :class:`SearchDescent` safety guard (engage force, hard back-off, travel cap,
        max steps) is unchanged, and the orientation stays frozen at the anchor.

        Args:
            t: Elapsed time (s).
            pos: Anchor TCP position ``(3,)`` (m).
            quat: Anchor TCP orientation ``(4,)`` ``[x, y, z, w]``.
            baseline: Approach-phase mean force ``(3,)`` (N), or None.
            logs: Mutable log-line list appended to in place.

        Returns:
            True when the spiral search engaged; False (staying on the learned path)
            when no approach axis is available and the vertical override is off.
        """
        if self.config.search_vertical_axis:
            # Vertical-port prior: descend straight world -z with the spiral in the
            # world XY plane, independent of the (possibly diagonal) motion axis.
            # _lateral_basis((0, 0, -1)) yields a world-XY (u, v), so the lateral
            # offset stays in world X/Y while the push accumulates in world -z.
            axis = _WORLD_DOWN_AXIS.copy()
        else:
            axis = self.axis_estimator.estimate()
            if axis is None:
                if not self._axis_unavailable_logged:
                    self._axis_unavailable_logged = True
                    logs.append(
                        f"[search] stall detected at t={t:.1f}s but approach axis "
                        f"could not be estimated ({len(self.axis_estimator)} samples); "
                        f"staying on the learned path"
                    )
                return False
        self.descent = SearchDescent(
            axis=axis,
            anchor_position=pos,
            anchor_quaternion=quat,
            baseline_force=baseline,
            radius_max=self.config.search_radius_max,
            turns=self.config.search_turns,
            z_step=self.config.search_z_step,
            omega=self.config.search_omega,
            engage_force=self.config.search_engage_force,
            travel_cap=self.config.search_travel_cap,
            max_steps=self.config.search_max_steps,
            yaw_amp=self.config.search_yaw_amp,
            contact_force_threshold=self.config.contact_force_threshold,
        )
        self.phase = GuardedPhase.DESCEND
        self._bearing_source = "search"
        baseline_norm = 0.0 if baseline is None else float(np.linalg.norm(baseline))
        max_steps_desc = (
            "unlimited" if self.config.search_max_steps == 0
            else str(self.config.search_max_steps)
        )
        gc = self.config
        logs.append(
            f"[search] HANDOFF at t={t:.1f}s: stall detected, engaging scripted "
            f"spiral search. "
            f"axis=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}) "
            f"anchor=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
            f"baseline|F|={baseline_norm:.2f}N radius_max={gc.search_radius_max*1e3:.1f}mm "
            f"turns={gc.search_turns:g} z_step={gc.search_z_step*1e3:.2f}mm "
            f"omega={gc.search_omega:g}rad engage_force={gc.search_engage_force:.1f}N "
            f"travel_cap={gc.search_travel_cap*1e3:.0f}mm max_steps={max_steps_desc} "
            f"yaw_amp={gc.search_yaw_amp:g}rad "
            f"force_thr={gc.contact_force_threshold:.1f}N"
        )
        return True

    def _maybe_reaim(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Re-query the provider and rate-limit-blend the descent axis toward it.

        Args:
            pos: Current TCP position ``(3,)`` (m).
            quat: Current TCP orientation ``(4,)`` ``[x, y, z, w]``.
        """
        assert self.bearing_provider is not None and self.descent is not None
        target, magnitude, ok, _axis = _split_reading(
            self.bearing_provider.predict(pos, quat)
        )
        if not ok or target is None:
            return
        t = np.asarray(target, dtype=np.float64).reshape(-1)
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            return
        if not (self.config.aux_min_mag <= magnitude <= self.config.aux_max_mag):
            return
        d = t - self.descent.anchor_position
        norm = float(np.linalg.norm(d))
        if norm < _AXIS_MIN_NORM:
            return
        self.descent.reaim(d / norm, self.config.reaim_rate)

    def _result(
        self,
        targets: np.ndarray | None,
        log_lines: tuple[str, ...],
        triggered_this_cycle: bool,
    ) -> GuardedStep:
        """Assemble a :class:`GuardedStep` snapshot of the current state.

        Args:
            targets: The pose targets for this cycle, or None.
            log_lines: Log lines produced this cycle.
            triggered_this_cycle: Whether the handoff happened this cycle.

        Returns:
            The populated :class:`GuardedStep`.
        """
        return GuardedStep(
            targets=targets,
            stiffness=self.config.stiffness(),
            log_lines=log_lines,
            phase=self.phase.value,
            steps=0 if self.descent is None else self.descent.steps,
            travel=0.0 if self.descent is None else self.descent.travel,
            force_delta=0.0 if self.descent is None else self.descent.last_force_delta,
            contacts=0 if self.descent is None else self.descent.contacts,
            triggered_this_cycle=triggered_this_cycle,
            bearing_source=self._bearing_source,
        )


def _interpolate_targets(
    prev_target: np.ndarray, new_target: np.ndarray, substeps: int
) -> np.ndarray:
    """Linearly interpolate position between two pose targets over ``substeps``.

    The orientation is taken from ``new_target`` for every sub-pose (the descent
    holds a constant orientation, so ``prev`` and ``new`` share it). Emitting
    several small sub-poses per step matches ``DeployACT``'s sub-stepped command
    cadence for a smooth impedance-reference stream.

    Args:
        prev_target: The ``(7,)`` pose target at the start of the step.
        new_target: The ``(7,)`` pose target at the end of the step.
        substeps: Number of sub-poses to emit (>= 1). The final sub-pose equals
            ``new_target``.

    Returns:
        An ``(substeps, 7)`` array of interpolated pose targets.
    """
    prev = np.asarray(prev_target, dtype=np.float64).reshape(7)
    new = np.asarray(new_target, dtype=np.float64).reshape(7)
    out = np.empty((substeps, 7), dtype=np.float64)
    for k in range(1, substeps + 1):
        frac = k / substeps
        out[k - 1, :3] = prev[:3] + (new[:3] - prev[:3]) * frac
        out[k - 1, 3:] = new[3:]
    return out


def _split_reading(
    reading: tuple,
) -> tuple[np.ndarray | None, float, bool, np.ndarray | None]:
    """Normalize a provider reading into ``(target, magnitude, ok, axis)``.

    Accepts the canonical 4-tuple ``(target, magnitude, ok, axis_base)`` and the
    legacy 3-tuple ``(target, magnitude, ok)`` (``axis`` defaults to None), so a
    3-D (offset-only) provider stays byte-identical.

    Args:
        reading: The ``PortBearingProvider.predict`` return (3- or 4-tuple).

    Returns:
        The 4-tuple ``(target_base, magnitude, ok, axis_base)`` with ``axis_base``
        None when the reading carried no explicit axis.
    """
    target = reading[0]
    magnitude = float(reading[1])
    ok = bool(reading[2])
    axis = reading[3] if len(reading) >= 4 else None
    return target, magnitude, ok, axis


def _unit_axis(axis: np.ndarray | None) -> np.ndarray | None:
    """Return ``axis`` as a finite ``(3,)`` unit vector, or None if unusable.

    Args:
        axis: A candidate approach axis ``(3,)`` in base_link, or None.

    Returns:
        The normalized axis, or None when ``axis`` is None, malformed, non-finite,
        or has near-zero norm (any of which falls the caller back to the
        offset-derived direction).
    """
    if axis is None:
        return None
    a = np.asarray(axis, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    norm = float(np.linalg.norm(a))
    if norm < _AXIS_MIN_NORM:
        return None
    return a / norm


class GuardedTraceWriter:
    """Appends guarded-descent log lines to a per-trial trace file.

    ``DeployACT`` emits the guarded ``[guarded]`` handoff/HOLD lines and the
    periodic step status through its ROS logger, but those only reliably reach the
    engine's captured stdout on some code paths (forensics caveat, 2026-07-19).
    This tiny append-only writer mirrors them to a plain file in the process's
    current working directory -- at runtime the per-trial directory -- so the
    guarded telemetry is always recoverable per trial. It holds no file handle
    open between writes (each call opens, appends, and closes) so a crashed or
    killed trial still leaves a complete trace; an empty batch writes nothing (no
    stray empty file is created).
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        """Initialize the writer.

        Args:
            path: Destination file. When ``None``, the file is
                ``guarded_trace.log`` inside ``AIC_GUARDED_TRACE_DIR`` if that
                environment variable is set (the eval runner exports the
                per-trial directory), else the process CWD. The 2026-07-19
                batches showed the policy node's CWD is the repo root, not the
                trial directory, so the CWD fallback interleaves trials into
                one file -- always prefer the env var in eval runs.
        """
        if path is None:
            trace_dir = os.environ.get("AIC_GUARDED_TRACE_DIR", "").strip()
            base = pathlib.Path(trace_dir) if trace_dir else pathlib.Path()
            self.path = base / DEFAULT_TRACE_FILENAME
        else:
            self.path = pathlib.Path(path)

    def write_lines(self, lines: Iterable[str]) -> int:
        """Append log lines to the trace file, one per line.

        Args:
            lines: Log lines to append (a trailing newline is added per line).

        Returns:
            The number of lines written (0 for an empty batch, which opens no
            file).

        Raises:
            OSError: If the file cannot be opened or written.
        """
        materialized = [str(line) for line in lines]
        if not materialized:
            return 0
        with self.path.open("a", encoding="utf-8") as fh:
            for line in materialized:
                fh.write(f"{line}\n")
        return len(materialized)
