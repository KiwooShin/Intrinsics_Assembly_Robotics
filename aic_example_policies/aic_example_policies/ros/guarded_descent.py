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
from typing import Mapping

import numpy as np

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

_QUAT_MIN_NORM: float = 1e-9
_AXIS_MIN_NORM: float = 1e-9


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
        ``AIC_GUARDED_ZSTIFFNESS``.

        Args:
            env: Environment mapping (e.g. ``os.environ``).

        Returns:
            The parsed :class:`GuardedDescentConfig`. When ``AIC_GUARDED`` is not
            exactly ``"1"`` the returned config has ``enabled=False`` and default
            thresholds.

        Raises:
            ValueError: If a provided override cannot be parsed as its numeric
                type.
        """
        enabled = env.get("AIC_GUARDED", "0").strip() == "1"

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

    def __init__(self, config: GuardedDescentConfig) -> None:
        """Initialize the controller from a config.

        Args:
            config: The parsed :class:`GuardedDescentConfig`. ``enabled`` is not
                enforced here (``DeployACT`` decides whether to construct the
                controller at all), but callers normally construct it only when
                ``config.enabled`` is True.
        """
        self.config = config
        self.stall = StallDetector(
            speed_threshold=config.speed_threshold,
            stall_window_s=config.stall_window_s,
            min_runtime_s=config.min_runtime_s,
        )
        self.axis_estimator = ApproachAxisEstimator(
            window=config.axis_window,
            min_displacement=config.axis_min_displacement,
        )
        self.descent: GuardedDescent | None = None
        self.phase = GuardedPhase.APPROACH
        self._last_position: np.ndarray | None = None
        self._last_time: float | None = None
        self._baseline_sum = np.zeros(3, dtype=np.float64)
        self._baseline_count = 0
        self._axis_unavailable_logged = False

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
            if self.stall.update(t, speed):
                axis = self.axis_estimator.estimate()
                if axis is None:
                    if not self._axis_unavailable_logged:
                        self._axis_unavailable_logged = True
                        logs.append(
                            f"[guarded] stall detected at t={t:.1f}s but approach axis "
                            f"could not be estimated ({len(self.axis_estimator)} samples); "
                            f"staying on the learned path"
                        )
                    return self._result(None, tuple(logs), False)
                baseline = self._baseline()
                self.descent = GuardedDescent(
                    axis=axis,
                    anchor_position=pos,
                    anchor_quaternion=quat,
                    baseline_force=baseline,
                    step_size=self.config.step_size,
                    travel_cap=self.config.travel_cap,
                    contact_force_threshold=self.config.contact_force_threshold,
                )
                self.phase = GuardedPhase.DESCEND
                triggered_this_cycle = True
                baseline_norm = 0.0 if baseline is None else float(np.linalg.norm(baseline))
                logs.append(
                    f"[guarded] HANDOFF at t={t:.1f}s: stall detected, engaging guarded "
                    f"descent. axis=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}) "
                    f"anchor=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                    f"baseline|F|={baseline_norm:.2f}N step={self.config.step_size*1e3:.1f}mm "
                    f"cap={self.config.travel_cap*1e3:.0f}mm force_thr="
                    f"{self.config.contact_force_threshold:.1f}N"
                )
            else:
                return self._result(None, tuple(logs), False)

        # Descending or holding: advance the state machine and emit targets.
        assert self.descent is not None
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
