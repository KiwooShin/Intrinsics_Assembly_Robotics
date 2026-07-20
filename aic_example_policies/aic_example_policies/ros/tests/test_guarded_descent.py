#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free guarded-descent probe used by ``DeployACT``.

These exercise :mod:`aic_example_policies.ros.guarded_descent` -- the stall
detector, approach-axis estimator, guarded-descent state machine, and the
``GuardedDescentController`` seam ``DeployACT`` calls -- and require neither ROS,
Gazebo, torch, nor a GPU.

Run with::

    PYTHONPATH=<repo>/aic_example_policies python -m unittest \
        aic_example_policies.ros.tests.test_guarded_descent -v
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
import unittest.mock

import numpy as np

from aic_example_policies.ros import pose_integration
from aic_example_policies.ros.guarded_descent import (
    DEFAULT_AUX_FIXED_TRAVEL,
    DEFAULT_CONTACT_FORCE_THRESHOLD,
    DEFAULT_SPECIALIST_DT,
    DEFAULT_SPECIALIST_EXEC_STEPS,
    DEFAULT_SPECIALIST_MAX_STEPS,
    DEFAULT_SPECIALIST_TRAVEL_CAP,
    DEFAULT_STEP_SIZE,
    DEFAULT_TRAVEL_CAP,
    ApproachAxisEstimator,
    GuardedDescent,
    GuardedDescentConfig,
    GuardedDescentController,
    GuardedPhase,
    GuardedTraceWriter,
    SpecialistDescent,
    StallDetector,
)


class StallDetectorTest(unittest.TestCase):
    """Tests for :class:`StallDetector` edge behavior."""

    def test_no_stall_when_speed_stays_high(self) -> None:
        """A trajectory that never slows below threshold never fires."""
        det = StallDetector(speed_threshold=0.01, stall_window_s=3.0, min_runtime_s=15.0)
        fired = [det.update(t=i * 0.5, speed=0.05) for i in range(80)]
        self.assertFalse(any(fired))
        self.assertFalse(det.triggered)

    def test_low_speed_at_startup_is_suppressed(self) -> None:
        """Low speed during the grace period must not trigger a stall."""
        det = StallDetector(speed_threshold=0.01, stall_window_s=3.0, min_runtime_s=15.0)
        # Continuously stationary from t=0, but every sample is inside the grace
        # window (< 15 s) so nothing fires.
        for i in range(30):  # t = 0.0 .. 14.5
            t = i * 0.5
            self.assertLess(t, 15.0)
            self.assertFalse(det.update(t=t, speed=0.0))
        self.assertFalse(det.triggered)

    def test_stationary_since_start_fires_once_after_grace(self) -> None:
        """Stationary-since-t=0 fires exactly at the first cycle past the grace."""
        det = StallDetector(speed_threshold=0.01, stall_window_s=3.0, min_runtime_s=15.0)
        fires = [det.update(t=i * 0.5, speed=0.0) for i in range(40)]  # t up to 19.5
        self.assertEqual(sum(fires), 1)
        # First sample with t >= 15.0 is i = 30 (t = 15.0); low span since 0 >= 3.
        self.assertTrue(fires[30])
        self.assertTrue(det.triggered)

    def test_sustained_stall_after_motion_fires_once(self) -> None:
        """Moving then stalling fires once, `stall_window_s` after the stop."""
        det = StallDetector(speed_threshold=0.01, stall_window_s=3.0, min_runtime_s=15.0)
        fires = []
        for i in range(60):
            t = i * 0.5
            speed = 0.05 if t < 20.0 else 0.0  # stops at t = 20.0
            fires.append(det.update(t=t, speed=speed))
        self.assertEqual(sum(fires), 1)
        fire_time = next(i * 0.5 for i, f in enumerate(fires) if f)
        # Stop at 20.0, window 3.0 -> first fire at t >= 23.0.
        self.assertGreaterEqual(fire_time, 23.0)
        self.assertLess(fire_time, 23.5)

    def test_recovery_resets_the_low_window(self) -> None:
        """A brief slow-down that recovers does not accumulate toward a stall."""
        det = StallDetector(speed_threshold=0.01, stall_window_s=3.0, min_runtime_s=15.0)
        det.update(t=16.0, speed=0.0)
        det.update(t=16.5, speed=0.0)
        self.assertIsNotNone(det.low_since)
        det.update(t=17.0, speed=0.05)  # recovered -> window resets
        self.assertIsNone(det.low_since)
        # New low span starts at 17.5; at 19.0 it is only 1.5 s -> no fire.
        for t in (17.5, 18.0, 18.5, 19.0):
            self.assertFalse(det.update(t=t, speed=0.0))
        self.assertFalse(det.triggered)

    def test_fires_only_once_then_latches(self) -> None:
        """After firing, subsequent updates return False."""
        det = StallDetector(min_runtime_s=1.0, stall_window_s=1.0)
        det.update(t=2.0, speed=0.0)
        first = det.update(t=3.5, speed=0.0)
        self.assertTrue(first)
        self.assertFalse(det.update(t=4.0, speed=0.0))
        self.assertFalse(det.update(t=4.5, speed=0.0))

    def test_reset_clears_latch(self) -> None:
        """`reset` returns the detector to its initial state."""
        det = StallDetector(min_runtime_s=1.0, stall_window_s=1.0)
        det.update(t=2.0, speed=0.0)
        det.update(t=3.5, speed=0.0)
        self.assertTrue(det.triggered)
        det.reset()
        self.assertFalse(det.triggered)
        self.assertIsNone(det.low_since)

    def test_invalid_thresholds_raise(self) -> None:
        """Out-of-range constructor arguments raise ValueError."""
        with self.assertRaises(ValueError):
            StallDetector(speed_threshold=0.0)
        with self.assertRaises(ValueError):
            StallDetector(stall_window_s=-1.0)
        with self.assertRaises(ValueError):
            StallDetector(min_runtime_s=-1.0)


class ApproachAxisEstimatorTest(unittest.TestCase):
    """Tests for :class:`ApproachAxisEstimator`."""

    def test_straight_line_axis(self) -> None:
        """A pure -Z approach yields the -Z unit axis."""
        est = ApproachAxisEstimator(window=8, min_displacement=1e-4)
        for _ in range(10):
            est.observe(np.array([0.0, 0.0, -0.01]))
        axis = est.estimate()
        assert axis is not None
        np.testing.assert_allclose(axis, [0.0, 0.0, -1.0], atol=1e-9)

    def test_diagonal_axis_is_unit_norm(self) -> None:
        """A diagonal approach yields a unit vector in that direction."""
        est = ApproachAxisEstimator(window=8)
        for _ in range(5):
            est.observe(np.array([0.006, 0.0, -0.008]))
        axis = est.estimate()
        assert axis is not None
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)
        np.testing.assert_allclose(axis, [0.6, 0.0, -0.8], atol=1e-9)

    def test_noisy_trajectory_recovers_dominant_direction(self) -> None:
        """Noisy displacements still average to the dominant -Z direction."""
        rng = np.random.default_rng(20260719)
        est = ApproachAxisEstimator(window=32, min_displacement=1e-5)
        for _ in range(32):
            disp = np.array([0.0, 0.0, -0.01]) + rng.normal(0.0, 0.001, size=3)
            est.observe(disp)
        axis = est.estimate()
        assert axis is not None
        # Dominant component is -Z; xy leakage stays small.
        self.assertLess(axis[2], -0.95)
        self.assertLess(abs(axis[0]), 0.2)
        self.assertLess(abs(axis[1]), 0.2)

    def test_tiny_displacements_are_ignored(self) -> None:
        """Sub-threshold moves are not recorded, so no axis is available."""
        est = ApproachAxisEstimator(window=8, min_displacement=1e-3)
        for _ in range(10):
            est.observe(np.array([0.0, 0.0, -1e-6]))  # far below threshold
        self.assertEqual(len(est), 0)
        self.assertIsNone(est.estimate())

    def test_cancelling_motion_returns_none(self) -> None:
        """Equal-and-opposite motion averages to ~zero -> no axis."""
        est = ApproachAxisEstimator(window=8, min_displacement=1e-6)
        for _ in range(4):
            est.observe(np.array([0.01, 0.0, 0.0]))
            est.observe(np.array([-0.01, 0.0, 0.0]))
        self.assertIsNone(est.estimate())

    def test_window_keeps_only_recent_samples(self) -> None:
        """Only the last `window` above-threshold samples drive the estimate."""
        est = ApproachAxisEstimator(window=4, min_displacement=1e-4)
        for _ in range(12):  # old +X motion
            est.observe(np.array([0.01, 0.0, 0.0]))
        for _ in range(4):  # recent +Y motion fills the window
            est.observe(np.array([0.0, 0.01, 0.0]))
        axis = est.estimate()
        assert axis is not None
        np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)

    def test_invalid_arguments_raise(self) -> None:
        """Bad constructor / observe arguments raise ValueError."""
        with self.assertRaises(ValueError):
            ApproachAxisEstimator(window=0)
        with self.assertRaises(ValueError):
            ApproachAxisEstimator(min_displacement=-1.0)
        est = ApproachAxisEstimator()
        with self.assertRaises(ValueError):
            est.observe(np.array([1.0, 2.0]))


class GuardedDescentTest(unittest.TestCase):
    """Tests for the :class:`GuardedDescent` stepping/back-off/cap state machine."""

    def _descent(self, **overrides: object) -> GuardedDescent:
        """Build a downward descent from a fixed anchor with test overrides."""
        kwargs: dict[str, object] = dict(
            axis=np.array([0.0, 0.0, -1.0]),
            anchor_position=np.array([0.1, 0.2, 0.5]),
            anchor_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            baseline_force=None,
            step_size=0.004,
            travel_cap=0.12,
            contact_force_threshold=12.0,
        )
        kwargs.update(overrides)
        return GuardedDescent(**kwargs)  # type: ignore[arg-type]

    def test_axis_is_normalized(self) -> None:
        """A non-unit axis is normalized on construction."""
        d = self._descent(axis=np.array([0.0, 0.0, -2.0]))
        np.testing.assert_allclose(d.axis, [0.0, 0.0, -1.0])

    def test_steps_advance_along_axis(self) -> None:
        """Each advance moves one step_size along the axis from the anchor."""
        d = self._descent()
        t1 = d.advance(force=None)
        self.assertEqual(d.steps, 1)
        np.testing.assert_allclose(t1[:3], [0.1, 0.2, 0.5 - 0.004])
        t2 = d.advance(force=None)
        self.assertEqual(d.steps, 2)
        np.testing.assert_allclose(t2[:3], [0.1, 0.2, 0.5 - 0.008])
        self.assertAlmostEqual(d.travel, 0.008)

    def test_orientation_is_held_constant(self) -> None:
        """The quaternion is unchanged across all commanded targets."""
        d = self._descent(anchor_quaternion=np.array([0.0, 0.0, 0.3827, 0.9239]))
        for _ in range(5):
            tgt = d.advance(force=None)
            np.testing.assert_allclose(tgt[3:], [0.0, 0.0, 0.3827, 0.9239])

    def test_travel_cap_holds_without_exceeding(self) -> None:
        """Travel never exceeds the cap; phase becomes HOLD at the cap."""
        d = self._descent(step_size=0.004, travel_cap=0.01)
        for _ in range(20):
            d.advance(force=None)
            self.assertLessEqual(d.travel, 0.01 + 1e-12)
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        self.assertEqual(d.steps, 2)  # 0.008 <= 0.01 < 0.012

    def test_contact_backoff_holds_one_step_back(self) -> None:
        """A baseline-relative force spike backs off one step and holds."""
        d = self._descent(baseline_force=np.array([19.0, 0.0, 0.0]),
                          contact_force_threshold=12.0)
        d.advance(force=np.array([19.0, 0.0, 0.0]))  # delta 0 -> step to 1
        d.advance(force=np.array([19.0, 0.0, 0.0]))  # step to 2
        self.assertEqual(d.steps, 2)
        # Spike: |F - baseline| = 16 N > 12 N -> back off to step 1 and hold.
        held = d.advance(force=np.array([35.0, 0.0, 0.0]))
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        self.assertEqual(d.steps, 1)
        self.assertEqual(d.contacts, 1)
        np.testing.assert_allclose(held[:3], [0.1, 0.2, 0.5 - 0.004])
        # Further advances just re-command the held target.
        again = d.advance(force=np.array([35.0, 0.0, 0.0]))
        self.assertEqual(d.steps, 1)
        np.testing.assert_allclose(again[:3], held[:3])

    def test_force_delta_tracked_when_below_threshold(self) -> None:
        """Sub-threshold force deltas are recorded but do not back off."""
        d = self._descent(baseline_force=np.array([19.0, 0.0, 0.0]),
                          contact_force_threshold=12.0)
        d.advance(force=np.array([24.0, 0.0, 0.0]))  # delta 5 N < 12 N
        self.assertAlmostEqual(d.last_force_delta, 5.0)
        self.assertEqual(d.phase, GuardedPhase.DESCEND)
        self.assertEqual(d.steps, 1)

    def test_no_baseline_disables_contact_guard(self) -> None:
        """Without a baseline, even a huge force does not back off."""
        d = self._descent(baseline_force=None)
        for _ in range(5):
            d.advance(force=np.array([500.0, 0.0, 0.0]))
        self.assertEqual(d.phase, GuardedPhase.DESCEND)
        self.assertEqual(d.steps, 5)

    def test_invalid_arguments_raise(self) -> None:
        """Malformed / out-of-range arguments raise ValueError."""
        with self.assertRaises(ValueError):
            self._descent(axis=np.array([0.0, 0.0, 0.0]))  # zero axis
        with self.assertRaises(ValueError):
            self._descent(step_size=0.0)
        with self.assertRaises(ValueError):
            self._descent(travel_cap=-1.0)
        with self.assertRaises(ValueError):
            self._descent(anchor_quaternion=np.array([0.0, 0.0, 0.0, 0.0]))


class GuardedDescentConfigTest(unittest.TestCase):
    """Tests for env parsing and the stiffness helper of the config."""

    def test_disabled_by_default(self) -> None:
        """An empty environment yields a disabled config with defaults."""
        cfg = GuardedDescentConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.step_size, DEFAULT_STEP_SIZE)
        self.assertEqual(cfg.travel_cap, DEFAULT_TRAVEL_CAP)
        self.assertEqual(cfg.contact_force_threshold, DEFAULT_CONTACT_FORCE_THRESHOLD)
        self.assertIsNone(cfg.z_stiffness)

    def test_enabled_flag(self) -> None:
        """`AIC_GUARDED=1` enables; any other value stays disabled."""
        self.assertTrue(GuardedDescentConfig.from_env({"AIC_GUARDED": "1"}).enabled)
        self.assertFalse(GuardedDescentConfig.from_env({"AIC_GUARDED": "0"}).enabled)
        self.assertFalse(GuardedDescentConfig.from_env({"AIC_GUARDED": "yes"}).enabled)

    def test_overrides_parse(self) -> None:
        """All `AIC_GUARDED_*` overrides parse into the config."""
        cfg = GuardedDescentConfig.from_env(
            {
                "AIC_GUARDED": "1",
                "AIC_GUARDED_SPEED": "0.02",
                "AIC_GUARDED_STALL_WINDOW": "2.0",
                "AIC_GUARDED_MIN_RUNTIME": "10.0",
                "AIC_GUARDED_AXIS_N": "12",
                "AIC_GUARDED_AXIS_MIN_DISP": "0.0005",
                "AIC_GUARDED_STEP": "0.006",
                "AIC_GUARDED_TRAVEL_CAP": "0.09",
                "AIC_GUARDED_FORCE": "15.0",
                "AIC_GUARDED_ZSTIFFNESS": "40.0",
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.speed_threshold, 0.02)
        self.assertEqual(cfg.stall_window_s, 2.0)
        self.assertEqual(cfg.min_runtime_s, 10.0)
        self.assertEqual(cfg.axis_window, 12)
        self.assertEqual(cfg.axis_min_displacement, 0.0005)
        self.assertEqual(cfg.step_size, 0.006)
        self.assertEqual(cfg.travel_cap, 0.09)
        self.assertEqual(cfg.contact_force_threshold, 15.0)
        self.assertEqual(cfg.z_stiffness, 40.0)

    def test_bad_override_raises(self) -> None:
        """A non-numeric override raises ValueError."""
        with self.assertRaises(ValueError):
            GuardedDescentConfig.from_env({"AIC_GUARDED": "1", "AIC_GUARDED_STEP": "abc"})

    def test_aux_fixed_travel_default(self) -> None:
        """`aux_fixed_travel` defaults to DEFAULT_AUX_FIXED_TRAVEL when unset."""
        self.assertEqual(
            GuardedDescentConfig.from_env({}).aux_fixed_travel, DEFAULT_AUX_FIXED_TRAVEL
        )
        self.assertEqual(GuardedDescentConfig().aux_fixed_travel, DEFAULT_AUX_FIXED_TRAVEL)

    def test_aux_fixed_travel_env_parses(self) -> None:
        """`AIC_GUARDED_AUX_TRAVEL` overrides the fixed deep-travel cap."""
        cfg = GuardedDescentConfig.from_env(
            {"AIC_GUARDED": "1", "AIC_GUARDED_AUX": "1", "AIC_GUARDED_AUX_TRAVEL": "0.08"}
        )
        self.assertEqual(cfg.aux_fixed_travel, 0.08)

    def test_aux_fixed_travel_bad_override_raises(self) -> None:
        """A non-numeric `AIC_GUARDED_AUX_TRAVEL` raises ValueError."""
        with self.assertRaises(ValueError):
            GuardedDescentConfig.from_env(
                {"AIC_GUARDED": "1", "AIC_GUARDED_AUX_TRAVEL": "deep"}
            )

    def test_stiffness_helper(self) -> None:
        """`stiffness()` returns None by default, else base with Z replaced."""
        self.assertIsNone(GuardedDescentConfig().stiffness())
        gains = GuardedDescentConfig(z_stiffness=40.0).stiffness()
        assert gains is not None
        self.assertEqual(gains, [90.0, 90.0, 40.0, 50.0, 50.0, 50.0])


class GuardedDescentControllerTest(unittest.TestCase):
    """Tests for the DeployACT-facing controller seam."""

    def _enabled_config(self, **overrides: object) -> GuardedDescentConfig:
        """Return an enabled config with small thresholds for fast tests."""
        base: dict[str, object] = dict(
            enabled=True,
            speed_threshold=0.01,
            stall_window_s=3.0,
            min_runtime_s=15.0,
            step_size=0.004,
            travel_cap=0.12,
            contact_force_threshold=12.0,
        )
        base.update(overrides)
        return GuardedDescentConfig(**base)  # type: ignore[arg-type]

    def _run_stream(
        self, ctrl: GuardedDescentController, substeps: int = 5
    ) -> list:
        """Drive a canned approach-then-stall stream and collect the steps.

        The TCP descends along -Z at 0.01 m/cycle until t ~ 19 s, then holds
        stationary; the wrist force is a constant 19 N baseline throughout.
        """
        steps = []
        pos = np.array([0.15, -0.05, 0.60])
        force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(120):
            t = i * 0.275
            if t < 19.0:  # approaching
                pos = pos + np.array([0.0, 0.0, -0.01])
            steps.append(ctrl.cycle(t, pos, quat, force, substeps=substeps))
        return steps

    def test_approach_returns_none_targets(self) -> None:
        """Before a stall the controller keeps the learned path (targets None)."""
        ctrl = GuardedDescentController(self._enabled_config())
        pos = np.array([0.15, -0.05, 0.60])
        force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(20):  # t up to ~5.2 s, well inside the grace period
            pos = pos + np.array([0.0, 0.0, -0.01])
            step = ctrl.cycle(i * 0.275, pos, quat, force, substeps=5)
            self.assertIsNone(step.targets)
            self.assertEqual(step.phase, GuardedPhase.APPROACH.value)
        self.assertFalse(ctrl.active)

    def test_handoff_then_descent(self) -> None:
        """A stall hands off to descent: 7-D targets and a single handoff cycle."""
        ctrl = GuardedDescentController(self._enabled_config())
        steps = self._run_stream(ctrl, substeps=5)
        self.assertTrue(ctrl.active)
        # Exactly one cycle reports the handoff.
        self.assertEqual(sum(1 for s in steps if s.triggered_this_cycle), 1)
        handoff_idx = next(i for i, s in enumerate(steps) if s.triggered_this_cycle)
        # Handoff logs a HANDOFF line.
        self.assertTrue(any("HANDOFF" in ln for ln in steps[handoff_idx].log_lines))
        # After handoff, targets are commanded and shaped (substeps, 7).
        post = steps[handoff_idx]
        assert post.targets is not None
        self.assertEqual(post.targets.shape, (5, 7))
        # Before handoff, every cycle keeps the learned path.
        self.assertTrue(all(s.targets is None for s in steps[:handoff_idx]))

    def test_descent_axis_is_downward(self) -> None:
        """The estimated axis reflects the -Z approach, so targets step down."""
        ctrl = GuardedDescentController(self._enabled_config())
        steps = self._run_stream(ctrl)
        handoff_idx = next(i for i, s in enumerate(steps) if s.triggered_this_cycle)
        first_targets = steps[handoff_idx].targets
        second_targets = steps[handoff_idx + 1].targets
        assert first_targets is not None and second_targets is not None
        # z strictly decreases as steps advance.
        self.assertLess(second_targets[-1, 2], first_targets[-1, 2])

    def test_travel_cap_stops_descent(self) -> None:
        """With a tiny cap, travel saturates and the phase reaches HOLD."""
        ctrl = GuardedDescentController(self._enabled_config(travel_cap=0.012))
        steps = self._run_stream(ctrl)
        self.assertTrue(ctrl.active)
        self.assertLessEqual(ctrl.descent.travel, 0.012 + 1e-12)
        self.assertEqual(steps[-1].phase, GuardedPhase.HOLD.value)

    def test_contact_backoff_via_controller(self) -> None:
        """A late force spike drives a contact back-off and a HOLD log line."""
        ctrl = GuardedDescentController(self._enabled_config())
        # Approach then stall with a constant baseline force.
        pos = np.array([0.15, -0.05, 0.60])
        base_force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        engaged = False
        saw_hold_log = False
        for i in range(160):
            t = i * 0.275
            if t < 19.0:
                pos = pos + np.array([0.0, 0.0, -0.01])
            # Once descending, inject a hard-contact spike.
            force = base_force
            if ctrl.active and ctrl.descent.steps >= 2:
                force = np.array([40.0, 0.0, 0.0])  # |F-base| = 21 N > 12 N
            step = ctrl.cycle(t, pos, quat, force, substeps=3)
            engaged = engaged or ctrl.active
            if any("HOLD" in ln for ln in step.log_lines):
                saw_hold_log = True
        self.assertTrue(engaged)
        self.assertTrue(saw_hold_log)
        self.assertGreaterEqual(ctrl.descent.contacts, 1)
        self.assertEqual(ctrl.descent.phase, GuardedPhase.HOLD)

    def test_stiffness_passthrough(self) -> None:
        """`stiffness` is None by default and a reduced-Z list when configured."""
        ctrl_default = GuardedDescentController(self._enabled_config())
        steps = self._run_stream(ctrl_default)
        self.assertTrue(all(s.stiffness is None for s in steps))

        ctrl_z = GuardedDescentController(self._enabled_config(z_stiffness=40.0))
        steps_z = self._run_stream(ctrl_z)
        descending = [s for s in steps_z if s.targets is not None]
        self.assertTrue(descending)
        for s in descending:
            self.assertEqual(s.stiffness, [90.0, 90.0, 40.0, 50.0, 50.0, 50.0])

    def test_no_axis_stays_on_learned_path(self) -> None:
        """A stall with no recorded approach motion does not engage descent."""
        # Never move: the axis estimator collects no above-threshold samples, so
        # even though the stall fires the controller stays on the learned path.
        ctrl = GuardedDescentController(self._enabled_config())
        pos = np.array([0.15, -0.05, 0.60])
        force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        logged = False
        for i in range(120):
            step = ctrl.cycle(i * 0.275, pos, quat, force, substeps=5)
            self.assertIsNone(step.targets)
            logged = logged or any("could not be estimated" in ln for ln in step.log_lines)
        self.assertFalse(ctrl.active)
        self.assertTrue(logged)

    def test_bad_substeps_raise(self) -> None:
        """A non-positive substeps count raises ValueError."""
        ctrl = GuardedDescentController(self._enabled_config())
        with self.assertRaises(ValueError):
            ctrl.cycle(0.0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), None, substeps=0)


class _FollowProvider:
    """Fake bearing provider: target = current TCP + a fixed base_link offset."""

    def __init__(self, offset: np.ndarray, ok: bool = True) -> None:
        self.offset = np.asarray(offset, dtype=np.float64)
        self.ok = ok
        self.calls = 0

    def predict(self, position, quaternion):
        self.calls += 1
        if not self.ok:
            return None, 0.0, False
        target = np.asarray(position, dtype=np.float64) + self.offset
        return target, float(np.linalg.norm(self.offset)), True


class _AlternatingProvider:
    """Fake provider whose offset alternates between two far-apart vectors."""

    def __init__(self, offset_a: np.ndarray, offset_b: np.ndarray) -> None:
        self.offsets = [np.asarray(offset_a, float), np.asarray(offset_b, float)]
        self.i = 0

    def predict(self, position, quaternion):
        off = self.offsets[self.i % 2]
        self.i += 1
        target = np.asarray(position, dtype=np.float64) + off
        return target, float(np.linalg.norm(off)), True


class _AxisProvider:
    """Fake 6-D provider: target = TCP + offset, plus an EXPLICIT approach axis.

    Returns the canonical 4-tuple ``(target, magnitude, ok, axis_base)``. ``axis``
    may be None to emulate a 3-D (offset-only) checkpoint on the 4-tuple path.
    """

    def __init__(self, offset: np.ndarray, axis: np.ndarray | None, ok: bool = True) -> None:
        self.offset = np.asarray(offset, dtype=np.float64)
        self.axis = None if axis is None else np.asarray(axis, dtype=np.float64)
        self.ok = ok
        self.calls = 0

    def predict(self, position, quaternion):
        self.calls += 1
        if not self.ok:
            return None, 0.0, False, None
        target = np.asarray(position, dtype=np.float64) + self.offset
        return target, float(np.linalg.norm(self.offset)), True, self.axis


class AuxBearingHandoffTest(unittest.TestCase):
    """Tests for the learned port-bearing (``port_aux``) handoff seam."""

    def _aux_config(self, **overrides: object) -> GuardedDescentConfig:
        base: dict[str, object] = dict(
            enabled=True,
            speed_threshold=0.01,
            stall_window_s=3.0,
            min_runtime_s=15.0,
            step_size=0.004,
            travel_cap=0.12,
            contact_force_threshold=12.0,
            use_aux_bearing=True,
            aux_min_mag=0.005,
            aux_max_mag=0.12,
            aux_consistency_std=0.01,
            aux_travel_margin=0.02,
            aux_fixed_travel=0.10,
            aux_min_samples=3,
        )
        base.update(overrides)
        return GuardedDescentConfig(**base)  # type: ignore[arg-type]

    def _run(self, ctrl: GuardedDescentController, substeps: int = 5) -> list:
        """Approach -Z then stall, as in the base controller stream."""
        steps = []
        pos = np.array([0.15, -0.05, 0.60])
        force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(120):
            t = i * 0.275
            if t < 19.0:
                pos = pos + np.array([0.0, 0.0, -0.01])
            steps.append(ctrl.cycle(t, pos, quat, force, substeps=substeps))
        return steps

    def _handoff(self, steps: list):
        idx = next(i for i, s in enumerate(steps) if s.triggered_this_cycle)
        return idx, steps[idx]

    def test_aux_target_drives_axis_and_cap(self) -> None:
        """A plausible aux offset aims the descent and clamps the travel cap."""
        prov = _FollowProvider(np.array([0.0, 0.0, -0.06]))  # 6 cm below, downward
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "aux")
        self.assertEqual(handoff.bearing_source, "aux")
        np.testing.assert_allclose(ctrl.descent.axis, [0.0, 0.0, -1.0], atol=1e-6)
        # travel_cap = min(0.12, |offset| + margin) = min(0.12, 0.06+0.02) = 0.08.
        self.assertAlmostEqual(ctrl.descent.travel_cap, 0.08, places=6)
        self.assertTrue(any("bearing=aux" in ln for ln in handoff.log_lines))

    def test_fallback_when_magnitude_out_of_range(self) -> None:
        """An implausibly large aux offset falls back to the motion axis."""
        prov = _FollowProvider(np.array([0.0, 0.0, -0.30]))  # 30 cm > aux_max_mag
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "motion-axis")
        self.assertAlmostEqual(ctrl.descent.travel_cap, 0.12, places=6)
        self.assertTrue(any("aux fallback" in ln for ln in handoff.log_lines))

    def test_fallback_when_inconsistent(self) -> None:
        """A noisy (high cross-frame spread) aux estimate falls back."""
        prov = _AlternatingProvider(
            np.array([0.0, 0.0, -0.05]), np.array([0.06, 0.0, -0.05])
        )  # ~6 cm apart >> aux_consistency_std
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "motion-axis")
        self.assertTrue(any("std" in ln for ln in handoff.log_lines))

    def test_fallback_when_pointing_away(self) -> None:
        """An aux target opposite the approach direction falls back."""
        prov = _FollowProvider(np.array([0.0, 0.0, +0.06]))  # +Z, away from -Z approach
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "motion-axis")
        self.assertTrue(any("points away" in ln for ln in handoff.log_lines))

    def test_fallback_when_provider_unavailable(self) -> None:
        """A provider that never returns ok leaves the buffer empty -> fallback."""
        prov = _FollowProvider(np.array([0.0, 0.0, -0.06]), ok=False)
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "motion-axis")
        self.assertTrue(any("aux samples" in ln for ln in handoff.log_lines))

    def test_use_aux_bearing_off_ignores_provider(self) -> None:
        """With use_aux_bearing False the provider is never consulted (byte-identical)."""
        prov = _FollowProvider(np.array([0.0, 0.0, -0.06]))
        ctrl = GuardedDescentController(
            self._aux_config(use_aux_bearing=False), bearing_provider=prov
        )
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(prov.calls, 0)  # provider never queried
        self.assertEqual(ctrl._bearing_source, "motion-axis")

    def test_none_provider_matches_no_aux(self) -> None:
        """No provider yields the exact motion-axis descent targets."""
        cfg = self._aux_config()
        ctrl_none = GuardedDescentController(cfg, bearing_provider=None)
        steps_none = self._run(ctrl_none)
        idx, handoff = self._handoff(steps_none)
        self.assertEqual(handoff.bearing_source, "motion-axis")
        # The next descent cycle emits motion-axis (-Z) targets, cap at default.
        self.assertAlmostEqual(ctrl_none.descent.travel_cap, 0.12, places=6)

    def test_reaim_blends_axis_over_descent(self) -> None:
        """With reaim on, a shifting aux target rotates the descent axis."""
        prov = _FollowProvider(np.array([0.0, 0.0, -0.06]))
        ctrl = GuardedDescentController(
            self._aux_config(reaim=True), bearing_provider=prov
        )
        steps = self._run(ctrl)
        idx, _ = self._handoff(steps)
        axis_at_handoff = ctrl.descent.axis.copy()
        # After handoff, shift the provider target sideways; reaim should tilt the
        # axis toward +X over subsequent descent cycles.
        prov.offset = np.array([0.05, 0.0, -0.06])
        pos = np.array([0.15, -0.05, -0.10])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        force = np.array([19.0, 0.0, 0.0])
        for j in range(5):
            ctrl.cycle(30.0 + j * 0.275, pos, quat, force, substeps=3)
        self.assertGreater(ctrl.descent.axis[0], axis_at_handoff[0])

    def test_explicit_axis_used_and_fixed_cap(self) -> None:
        """A 6-D explicit axis steers the descent and a FIXED travel cap applies."""
        # Offset is 6 cm down; the explicit axis agrees (-Z). With a 3-D provider
        # the cap would be |offset|+margin = 0.08; the 6-D path instead uses the
        # fixed cap min(travel_cap, aux_fixed_travel) = min(0.12, 0.10) = 0.10,
        # decoupled from |offset|.
        prov = _AxisProvider(np.array([0.0, 0.0, -0.06]), np.array([0.0, 0.0, -1.0]))
        ctrl = GuardedDescentController(
            self._aux_config(travel_cap=0.12, aux_fixed_travel=0.10),
            bearing_provider=prov,
        )
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "aux")
        np.testing.assert_allclose(ctrl.descent.axis, [0.0, 0.0, -1.0], atol=1e-6)
        self.assertAlmostEqual(ctrl.descent.travel_cap, 0.10, places=6)
        self.assertTrue(any("explicit6D" in ln for ln in handoff.log_lines))

    def test_explicit_axis_ignores_offset_magnitude(self) -> None:
        """The 6-D fixed cap ignores |offset| (a small offset still travels deep)."""
        # A 2 cm offset would cap a 3-D descent at 0.04 m; the 6-D fixed cap keeps
        # the full 0.10 m deep travel regardless of the (unreliable) magnitude.
        prov = _AxisProvider(np.array([0.0, 0.0, -0.02]), np.array([0.0, 0.0, -1.0]))
        ctrl = GuardedDescentController(
            self._aux_config(travel_cap=0.12, aux_fixed_travel=0.10),
            bearing_provider=prov,
        )
        steps = self._run(ctrl)
        self.assertEqual(ctrl._bearing_source, "aux")
        self.assertAlmostEqual(ctrl.descent.travel_cap, 0.10, places=6)

    def test_explicit_axis_sign_flipped_to_match_offset(self) -> None:
        """An explicit axis pointing away from the target is sign-flipped."""
        # The head emits +Z (away from the -Z approach/offset); the handoff must
        # flip it so the descent still advances toward the port (dot with offset
        # positive).
        prov = _AxisProvider(np.array([0.0, 0.0, -0.06]), np.array([0.0, 0.0, +1.0]))
        ctrl = GuardedDescentController(self._aux_config(), bearing_provider=prov)
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "aux")
        np.testing.assert_allclose(ctrl.descent.axis, [0.0, 0.0, -1.0], atol=1e-6)
        offset = np.array([0.0, 0.0, -0.06])
        self.assertGreater(float(np.dot(ctrl.descent.axis, offset)), 0.0)
        self.assertTrue(any("explicit6D" in ln for ln in handoff.log_lines))

    def test_none_axis_falls_back_to_offset_cap(self) -> None:
        """A 4-tuple provider with axis=None takes the byte-identical 3-D path."""
        prov = _AxisProvider(np.array([0.0, 0.0, -0.06]), axis=None)
        ctrl = GuardedDescentController(
            self._aux_config(travel_cap=0.12, aux_fixed_travel=0.10),
            bearing_provider=prov,
        )
        steps = self._run(ctrl)
        _, handoff = self._handoff(steps)
        self.assertEqual(ctrl._bearing_source, "aux")
        np.testing.assert_allclose(ctrl.descent.axis, [0.0, 0.0, -1.0], atol=1e-6)
        # |offset| + margin = 0.06 + 0.02 = 0.08 (NOT the fixed 0.10 cap).
        self.assertAlmostEqual(ctrl.descent.travel_cap, 0.08, places=6)
        self.assertTrue(any("offset3D" in ln for ln in handoff.log_lines))

    def test_none_axis_matches_legacy_3tuple(self) -> None:
        """The 4-tuple axis=None path is byte-identical to a legacy 3-tuple provider."""
        off = np.array([0.0, 0.0, -0.06])
        ctrl_legacy = GuardedDescentController(
            self._aux_config(), bearing_provider=_FollowProvider(off)
        )
        ctrl_none = GuardedDescentController(
            self._aux_config(), bearing_provider=_AxisProvider(off, axis=None)
        )
        steps_legacy = self._run(ctrl_legacy)
        steps_none = self._run(ctrl_none)
        np.testing.assert_allclose(ctrl_legacy.descent.axis, ctrl_none.descent.axis)
        self.assertAlmostEqual(
            ctrl_legacy.descent.travel_cap, ctrl_none.descent.travel_cap, places=9
        )
        _, hl = self._handoff(steps_legacy)
        _, hn = self._handoff(steps_none)
        assert hl.targets is not None and hn.targets is not None
        np.testing.assert_allclose(hl.targets, hn.targets)


class GuardedTraceWriterTest(unittest.TestCase):
    """Tests for the per-trial guarded telemetry file writer."""

    def test_writes_and_appends_lines(self) -> None:
        """Lines are appended in order across successive writes."""
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "guarded_trace.log"
            writer = GuardedTraceWriter(path)
            n1 = writer.write_lines(["[guarded] HANDOFF ...", "line-2"])
            n2 = writer.write_lines(("line-3",))
            self.assertEqual((n1, n2), (2, 1))
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["[guarded] HANDOFF ...", "line-2", "line-3"],
            )

    def test_empty_batch_writes_nothing(self) -> None:
        """An empty batch writes zero lines and creates no file."""
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "guarded_trace.log"
            self.assertEqual(GuardedTraceWriter(path).write_lines([]), 0)
            self.assertFalse(path.exists())

    def test_default_filename(self) -> None:
        """The default destination is guarded_trace.log."""
        self.assertEqual(GuardedTraceWriter().path.name, "guarded_trace.log")

    def test_default_honors_trace_dir_env(self) -> None:
        """AIC_GUARDED_TRACE_DIR redirects the default path per trial."""
        with tempfile.TemporaryDirectory() as d:
            with unittest.mock.patch.dict(
                os.environ, {"AIC_GUARDED_TRACE_DIR": d}
            ):
                writer = GuardedTraceWriter()
            self.assertEqual(
                writer.path, pathlib.Path(d) / "guarded_trace.log"
            )
            writer.write_lines(["[guarded] HANDOFF ..."])
            self.assertTrue(
                (pathlib.Path(d) / "guarded_trace.log").is_file()
            )

    def test_blank_trace_dir_env_falls_back_to_cwd(self) -> None:
        """A blank AIC_GUARDED_TRACE_DIR behaves like an unset one."""
        with unittest.mock.patch.dict(
            os.environ, {"AIC_GUARDED_TRACE_DIR": "  "}
        ):
            writer = GuardedTraceWriter()
        self.assertEqual(writer.path, pathlib.Path("guarded_trace.log"))

    def test_explicit_path_ignores_env(self) -> None:
        """An explicit path wins over the environment variable."""
        with tempfile.TemporaryDirectory() as d:
            explicit = pathlib.Path(d) / "custom.log"
            with unittest.mock.patch.dict(
                os.environ, {"AIC_GUARDED_TRACE_DIR": "/nonexistent"}
            ):
                writer = GuardedTraceWriter(explicit)
            self.assertEqual(writer.path, explicit)


class _ChunkProvider:
    """Fake specialist provider returning a fixed base_link TCP-velocity chunk.

    Mirrors the :class:`SpecialistActionProvider` protocol (``predict_chunk``) so
    the specialist descent can be exercised with no torch/checkpoint. ``chunk`` is
    a ``(K, 6)`` twist array; passing ``None`` emulates "no observation yet".
    """

    def __init__(self, chunk: np.ndarray | None) -> None:
        self.chunk = None if chunk is None else np.asarray(chunk, dtype=np.float64)
        self.calls = 0

    def predict_chunk(self, position, quaternion):
        self.calls += 1
        return self.chunk


class SpecialistConfigTest(unittest.TestCase):
    """Env parsing for the learned insertion specialist (``AIC_SPECIALIST``)."""

    def test_disabled_by_default(self) -> None:
        """An empty environment leaves the specialist off with defaults."""
        cfg = GuardedDescentConfig.from_env({})
        self.assertFalse(cfg.specialist_enabled)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.specialist_travel_cap, DEFAULT_SPECIALIST_TRAVEL_CAP)
        self.assertEqual(cfg.specialist_max_steps, DEFAULT_SPECIALIST_MAX_STEPS)
        self.assertEqual(cfg.specialist_exec_steps, DEFAULT_SPECIALIST_EXEC_STEPS)
        self.assertEqual(cfg.specialist_dt, DEFAULT_SPECIALIST_DT)

    def test_specialist_flag_forces_enabled(self) -> None:
        """`AIC_SPECIALIST=1` turns on the specialist AND the controller."""
        cfg = GuardedDescentConfig.from_env({"AIC_SPECIALIST": "1"})
        self.assertTrue(cfg.specialist_enabled)
        self.assertTrue(cfg.enabled)  # forced on so the stall machinery runs

    def test_specialist_off_keeps_guarded_semantics(self) -> None:
        """`AIC_SPECIALIST` unset keeps today's guarded on/off behavior exactly."""
        self.assertFalse(GuardedDescentConfig.from_env({"AIC_GUARDED": "1"}).specialist_enabled)
        self.assertTrue(GuardedDescentConfig.from_env({"AIC_GUARDED": "1"}).enabled)
        self.assertFalse(GuardedDescentConfig.from_env({"AIC_SPECIALIST": "0"}).enabled)
        self.assertFalse(GuardedDescentConfig.from_env({"AIC_SPECIALIST": "yes"}).enabled)

    def test_specialist_overrides_parse(self) -> None:
        """All `AIC_SPECIALIST_*` numeric overrides parse into the config."""
        cfg = GuardedDescentConfig.from_env(
            {
                "AIC_SPECIALIST": "1",
                "AIC_SPECIALIST_TRAVEL_CAP": "0.09",
                "AIC_SPECIALIST_MAX_STEPS": "40",
                "AIC_SPECIALIST_EXEC_STEPS": "2",
                "AIC_SPECIALIST_DT": "0.3",
            }
        )
        self.assertTrue(cfg.specialist_enabled)
        self.assertEqual(cfg.specialist_travel_cap, 0.09)
        self.assertEqual(cfg.specialist_max_steps, 40)
        self.assertEqual(cfg.specialist_exec_steps, 2)
        self.assertEqual(cfg.specialist_dt, 0.3)

    def test_specialist_bad_override_raises(self) -> None:
        """A non-numeric specialist override raises ValueError."""
        with self.assertRaises(ValueError):
            GuardedDescentConfig.from_env(
                {"AIC_SPECIALIST": "1", "AIC_SPECIALIST_MAX_STEPS": "lots"}
            )
        with self.assertRaises(ValueError):
            GuardedDescentConfig.from_env(
                {"AIC_SPECIALIST": "1", "AIC_SPECIALIST_TRAVEL_CAP": "deep"}
            )


class SpecialistDescentTest(unittest.TestCase):
    """Action->target integration, travel-cap clamp, and wrench back-off math."""

    ANCHOR = np.array([0.15, -0.05, 0.60])
    QUAT = np.array([0.0, 0.0, 0.0, 1.0])

    def _descent(self, provider: _ChunkProvider, **overrides: object) -> SpecialistDescent:
        kwargs: dict[str, object] = dict(
            provider=provider,
            anchor_position=self.ANCHOR,
            anchor_quaternion=self.QUAT,
            baseline_force=None,
            travel_cap=0.12,
            max_steps=0,
            exec_steps=4,
            dt_frame=0.275,
            step_size=0.004,
            contact_force_threshold=12.0,
        )
        kwargs.update(overrides)
        return SpecialistDescent(**kwargs)  # type: ignore[arg-type]

    def test_integration_matches_pose_integration(self) -> None:
        """Targets equal expand_twists + integrate_twist_chunk of the exec steps."""
        # 8-step chunk of a constant -Z 1 cm/frame twist; exec_steps=4, substeps=5.
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (8, 1))
        prov = _ChunkProvider(chunk)
        d = self._descent(prov, exec_steps=4)
        targets = d.advance(self.ANCHOR, self.QUAT, force=None, substeps=5)
        exec_chunk = chunk[:4]
        fine = pose_integration.expand_twists(exec_chunk, 5)
        expected = pose_integration.integrate_twist_chunk(
            self.ANCHOR, self.QUAT, fine, 0.275 / 5
        )
        self.assertEqual(targets.shape, (20, 7))  # 4 steps * 5 substeps
        np.testing.assert_allclose(targets, expected, atol=1e-9)
        self.assertEqual(d.steps, 4)
        # Net downward travel = 4 steps * 0.01 m/s * 0.275 s = 0.011 m.
        self.assertAlmostEqual(d.travel, 4 * 0.01 * 0.275, places=6)

    def test_exec_steps_truncates_chunk(self) -> None:
        """Only the first exec_steps twists are integrated per cycle."""
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (8, 1))
        d = self._descent(_ChunkProvider(chunk), exec_steps=2)
        targets = d.advance(self.ANCHOR, self.QUAT, force=None, substeps=3)
        self.assertEqual(targets.shape, (6, 7))  # 2 steps * 3 substeps
        self.assertEqual(d.steps, 2)

    def test_receding_horizon_reanchors_to_measured_pose(self) -> None:
        """Each cycle integrates from the passed (measured) pose, not the anchor."""
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk), exec_steps=4)
        moved = self.ANCHOR + np.array([0.0, 0.0, -0.05])  # arm advanced 5 cm
        targets = d.advance(moved, self.QUAT, force=None, substeps=1)
        # First target starts one integration step below the *measured* pose.
        np.testing.assert_allclose(
            targets[0, :3], moved + np.array([0.0, 0.0, -0.01 * 0.275]), atol=1e-9
        )

    def test_travel_cap_clamps_targets(self) -> None:
        """No commanded target exceeds travel_cap from the anchor; phase -> HOLD."""
        # A large single-step chunk that would jump ~11 cm in one exec cycle.
        chunk = np.tile(np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk), exec_steps=4, travel_cap=0.05)
        targets = d.advance(self.ANCHOR, self.QUAT, force=None, substeps=5)
        dists = np.linalg.norm(targets[:, :3] - self.ANCHOR, axis=1)
        self.assertLessEqual(float(dists.max()), 0.05 + 1e-9)
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        self.assertLessEqual(d.travel, 0.05 + 1e-9)

    def test_travel_cap_hold_recommands_measured_pose(self) -> None:
        """Once HOLD, further advances just re-command the current measured pose."""
        chunk = np.tile(np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk), exec_steps=4, travel_cap=0.05)
        d.advance(self.ANCHOR, self.QUAT, force=None, substeps=5)
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        held_pose = self.ANCHOR + np.array([0.0, 0.0, -0.03])
        held = d.advance(held_pose, self.QUAT, force=None, substeps=5)
        self.assertEqual(held.shape, (1, 7))
        np.testing.assert_allclose(held[0, :3], held_pose, atol=1e-9)

    def test_wrench_backoff_fires_and_holds(self) -> None:
        """A baseline-relative force spike retreats one step and holds."""
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(
            _ChunkProvider(chunk),
            exec_steps=4,
            baseline_force=np.array([19.0, 0.0, 0.0]),
            contact_force_threshold=12.0,
        )
        # First cycle (no spike) establishes the downward motion direction.
        d.advance(self.ANCHOR, self.QUAT, force=np.array([19.0, 0.0, 0.0]), substeps=1)
        self.assertEqual(d.phase, GuardedPhase.DESCEND)
        # Spike: |F - baseline| = 21 N > 12 N -> back off and HOLD.
        pos_now = self.ANCHOR + np.array([0.0, 0.0, -0.011])
        backoff = d.advance(pos_now, self.QUAT, force=np.array([40.0, 0.0, 0.0]), substeps=1)
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        self.assertEqual(d.contacts, 1)
        self.assertAlmostEqual(d.last_force_delta, 21.0, places=6)
        self.assertEqual(backoff.shape, (1, 7))
        # Retreat is one step_size *up* (opposite the -Z descent) from the measured pose.
        self.assertGreater(backoff[0, 2], pos_now[2])

    def test_no_baseline_disables_wrench_guard(self) -> None:
        """Without a baseline even a huge force does not back off."""
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk), baseline_force=None)
        d.advance(self.ANCHOR, self.QUAT, force=np.array([500.0, 0.0, 0.0]), substeps=1)
        self.assertEqual(d.phase, GuardedPhase.DESCEND)

    def test_max_steps_holds(self) -> None:
        """The max_steps safety bound freezes the descent once reached."""
        chunk = np.tile(np.array([0.0, 0.0, -0.001, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk), exec_steps=2, max_steps=2, travel_cap=1.0)
        d.advance(self.ANCHOR, self.QUAT, force=None, substeps=1)  # steps -> 2
        self.assertEqual(d.steps, 2)
        self.assertEqual(d.phase, GuardedPhase.DESCEND)
        out = d.advance(self.ANCHOR, self.QUAT, force=None, substeps=1)  # steps >= max
        self.assertEqual(d.phase, GuardedPhase.HOLD)
        self.assertEqual(out.shape, (1, 7))

    def test_none_chunk_holds_without_advancing(self) -> None:
        """A provider with no observation yet yields a hold (no step taken)."""
        d = self._descent(_ChunkProvider(None))
        out = d.advance(self.ANCHOR, self.QUAT, force=None, substeps=3)
        self.assertEqual(out.shape, (1, 7))
        self.assertEqual(d.steps, 0)
        self.assertEqual(d.phase, GuardedPhase.DESCEND)  # still descending, just idle
        np.testing.assert_allclose(out[0, :3], self.ANCHOR, atol=1e-9)

    def test_logs_specialist_lines(self) -> None:
        """Advancing appends `[specialist]` telemetry lines when a log list is given."""
        chunk = np.tile(np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]), (4, 1))
        d = self._descent(_ChunkProvider(chunk))
        logs: list[str] = []
        d.advance(self.ANCHOR, self.QUAT, force=None, substeps=1, logs=logs, t=20.0)
        self.assertTrue(any(ln.startswith("[specialist]") for ln in logs))

    def test_bad_arguments_raise(self) -> None:
        """Malformed constructor / chunk shapes raise ValueError."""
        prov = _ChunkProvider(np.zeros((4, 6)))
        with self.assertRaises(ValueError):
            self._descent(prov, travel_cap=0.0)
        with self.assertRaises(ValueError):
            self._descent(prov, exec_steps=0)
        with self.assertRaises(ValueError):
            self._descent(prov, dt_frame=0.0)
        with self.assertRaises(ValueError):
            self._descent(prov, max_steps=-1)
        # A malformed chunk (wrong width) raises on advance.
        d = self._descent(_ChunkProvider(np.zeros((4, 3))))
        with self.assertRaises(ValueError):
            d.advance(self.ANCHOR, self.QUAT, force=None, substeps=1)


class SpecialistSwitchTest(unittest.TestCase):
    """Controller switching: specialist-off is the exact scripted path."""

    def _config(self, **overrides: object) -> GuardedDescentConfig:
        base: dict[str, object] = dict(
            enabled=True,
            speed_threshold=0.01,
            stall_window_s=3.0,
            min_runtime_s=15.0,
            step_size=0.004,
            travel_cap=0.12,
            contact_force_threshold=12.0,
        )
        base.update(overrides)
        return GuardedDescentConfig(**base)  # type: ignore[arg-type]

    def _run(self, ctrl: GuardedDescentController, substeps: int = 5) -> list:
        steps = []
        pos = np.array([0.15, -0.05, 0.60])
        force = np.array([19.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(120):
            t = i * 0.275
            if t < 19.0:
                pos = pos + np.array([0.0, 0.0, -0.01])
            steps.append(ctrl.cycle(t, pos, quat, force, substeps=substeps))
        return steps

    def test_specialist_off_uses_scripted_descent(self) -> None:
        """With specialist_enabled False the descent is the scripted GuardedDescent."""
        chunk = np.tile(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]), (8, 1))
        prov = _ChunkProvider(chunk)
        ctrl = GuardedDescentController(
            self._config(specialist_enabled=False), specialist_provider=prov
        )
        self._run(ctrl)
        self.assertTrue(ctrl.active)
        self.assertIsInstance(ctrl.descent, GuardedDescent)
        self.assertEqual(prov.calls, 0)  # specialist never queried
        self.assertEqual(ctrl._bearing_source, "motion-axis")

    def test_specialist_off_matches_no_provider_byte_for_byte(self) -> None:
        """Passing a specialist provider but leaving it off yields identical targets."""
        chunk = np.tile(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]), (8, 1))
        ctrl_off = GuardedDescentController(
            self._config(specialist_enabled=False),
            specialist_provider=_ChunkProvider(chunk),
        )
        ctrl_bare = GuardedDescentController(self._config(specialist_enabled=False))
        steps_off = self._run(ctrl_off)
        steps_bare = self._run(ctrl_bare)
        for a, b in zip(steps_off, steps_bare):
            if a.targets is None:
                self.assertIsNone(b.targets)
            else:
                np.testing.assert_allclose(a.targets, b.targets)

    def test_specialist_on_engages_specialist_descent(self) -> None:
        """With specialist_enabled True the stall hands off to SpecialistDescent."""
        chunk = np.tile(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]), (8, 1))
        prov = _ChunkProvider(chunk)
        ctrl = GuardedDescentController(
            self._config(specialist_enabled=True), specialist_provider=prov
        )
        steps = self._run(ctrl)
        self.assertTrue(ctrl.active)
        self.assertIsInstance(ctrl.descent, SpecialistDescent)
        self.assertEqual(ctrl._bearing_source, "specialist")
        self.assertGreater(prov.calls, 0)
        # Exactly one handoff cycle, logged with a [specialist] HANDOFF line.
        self.assertEqual(sum(1 for s in steps if s.triggered_this_cycle), 1)
        idx = next(i for i, s in enumerate(steps) if s.triggered_this_cycle)
        self.assertTrue(
            any("[specialist] HANDOFF" in ln for ln in steps[idx].log_lines)
        )
        # After handoff, targets are commanded and are sub-stepped pose targets.
        post = steps[idx]
        assert post.targets is not None
        self.assertEqual(post.targets.shape[1], 7)
        # Pre-handoff cycles keep the learned path (targets None).
        self.assertTrue(all(s.targets is None for s in steps[:idx]))

    def test_specialist_on_without_provider_falls_back_to_scripted(self) -> None:
        """specialist_enabled but no provider wired stays on the scripted descent."""
        ctrl = GuardedDescentController(
            self._config(specialist_enabled=True), specialist_provider=None
        )
        self._run(ctrl)
        self.assertTrue(ctrl.active)
        self.assertIsInstance(ctrl.descent, GuardedDescent)
        self.assertEqual(ctrl._bearing_source, "motion-axis")


if __name__ == "__main__":
    unittest.main()
