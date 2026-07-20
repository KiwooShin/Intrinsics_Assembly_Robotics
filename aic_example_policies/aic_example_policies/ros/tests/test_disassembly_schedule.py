#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Unit tests for the ROS-free DisassembleCode perturbed-retract schedule.

Exercises :mod:`aic_example_policies.ros.disassembly_schedule` -- the pure geometry
of the perturbed retract (spiral radius band, lift-translate-reseat axial-then-
lateral ordering, config validation). Requires neither ROS, Gazebo, torch, nor a
GPU.

Run with::

    PYTHONPATH=aic_example_policies python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from aic_example_policies.ros import disassembly_schedule as ds


def _spiral_cfg(**kw) -> ds.RetractScheduleConfig:
    """Build a spiral RetractScheduleConfig with test-friendly defaults."""
    base = dict(
        n_steps=60,
        axial_span_m=0.025,
        radius_max_m=0.003,
        turns=3.0,
        azimuth0_rad=0.7,
        roll_amp_rad=0.03,
        pitch_amp_rad=0.04,
    )
    base.update(kw)
    return ds.RetractScheduleConfig(**base)


class DrawRadiusMaxTest(unittest.TestCase):
    """The log-spaced spiral radius draw stays inside the configured band."""

    def test_all_draws_in_band(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = ds.RADIUS_MIN_M, ds.RADIUS_MAX_M
        draws = [ds.draw_radius_max(rng, lo, hi) for _ in range(2000)]
        self.assertTrue(all(lo <= r <= hi for r in draws))
        # In-band: 0.2 mm .. 3.0 mm.
        self.assertGreaterEqual(min(draws), 0.2e-3)
        self.assertLessEqual(max(draws), 3.0e-3)

    def test_log_spacing_concentrates_mass_below_geometric_span(self) -> None:
        # A log-uniform over [lo, hi] has ~50% of its mass below the geometric mean.
        rng = np.random.default_rng(1)
        lo, hi = ds.RADIUS_MIN_M, ds.RADIUS_MAX_M
        gmean = math.sqrt(lo * hi)
        draws = np.array([ds.draw_radius_max(rng, lo, hi) for _ in range(5000)])
        frac_below = float(np.mean(draws < gmean))
        self.assertAlmostEqual(frac_below, 0.5, delta=0.05)

    def test_inverted_band_raises(self) -> None:
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            ds.draw_radius_max(rng, 0.003, 0.001)
        with self.assertRaises(ValueError):
            ds.draw_radius_max(rng, 0.0, 0.003)


class SpiralOffsetsTest(unittest.TestCase):
    """Archimedean spiral-out geometry."""

    def test_shape_and_columns(self) -> None:
        off = ds.spiral_offsets(_spiral_cfg(n_steps=40))
        self.assertEqual(off.shape, (40, 5))

    def test_axial_is_monotone_and_spans_full_range(self) -> None:
        cfg = _spiral_cfg()
        off = ds.spiral_offsets(cfg)
        axial = off[:, 0]
        self.assertTrue(np.all(np.diff(axial) >= -1e-12))  # non-decreasing
        self.assertAlmostEqual(axial[0], 0.0)
        self.assertAlmostEqual(axial[-1], cfg.axial_span_m)

    def test_lateral_radius_within_band_and_grows(self) -> None:
        cfg = _spiral_cfg(radius_max_m=0.0021)
        off = ds.spiral_offsets(cfg)
        radius = np.hypot(off[:, 1], off[:, 2])
        # Every waypoint's lateral radius is within [0, radius_max].
        self.assertLessEqual(float(radius.max()), cfg.radius_max_m + 1e-12)
        self.assertGreaterEqual(float(radius.min()), 0.0)
        # The peak is reached at the outermost step and lands in the 0.2-3.0 mm band.
        self.assertAlmostEqual(float(radius[-1]), cfg.radius_max_m, places=9)
        self.assertTrue(0.2e-3 <= radius.max() <= 3.0e-3)

    def test_azimuth_advances_configured_turns(self) -> None:
        cfg = _spiral_cfg(turns=3.0, roll_amp_rad=0.0, pitch_amp_rad=0.0)
        off = ds.spiral_offsets(cfg)
        # Unwrapped azimuth of the lateral vector spans ~turns full turns (drop the
        # r=0 start point, whose arctan2 is undefined; the span is then over n-2 of
        # the n-1 intervals, so allow ~0.1-turn slack).
        ang = np.unwrap(np.arctan2(off[1:, 2], off[1:, 1]))
        turns_measured = float(ang[-1] - ang[0]) / (2.0 * math.pi)
        self.assertAlmostEqual(turns_measured, cfg.turns, delta=0.15)

    def test_roll_pitch_bounded_by_amplitude(self) -> None:
        cfg = _spiral_cfg(roll_amp_rad=0.03, pitch_amp_rad=0.05)
        off = ds.spiral_offsets(cfg)
        self.assertLessEqual(float(np.abs(off[:, 3]).max()), 0.03 + 1e-12)
        self.assertLessEqual(float(np.abs(off[:, 4]).max()), 0.05 + 1e-12)


class LiftTranslateReseatTest(unittest.TestCase):
    """The recovery variant clears the bore axially BEFORE translating laterally."""

    def test_axial_precedes_lateral(self) -> None:
        cfg = _spiral_cfg(
            n_steps=60, lift_translate=True, lift_axial_m=0.004, lift_lateral_m=0.006
        )
        off = ds.lift_translate_reseat_offsets(cfg)
        radius = np.hypot(off[:, 1], off[:, 2])
        moved_lat = np.nonzero(radius > 1e-9)[0]
        self.assertGreater(moved_lat.size, 0, "variant must translate laterally")
        first_lat = int(moved_lat[0])
        # At the first lateral motion the bore is already cleared: axial has reached
        # the full lift height (segment 1 completed before segment 2 begins).
        self.assertGreaterEqual(off[first_lat, 0], cfg.lift_axial_m - 1e-9)
        # And no lateral offset exists before that index.
        self.assertTrue(np.all(radius[:first_lat] <= 1e-9))

    def test_segments_reach_expected_extents(self) -> None:
        cfg = _spiral_cfg(
            n_steps=90, lift_translate=True, lift_axial_m=0.004, lift_lateral_m=0.007
        )
        off = ds.lift_translate_reseat_offsets(cfg)
        radius = np.hypot(off[:, 1], off[:, 2])
        # Lateral translate reaches lift_lateral_m; axial reaches the full span.
        self.assertAlmostEqual(float(radius.max()), cfg.lift_lateral_m, places=6)
        self.assertAlmostEqual(float(off[:, 0].max()), cfg.axial_span_m, places=6)

    def test_axial_non_decreasing(self) -> None:
        cfg = _spiral_cfg(n_steps=45, lift_translate=True)
        off = ds.lift_translate_reseat_offsets(cfg)
        self.assertTrue(np.all(np.diff(off[:, 0]) >= -1e-12))


class BuildScheduleTest(unittest.TestCase):
    """Dispatch and sampling behaviour."""

    def test_dispatch_matches_variant(self) -> None:
        spiral = _spiral_cfg(lift_translate=False)
        recov = _spiral_cfg(lift_translate=True)
        np.testing.assert_allclose(
            ds.build_schedule(spiral), ds.spiral_offsets(spiral)
        )
        np.testing.assert_allclose(
            ds.build_schedule(recov), ds.lift_translate_reseat_offsets(recov)
        )

    def test_sample_config_is_seed_deterministic(self) -> None:
        a = ds.sample_config(np.random.default_rng(42), 50)
        b = ds.sample_config(np.random.default_rng(42), 50)
        self.assertEqual(a, b)

    def test_sample_config_draws_in_bands(self) -> None:
        d = ds.ScheduleDefaults()
        n_lift = 0
        for s in range(200):
            cfg = ds.sample_config(np.random.default_rng(s), 50, d)
            self.assertTrue(d.radius_min_m <= cfg.radius_max_m <= d.radius_max_m)
            self.assertTrue(0.0 <= cfg.azimuth0_rad < 2.0 * math.pi)
            self.assertTrue(
                d.roll_pitch_min_rad <= cfg.roll_amp_rad <= d.roll_pitch_max_rad
            )
            n_lift += int(cfg.lift_translate)
        # ~20% lift-translate-reseat variants (Bernoulli(lift_frac=0.2)).
        self.assertTrue(0.10 <= n_lift / 200.0 <= 0.30)


class ConfigValidationTest(unittest.TestCase):
    """Input validation on the config dataclasses."""

    def test_n_steps_too_small_raises(self) -> None:
        with self.assertRaises(ValueError):
            _spiral_cfg(n_steps=1)

    def test_lift_axial_must_be_less_than_span(self) -> None:
        with self.assertRaises(ValueError):
            _spiral_cfg(lift_translate=True, lift_axial_m=0.03, axial_span_m=0.025)

    def test_negative_amplitudes_raise(self) -> None:
        with self.assertRaises(ValueError):
            _spiral_cfg(roll_amp_rad=-0.01)

    def test_defaults_reject_inverted_band(self) -> None:
        with self.assertRaises(ValueError):
            ds.ScheduleDefaults(radius_min_m=0.003, radius_max_m=0.001)
        with self.assertRaises(ValueError):
            ds.ScheduleDefaults(lift_frac=1.5)


if __name__ == "__main__":
    unittest.main()
