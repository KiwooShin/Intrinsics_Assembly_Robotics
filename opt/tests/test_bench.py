"""Unit tests for opt.bench helpers (percentiles, sampler stats).

Torch import is guarded so the module skips cleanly on a torch-less interpreter.
The pure percentile/sampler-stats logic needs no GPU.
"""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401

try:
    from opt import bench

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class PercentileTest(unittest.TestCase):
    def test_median_odd(self) -> None:
        self.assertAlmostEqual(bench._percentile([3.0, 1.0, 2.0], 50), 2.0)

    def test_p90_interpolation(self) -> None:
        # 0..10 -> p90 = 9.0 with linear interpolation over 11 points.
        self.assertAlmostEqual(bench._percentile([float(i) for i in range(11)], 90), 9.0)

    def test_empty_and_single(self) -> None:
        self.assertEqual(bench._percentile([], 50), 0.0)
        self.assertEqual(bench._percentile([4.2], 90), 4.2)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class SamplerStatsTest(unittest.TestCase):
    def test_stats_from_manual_samples(self) -> None:
        s = bench.GpuSampler()
        s._util = [10.0, 30.0, 50.0]
        s._mem = [100.0, 200.0, 150.0]
        self.assertAlmostEqual(s.util_mean, 30.0)
        self.assertAlmostEqual(s.util_max, 50.0)
        self.assertAlmostEqual(s.mem_max, 200.0)

    def test_stats_empty_safe(self) -> None:
        s = bench.GpuSampler()
        self.assertEqual(s.util_mean, 0.0)
        self.assertEqual(s.util_max, 0.0)


if __name__ == "__main__":
    unittest.main()
