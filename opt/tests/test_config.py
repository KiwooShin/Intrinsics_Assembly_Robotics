"""Unit tests for opt.config dataclasses and validation (CPU-only)."""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401  (path side effect)
from opt import config


class TrainConfigTest(unittest.TestCase):
    """Validation and derived properties of TrainConfig."""

    def test_defaults_match_v2_baseline(self) -> None:
        c = config.TrainConfig()
        self.assertEqual(c.k, 16)
        self.assertEqual(c.img, 128)
        self.assertEqual(c.bs, 256)
        self.assertAlmostEqual(c.lr, 3e-4)
        self.assertFalse(c.ema_enabled)

    def test_ema_enabled_property(self) -> None:
        self.assertTrue(config.TrainConfig(ema_decay=0.999).ema_enabled)
        self.assertFalse(config.TrainConfig(ema_decay=0.0).ema_enabled)

    def test_frozen(self) -> None:
        c = config.TrainConfig()
        with self.assertRaises(Exception):
            c.lr = 1.0  # type: ignore[misc]

    def test_bad_epochs(self) -> None:
        with self.assertRaises(ValueError):
            config.TrainConfig(epochs=0)

    def test_bad_lr(self) -> None:
        with self.assertRaises(ValueError):
            config.TrainConfig(lr=0.0)

    def test_img_multiple_of_8(self) -> None:
        with self.assertRaises(ValueError):
            config.TrainConfig(img=100)
        # 96 and 128 are valid.
        config.TrainConfig(img=96)
        config.TrainConfig(img=128)

    def test_bad_ema(self) -> None:
        with self.assertRaises(ValueError):
            config.TrainConfig(ema_decay=1.0)
        with self.assertRaises(ValueError):
            config.TrainConfig(ema_decay=-0.1)


class ResultRecordsTest(unittest.TestCase):
    """Result dataclasses expose the expected derived fields."""

    def test_infer_hz(self) -> None:
        r = config.InferBenchResult(
            label="eager-bf16", batch=1, dtype="bf16", compiled=False,
            includes_h2d=True, iters=100, mean_ms=5.0, p50_ms=4.8, p90_ms=6.0,
            std_ms=0.3,
        )
        self.assertAlmostEqual(r.hz, 200.0)

    def test_trial_result_history_init(self) -> None:
        tc = config.TrialConfig(
            trial_id=0, lr=3e-4, k=16, img=128, weight_decay=1e-4,
            ema_decay=0.0, seed=0,
        )
        tr = config.TrialResult(config=tc)
        self.assertEqual(tr.history, [])
        self.assertTrue(tr.alive)


if __name__ == "__main__":
    unittest.main()
