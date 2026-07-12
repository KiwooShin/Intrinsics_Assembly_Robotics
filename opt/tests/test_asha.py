"""Unit tests for opt.asha successive-halving math (CPU-only)."""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401
from opt import asha
from opt.config import TrialConfig, TrialResult


class RungBudgetsTest(unittest.TestCase):
    def test_geometric_budgets(self) -> None:
        self.assertEqual(asha.rung_budgets(2, 32, 3), [2, 6, 18, 32])

    def test_single_rung_when_min_equals_max(self) -> None:
        self.assertEqual(asha.rung_budgets(10, 10, 3), [10])

    def test_last_rung_is_max(self) -> None:
        b = asha.rung_budgets(4, 50, 2)
        self.assertEqual(b[-1], 50)
        self.assertEqual(b[0], 4)
        self.assertTrue(all(b[i] < b[i + 1] for i in range(len(b) - 1)))

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            asha.rung_budgets(2, 32, 1)
        with self.assertRaises(ValueError):
            asha.rung_budgets(0, 32, 3)
        with self.assertRaises(ValueError):
            asha.rung_budgets(32, 2, 3)


class SurvivorsTest(unittest.TestCase):
    def test_keeps_top_fraction_lower_better(self) -> None:
        scores = {0: 0.5, 1: 0.1, 2: 0.9, 3: 0.3, 4: 0.7, 5: 0.2}
        # ceil(6/3) = 2 survivors, best-first.
        self.assertEqual(asha.survivors(scores, eta=3), [1, 5])

    def test_higher_better(self) -> None:
        scores = {0: 0.5, 1: 0.1, 2: 0.9}
        self.assertEqual(asha.survivors(scores, eta=3, lower_is_better=False), [2])

    def test_always_keeps_at_least_one(self) -> None:
        self.assertEqual(asha.survivors({7: 1.0}, eta=3), [7])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            asha.survivors({}, eta=3)


class SampleConfigsTest(unittest.TestCase):
    def test_deterministic(self) -> None:
        space = {"lr": [1e-4, 3e-4, 1e-3], "k": [8, 16]}
        a = asha.sample_configs(5, space, seed=0)
        b = asha.sample_configs(5, space, seed=0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        for cfg in a:
            self.assertIn(cfg["lr"], space["lr"])
            self.assertIn(cfg["k"], space["k"])

    def test_different_seed_differs(self) -> None:
        space = {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "k": [4, 8, 16, 32]}
        self.assertNotEqual(
            asha.sample_configs(8, space, seed=0),
            asha.sample_configs(8, space, seed=1),
        )

    def test_empty_candidates_raise(self) -> None:
        with self.assertRaises(ValueError):
            asha.sample_configs(3, {"lr": []}, seed=0)


class LeaderboardTest(unittest.TestCase):
    def _mk(self, tid: int, err: float) -> TrialResult:
        cfg = TrialConfig(
            trial_id=tid, lr=3e-4, k=16, img=128, weight_decay=1e-4,
            ema_decay=0.0, seed=0,
        )
        return TrialResult(config=cfg, epochs_done=10, val_first_action=err, val_l1=err * 10)

    def test_sorted_best_first(self) -> None:
        rows = [self._mk(0, 0.005), self._mk(1, 0.003), self._mk(2, 0.009)]
        md = asha.render_leaderboard(rows, baseline=0.006)
        # Trial 1 (0.003) should appear before trial 2 in the table body.
        i1 = md.index("| 1 | 1 |")
        i2 = md.index("| 3 | 2 |")
        self.assertLess(i1, i2)
        self.assertIn("vs_baseline", md)
        self.assertIn("Baseline", md)

    def test_no_baseline_column(self) -> None:
        md = asha.render_leaderboard([self._mk(0, 0.005)])
        self.assertNotIn("vs_baseline", md)


if __name__ == "__main__":
    unittest.main()
