"""Unit tests for opt.train_v3 (torch on CPU; end-to-end guarded by CUDA).

The module-level torch import is guarded so that ``unittest discover`` stays
green under a torch-less interpreter (the tests simply skip there).
"""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401

try:
    import torch

    from opt import train_v3
    from opt.config import TrainConfig

    _HAS_TORCH = True
except Exception:  # noqa: BLE001 - torch absent -> skip this module
    _HAS_TORCH = False

_HAS_CUDA = _HAS_TORCH and torch.cuda.is_available()


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class GlobExpandTest(unittest.TestCase):
    def test_dedup_and_sort(self) -> None:
        # Non-matching globs expand to nothing; empty parts are ignored.
        self.assertEqual(train_v3._expand_globs("/no/such/path_*, "), [])


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class EmaWeightsTest(unittest.TestCase):
    """EMA math runs correctly on CPU without a GPU."""

    def _model(self) -> "torch.nn.Module":
        m = torch.nn.Linear(3, 2)
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.fill_(0.0)
        return m

    def test_bad_decay(self) -> None:
        with self.assertRaises(ValueError):
            train_v3.EmaWeights(self._model(), decay=1.0)

    def test_update_moves_toward_params(self) -> None:
        m = self._model()
        ema = train_v3.EmaWeights(m, decay=0.5)
        with torch.no_grad():
            m.weight.fill_(3.0)  # shadow was 1.0
        ema.update(m)
        # shadow = 0.5*1 + 0.5*3 = 2.0
        self.assertAlmostEqual(float(ema._shadow["weight"].mean()), 2.0, places=5)

    def test_apply_and_restore(self) -> None:
        m = self._model()
        ema = train_v3.EmaWeights(m, decay=0.5)
        with torch.no_grad():
            m.weight.fill_(3.0)
        ema.update(m)  # shadow -> 2.0
        ema.apply_to(m)
        self.assertAlmostEqual(float(m.weight.detach().mean()), 2.0, places=5)
        ema.restore(m)
        self.assertAlmostEqual(float(m.weight.detach().mean()), 3.0, places=5)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class OptimizerTest(unittest.TestCase):
    def test_build_optimizer_cpu_falls_back(self) -> None:
        m = torch.nn.Linear(3, 2)
        # fused=True requested but CPU has no CUDA -> plain/foreach AdamW.
        opt = train_v3.build_optimizer(m, lr=1e-3, weight_decay=1e-4, fused=True)
        self.assertIsInstance(opt, torch.optim.AdamW)


@unittest.skipUnless(_HAS_CUDA, "CUDA required for end-to-end training")
class EndToEndTrainTest(unittest.TestCase):
    """A tiny real training run on the smoke dataset (GPU only)."""

    def test_overfit_smoke_runs(self) -> None:
        import glob

        eps = sorted(glob.glob("/home/kiwoos/training/smoke/ep_*"))
        if not eps:
            self.skipTest("smoke dataset not present")
        cfg = TrainConfig(
            train_globs=",".join(eps), val_globs="", epochs=3, bs=128,
            img=96, k=8, compile_mode="none", ema_decay=0.0, out="",
        )
        res = train_v3.train(cfg)
        self.assertGreater(res.throughput_fps, 0.0)
        self.assertTrue(res.best_val_first_action < float("inf"))
        self.assertFalse(res.fp8_active)  # torchao absent -> bf16 fallback


if __name__ == "__main__":
    unittest.main()
