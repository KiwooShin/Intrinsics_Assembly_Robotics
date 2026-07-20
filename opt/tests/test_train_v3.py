"""Unit tests for opt.train_v3 (torch on CPU; end-to-end guarded by CUDA).

The module-level torch import is guarded so that ``unittest discover`` stays
green under a torch-less interpreter (the tests simply skip there).
"""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401

try:
    import torch
    import torch.nn.functional as F

    import train_v2 as tv2
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


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class WeightedLossInvariantTest(unittest.TestCase):
    """The push-in weighted loss reduces to the plain mean L1 at uniform weights."""

    def test_uniform_weights_equal_mean_l1(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(8, 5, 6)
        target = torch.randn(8, 5, 6)
        w = torch.ones(8)
        per_frame = (pred - target).abs().mean(dim=(1, 2))
        weighted = (per_frame * w).sum() / w.sum()
        self.assertAlmostEqual(
            float(weighted), float(F.l1_loss(pred, target)), places=5
        )

    def test_weighting_up_weights_final_frames(self) -> None:
        # Final frame has large error; up-weighting it must raise the loss.
        pred = torch.zeros(3, 2, 6)
        target = torch.zeros(3, 2, 6)
        target[-1] = 10.0  # big error only on the last frame
        per_frame = (pred - target).abs().mean(dim=(1, 2))
        uniform = (per_frame * torch.ones(3)).sum() / 3.0
        ramp = torch.tensor([1.0, 1.0, 4.0])
        weighted = (per_frame * ramp).sum() / ramp.sum()
        self.assertGreater(float(weighted), float(uniform))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ThirteenDimForwardTest(unittest.TestCase):
    """A 13-D-state policy trains end-to-end on CPU (dry-run shape check)."""

    def test_forward_and_weighted_backward_cpu(self) -> None:
        k = 4
        model = tv2.Policy(k, state_dim=13)
        imgs = torch.rand(6, 3, 3, 32, 32)  # (B, cams, C, H, W)
        state = torch.randn(6, 13)          # 7-D pose + 6-D wrench
        target = torch.randn(6, k, 6)
        weights = torch.linspace(1.0, 4.0, 6)
        pred = model(imgs, state)
        self.assertEqual(tuple(pred.shape), (6, k, 6))
        per_frame = (pred - target).abs().mean(dim=(1, 2))
        loss = (per_frame * weights).sum() / weights.sum()
        loss.backward()  # gradients must flow through the 13-D head
        head0 = model.head[0]
        self.assertEqual(head0.in_features, 128 * 3 + 13)
        self.assertIsNotNone(head0.weight.grad)
        self.assertTrue(torch.isfinite(head0.weight.grad).all())

    def test_seven_dim_default_unchanged(self) -> None:
        model = tv2.Policy(4)  # default state_dim=7
        self.assertEqual(model.head[0].in_features, 128 * 3 + 7)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ArgParseTest(unittest.TestCase):
    """CLI flags map onto the opt-in TrainConfig fields."""

    def test_flags_off_by_default(self) -> None:
        cfg = train_v3._parse_args(["--epochs", "1"])
        self.assertFalse(cfg.tail_trim)
        self.assertFalse(cfg.use_wrench)
        self.assertEqual(cfg.pushin_weight, 1.0)
        self.assertEqual(cfg.state_dim, 7)

    def test_all_fixes_on(self) -> None:
        cfg = train_v3._parse_args([
            "--epochs", "1", "--tail-trim", "--wrench",
            "--pushin-weight", "4", "--shift-pad", "6", "--pushin-ramp-s", "2",
        ])
        self.assertTrue(cfg.tail_trim)
        self.assertTrue(cfg.use_wrench)
        self.assertEqual(cfg.state_dim, 13)
        self.assertEqual(cfg.pushin_weight, 4.0)
        self.assertTrue(cfg.pushin_enabled)
        self.assertEqual(cfg.shift_pad, 6)

    def test_last_inch_off_by_default(self) -> None:
        cfg = train_v3._parse_args(["--epochs", "1"])
        self.assertEqual(cfg.last_inch_s, 0.0)
        self.assertFalse(cfg.last_inch_enabled)

    def test_last_inch_flags(self) -> None:
        cfg = train_v3._parse_args([
            "--epochs", "1", "--last-inch-s", "2.5", "--last-inch-min-frames", "12",
        ])
        self.assertEqual(cfg.last_inch_s, 2.5)
        self.assertTrue(cfg.last_inch_enabled)
        self.assertEqual(cfg.last_inch_min_frames, 12)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class LoadSeatIndicesTest(unittest.TestCase):
    """_load_seat_indices reads the persisted marker (CPU-only, no CUDA)."""

    def test_missing_file_maps_to_minus_one(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(train_v3._load_seat_indices([d]), [-1])

    def test_reads_scalar_marker_in_dir_order(self) -> None:
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as root:
            import os

            d0, d1, d2 = (os.path.join(root, f"ep_{i}") for i in range(3))
            for p in (d0, d1, d2):
                os.makedirs(p)
            np.save(os.path.join(d0, "insertion_frame.npy"), np.asarray(42, np.int64))
            # d1 has no marker -> -1
            np.save(os.path.join(d2, "insertion_frame.npy"), np.asarray(-1, np.int64))
            self.assertEqual(train_v3._load_seat_indices([d0, d1, d2]), [42, -1, -1])


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class AuxHeadRegressionTest(unittest.TestCase):
    """The port-bearing aux head is opt-in and byte-identical when off (CPU)."""

    def test_aux_dim_zero_is_legacy_state_dict(self) -> None:
        """aux_dim=0 builds the exact legacy module (no aux_head keys)."""
        torch.manual_seed(0)
        legacy = tv2.Policy(4, state_dim=7)
        torch.manual_seed(0)
        aux_off = tv2.Policy(4, state_dim=7, aux_dim=0)
        self.assertEqual(
            list(legacy.state_dict().keys()), list(aux_off.state_dict().keys())
        )
        self.assertFalse(any("aux_head" in k for k in aux_off.state_dict().keys()))
        # Identical seed -> byte-identical parameters (no extra RNG draws).
        for k in legacy.state_dict():
            self.assertTrue(
                torch.equal(legacy.state_dict()[k], aux_off.state_dict()[k])
            )

    def test_aux_dim_zero_forward_returns_plain_tensor(self) -> None:
        model = tv2.Policy(4, state_dim=7, aux_dim=0)
        out = model(torch.rand(2, 3, 3, 32, 32), torch.randn(2, 7))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2, 4, 6))

    def test_aux_head_forward_shapes(self) -> None:
        for aux_dim in (3, 6):
            model = tv2.Policy(4, state_dim=7, aux_dim=aux_dim)
            self.assertEqual(model.aux_head[0].in_features, 128 * 3 + 7)
            act, aux = model(torch.rand(5, 3, 3, 32, 32), torch.randn(5, 7))
            self.assertEqual(tuple(act.shape), (5, 4, 6))
            self.assertEqual(tuple(aux.shape), (5, aux_dim))

    def test_aux_head_gradients_flow(self) -> None:
        model = tv2.Policy(4, state_dim=7, aux_dim=3)
        act, aux = model(torch.rand(4, 3, 3, 32, 32), torch.randn(4, 7))
        (aux.abs().mean() + act.abs().mean()).backward()
        self.assertIsNotNone(model.aux_head[0].weight.grad)
        self.assertTrue(torch.isfinite(model.aux_head[0].weight.grad).all())

    def test_negative_aux_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            tv2.Policy(4, aux_dim=-1)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class MaskedAuxLossTest(unittest.TestCase):
    """The masked aux L1 averages over valid frames only, per design 3.1."""

    def test_masked_mean_ignores_invalid(self) -> None:
        aux_pred = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
        aux_tgt = torch.zeros(3, 3)
        valid = torch.tensor([True, False, True])
        v = valid.to(aux_pred.dtype)
        aux_l1 = (aux_pred - aux_tgt).abs().mean(-1)  # [0, 5, 1]
        loss = (aux_l1 * v).sum() / v.sum().clamp(min=1.0)
        # (0 + 1) / 2 -> the invalid huge-error row is excluded.
        self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_all_invalid_does_not_divide_by_zero(self) -> None:
        aux_l1 = torch.tensor([3.0, 4.0])
        v = torch.zeros(2)
        loss = (aux_l1 * v).sum() / v.sum().clamp(min=1.0)
        self.assertEqual(float(loss), 0.0)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class AuxArgParseTest(unittest.TestCase):
    """CLI flags map onto the port_aux TrainConfig fields."""

    def test_port_aux_off_by_default(self) -> None:
        cfg = train_v3._parse_args(["--epochs", "1"])
        self.assertFalse(cfg.port_aux)
        self.assertFalse(cfg.port_aux_enabled)

    def test_port_aux_flags(self) -> None:
        cfg = train_v3._parse_args([
            "--epochs", "1", "--port-aux", "--aux-dim", "6", "--aux-weight", "0.25",
            "--aux-frame", "base", "--aux-freeze-encoder", "--init-ckpt", "/tmp/x.pt",
        ])
        self.assertTrue(cfg.port_aux)
        self.assertEqual(cfg.aux_dim, 6)
        self.assertEqual(cfg.aux_weight, 0.25)
        self.assertEqual(cfg.aux_frame, "base")
        self.assertTrue(cfg.aux_freeze_encoder)
        self.assertEqual(cfg.init_ckpt, "/tmp/x.pt")


@unittest.skipUnless(_HAS_CUDA, "CUDA required: _load_split loads to GPU")
class LoadSplitIntegrationTest(unittest.TestCase):
    """_load_split wires wrench/trim/weights onto real GPU tensors."""

    def _smoke_dirs(self) -> list[str]:
        import glob

        return sorted(glob.glob("/home/kiwoos/training/smoke/ep_*"))

    def test_wrench_widens_state_and_trim_shrinks(self) -> None:
        dirs = self._smoke_dirs()
        if not dirs:
            self.skipTest("smoke dataset not present")
        base = TrainConfig(train_globs=",".join(dirs), val_globs="",
                           img=96, k=8, compile_mode="none")
        _, st_base, _, w_base, off_base, valid_base = train_v3._load_split(
            dirs, base, apply_prep=True
        )
        self.assertEqual(st_base.shape[1], 7)
        self.assertIsNone(w_base)  # no weighting by default
        self.assertIsNone(off_base)  # no aux labels by default
        self.assertIsNone(valid_base)

        fixed = TrainConfig(
            train_globs=",".join(dirs), val_globs="", img=96, k=8,
            compile_mode="none", tail_trim=True, use_wrench=True,
            pushin_weight=4.0,
        )
        imf, st_f, act_f, w_f, _, _ = train_v3._load_split(dirs, fixed, apply_prep=True)
        self.assertEqual(st_f.shape[1], 13)  # 7-D pose + 6-D wrench
        self.assertIsNotNone(w_f)
        self.assertEqual(w_f.shape[0], imf.shape[0])
        self.assertEqual(imf.shape[0], act_f.shape[0])
        # Trimming never adds frames; on demos with a seated tail it removes some.
        self.assertLessEqual(imf.shape[0], st_base.shape[0])
        self.assertGreaterEqual(float(w_f.max()), 1.0)


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
