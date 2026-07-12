"""Unit tests for train_v2.py dataset math, model shape and checkpoint I/O.

Forces CPU (never touches the GPU) and is skipped entirely when torch is absent, so
it runs on the system interpreter (skipped) and the conda interpreter (executed).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Force CPU before torch initialises a CUDA context; keeps the GPU untouched.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    import train_v2


def _write_episode(d: Path, n: int, h: int = 8, w: int = 10,
                   vel_start: float = 0.0) -> None:
    rng = np.random.default_rng(int(vel_start) + 1)
    for name in ("center_images", "left_images", "right_images"):
        np.save(d / f"{name}.npy", rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8))
    np.save(d / "tcp_poses.npy", rng.standard_normal((n, 7)).astype(np.float32))
    vel = (np.arange(n * 6).reshape(n, 6).astype(np.float32) + vel_start)
    np.save(d / "tcp_velocities.npy", vel)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestActionChunks(unittest.TestCase):
    def test_shape_and_alignment(self) -> None:
        n, k = 4, 3
        vel = np.arange(n * 6).reshape(n, 6).astype(np.float32)
        ch = train_v2.build_action_chunks(vel, k)
        self.assertEqual(ch.shape, (n, k, 6))
        for t in range(n):
            for j in range(k):
                np.testing.assert_array_equal(ch[t, j], vel[min(t + j, n - 1)])

    def test_last_frame_is_padded_with_last_action(self) -> None:
        n, k = 5, 4
        vel = np.arange(n * 6).reshape(n, 6).astype(np.float32)
        ch = train_v2.build_action_chunks(vel, k)
        for j in range(k):
            np.testing.assert_array_equal(ch[n - 1, j], vel[n - 1])

    def test_k_one_equals_actions(self) -> None:
        vel = np.arange(3 * 6).reshape(3, 6).astype(np.float32)
        ch = train_v2.build_action_chunks(vel, 1)
        np.testing.assert_array_equal(ch, vel[:, None, :])

    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            train_v2.build_action_chunks(np.zeros((3, 6), np.float32), 0)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestNormStats(unittest.TestCase):
    def test_mean_std_correct(self) -> None:
        state = torch.randn(20, 7)
        act = torch.randn(20, 5, 6)
        ns = train_v2.compute_norm_stats(state, act, eps=1e-6)
        self.assertEqual(tuple(ns.smean.shape), (7,))
        self.assertEqual(tuple(ns.amean.shape), (6,))
        torch.testing.assert_close(ns.smean, state.mean(0))
        torch.testing.assert_close(ns.sstd, state.std(0) + 1e-6)
        torch.testing.assert_close(ns.amean, act.reshape(-1, 6).mean(0))
        torch.testing.assert_close(ns.astd, act.reshape(-1, 6).std(0) + 1e-6)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestPolicyForward(unittest.TestCase):
    def test_forward_shape_cpu(self) -> None:
        k = 4
        m = train_v2.Policy(k).to("cpu").eval()
        imgs = torch.randn(2, 3, 3, 32, 32)
        state = torch.randn(2, 7)
        with torch.no_grad():
            out = m(imgs, state)
        self.assertEqual(tuple(out.shape), (2, k, 6))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestLoadAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train_v2.DEV = "cpu"  # ensure tensors stay on CPU

    def test_shapes_dtypes_and_epid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d0 = Path(tmp) / "ep_0"; d0.mkdir()
            d1 = Path(tmp) / "ep_1"; d1.mkdir()
            _write_episode(d0, 3, vel_start=0.0)
            _write_episode(d1, 3, vel_start=100.0)
            imgs, state, act, epid = train_v2.load_all([str(d0), str(d1)], 16, 4)
            self.assertEqual(tuple(imgs.shape), (6, 3, 3, 16, 16))
            self.assertEqual(imgs.dtype, torch.float16)
            self.assertEqual(tuple(state.shape), (6, 7))
            self.assertEqual(state.dtype, torch.float32)
            self.assertEqual(tuple(act.shape), (6, 4, 6))
            self.assertEqual(act.dtype, torch.float32)
            self.assertEqual(epid.tolist(), [0, 0, 0, 1, 1, 1])

    def test_action_windowing_matches_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d0 = Path(tmp) / "ep_0"; d0.mkdir()
            _write_episode(d0, 4, vel_start=0.0)
            _, _, act, _ = train_v2.load_all([str(d0)], 16, 3)
            vel = np.load(d0 / "tcp_velocities.npy")
            expected = train_v2.build_action_chunks(vel, 3)
            np.testing.assert_allclose(act.numpy(), expected)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestCheckpointRoundtrip(unittest.TestCase):
    def test_save_and_reload_norm_stats(self) -> None:
        ns = train_v2.NormStats(
            smean=torch.randn(7), sstd=torch.rand(7) + 1.0,
            amean=torch.randn(6), astd=torch.rand(6) + 1.0,
        )
        model_state = {"w": torch.randn(3, 3)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "ckpt.pt")  # exercises makedirs of parent
            train_v2.save_checkpoint(path, model_state, ns, k=4, img=16)
            self.assertTrue(os.path.exists(path))
            ckpt = torch.load(path, map_location="cpu")
            self.assertIn("model", ckpt)
            self.assertEqual(ckpt["K"], 4)
            self.assertEqual(ckpt["img"], 16)
            torch.testing.assert_close(ckpt["amean"], ns.amean)
            torch.testing.assert_close(ckpt["astd"], ns.astd)
            torch.testing.assert_close(ckpt["smean"], ns.smean)
            torch.testing.assert_close(ckpt["sstd"], ns.sstd)


if __name__ == "__main__":
    unittest.main()
