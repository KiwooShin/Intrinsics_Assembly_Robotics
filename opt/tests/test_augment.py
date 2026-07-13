"""Unit tests for opt.augment (random-shift + proprioception dropout).

Pure-tensor logic runs on CPU (float32); the fp16/CUDA path is guarded so
``unittest discover`` stays green on a torch-less or GPU-less machine.
"""

from __future__ import annotations

import unittest

from opt.tests import _pathfix  # noqa: F401

try:
    import torch

    from opt import augment

    _HAS_TORCH = True
except Exception:  # noqa: BLE001 - torch absent -> skip this module
    _HAS_TORCH = False

_HAS_CUDA = _HAS_TORCH and torch.cuda.is_available()


def _gen(seed: int, device: str = "cpu") -> "torch.Generator":
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class RandomShiftTest(unittest.TestCase):
    """DrQ random-shift: bounds, determinism, dtype/shape preservation."""

    def test_shape_and_dtype_preserved(self) -> None:
        # Codebase shape: (B, cameras, channels, H, W).
        x = torch.rand(4, 3, 3, 16, 16, dtype=torch.float32)
        out = augment.random_shift(x, pad=4, generator=_gen(0))
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device, x.device)

    def test_pad_zero_is_identity(self) -> None:
        x = torch.rand(2, 3, 3, 8, 8)
        out = augment.random_shift(x, pad=0, generator=_gen(0))
        # No-op returns the input unchanged.
        self.assertTrue(torch.equal(out, x))

    def test_does_not_mutate_input(self) -> None:
        x = torch.rand(2, 3, 3, 12, 12)
        x0 = x.clone()
        _ = augment.random_shift(x, pad=4, generator=_gen(1))
        self.assertTrue(torch.equal(x, x0))

    def test_constant_image_is_preserved(self) -> None:
        # Replicate-pad + crop of a constant image is that same constant, so
        # no out-of-range garbage can appear (bounds sanity).
        x = torch.full((3, 3, 3, 10, 10), 0.7)
        out = augment.random_shift(x, pad=4, generator=_gen(2))
        self.assertTrue(torch.allclose(out, x))

    def test_shift_within_pad_bounds(self) -> None:
        # Column-ramp image: value at column c equals c (row-invariant). The
        # realized horizontal shift read from a non-edge column must satisfy
        # |shift| <= pad.
        h = w = 32
        pad = 8
        ramp = torch.arange(w, dtype=torch.float32).view(1, 1, 1, 1, w).expand(
            5, 1, 1, h, w
        ).contiguous()
        out = augment.random_shift(ramp, pad=pad, generator=_gen(3))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= w - 1))
        center = w // 2
        for m in range(ramp.shape[0]):
            shift = int(out[m, 0, 0, 0, center].item()) - center
            self.assertLessEqual(abs(shift), pad, f"image {m} shift {shift} > pad")

    def test_shifts_are_independent_per_image(self) -> None:
        h = w = 24
        pad = 6
        ramp = torch.arange(w, dtype=torch.float32).view(1, 1, 1, 1, w).expand(
            64, 1, 1, h, w
        ).contiguous()
        out = augment.random_shift(ramp, pad=pad, generator=_gen(4))
        center = w // 2
        shifts = {int(out[m, 0, 0, 0, center].item()) - center for m in range(64)}
        self.assertGreater(len(shifts), 1)  # not all images shifted identically

    def test_deterministic_under_seed(self) -> None:
        x = torch.rand(3, 3, 3, 16, 16)
        a = augment.random_shift(x, pad=4, generator=_gen(7))
        b = augment.random_shift(x, pad=4, generator=_gen(7))
        self.assertTrue(torch.equal(a, b))
        c = augment.random_shift(x, pad=4, generator=_gen(8))
        self.assertFalse(torch.equal(a, c))  # different seed -> different shift

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            augment.random_shift(torch.rand(2, 3, 3, 8, 8), pad=-1)
        with self.assertRaises(ValueError):
            augment.random_shift(torch.rand(8, 8), pad=4)  # <3 dims

    @unittest.skipUnless(_HAS_CUDA, "CUDA required for fp16 path")
    def test_fp16_cuda_preserves_dtype(self) -> None:
        x = torch.rand(4, 3, 3, 16, 16, dtype=torch.float16, device="cuda")
        out = augment.random_shift(x, pad=4, generator=_gen(0, "cuda"))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.device.type, "cuda")


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class ProprioDropoutTest(unittest.TestCase):
    """Proprioception dropout: probability statistics + determinism."""

    def test_shape_and_dtype_preserved(self) -> None:
        s = torch.randn(16, 7)
        out = augment.proprio_dropout(s, p=0.5, generator=_gen(0))
        self.assertEqual(out.shape, s.shape)
        self.assertEqual(out.dtype, s.dtype)

    def test_p_zero_is_identity(self) -> None:
        s = torch.randn(8, 7)
        out = augment.proprio_dropout(s, p=0.0, generator=_gen(0))
        self.assertTrue(torch.equal(out, s))

    def test_p_one_zeros_all(self) -> None:
        s = torch.randn(32, 7)
        out = augment.proprio_dropout(s, p=1.0, generator=_gen(0))
        self.assertTrue(torch.all(out == 0))

    def test_kept_rows_unchanged(self) -> None:
        s = torch.randn(200, 7)
        out = augment.proprio_dropout(s, p=0.5, generator=_gen(5))
        row_zeroed = (out == 0).all(dim=1)
        # Every non-zeroed row must equal the original exactly.
        self.assertTrue(torch.equal(out[~row_zeroed], s[~row_zeroed]))

    def test_probability_statistics(self) -> None:
        # Over a large batch the zeroed fraction approaches p.
        s = torch.ones(40000, 7)
        for p in (0.5, 0.8):
            out = augment.proprio_dropout(s, p=p, generator=_gen(11))
            zeroed = float((out == 0).all(dim=1).float().mean())
            self.assertAlmostEqual(zeroed, p, delta=0.02)

    def test_deterministic_under_seed(self) -> None:
        s = torch.randn(64, 7)
        a = augment.proprio_dropout(s, p=0.6, generator=_gen(3))
        b = augment.proprio_dropout(s, p=0.6, generator=_gen(3))
        self.assertTrue(torch.equal(a, b))

    def test_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            augment.proprio_dropout(torch.randn(4, 7), p=1.5)
        with self.assertRaises(ValueError):
            augment.proprio_dropout(torch.randn(4, 7), p=-0.1)
        with self.assertRaises(ValueError):
            augment.proprio_dropout(torch.randn(4, 7, 2), p=0.5)  # not 2-D


if __name__ == "__main__":
    unittest.main()
