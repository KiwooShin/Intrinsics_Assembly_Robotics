"""Unit tests for verify_dataset.py structural checks.

Includes a regression test for the zero-duration divide-by-zero bug. Uses small
synthetic .npy arrays; runs without ROS/Gazebo/GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import verify_dataset as vd


def _write_dataset(d: Path, n: int, timestamps: np.ndarray | None = None,
                   truncate: dict[str, int] | None = None) -> None:
    truncate = truncate or {}
    rng = np.random.default_rng(0)
    fields = {
        "center_images": rng.integers(0, 256, (n, 4, 5, 3), dtype=np.uint8),
        "left_images": rng.integers(0, 256, (n, 4, 5, 3), dtype=np.uint8),
        "right_images": rng.integers(0, 256, (n, 4, 5, 3), dtype=np.uint8),
        "tcp_velocities": rng.standard_normal((n, 6)).astype(np.float32),
        "tcp_poses": rng.standard_normal((n, 7)).astype(np.float32),
        "timestamps": (timestamps if timestamps is not None
                       else np.arange(n, dtype=np.float64) * 0.1),
    }
    for name, arr in fields.items():
        m = truncate.get(name)
        np.save(d / f"{name}.npy", arr[:m] if m is not None else arr)


class TestVerifyDataset(unittest.TestCase):
    def test_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_dataset(d, 10)
            report = vd.verify_dataset(str(d))
            self.assertEqual(report.n_frames, 10)
            self.assertTrue(report.length_match)
            self.assertTrue(report.monotonic)
            self.assertAlmostEqual(report.duration, 0.9, places=5)
            self.assertIsNotNone(report.rate_hz)

    def test_zero_duration_no_crash(self) -> None:
        # Regression: all-equal timestamps used to raise ZeroDivisionError.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_dataset(d, 6, timestamps=np.full(6, 42.0))
            report = vd.verify_dataset(str(d))
            self.assertEqual(report.duration, 0.0)
            self.assertIsNone(report.rate_hz)
            self.assertTrue(report.monotonic)  # non-decreasing (all equal)

    def test_non_monotonic_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ts = np.array([0.0, 0.3, 0.1, 0.5, 0.2], dtype=np.float64)
            _write_dataset(d, 5, timestamps=ts)
            report = vd.verify_dataset(str(d))
            self.assertFalse(report.monotonic)

    def test_length_mismatch_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_dataset(d, 8, truncate={"tcp_poses": 5})
            report = vd.verify_dataset(str(d))
            self.assertFalse(report.length_match)

    def test_missing_center_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                vd.verify_dataset(tmp)

    def test_load_arrays_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.save(d / "center_images.npy",
                    np.zeros((2, 4, 5, 3), dtype=np.uint8))
            arr = vd.load_arrays(str(d))
            self.assertIn("center_images", arr)
            self.assertNotIn("tcp_poses", arr)


if __name__ == "__main__":
    unittest.main()
