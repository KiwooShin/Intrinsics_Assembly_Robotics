#  Copyright (C) 2026 Intrinsic Innovation LLC  (Apache-2.0)
#
"""Offline smoke test for the deployed ACT-lite policy checkpoint.

This test validates that the checkpoint trained by ``train_v2.py`` can be loaded
and run for inference in the deployment interpreter *without* ROS, Gazebo, or the
full ``DeployACT`` node (which pulls in ``rclpy`` and the aic message packages).

It reconstructs the exact network architecture that ``DeployACT._Policy`` and
``train_v2.Policy`` define (a 3-camera CNN encoder + MLP head emitting a K-step,
6-D twist chunk), loads the real checkpoint's ``state_dict`` with ``strict=True``
(which fails loudly if the architecture ever drifts from the checkpoint), feeds a
dummy observation, and asserts a finite ``(1, K, 6)`` action chunk comes out both
on CPU and — when a CUDA device is present — on the GPU.

Run with::

    ~/venvs/aic-deploy/bin/python -m unittest discover \
        -s aic_example_policies/aic_example_policies/ros/tests -v
"""

from __future__ import annotations

import os
import pathlib
import unittest

try:
    import torch
    import torch.nn as nn

    _TORCH_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - environment guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = str(exc)


CKPT_PATH = pathlib.Path(
    os.environ.get("AIC_CKPT", "/home/kiwoos/training/ckpt/v2_wide.pt")
)


if torch is not None:

    class _Encoder(nn.Module):
        """Per-camera CNN encoder (mirrors ``DeployACT._Encoder``)."""

        def __init__(self, out: int = 128) -> None:
            """Build the 4-block strided-conv encoder.

            Args:
                out: Output feature dimension per camera image.
            """
            super().__init__()

            def blk(i: int, o: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU()
                )

            self.net = nn.Sequential(
                blk(3, 32),
                blk(32, 64),
                blk(64, 128),
                blk(128, out),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode a batch of images to feature vectors."""
            return self.net(x)

    class _Policy(nn.Module):
        """3-camera ACT-lite policy (mirrors ``DeployACT._Policy``)."""

        def __init__(self, k: int, state_dim: int = 7, feat: int = 128) -> None:
            """Build encoder + MLP head.

            Args:
                k: Action-chunk length (number of predicted future steps).
                state_dim: Dimension of the proprioceptive state (TCP pose = 7).
                feat: Per-camera feature dimension.
            """
            super().__init__()
            self.k = k
            self.enc = _Encoder(feat)
            self.head = nn.Sequential(
                nn.Linear(feat * 3 + state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, k * 6),
            )

        def forward(
            self, imgs: "torch.Tensor", state: "torch.Tensor"
        ) -> "torch.Tensor":
            """Predict a (B, K, 6) twist chunk from images and state.

            Args:
                imgs: Image tensor of shape ``(B, 3, 3, H, W)``.
                state: State tensor of shape ``(B, state_dim)``.

            Returns:
                Action chunk tensor of shape ``(B, K, 6)``.
            """
            b = imgs.shape[0]
            f = self.enc(imgs.view(b * 3, *imgs.shape[2:])).view(b, -1)
            return self.head(torch.cat([f, state], 1)).view(b, self.k, 6)


def _load_model(device: "torch.device") -> "tuple[_Policy, dict]":
    """Load the checkpoint and reconstruct the model on ``device``.

    Args:
        device: Torch device to place the model and normalization stats on.

    Returns:
        A tuple ``(model, ckpt)`` where ``model`` is in eval mode with the
        checkpoint weights loaded and ``ckpt`` is the raw checkpoint dict.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = _Policy(ckpt["K"]).to(device)
    model.load_state_dict(ckpt["model"])  # strict=True by default
    model.eval()
    return model, ckpt


@unittest.skipUnless(torch is not None, f"torch not importable: {_TORCH_IMPORT_ERROR}")
class DeployActOfflineTest(unittest.TestCase):
    """Checkpoint load + dummy-inference smoke tests."""

    def _run_inference(self, device: "torch.device") -> None:
        """Load the checkpoint and assert a finite denormalized chunk on device."""
        model, ckpt = _load_model(device)
        k = ckpt["K"]
        imgs = torch.rand(1, 3, 3, 128, 128, device=device)
        state = torch.randn(1, 7, device=device)
        smean, sstd = ckpt["smean"].to(device), ckpt["sstd"].to(device)
        amean, astd = ckpt["amean"].to(device), ckpt["astd"].to(device)
        state_n = (state - smean) / sstd
        with torch.inference_mode():
            pred = model(imgs, state_n)
        self.assertEqual(tuple(pred.shape), (1, k, 6))
        self.assertEqual(pred.device.type, device.type)
        act = pred[0] * astd + amean  # denormalize to physical twist units
        self.assertTrue(torch.isfinite(act).all(), "non-finite action produced")

    def test_checkpoint_infers_on_cpu(self) -> None:
        """The checkpoint loads and produces a finite (K, 6) chunk on CPU."""
        self._run_inference(torch.device("cpu"))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA not available"
    )
    def test_checkpoint_infers_on_gpu(self) -> None:
        """The checkpoint loads and produces a finite (K, 6) chunk on the GPU."""
        self._run_inference(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
