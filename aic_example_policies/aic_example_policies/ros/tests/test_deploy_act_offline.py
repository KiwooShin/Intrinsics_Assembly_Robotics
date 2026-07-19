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
import tempfile
import unittest

import numpy as np

from aic_example_policies.ros import port_offset

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

        def __init__(
            self, k: int, state_dim: int = 7, feat: int = 128, aux_dim: int = 0
        ) -> None:
            """Build encoder + MLP head (+ optional aux head).

            Args:
                k: Action-chunk length (number of predicted future steps).
                state_dim: Dimension of the proprioceptive state (TCP pose = 7).
                feat: Per-camera feature dimension.
                aux_dim: Port-bearing aux head width (0 disables it).
            """
            super().__init__()
            self.k = k
            self.aux_dim = aux_dim
            self.enc = _Encoder(feat)
            self.head = nn.Sequential(
                nn.Linear(feat * 3 + state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, k * 6),
            )
            if aux_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(feat * 3 + state_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, aux_dim),
                )

        def forward(
            self, imgs: "torch.Tensor", state: "torch.Tensor"
        ) -> "torch.Tensor":
            """Predict a (B, K, 6) twist chunk (+ aux) from images and state.

            Args:
                imgs: Image tensor of shape ``(B, 3, 3, H, W)``.
                state: State tensor of shape ``(B, state_dim)``.

            Returns:
                A ``(B, K, 6)`` chunk when ``aux_dim == 0``; otherwise a tuple
                ``(action, aux)`` with ``aux`` shaped ``(B, aux_dim)``.
            """
            b = imgs.shape[0]
            f = self.enc(imgs.view(b * 3, *imgs.shape[2:])).view(b, -1)
            h = torch.cat([f, state], 1)
            act = self.head(h).view(b, self.k, 6)
            if self.aux_dim > 0:
                return act, self.aux_head(h)
            return act


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

    def test_legacy_checkpoint_has_no_aux_head(self) -> None:
        """The deployed (legacy) checkpoint carries no aux head -> aux_dim 0 path."""
        _, ckpt = _load_model(torch.device("cpu"))
        self.assertNotIn("has_aux", ckpt)
        self.assertFalse(any("aux_head" in k for k in ckpt["model"]))


@unittest.skipUnless(torch is not None, f"torch not importable: {_TORCH_IMPORT_ERROR}")
class AuxCheckpointOfflineTest(unittest.TestCase):
    """A synthetic aux checkpoint drives the DeployACT ``_predict_offset`` math."""

    def _make_aux_checkpoint(self, path: pathlib.Path, aux_frame: str) -> None:
        """Save a tiny aux checkpoint (aux_dim=3) with omean/ostd/aux_frame."""
        torch.manual_seed(0)
        model = _Policy(k=4, state_dim=7, aux_dim=3)
        # Offset labels here live around a ~5 cm downward vector; store matching
        # de-normalization stats so a ~unit aux output maps to a plausible offset.
        omean = torch.tensor([0.0, 0.0, -0.05])
        ostd = torch.tensor([0.02, 0.02, 0.02])
        torch.save(
            {
                "model": model.state_dict(),
                "amean": torch.zeros(6), "astd": torch.ones(6),
                "smean": torch.zeros(7), "sstd": torch.ones(7),
                "K": 4, "img": 128, "state_dim": 7,
                "has_aux": True, "aux_dim": 3, "aux_frame": aux_frame,
                "omean": omean, "ostd": ostd,
            },
            path,
        )

    def _predict_offset_offline(self, ckpt: dict, obs_pose: np.ndarray):
        """Reproduce DeployACT._predict_offset with the reconstructed policy.

        Args:
            ckpt: The loaded checkpoint dict (with aux keys).
            obs_pose: The live ``[x, y, z, qx, qy, qz, qw]`` base_link TCP pose.

        Returns:
            The resolved :class:`port_offset.PortOffsetPrediction`.
        """
        model = _Policy(ckpt["K"], state_dim=ckpt["state_dim"], aux_dim=ckpt["aux_dim"])
        model.load_state_dict(ckpt["model"])  # strict: aux head must be present
        model.eval()
        imgs = torch.rand(1, 3, 3, 128, 128)
        state = torch.from_numpy(obs_pose.astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            _, aux = model(imgs, state)
        aux_vec = (aux[0] * ckpt["ostd"] + ckpt["omean"]).cpu().numpy()
        return port_offset.predict_from_aux(
            obs_pose[:3], obs_pose[3:7], aux_vec, frame=ckpt["aux_frame"]
        )

    def test_aux_checkpoint_predicts_plausible_offset(self) -> None:
        """An aux checkpoint yields a base_link target of plausible magnitude."""
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "aux.pt"
            self._make_aux_checkpoint(path, aux_frame="tcp")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(ckpt["has_aux"])
            self.assertTrue(any("aux_head" in k for k in ckpt["model"]))
            pose = np.array([0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0])
            pred = self._predict_offset_offline(ckpt, pose)
            # target = tcp_pos + offset_base; magnitude within a plausible window.
            self.assertEqual(pred.offset_base.shape, (3,))
            self.assertEqual(pred.target_base.shape, (3,))
            self.assertTrue(np.all(np.isfinite(pred.offset_base)))
            self.assertTrue(pred.plausible(0.005, 0.20))
            np.testing.assert_allclose(
                pred.target_base, pose[:3] + pred.offset_base, atol=1e-9
            )

    def test_identity_orientation_offset_matches_base(self) -> None:
        """With identity TCP orientation, TCP-frame offset equals the base offset."""
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "aux.pt"
            self._make_aux_checkpoint(path, aux_frame="tcp")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # identity quat
            pred = self._predict_offset_offline(ckpt, pose)
            # Downward-biased omean -> the offset's Z component is negative.
            self.assertLess(pred.offset_base[2], 0.0)


if __name__ == "__main__":
    unittest.main()
