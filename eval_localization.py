"""Held-out port-localization eval for a ``train_v3 --port-aux`` checkpoint.

This is THE decision metric for the privileged-DAgger port-localization pivot
(SESSION_REPORT.md cycle 6/7): given a trained aux-head checkpoint and a set of
held-out ``ds_dagger`` episodes, predict the TCP->port offset at every frame and
report the localization error against the episode's true label. If the median
held-out 3-D error is under the reach gate (default 10 mm) the seat is
*reachable* from the occluded sensing the deploy policy sees; if it is over the
gate, the occluded sensing lacks the information to localize the port.

The prediction path mirrors ``DeployACT`` *exactly* so the eval reflects
deployment (``aic_example_policies/.../ros/DeployACT.py``):

  1. Assemble the network inputs the way ``train_v2.load_all`` /
     ``train_v3._load_split`` do -- 3 cameras (left, center, right) resized to the
     checkpoint's ``img`` and normalized ``(x - 0.5) / 0.5``; the state is the
     7-D TCP pose, or the 13-D pose+wrench when the checkpoint was trained with
     ``--wrench`` -- normalized by the checkpoint's ``smean``/``sstd``.
  2. Run the ``port_aux`` head and de-normalize its output with the checkpoint's
     ``omean``/``ostd`` (``DeployACT._predict_offset``), giving the predicted
     offset in the checkpoint's ``aux_frame`` (``tcp`` by default).
  3. Resolve it into a base_link port target with the *same*
     ``port_offset.predict_from_aux`` the deploy loop uses.

The truth is the per-frame offset written beside each episode
(``port_offsets.npy``, TCP frame), cross-checked against a recomputation from the
true base_link port position (``port_target.npy``) via the identical
``port_offset`` geometry that produced the training labels.

The pure error-aggregation math (predicted vs. true arrays -> mm statistics), the
near-port masking, the deploy-target resolution, and the episode-split/glob logic
are all torch-free and unit-tested CPU-only (``tests/test_eval_localization.py``).
torch is imported lazily inside the checkpoint/inference helpers so this module --
and its tests -- import without a GPU or a torch build.

Typical usage::

    ~/miniconda3/bin/python eval_localization.py \\
        --ckpt ~/training/ckpt/dagger_aux.pt \\
        --episodes '~/training/ds_dagger/ep_*' \\
        --holdout-frac 0.2
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import math
import pathlib
import sys
from typing import Sequence

import numpy as np

# Reuse the canonical (ROS/torch-free) offset geometry from the deploy package so
# the eval frame convention and the deploy resolution can never diverge. The
# package root is ``<repo>/aic_example_policies`` (mirrors dagger_relabel.py and
# train_v3's repo-root insert).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _REPO_ROOT / "aic_example_policies"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from aic_example_policies.ros import port_offset  # noqa: E402

_LOG = logging.getLogger("eval_localization")

# Standard per-episode file names (written by dagger_relabel / prepare_dataset).
TCP_POSES_NAME = "tcp_poses.npy"
WRENCHES_NAME = "wrenches.npy"
PORT_OFFSETS_NAME = "port_offsets.npy"
PORT_TARGET_NAME = "port_target.npy"
_CAMERA_FILES = ("left_images.npy", "center_images.npy", "right_images.npy")

# Reach gate (m -> the seat is reachable when the median held-out 3-D error is
# below this). Expressed in mm at the CLI (``--gate-mm``, default 10).
DEFAULT_GATE_MM = 10.0
# Default near-port operating point (m): frames whose *true* remaining offset is
# within this radius, the last-inch regime where localization actually matters.
DEFAULT_NEAR_PORT_M = 0.03
_MM_PER_M = 1000.0


# =============================================================================
# Pure error-aggregation math (torch-free, unit-tested)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class OffsetErrorStats:
    """Aggregate localization-error statistics over a set of frames (mm).

    All magnitudes are millimeters. ``lateral`` is the in-plane (x, y) component
    of the error vector -- for a TCP-frame offset this is the misalignment
    perpendicular to the tool's approach axis, the component that decides whether
    the plug can enter the port; ``full_3d`` is the full Euclidean error and is
    frame-invariant (identical whether measured in the TCP or base_link frame).

    Attributes:
        n_frames: Number of frames the statistics were computed over.
        mean_3d_mm: Mean full-3-D error (mm).
        median_3d_mm: Median full-3-D error (mm) -- the reach-gate metric.
        p90_3d_mm: 90th-percentile full-3-D error (mm).
        mean_lat_mm: Mean lateral (x, y) error (mm).
        median_lat_mm: Median lateral (x, y) error (mm).
        p90_lat_mm: 90th-percentile lateral (x, y) error (mm).
    """

    n_frames: int
    mean_3d_mm: float
    median_3d_mm: float
    p90_3d_mm: float
    mean_lat_mm: float
    median_lat_mm: float
    p90_lat_mm: float


def offset_error_stats(
    predicted: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None
) -> OffsetErrorStats:
    """Aggregate per-frame predicted vs. true offset errors into mm statistics.

    Both arrays must be ``(N, 3)`` offsets expressed in the *same* frame. The
    per-frame error vector is ``predicted - true``; its full Euclidean norm is the
    3-D error and the norm of its first two (x, y) components is the lateral
    error. The returned statistics are in millimeters.

    Args:
        predicted: Predicted offsets shaped ``(N, 3)`` (m).
        true: True offsets shaped ``(N, 3)`` (m), same frame as ``predicted``.
        mask: Optional boolean ``(N,)`` frame selector; when given only the
            selected frames contribute (e.g. the near-port subset). At least one
            frame must be selected.

    Returns:
        The aggregated :class:`OffsetErrorStats` (mm).

    Raises:
        ValueError: If the arrays are not matching ``(N, 3)`` shapes, contain
            non-finite values, ``N`` is 0, or ``mask`` is the wrong shape /
            selects no frames.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    tru = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"predicted must be (N, 3), got {pred.shape}")
    if tru.shape != pred.shape:
        raise ValueError(
            f"true must match predicted shape {pred.shape}, got {tru.shape}"
        )
    if pred.shape[0] == 0:
        raise ValueError("predicted/true must contain at least one frame")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(tru))):
        raise ValueError("predicted/true contain non-finite values")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (pred.shape[0],):
            raise ValueError(
                f"mask must have shape ({pred.shape[0]},), got {m.shape}"
            )
        if not m.any():
            raise ValueError("mask selects no frames")
        pred, tru = pred[m], tru[m]

    err = pred - tru
    dist_3d = np.linalg.norm(err, axis=1) * _MM_PER_M
    dist_lat = np.linalg.norm(err[:, :2], axis=1) * _MM_PER_M
    return OffsetErrorStats(
        n_frames=int(pred.shape[0]),
        mean_3d_mm=float(dist_3d.mean()),
        median_3d_mm=float(np.median(dist_3d)),
        p90_3d_mm=float(np.percentile(dist_3d, 90.0)),
        mean_lat_mm=float(dist_lat.mean()),
        median_lat_mm=float(np.median(dist_lat)),
        p90_lat_mm=float(np.percentile(dist_lat, 90.0)),
    )


def near_port_mask(true_offsets: np.ndarray, radius_m: float) -> np.ndarray:
    """Return a boolean mask of frames within ``radius_m`` of the port.

    A frame is "near-port" when the magnitude of its *true* remaining offset is
    at most ``radius_m`` -- the last-inch operating point where localization
    matters most (mirrors ``train_v3``'s ``near_port_m`` val subset).

    Args:
        true_offsets: True per-frame offsets shaped ``(N, 3)`` (m).
        radius_m: Near-port radius (m); must be > 0.

    Returns:
        A boolean array shaped ``(N,)``, ``True`` where ``|offset| <= radius_m``.

    Raises:
        ValueError: If ``true_offsets`` is not ``(N, 3)`` or ``radius_m`` <= 0.
    """
    tru = np.asarray(true_offsets, dtype=np.float64)
    if tru.ndim != 2 or tru.shape[1] != 3:
        raise ValueError(f"true_offsets must be (N, 3), got {tru.shape}")
    if radius_m <= 0.0:
        raise ValueError(f"radius_m must be > 0, got {radius_m}")
    return np.linalg.norm(tru, axis=1) <= float(radius_m)


def resolve_deploy_targets(
    poses: np.ndarray, pred_offsets: np.ndarray, aux_frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve per-frame aux offsets into base_link via the deploy conversion.

    Calls the exact ``port_offset.predict_from_aux`` the deploy loop
    (``DeployACT._predict_offset``) uses on every frame, so the returned base_link
    port targets are byte-for-byte the ones the arm would aim at.

    Args:
        poses: TCP pose array shaped ``(N, 7)`` ``[x, y, z, qx, qy, qz, qw]`` in
            base_link.
        pred_offsets: Predicted (de-normalized) offsets shaped ``(N, 3)`` in
            ``aux_frame``.
        aux_frame: The frame ``pred_offsets`` live in: ``'tcp'`` or ``'base'``.

    Returns:
        ``(offset_base, target_base)`` each shaped ``(N, 3)`` -- the base_link
        TCP->port offset and the implied base_link port position per frame.

    Raises:
        ValueError: If shapes are wrong or ``aux_frame`` is unknown.
    """
    p = np.asarray(poses, dtype=np.float64)
    off = np.asarray(pred_offsets, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 7:
        raise ValueError(f"poses must be (N, 7), got {p.shape}")
    if off.shape != (p.shape[0], 3):
        raise ValueError(
            f"pred_offsets must be ({p.shape[0]}, 3), got {off.shape}"
        )
    if aux_frame not in ("tcp", "base"):
        raise ValueError(f"aux_frame must be 'tcp' or 'base', got {aux_frame!r}")
    offset_base = np.empty_like(off)
    target_base = np.empty_like(off)
    for i in range(p.shape[0]):
        pred = port_offset.predict_from_aux(
            p[i, :3], p[i, 3:7], off[i], frame=aux_frame
        )
        offset_base[i] = pred.offset_base
        target_base[i] = pred.target_base
    return offset_base, target_base


def true_offsets_from_target(
    poses: np.ndarray, port_target_base: np.ndarray, aux_frame: str
) -> np.ndarray:
    """Recompute per-frame true offsets from the true base_link port position.

    This is the exact geometry ``opt.port_labels`` uses to build the training
    targets (``port_offset.per_frame_tcp_offsets``), so it reproduces the label
    the aux head was trained against, in the checkpoint's ``aux_frame``.

    Args:
        poses: TCP pose array shaped ``(N, 7)`` in base_link.
        port_target_base: True port position ``[x, y, z]`` in base_link (m).
        aux_frame: ``'tcp'`` or ``'base'``.

    Returns:
        The true per-frame offsets shaped ``(N, 3)`` in ``aux_frame`` (m).
    """
    return port_offset.per_frame_tcp_offsets(
        np.asarray(poses, dtype=np.float64),
        np.asarray(port_target_base, dtype=np.float64),
        frame=aux_frame,
    )


# =============================================================================
# Pure episode-selection logic (torch-free, unit-tested)
# =============================================================================
def expand_episode_globs(spec: str) -> list[str]:
    """Expand a comma-separated set of episode-dir globs, sorted and de-duped.

    Mirrors ``train_v3._expand_globs`` so an eval ``--episodes`` spec selects the
    same directories the trainer would.

    Args:
        spec: Comma-separated glob patterns (``~`` is expanded).

    Returns:
        The matched directory paths, sorted within each pattern and de-duplicated
        across patterns (first occurrence wins).
    """
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.extend(sorted(glob.glob(str(pathlib.Path(part).expanduser()))))
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def held_out_split(episode_dirs: Sequence[str], holdout_frac: float) -> list[str]:
    """Return the held-out tail of a sorted episode list.

    Mirrors ``train_v2``'s "hold out the last N episodes" convention: the
    directories are sorted and the trailing ``ceil(holdout_frac * N)`` are
    returned (at least one). Deterministic given the same directory set.

    Args:
        episode_dirs: All candidate episode directories (order-independent).
        holdout_frac: Fraction to hold out, in ``(0, 1]``.

    Returns:
        The held-out episode directories, sorted ascending.

    Raises:
        ValueError: If ``episode_dirs`` is empty or ``holdout_frac`` not in
            ``(0, 1]``.
    """
    if not episode_dirs:
        raise ValueError("episode_dirs must be non-empty")
    if not 0.0 < holdout_frac <= 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1], got {holdout_frac}")
    dirs = sorted(episode_dirs)
    n_hold = min(len(dirs), max(1, math.ceil(holdout_frac * len(dirs))))
    return dirs[-n_hold:]


# =============================================================================
# Checkpoint / inference (torch imported lazily)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class CheckpointInfo:
    """Parsed aux-head checkpoint metadata (no torch tensors kept as fields).

    Attributes:
        path: Checkpoint path it was loaded from.
        k: Action-chunk length (K).
        img: Square input image size the encoder expects.
        state_dim: Proprioceptive state width (7 pose, 13 pose+wrench).
        use_wrench: Whether the 6-D wrist wrench is appended to the state.
        aux_dim: Aux-head output width (3 offset, 6 offset+axis); > 0 required.
        aux_frame: Frame the aux output lives in (``'tcp'`` / ``'base'``).
    """

    path: str
    k: int
    img: int
    state_dim: int
    use_wrench: bool
    aux_dim: int
    aux_frame: str


def load_checkpoint(path: str, device: str = "cpu"):
    """Load an aux-head checkpoint and its normalization tensors.

    torch is imported here (lazily) so the module imports GPU-free. ``device``
    defaults to ``cpu`` (``map_location='cpu'``) so the eval never contends the
    GPU a sim collection may be using.

    Args:
        path: Path to a ``train_v3 --port-aux`` checkpoint.
        device: Torch device string for ``map_location`` (default ``'cpu'``).

    Returns:
        A tuple ``(info, model, norm)`` where ``info`` is a
        :class:`CheckpointInfo`, ``model`` is the eval-mode ``train_v2.Policy``
        with the checkpoint weights loaded on ``device``, and ``norm`` is a dict
        of the de-normalization tensors ``{smean, sstd, omean, ostd}`` on
        ``device``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the checkpoint carries no aux head (``has_aux`` false or
            ``aux_dim`` 0) -- this eval requires a ``--port-aux`` checkpoint.
    """
    import torch  # lazy: keep module import GPU/torch-free

    import train_v2 as tv2  # repo-root Policy (single source of truth for arch)

    ckpt_path = pathlib.Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    has_aux = bool(ck.get("has_aux", False))
    aux_dim = int(ck.get("aux_dim", 0)) if has_aux else 0
    if aux_dim <= 0:
        raise ValueError(
            f"checkpoint {path} has no port_aux head (has_aux={has_aux}, "
            f"aux_dim={aux_dim}); retrain with train_v3 --port-aux."
        )
    state_dim = int(ck.get("state_dim", 7))
    info = CheckpointInfo(
        path=str(ckpt_path),
        k=int(ck["K"]),
        img=int(ck["img"]),
        state_dim=state_dim,
        use_wrench=bool(ck.get("use_wrench", state_dim == 13)),
        aux_dim=aux_dim,
        aux_frame=str(ck.get("aux_frame", "tcp")),
    )
    model = tv2.Policy(info.k, state_dim=info.state_dim, aux_dim=info.aux_dim)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    norm = {
        "smean": ck["smean"].to(device),
        "sstd": ck["sstd"].to(device),
        "omean": ck["omean"].to(device),
        "ostd": ck["ostd"].to(device),
    }
    return info, model, norm


def _load_camera_stack(ep_dir: pathlib.Path, img: int, device: str):
    """Load + preprocess the 3-camera image stack exactly like ``load_all``.

    Reproduces ``train_v2.load_all``'s preprocessing (per-camera
    ``permute -> /255 -> bilinear resize to (img, img) -> (x - 0.5) / 0.5``,
    stacked in left/center/right order) but keeps float32 for a CPU-capable
    forward pass (training uses fp16 + bf16 autocast on GPU; the numerics that
    matter -- resize, normalization, camera order -- are identical).

    Args:
        ep_dir: Episode directory holding the ``*_images.npy`` arrays.
        img: Square size to resize each camera to.
        device: Torch device string.

    Returns:
        A tuple ``(imgs, n)`` where ``imgs`` is a ``(N, 3, 3, img, img)`` float32
        tensor and ``n`` is the frame count.

    Raises:
        FileNotFoundError: If any camera array is missing.
    """
    import torch
    import torch.nn.functional as F

    cams = []
    n = -1
    for name in _CAMERA_FILES:
        path = ep_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{name} not found in {ep_dir}")
        arr = np.load(path)
        n = arr.shape[0]
        t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float() / 255.0
        t = F.interpolate(t, size=(img, img), mode="bilinear", align_corners=False)
        cams.append((t - 0.5) / 0.5)
    # cams is [left, center, right]; stack on the camera axis -> (N, 3, 3, H, W).
    return torch.stack(cams, 1), n


def predict_episode_offsets(ep_dir: str, info: CheckpointInfo, model, norm, device: str = "cpu"):
    """Predict per-frame port offsets for one episode (deploy-faithful).

    Assembles the checkpoint-matched inputs, runs the aux head, and de-normalizes
    its output with the checkpoint's ``omean``/``ostd`` -- the exact
    ``DeployACT._predict_offset`` prediction, minus the base_link rotation (kept
    separate so the caller can compare in either frame).

    Args:
        ep_dir: Episode directory.
        info: Parsed :class:`CheckpointInfo`.
        model: The loaded ``train_v2.Policy`` (eval mode).
        norm: The de-normalization tensor dict from :func:`load_checkpoint`.
        device: Torch device string.

    Returns:
        A tuple ``(pred_offsets, poses)`` where ``pred_offsets`` is the
        de-normalized ``(N, 3)`` offset in the checkpoint's ``aux_frame`` (m) and
        ``poses`` is the raw ``(N, 7)`` TCP pose array.

    Raises:
        FileNotFoundError: If a required per-frame array is missing.
        ValueError: If the wrench state is requested but ``wrenches.npy`` shape is
            wrong.
    """
    import torch

    ep = pathlib.Path(ep_dir)
    poses = np.load(ep / TCP_POSES_NAME).astype(np.float32)
    n = poses.shape[0]
    if info.use_wrench:
        wpath = ep / WRENCHES_NAME
        if not wpath.exists():
            raise FileNotFoundError(f"{WRENCHES_NAME} required for 13-D state: {ep}")
        wrench = np.load(wpath).astype(np.float32)
        if wrench.shape != (n, 6):
            raise ValueError(f"{wpath}: expected ({n}, 6), got {wrench.shape}")
        raw_state = np.concatenate([poses, wrench], axis=1)
    else:
        raw_state = poses
    imgs, n_img = _load_camera_stack(ep, info.img, device)
    if n_img != n:
        raise ValueError(f"{ep}: {n_img} image frames vs {n} pose frames")

    state = torch.from_numpy(raw_state).to(device)
    state = (state - norm["smean"]) / norm["sstd"]
    preds: list[np.ndarray] = []
    bs = 64
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            # train_v2.Policy.forward calls imgs.view(...); a plain slice is a
            # non-contiguous view (training gets a contiguous copy from tensor
            # indexing), so make the batch contiguous before the forward.
            out = model(imgs[sl].contiguous(), state[sl])
            aux = out[1]  # (nb, aux_dim) normalized
            aux = aux * norm["ostd"] + norm["omean"]  # de-normalize
            preds.append(aux[:, :3].cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float64), poses.astype(np.float64)


@dataclasses.dataclass(frozen=True)
class EpisodeEvalResult:
    """Per-episode localization eval outcome.

    Attributes:
        episode_dir: The episode directory evaluated.
        n_frames: Number of frames.
        stats_all: Error statistics over all frames (TCP-frame comparison).
        stats_near: Error statistics over the near-port subset, or None when no
            frame is near-port.
        pred_offsets: Predicted ``(N, 3)`` offsets in the checkpoint frame (m).
        true_offsets: True ``(N, 3)`` offsets in the checkpoint frame (m).
        pred_target_base: Deploy-resolved base_link port target for the final
            frame ``[x, y, z]`` (m), or None when no base target could be formed.
        true_target_base: True base_link port position ``[x, y, z]`` (m), or None
            when ``port_target.npy`` was absent.
    """

    episode_dir: str
    n_frames: int
    stats_all: OffsetErrorStats
    stats_near: OffsetErrorStats | None
    pred_offsets: np.ndarray
    true_offsets: np.ndarray
    pred_target_base: np.ndarray | None
    true_target_base: np.ndarray | None


def evaluate_episode(
    ep_dir: str, info: CheckpointInfo, model, norm, near_port_m: float, device: str = "cpu"
) -> EpisodeEvalResult:
    """Evaluate localization error for one held-out episode.

    Args:
        ep_dir: Episode directory.
        info: Parsed :class:`CheckpointInfo`.
        model: The loaded policy (eval mode).
        norm: The de-normalization tensor dict.
        near_port_m: Near-port radius (m) for the operating-point subset.
        device: Torch device string.

    Returns:
        The populated :class:`EpisodeEvalResult`.

    Raises:
        FileNotFoundError: If a required per-frame array is missing.
    """
    ep = pathlib.Path(ep_dir)
    pred_off, poses = predict_episode_offsets(ep_dir, info, model, norm, device)

    # Truth in the checkpoint's aux_frame. Prefer recomputing from the TRUE port
    # position (port_target.npy) via the exact training geometry -- this is the
    # label the head was trained against and is frame-consistent with the
    # checkpoint. Fall back to the cached port_offsets.npy (TCP frame).
    target_base = None
    tpath = ep / PORT_TARGET_NAME
    if tpath.exists():
        target_base = np.load(tpath).astype(np.float64).reshape(-1)
        true_off = true_offsets_from_target(poses, target_base, info.aux_frame)
        cached = ep / PORT_OFFSETS_NAME
        if cached.exists() and info.aux_frame == "tcp":
            stored = np.load(cached).astype(np.float64)
            if stored.shape == true_off.shape and not np.allclose(
                stored, true_off, atol=1e-4
            ):
                _LOG.warning(
                    "%s: cached port_offsets.npy disagrees with recomputed "
                    "target (max %.4f m); using recomputed.",
                    ep.name, float(np.abs(stored - true_off).max()),
                )
    else:
        cached = ep / PORT_OFFSETS_NAME
        if not cached.exists():
            raise FileNotFoundError(
                f"{ep}: needs port_target.npy or port_offsets.npy for the label"
            )
        true_off = np.load(cached).astype(np.float64)
        if info.aux_frame != "tcp":
            _LOG.warning(
                "%s: only cached TCP-frame port_offsets.npy available but "
                "checkpoint aux_frame=%s; comparing in TCP frame.",
                ep.name, info.aux_frame,
            )

    stats_all = offset_error_stats(pred_off, true_off)
    mask = near_port_mask(true_off, near_port_m)
    stats_near = offset_error_stats(pred_off, true_off, mask) if mask.any() else None

    # Deploy-resolved base_link target for the final (nearest) frame -- what the
    # arm would aim at, for a human-readable sanity line.
    pred_target_base = None
    if info.aux_frame in ("tcp", "base"):
        _, tgt = resolve_deploy_targets(poses[-1:], pred_off[-1:], info.aux_frame)
        pred_target_base = tgt[0]
    return EpisodeEvalResult(
        episode_dir=str(ep),
        n_frames=int(poses.shape[0]),
        stats_all=stats_all,
        stats_near=stats_near,
        pred_offsets=pred_off,
        true_offsets=true_off,
        pred_target_base=pred_target_base,
        true_target_base=target_base,
    )


def _fmt_stats(label: str, s: OffsetErrorStats) -> str:
    """Format an :class:`OffsetErrorStats` as a one-line report row."""
    return (
        f"{label:<14} n={s.n_frames:<5d} "
        f"3D[mean/med/p90]={s.mean_3d_mm:6.2f}/{s.median_3d_mm:6.2f}/"
        f"{s.p90_3d_mm:6.2f}mm  "
        f"lat[mean/med/p90]={s.mean_lat_mm:6.2f}/{s.median_lat_mm:6.2f}/"
        f"{s.p90_lat_mm:6.2f}mm"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate held-out port-localization error and gate on the median.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` when the median held-out 3-D error passes the
        gate, ``1`` when it exceeds the gate.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Held-out port-localization eval.")
    ap.add_argument("--ckpt", required=True, help="train_v3 --port-aux checkpoint.")
    ap.add_argument(
        "--episodes", required=True,
        help="Comma-separated held-out episode-dir globs (e.g. '~/training/ds_dagger/ep_*').",
    )
    ap.add_argument(
        "--holdout-frac", type=float, default=None,
        help="If set, evaluate only the last ceil(frac*N) sorted episodes.",
    )
    ap.add_argument(
        "--near-port-m", type=float, default=DEFAULT_NEAR_PORT_M,
        help="Near-port radius (m) for the operating-point subset.",
    )
    ap.add_argument(
        "--gate-mm", type=float, default=DEFAULT_GATE_MM,
        help="Reach gate on the median held-out 3-D error (mm).",
    )
    ap.add_argument("--device", default="cpu", help="Torch device (default cpu).")
    a = ap.parse_args(argv)

    episodes = expand_episode_globs(a.episodes)
    if not episodes:
        raise FileNotFoundError(f"no episodes match {a.episodes!r}")
    if a.holdout_frac is not None:
        episodes = held_out_split(episodes, a.holdout_frac)

    info, model, norm = load_checkpoint(a.ckpt, device=a.device)
    _LOG.info(
        "loaded %s (K=%d img=%d state_dim=%d aux_dim=%d aux_frame=%s) on %s",
        info.path, info.k, info.img, info.state_dim, info.aux_dim, info.aux_frame,
        a.device,
    )
    _LOG.info("evaluating %d held-out episodes", len(episodes))

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for ep in episodes:
        res = evaluate_episode(ep, info, model, norm, a.near_port_m, a.device)
        all_pred.append(res.pred_offsets)
        all_true.append(res.true_offsets)
        print(_fmt_stats(pathlib.Path(ep).name, res.stats_all))
        if res.stats_near is not None:
            print("  " + _fmt_stats("near-port", res.stats_near))
        if res.pred_target_base is not None and res.true_target_base is not None:
            derr = float(
                np.linalg.norm(res.pred_target_base - res.true_target_base) * _MM_PER_M
            )
            print(
                f"  deploy target (last frame): pred={res.pred_target_base.round(4)} "
                f"true={res.true_target_base.round(4)} |err|={derr:.2f}mm"
            )

    pred_cat = np.concatenate(all_pred, axis=0)
    true_cat = np.concatenate(all_true, axis=0)
    agg = offset_error_stats(pred_cat, true_cat)
    near = near_port_mask(true_cat, a.near_port_m)
    print("-" * 88)
    print(_fmt_stats("AGGREGATE", agg))
    if near.any():
        print("  " + _fmt_stats("near-port", offset_error_stats(pred_cat, true_cat, near)))

    passed = agg.median_3d_mm < a.gate_mm
    verdict = "PASS (seat reachable)" if passed else "FAIL (occluded sensing lacks info)"
    print(
        f"GATE: median held-out 3-D error {agg.median_3d_mm:.2f}mm "
        f"{'<' if passed else '>='} {a.gate_mm:.1f}mm -> {verdict}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
