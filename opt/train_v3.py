"""Improved ACT-lite trainer (train_v3).

Adds the Blackwell-class training wins justified by the opt/ research survey on
top of the already-fast train_v2 (whole dataset resident on GPU, bf16 autocast,
channels_last):

  * TF32 + ``set_float32_matmul_precision('high')`` for any fp32 matmuls.
  * Fused CUDA AdamW (``fused=True``) with decoupled weight decay.
  * Configurable ``torch.compile`` mode (default ``max-autotune``).
  * Optional EMA-of-weights evaluation (helps small imitation policies).
  * Optional torchao float8 training, which **degrades gracefully to bf16** when
    torchao is unavailable (as on this GB10 image) or a layer is unsupported.

Reuses ``train_v2.Policy`` and ``train_v2.load_all`` so the model architecture and
preprocessing exactly match the deployed checkpoint. Deterministic given a seed.

CLI example:
    ~/miniconda3/bin/python -m opt.train_v3 \\
        --train '~/training/ds_wide/ep_*' \\
        --val '~/training/ds_wide/ep_8,~/training/ds_wide/ep_9,~/training/ds_wide/ep_10' \\
        --epochs 120 --compile-mode max-autotune --ema-decay 0.999 \\
        --out ~/training/ckpt/v3_wide.pt
"""

from __future__ import annotations

import argparse
import glob
import logging
import pathlib
import sys
import time

# Make the repo root (train_v2.py) importable. torch/numpy resolve from the
# active interpreter's own site-packages (run with ~/miniconda3/bin/python); no
# extra site-packages path is injected, which would break a torch-less
# interpreter used only to run the pure-logic unit tests.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_v2 as tv2  # noqa: E402  (repo-root module)
from opt import augment  # noqa: E402
from opt import episode_prep  # noqa: E402
from opt import port_labels  # noqa: E402
from opt.config import TrainConfig, TrainResult  # noqa: E402

_LOG = logging.getLogger("opt.train_v3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_backend() -> None:
    """Enable TF32 / high-precision matmul and cudnn autotuning on CUDA."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_enable_fp8(model: torch.nn.Module, use_fp8: bool) -> bool:
    """Convert eligible Linear layers to torchao float8 training, if possible.

    Args:
        model: The policy module to convert in place.
        use_fp8: Whether float8 training was requested.

    Returns:
        True if float8 training was actually enabled; False if it was not
        requested or torchao/the hardware path is unavailable (bf16 fallback).
    """
    if not use_fp8:
        return False
    if not torch.cuda.is_available():
        _LOG.warning("fp8 requested but no CUDA device; falling back to bf16.")
        return False
    try:
        from torchao.float8 import convert_to_float8_training  # type: ignore

        convert_to_float8_training(model)
        _LOG.info("torchao float8 training enabled.")
        return True
    except Exception as exc:  # noqa: BLE001 - report any failure, keep training
        _LOG.warning("fp8 unavailable (%s); falling back to bf16.", exc)
        return False


class EmaWeights:
    """Exponential moving average of model parameters for evaluation.

    Maintains a detached shadow copy of every parameter and updates it after
    each optimizer step. ``apply_to``/``restore`` swap the shadow into the live
    model around evaluation so the reported metric reflects the EMA weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        """Initializes the shadow from the model's current parameters."""
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update the shadow toward the model's current parameters."""
        d = self.decay
        for name, p in model.named_parameters():
            self._shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap EMA weights into the model, backing up the live weights."""
        self._backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.copy_(self._shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore the live weights saved by :meth:`apply_to`."""
        for name, p in model.named_parameters():
            p.copy_(self._backup[name])
        self._backup = {}


def _expand_globs(spec: str) -> list[str]:
    """Expand a comma-separated set of episode-dir globs, sorted and de-duped."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.extend(sorted(glob.glob(str(pathlib.Path(part).expanduser()))))
    # Preserve order but drop duplicates.
    seen: set[str] = set()
    uniq = [d for d in out if not (d in seen or seen.add(d))]
    return uniq


def _load_seat_indices(dirs: list[str]) -> list[int]:
    """Load each episode's persisted seat frame index (``insertion_frame.npy``).

    Reads the small per-episode seat marker written by ``prepare_dataset`` (code
    change #3) so the last-inch window can end at the exact seat frame. This is a
    lightweight scalar ``np.load`` per directory (not the heavy image I/O of
    ``train_v2.load_all``); episodes without the file (older datasets) map to
    ``-1``, which :func:`episode_prep.last_inch_window` treats as "fall back to
    the speed-derived end".

    Args:
        dirs: Episode directories, in the same order ``train_v2.load_all`` loads
            them (so the returned list aligns positionally with those episodes).

    Returns:
        A list of per-episode seat frame indices (``int``), ``-1`` where absent
        or unreadable.
    """
    out: list[int] = []
    for d in dirs:
        path = pathlib.Path(d) / "insertion_frame.npy"
        seat = -1
        if path.exists():
            try:
                seat = int(np.load(path))
            except (ValueError, OSError) as exc:  # unreadable -> speed-derived end
                _LOG.warning("unreadable %s (%s); using speed-derived seat.", path, exc)
        out.append(seat)
    return out


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    fused: bool,
    params: "list[torch.nn.Parameter] | None" = None,
) -> torch.optim.Optimizer:
    """Build a (fused where supported) AdamW optimizer.

    Args:
        model: Module whose parameters are optimized.
        lr: Learning rate.
        weight_decay: Decoupled weight decay.
        fused: Request the fused CUDA kernel (ignored on CPU / if unsupported).
        params: Explicit parameter list to optimize (e.g. only the aux head for a
            frozen-encoder probe); defaults to ``model.parameters()``.

    Returns:
        A configured ``torch.optim.AdamW`` instance.
    """
    opt_params = list(model.parameters()) if params is None else list(params)
    use_fused = bool(fused and torch.cuda.is_available())
    try:
        return torch.optim.AdamW(
            opt_params, lr=lr, weight_decay=weight_decay, fused=use_fused
        )
    except (RuntimeError, ValueError) as exc:  # fused unsupported for these params
        _LOG.warning("fused AdamW unavailable (%s); using foreach.", exc)
        return torch.optim.AdamW(
            opt_params, lr=lr, weight_decay=weight_decay, foreach=True
        )


def _load_split(
    dirs: list[str], cfg: TrainConfig, apply_prep: bool
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Load a set of episodes to GPU tensors with the opt-in demo fixes applied.

    Reuses ``train_v2.load_all`` for the (heavy) GPU image/state/action loading,
    then optionally: concatenates the 6-D wrench onto the state (``cfg.use_wrench``
    -> 13-D), trims each episode's seated zero-velocity tail, and computes per-frame
    push-in loss weights. When ``cfg.last_inch_enabled`` the *inverse* selection is
    applied instead of tail-trim: only each demo's terminal approach->seat window
    is kept (using the exact ``insertion_frame.npy`` seat marker where present,
    else the speed-derived end) and the long lead-in is dropped, for an insertion
    specialist (mutually exclusive with ``tail_trim``). Trimming/last-inch/
    weighting are applied only to the training split (``apply_prep``); the
    validation split keeps every frame so its diagnostic metric stays comparable
    across runs. With every flag off the returned tensors are exactly
    ``load_all``'s output, ``weights`` is ``None`` and the two aux tensors are
    ``None`` (legacy path).

    When ``cfg.port_aux`` is set the hindsight port-offset labels + validity mask
    are built (in base_link, un-normalized, meters) from the *full* episode
    (:mod:`opt.port_labels`) and the **same** tail-trim keep-mask is applied to
    them so they stay aligned index-for-index with the images/state/actions
    (label-before-trim ordering, design section 3.2).

    Args:
        dirs: Episode directories to load.
        cfg: Training configuration (its flags decide wrench/trim/weighting/aux).
        apply_prep: Whether to apply tail-trim + push-in weighting (train only).

    Returns:
        ``(imgs, state, act, weights, offsets, valid)``. ``state`` is
        ``(N, cfg.state_dim)``; ``weights`` is a ``(N,)`` push-in weight tensor or
        ``None``; ``offsets`` is ``(N, aux_dim)`` raw offset labels (m) or
        ``None``; ``valid`` is a ``(N,)`` bool mask or ``None`` (both ``None``
        unless ``cfg.port_aux``).

    Raises:
        ValueError: If the built aux labels do not align with the loaded frames.
    """
    if cfg.use_wrench:
        imgs, state, act, epid, extra = tv2.load_all(
            dirs, cfg.img, cfg.k, load_extra=True
        )
        missing = int((~extra.has_wrench).sum().item())
        if missing:
            _LOG.warning(
                "%d/%d frames lack wrenches.npy; zero-filled. Deploy still sends "
                "the live wrench, but check the dataset if this is unexpected.",
                missing, state.shape[0],
            )
        state = torch.cat([state, extra.wrench], dim=1)  # (N, 13)
    else:
        imgs, state, act, epid = tv2.load_all(dirs, cfg.img, cfg.k)

    offsets: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    if cfg.port_aux:
        label_set = port_labels.build_labels(
            dirs,
            aux_dim=cfg.aux_dim,
            aux_frame=cfg.aux_frame,
            campaign_log=cfg.aux_label_glob,
        )
        if label_set.offsets.shape[0] != imgs.shape[0]:
            raise ValueError(
                "aux labels misaligned with loaded frames: "
                f"{label_set.offsets.shape[0]} labels vs {imgs.shape[0]} frames"
            )
        offsets = torch.from_numpy(label_set.offsets).to(imgs.device)
        valid = torch.from_numpy(label_set.valid).to(imgs.device)

    weights: torch.Tensor | None = None
    if apply_prep and cfg.last_inch_enabled:
        # Last-inch specialist: keep ONLY each demo's terminal approach->seat
        # window (the inverse of tail-trim) and drop the long lead-in. Mutually
        # exclusive with tail_trim (TrainConfig validates this).
        lookback = episode_prep.seconds_to_frames(cfg.last_inch_s, cfg.dt_frame)
        ramp = episode_prep.seconds_to_frames(cfg.pushin_ramp_s, cfg.dt_frame)
        epid_np = epid.detach().cpu().numpy()
        vel_np = act[:, 0].detach().cpu().float().numpy()
        # Exact seat marker per episode where persisted; -1 -> speed-derived end.
        # epid values are dir indices, so map per-dir markers to block order.
        seat_by_dir = _load_seat_indices(dirs)
        seat_list = [
            seat_by_dir[int(epid_np[s])]
            for s, _ in episode_prep.episode_bounds(epid_np)
        ]
        keep_np, w_np = episode_prep.build_last_inch_keep_and_weights(
            epid_np, vel_np,
            thr=cfg.tail_trim_threshold,
            min_frames=cfg.last_inch_min_frames,
            lookback_frames=lookback,
            pushin_ramp_frames=ramp,
            pushin_weight=cfg.pushin_weight,
            seat_indices=seat_list,
        )
        keep = torch.from_numpy(keep_np).to(imgs.device)
        imgs, state, act = imgs[keep], state[keep], act[keep]
        if offsets is not None:
            offsets, valid = offsets[keep], valid[keep]
        if cfg.pushin_enabled:
            weights = torch.from_numpy(w_np[keep_np]).to(state.device)
    elif apply_prep and (cfg.tail_trim or cfg.pushin_enabled):
        margin = episode_prep.seconds_to_frames(
            cfg.tail_trim_margin_s, cfg.dt_frame, minimum=0
        )
        ramp = episode_prep.seconds_to_frames(cfg.pushin_ramp_s, cfg.dt_frame)
        epid_np = epid.detach().cpu().numpy()
        # act[:, 0] is each frame's own twist (chunk step 0 has no clamping),
        # i.e. the raw tcp_velocity used for the m/s trim threshold.
        vel_np = act[:, 0].detach().cpu().float().numpy()
        keep_np, w_np = episode_prep.build_keep_and_weights(
            epid_np, vel_np,
            tail_trim=cfg.tail_trim,
            trim_threshold=cfg.tail_trim_threshold,
            trim_margin_frames=margin,
            pushin_ramp_frames=ramp,
            pushin_weight=cfg.pushin_weight,
        )
        keep = torch.from_numpy(keep_np).to(imgs.device)
        imgs, state, act = imgs[keep], state[keep], act[keep]
        if offsets is not None:
            offsets, valid = offsets[keep], valid[keep]
        if cfg.pushin_enabled:
            weights = torch.from_numpy(w_np[keep_np]).to(state.device)
    return imgs, state, act, weights, offsets, valid


def train(cfg: TrainConfig) -> TrainResult:
    """Train an ACT-lite policy per ``cfg`` and return its result record.

    Args:
        cfg: Training configuration.

    Returns:
        A :class:`TrainResult` with throughput and validation metrics.

    Raises:
        FileNotFoundError: If no training episodes match ``cfg.train_globs``.
    """
    setup_backend()
    seed_everything(cfg.seed)

    train_dirs = _expand_globs(cfg.train_globs)
    val_dirs = [d for d in _expand_globs(cfg.val_globs) if d]
    val_set = set(val_dirs)
    # A held-out val episode must not also train.
    train_dirs = [d for d in train_dirs if d not in val_set]
    if not train_dirs:
        raise FileNotFoundError(f"no training episodes match {cfg.train_globs!r}")

    imt, stt, act_t, w_t, off_t, valm_t = _load_split(train_dirs, cfg, apply_prep=True)
    if val_dirs:
        imv, stv, act_v, _, off_v, valm_v = _load_split(val_dirs, cfg, apply_prep=False)
    else:  # overfit mode: validate on the (prepared) training data.
        imv, stv, act_v, off_v, valm_v = imt, stt, act_t, off_t, valm_t

    smean, sstd = stt.mean(0), stt.std(0) + 1e-6
    amean, astd = act_t.reshape(-1, 6).mean(0), act_t.reshape(-1, 6).std(0) + 1e-6
    stt_n, stv_n = (stt - smean) / sstd, (stv - smean) / sstd
    act_t_n, act_v_n = (act_t - amean) / astd, (act_v - amean) / astd

    # Port-bearing aux head: normalize the offset labels with train-split stats
    # over the *valid* frames only (design section 3.1), and keep the raw
    # (meters) val labels for the de-normalized cm metric. omean/ostd are stored
    # in the checkpoint so deploy de-normalizes with the exact training scale.
    omean = ostd = off_t_n = off_v_n = None
    aux_dim = cfg.aux_dim if cfg.port_aux else 0
    if cfg.port_aux:
        assert off_t is not None and valm_t is not None
        if not bool(valm_t.any()):
            raise ValueError(
                "port_aux is on but no training frame is a valid (KEEP + "
                "inserted) label source; check the campaign_log join."
            )
        sel = off_t[valm_t]
        omean, ostd = sel.mean(0), sel.std(0) + 1e-6
        off_t_n = (off_t - omean) / ostd
        off_v_n = (off_v - omean) / ostd

    model = tv2.Policy(cfg.k, state_dim=cfg.state_dim, aux_dim=aux_dim).to(DEVICE)
    if cfg.init_ckpt:
        init = torch.load(cfg.init_ckpt, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        _LOG.info(
            "initialized weights from %s (missing=%d, unexpected=%d)",
            cfg.init_ckpt, len(missing), len(unexpected),
        )
    # Frozen-encoder probe: freeze encoder AND action head, train only the aux
    # head (design section 2.2). BN running stats are held by keeping the frozen
    # submodules in eval() (re-asserted each epoch after runnable.train()).
    frozen = bool(cfg.port_aux and cfg.aux_freeze_encoder)
    if frozen:
        model.enc.requires_grad_(False)
        model.head.requires_grad_(False)
    fp8_active = maybe_enable_fp8(model, cfg.use_fp8)
    runnable = model
    if cfg.compile_mode != "none":
        runnable = torch.compile(model, mode=cfg.compile_mode)
    opt_params = list(model.aux_head.parameters()) if frozen else None
    opt = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.fused_adam, opt_params)
    ema = EmaWeights(model, cfg.ema_decay) if cfg.ema_enabled else None

    # Dedicated RNG for input augmentation so shifts/dropout are reproducible
    # under cfg.seed and independent of the shuffle/model RNG stream.
    aug_gen = torch.Generator(device=DEVICE)
    aug_gen.manual_seed(cfg.seed)

    n_train = imt.shape[0]

    def run_epoch(
        images, states, actions, train_flag: bool, weights=None,
        aux_labels=None, valid=None,
    ) -> float:
        runnable.train(train_flag)
        if frozen and train_flag:
            # Keep the frozen encoder/action head (and their BN stats) in eval.
            model.enc.eval()
            model.head.eval()
        n = images.shape[0]
        idx = (
            torch.randperm(n, device=DEVICE)
            if train_flag
            else torch.arange(n, device=DEVICE)
        )
        tot = torch.zeros((), device=DEVICE)
        for i in range(0, n, cfg.bs):
            b = idx[i : i + cfg.bs]
            img_b, st_b = images[b], states[b]
            if train_flag:  # augment training batches only (never val/eval)
                if cfg.shift_pad > 0:
                    img_b = augment.random_shift(img_b, cfg.shift_pad, aug_gen)
                if cfg.proprio_dropout > 0.0:
                    st_b = augment.proprio_dropout(st_b, cfg.proprio_dropout, aug_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = runnable(img_b, st_b)
                pred, aux_pred = (out if cfg.port_aux else (out, None))
                if weights is None:
                    action_loss = F.l1_loss(pred, actions[b])
                else:  # push-in weighting: per-frame weighted mean of per-frame L1
                    per_frame = (pred - actions[b]).abs().mean(dim=(1, 2))
                    w_b = weights[b]
                    action_loss = (per_frame * w_b).sum() / w_b.sum()
                if cfg.port_aux and aux_labels is not None:
                    # Masked aux L1 over valid frames only, independent of the
                    # push-in weights (design section 3.1).
                    v_b = valid[b].to(aux_pred.dtype)
                    aux_l1 = (aux_pred - aux_labels[b]).abs().mean(-1)
                    aux_loss = (aux_l1 * v_b).sum() / v_b.sum().clamp(min=1.0)
                    loss = (
                        cfg.aux_weight * aux_loss
                        if frozen  # frozen probe optimizes the aux term alone
                        else action_loss + cfg.aux_weight * aux_loss
                    )
                else:
                    loss = action_loss
            if train_flag:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(model)
            tot += loss.detach() * len(b)
        return (tot / n).item()

    @torch.no_grad()
    def val_action_l1() -> tuple[float, float]:
        """Return (first-action, full-chunk) de-normalized L1 (m/s) on val."""
        runnable.eval()
        first_err = 0.0
        chunk_err = 0.0
        for i in range(0, imv.shape[0], cfg.bs):
            b = slice(i, i + cfg.bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = runnable(imv[b], stv_n[b])
            pred = (out[0] if cfg.port_aux else out).float()
            diff = (pred - act_v_n[b]).abs() * astd  # de-normalize, (nb, K, 6)
            first_err += diff[:, 0].mean(1).sum().item()
            chunk_err += diff.mean((1, 2)).sum().item()
        nv = imv.shape[0]
        return first_err / nv, chunk_err / nv

    @torch.no_grad()
    def val_offset_metrics() -> tuple[float, float, float]:
        """Return de-normalized port-offset val metrics in cm.

        Computes, over the *valid* held-out frames, the median Euclidean
        TCP->port error, the same restricted to the near-port operating point
        (true remaining distance <= ``cfg.near_port_m``), and the median lateral
        (perpendicular-to-offset) error -- the component that clips distractors
        (design section 2.5). Errors are frame-invariant so they are computed
        directly on the ``aux_frame`` vectors.

        Returns:
            ``(median_cm, nearport_median_cm, lateral_median_cm)``; a metric is
            NaN when its frame subset is empty.
        """
        runnable.eval()
        eucl_parts: list[torch.Tensor] = []
        lat_parts: list[torch.Tensor] = []
        near_parts: list[torch.Tensor] = []
        for i in range(0, imv.shape[0], cfg.bs):
            b = slice(i, i + cfg.bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = runnable(imv[b], stv_n[b])
            aux_pred = out[1].float()
            pred_off = aux_pred[:, :3] * ostd[:3] + omean[:3]  # meters, aux_frame
            true_off = off_v[b][:, :3]                          # meters (raw)
            v_b = valm_v[b]
            err = pred_off - true_off
            eucl = err.norm(dim=1)
            tnorm = true_off.norm(dim=1, keepdim=True).clamp(min=1e-9)
            along = (err * (true_off / tnorm)).sum(1)
            lateral = (eucl.pow(2) - along.pow(2)).clamp(min=0.0).sqrt()
            near = v_b & (true_off.norm(dim=1) <= cfg.near_port_m)
            eucl_parts.append(eucl[v_b])
            lat_parts.append(lateral[v_b])
            near_parts.append(eucl[near])

        def _median_cm(parts: list[torch.Tensor]) -> float:
            cat = torch.cat(parts) if parts else torch.empty(0, device=DEVICE)
            if cat.numel() == 0:
                return float("nan")
            return float(cat.median().item()) * 100.0

        return (
            _median_cm(eucl_parts),
            _median_cm(near_parts),
            _median_cm(lat_parts),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    final_train_loss, final_val_loss = 0.0, 0.0
    best_first, best_chunk = float("inf"), float("inf")
    final_chunk = 0.0
    best_offset_cm = float("inf")
    final_offset_cm = final_near_cm = final_lateral_cm = 0.0
    for ep in range(cfg.epochs):
        final_train_loss = run_epoch(
            imt, stt_n, act_t_n, True, weights=w_t,
            aux_labels=off_t_n, valid=valm_t,
        )
        if ep % 5 == 0 or ep == cfg.epochs - 1:
            if ema is not None:
                ema.apply_to(model)
            with torch.no_grad():
                final_val_loss = run_epoch(
                    imv, stv_n, act_v_n, False,
                    aux_labels=off_v_n, valid=valm_v,
                )
            first_err, final_chunk = val_action_l1()
            best_first = min(best_first, first_err)
            best_chunk = min(best_chunk, final_chunk)
            if cfg.port_aux:
                final_offset_cm, final_near_cm, final_lateral_cm = val_offset_metrics()
                if final_offset_cm == final_offset_cm:  # not NaN
                    best_offset_cm = min(best_offset_cm, final_offset_cm)
            if ema is not None:
                ema.restore(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0
    thr = cfg.epochs * n_train / wall
    ms_epoch = 1000.0 * wall / cfg.epochs

    ckpt_path = ""
    if cfg.out:
        if ema is not None:
            ema.apply_to(model)
        sd = model.state_dict()
        if ema is not None:
            ema.restore(model)
        out_path = pathlib.Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict = {
            "model": sd,
            "amean": amean.cpu(),
            "astd": astd.cpu(),
            "smean": smean.cpu(),
            "sstd": sstd.cpu(),
            "K": cfg.k,
            "img": cfg.img,
            # State dimensionality so DeployACT rebuilds the matching head and
            # (for 13-D) appends the live wrench. Old checkpoints lack these
            # keys; DeployACT defaults to 7-D / no-wrench when absent.
            "state_dim": cfg.state_dim,
            "use_wrench": cfg.use_wrench,
        }
        # Port-bearing aux keys, added only when the aux head is trained. Absent
        # on legacy checkpoints -> DeployACT builds a plain policy (byte-identical).
        if cfg.port_aux:
            ckpt["has_aux"] = True
            ckpt["aux_dim"] = cfg.aux_dim
            ckpt["aux_frame"] = cfg.aux_frame
            ckpt["omean"] = omean.cpu()
            ckpt["ostd"] = ostd.cpu()
        torch.save(ckpt, out_path)
        ckpt_path = str(out_path)

    return TrainResult(
        frames=n_train + (imv.shape[0] if val_dirs else 0),
        train_frames=n_train,
        val_frames=imv.shape[0],
        epochs=cfg.epochs,
        throughput_fps=thr,
        ms_per_epoch=ms_epoch,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        best_val_first_action=best_first,
        wall_s=wall,
        fp8_active=fp8_active,
        compile_mode=cfg.compile_mode,
        ckpt_path=ckpt_path,
        best_val_full_chunk=best_chunk,
        final_val_full_chunk=final_chunk,
        best_val_offset_cm=best_offset_cm,
        final_val_offset_cm=final_offset_cm,
        final_val_offset_nearport_cm=final_near_cm,
        final_val_offset_lateral_cm=final_lateral_cm,
    )


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse CLI arguments into a TrainConfig."""
    ap = argparse.ArgumentParser(description="Improved ACT-lite trainer (train_v3).")
    ap.add_argument("--train", dest="train_globs", default=TrainConfig.train_globs)
    ap.add_argument("--val", dest="val_globs", default=TrainConfig.val_globs)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--bs", type=int, default=TrainConfig.bs)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    ap.add_argument("--img", type=int, default=TrainConfig.img)
    ap.add_argument("--k", type=int, default=TrainConfig.k)
    ap.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay)
    ap.add_argument(
        "--compile-mode",
        default=TrainConfig.compile_mode,
        choices=["none", "default", "reduce-overhead", "max-autotune",
                 "max-autotune-no-cudagraphs"],
    )
    ap.add_argument("--no-fused-adam", dest="fused_adam", action="store_false")
    ap.add_argument("--fp8", dest="use_fp8", action="store_true")
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--out", default=TrainConfig.out)
    ap.add_argument("--shift-pad", type=int, default=TrainConfig.shift_pad,
                    help="DrQ random-shift radius in px on train images (0=off).")
    ap.add_argument("--proprio-dropout", type=float, default=TrainConfig.proprio_dropout,
                    help="Per-sample state-zeroing probability on train (0=off).")
    ap.add_argument("--tail-trim", dest="tail_trim", action="store_true",
                    help="Trim each demo's seated zero-velocity tail (train only).")
    ap.add_argument("--tail-trim-threshold", type=float,
                    default=TrainConfig.tail_trim_threshold,
                    help="Linear-speed (m/s) below which a trailing frame is stopped.")
    ap.add_argument("--tail-trim-margin-s", type=float,
                    default=TrainConfig.tail_trim_margin_s,
                    help="Seconds of frames kept after the last moving frame.")
    ap.add_argument("--last-inch-s", type=float, default=TrainConfig.last_inch_s,
                    help="Keep only each demo's terminal approach->seat window of "
                         "this many seconds (inverse of --tail-trim; 0=off).")
    ap.add_argument("--last-inch-min-frames", type=int,
                    default=TrainConfig.last_inch_min_frames,
                    help="Minimum frames kept in each last-inch window (>=1).")
    ap.add_argument("--wrench", dest="use_wrench", action="store_true",
                    help="Append the 6-D wrist wrench to the state (7-D -> 13-D).")
    ap.add_argument("--pushin-weight", type=float, default=TrainConfig.pushin_weight,
                    help="Peak push-in loss weight W (1.0=off / uniform L1).")
    ap.add_argument("--pushin-ramp-s", type=float, default=TrainConfig.pushin_ramp_s,
                    help="Seconds of frames over which the loss weight ramps to W.")
    ap.add_argument("--dt-frame", type=float, default=TrainConfig.dt_frame,
                    help="Recording frame period (s) for seconds->frames conversion.")
    ap.add_argument("--port-aux", dest="port_aux", action="store_true",
                    help="Train the port-bearing auxiliary head (TCP->port offset).")
    ap.add_argument("--aux-dim", type=int, default=TrainConfig.aux_dim,
                    choices=[3, 6], help="Aux output width: 3=offset, 6=offset+axis.")
    ap.add_argument("--aux-weight", type=float, default=TrainConfig.aux_weight,
                    help="Weight on the masked aux L1 term (0 disables the aux grad).")
    ap.add_argument("--aux-frame", default=TrainConfig.aux_frame,
                    choices=["tcp", "base"], help="Frame the offset labels live in.")
    ap.add_argument("--aux-freeze-encoder", dest="aux_freeze_encoder",
                    action="store_true",
                    help="Frozen probe: train only the aux head (needs --init-ckpt).")
    ap.add_argument("--aux-label-glob", default=TrainConfig.aux_label_glob,
                    help="Explicit campaign_log.csv (empty auto-derives per episode).")
    ap.add_argument("--init-ckpt", default=TrainConfig.init_ckpt,
                    help="Checkpoint to initialize weights from (non-strict).")
    ap.add_argument("--near-port-m", type=float, default=TrainConfig.near_port_m,
                    help="TCP->port distance (m) defining the near-port val subset.")
    a = ap.parse_args(argv)
    return TrainConfig(
        train_globs=a.train_globs, val_globs=a.val_globs, epochs=a.epochs, bs=a.bs,
        lr=a.lr, weight_decay=a.weight_decay, img=a.img, k=a.k, ema_decay=a.ema_decay,
        compile_mode=a.compile_mode, fused_adam=a.fused_adam, use_fp8=a.use_fp8,
        seed=a.seed, out=a.out, shift_pad=a.shift_pad, proprio_dropout=a.proprio_dropout,
        tail_trim=a.tail_trim, tail_trim_threshold=a.tail_trim_threshold,
        tail_trim_margin_s=a.tail_trim_margin_s, last_inch_s=a.last_inch_s,
        last_inch_min_frames=a.last_inch_min_frames, use_wrench=a.use_wrench,
        pushin_weight=a.pushin_weight, pushin_ramp_s=a.pushin_ramp_s, dt_frame=a.dt_frame,
        port_aux=a.port_aux, aux_dim=a.aux_dim, aux_weight=a.aux_weight,
        aux_frame=a.aux_frame, aux_freeze_encoder=a.aux_freeze_encoder,
        aux_label_glob=a.aux_label_glob, init_ckpt=a.init_ckpt, near_port_m=a.near_port_m,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: train and print the result record."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _parse_args(argv)
    res = train(cfg)
    print(
        f"[train_v3] frames={res.frames} train={res.train_frames} val={res.val_frames} "
        f"epochs={res.epochs} compile={res.compile_mode} fp8={res.fp8_active}"
    )
    print(
        f"[train_v3] fixes: tail_trim={cfg.tail_trim}(thr={cfg.tail_trim_threshold} "
        f"margin={cfg.tail_trim_margin_s}s) last_inch_s={cfg.last_inch_s}"
        f"(min={cfg.last_inch_min_frames}) state_dim={cfg.state_dim} "
        f"pushin_W={cfg.pushin_weight}(ramp={cfg.pushin_ramp_s}s) shift_pad={cfg.shift_pad}"
    )
    print(
        f"[train_v3] throughput={res.throughput_fps:.0f} fr/s  "
        f"{res.ms_per_epoch:.1f} ms/epoch  wall={res.wall_s:.1f}s"
    )
    print(
        f"[train_v3] final train_L1={res.final_train_loss:.4f}  "
        f"val_L1={res.final_val_loss:.4f}  "
        f"best val first-action |err|={res.best_val_first_action:.5f} m/s  "
        f"best val full-chunk |err|={res.best_val_full_chunk:.5f} m/s"
    )
    if cfg.port_aux:
        print(
            f"[train_v3] port_aux: dim={cfg.aux_dim} frame={cfg.aux_frame} "
            f"weight={cfg.aux_weight} freeze_encoder={cfg.aux_freeze_encoder} | "
            f"val_offset_cm={res.final_val_offset_cm:.2f} "
            f"near-port={res.final_val_offset_nearport_cm:.2f} "
            f"lateral={res.final_val_offset_lateral_cm:.2f} "
            f"(best={res.best_val_offset_cm:.2f})"
        )
    if res.ckpt_path:
        print(f"[train_v3] saved {res.ckpt_path}")


if __name__ == "__main__":
    main()
