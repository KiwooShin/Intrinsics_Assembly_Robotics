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


def build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float, fused: bool
) -> torch.optim.Optimizer:
    """Build a (fused where supported) AdamW optimizer.

    Args:
        model: Module whose parameters are optimized.
        lr: Learning rate.
        weight_decay: Decoupled weight decay.
        fused: Request the fused CUDA kernel (ignored on CPU / if unsupported).

    Returns:
        A configured ``torch.optim.AdamW`` instance.
    """
    use_fused = bool(fused and torch.cuda.is_available())
    try:
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, fused=use_fused
        )
    except (RuntimeError, ValueError) as exc:  # fused unsupported for these params
        _LOG.warning("fused AdamW unavailable (%s); using foreach.", exc)
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, foreach=True
        )


def _load_split(
    dirs: list[str], cfg: TrainConfig, apply_prep: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load a set of episodes to GPU tensors with the opt-in demo fixes applied.

    Reuses ``train_v2.load_all`` for the (heavy) GPU image/state/action loading,
    then optionally: concatenates the 6-D wrench onto the state (``cfg.use_wrench``
    -> 13-D), trims each episode's seated zero-velocity tail, and computes per-frame
    push-in loss weights. Trimming/weighting are applied only to the training split
    (``apply_prep``); the validation split keeps every frame so its diagnostic
    metric stays comparable across runs. With every flag off the returned tensors
    are exactly ``load_all``'s output and ``weights`` is ``None`` (legacy path).

    Args:
        dirs: Episode directories to load.
        cfg: Training configuration (its flags decide wrench/trim/weighting).
        apply_prep: Whether to apply tail-trim + push-in weighting (train only).

    Returns:
        ``(imgs, state, act, weights)`` where ``state`` is ``(N, cfg.state_dim)``
        and ``weights`` is a ``(N,)`` per-frame loss-weight tensor, or ``None``
        when push-in weighting is disabled (caller uses the plain mean L1).
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

    weights: torch.Tensor | None = None
    if apply_prep and (cfg.tail_trim or cfg.pushin_enabled):
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
        if cfg.pushin_enabled:
            weights = torch.from_numpy(w_np[keep_np]).to(state.device)
    return imgs, state, act, weights


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

    imt, stt, act_t, w_t = _load_split(train_dirs, cfg, apply_prep=True)
    if val_dirs:
        imv, stv, act_v, _ = _load_split(val_dirs, cfg, apply_prep=False)
    else:  # overfit mode: validate on the (prepared) training data.
        imv, stv, act_v = imt, stt, act_t

    smean, sstd = stt.mean(0), stt.std(0) + 1e-6
    amean, astd = act_t.reshape(-1, 6).mean(0), act_t.reshape(-1, 6).std(0) + 1e-6
    stt_n, stv_n = (stt - smean) / sstd, (stv - smean) / sstd
    act_t_n, act_v_n = (act_t - amean) / astd, (act_v - amean) / astd

    model = tv2.Policy(cfg.k, state_dim=cfg.state_dim).to(DEVICE)
    fp8_active = maybe_enable_fp8(model, cfg.use_fp8)
    runnable = model
    if cfg.compile_mode != "none":
        runnable = torch.compile(model, mode=cfg.compile_mode)
    opt = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.fused_adam)
    ema = EmaWeights(model, cfg.ema_decay) if cfg.ema_enabled else None

    # Dedicated RNG for input augmentation so shifts/dropout are reproducible
    # under cfg.seed and independent of the shuffle/model RNG stream.
    aug_gen = torch.Generator(device=DEVICE)
    aug_gen.manual_seed(cfg.seed)

    n_train = imt.shape[0]

    def run_epoch(
        images, states, actions, train_flag: bool, weights=None
    ) -> float:
        runnable.train(train_flag)
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
                pred = runnable(img_b, st_b)
                if weights is None:
                    loss = F.l1_loss(pred, actions[b])
                else:  # push-in weighting: per-frame weighted mean of per-frame L1
                    per_frame = (pred - actions[b]).abs().mean(dim=(1, 2))
                    w_b = weights[b]
                    loss = (per_frame * w_b).sum() / w_b.sum()
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
                pred = runnable(imv[b], stv_n[b]).float()
            diff = (pred - act_v_n[b]).abs() * astd  # de-normalize, (nb, K, 6)
            first_err += diff[:, 0].mean(1).sum().item()
            chunk_err += diff.mean((1, 2)).sum().item()
        nv = imv.shape[0]
        return first_err / nv, chunk_err / nv

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    final_train_loss, final_val_loss = 0.0, 0.0
    best_first, best_chunk = float("inf"), float("inf")
    final_chunk = 0.0
    for ep in range(cfg.epochs):
        final_train_loss = run_epoch(imt, stt_n, act_t_n, True, weights=w_t)
        if ep % 5 == 0 or ep == cfg.epochs - 1:
            if ema is not None:
                ema.apply_to(model)
            with torch.no_grad():
                final_val_loss = run_epoch(imv, stv_n, act_v_n, False)
            first_err, final_chunk = val_action_l1()
            best_first = min(best_first, first_err)
            best_chunk = min(best_chunk, final_chunk)
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
        torch.save(
            {
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
            },
            out_path,
        )
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
    ap.add_argument("--wrench", dest="use_wrench", action="store_true",
                    help="Append the 6-D wrist wrench to the state (7-D -> 13-D).")
    ap.add_argument("--pushin-weight", type=float, default=TrainConfig.pushin_weight,
                    help="Peak push-in loss weight W (1.0=off / uniform L1).")
    ap.add_argument("--pushin-ramp-s", type=float, default=TrainConfig.pushin_ramp_s,
                    help="Seconds of frames over which the loss weight ramps to W.")
    ap.add_argument("--dt-frame", type=float, default=TrainConfig.dt_frame,
                    help="Recording frame period (s) for seconds->frames conversion.")
    a = ap.parse_args(argv)
    return TrainConfig(
        train_globs=a.train_globs, val_globs=a.val_globs, epochs=a.epochs, bs=a.bs,
        lr=a.lr, weight_decay=a.weight_decay, img=a.img, k=a.k, ema_decay=a.ema_decay,
        compile_mode=a.compile_mode, fused_adam=a.fused_adam, use_fp8=a.use_fp8,
        seed=a.seed, out=a.out, shift_pad=a.shift_pad, proprio_dropout=a.proprio_dropout,
        tail_trim=a.tail_trim, tail_trim_threshold=a.tail_trim_threshold,
        tail_trim_margin_s=a.tail_trim_margin_s, use_wrench=a.use_wrench,
        pushin_weight=a.pushin_weight, pushin_ramp_s=a.pushin_ramp_s, dt_frame=a.dt_frame,
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
        f"margin={cfg.tail_trim_margin_s}s) state_dim={cfg.state_dim} "
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
    if res.ckpt_path:
        print(f"[train_v3] saved {res.ckpt_path}")


if __name__ == "__main__":
    main()
