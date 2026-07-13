"""W1 regularizer screening matrix: DrQ random-shift + proprioception dropout.

Runs the canonical-protocol experiment matrix (CLAUDE.md section 6) that screens
two cheap input regularizers on the ACT-lite policy:

  * random-shift image augmentation (arXiv:2004.13649, arXiv:2108.03298), and
  * proprioception dropout (arXiv:2509.18644),

against a no-augmentation baseline. Fixed budget 60 epochs, lr 1e-3, wd 1e-4;
train = every episode except the held-out val set (ds_wide/ep_8,9,10); crossed
over chunk K in {8, 16} with several seeds each.

Efficiency: the (large) image/state/action tensors are loaded onto the GPU once
per K and reused across every condition/seed; a single ``torch.compile``-d model
per K is re-initialized in place between runs, so the graph compiles only twice
for the whole matrix. Deterministic given the per-run seed.

Reported metrics are SECONDARY screening diagnostics (val first-action L1 and
full-chunk L1 in m/s); the winning conditions must be confirmed on the in-sim
scored suite before adoption (CLAUDE.md section 6).

CLI example:
    ~/miniconda3/bin/python -m opt.w1_matrix --seeds 3 --compile-mode default
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import logging
import pathlib
import statistics
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_v2 as tv2  # noqa: E402
from opt import augment  # noqa: E402
from opt.train_v3 import build_optimizer, seed_everything, setup_backend  # noqa: E402

_LOG = logging.getLogger("opt.w1_matrix")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _REPO_ROOT / "opt" / "results"
CKPT_DIR = pathlib.Path("~/training/ckpt").expanduser()

DEFAULT_TRAIN_GLOBS = "/home/kiwoos/training/ds_wide/ep_*,/home/kiwoos/training/smoke/ep_*"
DEFAULT_VAL_GLOBS = (
    "/home/kiwoos/training/ds_wide/ep_8,"
    "/home/kiwoos/training/ds_wide/ep_9,"
    "/home/kiwoos/training/ds_wide/ep_10"
)


@dataclasses.dataclass(frozen=True)
class Condition:
    """One regularizer setting screened by the matrix.

    Attributes:
        name: Short identifier used in the CSV / report.
        shift_pad: DrQ random-shift radius in px (0 disables).
        proprio_dropout: Per-sample state-zeroing probability (0 disables).
    """

    name: str
    shift_pad: int
    proprio_dropout: float


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Metrics for a single (condition, K, seed) training run (m/s L1)."""

    condition: str
    k: int
    seed: int
    shift_pad: int
    proprio_dropout: float
    best_first: float
    best_chunk: float
    final_first: float
    final_chunk: float
    final_train_l1: float
    final_val_l1: float
    wall_s: float
    throughput_fps: float


@dataclasses.dataclass
class _GpuData:
    """Normalized train/val tensors + norm stats resident on the GPU for one K."""

    imt: torch.Tensor
    stt: torch.Tensor
    act_t: torch.Tensor
    imv: torch.Tensor
    stv: torch.Tensor
    act_v: torch.Tensor
    stats: "tv2.NormStats"
    n_train: int
    n_val: int


def _expand(spec: str) -> list[str]:
    """Expand comma-separated globs into a sorted, unique directory list."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.extend(sorted(glob.glob(str(pathlib.Path(part).expanduser()))))
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def _build_data(train_dirs: list[str], val_dirs: list[str], img: int, k: int) -> _GpuData:
    """Load + normalize train/val tensors onto the GPU for one chunk length K."""
    imt, stt, act_t, _ = tv2.load_all(train_dirs, img, k)
    imv, stv, act_v, _ = tv2.load_all(val_dirs, img, k)
    stats = tv2.compute_norm_stats(stt, act_t)
    return _GpuData(
        imt=imt,
        stt=(stt - stats.smean) / stats.sstd,
        act_t=(act_t - stats.amean) / stats.astd,
        imv=imv,
        stv=(stv - stats.smean) / stats.sstd,
        act_v=(act_v - stats.amean) / stats.astd,
        stats=stats,
        n_train=imt.shape[0],
        n_val=imv.shape[0],
    )


def _reset_model(model: torch.nn.Module, k: int, seed: int) -> None:
    """Re-initialize ``model`` in place from a freshly-seeded ``Policy(k)``.

    Keeps the underlying parameter/buffer tensors (and thus any compiled graph)
    but overwrites their values with a new random init, so each run starts fresh
    without recompiling. ``seed_everything(seed)`` first makes the init and the
    subsequent training stream reproducible.

    Args:
        model: The live (possibly compiled) policy to reset.
        k: Chunk length used to build the reference model.
        seed: RNG seed for the fresh initialization.
    """
    seed_everything(seed)
    fresh = tv2.Policy(k)
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(fresh.state_dict())


@torch.no_grad()
def _evaluate(model: torch.nn.Module, data: _GpuData, bs: int) -> tuple[float, float, float]:
    """Return (first-action L1, full-chunk L1 in m/s, normalized val L1)."""
    model.eval()
    astd = data.stats.astd
    first_err = 0.0
    chunk_err = 0.0
    norm_l1 = torch.zeros((), device=DEVICE)
    nv = data.n_val
    for i in range(0, nv, bs):
        b = slice(i, i + bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(data.imv[b], data.stv[b]).float()
        diff = (pred - data.act_v[b]).abs() * astd  # de-normalized (nb, K, 6)
        first_err += diff[:, 0].mean(1).sum().item()
        chunk_err += diff.mean((1, 2)).sum().item()
        norm_l1 += F.l1_loss(pred, data.act_v[b].float(), reduction="sum")
    denom = nv * data.act_v.shape[1] * data.act_v.shape[2]
    return first_err / nv, chunk_err / nv, (norm_l1 / denom).item()


def train_one(
    model: torch.nn.Module,
    data: _GpuData,
    cond: Condition,
    k: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    bs: int,
) -> RunResult:
    """Train a single (condition, K, seed) run and return its screening metrics.

    Args:
        model: Live (possibly compiled) policy, reset in place before training.
        data: GPU-resident normalized tensors for this K.
        cond: Regularizer setting to apply to training batches only.
        k: Chunk length (for the record and reset).
        seed: RNG seed.
        epochs: Fixed epoch budget.
        lr: AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        bs: Mini-batch size.

    Returns:
        A :class:`RunResult` with best-over-training and final-epoch L1 metrics.
    """
    _reset_model(model, k, seed)
    opt = build_optimizer(model, lr, weight_decay, fused=True)
    aug_gen = torch.Generator(device=DEVICE)
    aug_gen.manual_seed(seed)
    n = data.n_train

    best_first, best_chunk = float("inf"), float("inf")
    final_first = final_chunk = final_val_l1 = 0.0
    last_train_l1 = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(epochs):
        model.train(True)
        idx = torch.randperm(n, device=DEVICE)
        tot = torch.zeros((), device=DEVICE)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            img_b, st_b = data.imt[b], data.stt[b]
            if cond.shift_pad > 0:
                img_b = augment.random_shift(img_b, cond.shift_pad, aug_gen)
            if cond.proprio_dropout > 0.0:
                st_b = augment.proprio_dropout(st_b, cond.proprio_dropout, aug_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.l1_loss(model(img_b, st_b), data.act_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)
        last_train_l1 = (tot / n).item()
        if ep % 5 == 0 or ep == epochs - 1:
            final_first, final_chunk, final_val_l1 = _evaluate(model, data, bs)
            best_first = min(best_first, final_first)
            best_chunk = min(best_chunk, final_chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0

    return RunResult(
        condition=cond.name, k=k, seed=seed, shift_pad=cond.shift_pad,
        proprio_dropout=cond.proprio_dropout, best_first=best_first,
        best_chunk=best_chunk, final_first=final_first, final_chunk=final_chunk,
        final_train_l1=last_train_l1, final_val_l1=final_val_l1, wall_s=wall,
        throughput_fps=epochs * n / wall,
    )


def _save_checkpoint(model: torch.nn.Module, data: _GpuData, k: int, img: int, path: pathlib.Path) -> None:
    """Save the live model's weights in DeployACT-compatible format."""
    target = getattr(model, "_orig_mod", model)
    tv2.save_checkpoint(str(path), target.state_dict(), data.stats, k, img)


def base_conditions() -> list[Condition]:
    """The five conditions run before the shift/dropout combo is chosen."""
    return [
        Condition("baseline", 0, 0.0),
        Condition("shift4", 4, 0.0),
        Condition("shift8", 8, 0.0),
        Condition("dropout0.5", 0, 0.5),
        Condition("dropout0.8", 0, 0.8),
    ]


def run_matrix(
    train_globs: str,
    val_globs: str,
    ks: list[int],
    seeds: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    bs: int,
    img: int,
    compile_mode: str,
    save_ckpt: bool,
) -> list[RunResult]:
    """Run the full W1 matrix and write CSV + report; return all run results."""
    setup_backend()
    val_dirs = _expand(val_globs)
    train_dirs = [d for d in _expand(train_globs) if d not in set(val_dirs)]
    if not train_dirs or not val_dirs:
        raise FileNotFoundError(
            f"need train and val episodes; got train={len(train_dirs)} val={len(val_dirs)}"
        )
    _LOG.info("train eps=%d val eps=%d", len(train_dirs), len(val_dirs))

    results: list[RunResult] = []
    for k in ks:
        _LOG.info("=== K=%d: loading data ===", k)
        data = _build_data(train_dirs, val_dirs, img, k)
        _LOG.info("K=%d train frames=%d val frames=%d", k, data.n_train, data.n_val)
        model = tv2.Policy(k).to(DEVICE)
        if compile_mode != "none":
            model = torch.compile(model, mode=compile_mode)

        # Track the best FINAL-epoch first-action run for this K so its exact
        # (final-weight) checkpoint can be saved in place, no extra re-run.
        best_final = float("inf")
        best_tag = ""

        def _maybe_save(res: RunResult) -> None:
            nonlocal best_final, best_tag
            if save_ckpt and res.final_first < best_final:
                best_final = res.final_first
                best_tag = f"{res.condition} seed={res.seed}"
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                _save_checkpoint(model, data, k, img, CKPT_DIR / f"w1_best_k{k}.pt")

        # Phase 1: five base conditions across seeds.
        for cond in base_conditions():
            for seed in seeds:
                res = train_one(model, data, cond, k, seed, epochs, lr, weight_decay, bs)
                results.append(res)
                _maybe_save(res)
                _LOG.info(
                    "K=%d %-11s seed=%d  first=%.5f chunk=%.5f  %.1fs %.0ffr/s",
                    k, cond.name, seed, res.best_first, res.best_chunk,
                    res.wall_s, res.throughput_fps,
                )

        # Phase 2: shift+dropout combo, using the better shift level for this K.
        best_shift = _pick_best_shift(results, k)
        combo = Condition(f"shift{best_shift}+dropout0.5", best_shift, 0.5)
        for seed in seeds:
            res = train_one(model, data, combo, k, seed, epochs, lr, weight_decay, bs)
            results.append(res)
            _maybe_save(res)
            _LOG.info(
                "K=%d %-16s seed=%d  first=%.5f chunk=%.5f  %.1fs %.0ffr/s",
                k, combo.name, seed, res.best_first, res.best_chunk,
                res.wall_s, res.throughput_fps,
            )
        if save_ckpt:
            _LOG.info("K=%d saved w1_best_k%d.pt from %s (final first=%.5f)",
                      k, k, best_tag, best_final)

        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_csv(results)
    _write_report(results, ks, seeds, epochs, lr, weight_decay)
    return results


def _pick_best_shift(results: list[RunResult], k: int) -> int:
    """Choose the shift radius (4 or 8) with lower mean first-action L1 at K."""
    means = {}
    for pad, name in ((4, "shift4"), (8, "shift8")):
        vals = [r.best_first for r in results if r.k == k and r.condition == name]
        means[pad] = statistics.mean(vals) if vals else float("inf")
    return 4 if means[4] <= means[8] else 8


def _agg(results: list[RunResult], cond: str, k: int) -> tuple[float, float, float, float, int]:
    """Return (first_mean, first_std, chunk_mean, chunk_std, n) over seeds."""
    rs = [r for r in results if r.condition == cond and r.k == k]
    if not rs:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    firsts = [r.best_first for r in rs]
    chunks = [r.best_chunk for r in rs]
    fs = statistics.stdev(firsts) if len(firsts) > 1 else 0.0
    cs = statistics.stdev(chunks) if len(chunks) > 1 else 0.0
    return statistics.mean(firsts), fs, statistics.mean(chunks), cs, len(rs)


def _write_csv(results: list[RunResult]) -> pathlib.Path:
    """Write one row per (condition, K, seed) to opt/results/w1_matrix.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "w1_matrix.csv"
    fields = [
        "condition", "k", "seed", "shift_pad", "proprio_dropout",
        "best_first_l1_mps", "best_chunk_l1_mps", "final_first_l1_mps",
        "final_chunk_l1_mps", "final_train_l1", "final_val_l1_norm",
        "wall_s", "throughput_fps",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r.condition, r.k, r.seed, r.shift_pad, r.proprio_dropout,
                f"{r.best_first:.6f}", f"{r.best_chunk:.6f}", f"{r.final_first:.6f}",
                f"{r.final_chunk:.6f}", f"{r.final_train_l1:.6f}", f"{r.final_val_l1:.6f}",
                f"{r.wall_s:.1f}", f"{r.throughput_fps:.0f}",
            ])
    return path


def _ordered_conditions(results: list[RunResult]) -> list[str]:
    """Condition names in first-seen order (baseline first, combo last)."""
    order: list[str] = []
    for r in results:
        if r.condition not in order:
            order.append(r.condition)
    return order


def _write_report(
    results: list[RunResult],
    ks: list[int],
    seeds: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> pathlib.Path:
    """Write the markdown summary (verdict table + per-K detail)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "w1_report.md"
    conds = _ordered_conditions(results)
    lines: list[str] = []
    lines.append("# W1 regularizer screening — random-shift + proprioception dropout")
    lines.append("")
    lines.append(
        f"Canonical protocol: train = all eps except val (ds_wide/ep_8,9,10), "
        f"budget {epochs} epochs, lr {lr:g}, wd {weight_decay:g}, bs 256, img 128, "
        f"K in {{{', '.join(str(k) for k in ks)}}}, seeds {seeds}."
    )
    lines.append("")
    lines.append(
        "> SECONDARY screening metrics only (val first-action L1 and full-chunk "
        "L1, m/s). Winners must be confirmed on the in-sim scored suite before "
        "adoption (CLAUDE.md section 6). Full-chunk L1 is NOT comparable across "
        "different K (larger K averages over farther, harder future steps)."
    )
    lines.append("")

    # Verdict table: experiment | verdict | metric.
    lines.append("## Summary (experiment | verdict | metric)")
    lines.append("")
    lines.append(
        "| experiment | K | verdict | first-action L1 (m/s) mean±std | "
        "full-chunk L1 (m/s) mean±std |"
    )
    lines.append("|---|---|---|---|---|")
    for k in ks:
        bmean, _, _, _, _ = _agg(results, "baseline", k)
        for cond in conds:
            fmean, fstd, cmean, cstd, n = _agg(results, cond, k)
            if n == 0:
                continue
            if cond == "baseline":
                verdict = "baseline"
            else:
                delta = 100.0 * (fmean - bmean) / bmean if bmean else float("nan")
                if delta <= -3.0:
                    verdict = f"win {delta:+.1f}%"
                elif delta >= 3.0:
                    verdict = f"loss {delta:+.1f}%"
                else:
                    verdict = f"~tie {delta:+.1f}%"
            lines.append(
                f"| {cond} | {k} | {verdict} | {fmean:.5f} ± {fstd:.5f} | "
                f"{cmean:.5f} ± {cstd:.5f} |"
            )
    lines.append("")
    lines.append(
        "_Verdict is vs the same-K baseline on first-action L1 (win/loss "
        "threshold ±3%). These are diagnostics, not the primary suite score._"
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the W1 screening matrix."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="W1 regularizer screening matrix.")
    ap.add_argument("--train", dest="train_globs", default=DEFAULT_TRAIN_GLOBS)
    ap.add_argument("--val", dest="val_globs", default=DEFAULT_VAL_GLOBS)
    ap.add_argument("--ks", default="8,16", help="comma-separated chunk lengths")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..n-1)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--compile-mode", default="default",
                    choices=["none", "default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--no-ckpt", dest="save_ckpt", action="store_false")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("run_matrix requires a CUDA device")
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    seeds = list(range(a.seeds))
    t0 = time.time()
    results = run_matrix(
        a.train_globs, a.val_globs, ks, seeds, a.epochs, a.lr, a.weight_decay,
        a.bs, a.img, a.compile_mode, a.save_ckpt,
    )
    print(f"\n[w1_matrix] {len(results)} runs, wall={time.time() - t0:.0f}s")
    print(f"[w1_matrix] wrote {RESULTS_DIR/'w1_matrix.csv'} and {RESULTS_DIR/'w1_report.md'}")


if __name__ == "__main__":
    main()
