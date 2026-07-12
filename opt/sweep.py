"""Budgeted hyperparameter search (ASHA-style successive halving) for ACT-lite.

Self-contained (no Optuna/Ray dependency, neither of which is installed on this
image): random-samples configs over {lr, chunk K, img size, weight decay, EMA
decay}, then runs successive halving with **trial continuation** -- every config
trains to a cheap first rung, the worst 1-1/eta are killed, and survivors keep
training (warm, from their existing weights) to the next rung, per Li et al.
(ASHA, arXiv:1810.05934) and Hyperband (arXiv:1603.06560). The selection metric
is held-out validation first-action L1 error (m/s), matching train_v2's report.

Held-out val defaults to ds_wide/ep_8,9,10 (train_exp.py's split). Results are
written to opt/results/sweep.csv (one row per trial-rung evaluation) and
opt/results/leaderboard.md. A baseline (train_v2 defaults) is trained to the full
budget for reference. Deterministic given --seed.

CLI example:
    ~/miniconda3/bin/python -m opt.sweep --n-configs 8 --min-epochs 4 \\
        --max-epochs 36 --eta 3 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import logging
import pathlib
import sys
import time

# Only the repo root is added to sys.path (for train_v2); torch comes from the
# active interpreter (run with ~/miniconda3/bin/python).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_v2 as tv2  # noqa: E402
from opt import asha  # noqa: E402
from opt.config import TrialConfig, TrialResult  # noqa: E402
from opt.train_v3 import (  # noqa: E402
    EmaWeights, build_optimizer, seed_everything, setup_backend,
)

_LOG = logging.getLogger("opt.sweep")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _REPO_ROOT / "opt" / "results"

# Discrete search space (random search + successive halving is the ASHA default).
SEARCH_SPACE: dict[str, list[object]] = {
    "lr": [1e-4, 3e-4, 1e-3],
    "k": [8, 16, 24],
    "img": [96, 128],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "ema_decay": [0.0, 0.999],
}


@dataclasses.dataclass
class _GpuDataset:
    """Normalized train/val tensors resident on the GPU for one (img, k)."""

    imt: torch.Tensor
    stt: torch.Tensor
    act_t: torch.Tensor
    imv: torch.Tensor
    stv: torch.Tensor
    act_v: torch.Tensor
    astd: torch.Tensor
    n_train: int


@dataclasses.dataclass
class _TrialState:
    """Live training state carried across rungs for one config."""

    result: TrialResult
    model: torch.nn.Module
    opt: torch.optim.Optimizer
    ema: EmaWeights | None
    key: tuple[int, int]


def _expand(spec: str) -> list[str]:
    """Expand comma-separated globs into a sorted, unique directory list."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.extend(sorted(glob.glob(str(pathlib.Path(part).expanduser()))))
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def _build_dataset(
    train_dirs: list[str], val_dirs: list[str], img: int, k: int
) -> _GpuDataset:
    """Load and normalize train/val tensors onto the GPU for one (img, k)."""
    imt, stt, act_t, _ = tv2.load_all(train_dirs, img, k)
    imv, stv, act_v, _ = tv2.load_all(val_dirs, img, k)
    smean, sstd = stt.mean(0), stt.std(0) + 1e-6
    amean, astd = act_t.reshape(-1, 6).mean(0), act_t.reshape(-1, 6).std(0) + 1e-6
    return _GpuDataset(
        imt=imt, stt=(stt - smean) / sstd, act_t=(act_t - amean) / astd,
        imv=imv, stv=(stv - smean) / sstd, act_v=(act_v - amean) / astd,
        astd=astd, n_train=imt.shape[0],
    )


def _train_epochs(state: _TrialState, ds: _GpuDataset, epochs: int, bs: int) -> float:
    """Train ``state`` for ``epochs`` more epochs; return last train L1."""
    model = state.model
    model.train(True)
    n = ds.n_train
    last = 0.0
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        tot = torch.zeros((), device=DEVICE)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.l1_loss(model(ds.imt[b], ds.stt[b]), ds.act_t[b])
            state.opt.zero_grad(set_to_none=True)
            loss.backward()
            state.opt.step()
            if state.ema is not None:
                state.ema.update(model)
            tot += loss.detach() * len(b)
        last = (tot / n).item()
    return last


@torch.no_grad()
def _evaluate(state: _TrialState, ds: _GpuDataset, bs: int) -> tuple[float, float]:
    """Return (val first-action L1 in m/s, normalized val L1) for ``state``."""
    model = state.model
    if state.ema is not None:
        state.ema.apply_to(model)
    model.eval()
    err = 0.0
    l1 = torch.zeros((), device=DEVICE)
    nv = ds.imv.shape[0]
    for i in range(0, nv, bs):
        b = slice(i, i + bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ds.imv[b], ds.stv[b]).float()
        err += ((pred[:, 0] - ds.act_v[b][:, 0]).abs() * ds.astd).mean(1).sum().item()
        l1 += F.l1_loss(pred, ds.act_v[b].float(), reduction="sum")
    if state.ema is not None:
        state.ema.restore(model)
    return err / nv, (l1 / (nv * ds.act_v.shape[1] * ds.act_v.shape[2])).item()


def run_sweep(
    train_globs: str,
    val_globs: str,
    n_configs: int,
    min_epochs: int,
    max_epochs: int,
    eta: int,
    bs: int,
    seed: int,
) -> tuple[list[TrialResult], float]:
    """Run the ASHA-style sweep and return (all trial results, baseline metric).

    Args:
        train_globs: Comma-separated training-episode globs.
        val_globs: Comma-separated held-out validation globs.
        n_configs: Number of configs to sample for the search.
        min_epochs: First-rung epoch budget.
        max_epochs: Final-rung epoch budget.
        eta: Successive-halving reduction factor.
        bs: Batch size.
        seed: Base RNG seed (trial i uses seed + i).

    Returns:
        Tuple of (final results for every sampled trial, baseline first-action
        L1 in m/s from the train_v2-default config trained to max_epochs).

    Raises:
        FileNotFoundError: If no training or validation episodes are found.
    """
    setup_backend()
    val_dirs = _expand(val_globs)
    train_dirs = [d for d in _expand(train_globs) if d not in set(val_dirs)]
    if not train_dirs or not val_dirs:
        raise FileNotFoundError(
            f"need both train and val episodes; got train={len(train_dirs)} "
            f"val={len(val_dirs)}"
        )
    _LOG.info("train eps=%d  val eps=%d", len(train_dirs), len(val_dirs))

    rungs = asha.rung_budgets(min_epochs, max_epochs, eta)
    _LOG.info("rung budgets (epochs): %s  eta=%d  n_configs=%d", rungs, eta, n_configs)

    sampled = asha.sample_configs(n_configs, SEARCH_SPACE, seed)
    dataset_cache: dict[tuple[int, int], _GpuDataset] = {}

    def get_dataset(img: int, k: int) -> _GpuDataset:
        key = (img, k)
        if key not in dataset_cache:
            _LOG.info("loading dataset img=%d k=%d ...", img, k)
            dataset_cache[key] = _build_dataset(train_dirs, val_dirs, img, k)
        return dataset_cache[key]

    # Build a live state per config.
    states: dict[int, _TrialState] = {}
    for tid, raw in enumerate(sampled):
        cfg = TrialConfig(
            trial_id=tid, lr=float(raw["lr"]), k=int(raw["k"]), img=int(raw["img"]),
            weight_decay=float(raw["weight_decay"]),
            ema_decay=float(raw["ema_decay"]), seed=seed + tid,
        )
        seed_everything(cfg.seed)
        model = tv2.Policy(cfg.k).to(DEVICE)
        opt = build_optimizer(model, cfg.lr, cfg.weight_decay, fused=True)
        ema = EmaWeights(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
        states[tid] = _TrialState(
            result=TrialResult(config=cfg), model=model, opt=opt, ema=ema,
            key=(cfg.img, cfg.k),
        )

    csv_rows: list[dict[str, object]] = []
    alive_ids = list(states.keys())
    prev_budget = 0
    for rung, budget in enumerate(rungs):
        add_epochs = budget - prev_budget
        prev_budget = budget
        scores: dict[int, float] = {}
        for tid in alive_ids:
            st = states[tid]
            ds = get_dataset(*st.key)
            seed_everything(st.result.config.seed + rung)  # per-rung determinism
            t0 = time.time()
            train_l1 = _train_epochs(st, ds, add_epochs, bs)
            first, val_l1 = _evaluate(st, ds, bs)
            st.result.wall_s += time.time() - t0
            st.result.epochs_done = budget
            st.result.train_l1 = train_l1
            st.result.val_l1 = val_l1
            st.result.val_first_action = first
            st.result.history.append((budget, first))
            scores[tid] = first
            csv_rows.append({
                "trial": tid, "rung": rung, "epochs": budget,
                "lr": st.result.config.lr, "k": st.result.config.k,
                "img": st.result.config.img, "wd": st.result.config.weight_decay,
                "ema": st.result.config.ema_decay, "train_l1": round(train_l1, 5),
                "val_l1": round(val_l1, 5), "val_first_action": round(first, 6),
            })
            _LOG.info("rung %d trial %d (lr=%.0e k=%d img=%d wd=%.0e ema=%.3f) "
                      "epochs=%d val_first=%.5f", rung, tid, st.result.config.lr,
                      st.result.config.k, st.result.config.img,
                      st.result.config.weight_decay, st.result.config.ema_decay,
                      budget, first)

        if rung < len(rungs) - 1:
            keep = set(asha.survivors(scores, eta))
            for tid in alive_ids:
                if tid not in keep:
                    states[tid].result.alive = False
                    states[tid].model = None  # type: ignore[assignment]
                    states[tid].opt = None  # type: ignore[assignment]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            alive_ids = [t for t in alive_ids if t in keep]
            _LOG.info("rung %d survivors -> %s", rung, alive_ids)

    _write_csv(csv_rows)
    baseline = _train_baseline(train_dirs, val_dirs, max_epochs, bs, seed)
    results = [st.result for st in states.values()]
    _write_leaderboard(results, baseline)
    return results, baseline


def _train_baseline(
    train_dirs: list[str], val_dirs: list[str], epochs: int, bs: int, seed: int
) -> float:
    """Train the train_v2-default config to full budget; return val first-action."""
    _LOG.info("training baseline (train_v2 defaults) to %d epochs ...", epochs)
    ds = _build_dataset(train_dirs, val_dirs, img=128, k=16)
    seed_everything(seed)
    model = tv2.Policy(16).to(DEVICE)
    opt = build_optimizer(model, lr=3e-4, weight_decay=0.0, fused=True)
    state = _TrialState(
        result=TrialResult(config=TrialConfig(
            trial_id=-1, lr=3e-4, k=16, img=128, weight_decay=0.0,
            ema_decay=0.0, seed=seed)),
        model=model, opt=opt, ema=None, key=(128, 16),
    )
    _train_epochs(state, ds, epochs, bs)
    first, _ = _evaluate(state, ds, bs)
    _LOG.info("baseline val first-action = %.5f m/s", first)
    return first


def _write_csv(rows: list[dict[str, object]]) -> pathlib.Path:
    """Write per-trial-rung rows to opt/results/sweep.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "sweep.csv"
    fields = ["trial", "rung", "epochs", "lr", "k", "img", "wd", "ema",
              "train_l1", "val_l1", "val_first_action"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_leaderboard(results: list[TrialResult], baseline: float) -> pathlib.Path:
    """Write the markdown leaderboard to opt/results/leaderboard.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "leaderboard.md"
    path.write_text(asha.render_leaderboard(results, baseline), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the sweep harness."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="ASHA-style ACT-lite hyperparameter sweep.")
    ap.add_argument("--train", dest="train_globs",
                    default="/home/kiwoos/training/ds_wide/ep_*")
    ap.add_argument("--val", dest="val_globs",
                    default="/home/kiwoos/training/ds_wide/ep_8,"
                            "/home/kiwoos/training/ds_wide/ep_9,"
                            "/home/kiwoos/training/ds_wide/ep_10")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=36)
    ap.add_argument("--eta", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("run_sweep requires a CUDA device")
    t0 = time.time()
    results, baseline = run_sweep(
        a.train_globs, a.val_globs, a.n_configs, a.min_epochs, a.max_epochs,
        a.eta, a.bs, a.seed,
    )
    best = min(results, key=lambda r: r.val_first_action)
    print(f"\n[sweep] wall={time.time() - t0:.0f}s  baseline val first-action="
          f"{baseline:.5f} m/s")
    print(f"[sweep] BEST trial {best.config.trial_id}: lr={best.config.lr:.0e} "
          f"k={best.config.k} img={best.config.img} wd={best.config.weight_decay:.0e} "
          f"ema={best.config.ema_decay:.3f} -> val first-action="
          f"{best.val_first_action:.5f} m/s "
          f"({100*(best.val_first_action-baseline)/baseline:+.1f}% vs baseline)")
    print(f"[sweep] wrote {RESULTS_DIR/'sweep.csv'} and {RESULTS_DIR/'leaderboard.md'}")


if __name__ == "__main__":
    main()
