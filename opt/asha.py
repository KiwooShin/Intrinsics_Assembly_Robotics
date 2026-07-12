"""Pure successive-halving (ASHA-style) scheduler math and leaderboard rendering.

Implements the budget/promotion arithmetic behind Successive Halving / Hyperband
(Li et al., 2018, arXiv:1810.05934 "A System for Massively Parallel Hyperparameter
Tuning") in a form that is deterministic and testable without torch or a GPU. The
sweep harness (opt/sweep.py) supplies the actual train-and-evaluate function; this
module only decides how many epochs each rung gets and which trials survive it.

Only the standard library is used so the unit tests run CPU-only.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from opt.config import TrialResult


def rung_budgets(min_epochs: int, max_epochs: int, eta: int) -> list[int]:
    """Compute the per-rung cumulative epoch budgets for successive halving.

    Budgets grow geometrically by ``eta`` from ``min_epochs`` up to and
    including ``max_epochs``: e.g. min=2, max=32, eta=3 -> [2, 6, 18, 32].

    Args:
        min_epochs: Epochs the first (cheapest) rung trains every trial to.
        max_epochs: Epochs the final rung trains survivors to.
        eta: Reduction factor (>= 2); also the geometric budget multiplier.

    Returns:
        Strictly increasing list of cumulative epoch budgets ending at
        ``max_epochs``.

    Raises:
        ValueError: If the arguments are inconsistent.
    """
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}")
    if min_epochs < 1:
        raise ValueError(f"min_epochs must be >= 1, got {min_epochs}")
    if max_epochs < min_epochs:
        raise ValueError(f"max_epochs ({max_epochs}) < min_epochs ({min_epochs})")
    budgets: list[int] = []
    b = min_epochs
    while b < max_epochs:
        budgets.append(b)
        b *= eta
    budgets.append(max_epochs)
    return budgets


def survivors(
    scores: Mapping[int, float],
    eta: int,
    lower_is_better: bool = True,
) -> list[int]:
    """Select the trials that advance to the next rung.

    Keeps the best ``ceil(n / eta)`` trials, matching successive halving's
    top-1/eta promotion rule.

    Args:
        scores: Mapping of trial id -> validation metric at the current rung.
        eta: Reduction factor; a 1/eta fraction is promoted.
        lower_is_better: If True (default, e.g. L1 error) smaller scores win.

    Returns:
        Trial ids of survivors, ordered best-first.

    Raises:
        ValueError: If ``eta`` < 2 or ``scores`` is empty.
    """
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}")
    if not scores:
        raise ValueError("scores must be non-empty")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    keep = max(1, math.ceil(len(ranked) / eta))
    return [trial_id for trial_id, _ in ranked[:keep]]


def sample_configs(
    n: int,
    space: Mapping[str, Sequence[object]],
    seed: int,
) -> list[dict[str, object]]:
    """Deterministically random-sample ``n`` hyperparameter dicts from a grid.

    Random search over a discrete space is the ASHA/Hyperband default base
    sampler (it dominates grid search under a fixed budget; Bergstra & Bengio
    2012, "Random Search for Hyper-Parameter Optimization").

    Args:
        n: Number of configs to draw.
        space: Mapping of hyperparameter name -> list of candidate values.
        seed: RNG seed for reproducibility.

    Returns:
        List of ``n`` dicts, one value drawn per hyperparameter.

    Raises:
        ValueError: If ``n`` < 1 or any candidate list is empty.
    """
    import random  # stdlib, local to keep module import side-effect free

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for key, values in space.items():
        if not values:
            raise ValueError(f"candidate list for '{key}' is empty")
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for _ in range(n):
        out.append({key: rng.choice(list(values)) for key, values in space.items()})
    return out


def render_leaderboard(results: Iterable[TrialResult], baseline: float | None = None) -> str:
    """Render a markdown leaderboard sorted by validation first-action error.

    Args:
        results: Trial results to rank (best/lowest first-action error first).
        baseline: Optional baseline first-action L1 (m/s) to show a delta
            column against; ``None`` omits the column.

    Returns:
        A markdown document string with a ranked table.
    """
    ranked = sorted(results, key=lambda r: r.val_first_action)
    lines: list[str] = ["# Sweep leaderboard", ""]
    header = "| rank | trial | lr | K | img | wd | ema | epochs | val_first|err|(m/s) | val_L1 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if baseline is not None:
        header += " vs_baseline |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)
    for rank, r in enumerate(ranked, 1):
        c = r.config
        row = (
            f"| {rank} | {c.trial_id} | {c.lr:.1e} | {c.k} | {c.img} | "
            f"{c.weight_decay:.1e} | {c.ema_decay:.3f} | {r.epochs_done} | "
            f"{r.val_first_action:.5f} | {r.val_l1:.4f} |"
        )
        if baseline is not None:
            delta = r.val_first_action - baseline
            pct = 100.0 * delta / baseline if baseline > 0 else 0.0
            row += f" {delta:+.5f} ({pct:+.1f}%) |"
        lines.append(row)
    if baseline is not None:
        lines += ["", f"_Baseline (train_v2 defaults) val first-action = {baseline:.5f} m/s._"]
    lines.append("")
    return "\n".join(lines)
