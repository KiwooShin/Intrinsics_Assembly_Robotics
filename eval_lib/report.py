"""Aggregation, reporting, and paired comparison for eval-suite results.

Produces (per CLAUDE.md reporting rules): an experiment-table row, mean + IQM
with stratified-bootstrap 95% CIs of total score, insertion success rate with a
Wilson CI, a per-stratum (rail x plug) breakdown, and a paired-comparison mode
whose verdict is significant iff the paired-bootstrap CI of the score difference
excludes zero. Outputs a markdown report + CSV + machine-readable JSON.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from . import stats
from .runner import TrialResult

RESULTS_CSV = "results.csv"
REPORT_MD = "report.md"
SUMMARY_JSON = "summary.json"

_CSV_FIELDS = [f.name for f in dataclasses.fields(TrialResult)]


@dataclasses.dataclass(frozen=True)
class Aggregate:
    """Suite-level aggregate statistics for one evaluation run.

    Attributes:
        name: Human-readable run name.
        n: Number of scored configs.
        mean: Mean total score with stratified-bootstrap CI.
        iqm: IQM total score with stratified-bootstrap CI.
        success: Insertion success rate with Wilson CI.
        n_success: Number of full insertions.
        outcome_counts: Count of each outcome label.
        per_stratum: Mapping of ``rail x plug`` label to (n, mean_total,
            n_success).
    """

    name: str
    n: int
    mean: stats.ConfidenceInterval
    iqm: stats.ConfidenceInterval
    success: stats.ConfidenceInterval
    n_success: int
    outcome_counts: dict[str, int]
    per_stratum: dict[str, tuple[int, float, int]]

    def verdict(self) -> str:
        """Return a one-word success/failure verdict for the experiment table.

        Returns:
            ``"PASS"`` if any full insertion occurred, else ``"FAIL"``.
        """
        return "PASS" if self.n_success > 0 else "FAIL"


def _stratum_label(result: TrialResult) -> str:
    """Return the ``rail x plug`` label used for stratum breakdowns.

    Args:
        result: A trial result row.

    Returns:
        A label such as ``"rail2_SFP"``.
    """
    return f"rail{result.rail}_{result.plug}"


def aggregate(results: list[TrialResult], name: str, seed: int = 0) -> Aggregate:
    """Compute suite-level aggregate statistics.

    Args:
        results: Per-config result rows.
        name: Run name for the report.
        seed: Bootstrap seed (determinism).

    Returns:
        The :class:`Aggregate`.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("cannot aggregate an empty result set")
    totals = np.array([r.total for r in results], dtype=float)
    strata = np.array([_stratum_label(r) for r in results])
    n = len(results)
    n_success = sum(1 for r in results if r.insertion_success)

    mean_ci = stats.stratified_bootstrap_ci(
        totals, strata, statistic=stats.mean_stat, seed=seed
    )
    if n >= 3:
        iqm_ci = stats.stratified_bootstrap_ci(
            totals, strata, statistic=stats.iqm, seed=seed + 1
        )
    else:
        point = float(np.mean(totals))
        iqm_ci = stats.ConfidenceInterval(point=point, lo=point, hi=point)
    success_ci = stats.wilson_ci(n_success, n)

    outcome_counts: dict[str, int] = {}
    for r in results:
        outcome_counts[r.outcome] = outcome_counts.get(r.outcome, 0) + 1

    per_stratum: dict[str, tuple[int, float, int]] = {}
    for label in sorted(set(strata.tolist())):
        rows = [r for r in results if _stratum_label(r) == label]
        cell_totals = [r.total for r in rows]
        cell_success = sum(1 for r in rows if r.insertion_success)
        per_stratum[label] = (
            len(rows),
            float(np.mean(cell_totals)),
            cell_success,
        )
    return Aggregate(
        name=name,
        n=n,
        mean=mean_ci,
        iqm=iqm_ci,
        success=success_ci,
        n_success=n_success,
        outcome_counts=outcome_counts,
        per_stratum=per_stratum,
    )


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Paired comparison between two evaluation runs A and B.

    Attributes:
        name_a: Name of run A.
        name_b: Name of run B.
        n: Number of matched configs.
        mean_diff: Paired-bootstrap CI of ``mean(A - B)``.
        iqm_diff: Paired-bootstrap CI of ``IQM(A - B)``.
        verdict: One of ``"A beats B"``, ``"B beats A"``, ``"inconclusive"``.
        per_config: List of ``(config_id, score_a, score_b, diff)`` tuples.
    """

    name_a: str
    name_b: str
    n: int
    mean_diff: stats.ConfidenceInterval
    iqm_diff: stats.ConfidenceInterval
    verdict: str
    per_config: list[tuple[str, float, float, float]]


def compare(
    results_a: list[TrialResult],
    results_b: list[TrialResult],
    name_a: str = "A",
    name_b: str = "B",
    seed: int = 0,
) -> Comparison:
    """Compare two runs over their matched configs (paired design).

    Args:
        results_a: Result rows from run A.
        results_b: Result rows from run B.
        name_a: Display name for A.
        name_b: Display name for B.
        seed: Bootstrap seed (determinism).

    Returns:
        The :class:`Comparison`.

    Raises:
        ValueError: If the two runs do not cover an identical config set.
    """
    by_a = {r.config_id: r for r in results_a}
    by_b = {r.config_id: r for r in results_b}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        raise ValueError("runs A and B share no config ids; cannot pair")
    if set(by_a) != set(by_b):
        only_a = sorted(set(by_a) - set(by_b))
        only_b = sorted(set(by_b) - set(by_a))
        raise ValueError(
            "paired comparison requires identical config sets; "
            f"only in A: {only_a}; only in B: {only_b}"
        )
    per_config: list[tuple[str, float, float, float]] = []
    diffs = []
    for cid in common:
        sa = by_a[cid].total
        sb = by_b[cid].total
        per_config.append((cid, sa, sb, sa - sb))
        diffs.append(sa - sb)
    diff_arr = np.array(diffs, dtype=float)
    mean_diff = stats.paired_bootstrap_ci(diff_arr, statistic=stats.mean_stat, seed=seed)
    if diff_arr.size >= 3:
        iqm_diff = stats.paired_bootstrap_ci(diff_arr, statistic=stats.iqm, seed=seed + 1)
    else:
        point = float(np.mean(diff_arr))
        iqm_diff = stats.ConfidenceInterval(point=point, lo=point, hi=point)
    verdict = _compare_verdict(mean_diff, name_a, name_b)
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        n=len(common),
        mean_diff=mean_diff,
        iqm_diff=iqm_diff,
        verdict=verdict,
        per_config=per_config,
    )


def _compare_verdict(
    mean_diff: stats.ConfidenceInterval, name_a: str, name_b: str
) -> str:
    """Derive the comparison verdict from the mean-difference CI.

    Args:
        mean_diff: Paired-bootstrap CI of ``mean(A - B)``.
        name_a: Display name for A.
        name_b: Display name for B.

    Returns:
        ``"<A> beats <B>"``, ``"<B> beats <A>"``, or ``"inconclusive"``.
    """
    if mean_diff.lo > 0.0:
        return f"{name_a} beats {name_b}"
    if mean_diff.hi < 0.0:
        return f"{name_b} beats {name_a}"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# Serialization / rendering
# --------------------------------------------------------------------------- #


def write_results_csv(path: str | Path, results: list[TrialResult]) -> None:
    """Write per-config result rows to CSV.

    Args:
        path: Destination CSV path.
        results: Result rows.
    """
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))


def read_results_csv(path: str | Path) -> list[TrialResult]:
    """Read per-config result rows back from CSV.

    Args:
        path: Path to a ``results.csv`` written by :func:`write_results_csv`.

    Returns:
        The list of :class:`TrialResult`.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"results csv not found: {p}")
    bool_fields = {"inserted", "insertion_success", "completed", "timed_out"}
    float_fields = {
        "total", "tier_1", "tier_2", "tier_3", "force_score", "contacts_score",
        "duration_score", "smoothness_score", "efficiency_score", "duration_s",
    }
    int_fields = {"rail", "port"}
    rows: list[TrialResult] = []
    with p.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            kwargs = {}
            for key, value in raw.items():
                if key in bool_fields:
                    kwargs[key] = str(value).lower() == "true"
                elif key in float_fields:
                    kwargs[key] = float(value)
                elif key in int_fields:
                    kwargs[key] = int(value)
                elif key == "error":
                    kwargs[key] = value or None
                else:
                    kwargs[key] = value
            rows.append(TrialResult(**kwargs))
    return rows


def _fmt_ci(ci: stats.ConfidenceInterval) -> str:
    """Format a CI as ``point [lo, hi]`` with two decimals.

    Args:
        ci: The confidence interval.

    Returns:
        The formatted string.
    """
    return f"{ci.point:.2f} [{ci.lo:.2f}, {ci.hi:.2f}]"


def render_report(agg: Aggregate) -> str:
    """Render a single-run markdown report.

    Args:
        agg: The suite-level aggregate.

    Returns:
        The markdown document as a string.
    """
    lines: list[str] = []
    lines.append(f"# Eval report: {agg.name}")
    lines.append("")
    lines.append("## Experiment table")
    lines.append("")
    lines.append("| Experiment | Verdict | Insertion success | Mean score | IQM score |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {agg.name} | {agg.verdict()} | "
        f"{agg.success.point * 100:.1f}% ({agg.n_success}/{agg.n}) "
        f"[{agg.success.lo * 100:.1f}, {agg.success.hi * 100:.1f}] | "
        f"{_fmt_ci(agg.mean)} | {_fmt_ci(agg.iqm)} |"
    )
    lines.append("")
    lines.append("## Score summary (total per config, max 100)")
    lines.append("")
    lines.append(f"- Configs scored: **{agg.n}**")
    lines.append(f"- Mean total (stratified-bootstrap 95% CI): **{_fmt_ci(agg.mean)}**")
    lines.append(f"- IQM total (stratified-bootstrap 95% CI): **{_fmt_ci(agg.iqm)}**")
    lines.append(
        f"- Insertion success (Wilson 95% CI): "
        f"**{agg.success.point * 100:.1f}%** "
        f"[{agg.success.lo * 100:.1f}%, {agg.success.hi * 100:.1f}%] "
        f"({agg.n_success}/{agg.n})"
    )
    lines.append("")
    lines.append("## Outcome distribution")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("|---|---|")
    for outcome in ("full", "partial", "proximity", "miss", "collision", "force"):
        lines.append(f"| {outcome} | {agg.outcome_counts.get(outcome, 0)} |")
    lines.append("")
    lines.append("## Per-stratum breakdown (rail x plug)")
    lines.append("")
    lines.append("| Stratum | N | Mean total | Insertions |")
    lines.append("|---|---|---|---|")
    for label in sorted(agg.per_stratum):
        count, mean_total, cell_success = agg.per_stratum[label]
        lines.append(f"| {label} | {count} | {mean_total:.2f} | {cell_success}/{count} |")
    lines.append("")
    return "\n".join(lines)


def render_comparison(cmp: Comparison) -> str:
    """Render a paired-comparison markdown report.

    Args:
        cmp: The comparison result.

    Returns:
        The markdown document as a string.
    """
    lines: list[str] = []
    lines.append(f"# Paired comparison: {cmp.name_a} vs {cmp.name_b}")
    lines.append("")
    lines.append(f"**Verdict: {cmp.verdict}**")
    lines.append("")
    lines.append(f"- Matched configs: **{cmp.n}**")
    lines.append(
        f"- Mean score difference A-B (paired-bootstrap 95% CI): "
        f"**{_fmt_ci(cmp.mean_diff)}** "
        f"({'significant' if cmp.mean_diff.excludes(0.0) else 'includes 0'})"
    )
    lines.append(
        f"- IQM score difference A-B (paired-bootstrap 95% CI): "
        f"**{_fmt_ci(cmp.iqm_diff)}** "
        f"({'significant' if cmp.iqm_diff.excludes(0.0) else 'includes 0'})"
    )
    lines.append("")
    lines.append("## Largest per-config differences")
    lines.append("")
    lines.append(f"| Config | {cmp.name_a} | {cmp.name_b} | A-B |")
    lines.append("|---|---|---|---|")
    ordered = sorted(cmp.per_config, key=lambda t: abs(t[3]), reverse=True)
    for cid, sa, sb, diff in ordered[:15]:
        lines.append(f"| {cid} | {sa:.2f} | {sb:.2f} | {diff:+.2f} |")
    lines.append("")
    return "\n".join(lines)


def _ci_dict(ci: stats.ConfidenceInterval) -> dict[str, float]:
    """Return a JSON-serializable representation of a CI.

    Args:
        ci: The confidence interval.

    Returns:
        A dict with ``point``/``lo``/``hi``/``level``.
    """
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "level": ci.level}


def write_report(
    out_dir: str | Path, results: list[TrialResult], name: str, seed: int = 0
) -> Aggregate:
    """Aggregate results and write CSV + markdown + JSON to ``out_dir``.

    Args:
        out_dir: Destination directory.
        results: Per-config result rows.
        name: Run name.
        seed: Bootstrap seed.

    Returns:
        The computed :class:`Aggregate`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_results_csv(out / RESULTS_CSV, results)
    agg = aggregate(results, name=name, seed=seed)
    (out / REPORT_MD).write_text(render_report(agg))
    summary = {
        "name": agg.name,
        "n": agg.n,
        "verdict": agg.verdict(),
        "mean_total": _ci_dict(agg.mean),
        "iqm_total": _ci_dict(agg.iqm),
        "insertion_success_rate": _ci_dict(agg.success),
        "n_success": agg.n_success,
        "outcome_counts": agg.outcome_counts,
        "per_stratum": {k: list(v) for k, v in agg.per_stratum.items()},
    }
    (out / SUMMARY_JSON).write_text(json.dumps(summary, indent=2))
    return agg


def write_comparison(out_dir: str | Path, cmp: Comparison) -> None:
    """Write a comparison markdown report + CSV to ``out_dir``.

    Args:
        out_dir: Destination directory.
        cmp: The comparison result.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(render_comparison(cmp))
    with (out / "compare.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_id", cmp.name_a, cmp.name_b, "diff"])
        for cid, sa, sb, diff in cmp.per_config:
            writer.writerow([cid, f"{sa:.4f}", f"{sb:.4f}", f"{diff:.4f}"])
    (out / "compare.json").write_text(
        json.dumps(
            {
                "name_a": cmp.name_a,
                "name_b": cmp.name_b,
                "n": cmp.n,
                "verdict": cmp.verdict,
                "mean_diff": _ci_dict(cmp.mean_diff),
                "iqm_diff": _ci_dict(cmp.iqm_diff),
            },
            indent=2,
        )
    )
