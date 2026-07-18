"""CLI for the fixed matched-seed evaluation harness.

Subcommands:
    gen      Generate the deterministic stratified suite (YAML configs + manifest).
    run      Run a policy/checkpoint over a suite and emit a scored report.
    compare  Paired comparison of two ``run`` output directories.

Every checkpoint runs the byte-identical suite (paired / matched-seed design,
rliable arXiv:2108.13264, research-plan finding #11). See ``eval_lib`` for the
implementation modules; this file is only argument parsing + orchestration.

Examples:
    python3 eval_suite.py gen --out eval_suite/ --n 50 --seed 20260712
    python3 eval_suite.py gen --out eval_suite_smoke/ --smoke
    python3 eval_suite.py run --checkpoint ckpt/v2_wide.pt \\
        --policy aic_example_policies.ros.CheatCode --suite eval_suite/ \\
        --out results/v2_wide/
    python3 eval_suite.py run --suite eval_suite/ --out results/dry/ --dry-run
    python3 eval_suite.py compare results/A results/B --out results/cmp_A_B/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval_lib import report, runner, suite


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for CLI progress output.

    Args:
        verbose: Emit DEBUG-level logs when True, else INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_gen(args: argparse.Namespace) -> int:
    """Generate and persist the evaluation suite.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on success).
    """
    n = suite.SMOKE_N if args.smoke else args.n
    members = suite.write_suite(
        out_dir=args.out,
        n=n,
        seed=args.seed,
        template_path=args.template,
        include_official=not args.no_official,
    )
    counts = suite.stratum_counts(members)
    n_official = sum(1 for m in members if m.source == "official")
    print(f"Wrote {len(members)} configs to {args.out}")
    print(f"  stratified: {len(members) - n_official}   official: {n_official}")
    print(f"  strata cells covered: {len(counts)}/20")
    print(f"  per-cell counts: min {min(counts.values())}  max {max(counts.values())}")
    print(f"  manifest: {Path(args.out) / 'manifest.csv'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a policy/checkpoint over the suite and write a report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on success).
    """
    name = args.name or Path(args.out).name
    env = (
        runner.SimEnv(policy_launch_cmd=args.policy_cmd)
        if args.policy_cmd is not None
        else None
    )
    results = runner.run_suite(
        suite_dir=args.suite,
        out_dir=args.out,
        policy=args.policy,
        checkpoint=args.checkpoint,
        dry_run=args.dry_run,
        limit=args.limit,
        env=env,
        timeout_s=args.timeout,
    )
    agg = report.write_report(args.out, results, name=name, seed=args.seed)
    print(report.render_report(agg))
    print(f"\nWrote report to {Path(args.out) / report.REPORT_MD}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run a paired comparison of two result directories.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on success).
    """
    results_a = report.read_results_csv(Path(args.dir_a) / report.RESULTS_CSV)
    results_b = report.read_results_csv(Path(args.dir_b) / report.RESULTS_CSV)
    name_a = args.name_a or Path(args.dir_a).name
    name_b = args.name_b or Path(args.dir_b).name
    cmp = report.compare(results_a, results_b, name_a=name_a, name_b=name_b, seed=args.seed)
    out_dir = args.out or f"results/compare_{name_a}_{name_b}"
    report.write_comparison(out_dir, cmp)
    print(report.render_comparison(cmp))
    print(f"\nWrote comparison to {Path(out_dir) / 'compare.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="generate the stratified suite")
    p_gen.add_argument("--out", required=True, help="output suite directory")
    p_gen.add_argument("--n", type=int, default=suite.DEFAULT_N,
                       help="number of stratified members (default 50)")
    p_gen.add_argument("--seed", type=int, default=suite.DEFAULT_SEED, help="master seed")
    p_gen.add_argument("--template", default=str(suite.DEFAULT_TEMPLATE),
                       help="eval_config.yaml template path")
    p_gen.add_argument("--smoke", action="store_true",
                       help=f"generate the {suite.SMOKE_N}+3 smoke subset")
    p_gen.add_argument("--no-official", action="store_true",
                       help="omit the 3 official eval_config trials")
    p_gen.set_defaults(func=cmd_gen)

    p_run = sub.add_parser("run", help="run a policy over a suite")
    p_run.add_argument("--suite", required=True, help="suite directory (with manifest.csv)")
    p_run.add_argument("--out", required=True, help="results output directory")
    p_run.add_argument("--policy", default="aic_example_policies.ros.CheatCode",
                       help="ROS policy param value (module.Class)")
    p_run.add_argument("--policy-cmd", default=None,
                       help="override the policy launch command (before --ros-args); "
                            "e.g. '/home/kiwoos/venvs/aic-deploy/bin/python "
                            "/home/kiwoos/ws_aic/install/lib/aic_model/aic_model' to run "
                            "a torch policy under the deploy venv interpreter")
    p_run.add_argument("--checkpoint", default=None,
                       help="checkpoint path exported as AIC_CHECKPOINT")
    p_run.add_argument("--name", default=None, help="run name for the report")
    p_run.add_argument("--dry-run", action="store_true",
                       help="fabricate scoring.yaml (no sim)")
    p_run.add_argument("--limit", type=int, default=None, help="run only first N configs")
    p_run.add_argument("--timeout", type=float, default=runner.DEFAULT_TRIAL_TIMEOUT_S,
                       help="per-trial completion timeout (s)")
    p_run.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="paired comparison of two runs")
    p_cmp.add_argument("dir_a", help="results dir for run A")
    p_cmp.add_argument("dir_b", help="results dir for run B")
    p_cmp.add_argument("--out", default=None, help="comparison output directory")
    p_cmp.add_argument("--name-a", default=None, help="display name for A")
    p_cmp.add_argument("--name-b", default=None, help="display name for B")
    p_cmp.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p_cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
