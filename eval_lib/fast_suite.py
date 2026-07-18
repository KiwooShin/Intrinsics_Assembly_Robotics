"""Derive a shortened-``time_limit`` copy of an evaluation suite.

The official protocol runs each trial with a task ``time_limit`` of 180 sim
seconds. A successful cable insertion is reached well under 30 sim seconds;
trials that never insert simply stall until the limit, so at the observed policy
real-time factor (RTF ~= 0.05, i.e. sim ~20x slower than wall) a single
non-inserting trial burns ~60 wall minutes purely waiting out the limit. Cutting
the limit to 60 sim seconds preserves the same tier-3 insertion signal (an insert
still happens in-window) while capping the wasted stall time at one third.

This module produces an *internal fast-protocol* suite: a byte-for-byte copy of a
source suite whose per-task ``time_limit`` is rewritten, with a ``suite_meta.yaml``
stamped so downstream reports carry the caveat that this deviates from the
official 180 s protocol and is for internal comparison only.

Everything here is pure file/YAML logic: it runs without ROS, Gazebo, or a GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_FAST_TIME_LIMIT_S: int = 60
OFFICIAL_TIME_LIMIT_S: int = 180

# Stamped into the derived suite_meta.yaml so report.py / readers can surface the
# protocol deviation. The exact string is asserted by the mission brief.
FAST_PROTOCOL_KEY: str = "internal_fast_protocol"


@dataclasses.dataclass(frozen=True)
class FastSuiteResult:
    """Summary of a derived fast suite.

    Attributes:
        out_dir: Directory the fast suite was written to.
        n_configs: Number of config files copied and rewritten.
        n_limits_rewritten: Total number of ``time_limit`` fields rewritten
            across all configs.
        time_limit_s: The task ``time_limit`` (sim seconds) written into every
            config.
    """

    out_dir: Path
    n_configs: int
    n_limits_rewritten: int
    time_limit_s: int


def set_task_time_limits(config: dict[str, Any], time_limit_s: int) -> int:
    """Rewrite every task ``time_limit`` in a parsed engine config, in place.

    Walks ``config['trials'][*]['tasks'][*]`` and sets ``time_limit`` on each
    task that already declares one (the field is left untouched where absent so
    unrelated task shapes are never invented).

    Args:
        config: A parsed engine config mapping (``eval_config``-style document).
        time_limit_s: New task time limit in simulated seconds (``> 0``).

    Returns:
        The number of ``time_limit`` fields that were rewritten.

    Raises:
        ValueError: If ``time_limit_s`` is not positive, or if ``config`` has no
            ``trials`` mapping to rewrite.
    """
    if time_limit_s <= 0:
        raise ValueError(f"time_limit_s must be positive, got {time_limit_s}")
    trials = config.get("trials")
    if not isinstance(trials, dict):
        raise ValueError("config has no 'trials' mapping to rewrite")
    rewritten = 0
    for trial in trials.values():
        if not isinstance(trial, dict):
            continue
        tasks = trial.get("tasks")
        if not isinstance(tasks, dict):
            continue
        for task in tasks.values():
            if isinstance(task, dict) and "time_limit" in task:
                task["time_limit"] = time_limit_s
                rewritten += 1
    return rewritten


def rewrite_config_file(src: Path, dst: Path, time_limit_s: int) -> int:
    """Read one config YAML, rewrite its task time limits, and write the copy.

    Args:
        src: Source config YAML path.
        dst: Destination config YAML path (parent dirs created as needed).
        time_limit_s: New task time limit in simulated seconds.

    Returns:
        The number of ``time_limit`` fields rewritten in this config.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        ValueError: If the config declares no task ``time_limit`` to rewrite.
    """
    if not src.is_file():
        raise FileNotFoundError(f"source config not found: {src}")
    with src.open() as fh:
        config = yaml.safe_load(fh)
    n = set_task_time_limits(config, time_limit_s)
    if n == 0:
        raise ValueError(f"no task time_limit found to rewrite in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return n


def build_fast_suite(
    src_suite_dir: str | Path,
    out_dir: str | Path,
    time_limit_s: int = DEFAULT_FAST_TIME_LIMIT_S,
) -> FastSuiteResult:
    """Derive a fast-protocol suite from an existing suite directory.

    The derived suite is identical to the source (same ``manifest.csv`` and the
    same set of stratified/official configs) except that every task
    ``time_limit`` is set to ``time_limit_s``. The copied ``suite_meta.yaml`` is
    stamped with the fast-protocol caveat.

    Args:
        src_suite_dir: Source suite directory (must contain ``manifest.csv`` and
            a ``configs/`` directory).
        out_dir: Destination directory for the derived suite.
        time_limit_s: Task time limit (sim seconds) for every config.

    Returns:
        A :class:`FastSuiteResult` describing what was written.

    Raises:
        FileNotFoundError: If the source suite, its manifest, or its ``configs/``
            directory is missing.
        ValueError: If the source suite contains no config files.
    """
    src = Path(src_suite_dir)
    out = Path(out_dir)
    manifest = src / "manifest.csv"
    configs_dir = src / "configs"
    if not manifest.is_file():
        raise FileNotFoundError(f"source manifest not found: {manifest}")
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"source configs dir not found: {configs_dir}")

    config_files = sorted(configs_dir.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"source suite has no configs: {configs_dir}")

    (out / "configs").mkdir(parents=True, exist_ok=True)
    total_rewritten = 0
    for cfg in config_files:
        total_rewritten += rewrite_config_file(
            cfg, out / "configs" / cfg.name, time_limit_s
        )

    # The manifest is copied verbatim: strata, poses and config-file paths are
    # unchanged, so every checkpoint still runs the byte-identical matched-seed
    # scene set -- only the task cutoff differs.
    shutil.copy2(manifest, out / "manifest.csv")

    _write_fast_meta(src, out, time_limit_s, len(config_files))

    logger.info(
        "wrote fast suite: %d configs, %d time_limit fields -> %s (time_limit=%ds)",
        len(config_files),
        total_rewritten,
        out,
        time_limit_s,
    )
    return FastSuiteResult(
        out_dir=out,
        n_configs=len(config_files),
        n_limits_rewritten=total_rewritten,
        time_limit_s=time_limit_s,
    )


def _write_fast_meta(
    src: Path, out: Path, time_limit_s: int, n_configs: int
) -> None:
    """Write the derived ``suite_meta.yaml`` with the fast-protocol caveat.

    Args:
        src: Source suite directory (its ``suite_meta.yaml`` is used as a base).
        out: Destination suite directory.
        time_limit_s: Task time limit stamped into the meta.
        n_configs: Number of configs in the derived suite.
    """
    meta: dict[str, Any] = {}
    src_meta = src / "suite_meta.yaml"
    if src_meta.is_file():
        with src_meta.open() as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            meta.update(loaded)
    meta[FAST_PROTOCOL_KEY] = f"time_limit={time_limit_s}"
    meta["fast_time_limit_s"] = time_limit_s
    meta["official_time_limit_s"] = OFFICIAL_TIME_LIMIT_S
    meta["derived_from"] = str(src)
    meta["n_configs"] = n_configs
    meta["protocol_note"] = (
        "INTERNAL FAST PROTOCOL: task time_limit shortened from "
        f"{OFFICIAL_TIME_LIMIT_S}s to {time_limit_s}s (sim seconds). Insertions "
        "occur well under 30 sim-s, so tier-3 signal is preserved while capping "
        "non-inserting stall cost. NOT the official 180s protocol -- for internal "
        "comparison only."
    )
    with (out / "suite_meta.yaml").open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src", required=True, help="source suite directory (with manifest.csv)"
    )
    parser.add_argument("--out", required=True, help="destination suite directory")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_FAST_TIME_LIMIT_S,
        help=f"task time_limit in sim seconds (default {DEFAULT_FAST_TIME_LIMIT_S})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    result = build_fast_suite(args.src, args.out, time_limit_s=args.time_limit)
    print(
        f"Wrote {result.n_configs} configs "
        f"({result.n_limits_rewritten} time_limit fields -> {result.time_limit_s}s) "
        f"to {result.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
