"""Pure-logic helpers for the Phase-0 demo-collection campaign.

The campaign shell script (``collect_campaign.sh``) delegates the two pieces of
non-trivial logic that are easy to get wrong in bash to this module:

* parsing the ``manifest.csv`` emitted by ``gen_config.py --mode strata`` into an
  ordered, filterable list of demo tasks, and
* deriving the stable per-episode output-directory name from a stratum + rep so a
  restarted campaign can skip episodes it already converted.

Everything here is pure (CSV + string manipulation only) so it imports and unit
tests without ROS, Gazebo, or a GPU. The ``__main__`` entry point prints the work
list as tab-separated rows for the shell loop to consume.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
from pathlib import Path

# Trailing ``_r<rep>`` in a strata config filename (``p0_<stratum>_r<rep>.yaml``).
_REP_RE = re.compile(r"_r(\d+)$")


@dataclasses.dataclass(frozen=True)
class DemoTask:
    """One unit of collection work derived from a manifest row.

    Attributes:
        config: Absolute path to the single-trial YAML config to collect.
        stratum: Stratified-cell name (e.g. ``sfp_rail2_sfp_port_0``).
        plug: Plug type, ``'sfp'`` or ``'sc'``.
        rep: Repetition index within the stratum (parsed from the config filename).
        episode_dir: Basename of the output episode directory (``ep_<stratum>_r<rep>``).
    """

    config: str
    stratum: str
    plug: str
    rep: int
    episode_dir: str


def parse_rep(config_path: str) -> int:
    """Extract the repetition index from a strata config filename.

    Args:
        config_path: Path whose stem ends in ``_r<rep>`` (e.g.
            ``.../p0_sfp_rail0_sfp_port_0_r3.yaml``).

    Returns:
        The integer repetition index (``3`` for the example above).

    Raises:
        ValueError: If the filename stem has no ``_r<rep>`` suffix.
    """
    stem = Path(config_path).stem
    match = _REP_RE.search(stem)
    if match is None:
        raise ValueError(
            f"config filename {config_path!r} has no trailing '_r<rep>' suffix"
        )
    return int(match.group(1))


def episode_dir_name(stratum: str, rep: int) -> str:
    """Return the stable output episode-directory basename for a stratum + rep.

    The name encodes both the stratum and the repetition so the campaign is
    resumable: an existing ``ep_<stratum>_r<rep>`` directory marks completed work.

    Args:
        stratum: Stratified-cell name.
        rep: Non-negative repetition index.

    Returns:
        ``ep_<stratum>_r<rep>``.

    Raises:
        ValueError: If ``stratum`` is empty or ``rep`` is negative.
    """
    if not stratum:
        raise ValueError("stratum must be a non-empty string")
    if rep < 0:
        raise ValueError(f"rep must be non-negative, got {rep}")
    return f"ep_{stratum}_r{rep}"


def load_tasks(manifest_path: str, plug: str | None = None) -> list[DemoTask]:
    """Parse a strata manifest CSV into an ordered list of demo tasks.

    Args:
        manifest_path: Path to the ``manifest.csv`` emitted by ``gen_config.py``.
        plug: Optional plug filter; when ``'sfp'`` or ``'sc'`` only tasks of that
            plug type are returned. ``None`` (default) returns every row in order.

    Returns:
        The demo tasks in manifest order (optionally filtered by ``plug``).

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.
        ValueError: If ``plug`` is not one of ``None``/``'sfp'``/``'sc'`` or a row
            is missing a required column.
    """
    if plug not in (None, "sfp", "sc"):
        raise ValueError(f"plug must be None, 'sfp' or 'sc', got {plug!r}")
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    tasks: list[DemoTask] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                config = row["config"]
                stratum = row["stratum"]
                row_plug = row["plug"]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(f"manifest row missing column {exc}") from exc
            if plug is not None and row_plug != plug:
                continue
            rep = parse_rep(config)
            tasks.append(
                DemoTask(
                    config=config,
                    stratum=stratum,
                    plug=row_plug,
                    rep=rep,
                    episode_dir=episode_dir_name(stratum, rep),
                )
            )
    return tasks


def _main(argv: list[str] | None = None) -> int:
    """Print the campaign work list as tab-separated rows for the shell loop.

    Emits one ``config\\tstratum\\tplug\\trep\\tepisode_dir`` line per demo task.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="path to strata manifest.csv")
    parser.add_argument(
        "--plug",
        choices=["sfp", "sc"],
        default=None,
        help="restrict output to one plug type (default: all)",
    )
    args = parser.parse_args(argv)
    for task in load_tasks(args.manifest, plug=args.plug):
        sys.stdout.write(
            f"{task.config}\t{task.stratum}\t{task.plug}\t{task.rep}\t{task.episode_dir}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
