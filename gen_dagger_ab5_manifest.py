"""Build the officials/ab5 DAgger collection manifest (the SEAT-TARGET poses).

``collect_dagger.sh`` feeds each manifest row through ``campaign_lib.load_tasks``,
which requires the ``config``, ``stratum`` and ``plug`` columns AND a ``config``
filename whose stem ends in ``_r<rep>`` (``campaign_lib.parse_rep`` derives the
repetition index from the filename). The real eval-suite configs are named
``official_1.yaml`` / ``cfg_001.yaml`` (no ``_r`` suffix), so this builder writes,
for each config x rep, a per-rep **symlink** ``<base>_r<rep>.yaml`` pointing at the
real config and points the manifest row at the symlink. That satisfies both
``parse_rep`` (rep parsed from the symlink name) and ``collect_dagger.sh`` (the
``[ -f "$CFG" ]`` existence check and the ``port_name`` / ``target_module_name``
greps both follow the symlink to the real config body).

The result is that

    MANIFEST=~/data/configs_dagger_ab5/manifest.csv OUT=~/training/ds_dagger_ab5 \\
        bash collect_dagger.sh

collects the deploy policy's stalls on the actual competition seat poses
(official_1/2/3, cfg_001, cfg_005), in-distribution for the seat, with N reps
each so there are enough stall samples to relabel.

Pure row-generation (``build_rows``, ``rep_config_filename``) is torch/ROS-free
and unit-tested (``tests/test_gen_dagger_ab5_manifest.py``); ``main`` performs the
filesystem side effects (symlinks + CSV).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib

# The five eval_suite_ab5 seat targets, with the stratum label the episode dir is
# named from and the plug type collect_dagger records. plug/port/module are also
# read straight from each config yaml by collect_dagger, so only plug (for the
# optional --plug filter + provenance) is duplicated here.
DEFAULT_REPS: int = 3
_CONFIG_SUBDIR = pathlib.Path("eval_suite_ab5") / "configs"


@dataclasses.dataclass(frozen=True)
class Ab5Config:
    """One eval-suite seat target to collect DAgger stalls on.

    Attributes:
        base: Config basename without extension (e.g. ``official_1``).
        stratum: Stratum label; the episode dir is ``ep_<stratum>_r<rep>``.
        plug: Plug type, ``'sfp'`` or ``'sc'``.
    """

    base: str
    stratum: str
    plug: str


# official_1/2, cfg_001, cfg_005 are SFP; official_3 is SC (verified from the
# config yamls' plug_type field).
AB5_CONFIGS: tuple[Ab5Config, ...] = (
    Ab5Config(base="official_1", stratum="ab5_official_1", plug="sfp"),
    Ab5Config(base="official_2", stratum="ab5_official_2", plug="sfp"),
    Ab5Config(base="official_3", stratum="ab5_official_3", plug="sc"),
    Ab5Config(base="cfg_001", stratum="ab5_cfg_001", plug="sfp"),
    Ab5Config(base="cfg_005", stratum="ab5_cfg_005", plug="sfp"),
)

MANIFEST_HEADER: tuple[str, ...] = ("config", "stratum", "plug")


def rep_config_filename(base: str, rep: int) -> str:
    """Return the per-rep symlink filename for a config base + repetition.

    The trailing ``_r<rep>`` is what ``campaign_lib.parse_rep`` reads to recover
    the repetition index, so the returned name round-trips through it.

    Args:
        base: Config basename without extension.
        rep: Non-negative repetition index.

    Returns:
        ``<base>_r<rep>.yaml``.

    Raises:
        ValueError: If ``base`` is empty or ``rep`` is negative.
    """
    if not base:
        raise ValueError("base must be a non-empty string")
    if rep < 0:
        raise ValueError(f"rep must be non-negative, got {rep}")
    return f"{base}_r{rep}.yaml"


@dataclasses.dataclass(frozen=True)
class ManifestRow:
    """One manifest row + the symlink it needs.

    Attributes:
        config: Absolute path to the per-rep symlink (the manifest ``config``).
        stratum: Stratum label.
        plug: Plug type.
        link_target: Absolute path to the real config the symlink points at.
    """

    config: str
    stratum: str
    plug: str
    link_target: str


def build_rows(
    configs: tuple[Ab5Config, ...],
    reps: int,
    link_dir: pathlib.Path,
    config_dir: pathlib.Path,
) -> list[ManifestRow]:
    """Build the manifest rows (and their symlink targets) for all config x rep.

    Args:
        configs: The seat-target configs to collect.
        reps: Number of repetitions per config (>= 1).
        link_dir: Directory the per-rep symlinks live in (the manifest ``config``
            column points here).
        config_dir: Directory holding the real ``<base>.yaml`` configs.

    Returns:
        One :class:`ManifestRow` per config x rep, in config-then-rep order.

    Raises:
        ValueError: If ``reps`` < 1.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    rows: list[ManifestRow] = []
    for cfg in configs:
        for rep in range(reps):
            link = link_dir / rep_config_filename(cfg.base, rep)
            target = config_dir / f"{cfg.base}.yaml"
            rows.append(
                ManifestRow(
                    config=str(link),
                    stratum=cfg.stratum,
                    plug=cfg.plug,
                    link_target=str(target),
                )
            )
    return rows


def write_manifest(rows: list[ManifestRow], manifest_path: pathlib.Path) -> None:
    """Write the ``config,stratum,plug`` manifest CSV consumed by campaign_lib.

    Args:
        rows: The manifest rows to write.
        manifest_path: Destination CSV path (parent dirs created).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MANIFEST_HEADER)
        for row in rows:
            writer.writerow([row.config, row.stratum, row.plug])


def create_symlinks(rows: list[ManifestRow]) -> None:
    """Create (or refresh) the per-rep config symlinks for every row.

    Args:
        rows: The manifest rows whose ``config`` symlinks point at ``link_target``.

    Raises:
        FileNotFoundError: If a symlink target (real config) does not exist.
    """
    for row in rows:
        target = pathlib.Path(row.link_target)
        if not target.exists():
            raise FileNotFoundError(f"config not found: {target}")
        link = pathlib.Path(row.config)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())


def main(argv: list[str] | None = None) -> int:
    """CLI: write the ab5 DAgger manifest + its per-rep config symlinks.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    repo_root = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build the officials/ab5 DAgger manifest.")
    ap.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path.home() / "data" / "configs_dagger_ab5",
        help="Directory for the manifest + per-rep config symlinks.",
    )
    ap.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help="Repetitions per seat config (default 3).",
    )
    ap.add_argument(
        "--config-dir", type=pathlib.Path, default=repo_root / _CONFIG_SUBDIR,
        help="Directory holding the real eval_suite_ab5 config yamls.",
    )
    a = ap.parse_args(argv)

    out_dir = a.out_dir.expanduser()
    rows = build_rows(AB5_CONFIGS, a.reps, out_dir, a.config_dir.expanduser())
    create_symlinks(rows)
    manifest_path = out_dir / "manifest.csv"
    write_manifest(rows, manifest_path)
    print(f"wrote {manifest_path} ({len(rows)} rows, {a.reps} reps x {len(AB5_CONFIGS)} configs)")
    print(f"symlinks in {out_dir}")
    for row in rows:
        print(f"  {row.stratum:<18} plug={row.plug:<3} {row.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
