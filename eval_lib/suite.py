"""Deterministic generation of the fixed matched-seed evaluation suite.

The suite is a fixed set of single-trial engine configs, stratified over
``rail{0-4} x plug{SFP,SC} x port{0,1}`` cells (20 cells) with the continuous
task axes sampled uniformly:

* board pose ``x`` in [0.15, 0.20], ``y`` in [-0.21, 0.05], ``yaw`` in [-pi, pi]
* grasp offset ``z`` in [0.040, 0.046]

Every config is generated from the official ``eval_config.yaml`` template the
same way ``gen_config.py`` does (whole document copied, ``trials`` replaced by a
single perturbed ``trial_1``) so the engine's ``scoring``/``task_board_limits``/
``robot`` blocks stay intact. The suite is persisted as one YAML per config plus
``manifest.csv``; because generation is fully seeded, *every* checkpoint runs the
byte-identical configs, which is the paired/matched-seed design of rliable
(arXiv:2108.13264) and finding #11 of the research plan.

The three official ``eval_config.yaml`` trials are included verbatim (content
unchanged, re-keyed to ``trial_1``) and flagged ``source == "official"``.

Note on SC fidelity: the SFP path mirrors ``gen_config.py``'s proven ``wide``
recipe exactly. The SC scene topology is approximated (SC uses its own board
mounts, not the NIC rails) and should be validated against the SC scene spec
before the first live SC run; the harness plumbing (strata, parsing, stats) is
unaffected by that approximation.
"""

from __future__ import annotations

import csv
import copy
import dataclasses
import math
import random
from pathlib import Path
from typing import Any

import yaml

# --- Continuous eval axes (match the eval distribution; finding #9). ---
BOARD_X_RANGE = (0.15, 0.20)
BOARD_Y_RANGE = (-0.21, 0.05)
BOARD_YAW_RANGE = (-math.pi, math.pi)
GRASP_Z_RANGE = (0.040, 0.046)

# --- NIC-card randomization bounds (aic_engine task_board_limits: nic_rail). ---
NIC_TRANSLATION_RANGE = (-0.0215, 0.0234)
CARD_YAW_RANGE = (-0.04, 0.04)

PLUGS = ("SFP", "SC")
RAILS = (0, 1, 2, 3, 4)
PORTS = (0, 1)

DEFAULT_SEED = 20260712
DEFAULT_N = 50
SMOKE_N = 12  # 12 stratified + 3 official == 15-config smoke suite

# Repo-root-relative location of the official template.
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "aic_engine" / "config" / "eval_config.yaml"
)

MANIFEST_COLUMNS = (
    "config_id",
    "source",
    "rail",
    "plug",
    "port",
    "port_name",
    "board_x",
    "board_y",
    "board_yaw",
    "grasp_z",
    "config_file",
)


@dataclasses.dataclass(frozen=True)
class Stratum:
    """A discrete evaluation cell: one ``rail x plug x port`` combination.

    Attributes:
        rail: Rail index in ``0..4``.
        plug: Plug family, ``"SFP"`` or ``"SC"``.
        port: Port index in ``{0, 1}``.
    """

    rail: int
    plug: str
    port: int

    def cell_id(self) -> str:
        """Return a stable string identifier for the cell."""
        return f"rail{self.rail}_{self.plug}_port{self.port}"

    def port_name(self) -> str:
        """Return the engine port name for this cell's plug/port."""
        prefix = "sfp_port" if self.plug == "SFP" else "sc_port"
        return f"{prefix}_{self.port}"


@dataclasses.dataclass(frozen=True)
class SuiteMember:
    """One config in the evaluation suite.

    Attributes:
        config_id: Unique id (e.g. ``"cfg_000"`` or ``"official_1"``).
        source: ``"stratified"`` or ``"official"``.
        stratum: The :class:`Stratum` this member belongs to.
        board_x: Sampled board pose x (meters).
        board_y: Sampled board pose y (meters).
        board_yaw: Sampled board pose yaw (radians).
        grasp_z: Sampled grasp gripper-offset z (meters).
        config_file: Suite-relative path of the written YAML config.
    """

    config_id: str
    source: str
    stratum: Stratum
    board_x: float
    board_y: float
    board_yaw: float
    grasp_z: float
    config_file: str

    def manifest_row(self) -> dict[str, Any]:
        """Return this member as a manifest CSV row."""
        return {
            "config_id": self.config_id,
            "source": self.source,
            "rail": self.stratum.rail,
            "plug": self.stratum.plug,
            "port": self.stratum.port,
            "port_name": self.stratum.port_name(),
            "board_x": f"{self.board_x:.6f}",
            "board_y": f"{self.board_y:.6f}",
            "board_yaw": f"{self.board_yaw:.6f}",
            "grasp_z": f"{self.grasp_z:.6f}",
            "config_file": self.config_file,
        }


def all_cells() -> list[Stratum]:
    """Return the 20 strata cells in a fixed, deterministic order.

    Returns:
        ``rail`` (outer) x ``plug`` x ``port`` (inner) ordering.
    """
    return [
        Stratum(rail=rail, plug=plug, port=port)
        for rail in RAILS
        for plug in PLUGS
        for port in PORTS
    ]


def assign_strata(n: int) -> list[Stratum]:
    """Assign ``n`` stratified members to cells by round-robin.

    Round-robin over the fixed cell order guarantees near-uniform coverage:
    each cell receives ``floor(n / 20)`` or ``ceil(n / 20)`` members.

    Args:
        n: Number of stratified members to assign (``> 0``).

    Returns:
        A list of length ``n`` of assigned strata.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    cells = all_cells()
    return [cells[i % len(cells)] for i in range(n)]


def _first_cable_pose(trial: dict[str, Any]) -> dict[str, Any]:
    """Return the pose dict of the (single) cable in a trial scene.

    Args:
        trial: A trial mapping with ``scene.cables``.

    Returns:
        The ``pose`` mapping of the first cable.

    Raises:
        KeyError: If the trial has no cable pose.
    """
    cables = trial["scene"]["cables"]
    first_key = next(iter(cables))
    return cables[first_key]["pose"]


def _apply_continuous_axes(
    trial: dict[str, Any],
    board_x: float,
    board_y: float,
    board_yaw: float,
    grasp_z: float,
) -> None:
    """Write the sampled continuous axes into a trial in place.

    Args:
        trial: The trial mapping to mutate.
        board_x: Board pose x.
        board_y: Board pose y.
        board_yaw: Board pose yaw.
        grasp_z: Grasp gripper-offset z.
    """
    pose = trial["scene"]["task_board"]["pose"]
    pose["x"] = board_x
    pose["y"] = board_y
    pose["yaw"] = board_yaw
    _first_cable_pose(trial)["gripper_offset"]["z"] = grasp_z


def _apply_sfp_target(
    trial: dict[str, Any], rail: int, port: int, rng: random.Random
) -> None:
    """Configure a NIC/SFP target: one NIC rail present, others absent.

    Mirrors ``gen_config.py``'s ``wide`` recipe.

    Args:
        trial: The trial mapping to mutate.
        rail: Target NIC rail index in ``0..4``.
        port: SFP port index in ``{0, 1}``.
        rng: Seeded RNG for the card translation/yaw jitter.
    """
    task_board = trial["scene"]["task_board"]
    for k in RAILS:
        key = f"nic_rail_{k}"
        if k == rail:
            task_board[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{k}",
                "entity_pose": {
                    "translation": rng.uniform(*NIC_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rng.uniform(*CARD_YAW_RANGE),
                },
            }
        else:
            task_board[key] = {"entity_present": False}
    task = trial["tasks"]["task_1"]
    task["cable_type"] = "sfp_sc"
    task["plug_type"] = "sfp"
    task["plug_name"] = "sfp_tip"
    task["port_type"] = "sfp"
    task["port_name"] = f"sfp_port_{port}"
    task["target_module_name"] = f"nic_card_mount_{rail}"


def _apply_sc_target(
    trial: dict[str, Any], rail: int, port: int, rng: random.Random
) -> None:
    """Configure an SC target from the SC template trial.

    SC uses its own board mounts rather than the NIC rails, so this only
    retargets the SC port/module and jitters the corresponding sc_rail. The
    ``rail`` stratum index is recorded for coverage; physically it selects
    ``sc_rail_{rail % 2}``. Provisional pending SC scene-spec confirmation.

    Args:
        trial: The SC trial mapping to mutate.
        rail: Rail stratum index in ``0..4`` (bookkeeping).
        port: SC port index in ``{0, 1}``.
        rng: Seeded RNG for sc_rail translation jitter.
    """
    task_board = trial["scene"]["task_board"]
    sc_key = f"sc_rail_{rail % 2}"
    if sc_key in task_board and isinstance(task_board[sc_key], dict):
        entity_pose = task_board[sc_key].get("entity_pose")
        if isinstance(entity_pose, dict):
            entity_pose["translation"] = rng.uniform(-0.05, 0.05)
    task = trial["tasks"]["task_1"]
    task["plug_type"] = "sc"
    task["port_type"] = "sc"
    task["port_name"] = f"sc_port_{port}"
    task["target_module_name"] = f"sc_port_{port}"


def build_config(
    template: dict[str, Any], stratum: Stratum, rng: random.Random
) -> tuple[dict[str, Any], SuiteMember]:
    """Build one stratified single-trial config and its member record.

    Args:
        template: The parsed ``eval_config.yaml`` template.
        stratum: The cell this config realises.
        rng: Seeded RNG for this config (determinism).

    Returns:
        A ``(config_dict, partial_member)`` tuple. The returned member has an
        empty ``config_id``/``config_file`` (filled in by :func:`generate_suite`).

    Raises:
        KeyError: If the template lacks the required source trial.
    """
    source_trial = "trial_1" if stratum.plug == "SFP" else "trial_3"
    config = copy.deepcopy(template)
    trial = copy.deepcopy(template["trials"][source_trial])
    board_x = rng.uniform(*BOARD_X_RANGE)
    board_y = rng.uniform(*BOARD_Y_RANGE)
    board_yaw = rng.uniform(*BOARD_YAW_RANGE)
    grasp_z = rng.uniform(*GRASP_Z_RANGE)
    _apply_continuous_axes(trial, board_x, board_y, board_yaw, grasp_z)
    if stratum.plug == "SFP":
        _apply_sfp_target(trial, stratum.rail, stratum.port, rng)
    else:
        _apply_sc_target(trial, stratum.rail, stratum.port, rng)
    config["trials"] = {"trial_1": trial}
    member = SuiteMember(
        config_id="",
        source="stratified",
        stratum=stratum,
        board_x=board_x,
        board_y=board_y,
        board_yaw=board_yaw,
        grasp_z=grasp_z,
        config_file="",
    )
    return config, member


def _infer_official_stratum(trial: dict[str, Any]) -> Stratum:
    """Best-effort stratum for an official trial (for coverage accounting).

    Args:
        trial: An official trial mapping.

    Returns:
        The inferred :class:`Stratum`.
    """
    task = trial["tasks"]["task_1"]
    plug = "SFP" if str(task.get("plug_type", "sfp")).lower() == "sfp" else "SC"
    target = str(task.get("target_module_name", ""))
    port_name = str(task.get("port_name", ""))
    rail = 0
    for token in target.replace("_", " ").split():
        if token.isdigit():
            rail = int(token)
            break
    rail = max(0, min(4, rail))
    port = 1 if port_name.endswith("1") else 0
    return Stratum(rail=rail, plug=plug, port=port)


def build_official_members(template: dict[str, Any]) -> list[tuple[dict[str, Any], SuiteMember]]:
    """Build the three official trials as verbatim single-trial suite members.

    Args:
        template: The parsed ``eval_config.yaml`` template.

    Returns:
        A list of ``(config_dict, partial_member)`` tuples, one per official
        trial, ordered ``trial_1``, ``trial_2``, ``trial_3``.
    """
    members: list[tuple[dict[str, Any], SuiteMember]] = []
    for idx, trial_key in enumerate(("trial_1", "trial_2", "trial_3"), start=1):
        if trial_key not in template.get("trials", {}):
            continue
        source_trial = copy.deepcopy(template["trials"][trial_key])
        stratum = _infer_official_stratum(source_trial)
        pose = source_trial["scene"]["task_board"]["pose"]
        grasp_z = _first_cable_pose(source_trial)["gripper_offset"]["z"]
        config = copy.deepcopy(template)
        config["trials"] = {"trial_1": source_trial}
        member = SuiteMember(
            config_id=f"official_{idx}",
            source="official",
            stratum=stratum,
            board_x=float(pose["x"]),
            board_y=float(pose["y"]),
            board_yaw=float(pose["yaw"]),
            grasp_z=float(grasp_z),
            config_file="",
        )
        members.append((config, member))
    return members


def load_template(template_path: str | Path = DEFAULT_TEMPLATE) -> dict[str, Any]:
    """Load and validate the eval-config template.

    Args:
        template_path: Path to ``eval_config.yaml``.

    Returns:
        The parsed template mapping.

    Raises:
        FileNotFoundError: If the template file is missing.
        ValueError: If the template lacks the required trials.
    """
    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"eval-config template not found: {path}")
    with path.open() as fh:
        template = yaml.safe_load(fh)
    if "trials" not in template or "trial_1" not in template["trials"]:
        raise ValueError(f"template {path} missing trials/trial_1")
    return template


def generate_suite(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    template_path: str | Path = DEFAULT_TEMPLATE,
    include_official: bool = True,
) -> tuple[list[tuple[dict[str, Any], SuiteMember]], dict[str, Any]]:
    """Generate the full suite deterministically (no disk writes).

    Args:
        n: Number of stratified members (official trials are added on top when
            ``include_official`` is True).
        seed: Master seed; member ``i`` uses ``random.Random(seed + i)``.
        template_path: Path to ``eval_config.yaml``.
        include_official: Whether to append the three official trials.

    Returns:
        A ``(members, meta)`` tuple where ``members`` is a list of
        ``(config_dict, SuiteMember)`` with ids/paths assigned, and ``meta`` is
        provenance metadata (seed, ranges, counts).
    """
    template = load_template(template_path)
    strata = assign_strata(n)
    built: list[tuple[dict[str, Any], SuiteMember]] = []
    for i, stratum in enumerate(strata):
        rng = random.Random(seed + i)
        config, member = build_config(template, stratum, rng)
        config_id = f"cfg_{i:03d}"
        member = dataclasses.replace(
            member, config_id=config_id, config_file=f"configs/{config_id}.yaml"
        )
        built.append((config, member))
    if include_official:
        for config, member in build_official_members(template):
            member = dataclasses.replace(
                member, config_file=f"configs/{member.config_id}.yaml"
            )
            built.append((config, member))
    meta = {
        "seed": seed,
        "n_stratified": n,
        "n_official": len(built) - n if include_official else 0,
        "n_total": len(built),
        "continuous_axes": {
            "board_x": list(BOARD_X_RANGE),
            "board_y": list(BOARD_Y_RANGE),
            "board_yaw": list(BOARD_YAW_RANGE),
            "grasp_z": list(GRASP_Z_RANGE),
        },
        "template": str(Path(template_path)),
    }
    return built, meta


def write_suite(
    out_dir: str | Path,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    template_path: str | Path = DEFAULT_TEMPLATE,
    include_official: bool = True,
) -> list[SuiteMember]:
    """Generate the suite and persist configs + ``manifest.csv`` to disk.

    Args:
        out_dir: Destination directory for the suite.
        n: Number of stratified members.
        seed: Master seed.
        template_path: Path to ``eval_config.yaml``.
        include_official: Whether to append the three official trials.

    Returns:
        The list of :class:`SuiteMember` written.
    """
    out = Path(out_dir)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    built, meta = generate_suite(
        n=n, seed=seed, template_path=template_path, include_official=include_official
    )
    members: list[SuiteMember] = []
    for config, member in built:
        config_path = out / member.config_file
        with config_path.open("w") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        members.append(member)
    write_manifest(out / "manifest.csv", members)
    with (out / "suite_meta.yaml").open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    return members


def write_manifest(path: str | Path, members: list[SuiteMember]) -> None:
    """Write the suite manifest CSV.

    Args:
        path: Destination CSV path.
        members: Suite members to record.
    """
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for member in members:
            writer.writerow(member.manifest_row())


def read_manifest(path: str | Path) -> list[SuiteMember]:
    """Read a suite manifest CSV back into :class:`SuiteMember` records.

    Args:
        path: Path to ``manifest.csv``.

    Returns:
        The list of members in file order.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    members: list[SuiteMember] = []
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            members.append(
                SuiteMember(
                    config_id=row["config_id"],
                    source=row["source"],
                    stratum=Stratum(
                        rail=int(row["rail"]),
                        plug=row["plug"],
                        port=int(row["port"]),
                    ),
                    board_x=float(row["board_x"]),
                    board_y=float(row["board_y"]),
                    board_yaw=float(row["board_yaw"]),
                    grasp_z=float(row["grasp_z"]),
                    config_file=row["config_file"],
                )
            )
    return members


def stratum_counts(members: list[SuiteMember]) -> dict[str, int]:
    """Count suite members per stratum cell.

    Args:
        members: Suite members.

    Returns:
        Mapping of ``cell_id`` to member count.
    """
    counts: dict[str, int] = {}
    for member in members:
        cell = member.stratum.cell_id()
        counts[cell] = counts.get(cell, 0) + 1
    return counts
