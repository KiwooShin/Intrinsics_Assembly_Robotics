"""Generate randomized single-trial eval configs for demo collection.

Three modes:

* ``near`` -- tight perturbation around the proven trial_1 SFP topology
  (nic_rail_2 / sfp_port_0) so CheatCode reliably succeeds.
* ``wide`` -- the full SFP eval distribution (random target NIC rail + port).
* ``strata`` -- stratified enumeration of the whole eval product
  (NIC rail {0..4} x port {sfp_port_0, sfp_port_1} for SFP, plus SC target rail
  {0, 1}) with continuous axes drawn uniformly per config, optional distractor
  NIC cards / SC ports on non-target rails, and a manifest CSV for later
  stratified analysis.

The scene keys and topology mirror ``aic_engine/config/eval_config.yaml`` exactly
(``nic_rail_i`` -> ``nic_card_mount_i``, ``sc_rail_i`` -> ``sc_port_i``, verified
against ``aic_engine/src/aic_engine.cpp``).
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import math
import os
import random
from typing import Any

import yaml

SRC = '/home/kiwoos/work/Intrinsics_Assembly_Robotics/aic_engine/config/eval_config.yaml'

# --- Eval-distribution ranges (match eval_config.yaml task_board_limits + trials) ---
NIC_MIN, NIC_MAX = -0.0215, 0.0234       # nic_rail translation limits (m)
SC_MIN, SC_MAX = -0.06, 0.055            # sc_rail translation limits (m)
CARD_YAW = 0.04                          # +/- entity yaw jitter (rad), within +/-10 deg
GRASP_Z_MIN, GRASP_Z_MAX = 0.040, 0.046  # cable gripper_offset z (m)
BOARD_X_MIN, BOARD_X_MAX = 0.15, 0.20    # board pose x (m)
BOARD_Y_MIN, BOARD_Y_MAX = -0.21, 0.05   # board pose y (m)
HOME_JITTER = 0.02                       # +/- home-joint jitter (rad)

# --- Board-yaw eval band (strata mode) ---
# Every official eval / sample-trial board yaw falls in one of two narrow clusters:
# a near-+/-pi cluster (board faces the robot, |yaw| ~ 3.0-3.14, e.g. 3.1, -3.1,
# 3.0, +/-pi) and a ~-1.8 side cluster (e.g. -1.8). Sampling U(-pi, pi) wasted
# demos on yaw~0 boards that face away from the robot (out-of-distribution).
# ``sample_eval_band_yaw`` reproduces the eval band instead.
YAW_PI_BAND_MIN = 2.6                     # |yaw| lower bound of the +/-pi cluster (rad)
YAW_SIDE_BAND_MIN, YAW_SIDE_BAND_MAX = -2.2, -1.4  # the ~-1.8 side cluster (rad)
YAW_PI_BAND_PROB = 0.7                    # P(draw from the +/-pi cluster)

NIC_RAILS: tuple[int, ...] = (0, 1, 2, 3, 4)
SC_RAILS: tuple[int, ...] = (0, 1)
SFP_PORTS: tuple[str, ...] = ('sfp_port_0', 'sfp_port_1')

# Template trials the strata builder derives from (SFP task vs SC task).
SFP_TEMPLATE_TRIAL = 'trial_1'
SC_TEMPLATE_TRIAL = 'trial_3'

_ABSENT: dict[str, Any] = {'entity_present': False}


def perturb(base: dict[str, Any], i: int, seed: int = 0, mode: str = 'near') -> dict[str, Any]:
    """Return a randomized copy of the base config near (or across) the eval values.

    Args:
        base: The parsed eval config to derive from (not mutated).
        i: Trial index; combined with ``seed`` to seed the RNG deterministically.
        seed: Base random seed. ``perturb`` with the same ``seed`` and ``i`` is
            reproducible.
        mode: ``'near'`` for tight perturbation around the proven trial_1 topology,
            or ``'wide'`` for the full SFP eval distribution.

    Returns:
        A new config dict containing a single ``trial_1`` with perturbed goal, grasp,
        and robot-start values.
    """
    rng = random.Random(seed + i)
    U = rng.uniform
    c = copy.deepcopy(base)
    c['trials'] = {'trial_1': copy.deepcopy(base['trials']['trial_1'])}
    tb = c['trials']['trial_1']['scene']['task_board']
    task = c['trials']['trial_1']['tasks']['task_1']
    if mode == 'near':
        # --- GOAL: board pose (position + yaw), tight around eval trial_1 ---
        tb['pose']['x'] += U(-0.015, 0.015)
        tb['pose']['y'] += U(-0.015, 0.015)
        tb['pose']['yaw'] += U(-0.08, 0.08)
        # target stays nic_rail_2 / sfp_port_0
        nic = tb['nic_rail_2']['entity_pose']
        nic['translation'] = U(-0.0215, 0.0234)
        nic['yaw'] = U(-0.04, 0.04)
    else:  # wide: full SFP eval distribution
        tb['pose']['x'] = 0.16 + U(-0.03, 0.04)        # eval x ~0.15-0.20
        tb['pose']['y'] = -0.21 + U(-0.02, 0.26)       # eval y ~-0.22..0.05
        tb['pose']['yaw'] = 3.1 + U(-0.2, 0.2)         # board faces robot (~pi)
        rail = rng.choice([0, 1, 2, 3, 4])             # random target NIC rail
        for k in range(5):
            r = tb[f'nic_rail_{k}']
            if k == rail:
                r['entity_present'] = True
                r['entity_name'] = f'nic_card_{k}'
                r['entity_pose'] = {'translation': U(-0.0215, 0.0234),
                                    'roll': 0.0, 'pitch': 0.0, 'yaw': U(-0.04, 0.04)}
            else:
                tb[f'nic_rail_{k}'] = {'entity_present': False}
        task['port_name'] = rng.choice(['sfp_port_0', 'sfp_port_1'])
        task['target_module_name'] = f'nic_card_mount_{rail}'
    # --- END: grasp offset (~2mm / ~0.04rad real grasp deviation) ---
    g = c['trials']['trial_1']['scene']['cables']['cable_0']['pose']
    g['gripper_offset']['x'] += U(-0.002, 0.002)
    g['gripper_offset']['y'] += U(-0.002, 0.002)
    g['gripper_offset']['z'] += U(-0.002, 0.002)
    g['roll'] += U(-0.04, 0.04); g['pitch'] += U(-0.04, 0.04); g['yaw'] += U(-0.04, 0.04)
    # --- START: small robot home-joint perturbation ---
    for j in c['robot']['home_joint_positions']:
        c['robot']['home_joint_positions'][j] += U(-0.02, 0.02)
    return c


@dataclasses.dataclass(frozen=True)
class Stratum:
    """One stratified eval cell (a discrete target choice).

    Attributes:
        plug: Plug type, ``'sfp'`` or ``'sc'``.
        target_rail: Index of the target rail (NIC rail for SFP, SC rail for SC).
        port_name: Task port name (``'sfp_port_0'``/``'sfp_port_1'`` for SFP,
            ``'sc_port_base'`` for SC).
    """

    plug: str
    target_rail: int
    port_name: str

    @property
    def name(self) -> str:
        """Return a stable, filesystem-safe identifier for the cell."""
        if self.plug == 'sfp':
            return f'sfp_rail{self.target_rail}_{self.port_name}'
        return f'sc_rail{self.target_rail}'


@dataclasses.dataclass
class StrataManifestRow:
    """One CSV manifest record describing an emitted strata config.

    Every sampled continuous value is recorded so a later analysis pass can group
    scored rollouts by stratum and by any continuous axis.
    """

    config: str
    stratum: str
    plug: str
    target_rail: int
    port_name: str
    target_module_name: str
    seed: int
    index: int
    board_x: float
    board_y: float
    board_yaw: float
    grasp_z: float
    target_translation: float
    target_yaw: float
    distractors_on: bool
    n_distractors: int
    distractor_rails: str
    cable_type: str


def enumerate_strata() -> list[Stratum]:
    """Enumerate the full stratified eval product.

    Returns:
        The 12 cells: NIC rail {0..4} x port {sfp_port_0, sfp_port_1} (10 SFP cells)
        plus SC target rail {0, 1} (2 SC cells).
    """
    cells: list[Stratum] = []
    for rail in NIC_RAILS:
        for port in SFP_PORTS:
            cells.append(Stratum('sfp', rail, port))
    for rail in SC_RAILS:
        cells.append(Stratum('sc', rail, 'sc_port_base'))
    return cells


def _present_nic(rail: int, rng: random.Random) -> dict[str, Any]:
    """Build a present NIC-card rail entry with randomized in-limit pose."""
    return {
        'entity_present': True,
        'entity_name': f'nic_card_{rail}',
        'entity_pose': {
            'translation': rng.uniform(NIC_MIN, NIC_MAX),
            'roll': 0.0, 'pitch': 0.0,
            'yaw': rng.uniform(-CARD_YAW, CARD_YAW),
        },
    }


def _present_sc(rail: int, rng: random.Random) -> dict[str, Any]:
    """Build a present SC-port rail entry with randomized in-limit pose."""
    return {
        'entity_present': True,
        'entity_name': f'sc_mount_{rail}',
        'entity_pose': {
            'translation': rng.uniform(SC_MIN, SC_MAX),
            'roll': 0.0, 'pitch': 0.0,
            'yaw': rng.uniform(-CARD_YAW, CARD_YAW),
        },
    }


def _encode_distractors(nic: list[int], sc: list[int]) -> str:
    """Render distractor placement as a compact ``nic:a|b;sc:c`` string for the manifest."""
    parts: list[str] = []
    if nic:
        parts.append('nic:' + '|'.join(str(k) for k in nic))
    if sc:
        parts.append('sc:' + '|'.join(str(k) for k in sc))
    return ';'.join(parts)


def sample_eval_band_yaw(rng: random.Random) -> float:
    """Sample a board yaw from the official eval-distribution band.

    Reproduces the two narrow yaw clusters seen across the official eval / sample
    trials instead of the full ``U(-pi, pi)`` circle (which wasted demos on yaw~0
    boards facing away from the robot). With probability :data:`YAW_PI_BAND_PROB`
    the yaw is drawn from the near-+/-pi cluster (``|yaw|`` uniform in
    ``[YAW_PI_BAND_MIN, pi]`` with a random sign); otherwise it is drawn from the
    ``~-1.8`` side cluster (``[YAW_SIDE_BAND_MIN, YAW_SIDE_BAND_MAX]``).

    Args:
        rng: Seeded RNG; all draws use it so callers stay reproducible.

    Returns:
        A board yaw in radians inside the eval band.
    """
    if rng.random() < YAW_PI_BAND_PROB:
        magnitude = rng.uniform(YAW_PI_BAND_MIN, math.pi)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return sign * magnitude
    return rng.uniform(YAW_SIDE_BAND_MIN, YAW_SIDE_BAND_MAX)


def build_strata_config(
    base: dict[str, Any],
    stratum: Stratum,
    index: int,
    seed: int = 0,
    distractors: bool = True,
) -> tuple[dict[str, Any], StrataManifestRow]:
    """Build one stratified single-trial config for ``stratum`` and its manifest row.

    Continuous axes (board pose x/y/yaw, grasp z, small grasp and home-joint jitter) are
    drawn uniformly from ``Random(seed + index)`` so every emitted config is reproducible.
    The target rail always carries the target entity; distractors (when enabled) are 1-2
    extra NIC cards on non-target NIC rails (plus the non-target SC port for SC tasks) and
    never collide with the target rail.

    Args:
        base: Parsed eval config providing the SFP (``trial_1``) and SC (``trial_3``)
            template trials (not mutated).
        stratum: The discrete eval cell to realize.
        index: Global emission index; combined with ``seed`` to seed the RNG.
        seed: Base random seed.
        distractors: When ``True``, spawn 1-2 distractor entities on non-target rails.

    Returns:
        A ``(config_dict, manifest_row)`` tuple. ``manifest_row.config`` is left empty for
        the caller to fill with the written path.

    Raises:
        KeyError: If ``base`` lacks the template trial required for the stratum's plug type.
    """
    rng = random.Random(seed + index)
    template_key = SC_TEMPLATE_TRIAL if stratum.plug == 'sc' else SFP_TEMPLATE_TRIAL
    if template_key not in base.get('trials', {}):
        raise KeyError(
            f"base config missing template trial '{template_key}' required for "
            f"plug '{stratum.plug}'"
        )
    c = copy.deepcopy(base)
    trial = copy.deepcopy(base['trials'][template_key])
    c['trials'] = {'trial_1': trial}
    tb = trial['scene']['task_board']
    task = trial['tasks']['task_1']
    target = stratum.target_rail
    distractor_nic: list[int] = []
    distractor_sc: list[int] = []

    # --- GOAL: board pose (continuous, full eval ranges) ---
    tb['pose']['x'] = rng.uniform(BOARD_X_MIN, BOARD_X_MAX)
    tb['pose']['y'] = rng.uniform(BOARD_Y_MIN, BOARD_Y_MAX)
    tb['pose']['yaw'] = sample_eval_band_yaw(rng)

    if stratum.plug == 'sfp':
        # NIC rails: only the target present unless distractors add more.
        for k in NIC_RAILS:
            tb[f'nic_rail_{k}'] = (
                _present_nic(k, rng) if k == target else copy.deepcopy(_ABSENT)
            )
        if distractors:
            others = [k for k in NIC_RAILS if k != target]
            distractor_nic = sorted(rng.sample(others, rng.randint(1, 2)))
            for k in distractor_nic:
                tb[f'nic_rail_{k}'] = _present_nic(k, rng)
        # SC rails: keep standard SFP board furniture (sc_rail_0 present, sc_rail_1 absent).
        tb['sc_rail_0'] = _present_sc(0, rng)
        tb['sc_rail_1'] = copy.deepcopy(_ABSENT)
        task['port_name'] = stratum.port_name
        task['target_module_name'] = f'nic_card_mount_{target}'
    else:  # sc
        # SC rails: only the target present unless distractors add the other.
        for k in SC_RAILS:
            tb[f'sc_rail_{k}'] = (
                _present_sc(k, rng) if k == target else copy.deepcopy(_ABSENT)
            )
        # NIC rails: absent by default; they become distractors when enabled.
        for k in NIC_RAILS:
            tb[f'nic_rail_{k}'] = copy.deepcopy(_ABSENT)
        if distractors:
            other_sc = 1 - target
            tb[f'sc_rail_{other_sc}'] = _present_sc(other_sc, rng)
            distractor_sc = [other_sc]
            distractor_nic = sorted(rng.sample(list(NIC_RAILS), rng.randint(1, 2)))
            for k in distractor_nic:
                tb[f'nic_rail_{k}'] = _present_nic(k, rng)
        task['port_name'] = 'sc_port_base'
        task['target_module_name'] = f'sc_port_{target}'

    # --- END: grasp offset (grasp z uniform in eval window + small jitter) ---
    cable_key = next(iter(trial['scene']['cables']))
    g = trial['scene']['cables'][cable_key]['pose']
    g['gripper_offset']['x'] += rng.uniform(-0.002, 0.002)
    g['gripper_offset']['y'] += rng.uniform(-0.002, 0.002)
    g['gripper_offset']['z'] = rng.uniform(GRASP_Z_MIN, GRASP_Z_MAX)
    g['roll'] += rng.uniform(-0.04, 0.04)
    g['pitch'] += rng.uniform(-0.04, 0.04)
    g['yaw'] += rng.uniform(-0.04, 0.04)

    # --- START: small robot home-joint perturbation ---
    for j in c['robot']['home_joint_positions']:
        c['robot']['home_joint_positions'][j] += rng.uniform(-HOME_JITTER, HOME_JITTER)

    rail_key = f'sc_rail_{target}' if stratum.plug == 'sc' else f'nic_rail_{target}'
    target_pose = tb[rail_key]['entity_pose']
    row = StrataManifestRow(
        config='',
        stratum=stratum.name,
        plug=stratum.plug,
        target_rail=target,
        port_name=task['port_name'],
        target_module_name=task['target_module_name'],
        seed=seed,
        index=index,
        board_x=tb['pose']['x'],
        board_y=tb['pose']['y'],
        board_yaw=tb['pose']['yaw'],
        grasp_z=g['gripper_offset']['z'],
        target_translation=target_pose['translation'],
        target_yaw=target_pose['yaw'],
        distractors_on=distractors,
        n_distractors=len(distractor_nic) + len(distractor_sc),
        distractor_rails=_encode_distractors(distractor_nic, distractor_sc),
        cable_type=task['cable_type'],
    )
    return c, row


def write_manifest(rows: list[StrataManifestRow], path: str) -> None:
    """Write manifest rows to ``path`` as CSV (header from :class:`StrataManifestRow`).

    Args:
        rows: Manifest rows to serialise.
        path: Destination CSV path (parent directory must already exist).
    """
    fieldnames = [f.name for f in dataclasses.fields(StrataManifestRow)]
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


def _run_strata(base: dict[str, Any], args: argparse.Namespace) -> None:
    """Emit ``reps`` configs per stratum plus a manifest CSV."""
    strata = enumerate_strata()
    rows: list[StrataManifestRow] = []
    global_index = 0
    for rep in range(args.reps):
        for stratum in strata:
            c, row = build_strata_config(
                base, stratum, global_index, args.seed, args.distractors
            )
            path = os.path.join(args.o, f'{args.prefix}_{stratum.name}_r{rep}.yaml')
            with open(path, 'w') as fh:
                yaml.safe_dump(c, fh, sort_keys=False)
            row.config = path
            rows.append(row)
            print(f"  {stratum.name:<28} rep{rep} target_rail={stratum.target_rail} "
                  f"n_distractors={row.n_distractors:>1} -> {path}")
            global_index += 1
    manifest_path = args.manifest or os.path.join(args.o, 'manifest.csv')
    write_manifest(rows, manifest_path)
    print(f"strata: wrote {len(rows)} configs ({len(strata)} cells x {args.reps} reps), "
          f"distractors={args.distractors}, manifest -> {manifest_path}")


def _run_perturb(base: dict[str, Any], args: argparse.Namespace) -> None:
    """Emit ``-n`` near/wide perturbed configs (legacy behaviour)."""
    print(f"mode={args.mode}  {'cfg':5} {'bx':>7} {'by':>7} {'yaw':>6} {'rail':>5} {'port':>11}")
    for i in range(args.n):
        c = perturb(base, i, args.seed, args.mode)
        p = f'{args.o}/{args.prefix}_{i}.yaml'
        with open(p, 'w') as fh:
            yaml.safe_dump(c, fh, sort_keys=False)
        tb = c['trials']['trial_1']['scene']['task_board']
        task = c['trials']['trial_1']['tasks']['task_1']
        rail = next((k for k in range(5) if tb[f'nic_rail_{k}'].get('entity_present')), '?')
        print(f"      {i:<5} {tb['pose']['x']:7.3f} {tb['pose']['y']:7.3f} {tb['pose']['yaw']:6.2f} "
              f"{str(rail):>5} {task['port_name']:>11}  -> {p}")


def main() -> None:
    """Parse CLI args and write perturbed (near/wide) or stratified configs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-n', type=int, default=5, help='config count for near/wide modes')
    ap.add_argument('-o', default=os.path.expanduser('~/data/configs'))
    ap.add_argument('--seed', type=int, default=20260618)
    ap.add_argument('--mode', choices=['near', 'wide', 'strata'], default='near')
    ap.add_argument('--prefix', default='cfg')
    ap.add_argument('--reps', type=int, default=1,
                    help='strata mode: configs emitted per stratum cell (12 cells)')
    ap.add_argument('--distractors', action=argparse.BooleanOptionalAction, default=True,
                    help='strata mode: spawn 1-2 distractor entities on non-target rails')
    ap.add_argument('--manifest', default='',
                    help='strata mode: manifest CSV path (default: <outdir>/manifest.csv)')
    a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True)
    with open(SRC) as fh:
        base = yaml.safe_load(fh)
    if a.mode == 'strata':
        _run_strata(base, a)
    else:
        _run_perturb(base, a)


if __name__ == '__main__':
    main()
