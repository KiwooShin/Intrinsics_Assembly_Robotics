"""Per-config evaluation runner: sim orchestration, scoring, result rows.

The sim-interaction layer is isolated behind :class:`SimRunner` (a thin,
mockable class that mirrors ``collect_one.sh``'s bringup/teardown). Tests and
``--dry-run`` swap in :class:`DryRunSimRunner`, which fabricates a plausible
``scoring.yaml`` so the identical parse -> classify -> aggregate path is
exercised without ROS, Gazebo, or a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from . import scoring
from .suite import SuiteMember, read_manifest

logger = logging.getLogger(__name__)

# Engine log markers (aic_engine/src/aic_engine.cpp). A trial reaches a terminal
# state on *any* of these lines, not only the success one:
#   - success banner (Engine::run_trials, "... completed successfully! Score:")
#   - per-trial scoring line, printed on BOTH the success and the
#     task-timeout/failure paths (Engine::score_trial,
#     "Finished scoring trial, total score is:")
#   - failure banner, printed after a failed/timed-out trial
#     (Engine::run_trials, "... failed or was not completed. Score:")
# Matching only the success marker made the runner wait out its whole timeout on
# any trial that did not insert, then kill the engine *before* it wrote
# scoring.yaml -> the trial was auto-scored 0 ("no scoring").
COMPLETION_MARKERS = (
    "completed successfully! Score:",
    "Finished scoring trial, total score is:",
    "failed or was not completed. Score:",
)
READY_MARKERS = ("No node with name", "Starting trial 'trial_1'")

# The engine prints a terminal marker (above) from handle_trial, then tears the
# model node down and only afterwards writes scoring.yaml (Engine::score_run).
# Once a marker is seen, wait up to this many wall-seconds for the file to
# appear before tearing the sim down, so a just-completed trial is never killed
# mid-score.
SCORING_WAIT_S = 120.0

# Per-trial completion timeout (wall seconds).
#
# The engine's task ``time_limit`` (180 s in the eval configs) is measured in
# SIMULATED time: ``wait_for_interruptible`` polls ``node_->now()`` and the
# engine node runs with ``use_sim_time:=true`` (aic_gz_bringup.launch.py). A
# trial that never inserts only reaches that limit -- and only then cancels the
# task, scores it, and writes scoring.yaml -- after 180 s of *sim* time. At the
# real-time factor observed while the learned policy runs inference
# (RTF ~= 0.05, i.e. sim ~20x slower than wall; measured from run.log ``t=``
# sim-clock stamps vs wall-clock log-line headers), 180 sim-s is ~3600 wall-s,
# plus sim bring-up (~2-3 min) and scoring/teardown (~2-3 min). The previous
# 600 s (10 min) timeout fired at ~30 sim-s, killing every slow/failed trial
# long before it could be scored. Size the default to comfortably exceed the
# worst case at a conservative RTF ~= 0.045 (180 / 0.045 ~= 4000 wall-s +
# overhead).
DEFAULT_TRIAL_TIMEOUT_S = 5400.0  # 90 minutes


@dataclasses.dataclass(frozen=True)
class SimEnv:
    """Environment for launching the simulator (mirrors ``collect_one.sh``).

    Attributes:
        ros_setup: ``setup.bash`` for the ROS distro.
        ws_setup: ``setup.bash`` for the built AIC workspace.
        rmw_implementation: ``RMW_IMPLEMENTATION`` value.
        gz_rendering_plugin_path: ``GZ_RENDERING_PLUGIN_PATH`` value.
        egl_vendor_library_filenames: ``__EGL_VENDOR_LIBRARY_FILENAMES`` value.
        display: X ``DISPLAY`` for headless rendering.
        zenoh_startup_s: Seconds to wait after starting ``rmw_zenohd``.
        policy_launch_cmd: Command that launches the policy node, before the
            ``--ros-args`` block. Defaults to ``ros2 run aic_model aic_model``,
            which runs the installed entry point under the system interpreter
            (``/usr/bin/python3``, no torch). Torch-backed policies such as
            ``DeployACT`` require the deploy venv interpreter, e.g.
            ``/home/kiwoos/venvs/aic-deploy/bin/python
            /home/kiwoos/ws_aic/install/lib/aic_model/aic_model``.
    """

    ros_setup: str = "/opt/ros/kilted/setup.bash"
    ws_setup: str = "/home/kiwoos/ws_aic/install/setup.bash"
    rmw_implementation: str = "rmw_zenoh_cpp"
    gz_rendering_plugin_path: str = (
        "/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins"
    )
    egl_vendor_library_filenames: str = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    display: str = ":2"
    zenoh_startup_s: int = 8
    policy_launch_cmd: str = (
        "/home/kiwoos/venvs/aic-deploy/bin/python -u "
        "/home/kiwoos/ws_aic/install/lib/aic_model/aic_model"
    )


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    """Low-level result of one sim launch.

    Attributes:
        completed: Whether the engine printed a completion marker.
        timed_out: Whether the wait for completion hit the timeout.
        duration_s: Wall-clock seconds spent running the trial.
        scoring_path: Path to the produced ``scoring.yaml`` (may not exist on
            failure).
        log_path: Path to the captured engine log (if any).
        error: Optional human-readable failure reason.
    """

    completed: bool
    timed_out: bool
    duration_s: float
    scoring_path: Path
    log_path: Path | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Aggregated per-config evaluation record (one manifest row scored).

    Attributes:
        config_id: Suite member id.
        source: ``"stratified"`` or ``"official"``.
        rail: Stratum rail index.
        plug: Stratum plug family.
        port: Stratum port index.
        outcome: Classified outcome label (see :class:`scoring.Outcome`).
        inserted: Whether a full correct-port insertion was scored.
        insertion_success: Alias of ``inserted`` used for the success rate.
        total: Total trial score.
        tier_1: Tier-1 score.
        tier_2: Tier-2 aggregate score.
        tier_3: Tier-3 task score.
        force_score: Tier-2 insertion-force sub-score.
        contacts_score: Tier-2 contacts sub-score.
        duration_score: Tier-2 duration sub-score.
        smoothness_score: Tier-2 smoothness sub-score.
        efficiency_score: Tier-2 efficiency sub-score.
        duration_s: Wall-clock seconds for the run.
        completed: Whether the engine completed the trial.
        timed_out: Whether the run timed out.
        error: Optional failure reason.
    """

    config_id: str
    source: str
    rail: int
    plug: str
    port: int
    outcome: str
    inserted: bool
    insertion_success: bool
    total: float
    tier_1: float
    tier_2: float
    tier_3: float
    force_score: float
    contacts_score: float
    duration_score: float
    smoothness_score: float
    efficiency_score: float
    duration_s: float
    completed: bool
    timed_out: bool
    error: str | None = None

    @classmethod
    def from_breakdown(
        cls,
        member: SuiteMember,
        breakdown: scoring.TrialScoreBreakdown,
        run: RunOutcome,
    ) -> "TrialResult":
        """Build a result row from a parsed breakdown and a run outcome.

        Args:
            member: The suite member that was run.
            breakdown: Parsed trial score.
            run: Low-level run outcome (timing, completion).

        Returns:
            The populated :class:`TrialResult`.
        """
        return cls(
            config_id=member.config_id,
            source=member.source,
            rail=member.stratum.rail,
            plug=member.stratum.plug,
            port=member.stratum.port,
            outcome=breakdown.outcome.value,
            inserted=breakdown.inserted,
            insertion_success=breakdown.inserted,
            total=breakdown.total,
            tier_1=breakdown.tier_1,
            tier_2=breakdown.tier_2,
            tier_3=breakdown.tier_3,
            force_score=breakdown.insertion_force_score,
            contacts_score=breakdown.contacts_score,
            duration_score=breakdown.duration_score,
            smoothness_score=breakdown.smoothness_score,
            efficiency_score=breakdown.efficiency_score,
            duration_s=run.duration_s,
            completed=run.completed,
            timed_out=run.timed_out,
            error=run.error,
        )

    @classmethod
    def failed(cls, member: SuiteMember, run: RunOutcome) -> "TrialResult":
        """Build a zero-score result row for a run that produced no scoring.

        Args:
            member: The suite member that was run.
            run: Low-level run outcome carrying the failure reason.

        Returns:
            A :class:`TrialResult` with zeroed scores and ``outcome == 'miss'``.
        """
        return cls(
            config_id=member.config_id,
            source=member.source,
            rail=member.stratum.rail,
            plug=member.stratum.plug,
            port=member.stratum.port,
            outcome=scoring.Outcome.MISS.value,
            inserted=False,
            insertion_success=False,
            total=0.0,
            tier_1=0.0,
            tier_2=0.0,
            tier_3=0.0,
            force_score=0.0,
            contacts_score=0.0,
            duration_score=0.0,
            smoothness_score=0.0,
            efficiency_score=0.0,
            duration_s=run.duration_s,
            completed=run.completed,
            timed_out=run.timed_out,
            error=run.error or "no scoring.yaml produced",
        )


class SimRunner:
    """Launches the simulator for one trial and returns its ``scoring.yaml``.

    This class is the single seam between the harness and ROS/Gazebo. It builds
    a bringup script mirroring ``collect_one.sh``: start ``rmw_zenohd``, launch
    ``aic_gz_bringup`` with the trial config and a *unique* ``AIC_RESULTS_DIR``
    (the engine overwrites ``scoring.yaml`` per run), start ``aic_model`` with
    the requested policy, wait for the engine completion marker, then tear down.
    It is intentionally not invoked by the unit tests (which mock it) so the
    tests need no ROS/GPU.
    """

    def __init__(
        self, env: SimEnv | None = None, timeout_s: float = DEFAULT_TRIAL_TIMEOUT_S
    ) -> None:
        """Initialize the runner.

        Args:
            env: Simulator environment; defaults to :class:`SimEnv` defaults.
            timeout_s: Per-trial completion timeout in seconds.
        """
        self.env = env or SimEnv()
        self.timeout_s = timeout_s

    def _bringup_script(
        self, config_path: Path, policy: str, results_dir: Path, log_path: Path
    ) -> str:
        """Return the bash bringup script for one trial.

        Args:
            config_path: Absolute path to the trial config YAML.
            policy: ROS ``policy`` parameter value (``module.Class``).
            results_dir: Unique ``AIC_RESULTS_DIR`` for this trial.
            log_path: File to capture the engine/model log to.

        Returns:
            The bash script as a string.
        """
        env = self.env
        completion_re = "|".join(m.replace("!", r"\!") for m in COMPLETION_MARKERS)
        ready_re = "|".join(READY_MARKERS)
        scoring_yaml = results_dir / "scoring.yaml"
        scoring_wait_iters = int(SCORING_WAIT_S // 2)
        return f"""#!/bin/bash
# ROS/ament ``setup.bash`` reference unbound shell vars (e.g.
# AMENT_TRACE_SETUP_FILES); under ``set -u`` a non-interactive bash would exit
# at the first ``source`` before the sim ever launches. Enable nounset only
# after the environment is sourced (mirrors ``collect_one.sh``, which omits it).
source {env.ros_setup}
source {env.ws_setup}
set -u
export RMW_IMPLEMENTATION={env.rmw_implementation}
export GZ_RENDERING_PLUGIN_PATH={env.gz_rendering_plugin_path}
export __EGL_VENDOR_LIBRARY_FILENAMES={env.egl_vendor_library_filenames}
export DISPLAY={env.display}
export AIC_RESULTS_DIR={results_dir}
mkdir -p "{results_dir}"

cleanup() {{
  # Protect this script's ancestor chain (the eval harness python and any
  # wrapper/launcher shells above it). Their argv can contain the aic_model
  # launcher path via --policy-cmd, so the grep below would otherwise match and
  # kill -9 the harness. The policy node is a *child* of this script, not an
  # ancestor, so it is still torn down between trials.
  KEEP=" "
  ANC=$$
  while [ "${{ANC:-0}}" -gt 1 ]; do
    KEEP="$KEEP$ANC "
    ANC=$(awk '{{print $4}}' /proc/$ANC/stat 2>/dev/null)
  done
  PIDS=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container|rmw_zenohd" \
    | grep -v grep | awk '{{print $2}}')
  for pid in $PIDS; do
    case "$KEEP" in
      *" $pid "*) ;;
      *) kill -9 "$pid" 2>/dev/null || true ;;
    esac
  done
  sleep 4
}}
trap cleanup EXIT
cleanup

ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
sleep {env.zenoh_startup_s}

# NOTE(2026-07-18): gazebo_gui:=false was tried here to save the ~150%-CPU GUI
# renderer, but the very first headless trial wedged hard at t=16 sim-s: the
# whole stack froze with the policy blocked in get_observation — the offscreen
# camera rendering appears to depend on the GL/EGL context the GUI client keeps
# alive on the VNC display. Reverted: the GUI stays. The 60-sim-s fast suite is
# the reliable wall-clock win instead.
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  aic_engine_config_file:={config_path} \
  ground_truth:=true start_aic_engine:=true launch_rviz:=false \
  > "{log_path}" 2>&1 &

for i in $(seq 1 45); do
  sleep 2
  if grep -qE "{ready_re}" "{log_path}" 2>/dev/null; then break; fi
done

{env.policy_launch_cmd} --ros-args -p use_sim_time:=true \
  -p policy:={policy} >> "{log_path}" 2>&1 &

# Poll for a terminal engine marker (success OR failure/timeout) or for the
# scoring.yaml artifact itself, whichever comes first. Matching only the success
# marker previously let this loop run to its full timeout on any trial that did
# not insert, killing the engine before it scored.
for i in $(seq 1 {int(self.timeout_s // 5)}); do
  sleep 5
  if [ -f "{scoring_yaml}" ]; then break; fi
  if grep -qE "{completion_re}" "{log_path}" 2>/dev/null; then break; fi
done
# A terminal marker precedes the scoring.yaml write (the engine cleans up and
# shuts the model node down in between), so wait bounded for the file before the
# teardown below so a just-completed trial is not killed mid-score.
for i in $(seq 1 {scoring_wait_iters}); do
  if [ -f "{scoring_yaml}" ]; then break; fi
  sleep 2
done
sleep 4
"""

    def run_trial(
        self, member: SuiteMember, config_path: Path, policy: str, results_dir: Path
    ) -> RunOutcome:
        """Run one trial in the simulator and return its outcome.

        Args:
            member: Suite member being run (for logging context).
            config_path: Absolute path to the trial config YAML.
            policy: ROS ``policy`` parameter value.
            results_dir: Unique output directory (becomes ``AIC_RESULTS_DIR``).

        Returns:
            A :class:`RunOutcome` pointing at the produced ``scoring.yaml``.
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "run.log"
        script_path = results_dir / "bringup.sh"
        script = self._bringup_script(config_path, policy, results_dir, log_path)
        script_path.write_text(script)
        scoring_path = results_dir / "scoring.yaml"
        start = time.monotonic()
        timed_out = False
        error: str | None = None
        try:
            # start_new_session=True puts the bringup (and the whole sim process
            # tree it launches) in its own session/process group. At trial
            # completion the engine shuts the model node down and ``ros2 launch``
            # signals its *process group*; without this isolation that group is
            # the harness's own group, so the shutdown signal would kill the
            # harness mid-run (observed: harness dies at insert-return with no
            # traceback). Isolating the session confines that signal to the sim.
            subprocess.run(
                ["bash", str(script_path)],
                timeout=self.timeout_s + 120.0,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            error = "sim bringup exceeded hard timeout"
        duration_s = time.monotonic() - start
        completed = _log_has_completion(log_path)
        if not scoring_path.is_file() and not completed:
            error = error or "engine produced no scoring.yaml"
        return RunOutcome(
            completed=completed,
            timed_out=timed_out,
            duration_s=duration_s,
            scoring_path=scoring_path,
            log_path=log_path,
            error=error,
        )


class DryRunSimRunner:
    """Fabricates a plausible ``scoring.yaml`` without touching ROS.

    Used by ``--dry-run`` and by unit tests. Scores are derived deterministically
    from the config id so that a dry run produces a varied-but-reproducible
    result set that flows through the entire parse/classify/aggregate pipeline.
    """

    def __init__(self, insertion_bias: float = 0.4) -> None:
        """Initialize the fake runner.

        Args:
            insertion_bias: Fraction of configs (by hash) that "fully insert".
        """
        self.insertion_bias = insertion_bias

    def run_trial(
        self, member: SuiteMember, config_path: Path, policy: str, results_dir: Path
    ) -> RunOutcome:
        """Fabricate a scoring file and return a successful outcome.

        Args:
            member: Suite member being run (seeds the fabricated scores).
            config_path: Ignored (kept for interface parity).
            policy: Ignored (kept for interface parity).
            results_dir: Directory to write the fabricated ``scoring.yaml``.

        Returns:
            A :class:`RunOutcome` marked completed.
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        scoring_path = results_dir / "scoring.yaml"
        doc = fabricate_scoring_dict(member, self.insertion_bias)
        with scoring_path.open("w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False)
        return RunOutcome(
            completed=True,
            timed_out=False,
            duration_s=0.0,
            scoring_path=scoring_path,
            log_path=None,
            error=None,
        )


def _hash_unit(text: str, salt: str = "") -> float:
    """Map a string to a deterministic float in ``[0, 1)``.

    Args:
        text: Input string.
        salt: Optional salt to decorrelate multiple draws from one key.

    Returns:
        A pseudo-uniform value in ``[0, 1)``.
    """
    digest = hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def fabricate_scoring_dict(member: SuiteMember, insertion_bias: float) -> dict[str, Any]:
    """Build a schema-accurate fake ``scoring.yaml`` document for one member.

    Args:
        member: Suite member seeding the fabricated scores.
        insertion_bias: Fraction of configs that fully insert.

    Returns:
        A mapping matching the engine's ``scoring.yaml`` schema.
    """
    key = member.config_id
    roll = _hash_unit(key, "insert")
    inserted = roll < insertion_bias
    if inserted:
        tier_3 = scoring.TIER3_FULL_INSERTION
        tier_3_msg = "Cable insertion successful."
    elif roll < insertion_bias + 0.2:
        tier_3 = 38.0 + 12.0 * _hash_unit(key, "partial")
        tier_3_msg = "Partial insertion detected."
    elif roll < insertion_bias + 0.5:
        tier_3 = 25.0 * _hash_unit(key, "prox")
        tier_3_msg = "No insertion detected. Final plug port distance."
    else:
        tier_3 = 0.0
        tier_3_msg = "Task not completed."
    force = -12.0 if _hash_unit(key, "force") < 0.1 else 0.0
    contacts = -24.0 if _hash_unit(key, "contact") < 0.1 else 0.0
    if tier_3 > 0:
        duration = 12.0 * _hash_unit(key, "dur")
        smoothness = 6.0 * _hash_unit(key, "jerk")
        efficiency = 6.0 * _hash_unit(key, "eff")
    else:
        duration = smoothness = efficiency = 0.0
    tier_2 = force + contacts + duration + smoothness + efficiency
    tier_1 = 1.0
    total = tier_1 + tier_2 + tier_3
    return {
        "total": total,
        "trial_1": {
            "tier_1": {"score": tier_1, "message": "Model validation succeeded."},
            "tier_2": {
                "score": tier_2,
                "message": "Scoring succeeded.",
                "categories": {
                    scoring.CAT_INSERTION_FORCE: {"score": force, "message": ""},
                    scoring.CAT_CONTACTS: {"score": contacts, "message": ""},
                    scoring.CAT_DURATION: {"score": duration, "message": ""},
                    scoring.CAT_SMOOTHNESS: {"score": smoothness, "message": ""},
                    scoring.CAT_EFFICIENCY: {"score": efficiency, "message": ""},
                },
            },
            "tier_3": {"score": tier_3, "message": tier_3_msg},
        },
    }


def _log_has_completion(log_path: Path) -> bool:
    """Return whether the engine log contains a completion marker.

    Args:
        log_path: Path to the captured engine log.

    Returns:
        True if any completion marker is present.
    """
    if not log_path.is_file():
        return False
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return False
    return any(marker in text for marker in COMPLETION_MARKERS)


def score_member(
    member: SuiteMember,
    run: RunOutcome,
) -> TrialResult:
    """Parse a run's ``scoring.yaml`` into a :class:`TrialResult`.

    Args:
        member: The suite member that was run.
        run: The low-level run outcome.

    Returns:
        A populated :class:`TrialResult`; a zero-score row if parsing fails.
    """
    if not run.scoring_path.is_file():
        return TrialResult.failed(member, run)
    try:
        result = scoring.parse_scoring_file(run.scoring_path)
        breakdown = result.single_trial()
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("failed to parse %s: %s", run.scoring_path, exc)
        return TrialResult.failed(
            member, dataclasses.replace(run, error=f"parse error: {exc}")
        )
    return TrialResult.from_breakdown(member, breakdown, run)


def run_suite(
    suite_dir: str | Path,
    out_dir: str | Path,
    policy: str,
    checkpoint: str | Path | None = None,
    runner: Any | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    env: SimEnv | None = None,
    timeout_s: float = DEFAULT_TRIAL_TIMEOUT_S,
) -> list[TrialResult]:
    """Run every suite member and return per-config result rows.

    Exports the checkpoint path into the environment so the policy class can load
    its weights; the delivery mechanism is centralised here and easy to change.
    Two variable names are set for compatibility: ``AIC_CKPT`` (read by
    ``aic_example_policies.ros.DeployACT``) and the generic ``AIC_CHECKPOINT``.
    Child processes (the bringup bash script and the policy node it launches)
    inherit these via the environment.

    Args:
        suite_dir: Directory containing ``manifest.csv`` and ``configs/``.
        out_dir: Destination for per-trial artifacts and result rows.
        policy: ROS ``policy`` parameter value (``module.Class``).
        checkpoint: Optional checkpoint path exported as ``AIC_CKPT`` and
            ``AIC_CHECKPOINT``.
        runner: Sim-interaction object with ``run_trial(...)``; defaults to
            :class:`SimRunner` (or :class:`DryRunSimRunner` when ``dry_run``).
        dry_run: If True, use the fabricating runner (no sim).
        limit: If set, only run the first ``limit`` members.
        env: Simulator environment for the default :class:`SimRunner`.
        timeout_s: Per-trial completion timeout.

    Returns:
        The list of :class:`TrialResult` rows in manifest order.

    Raises:
        FileNotFoundError: If the manifest or a config file is missing.
    """
    suite_path = Path(suite_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    members = read_manifest(suite_path / "manifest.csv")
    if limit is not None:
        members = members[:limit]
    if runner is None:
        runner = DryRunSimRunner() if dry_run else SimRunner(env=env, timeout_s=timeout_s)
    if checkpoint is not None:
        os.environ["AIC_CKPT"] = str(checkpoint)
        os.environ["AIC_CHECKPOINT"] = str(checkpoint)

    results: list[TrialResult] = []
    trials_root = out_path / "trials"
    for i, member in enumerate(members, start=1):
        config_path = (suite_path / member.config_file).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"config for {member.config_id} not found: {config_path}")
        results_dir = trials_root / member.config_id
        logger.info(
            "[%d/%d] %s (%s) -> %s",
            i,
            len(members),
            member.config_id,
            member.stratum.cell_id(),
            results_dir,
        )
        run = runner.run_trial(member, config_path, policy, results_dir)
        result = score_member(member, run)
        results.append(result)
        # Keep a copy of the scoring file alongside the run for provenance.
        if run.scoring_path.is_file() and run.scoring_path.parent != results_dir:
            shutil.copy2(run.scoring_path, results_dir / "scoring.yaml")
    return results
