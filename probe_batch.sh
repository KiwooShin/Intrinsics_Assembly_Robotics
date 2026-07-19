#!/usr/bin/env bash
# Self-driving guarded-descent PROBE batch: runs the deployed policy with the
# opt-in guarded-descent extension (AIC_GUARDED=1) over the 3 official configs x
# N reps each, sequentially, resumably, with no agent babysitting (CLAUDE.md §6
# fix after repeated agent-waiter stalls). Modeled EXACTLY on eval_batch.sh:
# flock single-instance lock, preflight orphan sweep, per-unit skip-if-done
# resume, and a DONE marker line watchdogs grep for.
#
# The probe answers the day-1 judge question: once the learned policy stalls
# 0.05-0.08 m short of the port, does a wrench-guarded scripted descent along the
# estimated approach axis recover the insertion? Guarded descent is a model-free
# fallback layered on the deployed node; it only activates AFTER StallDetector
# fires, so with the flag on but no stall the run is identical to the deployed
# policy. See aic_example_policies/ros/guarded_descent.py.
#
# Usage:
#   nohup bash probe_batch.sh >> results/probe_guarded/probe_batch.log 2>&1 &
# Env (all optional; defaults reproduce the adopted deployment + guarded probe):
#   SUITE_SRC   suite dir holding the official configs (default eval_suite_ab5)
#   CONFIGS     space-separated config ids     (default "official_1 official_2 official_3")
#   REPS        reps per config                (default 3)
#   POLICY      policy class    (default aic_example_policies.ros.DeployACT)
#   CKPT        checkpoint      (default /home/kiwoos/training/ckpt/v2_wide.pt, the adopted ckpt)
#   OUT         output root     (default results/probe_guarded)
#   AIC_ENSEMBLE   temporal ensembling (default 1 — the adopted v2_wide deployment)
#   AIC_GUARDED_*  guarded-descent overrides (speed/window/step/cap/force/zstiffness);
#                  passed straight through to the policy node (see guarded_descent.py).
# A unit (<cfg>_r<i>) is SKIPPED if its trial dir already holds a PROBEDONE
# marker (resume). Emits "PROBEBATCHDONE" on completion.
set -u
cd "$(dirname "$0")"

# Single-instance lock: two concurrent batches share the sim and fratricide each
# other's processes (observed 2026-07-18). flock refuses a second start. A
# DIFFERENT lock file from eval_batch.sh so the two never run at once either --
# they still share the one global sim/zenoh graph.
LOCK=/tmp/aic_probe_batch.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[probe_batch] another instance holds $LOCK — refusing to start"
  exit 1
fi
# Also refuse to start while an eval batch holds ITS lock: both drive the one
# global sim and would fratricide each other's trials.
if [ -e /tmp/aic_eval_batch.lock ] && ! flock -n -w 0 -x /tmp/aic_eval_batch.lock true 2>/dev/null; then
  echo "[probe_batch] eval_batch.sh appears to be running (/tmp/aic_eval_batch.lock busy) — refusing to start"
  exit 1
fi

# PREFLIGHT orphan sweep (WEDGE-DEBUG 2026-07-18 fratricide fix). Run ONCE, at
# batch start, BEFORE the first trial: kill leaked PPID==1 orphans left by a
# purged sim or a crashed prior batch. Bracketed patterns keep the matcher from
# matching itself; PPID==1 plus pid!=$$ guarantees we never touch this batch's
# own process tree (our live children are never reparented to init). Identical
# to eval_batch.sh — the same global ROS/gz graph is being cleaned.
preflight_orphan_sweep() {
  local self=$$
  local pat='aic_[a]dapter|static_[t]ransform_publisher|[r]elay|gz [s]im|aic_[e]ngine|rmw_[z]enohd|aic_[m]odel|component_[c]ontainer'
  ps -eo pid=,ppid=,args= 2>/dev/null \
    | awk -v self="$self" -v pat="$pat" '$2==1 && $1!=self && $0~pat {print $1}' \
    | while read -r pid; do
        [ -n "$pid" ] || continue
        echo "[probe_batch] preflight: killing orphan pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
      done
}
preflight_orphan_sweep

SUITE_SRC="${SUITE_SRC:-eval_suite_ab5}"
CONFIGS="${CONFIGS:-official_1 official_2 official_3}"
REPS="${REPS:-3}"
POLICY="${POLICY:-aic_example_policies.ros.DeployACT}"
CKPT="${CKPT:-/home/kiwoos/training/ckpt/v2_wide.pt}"
OUT="${OUT:-results/probe_guarded}"
# DeployACT (a torch policy) MUST run under the deploy venv interpreter; the
# default `ros2 run aic_model aic_model` uses /usr/bin/python3 (no torch), so the
# node crashes on import, never registers, and every trial scores 0. Pin the venv
# launcher. PYTHONUNBUFFERED so the policy's stdout/stderr (startup line incl. the
# guarded_descent=ON banner, HANDOFF + phase-change lines) are flushed to run.log.
POLICY_CMD="${POLICY_CMD:-/home/kiwoos/venvs/aic-deploy/bin/python -u /home/kiwoos/ws_aic/install/lib/aic_model/aic_model}"

# THE probe switch: enable the guarded-descent extension in the policy node. The
# policy node inherits this (and any AIC_GUARDED_* / AIC_ENSEMBLE overrides) via
# the environment: probe_batch.sh -> eval_suite.py -> runner Popen(bash bringup)
# -> POLICY_CMD, none of which resets the env. Default AIC_ENSEMBLE=1 reproduces
# the adopted v2_wide deployment on top of which the probe is layered.
export AIC_GUARDED=1
export AIC_ENSEMBLE="${AIC_ENSEMBLE:-1}"

mkdir -p "$OUT/_suites"
echo "==== [probe_batch] START $(date '+%F %T') configs=[$CONFIGS] reps=$REPS ckpt=$CKPT ===="
echo "==== [probe_batch] AIC_GUARDED=$AIC_GUARDED AIC_ENSEMBLE=$AIC_ENSEMBLE guarded overrides: $(env | grep '^AIC_GUARDED_' | tr '\n' ' ') ===="

for cfg in $CONFIGS; do
  # Pull the config's manifest row once; it carries stratum/pose columns the
  # runner's SuiteMember needs. Fail loudly if the config is unknown.
  row="$(awk -F, -v c="$cfg" '$1==c {print; exit}' "$SUITE_SRC/manifest.csv")"
  if [ -z "$row" ]; then
    echo "==== [probe_batch] ERROR config '$cfg' not found in $SUITE_SRC/manifest.csv — skipping ===="
    continue
  fi
  abscfg="$PWD/$SUITE_SRC/configs/${cfg}.yaml"
  if [ ! -f "$abscfg" ]; then
    echo "==== [probe_batch] ERROR config file $abscfg missing — skipping ===="
    continue
  fi

  for i in $(seq 1 "$REPS"); do
    unit="${cfg}_r${i}"
    trial_dir="$OUT/trials/$unit"
    if [ -f "$trial_dir/PROBEDONE" ]; then
      echo "==== [probe_batch] SKIP_DONE $unit ($trial_dir/PROBEDONE exists) ===="
      continue
    fi

    # Build a 1-member temp suite whose only member is this rep: config_id renamed
    # to <cfg>_r<i> (so the runner writes trials/<cfg>_r<i>) and config_file set to
    # the absolute official-config path (so it resolves regardless of suite dir).
    suite_dir="$OUT/_suites/$unit"
    mkdir -p "$suite_dir"
    head -1 "$SUITE_SRC/manifest.csv" > "$suite_dir/manifest.csv"
    echo "$row" | awk -F, -v OFS=, -v id="$unit" -v cf="$abscfg" '{$1=id; $NF=cf; print}' \
      >> "$suite_dir/manifest.csv"

    echo "==== [probe_batch] RUN $unit ckpt=$CKPT $(date '+%F %T') ===="
    python3 eval_suite.py run --suite "$suite_dir/" --policy "$POLICY" \
      --checkpoint "$CKPT" --out "$OUT/" --policy-cmd "$POLICY_CMD" --name "$unit"
    rc=$?
    echo "==== [probe_batch] EXIT $unit rc=$rc $(date '+%F %T') ===="

    # Mark done only if the trial actually scored (scoring.yaml written); a failed
    # trial is left unmarked so a resumed batch retries it (mirrors eval_batch's
    # summary.json gate). run.log holds the wrench trace + guarded HANDOFF/phase
    # lines; scoring.yaml holds the engine score, as with every eval run.
    if [ -f "$trial_dir/scoring.yaml" ]; then
      : > "$trial_dir/PROBEDONE"
      echo "==== [probe_batch] PROBEDONE $unit ($trial_dir/scoring.yaml present) ===="
    else
      echo "==== [probe_batch] NOSCORE $unit (no scoring.yaml — will retry on resume) ===="
    fi
  done
done
echo "PROBEBATCHDONE $(date '+%F %T')"
