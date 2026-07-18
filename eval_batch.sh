#!/usr/bin/env bash
# Self-driving evaluation batch: runs eval_suite.py over a queue of checkpoints
# sequentially, resumably, with no agent babysitting (CLAUDE.md §6 fix after
# repeated agent-waiter stalls). Mirrors collect_campaign.sh robustness.
#
# Usage:
#   nohup bash eval_batch.sh >> results/eval_batch.log 2>&1 &
# Env:
#   SUITE   suite dir (default eval_suite_smoke)
#   POLICY  policy class (default aic_example_policies.ros.DeployACT)
#   QUEUE   space-separated "name:ckpt_path" pairs (default: the P1 comparison)
# A run is SKIPPED if its results dir already has a summary.json (resume).
# Emits "EVALBATCHDONE" on completion — watchdogs grep for it.
set -u
cd "$(dirname "$0")"

# Single-instance lock: two concurrent batches share the sim and fratricide
# each other's processes (observed 2026-07-18). flock refuses a second start.
LOCK=/tmp/aic_eval_batch.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[eval_batch] another instance holds $LOCK — refusing to start"
  exit 1
fi

# PREFLIGHT orphan sweep (WEDGE-DEBUG 2026-07-18 fratricide fix). Run ONCE, at
# batch start, BEFORE the first trial: kill leaked PPID==1 orphans left by a
# purged sim or a crashed prior batch (aic_adapter, static_transform_publisher,
# relay, gz sim, aic_engine, rmw_zenohd, aic_model, component_container get
# reparented to init when their bringup dies). This is NOT a between-run kill:
# runner.py now reaps each trial's own process group on every exit path, so
# there is no global pkill between runs (that was the fratricide -- it reaped
# peer trials). Bracketed patterns keep the matcher from matching itself;
# PPID==1 plus pid!=$$ guarantees we never touch this batch's own process tree
# (our live children are never reparented to init).
preflight_orphan_sweep() {
  local self=$$
  local pat='aic_[a]dapter|static_[t]ransform_publisher|[r]elay|gz [s]im|aic_[e]ngine|rmw_[z]enohd|aic_[m]odel|component_[c]ontainer'
  ps -eo pid=,ppid=,args= 2>/dev/null \
    | awk -v self="$self" -v pat="$pat" '$2==1 && $1!=self && $0~pat {print $1}' \
    | while read -r pid; do
        [ -n "$pid" ] || continue
        echo "[eval_batch] preflight: killing orphan pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
      done
}
preflight_orphan_sweep

SUITE="${SUITE:-eval_suite_smoke}"
POLICY="${POLICY:-aic_example_policies.ros.DeployACT}"
QUEUE="${QUEUE:-p1_k16:/home/kiwoos/training/ckpt/p1_k16.pt v2_wide:/home/kiwoos/training/ckpt/v2_wide.pt p1_k8:/home/kiwoos/training/ckpt/p1_k8.pt}"
# DeployACT (a torch policy) MUST run under the deploy venv interpreter; the
# default `ros2 run aic_model aic_model` uses /usr/bin/python3 (no torch), so the
# node crashes on import, never registers, and every trial scores 0 ("Model
# validation failed"). Pin the venv launcher. PYTHONUNBUFFERED so the policy's
# stdout/stderr (startup line, tracebacks, per-step pred logs) are flushed to
# run.log instead of being lost when the node is SIGKILL'd at teardown.
POLICY_CMD="${POLICY_CMD:-/home/kiwoos/venvs/aic-deploy/bin/python -u /home/kiwoos/ws_aic/install/lib/aic_model/aic_model}"

for pair in $QUEUE; do
  name="${pair%%:*}"
  ckpt="${pair#*:}"
  out="results/${name}_smoke"
  if [ -f "$out/summary.json" ]; then
    echo "==== [eval_batch] SKIP_DONE $name ($out/summary.json exists) ===="
    continue
  fi
  echo "==== [eval_batch] RUN $name ckpt=$ckpt $(date '+%F %T') ===="
  python3 eval_suite.py run --suite "$SUITE/" --policy "$POLICY" \
    --checkpoint "$ckpt" --out "$out/" --policy-cmd "$POLICY_CMD"
  rc=$?
  echo "==== [eval_batch] EXIT $name rc=$rc $(date '+%F %T') ===="
  # NO between-run kill. runner.py now reaps each trial's own process group on
  # every exit path (os.killpg on success/timeout/exception), so there are no
  # stragglers to sweep here. A global pkill between runs is exactly the
  # fratricide this change removes: it killed peer/sibling batches' processes.
done
echo "EVALBATCHDONE $(date '+%F %T')"
