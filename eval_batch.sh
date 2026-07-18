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

SUITE="${SUITE:-eval_suite_smoke}"
POLICY="${POLICY:-aic_example_policies.ros.DeployACT}"
QUEUE="${QUEUE:-p1_k16:/home/kiwoos/training/ckpt/p1_k16.pt v2_wide:/home/kiwoos/training/ckpt/v2_wide.pt p1_k8:/home/kiwoos/training/ckpt/p1_k8.pt}"

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
    --checkpoint "$ckpt" --out "$out/"
  rc=$?
  echo "==== [eval_batch] EXIT $name rc=$rc $(date '+%F %T') ===="
  # Belt-and-braces: kill any straggler sim procs between runs (bracket-safe).
  pkill -9 -f 'gz [s]im' 2>/dev/null
  pkill -9 -f 'aic_[e]ngine' 2>/dev/null
  pkill -9 -f 'rmw_[z]enohd' 2>/dev/null
  pkill -9 -f 'aic_[m]odel' 2>/dev/null
  sleep 5
done
echo "EVALBATCHDONE $(date '+%F %T')"
