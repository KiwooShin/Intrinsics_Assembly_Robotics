#!/bin/bash
# Per-demo storage-light pipeline: for each config -> collect, keep only full-insert
# successes (score>=THRESH), convert+trim to an episode, then DELETE the raw bag.
# Usage: collect_convert.sh <cfg_dir> <prefix> <N> <out_dataset_dir> [start_ep_idx]
CFG_DIR=${1:-/home/kiwoos/data/configs}
PREFIX=${2:-wide}
N=${3:-12}
OUT=${4:-/home/kiwoos/training/wide}
EP=${5:-0}            # next episode index to write
THRESH=60            # full insertion ~93, partial ~38
mkdir -p "$OUT"
keep=0; fail=0
echo "=== collect_convert: $N x $PREFIX -> $OUT (start ep_$EP) $(date +%H:%M:%S) ==="
for i in $(seq 0 $((N-1))); do
  CFG=$CFG_DIR/${PREFIX}_${i}.yaml
  [ -f "$CFG" ] || { echo "  [$i] missing $CFG"; continue; }
  echo "---- [$i] $(date +%H:%M:%S) collect ----"
  bash /home/kiwoos/collect_one.sh "$CFG" "${PREFIX}_${i}" >/dev/null 2>&1
  BAG=$(ls -dt /home/kiwoos/data/demos/${PREFIX}_${i}_* 2>/dev/null | head -1)
  SCORE=$(tail -1 /home/kiwoos/data/logs/one_log.txt | grep -oP 'score=\K[0-9.]+')
  [ -z "$SCORE" ] && SCORE=0
  good=$(awk -v s="$SCORE" -v t="$THRESH" 'BEGIN{print (s+0>=t)?1:0}')
  if [ -n "$BAG" ] && [ "$good" = "1" ]; then
    python3 /home/kiwoos/training/prepare_dataset.py "$BAG" "$OUT/ep_${EP}" >/tmp/cc_conv.log 2>&1
    ins=$(grep -oP 'insertion_events: \K[0-9]+' /tmp/cc_conv.log | head -1)
    if [ -f "$OUT/ep_${EP}/tcp_velocities.npy" ] && [ "${ins:-0}" -ge 1 ]; then
      nf=$(grep -oP 'Synchronized frames: \K[0-9]+' /tmp/cc_conv.log | head -1)
      echo "  [$i] KEEP ep_${EP} score=$SCORE frames=$nf"
      EP=$((EP+1)); keep=$((keep+1))
    else
      echo "  [$i] DROP (convert/ins fail) score=$SCORE"; rm -rf "$OUT/ep_${EP}"; fail=$((fail+1))
    fi
  else
    echo "  [$i] DROP (score=$SCORE < $THRESH)"; fail=$((fail+1))
  fi
  [ -n "$BAG" ] && rm -rf "$BAG"      # always delete the bag
done
echo "=== DONE $(date +%H:%M:%S): kept=$keep fail=$fail | episodes in $OUT ==="
du -sh "$OUT"/ep_* 2>/dev/null | tail -5
df -h ~ | tail -1
echo "CCDONE"
