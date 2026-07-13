#!/bin/bash
# Phase-0 stratified demo-collection CAMPAIGN.
#
# Loops over a strata manifest (gen_config.py --mode strata) and, per demo, does a
# full clean bringup, records a CheatCode (ground_truth) demo, score-filters, converts
# the bag to a trimmed episode (wrench + joints included), deletes the bag immediately,
# and appends a row to campaign_log.csv. Robust: per-demo wall-clock timeout, failures
# are logged + skipped (never wedges the loop), and it is resumable (already-converted
# episodes are skipped). Follows the proven collect_convert.sh pattern.
#
# Env overrides:
#   MANIFEST  strata manifest.csv           (default ~/data/configs_phase0/manifest.csv)
#   OUT       dataset output dir            (default ~/training/ds_phase0)
#   PLUG      plug filter: ''|sfp|sc        (default '' = all cells)
#   LIMIT     process at most N tasks, 0=all(default 0)
#   THRESH    keep score >= THRESH          (default 60)
#   TIMEOUT   per-demo wall-clock seconds   (default 1500 = 25 min)
#   KEEP_BAG  non-empty = do NOT delete bags(default '' = delete, storage-light)
#
# Usage: bash collect_campaign.sh                             # full campaign, all cells
#        MANIFEST=/path/verify.csv bash collect_campaign.sh   # verify run (2-row manifest)
#        PLUG=sfp bash collect_campaign.sh                    # SFP-only campaign
set -u

REPO=/home/kiwoos/work/Intrinsics_Assembly_Robotics
COLLECT_ONE=$REPO/collect_one.sh
PREP=$REPO/prepare_dataset.py
LIB=$REPO/campaign_lib.py

MANIFEST=${MANIFEST:-$HOME/data/configs_phase0/manifest.csv}
OUT=${OUT:-$HOME/training/ds_phase0}
PLUG=${PLUG:-}
LIMIT=${LIMIT:-0}
THRESH=${THRESH:-60}
TIMEOUT=${TIMEOUT:-1500}
KEEP_BAG=${KEEP_BAG:-}

LOG=$OUT/campaign_log.csv
DEMO_LOG_DIR=$OUT/demo_logs
PY=python3

mkdir -p "$OUT" "$DEMO_LOG_DIR"
[ -f "$LOG" ] || echo "timestamp,config,stratum,plug,rep,episode_dir,score,frames,insertion_events,wall_clock_s,status" > "$LOG"

log_row() {  # ts cfg stratum plug rep epdir score frames ins wall status
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11}" >> "$LOG"
}

cleanup_sim() {
  local pids
  pids=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container|rmw_zenohd|ros2 bag record" \
         | grep -v grep | awk '{print $2}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
}

# Recorder teardown — called unconditionally after EVERY collect_one.sh return, before
# conversion. collect_one.sh's own EXIT-trap teardown has been observed to leak the bag
# recorder (it then records the idle sim indefinitely, growing the bag until the disk
# fills). SIGINT first so the MCAP finalizes cleanly, then hard-kill the sim stack.
stop_stragglers() {
  if pgrep -f "ros2 bag record" >/dev/null 2>&1; then
    echo "[campaign]   WARN straggler bag recorder -> SIGINT + finalize"
    pkill -INT -f "ros2 bag record" 2>/dev/null
    sleep 6
  fi
  cleanup_sim
}

ensure_vnc() {
  if ! ls /tmp/.X11-unix/X2 >/dev/null 2>&1; then
    echo "[campaign] VNC :2 missing -> starting"
    vncserver :2 -geometry 1920x1080 -depth 24 -localhost no >/tmp/vnc_campaign.log 2>&1 || true
    sleep 4
  fi
}

echo "=== collect_campaign START $(date '+%F %T') ==="
echo "    manifest=$MANIFEST out=$OUT plug='${PLUG:-all}' limit=$LIMIT thresh=$THRESH timeout=${TIMEOUT}s"
ensure_vnc
cleanup_sim   # clear any orphan sim before the first clean bringup

# Build the work list via the tested pure-python helper (robust CSV parsing).
PLUG_ARG=()
[ -n "$PLUG" ] && PLUG_ARG=(--plug "$PLUG")
mapfile -t TASKS < <("$PY" "$LIB" "$MANIFEST" "${PLUG_ARG[@]}")
TOTAL=${#TASKS[@]}
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL" ] && TOTAL_RUN=$LIMIT || TOTAL_RUN=$TOTAL
echo "[campaign] $TOTAL tasks in manifest; will process $TOTAL_RUN"

keep=0; drop=0; fail=0; skip=0; i=0
for line in "${TASKS[@]}"; do
  i=$((i+1))
  [ "$LIMIT" -gt 0 ] && [ "$i" -gt "$LIMIT" ] && break
  IFS=$'\t' read -r CFG STRATUM TPLUG REP EPDIR <<< "$line"
  EP="$OUT/$EPDIR"
  TS=$(date '+%Y%m%d_%H%M%S')

  # --- resumable: skip fully-converted A0 episodes (need wrench+joints present) ---
  if [ -f "$EP/tcp_velocities.npy" ] && [ -f "$EP/wrenches.npy" ] && [ -f "$EP/joint_positions.npy" ]; then
    echo "[campaign] DEMO $i/$TOTAL_RUN $STRATUM -> SKIP_EXISTS ($EPDIR)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "SKIP_EXISTS"
    skip=$((skip+1)); continue
  fi

  if [ ! -f "$CFG" ]; then
    echo "[campaign] DEMO $i/$TOTAL_RUN $STRATUM -> FAIL_NOCFG ($CFG)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "FAIL_NOCFG"
    fail=$((fail+1)); continue
  fi

  echo "---- [campaign] DEMO $i/$TOTAL_RUN $(date '+%T') stratum=$STRATUM rep=$REP plug=$TPLUG ----"
  DLOG=$DEMO_LOG_DIR/${EPDIR}_${TS}.log
  start=$(date +%s)

  # Full clean bringup + record + CheatCode all live inside collect_one.sh (its trap
  # cleans the sim on exit). SIGINT first so that trap runs; hard SIGKILL 30s later.
  timeout --signal=INT --kill-after=30 "$TIMEOUT" \
      bash "$COLLECT_ONE" "$CFG" "$EPDIR" > "$DLOG" 2>&1
  rc=$?
  end=$(date +%s); wall=$((end-start))

  # Belt-and-braces: kill any leaked recorder/sim NOW, before conversion (see above).
  stop_stragglers

  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "[campaign]   FAIL_TIMEOUT after ${wall}s (rc=$rc)"
    BAG=$(grep -oP 'bag=\K\S+' "$DLOG" | tail -1)
    [ -n "${BAG:-}" ] && rm -rf "$BAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_TIMEOUT"
    fail=$((fail+1)); continue
  fi

  BAG=$(grep -oP 'bag=\K\S+' "$DLOG" | tail -1)
  SCORE=$(grep -oP 'RESULT score=\K[0-9.]+' "$DLOG" | tail -1)
  DONEFLAG=$(grep -oP 'RESULT score=\S+ done=\K[0-9]' "$DLOG" | tail -1)
  [ -z "${SCORE:-}" ] && SCORE=0

  if [ -z "${BAG:-}" ] || [ ! -d "$BAG" ]; then
    echo "[campaign]   FAIL_NOBAG (score=$SCORE done=${DONEFLAG:-?})"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "$SCORE" "" "" "$wall" "FAIL_NOBAG"
    fail=$((fail+1)); continue
  fi

  good=$(awk -v s="$SCORE" -v t="$THRESH" 'BEGIN{print (s+0>=t)?1:0}')
  if [ "$good" = "1" ]; then
    "$PY" "$PREP" "$BAG" "$EP" > "${DLOG}.conv" 2>&1
    FRAMES=$(grep -oP 'Synchronized frames: \K[0-9]+' "${DLOG}.conv" | head -1)
    INS=$(grep -oP 'insertion_events: \K[0-9]+' "${DLOG}.conv" | head -1)
    if [ -f "$EP/tcp_velocities.npy" ] && [ "${INS:-0}" -ge 1 ]; then
      echo "[campaign]   KEEP $EPDIR score=$SCORE frames=${FRAMES:-?} ins=${INS:-?} wall=${wall}s"
      log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "$SCORE" "${FRAMES:-}" "${INS:-}" "$wall" "KEEP"
      keep=$((keep+1))
    else
      echo "[campaign]   DROP_CONVERT $EPDIR score=$SCORE ins=${INS:-0}"
      rm -rf "$EP"
      log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "$SCORE" "${FRAMES:-}" "${INS:-0}" "$wall" "DROP_CONVERT"
      fail=$((fail+1))
    fi
  else
    echo "[campaign]   DROP_SCORE $EPDIR score=$SCORE < $THRESH (done=${DONEFLAG:-?})"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "$SCORE" "" "" "$wall" "DROP_SCORE"
    drop=$((drop+1))
  fi

  if [ -z "$KEEP_BAG" ]; then
    rm -rf "$BAG"   # always delete the raw bag (storage-light)
  else
    echo "[campaign]   KEEP_BAG set -> retaining $BAG"
  fi
done

echo "=== collect_campaign DONE $(date '+%F %T'): keep=$keep drop=$drop fail=$fail skip=$skip ==="
du -sh "$OUT"/ep_* 2>/dev/null | tail -3
df -h ~ | tail -1
echo "CAMPAIGNDONE"
