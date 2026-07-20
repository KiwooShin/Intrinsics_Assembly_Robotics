#!/bin/bash
# Assembly-by-disassembly demo collection CAMPAIGN (InsertionNet-backward).
#
# For each seated config in a manifest this does a full clean bringup with
# ground_truth:=true, records a bag (collect_one.sh topic set), and runs the
# scripted DisassembleCode oracle. DisassembleCode drives the gripper-welded plug
# down to NEAR seat depth and immediately begins a slow, laterally-perturbed
# RETRACT -- never dwelling long enough to fire the port's 1 s continuous-contact
# touch latch, so the seat stays reversible (no weld). The recorded perturbed
# retract is then converted to an episode (prepare_dataset.py) and TIME-REVERSED
# (reverse_disasm.py) into an insertion demo whose action label CONTAINS lateral
# correction (which the pure-vertical oracle last-inch lacks). The training episode
# lands in OUT/ep_<stratum>_r<rep>.
#
# DisassembleCode is SCRIPTED (no torch), so it runs under PLAIN ROS -- the same
# `ros2 run aic_model aic_model -p policy:=...` launcher CheatCode uses, NOT the
# deploy venv (unlike collect_dagger.sh's DeployACT).
#
# Detached / resumable / DONE-markered (CLAUDE §6 agent-waiter ban): launch with
# nohup, verify the first unit converts, then report and exit; the watchdog tails
# the log. Already-converted episodes (OUT/<ep>/tcp_velocities.npy present) are
# skipped, so a re-run resumes. The raw bag AND the intermediate un-reversed
# episode are deleted immediately after the reversal (storage-light).
#
# Env overrides:
#   MANIFEST     strata manifest.csv      (default ~/data/configs_phase0/manifest.csv)
#   OUT          episode output root      (default ~/training/ds_disasm)
#   PLUG         plug filter ''|sfp|sc    (default sfp -- SFP rails 0/1/2 x port_0/1)
#   LIMIT        process at most N tasks  (default 0 = all)
#   TIMEOUT      per-trial wall-clock s   (default 900 = 15 min)
#   BASE_SEED    base RNG seed; each trial i uses DISASM_SEED=BASE_SEED+i (default 0)
#   KEEP_BAG     non-empty = keep raw bags (default '' = delete)
#   DISASM_*     retract-schedule knobs passed through to DisassembleCode verbatim
#                (DISASM_RETRACT_START_Z, DISASM_AXIAL_SPAN_M, DISASM_AXIAL_STEP_M,
#                 DISASM_DT, DISASM_TURNS, DISASM_RADIUS_MIN_M, DISASM_RADIUS_MAX_M,
#                 DISASM_ROLL_PITCH_MIN_RAD, DISASM_ROLL_PITCH_MAX_RAD,
#                 DISASM_LIFT_FRAC, DISASM_LIFT_AXIAL_{MIN,MAX}_M,
#                 DISASM_LIFT_LATERAL_{MIN,MAX}_M)
#
# Usage:
#   nohup bash collect_disassembly.sh >> ~/training/ds_disasm/disasm.log 2>&1 &
#   MANIFEST=/path/verify.csv LIMIT=1 bash collect_disassembly.sh   # first-unit verify
set -u

REPO=/home/kiwoos/work/Intrinsics_Assembly_Robotics
LIB=$REPO/campaign_lib.py
PREP=$REPO/prepare_dataset.py
REVERSE=$REPO/reverse_disasm.py
PY=python3

MANIFEST=${MANIFEST:-$HOME/data/configs_phase0/manifest.csv}
OUT=${OUT:-$HOME/training/ds_disasm}
PLUG=${PLUG:-sfp}
LIMIT=${LIMIT:-0}
TIMEOUT=${TIMEOUT:-900}
BASE_SEED=${BASE_SEED:-0}
KEEP_BAG=${KEEP_BAG:-}

# Export every DISASM_* knob so it propagates to the (grandchild) model node.
for v in $(env | grep -oE '^DISASM_[A-Z_]+' || true); do export "$v"; done

LOG=$OUT/disasm_log.csv
DEMO_LOG_DIR=$OUT/demo_logs
DEMO_DIR=$HOME/data/disasm_bags
RAW_DIR=$HOME/data/disasm_raw
mkdir -p "$OUT" "$DEMO_LOG_DIR" "$DEMO_DIR" "$RAW_DIR"
[ -f "$LOG" ] || echo "timestamp,config,stratum,plug,rep,episode_dir,frames,insertion_events,latched,wall_clock_s,status" > "$LOG"

log_row() {  # ts cfg stratum plug rep epdir frames ins latched wall status
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11}" >> "$LOG"
}

cleanup_sim() {
  local pids
  pids=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container|rmw_zenohd|ros2 bag record" \
         | grep -v grep | awk '{print $2}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
}

stop_stragglers() {
  if pgrep -f "ros2 bag record" >/dev/null 2>&1; then
    echo "[disasm]   WARN straggler bag recorder -> SIGINT + finalize"
    pkill -INT -f "ros2 bag record" 2>/dev/null
    sleep 6
  fi
  cleanup_sim
}

ensure_vnc() {
  if ! ls /tmp/.X11-unix/X2 >/dev/null 2>&1; then
    echo "[disasm] VNC :2 missing -> starting"
    vncserver :2 -geometry 1920x1080 -depth 24 -localhost no >/tmp/vnc_disasm.log 2>&1 || true
    sleep 4
  fi
}

# One disassembly rollout: bringup (ground_truth) + record bag + run DisassembleCode
# (plain ROS), wait for the engine completion line (or the caller's timeout), then
# stop the bag cleanly. Echoes "BAG=<path>" on stdout for the caller.
run_disasm_trial() {  # cfg tag rlog seed
  local CONFIG=$1 TAG=$2 RLOG=$3 SEED=$4
  local BPATH=$DEMO_DIR/${TAG}

  source /opt/ros/kilted/setup.bash
  source /home/kiwoos/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export DISPLAY=:2
  export DISASM_SEED=$SEED

  cleanup_sim
  ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
  sleep 8

  echo "[disasm] launching sim (ground_truth) cfg=$CONFIG seed=$SEED"
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    aic_engine_config_file:="$CONFIG" \
    ground_truth:=true start_aic_engine:=true launch_rviz:=false \
    > "$RLOG" 2>&1 &

  local READY=0 i
  for i in $(seq 1 45); do
    sleep 2
    if grep -qE "No node with name|Starting trial 'trial_1'" "$RLOG" 2>/dev/null; then
      READY=1; echo "[disasm] engine ready after $((i*2))s"; break
    fi
  done
  [ $READY -eq 0 ] && echo "[disasm] ERROR: engine never became ready" && return 1

  # Disable each port touch-latch for COLLECTION ONLY (latch diagnosis 2026-07-20):
  # the upstream TouchPlugin welds the plug tip to the seat sensor plate after 1s of
  # contact, which fires during the slow perturbed retract. It is a SCORING sensor, not
  # needed for data-gen; the reversed episode's seat marker is geometric (reverse_disasm
  # sets insertion_frame=N-1), so labeling is unaffected. This is a runtime gz-transport
  # call that auto-reverts each run -- scoring worlds/assets are never edited. If the
  # disable fails for any reason, the insertion_events>=1 gate still rejects a weld.
  sleep 2
  local nsvc=0 svc
  for svc in $(gz service -l 2>/dev/null | grep -iE 'port.*/enable$'); do
    if gz service -s "$svc" --reqtype gz.msgs.Boolean --reptype gz.msgs.Empty \
         --timeout 2000 --req 'data: false' > /dev/null 2>&1; then
      echo "[disasm] touch-latch disabled: $svc"; nsvc=$((nsvc+1))
    fi
  done
  [ "$nsvc" -eq 0 ] && echo "[disasm] WARNING: no touch-latch enable service disabled (latch may still fire; gate will reject welds)"

  echo "[disasm] bag record -> $BPATH"
  ros2 bag record \
    /left_camera/image /center_camera/image /right_camera/image \
    /left_camera/camera_info /center_camera/camera_info /right_camera/camera_info \
    /aic_controller/controller_state /aic_controller/pose_commands \
    /joint_states /fts_broadcaster/wrench \
    /scoring/tf /scoring/insertion_event /tf /tf_static \
    -o "$BPATH" > /tmp/bag_disasm.log 2>&1 &
  sleep 3

  echo "[disasm] starting DisassembleCode (plain ROS, scripted teacher)"
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DisassembleCode >> "$RLOG" 2>&1 &

  local DONE=0
  for i in $(seq 1 72); do
    sleep 10
    if grep -qE "All Tasks Completed for trial 'trial_1'|completed successfully! Score:|Finished scoring trial|Engine Stopped" "$RLOG" 2>/dev/null; then
      DONE=1; echo "[disasm] trial complete after ~$((i*10))s"; break
    fi
  done
  [ $DONE -eq 0 ] && echo "[disasm] WARNING: completion not detected (timeout window)"

  sleep 4
  local BAGPID
  BAGPID=$(ps aux | grep "ros2 bag record" | grep -v grep | awk '{print $2}')
  [ -n "$BAGPID" ] && kill -INT $BAGPID 2>/dev/null
  sleep 5
  echo "BAG=$BPATH"
  return 0
}

echo "=== collect_disassembly START $(date '+%F %T') ==="
echo "    manifest=$MANIFEST out=$OUT plug='${PLUG:-all}' limit=$LIMIT timeout=${TIMEOUT}s base_seed=$BASE_SEED"
env | grep -E '^DISASM_' | sed 's/^/    knob /' || true
ensure_vnc
cleanup_sim

PLUG_ARG=()
[ -n "$PLUG" ] && PLUG_ARG=(--plug "$PLUG")
mapfile -t TASKS < <("$PY" "$LIB" "$MANIFEST" "${PLUG_ARG[@]}")
TOTAL=${#TASKS[@]}
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL" ] && TOTAL_RUN=$LIMIT || TOTAL_RUN=$TOTAL
echo "[disasm] $TOTAL tasks in manifest; will process $TOTAL_RUN"

keep=0; fail=0; skip=0; i=0
for line in "${TASKS[@]}"; do
  i=$((i+1))
  [ "$LIMIT" -gt 0 ] && [ "$i" -gt "$LIMIT" ] && break
  IFS=$'\t' read -r CFG STRATUM TPLUG REP EPDIR <<< "$line"
  EP="$OUT/$EPDIR"
  RAW="$RAW_DIR/$EPDIR"
  TS=$(date '+%Y%m%d_%H%M%S')
  SEED=$((BASE_SEED + i))

  # --- resumable: skip fully-reversed episodes ---
  if [ -f "$EP/tcp_velocities.npy" ] && [ -f "$EP/insertion_frame.npy" ]; then
    echo "[disasm] TRIAL $i/$TOTAL_RUN $STRATUM -> SKIP_EXISTS ($EPDIR)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "SKIP_EXISTS"
    skip=$((skip+1)); continue
  fi
  if [ ! -f "$CFG" ]; then
    echo "[disasm] TRIAL $i/$TOTAL_RUN $STRATUM -> FAIL_NOCFG ($CFG)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "FAIL_NOCFG"
    fail=$((fail+1)); continue
  fi

  echo "---- [disasm] TRIAL $i/$TOTAL_RUN $(date '+%T') stratum=$STRATUM rep=$REP plug=$TPLUG seed=$SEED ----"
  DLOG=$DEMO_LOG_DIR/${EPDIR}_${TS}.log
  start=$(date +%s)

  timeout --signal=INT --kill-after=30 "$TIMEOUT" \
      bash -c "$(declare -f cleanup_sim run_disasm_trial); \
               export DEMO_DIR='$DEMO_DIR'; \
               run_disasm_trial '$CFG' '${EPDIR}_${TS}' '$DLOG' '$SEED'" > "${DLOG}.trial" 2>&1
  rc=$?
  end=$(date +%s); wall=$((end-start))
  stop_stragglers

  BAG=$(grep -oP 'BAG=\K\S+' "${DLOG}.trial" | tail -1)
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "[disasm]   FAIL_TIMEOUT after ${wall}s (rc=$rc)"
    [ -n "${BAG:-}" ] && rm -rf "$BAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_TIMEOUT"
    fail=$((fail+1)); continue
  fi
  if [ -z "${BAG:-}" ] || [ ! -d "$BAG" ]; then
    echo "[disasm]   FAIL_NOBAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_NOBAG"
    fail=$((fail+1)); continue
  fi

  # --- convert the recorded retract -> raw episode, then TIME-REVERSE -> insertion.
  rm -rf "$RAW"
  "$PY" "$PREP" "$BAG" "$RAW" > "${DLOG}.convert" 2>&1
  conv_rc=$?
  INS=$(grep -oP 'insertion_events: \K[0-9]+' "${DLOG}.convert" | head -1)
  FRAMES=$(grep -oP 'Synchronized frames: \K[0-9]+' "${DLOG}.convert" | head -1)
  LATCHED=False
  [ "${INS:-0}" -ge 1 ] && LATCHED=True   # a seat fired -> the retract likely latched

  # HARD GATE (verify-disasm-correctness workflow, 2026-07-20): a plug that tripped the
  # irreversible 1s touch-latch DURING the ~21s descent must NOT be time-reversed into
  # the dataset as a fake seat -- the one silent-corruption path. Reject it outright
  # (do NOT reverse or keep). The TouchPlugin timer resets only on contact BREAK, so a
  # sustained-contact descent (chiefly SC at -0.013, 3mm below its -0.010 seat floor)
  # can weld mid-descent; SFP at -0.013 stays 2mm shallower than its -0.015 floor and is
  # geometry-safe, but the gate applies to both for defense in depth.
  if [ "$LATCHED" = "True" ]; then
    echo "[disasm]   FAIL_LATCHED $EPDIR insertion_events=${INS:-0} (welded mid-descent; rejected, not reversed)"
    rm -rf "$EP" "$RAW"
    [ -z "$KEEP_BAG" ] && rm -rf "$BAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "${FRAMES:-}" "${INS:-0}" "True" "$wall" "FAIL_LATCHED"
    fail=$((fail+1)); continue
  fi

  if [ "$conv_rc" -eq 0 ] && [ -f "$RAW/tcp_velocities.npy" ]; then
    "$PY" "$REVERSE" "$RAW" "$EP" > "${DLOG}.reverse" 2>&1
    rev_rc=$?
  else
    rev_rc=1
  fi

  if [ "${rev_rc:-1}" -eq 0 ] && [ -f "$EP/tcp_velocities.npy" ] && [ -f "$EP/insertion_frame.npy" ]; then
    STATUS=KEEP   # latched episodes are already rejected above -> this is a clean seat-free retract
    echo "[disasm]   $STATUS $EPDIR frames=${FRAMES:-?} insertion_events=${INS:-0} wall=${wall}s"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "${FRAMES:-}" "${INS:-0}" "$LATCHED" "$wall" "$STATUS"
    keep=$((keep+1))
  else
    echo "[disasm]   FAIL_CONVERT $EPDIR (conv_rc=$conv_rc rev_rc=${rev_rc:-1}); see ${DLOG}.convert/.reverse"
    rm -rf "$EP"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "${INS:-0}" "$LATCHED" "$wall" "FAIL_CONVERT"
    fail=$((fail+1))
  fi

  rm -rf "$RAW"                       # drop the un-reversed intermediate
  if [ -z "$KEEP_BAG" ]; then
    rm -rf "$BAG"                     # storage-light: delete the raw bag (CLAUDE §6)
  else
    echo "[disasm]   KEEP_BAG set -> retaining $BAG"
  fi
done

echo "=== collect_disassembly DONE $(date '+%F %T'): keep=$keep fail=$fail skip=$skip ==="
du -sh "$OUT"/ep_* 2>/dev/null | tail -3
df -h ~ | tail -1
echo "DISASMDONE"
