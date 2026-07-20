#!/bin/bash
# Privileged-DAgger collection CAMPAIGN — deploy-policy stall states, ground-truth
# port TF, offline relabel.
#
# For each config in a manifest this does a full clean bringup with
# ground_truth:=true, records a bag (same topics as collect_one.sh PLUS the
# /scoring/tf* port transforms), runs the DEPLOY policy (DeployACT, NOT CheatCode)
# under the deploy venv with the adopted deploy env (AIC_CKPT/AIC_GUARDED[/AUX]),
# then relabels the bag OFFLINE with dagger_relabel.py: it reads the true port
# entrance TF from the bag, selects the policy's stall window, and emits a training
# episode (prepare_dataset layout + port_target.npy) for retraining the aux head.
# The raw bag is deleted immediately after conversion (CLAUDE §6). NO score filter
# — we WANT the non-seating stalls.
#
# Detached / resumable / DONE-markered (CLAUDE §6 agent-waiter ban): launch with
# nohup, verify the first unit converts, then report and exit; the watchdog tails
# the log. Already-converted episodes (OUT/<ep>/port_target.npy present) are
# skipped, so a re-run resumes.
#
# Env overrides:
#   MANIFEST     strata manifest.csv        (default ~/data/configs_phase0/manifest.csv)
#   OUT          episode output root        (default ~/training/ds_dagger)
#   CKPT         deploy checkpoint -> AIC_CKPT (default ~/training/ckpt/v2_wide.pt)
#   AIC_GUARDED  guarded-descent gate       (default 1)
#   AIC_GUARDED_AUX  learned-bearing gate   (default '' = off; set 1 with an aux ckpt)
#   AIC_* (any)  extra deploy knobs are passed through to the model node verbatim
#   POLICY_PY    deploy venv launcher       (default the aic-deploy venv aic_model)
#   PLUG         plug filter ''|sfp|sc      (default '' = all)
#   LIMIT        process at most N tasks    (default 0 = all)
#   TIMEOUT      per-trial wall-clock secs  (default 1500 = 25 min)
#   KEEP_BAG     non-empty = keep raw bags  (default '' = delete, storage-light)
#
# Usage:
#   nohup bash collect_dagger.sh >> ~/training/ds_dagger/dagger.log 2>&1 &
#   MANIFEST=/path/verify.csv LIMIT=1 bash collect_dagger.sh   # first-unit verify
set -u

REPO=/home/kiwoos/work/Intrinsics_Assembly_Robotics
LIB=$REPO/campaign_lib.py
RELABEL=$REPO/dagger_relabel.py
PY=python3

MANIFEST=${MANIFEST:-$HOME/data/configs_phase0/manifest.csv}
OUT=${OUT:-$HOME/training/ds_dagger}
CKPT=${CKPT:-$HOME/training/ckpt/v2_wide.pt}
PLUG=${PLUG:-}
LIMIT=${LIMIT:-0}
TIMEOUT=${TIMEOUT:-1500}
KEEP_BAG=${KEEP_BAG:-}

# Deploy env (mirrors eval_batch.sh precedent). DeployACT is a torch policy, so it
# MUST run under the deploy venv interpreter; plain `ros2 run aic_model aic_model`
# uses /usr/bin/python3 (no torch) and the node crashes on import -> 0 score.
export AIC_CKPT=$CKPT
export AIC_GUARDED=${AIC_GUARDED:-1}
# AIC_GUARDED_AUX is only meaningful with an aux-head checkpoint; left unset = off.
[ -n "${AIC_GUARDED_AUX:-}" ] && export AIC_GUARDED_AUX
POLICY_PY=${POLICY_PY:-/home/kiwoos/venvs/aic-deploy/bin/python}
POLICY_LAUNCHER=${POLICY_LAUNCHER:-/home/kiwoos/ws_aic/install/lib/aic_model/aic_model}

LOG=$OUT/dagger_log.csv
DEMO_LOG_DIR=$OUT/demo_logs
DEMO_DIR=$HOME/data/dagger_bags
mkdir -p "$OUT" "$DEMO_LOG_DIR" "$DEMO_DIR"
[ -f "$LOG" ] || echo "timestamp,config,stratum,plug,rep,episode_dir,frames,stalled,max_offset_cm,wall_clock_s,status" > "$LOG"

log_row() {  # ts cfg stratum plug rep epdir frames stalled maxoff wall status
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
    echo "[dagger]   WARN straggler bag recorder -> SIGINT + finalize"
    pkill -INT -f "ros2 bag record" 2>/dev/null
    sleep 6
  fi
  cleanup_sim
}

ensure_vnc() {
  if ! ls /tmp/.X11-unix/X2 >/dev/null 2>&1; then
    echo "[dagger] VNC :2 missing -> starting"
    vncserver :2 -geometry 1920x1080 -depth 24 -localhost no >/tmp/vnc_dagger.log 2>&1 || true
    sleep 4
  fi
}

# One deploy rollout: bringup (ground_truth) + record bag + run DeployACT, wait for
# the engine completion line (or the caller's timeout), then stop the bag cleanly.
# Echoes "BAG=<path>" on stdout for the caller. Mirrors collect_one.sh, but runs
# the DEPLOY policy under the venv and records /scoring/tf* for the port TF.
run_deploy_trial() {  # cfg tag rlog
  local CONFIG=$1 TAG=$2 RLOG=$3
  local BPATH=$DEMO_DIR/${TAG}

  source /opt/ros/kilted/setup.bash
  source /home/kiwoos/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export DISPLAY=:2

  cleanup_sim
  ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
  sleep 8

  # Per-trial scoring dir so we can detect completion by the scoring.yaml the
  # engine always writes at trial end (the deploy policy does NOT emit CheatCode's
  # "All Tasks Completed" line, so a marker-only wait hangs until timeout).
  local SCORE_DIR="$BPATH.scoring"
  mkdir -p "$SCORE_DIR"
  export AIC_RESULTS_DIR="$SCORE_DIR"

  echo "[dagger] launching sim (ground_truth) cfg=$CONFIG"
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    aic_engine_config_file:="$CONFIG" \
    ground_truth:=true start_aic_engine:=true launch_rviz:=false \
    > "$RLOG" 2>&1 &

  local READY=0 i
  for i in $(seq 1 45); do
    sleep 2
    if grep -qE "No node with name|Starting trial 'trial_1'" "$RLOG" 2>/dev/null; then
      READY=1; echo "[dagger] engine ready after $((i*2))s"; break
    fi
  done
  [ $READY -eq 0 ] && echo "[dagger] ERROR: engine never became ready" && return 1

  echo "[dagger] bag record -> $BPATH (with /scoring/tf*)"
  ros2 bag record \
    /left_camera/image /center_camera/image /right_camera/image \
    /left_camera/camera_info /center_camera/camera_info /right_camera/camera_info \
    /aic_controller/controller_state /aic_controller/pose_commands \
    /joint_states /fts_broadcaster/wrench \
    /scoring/tf /scoring/tf_static /scoring/insertion_event \
    /tf /tf_static \
    -o "$BPATH" > /tmp/bag_dagger.log 2>&1 &
  sleep 3

  echo "[dagger] starting DEPLOY policy (DeployACT) AIC_CKPT=$AIC_CKPT AIC_GUARDED=$AIC_GUARDED AIC_GUARDED_AUX=${AIC_GUARDED_AUX:-0}"
  PYTHONUNBUFFERED=1 "$POLICY_PY" -u "$POLICY_LAUNCHER" \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DeployACT >> "$RLOG" 2>&1 &

  local DONE=0
  for i in $(seq 1 72); do
    sleep 10
    # Primary: the engine wrote scoring.yaml (fires for seat OR non-seat, deploy
    # or oracle). Secondary: the legacy CheatCode completion markers.
    if [ -f "$SCORE_DIR/scoring.yaml" ] || \
       grep -qE "All Tasks Completed for trial 'trial_1'|completed successfully! Score:|Finished scoring trial|Engine Stopped" "$RLOG" 2>/dev/null; then
      DONE=1; echo "[dagger] trial complete after ~$((i*10))s"; break
    fi
  done
  [ $DONE -eq 0 ] && echo "[dagger] WARNING: completion not detected (timeout window)"

  sleep 4
  local BAGPID
  BAGPID=$(ps aux | grep "ros2 bag record" | grep -v grep | awk '{print $2}')
  [ -n "$BAGPID" ] && kill -INT $BAGPID 2>/dev/null
  sleep 5
  echo "BAG=$BPATH"
  return 0
}

echo "=== collect_dagger START $(date '+%F %T') ==="
echo "    manifest=$MANIFEST out=$OUT ckpt=$CKPT plug='${PLUG:-all}' limit=$LIMIT timeout=${TIMEOUT}s"
echo "    deploy: AIC_GUARDED=$AIC_GUARDED AIC_GUARDED_AUX=${AIC_GUARDED_AUX:-0} venv=$POLICY_PY"
ensure_vnc
cleanup_sim

PLUG_ARG=()
[ -n "$PLUG" ] && PLUG_ARG=(--plug "$PLUG")
mapfile -t TASKS < <("$PY" "$LIB" "$MANIFEST" "${PLUG_ARG[@]}")
TOTAL=${#TASKS[@]}
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL" ] && TOTAL_RUN=$LIMIT || TOTAL_RUN=$TOTAL
echo "[dagger] $TOTAL tasks in manifest; will process $TOTAL_RUN"

keep=0; fail=0; skip=0; i=0
for line in "${TASKS[@]}"; do
  i=$((i+1))
  [ "$LIMIT" -gt 0 ] && [ "$i" -gt "$LIMIT" ] && break
  IFS=$'\t' read -r CFG STRATUM TPLUG REP EPDIR <<< "$line"
  EP="$OUT/$EPDIR"
  TS=$(date '+%Y%m%d_%H%M%S')

  # --- resumable: skip fully-relabeled episodes (need the port target present) ---
  if [ -f "$EP/port_target.npy" ] && [ -f "$EP/tcp_poses.npy" ]; then
    echo "[dagger] TRIAL $i/$TOTAL_RUN $STRATUM -> SKIP_EXISTS ($EPDIR)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "SKIP_EXISTS"
    skip=$((skip+1)); continue
  fi
  if [ ! -f "$CFG" ]; then
    echo "[dagger] TRIAL $i/$TOTAL_RUN $STRATUM -> FAIL_NOCFG ($CFG)"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "0" "FAIL_NOCFG"
    fail=$((fail+1)); continue
  fi

  echo "---- [dagger] TRIAL $i/$TOTAL_RUN $(date '+%T') stratum=$STRATUM rep=$REP plug=$TPLUG ----"
  DLOG=$DEMO_LOG_DIR/${EPDIR}_${TS}.log
  start=$(date +%s)

  # Run one trial under a wall-clock timeout in a fresh shell. All AIC_* deploy
  # knobs are already `export`ed above, so they propagate to the model node; the
  # non-exported launcher/dir vars are exported explicitly here (the `VAR=val fn`
  # prefix form does NOT export to grandchildren for a shell function).
  timeout --signal=INT --kill-after=30 "$TIMEOUT" \
      bash -c "$(declare -f cleanup_sim run_deploy_trial); \
               export POLICY_PY='$POLICY_PY' POLICY_LAUNCHER='$POLICY_LAUNCHER' DEMO_DIR='$DEMO_DIR'; \
               run_deploy_trial '$CFG' '${EPDIR}_${TS}' '$DLOG'" > "${DLOG}.trial" 2>&1
  rc=$?
  end=$(date +%s); wall=$((end-start))
  stop_stragglers

  BAG=$(grep -oP 'BAG=\K\S+' "${DLOG}.trial" | tail -1)
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "[dagger]   FAIL_TIMEOUT after ${wall}s (rc=$rc)"
    [ -n "${BAG:-}" ] && rm -rf "$BAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_TIMEOUT"
    fail=$((fail+1)); continue
  fi
  if [ -z "${BAG:-}" ] || [ ! -d "$BAG" ]; then
    echo "[dagger]   FAIL_NOBAG"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_NOBAG"
    fail=$((fail+1)); continue
  fi

  # --- OFFLINE relabel: true port TF + stall window -> training episode. No score
  # filter: we keep every stall (the whole point is the non-seating distribution).
  # The entrance frame is auto-detected (target-scoped via /scoring/tf); pass
  # PORT_FRAME=... to force a specific frame if a bag has distractor ambiguity.
  # The board hosts multiple NIC mounts, each with an sfp_port_0/1 entrance frame,
  # so port_name alone is ambiguous. Build the exact entrance frame from the
  # config's target_module_name + port_name (frame = task_board/{module}/{port}
  # _link_entrance) and pass it as --port-frame (overrides detection, no ambiguity).
  # PORT_FRAME env still overrides if the caller sets it.
  PORT_NAME=$(grep -m1 -E '^[[:space:]]*port_name:' "$CFG" | awk '{print $2}')
  MODULE_NAME=$(grep -m1 -E '^[[:space:]]*target_module_name:' "$CFG" | awk '{print $2}')
  AUTO_FRAME=""
  [ -n "$PORT_NAME" ] && [ -n "$MODULE_NAME" ] && \
    AUTO_FRAME="task_board/${MODULE_NAME}/${PORT_NAME}_link_entrance"
  USE_FRAME=${PORT_FRAME:-$AUTO_FRAME}
  "$PY" "$RELABEL" "$BAG" "$EP" \
      ${USE_FRAME:+--port-frame "$USE_FRAME"} \
      --campaign-log "$OUT/campaign_log.csv" \
      --config "$CFG" --stratum "$STRATUM" --plug "$TPLUG" --rep "$REP" \
      > "${DLOG}.relabel" 2>&1
  relabel_rc=$?
  FRAMES=$(grep -oP 'Done: \K[0-9]+' "${DLOG}.relabel" | head -1)
  STALLED=$(grep -oP 'stalled=\K(True|False)' "${DLOG}.relabel" | head -1)
  MAXOFF=$(grep -oP 'max\|off\|=\K[0-9.]+' "${DLOG}.relabel" | head -1)

  if [ "$relabel_rc" -eq 0 ] && [ -f "$EP/port_target.npy" ]; then
    echo "[dagger]   KEEP $EPDIR frames=${FRAMES:-?} stalled=${STALLED:-?} maxoff=${MAXOFF:-?}cm wall=${wall}s"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "${FRAMES:-}" "${STALLED:-}" "${MAXOFF:-}" "$wall" "KEEP"
    keep=$((keep+1))
  else
    echo "[dagger]   FAIL_RELABEL $EPDIR (rc=$relabel_rc); see ${DLOG}.relabel"
    rm -rf "$EP"
    log_row "$TS" "$CFG" "$STRATUM" "$TPLUG" "$REP" "$EPDIR" "" "" "" "$wall" "FAIL_RELABEL"
    fail=$((fail+1))
  fi

  if [ -z "$KEEP_BAG" ]; then
    rm -rf "$BAG"   # storage-light: delete the raw bag after conversion (CLAUDE §6)
  else
    echo "[dagger]   KEEP_BAG set -> retaining $BAG"
  fi
done

echo "=== collect_dagger DONE $(date '+%F %T'): keep=$keep fail=$fail skip=$skip ==="
du -sh "$OUT"/ep_* 2>/dev/null | tail -3
df -h ~ | tail -1
echo "DAGGERDONE"
