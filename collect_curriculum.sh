#!/usr/bin/env bash
# collect_curriculum.sh — M3 offset-staged ORACLE demo collection (RUN #4).
#
# Per episode: stage the plug CURR_LAT_OFFSET_MM off the port at CURR_STANDOFF_M,
# then the CheatCode-style oracle ladder descends to the TRUE port and seats — the
# recorded episode therefore CONTAINS the lateral-correction motion the base corpus
# lacks. Forward-ordered demos ending at the natural seat weld (no reversal, no
# latch gymnastics). Bags convert via prepare_dataset then delete (CLAUDE.md §6).
#
# Offsets: a fixed ladder of (offset_mm, azimuth_deg) pairs covering 0.5-4mm across
# 8 bearings, EPS episodes total (default 24). Resumable via ep-dir presence; ends
# with CURRCOLLECTDONE.
#
# Env: CFG (default official_2 yaml), EPS (default 24), OUT (default ~/training/ds_curr),
#      TIMEOUT per-episode (default 1300), KEEP_BAG=1 to retain bags.
set -u
CFG="${CFG:-$PWD/eval_suite/configs/official_2.yaml}"
EPS="${EPS:-24}"
OUT="${OUT:-$HOME/training/ds_curr}"
TIMEOUT="${TIMEOUT:-1300}"
PY=$HOME/miniconda3/bin/python
PREP=$PWD/prepare_dataset.py
POLICY_PY="$HOME/venvs/aic-deploy/bin/python"
BAGDIR=$HOME/data/curr_bags
mkdir -p "$OUT" "$BAGDIR"

cleanup_sim() {
  local pids
  # Match every per-trial ROS launch child (robot_state_publisher + topic_tools/relay
  # leaked ~0.5GB each across trials and OOM'd training — widen the net). Exclude the
  # bash -c wrapper whose cmdline embeds this function text (declare -f), or we self-kill.
  pids=$(ps aux | grep -E "gz sim|aic_model|aic_engine|aic_adapter|static_transform_publisher|component_container|rmw_zenohd|robot_state_publisher|topic_tools/relay|ros_gz_bridge|parameter_bridge|ros2 bag record" \
         | grep -vE "grep|declare -f|cleanup_sim|run_trial|run_ep" | awk '{print $2}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
}

run_ep() {  # lat_mm az_deg bag_path log_path
  local LAT=$1 AZ=$2 BAG=$3 LOG=$4
  source /opt/ros/kilted/setup.bash
  source /home/kiwoos/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export DISPLAY=:2
  export CURR_MODE=oracle
  export CURR_LAT_OFFSET_MM="$LAT"
  export CURR_LAT_AZIMUTH_DEG="$AZ"
  export CURR_STANDOFF_M="${STANDOFF_M:-0.02}"
  export CURR_CORRECT_AT_MM="${CORRECT_AT_MM:-0}"
  export CURR_CKPT=unused-in-oracle-mode

  cleanup_sim
  ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
  sleep 8
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    aic_engine_config_file:="$CFG" \
    ground_truth:=true start_aic_engine:=true launch_rviz:=false > "$LOG" 2>&1 &
  local READY=0 i
  for i in $(seq 1 45); do
    sleep 2
    grep -qE "No node with name|Starting trial 'trial_1'" "$LOG" 2>/dev/null \
      && { READY=1; break; }
  done
  [ $READY -eq 0 ] && echo "[collect] ERROR: engine never ready" && return 1

  ros2 bag record \
    /left_camera/image /center_camera/image /right_camera/image \
    /left_camera/camera_info /center_camera/camera_info /right_camera/camera_info \
    /aic_controller/controller_state /aic_controller/pose_commands \
    /joint_states /fts_broadcaster/wrench \
    /scoring/tf /scoring/insertion_event /tf /tf_static \
    -o "$BAG" > /tmp/bag_curr.log 2>&1 &
  sleep 3

  "$POLICY_PY" -u "$HOME/ws_aic/install/lib/aic_model/aic_model" --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CurriculumInsert >> "$LOG" 2>&1 &

  local DONE=0
  for i in $(seq 1 90); do
    sleep 10
    grep -qE "CurriculumInsert\(oracle\): dwell at seat|insert_cable\(\) returned" "$LOG" 2>/dev/null \
      && { DONE=1; break; }
  done
  sleep 8   # let the dwell + weld land in the bag
  local BAGPID
  BAGPID=$(ps aux | grep "ros2 bag record" | grep -v grep | awk '{print $2}')
  [ -n "$BAGPID" ] && kill -INT $BAGPID 2>/dev/null
  sleep 5
  [ $DONE -eq 0 ] && echo "[collect] WARNING: completion not detected"
  return 0
}

# Fixed offset ladder: 8 azimuths x offsets cycling 0.5/1/2/3/4mm -> EPS episodes.
OFFSETS=(0.5 1 2 3 4)
echo "=== collect_curriculum START $(date '+%F %T') eps=$EPS out=$OUT ==="
keep=0; fail=0; i=0
while [ $i -lt "$EPS" ]; do
  LAT=${OFFSETS[$((i % 5))]}
  AZ=$(( (i * 45) % 360 ))
  EPDIR="$OUT/ep_curr_lat${LAT}_az${AZ}_$((i / 5))"
  i=$((i+1))
  if [ -f "$EPDIR/tcp_velocities.npy" ]; then
    echo "[collect] EP $i/$EPS SKIP_EXISTS $(basename "$EPDIR")"; continue
  fi
  TS=$(date '+%Y%m%d_%H%M%S')
  BAG=$BAGDIR/curr_${TS}
  LOG=$OUT/$(basename "$EPDIR").log
  echo "---- [collect] EP $i/$EPS lat=${LAT}mm az=${AZ} $(date '+%T') ----"
  timeout --signal=INT --kill-after=30 "$TIMEOUT" \
    bash -c "$(declare -f cleanup_sim run_ep); STANDOFF_M='${STANDOFF_M:-0.02}' CORRECT_AT_MM='${CORRECT_AT_MM:-0}' \
             CFG='$CFG' POLICY_PY='$POLICY_PY' run_ep '$LAT' '$AZ' '$BAG' '$LOG'"
  cleanup_sim
  if [ -d "$BAG" ]; then
    "$PY" "$PREP" "$BAG" "$EPDIR" > "$LOG.convert" 2>&1
    if [ -f "$EPDIR/tcp_velocities.npy" ]; then
      INS=$(grep -oP 'insertion_events: \K[0-9]+' "$LOG.convert" | head -1)
      FRAMES=$(grep -oP 'Synchronized frames: \K[0-9]+' "$LOG.convert" | head -1)
      echo "[collect]   KEEP $(basename "$EPDIR") frames=${FRAMES:-?} insertion_events=${INS:-0}"
      keep=$((keep+1))
    else
      echo "[collect]   FAIL_CONVERT $(basename "$EPDIR")"; rm -rf "$EPDIR"; fail=$((fail+1))
    fi
    [ -z "${KEEP_BAG:-}" ] && rm -rf "$BAG"
  else
    echo "[collect]   FAIL_NOBAG"; fail=$((fail+1))
  fi
done
echo "=== collect_curriculum DONE $(date '+%F %T'): keep=$keep fail=$fail ==="
df -h ~ | tail -1
echo "CURRCOLLECTDONE"
