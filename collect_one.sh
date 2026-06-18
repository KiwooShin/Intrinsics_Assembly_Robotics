#!/bin/bash
# Single-trial CheatCode demo (one SFP insertion). Correct success/score detection.
DEMO_DIR=~/data/demos
LOG_DIR=~/data/logs
mkdir -p "$DEMO_DIR" "$LOG_DIR"

source /opt/ros/kilted/setup.bash
source /home/kiwoos/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DISPLAY=:2
CONFIG=${1:-/home/kiwoos/data/single_trial.yaml}
TAG=${2:-one}

TS=$(date +%Y%m%d_%H%M%S)
RLOG=$LOG_DIR/${TAG}_${TS}.log
BPATH=$DEMO_DIR/${TAG}_${TS}

cleanup() {
  PIDS=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container|rmw_zenohd|ros2 bag" | grep -v grep | awk '{print $2}')
  [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null || true
  sleep 4
}
trap cleanup EXIT

cleanup
echo "[one] $(date +%H:%M:%S) starting zenoh"
ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
sleep 8

echo "[one] launching sim (single trial, ground_truth)"
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  aic_engine_config_file:=$CONFIG \
  ground_truth:=true start_aic_engine:=true launch_rviz:=false \
  > "$RLOG" 2>&1 &

# Engine ready = it begins polling for the (not-yet-started) model node
READY=0
for i in $(seq 1 45); do
  sleep 2
  if grep -qE "No node with name|Starting trial 'trial_1'" "$RLOG" 2>/dev/null; then
    READY=1; echo "[one] engine ready after $((i*2))s"; break
  fi
done
[ $READY -eq 0 ] && echo "[one] ERROR: engine never became ready" && exit 1

echo "[one] starting bag record -> $BPATH"
ros2 bag record \
  /left_camera/image /center_camera/image /right_camera/image \
  /left_camera/camera_info /center_camera/camera_info /right_camera/camera_info \
  /aic_controller/controller_state /aic_controller/pose_commands \
  /joint_states /fts_broadcaster/wrench \
  /scoring/tf /scoring/insertion_event /tf /tf_static \
  -o "$BPATH" > /tmp/bag_one.log 2>&1 &
sleep 3

echo "[one] starting CheatCode model"
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode >> "$RLOG" 2>&1 &

# Wait for the real engine completion line (up to 12 min)
DONE=0
for i in $(seq 1 72); do
  sleep 10
  if grep -qE "All Tasks Completed for trial 'trial_1'|completed successfully! Score:|Finished scoring trial" "$RLOG" 2>/dev/null; then
    DONE=1; echo "[one] trial complete after ~$((i*10))s"; break
  fi
done
[ $DONE -eq 0 ] && echo "[one] WARNING: completion not detected (timeout)"

# Let post-completion frames flush, then stop bag cleanly so MCAP finalizes
sleep 4
BAGPID=$(ps aux | grep "ros2 bag record" | grep -v grep | awk '{print $2}')
[ -n "$BAGPID" ] && kill -INT $BAGPID 2>/dev/null
sleep 5

SCORE=$(grep -oP "completed successfully! Score: \K[0-9.]+" "$RLOG" | tail -1)
[ -z "$SCORE" ] && SCORE=$(grep -oP "total score is: \K[0-9.]+" "$RLOG" | tail -1)
INS=$(grep -c "scoring/insertion_event" /tmp/bag_one.log 2>/dev/null || echo "?")
echo "[one] RESULT score=${SCORE:-NONE} done=$DONE bag=$BPATH"
echo "$TS score=${SCORE:-NONE} done=$DONE bag=$BPATH" >> "$LOG_DIR/one_log.txt"
echo "[one] FINISHED"
