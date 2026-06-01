#!/bin/bash
# Collect N sample demos (SFP only, single trial per run)

N=${1:-5}
DEMO_DIR=~/data/demos
LOG_DIR=~/data/logs
mkdir -p $DEMO_DIR $LOG_DIR

source /opt/ros/kilted/setup.bash
source /home/kiwoos/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DISPLAY=:2
CONFIG=/home/kiwoos/ws_aic/install/share/aic_engine/config/eval_config.yaml

cleanup() {
  PIDS=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container" | grep -v grep | awk '{print $2}')
  [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null || true
  sleep 3
}

SUCCESS=0; RUNS=0

ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
ZENOH_PID=$!
sleep 8
echo "Zenoh started | Target: $N samples"

while [ $SUCCESS -lt $N ]; do
  RUNS=$((RUNS+1))
  TS=$(date +%Y%m%d_%H%M%S)
  RLOG=$LOG_DIR/sample_run_${RUNS}.log
  echo "--- Run $RUNS | Success $SUCCESS/$N | $(date +%H:%M:%S) ---"

  cleanup

  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    aic_engine_config_file:=$CONFIG \
    ground_truth:=true start_aic_engine:=true launch_rviz:=false \
    > $RLOG 2>&1 &

  READY=0
  for i in $(seq 1 25); do
    sleep 2
    if grep -q "No node with name" $RLOG 2>/dev/null; then
      READY=1; echo "  Engine ready"; break
    fi
  done
  [ $READY -eq 0 ] && echo "  Timeout" && cleanup && continue

  # Record bag with cameras + state
  BPATH="$DEMO_DIR/sample_${SUCCESS}_${TS}"
  ros2 bag record \
    /left_camera/image /center_camera/image /right_camera/image \
    /aic_controller/controller_state \
    /joint_states /fts_broadcaster/wrench \
    /scoring/tf \
    -o $BPATH > /tmp/bag.log 2>&1 &
  BAG_PID=$!
  sleep 2
  echo "  Bag: $BPATH"

  ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode \
    >> $RLOG 2>&1 &

  # Wait up to 10 min for scoring
  for i in $(seq 1 120); do
    sleep 10
    grep -q "Complete Scoring Results" $RLOG 2>/dev/null && break
  done

  kill $BAG_PID 2>/dev/null || true
  sleep 2

  INS=$(grep -c "Cable insertion successful" $RLOG 2>/dev/null || echo "0")
  SCORE=$(grep "total:" $RLOG 2>/dev/null | tail -1 | grep -oP 'total: K[0-9]+.[0-9]+' || echo "0")
  echo "  Score: $SCORE | Insertions: $INS"

  if [ "$INS" -ge "2" ] 2>/dev/null; then
    SUCCESS=$((SUCCESS+1))
    echo "  SUCCESS $SUCCESS — saved: $BPATH"
    echo "$TS score=$SCORE bag=$BPATH" >> $LOG_DIR/sample_log.txt
    SIZE=$(du -sh $BPATH | cut -f1)
    echo "  Bag size: $SIZE"
  else
    echo "  FAIL — removing bag"
    rm -rf $BPATH
  fi
  cleanup
done

kill $ZENOH_PID 2>/dev/null || true
echo "=== Done: $SUCCESS samples in $RUNS runs ==="
