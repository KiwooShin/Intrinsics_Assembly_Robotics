#!/usr/bin/env bash
# eval_curriculum.sh — RUN #4 curriculum milestone eval (M1+: staged start -> learned insertion).
#
# Runs CurriculumInsert (privileged staging above the aligned port, then the CURR_CKPT
# specialist) on a config for REPS trials, records per-trial insertion_events + engine
# score from scoring.yaml. Detached/resumable: a rep with results/<OUT>/trials/<unit>/DONE
# is skipped; ends with CURRDONE.
#
# Env:
#   CFG      config yaml            (default eval_suite/configs/official_2.yaml)
#   NAME     config id for dirs     (default official_2)
#   REPS     reps                   (default 3)
#   CKPT     specialist checkpoint  (default ~/training/ckpt/insert_m1_wrench_k4.pt)
#   OUT      output root            (default results/curr_m1)
#   LAT_MM / LAT_AZ_DEG / STANDOFF_M / BUDGET_S  -> CURR_* knobs
#   TIMEOUT  per-trial cap seconds  (default 900)
set -u
CFG="${CFG:-$PWD/eval_suite/configs/official_2.yaml}"
NAME="${NAME:-official_2}"
REPS="${REPS:-3}"
CKPT="${CKPT:-$HOME/training/ckpt/insert_m1_wrench_k4.pt}"
OUT="${OUT:-results/curr_m1}"
TIMEOUT="${TIMEOUT:-1300}"
POLICY_PY="$HOME/venvs/aic-deploy/bin/python"
mkdir -p "$OUT/trials"

cleanup_sim() {
  local pids
  pids=$(ps aux | grep -E "gz sim|aic_model|aic_engine|component_container|rmw_zenohd" \
         | grep -v grep | awk '{print $2}')
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
}

run_trial() {  # unit trial_dir
  local UNIT=$1 TDIR=$2
  source /opt/ros/kilted/setup.bash
  source /home/kiwoos/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export DISPLAY=:2
  export CURR_CKPT="$CKPT"
  export CURR_LAT_OFFSET_MM="${LAT_MM:-0}"
  export CURR_LAT_AZIMUTH_DEG="${LAT_AZ_DEG:-0}"
  export CURR_STANDOFF_M="${STANDOFF_M:-0.02}"
  export CURR_BUDGET_S="${BUDGET_S:-30}"

  cleanup_sim
  ros2 run rmw_zenoh_cpp rmw_zenohd > /dev/null 2>&1 &
  sleep 8
  echo "[curr] launching sim (ground_truth for staging TF) cfg=$CFG"
  # AIC_RESULTS_DIR so scoring.yaml lands in the trial dir (collect_dagger pattern).
  AIC_RESULTS_DIR="$TDIR" ros2 launch aic_bringup aic_gz_bringup.launch.py \
    aic_engine_config_file:="$CFG" \
    ground_truth:=true start_aic_engine:=true launch_rviz:=false \
    > "$TDIR/run.log" 2>&1 &
  local READY=0 i
  for i in $(seq 1 45); do
    sleep 2
    grep -qE "No node with name|Starting trial 'trial_1'" "$TDIR/run.log" 2>/dev/null \
      && { READY=1; echo "[curr] engine ready after $((i*2))s"; break; }
  done
  [ $READY -eq 0 ] && echo "[curr] ERROR: engine never ready" && return 1

  echo "[curr] starting CurriculumInsert (venv python, ckpt=$(basename "$CKPT"))"
  "$POLICY_PY" -u "$HOME/ws_aic/install/lib/aic_model/aic_model" --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CurriculumInsert >> "$TDIR/run.log" 2>&1 &

  local DONE=0
  for i in $(seq 1 90); do
    sleep 10
    if [ -f "$TDIR/scoring.yaml" ] || \
       grep -qE "Finished scoring trial|Engine Stopped|completed successfully" "$TDIR/run.log" 2>/dev/null; then
      DONE=1; echo "[curr] trial complete after ~$((i*10))s"; break
    fi
  done
  [ $DONE -eq 0 ] && echo "[curr] WARNING: completion not detected"
  sleep 5
  return 0
}

echo "=== eval_curriculum START $(date '+%F %T') cfg=$NAME reps=$REPS ckpt=$(basename "$CKPT") lat=${LAT_MM:-0}mm out=$OUT ==="
for r in $(seq 1 "$REPS"); do
  UNIT="${NAME}_r${r}"
  TDIR="$OUT/trials/$UNIT"
  if [ -f "$TDIR/DONE" ]; then echo "[curr] SKIP_DONE $UNIT"; continue; fi
  mkdir -p "$TDIR"
  echo "---- [curr] TRIAL $UNIT $(date '+%T') ----"
  timeout --signal=INT --kill-after=30 "$TIMEOUT" \
    bash -c "$(declare -f cleanup_sim run_trial); CFG='$CFG' CKPT='$CKPT' \
             LAT_MM='${LAT_MM:-0}' LAT_AZ_DEG='${LAT_AZ_DEG:-0}' \
             STANDOFF_M='${STANDOFF_M:-0.02}' BUDGET_S='${BUDGET_S:-45}' \
             POLICY_PY='$POLICY_PY' run_trial '$UNIT' '$TDIR'"
  cleanup_sim
  if [ -f "$TDIR/scoring.yaml" ]; then
    INS=$(grep -cE 'insertion' "$TDIR/scoring.yaml" 2>/dev/null || true)
    SCORE=$(grep -m1 -oE 'score: [0-9.-]+' "$TDIR/scoring.yaml" 2>/dev/null || true)
    echo "[curr]   $UNIT scored ($SCORE, insertion-lines=$INS)"
    : > "$TDIR/DONE"
  else
    echo "[curr]   $UNIT NOSCORE (no scoring.yaml — will retry on resume)"
  fi
done
echo "=== eval_curriculum DONE $(date '+%F %T') ==="
echo "CURRDONE"
