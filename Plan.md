# 2-Week Sprint Plan — AI for Industry Challenge

**Goal:** Train a high-scoring policy for the Intrinsic AIC competition within 2 weeks.
**Competition:** $180K prize | UR5e cable insertion into NIC card ports | Evaluated in Gazebo
**Compute:** NVIDIA DGX-Spark (Grace Blackwell, ~H100-class GPU)
**Target score:** 250–280 / 300 (full insertion, <5s, no collisions)
**Audience:** Also a portfolio artifact for frontier AI robotics lab hiring managers

---

## Scoring Cheat Sheet

```
Per trial max: 100 pts × 3 trials = 300 pts total
```

| Component | Pts | Win condition |
|---|---|---|
| Model validity | 1 | Node loads, lifecycle works |
| **Task duration** | **12** | **Complete in < 5 seconds** |
| Trajectory smoothness | 6 | Low jerk (Savitzky-Golay filtered) |
| Path efficiency | 6 | Go directly to port |
| Force penalty | -12 | Never exceed 20N for >1s |
| Contact penalty | -24 | Zero collisions with enclosure/board |
| **Full correct insertion** | **75** | **Plug fully seated in right port** |
| Wrong port | -12 | Never — worse than not inserting |
| Partial insertion | 38–50 | Plug entered port bounding box |
| Proximity only | 0–25 | Plug near port but not inserted |

> **Key rule:** Tier 2 scores (duration, smoothness, efficiency) only count if Tier 3 score > 0 (plug near or in port). Motion quality is irrelevant if you never reach the port.

**Score milestones:**
- Proximity only: ~78/300
- Partial insertion: ~189/300
- Full insertion, slow: ~240/300
- Full insertion, <5s, smooth: ~297/300 ← target

---

## Environment Status (Completed)

- ✅ ROS 2 Kilted Kaiju built from source (native ARM64 on DGX-Spark)
- ✅ Gazebo running at **500Hz / 100% RTF**
- ✅ WaveArm policy running, robot visible in Gazebo/VNC
- ✅ Pixi (0.67.2), Distrobox, Docker installed
- ✅ OGRE2 rendering via apt symlink
- ✅ Global Illumination disabled (performance)
- ✅ Cameras disabled for WaveArm demo (re-enable for training)

**Standard restart command (run before every session):**
```bash
kill -9 $(ps aux | grep -E "rviz2|gz sim|zenohd|aic_model|aic_engine|component_container" | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 3

source ~/.bashrc
export GZ_RENDERING_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gz-rendering-9/engine-plugins
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DISPLAY=:2

ros2 run rmw_zenoh_cpp rmw_zenohd &
sleep 6
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false start_aic_engine:=true launch_rviz:=false > /tmp/gz_launch.log 2>&1 &

# Wait for engine, then:
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
```

---

## Week 1 — Baselines, Data, First Model

### Day 1 — Establish Ground Truth (Today)

**Goal:** Know what a perfect score looks like. Know where RunACT currently stands.

**Step 1: Run CheatCode (score ceiling)**
```bash
# Re-enable cameras for CheatCode (needs them)
# Edit aic_description/urdf/ur_gz.urdf.xacro: use_sim_cam default="${True}"

ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=true launch_rviz:=false
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```
Record score. **Expected: ~100/trial = 300/300.** If not, debug before proceeding.

**Step 2: Run with the official eval config**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  aic_engine_config_file:=$(ros2 pkg prefix aic_engine)/share/aic_engine/config/eval_config.yaml \
  ground_truth:=true start_aic_engine:=true launch_rviz:=false
```
This is the EXACT config used on the cloud evaluator. All testing must use this.

**Step 3: Run RunACT (Day 0 baseline)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false start_aic_engine:=true launch_rviz:=false
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT
```
Record score. This is your starting point.

**Step 4: Read the CheatCode source**
```
aic_example_policies/aic_example_policies/ros/CheatCode.py
```
Understand: how it perceives the port, how it moves to it, what commands it sends. This is your reference implementation.

---

### Day 2 — Data Collection

**Goal:** Collect 500 demonstrations with CheatCode. This feeds everything downstream.

**Enable cameras first:**
```bash
# aic_description/urdf/ur_gz.urdf.xacro line 45:
# default="${True}"   ← change back to this
# basler_camera_macro.xacro: update_rate back to 20.0, resolution 1152x1024
```

**Collect demonstrations across the full eval distribution:**
```bash
# Terminal 1: Start sim with ground truth
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  aic_engine_config_file:=...eval_config.yaml \
  ground_truth:=true start_aic_engine:=true launch_rviz:=false

# Terminal 2: Record all topics
ros2 bag record \
  /left_camera/image /center_camera/image /right_camera/image \
  /left_camera/depth /center_camera/depth \
  /joint_states /fts_broadcaster/wrench \
  /aic_controller/controller_state /aic_controller/pose_commands \
  /tf /tf_static \
  -o ~/data/cheatcode_sfp_demos &

# Terminal 3: Run CheatCode for many trials
# Modify eval_config.yaml to cycle through all 5 NIC rails
# Collect ~100 demos per rail position
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

**Target: 500 SFP demos + 100 SC plug demos.** Save trial outcomes (success/fail) alongside bags.

**Data format needed:**
- Per timestep: `[left_img, center_img, right_img, depth, joint_pos, wrench, tcp_pose]`
- Action: `tcp_velocity_command` (6D twist) from `/aic_controller/pose_commands`
- Episode boundary: trial start/end from aic_engine

---

### Day 3-4 — Train 3D Diffusion Policy (DP3)

**Why DP3:** Point clouds from the wrist RGB-D camera encode the 3D geometry of the NIC port with metric accuracy. Millimeter displacements are embedded spatially — not inferred from pixel offsets. Validated on connector insertion at 85% success with 40 demonstrations.

**Setup:**
```bash
git clone https://github.com/YanjieZe/3D-Diffusion-Policy ~/DP3
cd ~/DP3
conda activate base  # or set up new env
pip install -r requirements.txt
```

**Data preprocessing:**
```python
# Convert rosbags → DP3 dataset format
# For each timestep:
#   - Project depth image → point cloud (use camera intrinsics)
#   - Crop to 10cm³ region around gripper (port is always nearby)
#   - Subsample to 1024 points
#   - Action: tcp_velocity 6D twist
# Save as zarr or HDF5
```

**Training config:**
```yaml
# dp3_aic_sfp.yaml
task_name: aic_sfp_insertion
obs_type: pointcloud          # 3D point cloud input
action_type: tcp_velocity     # 6D Cartesian velocity output
chunk_size: 16                # predict 16 steps at once (action chunking)
n_diffusion_steps: 10         # DDIM for fast inference
batch_size: 256               # DGX-Spark can handle this
n_epochs: 3000
learning_rate: 1e-4
```

```bash
python train.py --config dp3_aic_sfp.yaml
# Training time: ~4-8 hours on DGX-Spark for 3000 epochs
```

**Evaluation during training** (every 100 epochs):
```bash
# Run policy against eval_config.yaml, record score
# Target by end of Day 4: consistent proximity score (25 pts/trial → ~78/300)
```

---

### Day 5 — Evaluate DP3 + Add Force-Aware Insertion

**Step 1: Wrap DP3 in the aic_model Policy framework:**
```python
# my_policy/dp3_policy.py
import torch
from aic_model import Policy
from dp3 import DP3Model

class DP3InsertionPolicy(Policy):
    def __init__(self):
        self.model = DP3Model.load("~/DP3/checkpoints/best.ckpt")
        self.model.eval()
    
    def insert_cable(self, get_observation, move_robot, send_feedback, task):
        target_port = task.target_port  # e.g., "SFP_PORT_0"
        
        while not self.is_done(get_observation()):
            obs = get_observation()
            
            # Build point cloud from depth image
            pcd = self.depth_to_pointcloud(obs.center_image, obs.center_camera_info)
            
            # DP3 inference
            action_chunk = self.model.predict(pcd, obs.controller_state)
            
            # Send each action in chunk
            for action in action_chunk:
                move_robot(self.to_motion_update(action))
```

**Step 2: Add force-aware insertion layer for the last millimeter:**
```python
def force_insertion(self, get_observation, move_robot):
    """Switch to force control when near port."""
    MAX_FORCE = 15.0  # Safety margin below 20N penalty threshold
    
    while not self.inserted(get_observation()):
        obs = get_observation()
        fx = obs.wrist_wrench.wrench.force.x
        fy = obs.wrist_wrench.wrench.force.y
        fz = obs.wrist_wrench.wrench.force.z
        total_force = (fx**2 + fy**2 + fz**2) ** 0.5
        
        if total_force > MAX_FORCE:
            send_feedback(f"Force {total_force:.1f}N — reducing")
            move_robot(self.retreat(0.001))  # back off 1mm
        else:
            move_robot(self.push_forward(fz=8.0))  # gentle 8N push
```

**Run and score. Target by end of Day 5: ~130-150/300**

---

### Day 6-7 — RLPD Online Refinement in Gazebo

**Goal:** Use the DP3 demonstrations as a warm start for online RL. RLPD (the algorithm inside SERL/HIL-SERL) achieves USB connector insertion in 19 minutes of real-world RL. In sim, convergence is faster.

```bash
git clone https://github.com/rail-berkeley/serl ~/serl
```

**Setup RLPD with Gazebo:**
```python
# reward function: use aic_engine scoring topics
# - +1 for each mm of insertion depth (continuous)
# - +75 for full insertion
# - -12 for wrong port
# - -1 for force > 15N

# Offline dataset: CheatCode demonstrations (500 demos)
# Online: run DP3 policy, collect failures, learn from them
# Batch: 50% offline demos, 50% online rollouts (RLPD symmetric sampling)
```

**Training loop:**
```bash
# Terminal 1: Gazebo sim running
# Terminal 2: RLPD training loop
python serl_train.py \
  --offline_data ~/data/cheatcode_sfp_demos \
  --policy dp3_checkpoint_best.ckpt \
  --n_utd 4 \
  --batch_size 256
```

**Expected:** SERL converges on USB insertion in 19–120 min in sim. Target: >60% full insertion by end of Day 7.

**Score target by end of Week 1: ~190/300** (partial insertion consistently)

---

## Week 2 — Polish, Generalize, Submit

### Day 8 — SC Plug Generalization (Trial 3)

Trial 3 is a different plug type (SC fiber optic, not SFP). Your model must generalize.

**Strategy — task-conditioned policy:**
```python
def insert_cable(self, get_observation, move_robot, send_feedback, task):
    plug_type = task.plug_type  # "SFP_MODULE" or "SC_PLUG"
    
    if plug_type == "SFP_MODULE":
        model = self.sfp_model
    else:
        model = self.sc_model  # trained separately, same architecture
    
    # Rest of insertion loop unchanged
```

**Collect SC plug demonstrations:**
```bash
# Edit eval_config.yaml: use Trial 3 (SC) config
# Run CheatCode with ground_truth:=true for 100 demos
ros2 bag record ... -o ~/data/cheatcode_sc_demos
```

**Train SC DP3 checkpoint:** Fine-tune from SFP checkpoint (faster convergence — shared insertion physics).

```bash
python train.py --config dp3_aic_sc.yaml \
  --pretrained ~/DP3/checkpoints/sfp_best.ckpt \
  --n_epochs 1000  # fewer epochs needed with pretrained init
```

---

### Day 9 — Speed Optimization (<5 Seconds)

**12 points depend on completing in <5 seconds.** Profile current timing:

```python
import time
start = time.time()
# policy runs
duration = time.time() - start
print(f"Duration: {duration:.2f}s")  # target: <5.0s
```

**Optimizations:**
1. **Reduce DP3 denoising steps:** 10 → 4 steps (DDIM, 2.5× faster inference)
2. **Larger action chunk size:** predict 32 steps at once (fewer inference calls)
3. **Precompute port estimate:** from first observation before moving
4. **Async inference:** run next prediction while executing current action chunk
5. **Tune TCP velocity limits** in controller config (check `aic_controller` params)

```python
# DDIM with fewer steps — faster inference, minimal quality loss
model.set_inference_steps(4)  # was 10
```

**Smooth trajectories for 6 jerk points:**
```python
from scipy.signal import savgol_filter
# Post-process action chunk before sending
smoothed = savgol_filter(action_chunk, window_length=15, polyorder=2, axis=0)
```

---

### Day 10 — Full Insertion + NIC Rail Stress Test

**Test all 5 NIC card positions and both port targets:**
```yaml
# Modify eval_config.yaml to test all combinations:
# NIC rails: nic_rail_0 through nic_rail_4
# Ports: SFP_PORT_0 and SFP_PORT_1
# SC ports: SC_PORT_0 and SC_PORT_1
```

```bash
for rail in 0 1 2 3 4; do
  # Update eval_config.yaml with rail=$rail
  # Run 10 trials, record scores
  echo "Rail $rail: $(run_trials 10)"
done
```

**Failure analysis:** For any failure, inspect:
- Did the policy identify the correct port? (Task description parsing)
- Did it align the plug axis? (Rotation error before contact)
- Did force exceed 20N? (Check wrench topic)
- Did it collide with the board? (Check contacts topic)

---

### Day 11 — Integration + Ablation Study

Run a clean 30-trial evaluation against `eval_config.yaml`:

| Metric | Target | Record |
|---|---|---|
| Score (avg/trial) | >85 | |
| Full insertion rate | >80% | |
| Mean duration | <5s | |
| Force violations | 0 | |
| Contact violations | 0 | |

If force violations occur: tune the 15N threshold, slow insertion speed.
If contact violations occur: add a safety bounding box check before movement.
If wrong port insertions occur: add explicit port identity verification from task message.

---

### Day 12 — Package as Docker Container

The submission is a Docker container with your `aic_model` node inside.

```bash
# Use the provided aic_model Dockerfile template
cd /home/kiwoos/work/Intrinsics_Assembly_Robotics/docker/aic_model/
```

**Edit `Dockerfile` to include your model checkpoint:**
```dockerfile
FROM ros:kilted-ros-base

# Install pixi (pinned version)
RUN curl -fsSL https://pixi.sh/install.sh | sh
RUN /root/.pixi/bin/pixi self-update --version 0.67.2

COPY . /ws_aic/src/aic
COPY my_policy/ /ws_aic/src/aic/my_policy/
COPY checkpoints/ /checkpoints/  # DP3 checkpoint

WORKDIR /ws_aic/src/aic
RUN pixi install

ENV MODEL_CHECKPOINT=/checkpoints/dp3_best.ckpt
```

**Local verification before submit:**
```bash
# Build your model container
docker build -t aic_model:latest -f docker/aic_model/Dockerfile .

# Run eval container + your model container together
export DBX_CONTAINER_MANAGER=docker
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true &

# Run your model in the container
docker run --rm --network host aic_model:latest \
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=my_policy.DP3InsertionPolicy
```

---

### Day 13 — First Portal Submission

```bash
# Tag and push to Intrinsic's registry
docker tag aic_model:latest <your-registry>/aic_model:v1
docker push <your-registry>/aic_model:v1
# Submit via the AIC portal
```

**Monitor the official score.** Compare with local eval_config.yaml score. If there's a gap, investigate.

---

### Day 14 — Buffer + Second Submission

Use the day to fix any issues found from portal feedback:
- Score lower than local? → Likely import timing issue or missing dependency in container
- Force violations on portal? → Tighten safety margin, add graceful retry
- Wrong port on portal? → Harden port identification from task message

Submit improved version.

---

## Key Technical Decisions Explained

### Why DP3 over plain Diffusion Policy
Point clouds encode 3D geometry metrically. A 1mm misalignment is 1mm in the point cloud — not an ambiguous pixel offset that depends on camera angle and lighting. For connector insertion requiring <1mm precision, this matters enormously. Published results: 85% success with 40 demos on connector insertion tasks.

### Why RLPD/SERL over pure imitation learning
CheatCode demonstrations give near-perfect trajectories, but slight grasp deviations (~2mm, ~0.04 rad as stated in the qual doc) cause IL models to fail. Online RL adapts to exactly the distribution of errors it sees. SERL literature shows USB insertion → 100% success in 2.5 hours of human-guided RL. In sim, convergence is faster.

### Why force control for the last millimeter
The F/T sensor at the wrist is the only sensor that can detect the plug contacting the port rim. Vision cannot resolve sub-millimeter alignment. A pure vision policy will either miss the port or slam into it. Hybrid: DP3 for approach + force control for the 1–3mm insertion stroke.

### Why the <5s target matters so much
12 points for duration vs 6 for smoothness and 6 for efficiency. Duration is the highest single Tier 2 reward. The plug starts within centimeters of the port. A policy that takes >60 seconds scores the same as one that never reaches the port.

### Why task-conditioning for SC generalization
Trial 3 is a different plug type. A single model conditioned on the task message (plug type, target port ID) generalizes better than two separate models — it learns shared insertion physics and only adapts the geometry. Fine-tuning from the SFP checkpoint reduces SC training time from ~8 hours to ~2 hours.

---

## Reference Architecture (What You're Building)

```
Observation (20Hz)
  ├── RGB-D → Point Cloud (1024 pts, cropped 10cm³ around TCP)
  ├── Joint States (7-DOF)
  ├── F/T Wrench (6D)
  └── TCP Pose + Velocity

          │
          ▼
  ┌─────────────────────────────────┐
  │  Phase 1: DP3 Approach          │  (point cloud → TCP velocity chunk)
  │  - 3D MLP point encoder         │  Until plug near port
  │  - DDIM diffusion (4 steps)     │
  │  - Action chunk size 32         │
  │  - Savitzky-Golay smoothing     │
  └────────────┬────────────────────┘
               │ plug at port entrance
               ▼
  ┌─────────────────────────────────┐
  │  Phase 2: Force Controller      │  (F/T → TCP velocity)
  │  - Target insertion force: 8N   │  Until full insertion detected
  │  - Lateral force → correction   │
  │  - Hard limit: 15N (5N margin)  │
  └─────────────────────────────────┘
               │ insertion event detected
               ▼
          Done ✓ (75 pts + Tier 2)
```

---

## Research Foundation

| Paper | Key Result | Used For |
|---|---|---|
| **DP3** (Ze et al., RSS 2024) | 85% insertion success, 40 demos | Primary policy |
| **HIL-SERL** (Luo et al., Science Robotics 2024) | 100% USB insertion, 2.5h | RL refinement |
| **RLPD** (Ball et al., ICML 2023) | Offline+online RL, symmetric sampling | Core RL algorithm |
| **ACT** (Zhao et al., RSS 2023) | Action chunking, temporal ensemble | Ablation baseline |
| **Diffusion Policy** (Chi et al., RSS 2023) | Multimodal action distribution | 2D image baseline |
| **VITaL** (Garratt et al., IROS 2024) | 85% USB insertion with F/T pretraining | F/T sensor integration |
| **DINOBot** (Di Palo et al., ICRA 2024) | One-shot via DINOv2 alignment | Port localization fallback |

---

## Recruiter One-Liner

> *"Competing in Intrinsic's AI for Industry Challenge ($180K prize), building a DP3 + HIL-SERL pipeline for dexterous cable connector insertion on a UR5e robot in Gazebo, targeting >250/300 across randomized SFP and SC plug insertion trials on NVIDIA DGX-Spark."*

---

## Environment Quick Reference

```bash
# Check simulation speed
ros2 topic hz /joint_states  # should be ~500Hz

# Check GPU usage
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Run official eval config
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  aic_engine_config_file:=$(ros2 pkg prefix aic_engine)/share/aic_engine/config/eval_config.yaml \
  ground_truth:=false start_aic_engine:=true launch_rviz:=false

# Re-enable cameras (needed for training)
# aic_description/urdf/ur_gz.urdf.xacro → use_sim_cam default="${True}"
# aic_assets/models/Basler Camera/basler_camera_macro.xacro → update_rate 20.0, 1152×1024

# Disable cameras (for fast physics demo)
# aic_description/urdf/ur_gz.urdf.xacro → use_sim_cam default="${False}"

# DGX-Spark IP
hostname -I | awk '{print $1}'

# VNC viewer on Mac → connect to <ip>:5902
```
