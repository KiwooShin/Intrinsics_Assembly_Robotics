# Camera-Only Imitation Learning for Robotic Cable Insertion

**A learned visuomotor policy that seats SFP/SC fiber-optic plugs into NIC ports with a UR5e — privileged-staged above the known port, then driven home from RGB cameras + wrist force with no ground-truth pose fed to the network.** (Camera-only *port localization* from a cold start is shown to be sensor-limited on this rig — the 128&nbsp;px cameras can't resolve a 2&nbsp;mm port; see the write-up.)
Built on the [Intrinsic AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) toolkit (Gazebo simulation, real contact physics, force-aware scoring). This fork adds the full research pipeline: oracle distillation, ACT-style policy learning, a hardened evaluation harness, and a statistics-first experiment log.

## Milestones

The **RUN&nbsp;#4 curriculum** reopened the insertion problem: privileged-stage the plug above the port, learn the *seating* skill from vision + force, then grow the lateral offset stage by stage. That produced the project's first learned-policy seats — and a systematic study of exactly how far off-center they generalize, and why. The full interactive write-up (dose-response, code-verified mechanism, and the partial-observability analysis) is [`dashboard/showcase.html`](dashboard/showcase.html) *(open in a browser)*; the running lab notebook is [`SESSION_REPORT.md`](SESSION_REPORT.md).

### ✦ Featured — two connectors, one system (SFP + SC side by side)

![Combined SFP + SC demo: two connector types seated by one system](docs/media/combined_sfp_sc_together_2026-07-22_22h.gif)

The same UR5e and the same curriculum framework seating **two very different connectors together**. **Left:** an **SFP fiber plug** dropping straight into an aligned NIC port (world-z insertion, `m3c`/`m4` model, **93/100**). **Right:** an **SC bayonet** following a **physically rotated** port's own insertion axis (`sc_insert` model, **87/100**). Two connector geometries, two insertion strategies, one learned system — both seat.

### Milestone 1 — First learned-policy seat

![Milestone 1: first learned-policy seat](docs/media/milestone1_first_seat_2026-07-22_08h.gif)

The project's first insertion by a **learned** policy — every prior seat came from the scripted oracle. Privileged-staged above the aligned port (true port TF), the policy then descends and seats the SFP plug from 3 RGB cameras + wrist force with **no ground-truth pose fed to the network**. Best demo trial **93/100** (tier-3); **3/3** on the matched-seed suite (mean **~88/100**).

### Milestone 2 — Off-center insertion (2&nbsp;mm)

![Milestone 2: seating a 2mm-offset plug](docs/media/milestone2_offset_2mm_2026-07-22_09h.gif)

Growing the staged lateral offset, the policy learns to correct at the mouth and seat an off-center plug — a **verified 2&nbsp;mm staged seat scores 92.9, and even 3&nbsp;mm seats (93.3)**, both shown under **Dynamic rollouts** below. Off-center recovery is the policy's headline dynamic skill — the plug is commanded well off-center and has to travel and correct during a *blind* descent (the offset isn't visible until contact). It does not always succeed; reliability at a given offset varies scene-to-scene (see **Dynamic rollouts** below), and the deeper partial-observability limit — plus a contact-gated fix that was built, tested, and honestly shown to arrive too late — is dissected in the write-up.

### Milestone 3 — Robustness across 10 random locations

![Milestone 3: 10 random locations](docs/media/milestone3_ten_random_locations_2026-07-22_14h.gif)

The **same seating policy on 10 fully-randomized scenes** — different board poses, NIC rails, and ports — seats **9/10 at ~93/100**, up from **7/10** before robustifying on a wider, pose-diverse corpus (`m4` vs `m3c`). The lone miss is a rare execution-failure/runaway (~10% across the distribution), disclosed in the tally rather than cropped out. Privileged-staged curriculum (a training-legal capability study; the *blind* full task is camera-resolution-limited — see the write-up).

### Milestone 4 — SC (angled) fiber insertion — a separate model for a harder task

![Milestone 4: SC angled insertion](docs/media/milestone4_sc_insertion_2026-07-22_18h.gif)

A genuinely harder *second* connector task. The **SC/LC fiber port is physically rotated**, so a top-down descent rams it — the plug must be inserted along a **pose-conditioned axis**. A **separate `sc_insert` model** (distinct from the SFP policy; privileged-staged, then 3 RGB cameras + wrist force with no ground-truth pose fed to the network) learns this and **seats 5/6 SC poses at ~85–87**. Solving it required matching the reference oracle's slow continuous descent so the impedance controller converges onto the tilted axis (a top-down or fast descent leaves the plug ~3.5 mm off). Honest scope: the scripted oracle itself only cleanly seats ~37 % of SC poses, so this is a proof of *learned* capability on the seating subset; the SFP task keeps its own `m3c`/`m4` models.

### ⚡ Dynamic rollouts — challenging offsets, honestly scored

The hard, high-movement cases. From a **privileged-staged start** the plug is commanded **well off-center**, so it has to *travel* and correct a large offset during the blind descent (three RGB cameras + wrist force; the network sees **no ground-truth pose**). Some seat, some don't — and the seat/miss split at a given offset depends on the scene, not just the magnitude, so both are shown.

| Large-offset recovery | Large-offset recovery |
|:---:|:---:|
| ![3 mm offset corrected and seated](docs/media/dynamic_offset_edge_3p0mm_2026-07-22_23h.gif) | ![2 mm offset corrected and seated](docs/media/dynamic_offset_edge_2p0mm_2026-07-22_23h.gif) |
| **3 mm offset → seated · 93/100.** A large lateral correction swings in during descent. | **2 mm offset → seated · 92.9/100.** The plug walks over to the port and drops in. |
| ![1.2 mm offset drifts past — not seated](docs/media/dynamic_offset_edge_1p2mm_2026-07-22_23h.gif) | ![SC angled insertion, seated](docs/media/dynamic_sc_angled_s1_2026-07-22_23h.gif) |
| **1.2 mm offset → NOT seated · 57/100.** The plug drifts past before it can feel the port — an honest failure, shown not hidden. | **SC angled insertion → seated · 87/100.** A long, visibly tilted descent along the rotated port's own axis. |

*Each GIF: title card → the full three-camera rollout (left / center / right, ~72 frames) → the measured engine result.*

<details>
<summary><b>Earlier runs</b> — camera-only approach, pre-curriculum</summary>

| Scripted oracle — a full SFP insertion (the task) | Learned policy — camera-only rollout |
|:---:|:---:|
| ![Oracle demo: complete SFP insertion](docs/media/oracle_demo_sfp_rail0_sfp_port_0.gif) | ![Learned policy rollout on official_1](docs/media/policy_p1_k16_official_1.gif) |
| `CheatCode` oracle (reads ground-truth port poses) seats the plug — engine ≈93/100. The teacher and per-trial upper bound. | ACT policy (3 RGB cams + TCP state, ~0.75 M params) drives a clean approach on an official config. 2.5× speed, three-camera view. |

| Adopted checkpoint `v2_wide` on official_2 — engine 41.6 |
|:---:|
| ![Adopted checkpoint v2_wide on official_2](docs/media/policy_v2_wide_official_2.gif) |
| Characteristic pre-curriculum behavior on a harder pose: a clean camera-only approach that stalls ~5–8&nbsp;cm from the port (the **last-inch attractor**). The curriculum milestones above are how the seat was finally reached. |

</details>

*Milestone GIFs: a title card, then the 3-camera rollout (left / center / right) with the measured result. Earlier-run GIFs are 2.5× timelapses.*

## Results

| Evaluation | Score | Notes |
|---|---:|---|
| **Best official-config eval (3 trials, /300)** | **119.4** | adopted checkpoint `v2_wide` — ~40/trial **tier-2 proximity/directed-approach**, **no tier-3 seat**; no camera-only checkpoint seats the official poses (the seats come from the curriculum's privileged-staged start) |
| Oracle (scripted, ground-truth poses) | ≈93 / 100 per trial | distillation teacher; 97.5 % keep-rate over a 40-config collection campaign |
| 180-s matched-seed suite, 15 configs (mean) | v2_wide 7.75 · p2_k8 5.29 · p1_k16 3.03 | harder stratified poses; officials-only sums: 97.7 / 64.9 / 68.8 (/300) |
| Unit tests | 295 green | pure-logic seams for config gen, dataset prep, chunk ensembling, eval harness |

**The central research finding:** on poses harder than the official ones, every checkpoint approaches cleanly and then stalls 0.05–0.08 m from the port for the full trial — a **last-inch fixed-point attractor** caused by mode-averaging of a deterministic ACT head toward the demos' zero-velocity endings ([Zhao et al., 2023, arXiv:2304.13705](https://arxiv.org/abs/2304.13705)). The ranked fixes (last-inch DAgger, CVAE action head, residual RL — [ResiP](https://arxiv.org/abs/2407.16677) / [RLDG](https://arxiv.org/abs/2412.09858)) are laid out in [SESSION_REPORT.md](./SESSION_REPORT.md).

**The evaluation-rigor finding:** identical config + seed trials vary by **σ ≈ 3–18 points** run-to-run (Gazebo/ROS timing nondeterminism), enough to flip trial outcomes. A promising single-run A/B result (temporal ensembling, "+3.9, wins on 5/5 configs") was **overturned** by a 15-config paired-bootstrap confirmation and a 3-repetitions-per-arm experiment (30 trials): no real effect, added variance. All adoption decisions here are therefore CI-backed at n≥3 — the negative result and the noise-floor measurement are documented as first-class deliverables.

## Method

```
gen_config.py                 # stratified randomized scene configs (rail × plug × port × yaw band)
  └─ collect_campaign.sh      # CheatCode oracle rollouts → rosbags (resumable, score-gated keeps)
       └─ prepare_dataset.py  # bag → synced .npy episodes (3 cams + TCP state, task-window trimmed)
            └─ train_v3       # ACT-style CNN + action chunking (K×6 TCP-velocity chunks, ~17 min/run)
                 └─ DeployACT # receding-horizon MODE_POSITION execution @ ~18 Hz
eval_suite.py + eval_batch.sh # matched-seed suites, IQM + paired-bootstrap CIs, resumable batches
```

- **Observation → action:** 3 RGB cameras + 7-D TCP state → chunk of K×6 TCP velocities; execute the first 4 actions, re-predict (receding horizon).
- **Why image-based ACT (not point-cloud DP3):** the sim bridges RGB only — no depth topics.
- **Deployment insight that unlocked scoring:** the stock velocity mode integrates its reference open-loop and freezes the arm; re-anchoring every chunk through position-mode targets took the *proximity* score from 36 → 119.4 (tier-2 only — a directed approach, not a seat).
- **Dataset:** 93 oracle episodes across stratified + failure-driven collection phases.

## Engineering highlights

- **Resumable, self-healing eval harness** — flock-guarded batches, orphan sweeps, skip-if-done resume, DONE markers; survived a 48 h autonomous run with a 5-minute liveness watchdog.
- **Statistics-first protocol** — matched-seed suites, IQM + paired-bootstrap CIs, measured sim noise floor, n≥3 repetition rule codified after it overturned a false positive.
- **295 unit tests** (stdlib `unittest`, no ROS/GPU needed) over every pure-logic seam.
- **Full lab notebook** — [`SESSION_REPORT.md`](./SESSION_REPORT.md) records every experiment with tables, verdicts, and paper citations; [`progress.md`](./progress.md) is the 2-hourly run log.

## Where to look

| File | What it is |
|---|---|
| [`SESSION_REPORT.md`](./SESSION_REPORT.md) | The lab notebook: every experiment, result table, and adoption decision |
| [`Plan.md`](./Plan.md) / [`ResearchPlan.md`](./ResearchPlan.md) | Architecture rationale and research plan |
| [`aic_example_policies/.../DeployACT.py`](./aic_example_policies/aic_example_policies/ros/DeployACT.py) | The deployed policy node (receding horizon + opt-in chunk ensembling) |
| [`eval_suite.py`](./eval_suite.py), [`eval_batch.sh`](./eval_batch.sh) | Matched-seed evaluation suites + resumable batch runner |
| [`train_smoke.py`](./train_smoke.py), [`prepare_dataset.py`](./prepare_dataset.py) | Training and dataset pipeline |

---

# AI for Industry Challenge Toolkit (upstream)

[![build](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml)
[![style](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml)

The **AI for Industry Challenge** is an open competition for developers and roboticists aimed at solving some of the hardest, high-impact problems in robotics and manufacturing. This repository contains the official toolkit; for registration details, official rules, and FAQs, visit the [event page](https://www.intrinsic.ai/events/ai-for-industry-challenge).

## Toolkit Guide

Welcome to the AIC toolkit documentation. This guide walks you through the complete workflow for participating in the challenge — from understanding the requirements to submitting your solution.

Follow the sections below to navigate through each phase of the process.

1. **📖 Understand the Challenge**
   - Read the [Challenge Overview](./docs/overview.md) to understand the goals.
   - Review the [Qualification Phase](./docs/phases.md#qualification-phase-train-your-model) to understand what you'll be building.
   - Review the [Scoring Guide](./docs/scoring.md) to understand how you'll be scored.

2. **🔧 Set Up Your Environment**
   - Follow the [Getting Started](./docs/getting_started.md) guide to set up and validate your development environment.
   - Run the evaluation container and set up your local workspace with Pixi.

3. **💻 Develop Your Policy**
   - Explore the [Scene Description](./docs/scene_description.md) to learn how to customize and explore the environment.
   - Review [AIC Interfaces](./docs/aic_interfaces.md) to understand available interfaces to communicate with sensors and actuators.
   - Consult [AIC Controller](./docs/aic_controller.md) to learn about controlling the robot.
   - Consult the [Challenge Rules](./docs/challenge_rules.md) to ensure compliance.
   - Start with the [Policy Integration Guide](./docs/policy.md) to implement your solution.
   - See [Participant Utilities](./docs/participant_utilities.md) for a list of helpful tools.

4. **🧪 Test Your Solution**
   - Use the provided simulation environment to test your policy.
   - Run `aic_engine` with the `sample_config` in [`aic_engine/config/`](./aic_engine/config/) to test different scenarios. For more information on running the `aic_engine` with different configs, see the [aic_engine README file](./aic_engine/README.md).
   - Create your own test scenarios by following the configuration example in [`aic_engine/config/`](./aic_engine/config/) to run with `aic_engine`.
   - Refer to [Troubleshooting](./docs/troubleshooting.md) if you encounter issues.

5. **📦 Submit Your Entry**
   - Package your solution following the [Submission Guidelines](./docs/submission.md).
   - Test your container locally before submitting following [these instructions](./docs/submission.md#verify-locally).
   - Submit through the official portal following [these instructions](./docs/submission.md#2-upload-your-image-to-our-registry).

---

## Toolkit Architecture

![AIC Competition Components](../media/aic_competition_components.png)

The AI for Industry Challenge toolkit is divided into **two main components**:

### 1. Evaluation Component (Provided - Run by Organizers)

This component provides the complete evaluation infrastructure:
- **`aic_engine`** - Orchestrates trials and computes scores.
- **`aic_bringup`** - Launches simulation environment (Gazebo, robot, sensors).
- **`aic_controller`** - Low-level robot control with force management.
- **`aic_adapter`** - Sensor fusion and data synchronization.

**What you receive:** Standard ROS sensor topics providing camera images, joint states, force/torque measurements, and TF frames.

### 2. Participant Model Component (Your Implementation - What You Submit)

This is what you develop and submit:
- **A ROS 2 node** that follows the behavioral requirements defined in [Challenge Rules](./docs/challenge_rules.md).
- **Your custom logic** - Code to process sensor data and command the robot to insert cables.

**What you provide:** A container with a ROS 2 Lifecycle node named `aic_model` that responds to the `/insert_cable` action and outputs robot motion commands via standard ROS topics/services.

**Convenient Entry Point:** We provide an `aic_model` framework that handles all the ROS 2 boilerplate and lifecycle management. You simply implement a Python policy class that gets dynamically loaded at runtime. See the [Policy Integration Guide](./docs/policy.md) for details.

### Development and Submission Workflow

> [!IMPORTANT]
> **ROS 2 Distribution:** The official evaluation of all submissions will be conducted using **ROS 2 Kilted Kaiju**. If you choose to develop or test your policy using a different ROS 2 distribution (e.g., Humble or Jazzy), it is entirely your responsibility to ensure compatibility and support. Please note that **inter-distro communication is not guaranteed and not officially supported**.

**Development Options:**
- Develop inside a container (recommended - matches evaluation environment).
- OR develop in native Ubuntu 24.04 environment (requires all dependencies).

**Submission Requirements:**
- Package your solution using the provided `aic_model` Dockerfile.
- Submit your container - it must respond to standard ROS inputs and command the robot to insert cables.
- Your container interfaces with the evaluation component via ROS topics.

---
## Repository Structure

```
aic/
├── aic_adapter/          # Adapter for interfacing between model and controller
├── aic_assets/           # 3D models and simulation assets
├── aic_bringup/          # Launch files for starting the challenge environment
├── aic_controller/       # Robot controller implementation
├── aic_description/      # Robot and environment URDF/SDF descriptions
├── aic_engine/           # Trial orchestration and validation engine
├── aic_example_policies/ # Example policy implementations
├── aic_gazebo/           # Gazebo-specific plugins and configurations
├── aic_interfaces/       # ROS 2 message, service, and action definitions
├── aic_model/            # Template for participant policy implementation
├── aic_scoring/          # Scoring system implementation
├── aic_utils/            # Utility packages and tools
├── docker/               # Docker container definitions
└── docs/                 # Comprehensive documentation
```

---

## Key Packages for Participants

### `aic_model` - Convenient Policy Framework (Recommended)
This package provides a ready-to-use ROS 2 Lifecycle node that dynamically loads and executes your Python policy implementation. It handles all ROS 2 boilerplate, lifecycle management, and challenge rule compliance, allowing you to focus on implementing your policy logic.
- **Location**: `aic_model/`.
- **Documentation**: [Policy Integration Guide](./docs/policy.md).
- **Tutorial**: [Creating a New Policy Node](./docs/policy.md#tutorial-creating-a-new-policy-node).

> **Note:** While we recommend using this framework, you may implement your own ROS 2 node from scratch as long as it adheres to the [Challenge Rules](./docs/challenge_rules.md).

### `aic_interfaces` - Communication Protocols
Defines all ROS 2 messages, services, and actions used in the challenge.
- **Location**: `aic_interfaces/`.
- **Documentation**: [AIC Interfaces](./docs/aic_interfaces.md).

### `aic_example_policies` - Reference Implementations
Example policies demonstrating different approaches and techniques.
- **Location**: `aic_example_policies/`.
- **README**: [aic_example_policies/README.md](./aic_example_policies/README.md).

### `aic_bringup` - Launch the Environment
Launch files to start the simulation, robot, and scoring systems.
- **Location**: `aic_bringup/`.
- **README**: [aic_bringup/README.md](./aic_bringup/README.md).

### `aic_engine` - Trial Orchestrator
Manages trial execution, validates participant models, and collects scoring data.
- **Location**: `aic_engine/`.
- **README**: [aic_engine/README.md](./aic_engine/README.md).

---

## Additional Documentation

### Challenge Information

* **[Challenge Overview](./docs/overview.md):** High-level summary of the competition goals and structure.
* **[Competition Phases](./docs/phases.md):** Details on Qualification, Phase 1, and Phase 2.
* **[Qualification Phase](./docs/qualification_phase.md):** Detailed technical overview of the qualification phase trials and scoring.
* **[Challenge Rules](./docs/challenge_rules.md):** Required behavior for participant models.
* **[Scoring](./docs/scoring.md):** Metrics and methods used to evaluate performance.
* **[Scoring Test Examples](./docs/scoring_tests.md):** Reproducible examples exercising each scoring tier with exact commands.

### Technical Documentation

* **[Getting Started](./docs/getting_started.md):** How to set up your local development environment.
* **[Policy Integration](./docs/policy.md):** Guide to implementing your policy in the `aic_model` framework.
* **[AIC Interfaces](./docs/aic_interfaces.md):** ROS 2 topics, services, and actions available to your policy.
* **[AIC Controller](./docs/aic_controller.md):** Understanding the robot controller and motion commands.
* **[Scene Description](./docs/scene_description.md):** Technical details of the simulation environment.
* **[Task Board Description](./docs/task_board_description.md):** Physical layout and specifications of the task board.
* **[Troubleshooting](./docs/troubleshooting.md):** Common issues and debugging strategies.

### Reference Materials

* **[Glossary](./docs/glossary.md):** Terminology and definitions used throughout the AI for Industry Challenge

### Submission

* **[Submission Guidelines](./docs/submission.md):** How to package and submit your final model.

---


## Support and Resources

- **Discussions**: Engage in conversations and ask questions about the challenge on [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/). The community is encouraged to participate in discussions and assist each other.
- **Issues**: Report any bugs or technical issues via [GitHub Issues](https://github.com/intrinsic-dev/aic/issues). Please refrain from using the Issue tracker for general questions about the challenge.
  - **Note:**: Review the list of [known issues](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22known%20issue%22) and [bugs](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) before opening a new ticket.
- **Event Page**: Visit the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) for official updates.

---

## License

This project is licensed under the Apache License 2.0 - see the individual package files for details.
The [aic_isaac](./aic_utils/aic_isaac/) folder contains files licensed under BSD-3 - see [aic_isaac/LICENSE](./aic_utils/aic_isaac/LICENSE).
