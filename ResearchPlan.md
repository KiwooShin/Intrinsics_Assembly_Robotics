# Research & Execution Plan — 48h Autonomous Run (2026-07-12 → 07-14)

Synthesized by the orchestrator from 5 recon sub-agent reports (Phase-1
requirements, code/environment audit, and 3 literature lanes). Governed by
[CLAUDE.md](./CLAUDE.md). Progress cadence: [progress.md](./progress.md) every 2 h.

---

## 0. Situation

- **Task:** qualification-style trials remain the training target: UR5e inserts
  SFP/SC plug into NIC port, 3 trials × 100 pts (75 insertion; 12 duration<5s;
  6 smoothness; 6 path efficiency; −12 force >20 N/1 s; −24 any off-limit contact).
  Tier-2 bonuses gated on Tier-3 > 0. Distractor NIC cards/ports appear in eval
  scenes → policy must be task-conditioned (port selection from `Task.msg`).
- **⚠ Phase-1 spec is NOT public** (docs say "Coming Soon") and deadlines
  conflict: local docs say Phase-1 eval **Jul 14–21**; the live event page says
  results **Aug 4**. USER ACTION: confirm real deadline/spec via portal/email.
  Until then we maximize the qualification-task score (Phase 1 "incorporates
  trained models", so this work carries forward regardless).
- **Assets:** validated demo pipeline (oracle → trim → .npy), 16 episodes,
  ACT-lite checkpoint `v2_wide.pt`, efficient trainer (~5,600 fr/s).
- **Blocker being fixed now:** torch/rclpy interpreter split (venv overlay fix
  in progress) — gates all in-sim scoring.

## 1. Findings that set strategy (evidence → decision)

| # | Finding | Source | Decision |
|---|---------|--------|----------|
| 1 | Connector-insertion BC reaches 92.8–100% with plain CNN + static cam + **100–300 demos**; transformer vision backbones did 7–8× WORSE; **val MSE/L1 does not predict success** | arXiv:2602.22100 | Keep CNN encoders; scale demos 16→150+; select checkpoints by scored rollouts, never val L1 |
| 2 | Generalization scales with **config diversity**, not demo count per config (~power law; ≥0.9 success at ~32 diverse cells × ~50 demos) | arXiv:2410.18647 (ICLR'25 oral) | Stratified collection over rail×plug×port cells; ≤1–2 demos per exact config |
| 3 | **Action chunking + noise-injected collection** (execute oracle+noise, log clean label) provably kills compounding error — no fancy active learning needed | arXiv:2507.09061; DART (CoRL'17) | Add noise-injection mode to the collector; oracle relabeling is free |
| 4 | Queryable-expert DAgger is our textbook setting; failure-driven top-up beats uniform once easy gains are in | DAgger; arXiv:2410.14868; arXiv:2508.05310 | Phase-2 collection targets configs where the current policy scores worst |
| 5 | Proprioceptive state becomes a shortcut → policy ignores vision, kills spatial generalization; fix = high-rate state dropout (needs relative actions ✓) | arXiv:2509.18644 | Train with TCP-pose dropout ~50–80% |
| 6 | F/T input token lifts contact-phase success (80% vs 50% vision-only) | Haptic-ACT arXiv:2409.11925 | Record + feed wrench; pipeline currently DROPS it — fix prepare_dataset |
| 7 | Far-future chunk error = multimodality symptom; CVAE/VQ heads fix it; random pixel-shift aug is near-mandatory even when train≈eval (73%→27% without) | ACT 2304.13705; VQ-BeT 2403.03181; robomimic 2108.03298; DrQ 2004.13649 | Add shift aug now; A/B CVAE (LeRobot ACT) and VQ-BeT vs L1 head |
| 8 | Chunk-boundary jerk/stalls leak smoothness+duration points; LiPo post-processor smooths without retraining | LiPo arXiv:2506.05165 | Wrap deployment with blending/jerk-min smoothing |
| 9 | Realism DR is pointless when eval == training sim; only task-instance randomization matters (match eval distribution) | ICLR'22 DR study; Procgen 1812.02341 | No visual/physics DR; sample eval-config ranges exactly |
| 10 | Residual RL on frozen chunked BC: 5%→99% on 0.2 mm peg; engine sub-scores = ready shaped reward; RLDG: distill RL back into BC for clean deployment | ResiP 2407.16677; 2509.19301; RLDG 2412.09858; Ng'99 shaping | Stage-2 (post-BC-baseline): residual RL prototype, then re-distill |
| 11 | Paired matched-seed eval + sequential testing separates policies at small n; IQM + bootstrap CIs | rliable 2108.13264; 2405.05439; TRI STEP | Fixed stratified 50-config eval suite, same seeds for every checkpoint |

**SOTA models considered** (rule §4): SmolVLA-450M (`lerobot/smolvla`, Apache-2.0
— best VLA fit if demos >100; deferred), π0/π0.5/π0-FAST (openpi — heavy;
borrowed idea: knowledge insulation), GR00T N1.5-3B (`nvidia/GR00T-N1.5-3B` —
license non-commercial, verify challenge compatibility before any use), HPT
(`liruiw/hpt` — weak fit), VQ-BeT + LeRobot ACT (adopt as A/B candidates),
DINOv2/SAM-2 (perception fallback only), DP3/FlowPolicy/MP1 (REJECTED — no
depth topics in sim). Full citation list lives in Research.md + this table.

## 2. Execution plan (48 h, heartbeat-driven)

Two always-on lanes + experiment waves. The sim is the scarce resource — it
runs data collection or eval at all times; the GPU trains between.

**Lane A — Data (continuous):**
- A0 (now): fix pipeline to record wrench + joints into episodes (26-D-capable
  state); add stratified `gen_config` sampling with seeds; storage-light as ever.
- A1: Phase-0 coverage — ~40–50 demos, 1 per stratum (rail 0-4 × plug SFP/SC ×
  port × distractor presence), continuous axes uniform.
- A2: Phase-1 volume — ~70–90 noise-injected demos at fresh random configs.
- A3: Phase-2 top-up — ~40–60 demos at the worst-scoring cells of the current
  best policy (auto-DAgger).

**Lane B — Train/Eval (continuous):**
- B0 (now): venv runtime fix → score `v2_wide.pt` in-sim = baseline row.
- B1: eval harness `eval_suite.py` — fixed stratified 50-config suite, matched
  seeds, parses scoring.yaml, emits the rule-mandated experiment table.
- B2: retrain at every +30–50 demos; score on suite; keep leaderboard.

**Experiment waves (GPU, between retrains):**
- W1: input/regularization — pixel-shift aug, TCP-pose dropout, wrench token.
- W2: head A/B — L1 (control) vs LeRobot-ACT CVAE vs VQ-BeT, same data.
- W3: deployment quality — LiPo smoothing wrapper, chunk-size 16 vs 32–48,
  receding-horizon execution; measure duration/smoothness sub-scores.
- W4 (stretch, if BC ≥ partial-insertion consistently): ResiP-style residual RL
  on engine shaped reward; RLDG re-distillation.
- Every wave ends with an **analysis sub-agent** pass (rule §4) comparing results
  to the cited literature and updating next-wave priorities.

**Success criteria:** baseline scored by H+6; ≥100 total demos by H+24; a
policy with repeatable full insertions on the eval suite by H+40; best-model
demo video + final report by H+48.

## 3. Visualization & demo plan

1. **Per-milestone insertion videos** (`make_video.py`, 3-cam side-by-side) →
   `demo/`, newest linked from progress.md. Include one CheatCode reference
   video vs learned-policy video at the same config ("oracle vs learned").
2. **Live scoreboard dashboard** (HTML artifact, updated each heartbeat):
   experiment table (title / success / key metric), demo-count vs score curve,
   per-stratum heatmap (rail × plug success), score-tier waterfall
   (proximity → partial → full → bonuses).
3. **Fancy final demo (H+40→48):** 60–90 s highlight reel — task board intro
   frames, montage of insertions across randomized configs, training/score
   progression chart, side-by-side oracle vs policy, final scoreboard. Rendered
   with ffmpeg from collected rollout frames; kept in `demo/` (not committed).
4. **Stretch:** action-chunk visualization overlay (predicted 16-step TCP
   trajectory projected into the center camera) — compelling recruiter-facing
   artifact and a genuine debugging tool.

## 4. Risks

| Risk | Mitigation |
|------|------------|
| Phase-1 deadline is actually Jul 14 | Plan already maximizes qualification-task policy; user confirms spec ASAP |
| Runtime fix fails (venv/torch aarch64) | Fallbacks: pip --user (careful), inference sidecar over ROS topic |
| Closed-loop rollout much worse than val L1 suggests | Expected (covariate shift); that is what Lanes A2/A3 + DAgger exist for |
| Sim throughput limits demos (<100 in 48 h) | Lean harder on noise-injection (more coverage per demo) + augmentation; consider parallel Gazebo instance if RAM/GPU allow |
| GPU shared with g1nav project | Short training runs; never kill foreign processes |
