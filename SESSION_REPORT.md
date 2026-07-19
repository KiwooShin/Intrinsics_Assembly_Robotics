# Autonomous Session Report — 2026-06-18

Running log of the unattended session (user away ~10h). See [Plan.md](./Plan.md) for the full plan.

## TL;DR
End-to-end imitation pipeline is **verified**: generate randomized scene → CheatCode demo → trim → train ACT-style policy → validate. Overfit confirmed; training is GPU-efficient (95% util); proper held-out validation shows few near-identical demos don't generalize → now expanding **diversity** and designing smart/efficient data generation.

## Verified results
| Check | Result |
|---|---|
| Dataset generation | 5 near-eval demos collected, all CheatCode ~93/100 |
| Trim correctness | task-window trim drops idle+reset (≈530 clean frames/ep, ends seated) |
| Overfit (val==train) | first-action err **0.0005 m/s** (~1% range), uniform across 16-step chunk → model fits data |
| Held-out (train 4 / val 1) | train 0.075 vs **val 0.34** (norm L1); val first-action **0.0024 m/s** → limited generalization |
| GPU efficiency | util **0→95%**, throughput **1,326 → 5,608 fr/s** (~4.2×) |

## Efficiency analysis (Phase 2)
- Baseline was both CPU-loader-bound (~2,600 fr/s ceiling) AND launch-overhead-bound (24.7 ms/step for a 0.75M model in fp32/bs64).
- Fixes (in `train_v2.py`): **entire dataset on GPU** (0.78 GB fp16 → no CPU loader), **bf16 autocast**, **channels_last**, **torch.compile**, **bs256**, on-GPU loss accumulation (one sync/epoch).
- Result: 95% GPU util, 5,608 fr/s. For datasets too big for VRAM later, fall back to a cached/streaming loader.

## Pipeline (scripts in repo + ~/training)
- `gen_config.py --mode {near,wide}` — randomized single-trial SFP configs (near eval, or full SFP eval distribution: all rails, both ports, wide board pose).
- `collect_one.sh` / `collect_set.sh` — CheatCode demo collection (correct scoring detection).
- `collect_convert.sh` — **storage-light**: per-demo collect → score-filter (≥60) → trim-convert → **delete bag** (never holds >1 of the ~8 GB bags).
- `prepare_dataset.py` — bag→.npy + task-window trim.
- `train_v2.py` — efficient ACT-style trainer (all-GPU, bf16, compile, proper val split).
- `verify_dataset.py`, `make_video.py`, `profile_loader.py`, `profile2.py`.

## Generalization experiment (Phase 3 result)
Collected **11 wide** demos (full SFP eval distribution: all rails, both ports, wide board pose) via the storage-light pipeline; 1 of 12 dropped (partial insert, score 65, no insertion_event — filtered correctly). Same held-out wide val set (`ds_wide/ep_8,9,10`), three training conditions:

| Train set | Val L1 (16-step chunk) | Val first-action |err| |
|---|---|---|
| NEAR only (5) | ~0.58 | 0.00527 m/s |
| WIDE (8) | ~0.37 | 0.00376 m/s |
| NEAR+WIDE (13) | ~0.30 | **0.00296 m/s** |

**Findings:**
- Diversity + count improve held-out generalization **monotonically** (first-action err −44% near→near+wide). Near-eval-only data does NOT cover the wide distribution.
- The full 16-step chunk generalizes worse than the first action (val 0.30 vs train 0.057) → the multimodal far-future a deterministic head can't capture; motivates (a) more data and (b) full LeRobot ACT with the CVAE latent + temporal ensembling.
- Pipeline scales cleanly; GPU-efficient throughout.

Final model trained on all 16 episodes → `~/training/ckpt/v2_wide.pt` (for in-sim deployment via `DeployACT`).

## ⚠️ Deployment-runtime blocker (needs a decision)
`DeployACT.py` is written and already importable (symlink-install), but **the policy cannot run in-sim yet** because of a Python/torch mismatch in this from-source setup:
- `aic_model` runs under **`/usr/bin/python3` (3.12)** = the from-source ROS Kilted runtime → has `rclpy`, **no `torch`**.
- The torch I trained with is **miniconda `python3` (3.13)** → has `torch`, but its `rclpy` is ABI-incompatible (built for 3.12).
- `pixi.toml` is the *intended* runtime (co-installs ROS + lerobot + torch), but **no pixi env is built**, and pixi would use binary ROS, not the tuned from-source Gazebo.

This also blocks the shipped `RunACT`. Options (user to choose — all touch the environment, so deferred):
1. **`pip install --user torch` (cu128 aarch64) into `/usr/bin/python3`** — smallest change, keeps the tuned from-source sim. Risk: could shadow system numpy/deps used by from-source ROS. *(recommended, with care)*
2. **`pixi install`** then run sim+policy via `pixi run` — repo-intended, isolated, but uses binary ROS (re-validate Gazebo rendering/perf).
3. **Inference sidecar**: keep `aic_model` on /usr/bin/python3; run torch inference in a miniconda process; bridge over a ROS topic/service. More code, no env risk.

## Next (once runtime resolved)
- Test `DeployACT` in-sim, then build the **failure-driven loop**: run the learned policy across configs, score with the engine, generate CheatCode demos where it fails, retrain.
- Move to **LeRobot ACT** for the competitive model (CVAE latent + temporal ensemble → better chunk/multimodal generalization), reusing this data pipeline. Note: LeRobot ACT also needs torch in the runtime (same blocker), and its `RunACT` uses a **26-D state** (pose+vel+err+joints) vs my pose-7 — align state when switching.
- Continue step-wise data expansion (next: ~40 demos) and re-measure the val curve.

## Strategy: generating additional data + training (Phase 4 design)
1. **Coverage-first, step-wise.** Diversity matters more than count with few demos. Expand 5 → ~15 → ~40, sampling the full eval distribution; measure the val curve at each step and stop when it plateaus / hits target.
2. **Storage-light by default.** Always convert-and-delete bags (done). Optional next: a live recorder node that writes the trimmed episode directly from topics, skipping the 8 GB bag entirely.
3. **Failure-driven (DAgger-style).** Wrap the trained model as an `aic_model` Policy, run it (not CheatCode) across many configs, score with the engine, find where it fails (wrong port / collision / no insertion), then generate CheatCode demos AT those failing configs and retrain. Focuses data where the policy is weak. (Building the policy-deployment wrapper also serves the final submission.)
4. **On-the-fly training loop (advanced).** Interleave a generator process (CheatCode rollouts → episodes) with training via a bounded replay buffer; discard old episodes once learned. Keeps storage flat regardless of total demos seen.

## Open decisions for the user (when back)
- Stay with the standalone ACT-lite trainer, or invest in full LeRobot ACT (CVAE latent + temporal ensemble) for better multimodal/far-future prediction?
- Target dataset size / compute budget for the first generalizing model.
- Priority: breadth (cover SFP+SC, clutter) vs depth (nail SFP trials 1–2 first).

## Experiment — SC oracle validation (2026-07-17 ~20:30, SC-VAL sub-agent)

| Experiment | Verdict | Key metric |
|---|---|---|
| SC entrance-frame retarget (floor 0.0) | ⚠️ Partial | p0 config full insertion 94.28; official trial_3 stopped 0.01 m short (65.18) |
| Floor tune −0.005 | ✅ Success | official trial_3 **94.10 full**, p0 94.07 full; contacts 0, force 0 everywhere |
| SC oracle overall | ✅ Fixed (19.07 → ~94) | CheatCode now valid for SC; 8-demo SC collection launched |

Phase-0 campaign final: **39/40 KEEPs** (97.5%), scores 92.6–94.0. Dataset:
39 Phase-0 SFP + 16 legacy + up to 8 SC incoming = ~63 episodes.

## Experiment — arm unfreeze via position-mode deployment (2026-07-12 ~17:00)

| Experiment | Verdict | Key metric |
|---|---|---|
| MODE_VELOCITY deadband hypothesis | ❌ Refuted | cmd 0.028 m/s moved the arm; clamp is 0.25 m/s, no deadband |
| Root cause: open-loop reference drift | ✅ Confirmed | aic_controller.cpp:1060 — reference never re-anchored to measured TCP |
| Receding-horizon MODE_POSITION fix | ✅ Success | eval_config total **36.1 → 119.4/300 (+230%)**; all trials directed approach |
| Insertion | ❌ Not yet | stalls 5–6 cm short: near-port view ≈ seated end-state → ~0 velocity pred |

Deployment recipe now: integrate first 4 predicted twists from measured TCP
pose into absolute targets (~18 Hz sub-stepped), MODE_POSITION with CheatCode
stiffness/damping; dt=0.275 s from episode timestamps. Next lever: DAgger
demos at near-port stall states + more diverse data (Phase-0 campaign).

## Diagnosis — CheatCode SC failure (2026-07-12 ~18:00, SC-DIAG sub-agent)

| Experiment | Verdict | Key evidence |
|---|---|---|
| CheatCode on generated SC config | ❌ Fail (19.07) | Gripper finger hit the TARGET sc_port_0 body (−24); plug stalled 2 cm out |
| Distractor-contact hypothesis | ❌ Ruled out | Contact partner was the target port, not a distractor |
| Root cause: no SC branch in CheatCode | ✅ Confirmed | Generic SFP-calibrated descent targets `{port}_link`; SC uses rotated frame + separate `_entrance` frame (offset −0.0156) CheatCode never uses |
| gen_config strata yaw U(−π,π) | 🐛 Bug found | Eval yaws cluster in {±3.1, 3.0, −1.8}; yaw≈0 boards are OOD/unreachable |

Fixes dispatched: eval-band yaw sampler + in-place config regen + campaign
relaunch (resumable, episode names unchanged); CheatCode SC retarget to the
entrance frame with shallower descent floor (sim validation post-campaign,
success = ≥85 with contacts 0). If SC stays unsolved, ceiling ≈ 200/300
(trial 3 forfeited) — SC oracle fix is the highest-value single item after
data volume.

## Analysis — proof sweep + benchmarks (2026-07-12, analysis sub-agent)

**Headline:** the ASHA sweep's −15.6% "win" (val first-action 0.00316 vs 0.00374)
is on the first-action L1 proxy that our own Finding #1 says does NOT predict
success — treat it as hyperparameter narrowing, not a validated policy gain. It
must be confirmed on the in-sim scored suite before adoption.

- **K=8 > K=16 is a selection-metric artifact for deployment.** The objective
  scores only action[0]; a shorter chunk trivially wins near-term L1. ACT
  (arXiv:2304.13705) shows rollout success peaks at a ~2 s horizon (k≈100 @ 50 Hz
  = K≈8 @ 4 Hz), and BID (arXiv:2408.17355) + arXiv:2507.09061 show longer chunks
  help open-loop stability. Decision hinges on execution mode: for action[0]-only
  re-planning (h=1) K=8 is legitimate; for open-loop chunks, prefer K=16. Settle
  by running BOTH Ks through the matched-seed suite (h=1 and h=K), ranked by task
  score, not L1.
- **All EMA trials pruned at rung 0 = decay/budget mismatch, not bad luck.** EMA
  decay 0.999 (≈1000-step horizon) with no warmup, seeded from random init, over
  ~100 steps at rung 0 retains 0.999^100 ≈ 90% of the initialization — hence
  val_L1 ≈ 0.68 (random level). Remove ema=0.999 from the broad search; re-add
  only at final full-budget training with warmup/bias-correction or a step-matched
  decay ~0.99 (EMA dynamics: arXiv:2411.18704).
- **lr 1e-3 > 3e-4 is the un-applied linear-scaling correction** (bs64→bs256 in
  Phase 2 never re-tuned lr; Goyal et al. arXiv:1706.02677 predicts ~1.2e-3). But
  no 3e-4 config survived to 54 epochs, so ASHA's known fast-early-learner bias
  (arXiv:1810.05934, 1603.06560) may have culled a competitive slow starter —
  confirm with a full-budget, no-early-stop head-to-head including 3e-4.
- **eager-bf16 (0.271 ms) slower than eager-fp32 (0.213 ms) at bs1/0.75M is
  expected** (overhead-bound; autocast casts cost more than unrealized Tensor-Core
  savings). Ship compiled-bf16 (0.112 ms) for inference; never eager-bf16.
  torch.compile gives +16–22% train throughput. Inference has ~900× headroom
  under 4 Hz — cheap enough for temporal ensembling / receding-horizon h=1.
- **Cross-session comparability warning:** today's 0.00316 (8 WIDE-only eps, K8,
  lr1e-3, 54 ep) is NOT comparable to June's 0.00296 (13 NEAR+WIDE eps, K16,
  lr3e-4, 60 ep) — different train set, K, lr, budget. Going forward, track ONE
  canonical metric: in-sim suite score (primary; rliable IQM+CI, arXiv:2108.13264),
  pinned train-manifest hash, fixed budget, both K's, mean of 3 seeds; val-L1 is a
  secondary diagnostic only.

## 2026-07-18 Batch smoke60 — analysis

Analysis sub-agent review of the first fully-valid batch through the hardened
harness (completed 17:01, `eval_suite_smoke60`, 15 configs = 12 stratified +
3 official, **internal fast protocol time_limit=60 sim-s**, NOT the official
180 s). Canonical metric per CLAUDE.md §6 = matched-seed in-sim eval-suite
score (IQM + bootstrap CI); val-L1 is secondary only. All three checkpoints
share the same 15 matched-seed configs.

### Summary table (one row per checkpoint)

| Checkpoint (topic) | Verdict | Insertions | Mean [95% CI] | IQM [95% CI] | Outcomes (miss/coll/prox) | Off-limit collisions | Officials sum /300 |
|---|---|---|---|---|---|---|---|
| **p1_k16** — new retrain, K=16, 60-ep set (incl 5 SC); best point estimate | FAIL (approach-only; best of 3) | 0/15 | **6.39** [-0.22, 13.27] | **3.76** [-1.67, 11.39] | 8 / 4 / 3 | **2** (cfg_002, cfg_007) | 78.6 |
| **p1_k8** — new retrain, K=8, same set | FAIL (approach-only) | 0/15 | 5.35 [0.05, 11.41] | 2.85 [1.0, 9.64] | 10 / 2 / 3 | **1** (cfg_007) | 75.1 |
| **v2_wide** — prior best, known 119.4/300 on official 180 s | FAIL @60s / but known-inserting @180s | 0/15 | 2.78 [-2.01, 7.58] | 1.00 [-1.67, 4.38] | 10 / 3 / 2 | **2** (cfg_002, cfg_009) | 71.3 (was 119.4 @180s) |

Paired bootstraps (all **inconclusive**, CIs include 0): k16−v2 mean +3.61
[-3.63, 13.83], IQM 0.00 [-0.72, 3.78]; k16−k8 mean +1.04 [-5.56, 8.46], IQM
-0.16 [-4.08, 4.71]; k8−v2 mean +2.57 [-4.54, 9.68], IQM +1.19 [-4.71, 10.17].
Point-estimate order k16 > k8 > v2; **none statistically separable.** 8/15
configs floor at exactly +1.0 on all three checkpoints.

### Score-band decode (what the numbers mean)

Total score at 60 s collapses to three discrete bands, none of which is
insertion: **collision** (total −23 = tier-1 +1 with a −24 off-limit contact
penalty; −11 = stratum mean of one −23 + one +1), **miss floor** (+1.0 =
tier-1 "Model validation succeeded" only; tier-2 and tier-3 both 0 because the
plug never entered the max bounding radius), **proximity** (+7 … +40 = plug
reached bounding radius, earned duration/efficiency/smoothness bonuses but
never seated). Tier-3 reads "Distance computation failed, tf between cable and
port not found" on every floored trial. **0/45 insertions across the entire
batch.**

### (a) Insertion-floor effect — the fast protocol currently measures approach quality only

`suite_meta.yaml` states the design assumption: "insertions occur well under
30 sim-s, so tier-3 signal is preserved" by capping at 60 s. **That assumption
is false for the current policies.** With 0/45 insertions, tier-3 (the bulk of
the 100-pt scale) never fires; the metric degenerates to a proxy for "does the
arm approach the port cleanly without an off-limit contact." Consequence: the
60 s suite has almost no discriminating power right now — 9/15 configs floor
identically and every paired bootstrap is inconclusive. IQM in particular is
pinned near the +1.0 floor (its middle-mass is all misses), so it cannot rank
these checkpoints; the mean separates them only through the 3 official tails,
with CIs that span 0.

**Critical caveat vs §6:** the 60 s screen is a legitimate cheap
regression/screening metric but MUST NOT be the sole adoption gate, and it is
actively **unfair to slower-but-inserting policies.** Direct evidence:
v2_wide scored **119.4/300 on the official 180 s eval** but only **71.3/300**
on the same-family officials here at 60 s — a ~48-pt truncation, with
official_1 falling from a scoring result to the +1.0 floor. v2's successful
insertions evidently complete **between 60 and 180 sim-s**, so the 60 s cutoff
truncates its insertion phase. The point-estimate ranking (k16 > k8 > v2)
therefore may **invert** at full budget: the p1 retrains lead on *approach*,
but v2 is the only checkpoint with demonstrated *insertion* capability. Per §6
("primary = matched-seed suite IQM+CI … prefer mean of 3 seeds"), adoption
requires the 180 s protocol where insertion can register.

### (b) Officials-vs-strata gap = a yaw/pose *coverage* gap, not officials being OOD-hard

Officials average ~26 pts (k16 26.2, k8 25.0, v2 23.8); the 12 stratified
configs average ~1.4 pts (k16). The gap is the **reverse** of "officials are
harder OOD." Mapping totals onto `manifest.csv` board_yaw:

- **Floored (+1.0 miss) cells** cluster at **moderate |yaw| ≈ 0.6–1.5**:
  cfg_000 (−1.49), cfg_001 (0.84), cfg_003 (0.63), cfg_004 (−1.45),
  cfg_006 (−0.76), cfg_008 (−1.23), cfg_010 (−0.63), cfg_011 (0.91).
- **Scoring (proximity) cells + all officials** sit at **extreme |yaw| ≈ 1.3–3.1**:
  cfg_005 (2.67), cfg_009 (1.29), official_1 (3.10), official_2 (−3.10),
  official_3 (−1.80).

The boundary is ~|yaw| ≈ 1.3: below it the policy never initiates a correct
approach (miss); above it it reaches proximity. This exactly confirms the
prior SESSION_REPORT finding ("gen_config strata yaw U(−π,π) BUG: eval yaws
cluster in {±3.1, 3.0, −1.8}; yaw≈0 boards are OOD/unreachable"). The 60-ep
demonstration set covers the extreme-yaw canonical modes (where officials
live) and undersamples the moderate-yaw band the stratified suite probes.
Orthogonally, **rail0 fails uniformly** — all four rail0 configs floor or
collide on all three checkpoints (0 positive scores), independent of yaw.
Distractors are present in BOTH stratified and official configs (6+
entity_present per config), so distractors are not the differentiator; pose
coverage is. This is a textbook behavior-cloning support/coverage failure:
compounding error off the demonstrated manifold (DAgger, Ross et al. 2011,
arXiv:1011.0686), and ACT/ALOHA's dependence on dense teleop coverage of the
target distribution (Zhao et al. 2023, arXiv:2304.13705).

### (c) Collision asymmetry (k16 4 vs k8 2) — mostly an artifact of k16 reaching more ports; genuine off-limit gap is +1 and within noise

`outcome_counts` "collision" flags **any** contact event, conflating two
different things. Separating them:

- **Genuine off-limit collisions** (tool_link/gripper into a mount housing →
  −24 penalty → −23 total): **k16 = 2** (cfg_002 nic_card_mount_0, cfg_007
  nic_card_mount_2), **k8 = 1** (cfg_007), **v2 = 2** (cfg_002, cfg_009).
- **Near-port glancing contacts during a scoring proximity approach**
  (still positive total): k16's other two "collisions" are cfg_005 (+15.27,
  gripper-finger graze) and official_1 (+11.60, tool_link touch on the card
  link, not the off-limit housing). These occur **because k16 reaches
  proximity on more configs** (e.g. cfg_009 +40, official_1 +12 where k8
  floors at +1) — more approaches mechanically create more grazing
  opportunities near the port.

So the headline 4-vs-2 overstates the effect. The genuine off-limit delta is
k16 2 vs k8 1 — one config (n=1 seed, 15 configs) — and the per-config picture
is non-monotone (cfg_002: k16+v2 collide, k8 misses; cfg_007: k16+k8 collide,
v2 misses; cfg_009: v2 collides, k16 gets +40). The longer K=16 open-loop
horizon (16 actions between observations vs 8) is a **plausible** mechanism for
reduced reactivity → overshoot into contact, consistent with the chunking
trade-off (larger chunk lowers re-planning/compounding error but raises
open-loop drift; ACT arXiv:2304.13705; BID arXiv:2408.17355; arXiv:2507.09061),
but the data **cannot adjudicate it** at this sample size. Verdict: collision
asymmetry not established.

### (d) Harness-health checks — both prior failure modes are ABSENT (positive validation of the hardened harness)

- **DeployACT startup latency / first-trial cfg_000 GetState-teardown hang: NOT
  reproduced.** `/aic_model/get_state` is available promptly and lifecycle
  `configure` takes a **uniform ~6 s on every trial** (cfg_000, cfg_001,
  official_3 all ≈6 s: configured ~6 s after "Configuring…"). The previously
  documented first-trial hang does not appear; startup is not a confound in
  this batch. Per-trial wall time is ~5.8–6.5 min (≈300–390 s of active rollout
  for 60 sim-s, i.e. ~5× real-time), stable across the run.
- **p1 joint-bound spikes: NOT observed.** `JointSaturationLimiter` is
  configured for all arm joints (wrist ±6.283, elbow ±3.142) and gripper, but
  there are **no saturation/out-of-range/violation events** in any run.log —
  only the limiter setup INFO lines. Note: because the saturation limiter
  clamps silently, open-loop overshoot would manifest as the arm stalling at a
  clamped pose rather than a logged spike, which is consistent with the miss
  floor (arm drives toward an undertrained target and stops short).

### (e) Literature alignment

- **Coverage / OOD floor (primary root cause):** DAgger (Ross, Gordon, Bagnell,
  "A Reduction of Imitation Learning…", arXiv:1011.0686) — BC error compounds
  off the demonstrated support; the fix is targeted data aggregation at the
  states the learner actually visits. HG-DAgger (Kelly et al., arXiv:1810.02890)
  for human-gated aggregation on failure states. ACT/ALOHA (Zhao et al.,
  arXiv:2304.13705) and Diffusion Policy (Chi et al., arXiv:2303.04137) both
  depend on dense demonstration coverage of the deployment distribution — 60
  episodes over a U(−π,π)×wide-x/y domain is far too sparse, matching the
  moderate-yaw floor we observe.
- **Chunk length / open-loop horizon (collision hypothesis):** ACT action
  chunking (arXiv:2304.13705) trades compounding error against open-loop drift;
  BID / bidirectional decoding (arXiv:2408.17355) and arXiv:2507.09061 show
  longer chunks aid open-loop stability but reduce reactivity — the qualitative
  frame for k16's marginally higher off-limit contact count, though our n=1 can't
  confirm it.
- **Metric methodology:** rliable IQM + stratified bootstrap CIs (Agarwal et al.,
  arXiv:2108.13264) — exactly why we report IQM+CI; the batch shows why n=1 seed
  with a floored middle-mass yields inconclusive CIs.

### Recommendations for the remaining ~24 h

**1. Run the 180 s full-budget eval overnight — but as a matched-seed
HEAD-TO-HEAD, not p1_k16 alone.** Include **p1_k16 AND v2_wide** (add p1_k8 if
budget allows). Rationale: the 60 s screen truncated v2's known insertions
(119.4→71.3 on officials); comparing a 180 s p1_k16 against a 60 s-crippled v2
would be invalid. Only at 180 s can tier-3 insertion register and settle whether
p1_k16's approach lead converts to actual insertions or whether v2's insertion
capability wins. Budget: ~15 min/trial × 15 = ~3.75 h/checkpoint → k16+v2 ≈
7.5 h, +k8 ≈ 11 h — fits comfortably overnight. Pin the manifest hash, config
(K, epoch budget), and seed set on every row (§6). **Launch as a detached,
resumable script with a DONE marker per §6 agent-waiter ban.**

**2. Phase-2 failure-driven collection targets the floored cells (DAgger-style),
prioritized:**
   - **P1 — rail0 (highest):** all four rail0 cells fail on every checkpoint.
     Collect both plugs (SFP+SC) × both ports at rail0. **~16–20 demos.**
   - **P2 — moderate-yaw band |yaw| ∈ [0.5, 1.5], port0 emphasis, rails 1–2:**
     the undersampled band that floors (cfg_004, 006, 008, 010, 011).
     **~16–20 demos.**
   - **P3 — the 2 off-limit-collision cells** cfg_002 (rail0_SC, yaw 1.32) and
     cfg_007 (rail1_SC, yaw 2.84): clean contact-free approach demos to teach
     avoidance. **~6–8 demos.**
   - **Method:** seed each demo at the exact failed board pose from
     `manifest.csv` (replay cfg_000/003/004/006/008/010/011 + cfg_002/007
     poses) ± a small neighborhood (±0.05 rad yaw, ±1 cm x/y, grasp_z within
     the [0.040, 0.046] band), ~4–6 demos/cell. **Total ≈ 40–48 demos** roughly
     doubling effective coverage in the failed region. Then retrain (both K=8
     and K=16 per §6), re-screen on the 60 s suite for regression, and 180 s
     eval the winner.

**3. Keep the 60 s suite as a cheap screening/regression gate only** (it detects
gross approach regressions fast), but record adoption decisions on the 180 s
matched-seed IQM+CI, mean of ≥3 seeds where feasible (§6). Do NOT adopt on the
60 s point estimates — they rank approach quality, not task success.

Note: 0/45 insertions means neither p1 retrain has yet matched v2's known
insertion capability; the SC branch and the moderate-yaw coverage hole remain
the two highest-value fixes (consistent with the prior SC-oracle and yaw-bug
findings in this report).

## 2026-07-18 Phase-2 collection — final tally + SC oracle follow-up

**Campaign:** 40 failure-driven configs (seed 20260718, all distractor + eval-band
yaw) → **33 KEEP / 7 DROP** (82.5%), 17:20–22:28, zero harness incidents,
resumable driver, bags deleted per storage rule. Dataset now **77 episodes**
(ds_phase0 44 + ds_phase2 33; 69 SFP + 8 SC), all with wrench/joints.

| Cell group | Attempted | KEEP | Drops (score, cause) |
|---|---|---|---|
| SFP (port_0 boost + port_1) | 32 | 30 | 56.2, 57.7 (near-threshold oracle misses) |
| SC rail0 | 4 | 1 (93.8) | 65.0 ins=0 (floor partial), 28.1, 42.9 (poor runs) |
| SC rail1 | 4 | 2 (94.1, 94.0) | 62.8 ins=0 (floor partial), 58.6 (near-threshold) |

**SC oracle follow-up (TOP priority for any future session):** under eval-band
yaw + distractors the SC CheatCode keep-rate fell to 3/8 (vs 5/8 in Phase-0
band). Two distinct failure modes: (a) partial-insert at the −0.005 descent
floor with no insertion event (62.8/65.0) — micro-tune the floor deeper
(≈−0.007) and re-validate zero-contact; (b) outright poor approaches on some
eval-band SC poses (28.1/42.9/58.6) — the entrance-frame retarget needs a
pose-conditioned approach waypoint. Neither blocks tonight's retrain; SFP
coverage of the floored strata (the eval batch's main gap) is complete.
