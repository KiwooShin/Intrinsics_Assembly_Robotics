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
