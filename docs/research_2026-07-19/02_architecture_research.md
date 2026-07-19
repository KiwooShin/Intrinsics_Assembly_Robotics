# Policy Architectures for mm-Precision RGB Insertion — Research Brief for the Intrinsic AIC Stack

Scope: what the 2023–2026 literature actually demonstrates for mm/sub-cm insertion from RGB, mapped onto THIS stack's specific failure (a deterministic ACT head that mode-averages to zero velocity in the last inch → 0/45 insertions on the hard suite) and constraints (RGB-only, ~0.75M-param CNN, single GB10, 93 demos expandable, wrench recorded-but-unused, sim-only with a ground-truth engine reward).

---

## 0. The single most important framing fact

Your root cause — *"near the port the observation ≈ the seated/terminal demo frames, whose oracle velocity is zero, so an L1/L2-regressed head returns the mean ≈ 0 → no motion → stall at 0.05–0.08 m"* — is **the textbook motivation for every expressive-head architecture below**. The literature is unambiguous: a mean-regression (deterministic L1/L2) BC head *"averages across modes and produces an action in-between two valid behaviors, which is typically invalid itself,"* whereas CVAE, diffusion, and flow-matching heads *"commit to one mode within each rollout"* and can push through instead of averaging to a stop ([Diffusion Policy](https://arxiv.org/abs/2303.04137), Chi et al.). **This means your highest-leverage move is not more data or a bigger backbone — it is replacing the deterministic head with a multimodal one, on the same RGB encoder.** Everything below is ranked against that.

Second framing fact, from FMB (below): on real connector insertion, **force/torque is decisive** — RGB+D+τ scored 11/25 vs RGB-only 2/25. You *already record* `/fts_broadcaster/wrench` but the policy ignores it. That is a second near-free lever.

Third: you have **no sim-to-real gap** (oracle teacher in the same Gazebo the engine scores) and a **ground-truth sparse reward** (the insertion event). That makes sim residual-RL methods — usually gated by transfer risk — unusually well-matched here.

---

## 1. ACT with CVAE head + temporal ensembling (as originally intended)

**Paper:** *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ALOHA/ACT), Zhao et al. 2023 — arXiv:2304.13705 — https://arxiv.org/abs/2304.13705

- **What it is / why it matters here:** ACT is exactly your architecture *minus the two pieces you dropped*: a **CVAE latent `z`** (encoder compresses the demo action chunk + joints into `z`; at test time `z=0`/prior) so the decoder can represent a *distribution* over chunks rather than the mean, plus **temporal ensembling** (exp-weighted blend of overlapping chunks). Your own notes (arXiv:2304.13705 motivation) already identify the deterministic head as the mode-averaging culprit.
- **Reported precision results:** ~80–90% on fine bimanual tasks (thread-in-slot, battery insertion, cup-into-slot) from ~50 demos/task at 50 Hz; the *ablation* shows removing the CVAE collapses precise-task success (their "no-CVAE" row drops sharply, esp. on tasks with human/teacher timing variability — your exact situation).
- **Data / compute:** ~50 demos/task; trains in tens of minutes on one GPU; ~80M params typical (yours at 0.75M is far smaller — fine for a single narrow task, but consider widening the decoder). Inference trivially <4 Hz on GB10.
- **Verdict for THIS stack — ADOPT the CVAE, treat temporal ensembling as already-closed.** You have already *empirically closed* temporal ensembling on the deterministic head: n=3/arm, 30 trials, OFF 23.1±2.1 vs ON 20.1±4.2, added variance + a new miss. That result is *expected* — TE was designed to smooth an *already-multimodal* CVAE decoder, not to rescue a mean head; blending stale mean-chunks just injects noise. **Do not revisit TE until the head is multimodal.** The CVAE latent itself is the un-tested half and is the cheapest direct hit on your root cause. Risk: CVAE alone is a *weaker* multimodal model than diffusion/flow (a single Gaussian latent); if the near-port distribution is sharply bimodal (push vs stop), a diffusion/flow head (Sec. 2/3) represents it better. Recommendation: implement the CVAE as the low-risk fallback, but prefer a diffusion/flow action head as primary.

Supporting chunk-length evidence (already in your notes): BID *Bidirectional Decoding* (arXiv:2408.17355) and arXiv:2507.09061 on chunk horizon; keep K=16 for open-loop stability with the receding-horizon h=4 deploy.

---

## 2. Diffusion Policy and its insertion results

**Paper:** *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Chi et al. 2023 — arXiv:2303.04137 — https://arxiv.org/abs/2303.04137 · project: https://diffusion-policy.cs.columbia.edu/

- **Why it is the strongest match to your failure:** DP models the *conditional action distribution* as a denoising process; it *"learns multi-modal behavior and commits to only one mode within each rollout … sampling from the full range of demonstrated behaviors without averaging them away,"* and shows **robust recovery** (when it lands in a slightly wrong intermediate state it can push out of it because it samples a broad distribution rather than following a deterministic path). That is precisely the "escape the zero-velocity fixed point" behavior you need.
- **Insertion-class results:** DP dominates RoboMimic/PushT and is the de-facto BC head for contact-rich work. On real precise insertion it is the *base policy* that residual-RL papers build on (ResiP, Sec. 8) and that force-conditioned variants extend (Sec. 6). Two-stage "reach (diffusion) → insert (RL/impedance)" frameworks report **100% on polygon pegs and ~90% on Ethernet/waterproof connectors** with ±2 mm hole uncertainty ([Diffusion-Based Impedance Learning](https://arxiv.org/abs/2509.19696); two-stage reach-to-insert works). Caveat from FMB: a *vanilla* ResNet diffusion policy scored 0/10 on FMB's hardest multi-stage tasks — DP is not magic on long-horizon/low-data; the head helps precision/multimodality, not coverage.
- **Data / compute:** comparable to ACT (tens–hundreds of demos); a CNN-UNet or transformer denoiser; 100-step DDPM inference is the cost, but DDIM/consistency distillation ([Consistency Policy](https://arxiv.org/abs/2405.07503)) gets it to 1–3 steps — at your 4 Hz control rate even 10–20 DDIM steps fit comfortably on GB10 (you have ~900× inference headroom under 4 Hz).
- **Verdict for THIS stack — TOP CANDIDATE for the head swap.** Keep your 3×128px CNN encoder + 7D TCP state; replace the deterministic twist regressor with a **diffusion (or flow, Sec. 3) action head** predicting the K×6 twist chunk. RGB-only, ~same param budget class, single-GPU, inference well within 4 Hz. This is the most direct architectural fix for the mode-averaging attractor and is well-trodden for insertion. Pair with your planned last-inch DAgger (arXiv:1011.0686) at the observed 0.05–0.08 m stall states to give the head a non-zero closing-velocity mode to sample.

3D variant note: **3D Diffusion Policy (DP3)** (arXiv:2403.03954) is more sample-efficient but needs a point cloud — you have RGB-only, no depth bridged — so DP3 is out unless you lift RGB→pseudo-3D (Sec. 5).

---

## 3. Flow-matching policies

**Papers:** π0 (Sec. 4) uses a flow-matching action expert; SmolVLA (Sec. 4) uses a flow-matching expert; general: flow matching = a straighter, fewer-step alternative to diffusion for the same multimodal action-distribution modeling.

- **Why relevant:** Flow matching gives the *same* multimodal-commit property as diffusion with **1–10 integration steps** instead of 50–100 denoising steps — cheaper inference, smoother chunks, and it is the head choice in the newest VLAs (π0, π0.5, SmolVLA, GR00T's DiT is diffusion-flavored). ForceFlow (arXiv:2605.11048, "Learning to Feel and Act via Contact-Driven Flow Matching") and RFS (residual flow steering, arXiv:2602.01789) show flow heads are actively used for contact-rich/insertion.
- **Reported results:** as the head inside π0 (57.6%→97.6% after RL fine-tune) and SmolVLA (78.3% real avg on SO-100); as a standalone small head it matches diffusion on RoboMimic-class tasks at lower inference cost.
- **Verdict for THIS stack — EQUAL-BEST head choice with diffusion, slightly preferred on compute.** A small **flow-matching action expert** on your existing encoder is the cleanest, cheapest expressive head: fewer sampling steps than diffusion, native to the LeRobot code you already reference, and identical benefit against mode-averaging. If you implement one expressive head, make it flow-matching or diffusion; CVAE is the fallback.

---

## 4. Pretrained VLA fine-tuning — π0 / OpenVLA / SmolVLA / GR00T-N1 (feasibility on ONE GB10, 93 eps)

**GB10 reality check (DGX Spark):** 128 GB unified LPDDR5X, **273 GB/s** bandwidth, ~31 TFLOPS FP32 / ~500 TFLOPS dense NVFP4 (1 PFLOP only with 2:4 sparsity) — https://www.nvidia.com/en-us/products/workstations/dgx-spark/. **Implication:** memory is *abundant* (any 7B model fits, even full FT); **compute/bandwidth is the bottleneck**, and it is *shared with Gazebo* (your note: heavy GPU training drops sim to ~0.05× RTF). So the constraint is not "does it fit" but "training is slow and contends with collection/eval — serialize."

| Model | Params | arXiv | Single-GB10 fine-tune | Demos to ~80% | Fit verdict |
|---|---|---|---|---|---|
| **SmolVLA** | 450M (100M flow-expert) | [2506.01844](https://arxiv.org/abs/2506.01844) | **Yes — trains on ONE consumer GPU** | ~50–200; 51.7→78.3% real w/ community pretrain | **Best VLA fit.** Small, flow-matching head, LeRobot-native, RGB, fast enough to co-exist with sim. |
| **π0 / π0.5** (openpi) | ~3.3B | [2410.24164](https://arxiv.org/abs/2410.24164); πRL [2510.09976](https://arxiv.org/abs/2510.09976) | **Yes — LoRA needs >22.5 GB (fits 128 GB easily)**; full FT >70 GB also fits | ~10 hr robot data pretrain prior; LoRA on ~100+ demos | Feasible but heavy; JAX/openpi stack; slow train on GB10; contends hard w/ sim. |
| **OpenVLA** | 7B | [2406.09246](https://arxiv.org/abs/2406.09246) | LoRA (r32) ~72 GB w/ bs16 (fits 128 GB); 10–15 hr on one A100 → **longer on GB10** | 100–500; strong at 100 on WidowX | Feasible memory-wise; **discrete-token autoregressive head is slower per step**, and 7B is overkill for one narrow sim insertion. |
| **GR00T N1 / N1.5 / N1.7** | 2.2B (1.34B VLM Eagle-2 + DiT head) | [2503.14734](https://arxiv.org/abs/2503.14734) | Fine-tune scripts + HF ckpts; DiT 16-action chunk 63.9 ms on L40 → GB10 slower but <4 Hz ok | Improved data-efficiency (N1.5); LeRobot SO-101 tutorial exists | Feasible; humanoid-oriented; DiT action head already diffusion-style. Overkill unless you want the visual prior. |

**Feasibility verdict for VLAs on THIS task:** *All are runnable on GB10's 128 GB.* The real questions are (a) does a web-scale prior help **sub-cm insertion precision** — the literature says pretraining buys **visual/semantic generalization and ~2× data efficiency**, not fundamentally better last-inch precision (that comes from the expressive head + closed-loop correction, which a small model also has); and (b) GPU contention — a multi-hour π0/OpenVLA LoRA run starves Gazebo, throttling your collection/eval loop.

- **Recommended VLA if you go this route: SmolVLA (450M).** It *is* a flow-matching-headed policy on a pretrained VLM, trainable on one GPU, RGB, LeRobot-native — you get the expressive head (Sec. 3) *and* a visual prior in one package, at a size that co-exists with the sim. π0-via-openpi-LoRA is the next step up if SmolVLA underfits your visual diversity. **Deprioritize OpenVLA (7B, discrete-token, slow) and GR00T (humanoid-scale) for this single narrow sim task.**
- **Data:** 93 eps is at the *low end* for VLA fine-tuning (they like 100–500). Your DAgger/failure-driven expansion to ~150–250 eps would put SmolVLA/π0 LoRA in a comfortable range.
- **Key caveat:** a VLA's benefit over a well-built small flow/diffusion policy on *one* task, *one* embodiment, *RGB in sim* is unproven-to-marginal; the bottleneck here is precision+multimodality+contact, not semantics. **Rank VLAs below the head-swap and force-conditioning moves.**

---

## 5. Point-cloud-free 3D approaches usable with RGB-only

You have RGB-only, no depth topics bridged — so DP3/point-cloud methods are out, but "lift 2D→pseudo-3D" is directly applicable and could sharpen the depth-along-insertion-axis perception that the last inch needs.

- **Lift3D** — *Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation*, arXiv:2411.18623 — https://arxiv.org/html/2411.18623 — lifts 2D foundation features to 3D without a depth sensor; beats RVT-2-class baselines on RLBench.
- **NoReal3D** — *No Need for Real 3D: Fusing 2D Vision with Pseudo-3D Representations*, arXiv:2509.16532 — plug-and-play pseudo-3D from **monocular RGB**, no auxiliary inputs; first 2D→3D lift with no depth.
- **VO-DP** — *Vision-Only Diffusion Policy* (VGGT + DINOv2 semantic + geometry features), arXiv:2510.15530 — single-view RGB, compresses semantic+geometric features into a diffusion policy; explicitly targets the "point clouds are noisy/fragile" problem.
- **StereoPolicy**, arXiv:2605.09989 — if you can render a second camera view in Gazebo, stereo gives metric depth cheaply.

**Verdict — SECONDARY but cheap partial win.** You don't need a full 3D policy; the highest-value slice is a **stronger pretrained visual encoder (DINOv2/VGGT features)** replacing/augmenting your 128px CNN, which improves near-port depth discrimination (how far to push) with RGB-only. This is orthogonal to and stackable with the head swap. Rendering a stereo pair in sim to recover metric depth is a legitimate alternative to bridging a depth topic. Lower priority than Sec. 2/3/6 because your failure is a *head/behavior* failure, not primarily a perception failure (approaches are clean; the arm reaches 5–8 cm then stalls).

---

## 6. Force/wrench-conditioned policies for contact-rich insertion

**The load-bearing datapoint — FMB:** *A Functional Manipulation Benchmark*, Luo et al. 2024 — arXiv:2401.08553 — https://arxiv.org/html/2401.08553 — on real **insertion**, input modality is decisive: **RGB+D+τ = 11/25 vs RGB-only = 2/25.** Their BC baselines (ResNet-34 / Transformer) take vision **and force/torque**. **You record `/fts_broadcaster/wrench` in every dataset and throw it away.** This is the second-cheapest high-leverage change after the head swap.

Recent force-conditioned architectures (all 2025–2026):
- **PhaForce** — *Phase-Scheduled Visual-Force Policy … Slow Planning + Fast Correction*, arXiv:2603.08342 — decouples a slow visual planner from a fast force-correction loop; directly models the "approach then seat" phase structure you have (tier_2/tier_3).
- **FILIC** — *Dual-Loop Force-Guided Imitation Learning with Impedance Torque Control*, arXiv:2509.17053 — force-guided IL + impedance inner loop for contact-rich insertion.
- **Tactile/Force-Conditioned Diffusion Policy** — arXiv:2510.13324 — diffusion policy over 6D wrench for force-aware manipulation; FARM-style tactile→force conditioning.
- **FORGE** — *Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty*, arXiv:2408.04587 — extends IndustReal-style RL with force sensing + a force threshold, exactly the "avoid excessive force" the base IndustReal lacked.
- **ForceFlow** — arXiv:2605.11048 — contact-driven flow matching (force + flow head combined).

**Verdict for THIS stack — ADOPT force conditioning (rank #2 move).** Concrete, low-risk, uses existing data: (a) concatenate the recorded 6D wrench into your policy's state vector (7D→13D), and (b) if you want the FMB gain fully, adopt a **phase/impedance structure** (PhaForce/FILIC) so the last-inch seating is force-regulated rather than pure position. This targets tier_2 (force/contact) *and* the seating that tier_3 rewards, and it is the empirically single most impactful modality change for insertion in the literature. Bonus: force feedback gives the policy a *non-visual* signal that the plug is "not yet seated," which directly counters the "near-port view ≈ seated view → zero velocity" ambiguity that vision alone cannot disambiguate.

---

## 7. Connector / cable / peg insertion benchmarks (the closest analogues)

| System | arXiv | Task | Reported success | Modality | Relevance to you |
|---|---|---|---|---|---|
| **IndustReal** | [2305.17110](https://arxiv.org/abs/2305.17110) | Peg insert, gear mesh, NIST-style | 83–99% over 600 trials sim→real; peg 80/88%, gears 97.5/100%, connectors 100% | Proprio + goal (**no vision, no force in policy**); SAPU + SDF reward + curriculum | Proves sim-trained RL insertion works to ~mm; SBC curriculum (start near goal, recede) is directly portable to your engine reward. No sim-to-real gap for you = even easier. |
| **AutoMate** | [2407.08028](https://arxiv.org/abs/2407.08028) | 100-assembly dataset, specialist+generalist | Sim ~80%+ (80 assemblies); real specialist 90%, generalist 84.5–86% over 20 assemblies | Sim RL + sim-to-real | Shows a *generalist* insertion policy over diverse geometries — your SFP+SC dual-plug generalization problem. |
| **FMB** | [2401.08553](https://arxiv.org/abs/2401.08553) | Functional multi-stage + insertion | Insertion RGB+D+τ 11/25 vs RGB-only 2/25; vanilla diffusion 0/10 on hardest | Vision + **force/torque** | The force-is-decisive datapoint; also the "coverage/long-horizon still hard" warning. |
| **HIL-SERL** | [2410.21845](https://arxiv.org/abs/2410.21845) (Science Robotics) | RAM insertion, USB, assembly, dual-arm | **~100%** within 1–2.5 hr real training; near-perfect + fast cycle | Vision (pixel) + human corrections, RL | The strongest real insertion result anywhere; your analogue = engine reward replaces the human/classifier reward. |
| **RLDG** | [2412.09858](https://arxiv.org/abs/2412.09858) | Connector insertion, precise assembly | RL-generated data → generalist +30–40% over human demos | Vision, HIL-SERL backbone | Distill RL insertion into your BC policy; RL data > human demos for precision. |
| **ResiP** | [2407.16677](https://arxiv.org/abs/2407.16677) | Precise **visual** multi-part assembly | Residual RL substantially beats frozen BC on high-precision assembly, **from RGB** | **RGB** + frozen diffusion BC + residual RL, sparse reward | *Your exact setup*: frozen chunked BC (diffusion) + closed-loop residual RL from RGB, sparse reward = your engine insertion event. |
| **EasyInsert** | [2505.16187](https://arxiv.org/abs/2505.16187) | Data-efficient generalizable insertion | Data-efficient, generalizes across connectors | — | Worth reading for the low-demo regime you're in. |
| **SERL** | [2401.16013](https://arxiv.org/abs/2401.16013) | PCB/connector insertion | Near-100% sample-efficient real RL | Vision + force | HIL-SERL's predecessor; PCB connector insertion specifically. |

---

## 8. Ranked top-3 architecture moves toward >90 avg

Your target (>90/100 avg ≈ oracle) requires **reliable full insertions almost everywhere**, which means solving the last-inch attractor *and* the moderate-yaw/rail0 coverage holes. Data expansion (DAgger at stall states, rail0/moderate-yaw coverage) is assumed in parallel — these are the *architecture* moves.

### #1 — Swap the deterministic head for an expressive multimodal action head (flow-matching or diffusion), on the existing RGB encoder
- **Why:** directly kills the mode-averaging fixed point that produces 0/45 insertions — the failure the whole literature says this head class exists to fix ([Diffusion Policy 2303.04137](https://arxiv.org/abs/2303.04137); [ACT/CVAE 2304.13705](https://arxiv.org/abs/2304.13705)). Flow-matching (Sec. 3) preferred for cheap 1–10-step inference; CVAE is the low-risk fallback; keep RGB-only, keep ~0.75M-class encoder, keep receding-horizon K=16/h=4.
- **Compute/fit:** trains in your existing ~17-min budget class; inference << 4 Hz on GB10; no new sensors.
- **Expected impact:** unlocks *nonzero* insertions (the binary 0→>0 flip your run never achieved); expect the last-inch stalls that currently score 34–43 to start seating. This is the necessary precondition for >90 — and the only move that also *retroactively re-enables* temporal ensembling (which you correctly shelved for the mean head).
- **Pair with:** last-inch DAgger at 0.05–0.08 m stall states ([1011.0686](https://arxiv.org/abs/1011.0686)/[1810.02890](https://arxiv.org/abs/1810.02890)) so the multimodal head has a non-zero-velocity "push-in" mode to sample.

### #2 — Add force/wrench conditioning (+ phase/impedance structure) using data you already record
- **Why:** FMB's insertion result (RGB+D+τ 11/25 vs RGB-only 2/25) is the strongest single-modality effect in the insertion literature, and your wrench is recorded-but-unused. Force disambiguates "near-port-but-not-seated" (which vision cannot, and which *causes* your zero-velocity ambiguity), and regulates tier_2 contact penalties. Architectures: [PhaForce 2603.08342](https://arxiv.org/pdf/2603.08342), [FILIC 2509.17053](https://arxiv.org/abs/2509.17053), [Force-Conditioned Diffusion 2510.13324](https://arxiv.org/html/2510.13324v1), [FORGE 2408.04587](https://arxiv.org/html/2408.04587).
- **Compute/fit:** minimal — concatenate 6D wrench into state (7→13D) for a fast win; add an impedance/force inner loop (you already deploy CheatCode stiffness/damping in MODE_POSITION) for the full gain.
- **Expected impact:** targets tier_2 *and* the seating that gates tier_3; also cuts the SC −23 wrist-into-mount collisions (force-reactive stop). Stackable with #1.

### #3 — Residual RL in sim on the frozen expressive BC policy, rewarded by the engine insertion event
- **Why:** you have the two ingredients residual-RL insertion papers need and usually lack — **no sim-to-real gap** and a **ground-truth sparse reward**. [ResiP 2407.16677](https://arxiv.org/abs/2407.16677) is *your exact recipe*: frozen chunked (diffusion) BC + closed-loop residual RL from **RGB**, sparse reward → large precision gains on multi-part assembly. [RLDG 2412.09858](https://arxiv.org/abs/2412.09858) and [HIL-SERL 2410.21845](https://arxiv.org/abs/2410.21845) show RL reaches ~100% on connector/RAM insertion; IndustReal's SBC curriculum ([2305.17110](https://arxiv.org/abs/2305.17110)) (start near the seated goal, recede) is directly portable.
- **Compute/fit:** highest engineering + GPU cost; **GPU contention with Gazebo is the real risk** (training starves sim to 0.05× RTF) — must serialize RL rollout-collection vs update, and the on-policy sim rollouts are the throughput limit at ~6.5 min/trial. Start from a well-fit #1 policy so RL only learns the last-inch residual (small action space, fast convergence — ResiP/HIL-SERL converge in ~1–2 hr of interaction).
- **Expected impact:** the only move with a demonstrated path to *≥90–100%* insertion (the RL insertion papers' headline numbers). It is the ceiling-raiser once #1 gives it a competent base policy and #2 gives it force-aware seating.

**Sequencing:** #1 first (precondition, cheap), #2 concurrently (cheap, uses existing data), #3 last (highest ceiling, highest cost, needs #1's base). VLA fine-tuning (SmolVLA/π0-LoRA, Sec. 4) is a *substitute* for #1 that also brings a visual prior — attractive if your visual diversity (distractors, poses) is the binding constraint after #1+#2, but not ahead of them.

---

## 9. Full citation list (title · arXiv · URL)

- ACT/ALOHA — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — 2304.13705 — https://arxiv.org/abs/2304.13705
- Diffusion Policy — *Visuomotor Policy Learning via Action Diffusion* — 2303.04137 — https://arxiv.org/abs/2303.04137
- 3D Diffusion Policy (DP3) — *3D Diffusion Policy* — 2403.03954 — https://arxiv.org/abs/2403.03954
- Consistency Policy — 2405.07503 — https://arxiv.org/abs/2405.07503
- Bidirectional Decoding (BID) — 2408.17355 — https://arxiv.org/abs/2408.17355
- π0 — *A Vision-Language-Action Flow Model for General Robot Control* — 2410.24164 — https://arxiv.org/abs/2410.24164
- πRL — *Online RL Fine-tuning for Flow-based VLA* — 2510.09976 — https://arxiv.org/html/2510.09976
- OpenVLA — 2406.09246 — https://arxiv.org/abs/2406.09246
- SmolVLA — 2506.01844 — https://arxiv.org/abs/2506.01844
- GR00T N1 — 2503.14734 — https://arxiv.org/abs/2503.14734
- AutoMate — 2407.08028 — https://arxiv.org/abs/2407.08028
- IndustReal — 2305.17110 — https://arxiv.org/abs/2305.17110
- FORGE (force-guided IndustReal) — 2408.04587 — https://arxiv.org/html/2408.04587
- FMB — 2401.08553 — https://arxiv.org/html/2401.08553
- RLDG — 2412.09858 — https://arxiv.org/abs/2412.09858
- HIL-SERL — 2410.21845 — https://arxiv.org/abs/2410.21845
- SERL — 2401.16013 — https://arxiv.org/abs/2401.16013
- ResiP (Residual RL for Precise Visual Assembly) — 2407.16677 — https://arxiv.org/abs/2407.16677
- EasyInsert — 2505.16187 — https://arxiv.org/abs/2505.16187
- Diffusion-Based Impedance Learning — 2509.19696 — https://arxiv.org/abs/2509.19696
- PhaForce (phase-scheduled visual-force) — 2603.08342 — https://arxiv.org/pdf/2603.08342
- FILIC (force-guided IL + impedance) — 2509.17053 — https://arxiv.org/abs/2509.17053
- Force/Tactile-Conditioned Diffusion Policy — 2510.13324 — https://arxiv.org/html/2510.13324v1
- ForceFlow (contact-driven flow matching) — 2605.11048 — https://arxiv.org/pdf/2605.11048
- Lift3D — 2411.18623 — https://arxiv.org/html/2411.18623
- NoReal3D (pseudo-3D from monocular RGB) — 2509.16532 — https://arxiv.org/html/2509.16532v1
- VO-DP (vision-only diffusion policy, VGGT+DINOv2) — 2510.15530 — https://arxiv.org/pdf/2510.15530
- LiPo (policy composition) — 2506.05165 — https://arxiv.org/abs/2506.05165
- DAgger — 1011.0686 — https://arxiv.org/abs/1011.0686 · HG-DAgger — 1810.02890 — https://arxiv.org/abs/1810.02890
- GB10 / DGX Spark specs — https://www.nvidia.com/en-us/products/workstations/dgx-spark/ · openpi (π0 LoRA) — https://github.com/Physical-Intelligence/openpi

(Two arXiv IDs — DP3 2403.03954 and Consistency Policy 2405.07503 — are cited from prior knowledge, not this session's search hits; verify before formal write-up. All others are corroborated by the searches above.)