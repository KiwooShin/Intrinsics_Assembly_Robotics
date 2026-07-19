I have a comprehensive evidence base. Here is the research report.

---

# Breaking the Last-Inch Attractor: Training/Data Methods for Near-Oracle Insertion Reliability

**Scope:** training- and data-side interventions to convert your ACT-style BC policy from "clean approach → stall at 0.05–0.08 m" into reliable full insertion (target avg >90/100), given your verified root cause (deterministic-head mode-averaging toward the oracle's zero-velocity seated endings) and your stack constraints (single-env Gazebo at ~0.05× RTF under GPU load, 6.5 min/trial, 17 min/retrain, wrench recorded-but-unused, CheatCode privileged oracle, engine insertion event as a ground-truth success signal).

## 0. Two corrections + one reframe that change the ranking

**Citation corrections (verified):**
- The ID you cited for "juicer-style data aggregation," **arXiv:2402.17768, is actually _Diffusion Meets DAgger_ (DMD)** — a different (also-relevant) paper. **JUICER is arXiv:2404.03729.** Both are covered below.
- **LiPo (arXiv:2506.05165) is NOT residual RL or policy composition.** It is a lightweight post-optimizer that *smooths action-chunk discontinuities at chunk boundaries* (overlap scheduling + linear blending). It is a motion-quality/jerk lever, closely related to the ACT temporal-ensembling you already closed-negative — not a last-inch fix. De-prioritize it; details in §7.

**The reframe (most important literature finding for your project):** ACT's own ablation (arXiv:2304.13705, Table in §3.4) reports that **"removal of the CVAE objective makes almost no difference on _scripted_ data" (unimodal/deterministic), while on human data success collapses 35.3% → 2%.** Your teacher is a *scripted* CheatCode oracle. So the standard "add a CVAE latent" recommendation (your prior rec #2) is the one most likely to under-deliver: the CVAE is the fix for *human action noise*, which your data does not have. Your bimodality is narrower and more mechanical — **observation aliasing at the terminal**: the near-port *approaching* frame (nonzero closing velocity) is visually near-identical to the *seated* frame (zero velocity), and the deterministic L1 head regresses the mean of the two → ~0 twist → stall. This reframe elevates two under-weighted moves — **(a) disambiguate the observation (add wrench + TCP-to-port height, both already available)** and **(b) curate/reset the data so the terminal zero-velocity tail stops aliasing the approach** — above the CVAE, because they attack the actual mechanism (aliasing), not action noise the scripted oracle never had.

---

## 1. DAgger family — last-inch corrective demos at the observed stall states

**Methods & evidence.**
- **DAgger** (Ross, Gordon, Bagnell 2011, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)) is the formal fix for exactly your failure: BC error compounds off the demonstrated support; aggregate expert labels *at the states the learner actually visits* (your 0.05–0.08 m stall states) and retrain. This is the canonical treatment for compounding-error + coverage floors, and your smoke60 analysis already fingered it as the primary root cause.
- **HG-DAgger** (Kelly et al. 2019, [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)) generalizes DAgger to a *gated* takeover: an expert seizes control at failure regions and hands back, and it learns an uncertainty threshold marking where the novice is unsafe. In your stack the "human" is replaced by the **privileged CheatCode oracle** and the gate is automatic (engine stall detection / plug-port distance plateau) — i.e., **auto-gated DAgger**, which removes HG-DAgger's main cost (human labor) entirely.
- **JUICER** (Ankile et al. 2024, [arXiv:2404.03729](https://arxiv.org/abs/2404.03729)) is the most on-point: it annotates *bottleneck states* (precision-critical regions = your last inch), synthesizes **counterfactual corrective trajectories near those bottlenecks**, and augments coverage there. Effect: a policy from **10 demos + JUICER beats 50 demos without it** on simulated multi-part assembly. This is precisely "manufacture last-inch corrective data without new teleop."
- **DMD** (Zhang et al. 2024, [arXiv:2402.17768](https://arxiv.org/abs/2402.17768)) synthesizes the OOD recovery states + on-manifold corrective labels using diffusion instead of live rollouts: **80% success from 8 demos** on Franka pushing. Caveat: it is *eye-in-hand* specific (it renders novel wrist-cam views); you have 3 fixed cams, so the image-synthesis machinery is a poor fit, but the concept (synthesize corrective data off the demonstrated manifold) carries.

**Fit to your stack (excellent).** You have the one ingredient that makes DAgger cheap and exact: a **privileged oracle that reads ground-truth poses**. You can (i) roll your current BC policy until the engine/plug-port distance plateaus at 0.05–0.08 m, (ii) freeze that exact state, (iii) let CheatCode label a **decisive nonzero closing action** from there, (iv) aggregate + retrain (17 min). Critically, **also curate the terminal tail**: trim or down-weight the long run of near-zero-velocity seated frames whose RGB aliases the near-port approach — this is a one-line change to `prepare_dataset.py` trimming and directly reduces the mean the head averages toward.

**Effect size (expected).** DAgger-family results routinely move BC from partial to near-complete on the *aggregated region*; JUICER's 10>50 and DMD's 8-demo-80% show large gains from *small* targeted corrective sets. On your task the mechanism is unusually clean (single, localized, geometrically-defined bottleneck), so this is the highest-probability route from 0 → nonzero insertions.

**Cost:** low. Reuses collection + trim + retrain infra you already hardened. Main new code: stall-state capture + reset-to-captured-state (you have GT poses, so this is programmatic). **Risk:** low — worst case is neutral if the aliasing (not coverage) dominates, which §2 hedges.

---

## 2. Force/wrench (+ TCP height) in the observation — disambiguate approach from seated

**Why this is mechanistically the cleanest attack on the attractor.** The attractor exists because *approaching-near-port* and *seated-done* produce near-identical RGB but opposite target velocities. **Wrench breaks the tie for free:** during approach wrench ≈ 0; at seating/contact wrench spikes. Add the 6-D wrench (and the scalar TCP-to-port axial distance, which you can supply from the same TCP state you already feed) to the observation and the head can condition "keep pushing" vs "stop" on a signal that *actually differs between the two modes* — collapsing the bimodality at its source rather than representing it.

**Evidence.** Force proprioception consistently lifts contact-rich insertion over vision-only: FILIC ([arXiv:2509.17053](https://arxiv.org/abs/2509.17053)) reports joint-torque obs **80%** and estimated EE-force obs **90%** vs position-only on peg-in-hole; force-augmented proprioception improves success "in all settings." RLDG's and HIL-SERL's insertion specialists both include **wrench in the observation** (below). General contact-rich surveys converge on force/torque being critical below ~1 mm clearance.

**Fit to your stack (near-free, under-exploited).** `/fts_broadcaster/wrench` **already exists and is already recorded in every dataset** — your policy simply doesn't consume it. This is the lowest-cost structural change available: extend the 7-D TCP state to ~13-D (add 6-D wrench) ± 1-D axial distance, re-normalize, retrain (17 min). **Risk:** low; the only caveats are (a) sim wrench fidelity in Gazebo (validate the signal is non-degenerate at contact) and (b) re-tuning input normalization. This move is currently the biggest gap between your setup and the contact-rich IL/RL literature, and it is nearly free.

---

## 3. Residual RL on the frozen BC policy — learn the last inch RL cannot get wrong

**The flagship result (ResiP, [arXiv:2407.16677](https://arxiv.org/abs/2407.16677)).** Freeze the chunked BC policy, add a *fully closed-loop* residual `a = a_base + α·a_res` (α≤1, 10-D EE-pose), train the residual with **PPO on a sparse binary "assembled" reward**. It explicitly targets your exact two failure causes: *distribution shift* and *lack of closed-loop correction from action chunking*. Effect sizes are dramatic:

| Task | BC (diffusion) | + Residual RL |
|---|---|---|
| peg-in-hole | 5% | **99%** |
| one_leg (low/med rand) | 54% / 29% | **98% / 76%** |
| round_table (low/med) | 12% / 4% | **94% / 77%** |
| lamp (low/med) | 7% / 2% | **97% / 70%** |

**The catch for your stack:** ResiP used **1024 parallel Isaac Gym envs, up to 500M env steps**. That is *on-policy* PPO and is **infeasible in single-env Gazebo at 0.05× RTF** — you'd need years of wall-clock.

**The fix: go OFF-policy** ([Residual Off-Policy RL for Finetuning BC, arXiv:2509.19301](https://arxiv.org/abs/2509.19301)). Same frozen-base + additive-residual + **sparse binary reward**, but DDPG/TD3 + RLPD + REDQ ensembling. Reported **~200× sample efficiency over PPO** (BoxCleanup converges 200k vs 40M steps), and **real-world 14%→64% in 134 rollouts (~15 min)** and **23%→64% in 343 episodes (~76 min)** on high-DoF manipulation. Off-policy is the *only* residual-RL family whose sample budget (10²–10⁵ interactions) fits a slow single-env simulator.

**HIL-SERL** (Luo et al., Science Robotics 2025, [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)) is the proof point for *full* (not just residual) online insertion RL: **~100% on RAM insertion / SSD assembly in 1–2.5 h** with a sparse learned success classifier + a few demos + human corrections, ~2× fewer interventions than baselines. In your setup the **engine's insertion event replaces the learned classifier** (a perfect, free reward), and your BC policy replaces the demo-init — you could run HIL-SERL-style online RL with automatic reward and *no human*, using the CheatCode oracle for interventions.

**Fit/cost/risk.** Highest ceiling of any method here (this is the literature's answer to "BC saturates on contact-rich insertion"), and the reward you need (insertion event) is already emitted per trial. **Cost: high** — new RL loop, replay buffer, reward wiring, and you must reconcile Gazebo throughput (serialize vs training per your GPU-RTF rule; consider a lightweight contact sim or reduced-fidelity fast env for the residual, then validate in Gazebo). **Risk: medium-high** — RL exploration can regress the clean approach; mitigate with small α, residual-only (base frozen), and an action-magnitude clamp. Recommend the **off-policy** variant, seeded from `v2_wide`, reward = insertion event, action = residual on your existing twist/pose target.

---

## 4. Coverage + data-generation + augmentation — fix the orthogonal miss floor

Even a perfect last-inch fix cannot score on the **8–9/15 configs that _miss entirely_** (moderate |yaw|∈[0.5,1.5], rail0). That is a classic BC support/coverage hole, independent of the attractor, and it gates everything.

- **MimicGen** (Mandlekar et al. 2023, [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)): transform a *small* demo set to *new object poses* using known object frames → **~200 demos → 50k across 18 tasks incl. high-precision assembly**, "compares favorably to collecting more human demos." You already read **ground-truth board/port poses**, so you can MimicGen-style **replay/re-target existing demos into the undersampled yaw/rail cells** almost for free (no new oracle rollouts), directly filling the miss-floor band your analysis identified.
- **Image augmentation** (RAD, Laskin et al. [arXiv:2004.14990](https://arxiv.org/abs/2004.14990); DrQ, Kostrikov et al. [arXiv:2004.13649](https://arxiv.org/abs/2004.13649)): **geometric augs (random crop / ±few-pixel shift) dominate; photometric (color jitter, cutout) are largely ineffective** for sample-efficient control. Diffusion-Policy practice concurs (random crop is standard). On your 3×128px cams this is a near-free regularizer for precision + distractor robustness. **Add random-shift/crop; skip heavy color aug.**
- **Domain randomization** (textures/lighting/camera, standard sim-to-real practice; ResiP applies it in the distillation phase): mild DR over distractor placement, board texture, and lighting hardens against the distractor-mount collisions (your −23 SC failures) without new demos.

**Cost:** low (MimicGen-replay reuses your oracle + GT poses; aug is a dataloader change). **Risk:** low. **This is the cheapest way to lift the miss-floor configs from +1.0 to at-least-proximity**, which is a prerequisite for the last-inch fix to even register on those cells.

---

## 5. Curriculum + automatic recovery-demo generation via privileged resets (assembly-by-disassembly)

This is the systematic, automated version of §1, and it is the direct sim recipe for "generate corrective demos near failure/goal states."

- **Reverse Curriculum Generation** (Florensa et al., CoRL 2017, [arXiv:1707.05300](https://arxiv.org/abs/1707.05300)): start from states *adjacent to the goal* and expand outward as the agent succeeds; demonstrated on a 7-DoF **key-insertion** task otherwise unsolvable by standard RL. For you: reset the plug **seated**, retract along the insertion axis in small increments to synthesize a graded set of near-port init states — exactly the last inch.
- **IndustReal** (Tang et al. 2023, [arXiv:2305.17110](https://arxiv.org/abs/2305.17110)): **83–99% zero-shot sim-to-real over 600 trials** on contact-rich assembly, via (i) **SDF-based dense reward** (resolves insertion symmetry/where "done" is), (ii) **Sampling-Based Curriculum (SBC)** that starts near-engaged and expands — preventing overfitting to over-constrained initial contact, and (iii) a **policy-level action integrator** (they integrate the policy's velocity commands into pose targets — structurally identical to your receding-horizon integrator, and worth mirroring their formulation). SBC is the principled curriculum for your "easy poses → eval-band yaw" ramp.
- **AutoMate** (Tang et al. 2024, [arXiv:2407.08028](https://arxiv.org/abs/2407.08028)): **specialist policies solve 80 assemblies at 80%+**, distilled into an 80%+ generalist, via **assembly-by-disassembly** (generate init states by *disassembling* from the seated pose — the automatic reverse-curriculum reset you want), **RL-with-imitation-objective + DTW** (imitation regularizer keeps RL near the oracle trajectory — a hedge against RL wrecking your clean approach), and **curriculum-based RL fine-tuning**. This is the closest published blueprint to "privileged-oracle + sim + insertion goal → reliable insertion across poses."

**Fit/cost/risk.** You have privileged pose access and programmatic resets, so assembly-by-disassembly is directly implementable. **Cost: medium** (reset/disassembly harness + curriculum scheduler). **Risk: low-medium.** Pairs naturally with §1 (data) or §3 (RL). SBC + assembly-by-disassembly is how you'd auto-generate the last-inch corrective distribution at scale rather than hand-collecting stall states.

---

## 6. Scripted-oracle → policy distillation gap, and the RLDG insight

Your oracle scores ~93/100 but the student stalls — a **distillation gap**. Two literature points:
- **RLDG** (Xu et al. 2024, [arXiv:2412.09858](https://arxiv.org/abs/2412.09858)): distilling **RL-generated** trajectories into a generalist beats distilling human demos — **connector insertion 96%→~100%, FMB insertion 67%→100%, unseen-connector 90%→100%**, +30–50% generalization. The mechanism (their Fig. 8) is decisive: **RL actions concentrate probability toward the optimal insertion direction; demo actions cluster around the action-space center** → RL targets are more *optimal and consistent*, so distillation succeeds where human/loose demos saturate. **Direct implication for you:** your oracle's *zero-velocity seated endings* are precisely the "clusters toward the center/stop" pathology — an RL (or even just a decisively-tuned oracle) specialist would supply **monotonic closing-velocity** targets that distill without collapsing. RLDG's specialists use **wrench in the obs** (reinforces §2).
- The general reduction: DAgger (§1) is the formal cure for the distillation/coverage gap; the practical cure here is either (a) make the *teacher's* last-inch action decisively nonzero (fix the oracle's terminal deceleration / curate the tail) or (b) source last-inch targets from RL (§3). Note your own SC-oracle fragility (3/8 keep under eval-band yaw) is a *teacher-quality* gap that caps the student regardless of method — worth the flagged floor micro-tune (≈−0.007) + pose-conditioned waypoint before scaling any distillation.

---

## 7. Action-space, chunk-length, and the LiPo correction

- **Action parameterization.** Your velocity-target head interacts badly with the attractor: "predicted mean ≈ 0" maps to "no motion." A **delta-pose** or **impedance-target** parameterization where the *neutral output holds position* and a learned command produces a decisive axial push can be more robust near the terminal; ResiP/IndustReal/AutoMate all operate in EE-pose (+ integrator) rather than raw velocity. Consider re-parameterizing the last-inch action as an axial delta-pose with a nonzero learned setpoint, or an impedance target that keeps commanding into the port under contact (this is also what makes force-controlled insertion forgiving). Low-to-medium cost; medium risk (touches the controller you already fixed once).
- **Chunk length K.** ACT ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)) shows rollout success peaks at a ~2 s horizon; BID ([arXiv:2408.17355](https://arxiv.org/abs/2408.17355)) shows longer chunks aid open-loop stability but reduce reactivity (your k16-vs-k8 collision asymmetry, though your n=1 can't confirm). This is a *tuning* knob, not a last-inch cure — settle K on the 180 s scored suite, not L1.
- **LiPo** ([arXiv:2506.05165](https://arxiv.org/abs/2506.05165)) — **corrected role:** it smooths *chunk-boundary discontinuities* via overlap scheduling + linear blending. It is a jerk/motion-quality lever, adjacent to the temporal ensembling you already tested and closed-negative (m=0.01, n=3/arm: OFF 23.1±2.1 vs ON 20.1±4.2). It will **not** break the attractor. Keep it shelved unless you adopt a genuinely multimodal head (then blending has a defined role).

---

## 8. Multimodal head — where CVAE/diffusion actually sits (below the data/obs fixes)

**The caveat is the headline (see §0):** ACT's ablation shows **CVAE ≈ deterministic on scripted data**, so a plain CVAE latent (your prior rec #2) is unlikely to pay off on a scripted-oracle stack — the CVAE cures *human* action noise you don't have. Your bimodality is *observation aliasing at the terminal*, which a CVAE with latent=0 at inference still averages over.

**If you want a truly multimodal head, use a mode-selecting generator, not a Gaussian-CVAE mean:** Diffusion Policy (Chi et al. 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)) or VQ-BeT ([arXiv:2403.03181](https://arxiv.org/abs/2403.03181)) can represent "push-in" as a *distinct sampled mode* rather than a mean. **But sampling a bimodal terminal 50/50 risks drawing "stop" as often as "push,"** so a multimodal head only reliably helps **combined with §2 (wrench/height disambiguation) or §1 (curated data that removes the aliasing)** — at which point the distribution is no longer bimodal and the multimodal head is less necessary. Net: multimodal-head is a *complement* to the aliasing fixes, not a substitute, and ranks below them. **Cost: high** (new head, ~3–10× inference — still within your ~900× 4 Hz headroom); **risk: medium** (mode-mis-selection, as your ensembling A/B already hinted).

---

## TOP 5 RANKED METHOD MOVES

Ranked by (expected effect on insertion reliability) × (fit to your exact stack) ÷ (cost + risk). The first two are near-free and attack the verified mechanism directly; do them first and in combination.

**#1 — Auto-DAgger + terminal-tail curation: privileged-reset last-inch corrective demos.**
Roll BC to the 0.05–0.08 m stall, freeze the state (you have GT poses), let CheatCode label a *decisive closing action*, aggregate + retrain; simultaneously trim/down-weight the zero-velocity seated tail that aliases the approach. *Papers:* DAgger [1011.0686], HG-DAgger [1810.02890], JUICER [2404.03729], DMD [2402.17768]. *Effect:* highest-probability 0→nonzero insertions (JUICER: 10 demos beat 50; DMD: 8→80%). *Cost:* low (reuses collect+trim+17-min retrain). *Risk:* low.

**#2 — Put wrench (+ TCP-to-port axial distance) in the observation.**
Disambiguates approach (wrench≈0) from seated (wrench high), collapsing the aliasing that zeros the head. *Papers:* FILIC [2509.17053]; wrench-in-obs in RLDG [2412.09858] & HIL-SERL [2410.21845]. *Effect:* directly removes the mode-averaging cause; force-obs adds ~10–20 pp on peg-in-hole in the literature. *Cost:* **near-zero — wrench is already recorded, just unused.** *Risk:* low (validate Gazebo wrench fidelity + re-normalize).

**#3 — Automatic recovery-demo generation via assembly-by-disassembly + sampling-based curriculum.**
Reset seated, retract along the insertion axis to synthesize graded near-port init states; curriculum from near-engaged → eval-band yaw. The scalable, automated form of #1. *Papers:* AutoMate [2407.08028], IndustReal SBC [2305.17110], Reverse Curriculum [1707.05300]. *Effect:* large + systematic (AutoMate/IndustReal 80–99% on contact-rich assembly). *Cost:* medium (reset/disassembly harness + scheduler). *Risk:* low-medium.

**#4 — Coverage + augmentation to lift the miss-floor (prerequisite, orthogonal to the attractor).**
MimicGen-style privileged-pose replay into the undersampled |yaw|∈[0.5,1.5]/rail0 cells + geometric image aug + mild DR. *Papers:* MimicGen [2310.17596], RAD [2004.14990], DrQ [2004.13649]. *Effect:* moves 8–9/15 entirely-missing configs to at-least-proximity so #1–#3 can register there. *Cost:* low. *Risk:* low.

**#5 — Off-policy residual RL on the frozen BC policy, reward = engine insertion event.**
Highest ceiling; the literature's definitive contact-rich-insertion fix — but use the *off-policy* variant to fit single-env Gazebo, not ResiP's 500M-step PPO. *Papers:* ResiP [2407.16677] (effect: 5%→99% peg-in-hole), Residual Off-Policy RL [2509.19301] (~200× more sample-efficient; 14%→64% in ~15 min real), HIL-SERL [2410.21845] (~100% insertion in 1–2.5 h). *Effect:* largest potential (learns the last inch BC cannot). *Cost:* high (RL loop + Gazebo-throughput reconciliation). *Risk:* medium-high (guard the clean approach with small α, frozen base, imitation/DTW regularizer à la AutoMate). Pursue after #1–#2 confirm the attractor moves.

**Explicitly demoted:** plain CVAE latent (ACT ablation: ≈deterministic on *scripted* data — §0/§8); ACT temporal ensembling (you already closed-negative, n=3); LiPo (chunk-boundary smoothing, not a last-inch fix). A multimodal *diffusion/VQ-BeT* head is worth it only *paired with* #1/#2, not as a standalone substitute.

**Sequencing note (fits your constraints):** #1+#2 share one retrain (~17 min) and require no RL infra — run them together first; they are the cheapest test of whether the attractor is aliasing-driven (it is, per your forensics). Add #4 in the same retrain. Reserve #3 and #5 for after you confirm nonzero insertions, since they carry harness/RL cost against your GPU-vs-Gazebo serialization rule. Adopt everything on the **180 s matched-seed suite with n≥3 reps** (your measured ±3–18 pt noise floor makes single-run per-config claims unresolvable).

---

## Sources
- DAgger — Ross, Gordon, Bagnell 2011 — https://arxiv.org/abs/1011.0686
- HG-DAgger — Kelly et al. 2019 — https://arxiv.org/abs/1810.02890
- Diffusion Meets DAgger (DMD; the ID you cited as "JUICER") — Zhang et al. 2024 — https://arxiv.org/abs/2402.17768
- JUICER (correct ID) — Ankile et al. 2024 — https://arxiv.org/abs/2404.03729
- MimicGen — Mandlekar et al. 2023 — https://arxiv.org/abs/2310.17596
- ResiP / From Imitation to Refinement — Ankile et al. 2024 — https://arxiv.org/abs/2407.16677
- Residual Off-Policy RL for Finetuning BC — 2025 — https://arxiv.org/abs/2509.19301
- RLDG — Xu et al. 2024 — https://arxiv.org/abs/2412.09858
- HIL-SERL — Luo et al. 2024/2025 (Science Robotics) — https://arxiv.org/abs/2410.21845
- IndustReal — Tang et al. 2023 — https://arxiv.org/abs/2305.17110
- AutoMate — Tang et al. 2024 — https://arxiv.org/abs/2407.08028
- Reverse Curriculum Generation — Florensa et al. 2017 — https://arxiv.org/abs/1707.05300
- FILIC (force-guided IL for contact-rich) — 2025 — https://arxiv.org/abs/2509.17053
- RAD (data augmentation for control) — Laskin et al. 2020 — https://arxiv.org/abs/2004.14990
- DrQ (data-regularized Q) — Kostrikov et al. 2020 — https://arxiv.org/abs/2004.13649
- ACT / ALOHA (CVAE ablation: ≈deterministic on scripted data) — Zhao et al. 2023 — https://arxiv.org/abs/2304.13705
- Diffusion Policy (multimodal head) — Chi et al. 2023 — https://arxiv.org/abs/2303.04137
- VQ-BeT (multimodal head) — Lee et al. 2024 — https://arxiv.org/abs/2403.03181
- BID / Bidirectional Decoding (chunk-length horizon) — Liu et al. 2024 — https://arxiv.org/abs/2408.17355
- LiPo (chunk-boundary smoothing; NOT residual RL) — Son & Park 2025 — https://arxiv.org/abs/2506.05165