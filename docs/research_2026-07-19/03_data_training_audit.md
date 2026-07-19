I have everything I need. Here is the audit.

---

# Data + Training Audit — Intrinsic AIC Insertion Policy

Scope: the 93-episode ACT-lite dataset + `train_v2`/`train_v3` recipe behind `v2_wide` (submission) and the `p1/p2` retrains. Verified against `SESSION_REPORT.md`, `CLAUDE.md §6`, both `campaign_log.csv`, both `configs_phase*/manifest.csv`, `eval_suite_smoke/manifest.csv` + `suite_meta.yaml`, `aic_engine/config/eval_config.yaml`, `opt/train_v3.py`, `opt/config.py`, `opt/augment.py`, `opt/results/w1_report.md`, and the `p2_k*_train.log`.

Key file paths (absolute):
- Trainers: `/home/kiwoos/training/train_v2.py` (shipped `v2_wide`), `/home/kiwoos/work/Intrinsics_Assembly_Robotics/opt/train_v3.py` (p1/p2), `/home/kiwoos/work/Intrinsics_Assembly_Robotics/opt/config.py` (defaults), `/home/kiwoos/work/Intrinsics_Assembly_Robotics/opt/augment.py`
- Data prep/trim: `/home/kiwoos/training/prepare_dataset.py`
- Config gen: `/home/kiwoos/training/gen_config.py`
- Manifests: `/home/kiwoos/data/configs_phase0/manifest.csv`, `/home/kiwoos/data/configs_phase2/manifest.csv`
- Eval distribution: `/home/kiwoos/work/Intrinsics_Assembly_Robotics/aic_engine/config/eval_config.yaml`, `/home/kiwoos/work/Intrinsics_Assembly_Robotics/eval_suite_smoke/manifest.csv`

---

## (1) What the 93 episodes actually cover vs the eval distribution

**Composition (verified counts):** `ds_phase0` 44 (39 SFP + 5 SC) + `ds_phase2` 33 (30 SFP + 3 SC) + `ds_wide` 11 (SFP) + `smoke` 5 (SFP) = **93**. **SC = 8 / 93 = 8.6 %**; SFP = 85 / 91.4 %. Every episode ≈ 530 frames, dt ≈ 0.275 s (~145 s task window), trimmed to `[first pose_command, last + 0.3 s]`.

**Yaw is the dominant hole.** Tally of board_yaw over the episodes actually on disk (joined campaign_log KEEP → manifest):

| Source | \|yaw\|≥2.0 | 1.3≤\|yaw\|<2.0 | \|yaw\|<1.3 |
|---|---|---|---|
| ds_phase0 (44) | 39 | 5 | **0** |
| ds_phase2 (33) | 29 | 4 | **0** |
| ds_wide (11) | 11 (yaw 3.1±0.2) | 0 | **0** |
| smoke (5) | 5 (yaw 3.1±0.08) | 0 | **0** |

The **entire dataset lives at \|board_yaw\| ≥ ~1.3, ~90 % at \|yaw\| ≥ 2.0** ("board faces the robot, ~π"). The generators enforce this: `gen_config.py` wide mode fixes `yaw = 3.1 ± 0.2`; near mode fixes `3.1 ± 0.08`; the phase0/phase2 samplers used the eval-band cluster after the earlier `U(-π,π)` bug was found.

The eval suite (`suite_meta.yaml`) samples `board_yaw ~ U(-π, π)` — the **full circle**. So:

| Eval subset | Yaw values | Covered by demos? |
|---|---|---|
| **Official trial_1/2/3** | 3.1, -3.1, -1.8 | **Yes** — all \|yaw\|≥1.8, in-distribution (why `v2_wide` inserts on the real /300) |
| **Stratified cfg_000–011** | -1.49, 0.84, 1.32, 0.63, -1.45, 2.67, -0.76, 2.84, -1.23, 1.29, -0.63, 0.91 | **8 of 12 fall in \|yaw\|<1.5** — the empty band. These are exactly the cells that floor at +1.0 |

**Enumerated coverage holes (most to least load-bearing):**

1. **Moderate/low yaw \|yaw\| ∈ [0, 1.5]: zero demos.** 8/12 stratified cells sit here; all floor (miss). This is a genuine BC support gap — the policy never initiates a correct approach because it has never seen the board at that orientation. *Caveat: the true official /300 never tests this band, so it is a robustness/stratified-suite hole, not a submission-metric hole.*
2. **SC rail 2: never demonstrated.** SC demos exist only for `sc_rail0` and `sc_rail1` (both phases). Eval SC includes `cfg_010`/`cfg_011` on **rail 2** → fully OOD. SC ports: demos give sc_port_0 (rail0) and sc_port_1 (rail1) only.
3. **SC drastically under-represented: 8.6 % of data vs 33 % of officials / 50 % of the stratified suite.** Only 8 SC demos total, 5 of them at scores that are fine but 3 of the last 8 collected were near-threshold — SC is both sparse and low-quality (oracle keep-rate 3/8 under eval-band yaw+distractors).
4. **Distractor density is lighter in demos.** Demo `n_distractors` = 1–3 rails; eval configs carry 6+ present entities. Same *kind*, lower *clutter*.
5. **rail0 "uniform failure" is confounded with the yaw hole, not a rail-count gap.** rail0 actually has the *most* SFP demos (19), yet all four rail0 stratified cells fail — but all four also have moderate yaw (-1.49/0.84/1.32/0.63). The failure tracks yaw, not rail. Don't over-collect rail0.

**Well-covered (do not spend budget here):** SFP rails 0–4 (11/10/8/7/8 phase0), both SFP ports (19/20), board_x 0.15–0.20, board_y −0.21–0.05, grasp_z 0.040–0.046 — all match eval ranges.

---

## (2) Is 93 episodes fundamentally too few?

**Not in the raw sense — the binding constraint is representation + coverage, not count.** Two pieces of direct evidence from this repo:

- `v2_wide` was trained on only **16 near-eval demos** and still produces genuine insertions on the official /300 (119.4). ACT/ALOHA (Zhao et al., arXiv:2304.13705) reports strong results at **~50 demos per single narrow setup**; 16 near-eval demos on the narrow official pose is the same regime.
- Phase-2 **added 33 targeted demos → best-ever val L1 (0.00129, −40 % vs p1) → still 0/45 insertions and the *worst* officials.** This is a controlled A/B showing more data did not move the task metric. Count is not the binding lever right now.

**But 93 is far too sparse for the *breadth* the stratified suite demands.** Published per-condition densities: ACT 50/task, Diffusion Policy (Chi et al., arXiv:2303.04137) ~100–200/task, RoboMimic (Mandlekar et al., arXiv:2108.03298) 200 proficient-human/task — each covering **one** setup. Here 93 demos are spread over a discrete grid of ≥20 cells (2 plugs × 5 rails × 2 ports) *times* continuous `U(-π,π)` yaw × wide x/y → **~4–5 demos/cell, ~2/cell for SC, and 0 in the moderate-yaw band**. That is ~10× below ACT's per-condition density and, for the empty yaw band, undefined. Ross et al.'s DAgger analysis (arXiv:1011.0686) is the relevant frame: BC error compounds off the demonstrated support, and 8/12 stratified cells are off-support.

**Verdict:** for the official /300, 93 (even 16) is adequate; for the target of >90/trial *everywhere including stratified*, count is a secondary constraint behind (a) the deterministic-head multimodality (Section 3/4) and (b) filling the specific empty strata (Section 1). Blindly scaling to hundreds of the same \|yaw\|≥2 SFP demos would not help.

---

## (3) What the oracle demos cannot teach — and the L1↔score decoupling

The teacher is a **privileged scripted oracle** (reads ground-truth poses, ~93/100, contacts 0). Four things it structurally cannot put in the data:

1. **Recovery / off-manifold correction.** Every demo is a monotone privileged success from a good grasp; there is **no demonstration of "you are stalled at 5–8 cm, back off and re-approach."** When the student visits its own stalled state (which the expert never visits), there is no target action → it sits. Classic covariate shift (Ross et al., arXiv:1011.0686); also the privileged-teacher→vision-student gap (Learning by Cheating, Chen et al., arXiv:1912.12294): the oracle acts on info the RGB student cannot observe, and near the port the RGB is ambiguous (near-port view ≈ seated view).
2. **A firm non-zero closing push near the port.** The oracle decelerates to **zero velocity at seating**, and `prepare_dataset.py` deliberately keeps the seated tail (`t_end = last cmd + 0.3 s`, comment: "keep the final seated frame"). So near-port frames carry a **bimodal action label** — "still pushing in (small +v)" vs "seated, done (0 v)" — for near-identical images. A deterministic L1 head returns the **mean → ~0** → receding-horizon target ≈ current TCP → the last-inch fixed-point attractor documented across every 0/45 trial. This is precisely ACT's stated motivation for a CVAE (arXiv:2304.13705, §"multimodality").
3. **Force-guided search.** The wrench IS recorded (`wrenches.npy` in every episode) but (a) the policy never consumes it and (b) **the oracle itself is pose-scripted with ~zero contact force**, so even if the model read wrench there is **no force→action correlation in the data to imitate**. The demos contain no "feel resistance → wiggle/re-seat" behavior at all.
4. **Any within-state action diversity.** The oracle is deterministic given ground truth: one trajectory per state. There is no distribution to spread a stochastic head over even if you added one — the *data* must be made multimodal (or corrective), not just the model.

**val-L1 ↔ task-score decoupling — the evidence is unambiguous:**

- `p2_k8` best val L1 **0.00129** (−40 % vs p1) → **worst officials (64.9) and 0 insertions**; `v2_wide` worse L1 but the only inserter (SESSION_REPORT 2026-07-19 (b)/(d)).
- The ASHA sweep's −15.6 % first-action-L1 "win" and the W1 shift-aug −15 % L1 win **both failed to produce any insertion**.
- Mechanistically L1 is dominated by the hundreds of well-behaved approach + seated frames; the ~30–50 last-inch push-in frames are a tiny minority, and the loss is actually *minimized* by predicting the zero-velocity mode there. L1 measures approach tracking; it is blind to the one behavior that scores tier-3. Per CLAUDE.md §6, L1 is correctly demoted to a secondary diagnostic — this audit confirms it is nearly anti-correlated with insertions at the margin.

---

## (4) Training-recipe critique

Shipped submission `v2_wide` = `train_v2.py`: **lr 3e-4, bs 256, 60 epochs, K=16, img 128, L1, deterministic head, NO augmentation, no EMA, no weight decay.** The p1/p2 retrains (`opt/train_v3.py`) add TF32 + fused AdamW + `max-autotune` + `wd 1e-4` and *optional* shift-aug; but **`TrainConfig` defaults `shift_pad=0, proprio_dropout=0, ema_decay=0`**, and the `p2_k*_train.log` do not echo any `--shift-pad`, so it is not verifiable that the adopted checkpoints carry augmentation (they use `lr 3e-4` default too, though W1 ran at 1e-3). Concrete deficiencies, ranked by impact on insertion rate:

1. **Deterministic head (no CVAE) — the root architectural defect.** ACT's L1+CVAE exists specifically to stop mean-collapse on multimodal action labels; dropping the CVAE guarantees the zero-velocity attractor of Section 3.2. No recipe tweak fixes this while the head stays deterministic. (arXiv:2304.13705.)
2. **Unweighted L1 over the whole chunk/episode.** The critical last-inch is <10 % of frames; the loss is dominated by approach + seated tail. There is **no phase weighting, no action-magnitude weighting, and no down-weighting of the seated tail** — so the objective actively rewards predicting ~0 near the port.
3. **Wrench ignored despite being recorded and being a *scored* channel (tier-2).** State is pose-only 7D. For contact-rich seating, force is the standard disambiguator between "seated" and "pushing." Feeding `wrenches.npy` is a data-already-present change (no new collection).
4. **128 px, scene cams only, no eye-in-hand.** After AREA-resize 288×256 → 128, a few-mm port at 5 cm is near sub-pixel; the 3 cameras are wide scene views with no close-up of the plug/port interface. This caps last-inch *visual* precision independent of the head. robomimic (arXiv:2108.03298) shows random-shift is the single most important visuomotor-BC aug — and the shipped model has **none**.
5. **No augmentation in the submission checkpoint.** W1 (`opt/results/w1_report.md`) already measured shift4/shift8 at −15 %/−16 % L1 (K=8) and −5–7 % (K=16); the shipped `v2_wide` predates it and runs raw. Even as an L1 win it is free regularization the submission lacks.
6. **lr under-scaled.** bs was raised 64→256 in Phase-2 without re-scaling lr; linear-scaling (Goyal et al., arXiv:1706.02677) predicts ~1.2e-3, W1/analysis prefer 1e-3, but `v2_wide` and the `TrainConfig` default sit at 3e-4. Minor, free.
7. **Epoch budget is not the bottleneck.** train L1 0.068 vs val 0.097 = mild gap, not severe overfit or underfit; more than 60 epochs will not move insertions. Do not spend compute here.
8. **Quaternion in the 7D state is mean/std-normalized component-wise** — geometrically improper (though small); a relative/6D rotation representation would be cleaner if the state is ever reworked.

Net: the recipe is a *fast, clean L1 regressor* that is well-tuned for the wrong objective. The gains available from recipe tweaks (aug, lr, weighting) are real but bounded; the ceiling is set by defects #1–#4.

---

## (5) Cheapest data-side wins, ranked by expected insertion-rate impact

Ordered by expected Δ(insertion rate) per unit effort. Items 1–2 are near-free relabels of existing data.

1. **Trim / down-weight the zero-velocity seated tail (near-free; highest expected impact).** `prepare_dataset.py` intentionally keeps the seated frame; those + the deceleration ramp are exactly the "v≈0 at a near-port view" samples that pull the deterministic mean to zero. **Relabel existing 93 episodes** to drop the terminal `|v|≈0` frames (or weight the loss by action magnitude / inverse plug-port distance). This directly de-biases the near-port action toward push-in and attacks the fixed-point attractor with **no new collection**. Pair with a magnitude-weighted L1.
2. **Rebalance SC in the loss + oversample SC (near-free), then a small targeted SC collection.** SC is 8.6 % of data vs 33–50 % of eval and causes the −23 collisions. Free step: class-weight/oversample the 8 existing SC episodes. Cheap step: after the oracle floor micro-tune (≈−0.007) + pose-conditioned entrance waypoint (the known SC follow-up), collect **~12–16 SC demos at rail2 (currently 0) + both ports + eval-band yaw**. Fixes trial_3 and the SC wrist-into-mount −23s.
3. **Last-inch DAgger at observed stall states (moderate effort, high impact).** Seed CheatCode demos at the replicated 5–8 cm stall poses (± small neighborhood) so the data contains *non-zero closing velocity conditioned on near-port RGB*. This is the collection-side complement to #1 and the SESSION_REPORT's #1 lever (Ross et al. arXiv:1011.0686; HG-DAgger, Kelly et al. arXiv:1810.02890). ~30–50 demos.
4. **Feed the already-recorded wrench into the state (data present, model change only).** No collection cost; provides the seated-vs-pushing signal and lets any future corrective demos actually be force-conditioned. Enables #3 to teach force-reactivity.
5. **Fill the moderate-yaw band \|yaw\| ∈ [0.5, 1.5] — only if the true eval tests it.** ~16–20 demos across the empty band would lift the 8 floored stratified cells. **Rank this LOW for the /300** (officials are all \|yaw\|≥1.8, already covered) and HIGH only if the target explicitly includes the stratified suite. Do not confuse it with rail0 (that's the same yaw hole).
6. **Match distractor density + add a few exact-official-pose reps (cheap insurance).** Raise demo clutter to ~6 entities and add reps at the literal official yaws {3.1, −3.1, −1.8} to harden the submission that already works there. Marginal but safe.

**Bottom line for the audit:** the dataset is *broad-but-shallow and yaw-clipped*, SC is starved and rail2-blind, and — decisively — the demos encode a **bimodal, recovery-free, force-free** near-port signal that a deterministic L1 head averages to a dead stop. The Phase-2 experiment proves more same-shaped data won't convert (best L1, 0 insertions). The cheapest high-leverage moves are **data relabels** (drop the seated tail, magnitude-weight the loss, oversample SC) and **stall-state DAgger**, all of which target the last-inch attractor directly; broad recollection and more epochs are low-yield until the multimodality is represented (CVAE) or the seated-tail bias is removed.