# DATA-FIRST PLAN: Breaking the Last-Inch Attractor to >90/trial

**Thesis (opinionated, and the bet that separates this draft):** The deterministic head is *not* the disease — ACT's own ablation shows CVAE ≈ deterministic on **scripted** teachers (35.3→2% collapse is a *human-noise* fix we don't need). Our bimodality is **terminal observation aliasing** (near-port view ≈ seated view, opposite velocities) baked into the *data* by a seated zero-velocity tail. Therefore the attractor is killable with **data + one 6-dim input change**, no diffusion/flow head. We sequence expressive-head/residual-RL as *contingencies behind kill-gates*, not the main line. If Phase 0 fails to move tier-3, the thesis is wrong and we escalate — cheaply and early.

**Eval instrumentation (binds every gate):** matched-seed suite, **n≥3 reps**, IQM+CI, discard `completed=False` harness hangs (~1070 s) and re-run. Two tiers: **Focused screen** = 3 officials + 6 reached stratified ×3 = 27 trials ≈ **3 h wall**; **Full gate** = 15-config ×3 = 45 trials ≈ **5 h wall**. Overturn bar for any adoption: **≥1 genuine insertion** and IQM +≥5 pts (sub-5 unresolvable at n=3). **Serialize GPU:** train (17 min, Gazebo→0.05×) *then* collect/eval (Gazebo full-speed) — never overlap. Orchestrator schedules train-blocks vs collect/eval-blocks; sub-agents fan out collection only inside full-speed windows.

---

## Phase 0 — Aliasing kill-shot (relabel + wrench + weighted loss)
**Goal:** test the thesis for near-zero cost; get first tier-3 movement.
**Tasks (one retrain):** (1) **Trim the seated zero-velocity tail** in `prepare_dataset.py` (drop terminal `|v|≈0` frames that alias the approach) across all 93 eps. (2) **Magnitude/inverse-distance-weighted L1** so the <10% last-inch push-in frames stop being drowned by approach+seated frames. (3) **Wrench→state, 7D→13D** (already recorded; the *only* permitted architecture change — validate `/fts_broadcaster/wrench` is non-degenerate at contact first; if degenerate, substitute the scalar TCP-to-port axial distance from existing state). (4) **Geometric aug** (random ±4–8 px shift/crop; skip color — RAD/DrQ: photometric is inert).
**Wall-clock:** relabel/loss/aug code ~2 h; validate wrench signal ~0.5 h; retrain 17 min; **focused screen 3 h**. → **~1 day.**
**Gate:** focused screen, n=3, vs v2_wide baseline. **PASS = officials IQM +≥5 and ≥1 stratified reached-config gains pure tier-3 depth** (partial→deeper, or first full insertion).
**Expected:** officials ~40→~55/trial (partials deepen; wrench + tail-drop de-bias toward push-in — mechanism-consistent with the ENSEMBLE-AB pure-tier-3 gains); stratified mean 8→~18. Not full yet — needs corrective states (Phase 1).
**KILL:** if tier-3 does **not** move at all (no deeper partials, no new insertion) despite tail-drop+wrench+weighting → aliasing is *not* the dominant mechanism → **the data-first thesis is falsified**; jump straight to minimal flow/diffusion head on the same encoder (Architecture brief #1) before spending DAgger budget. This is the plan's honest failure test, run in day one.

## Phase 1 — Last-inch auto-DAgger (the decisive fix)
**Goal:** convert reached configs (all 3 officials + ~5 SFP stratified) from stall→seat.
**Tasks:** (1) **Stall-state harness:** roll Phase-0 policy, detect plug-port distance plateau at 0.05–0.08 m, freeze state (we have GT poses → programmatic reset). (2) **CheatCode labels a decisive, monotonic non-zero closing action** from each stall (RLDG insight: teacher targets must concentrate toward the insertion direction — *not* the oracle's decelerating tail). (3) Collect **~40–50 corrective demos** at stalls ± small neighborhood; **aggregate (don't replace)**, cap corrective fraction ≤35% to protect the approach. (4) Retrain.
**Wall-clock:** harness ~3 h; collect 50×6.5 min ≈ **5.5 h**; retrain 17 min; **full gate 5 h**. → **~1.5–2 days.**
**Gate:** full 15-config, n=3. **PASS = all 3 officials seat (≥2/3 reps each) → official /300 ≥ 255**; stratified mean ≥30.
**Expected:** officials 55→**85–90**/trial (**/300 → ~255–270**); stratified mean 18→**~32** (forensics ceiling for "last-inch-only" — reached SFP configs seat, misses/SC unchanged). This is the **+160 official prize** the whole eval hinges on.
**KILL:** if corrective demos don't yield ≥1 *new* insertion beyond Phase 0 at n=3, **or** they regress the approach (à la ensembling-ON corrupting official_1) → last-inch *data* is insufficient → escalate to **force-guarded scripted insertion primitive spliced onto the learned approach** (hybrid, Datameister/MacCody-proven, rules-legal) as the seating stage, and shelve pure-BC seating.

## Phase 2 — SC oracle repair (teacher bug, not model bug)
**Goal:** fix trial_3 (SC official) + eliminate the −23 wrist-into-mount collisions that make any SC-trained checkpoint unshippable.
**Tasks:** (1) **Micro-tune descent floor ≈−0.007** + **pose-conditioned entrance waypoint**; **re-validate zero-contact** under eval-band yaw+distractors (target keep-rate ≥6/8, up from 3/8). (2) Only *then* collect **~12–16 SC demos** at **rail2 (currently 0 demos), both ports, eval-band yaw**. (3) Class-weight/oversample SC in loss (8.6%→~30%). Retrain.
**Wall-clock:** oracle tune+revalidate ~4 h (iterative, cheap trials); collect 16×6.5 ≈ 1.8 h; retrain 17 min; **full gate 5 h**. → **~1–1.5 days.**
**Gate:** full 15-config, n=3. **PASS = official_3 seats (/300 → ~279–285); zero clean −23 across all SC configs; SC stratified mean ≥ +30.**
**Expected:** official /300 → **~279–285**; stratified mean 32→**~50–68** (SC configs miss/−23 → proximity/seat).
**KILL:** if repaired oracle still keeps <5/8 under eval-band yaw+distractors → SC teacher fundamentally can't demo eval SC → **do not train on broken demos** (Phase-2 proved more bad SC data makes SC *worse*, −5.2 vs v2's +5.6); fall back to force-guarded SC primitive, or accept SC configs as a bounded loss and bank the SFP gains.

## Phase 3 — Coverage + recovery (the stratified-only work)
**Goal:** lift the 10/15 miss-floor configs — moderate-|yaw|∈[1.2,1.5] SFP-port_0 (cfg_000/004/008) + rail0 cells — from +1.0 to seated. *Skip if only the official /300 matters — officials are all |yaw|≥1.8, already covered.*
**Tasks:** (1) **MimicGen-style privileged replay:** re-target existing SFP demos into the empty |yaw|∈[0.5,1.5] band + rail0 via GT frames (oracle re-rollout for valid images+contacts), ~16–20 demos. (2) **Assembly-by-disassembly recovery demos:** reset seated, retract along insertion axis in increments → graded near-port inits (IndustReal SBC / reverse-curriculum). (3) Mild DR on distractor placement/lighting. Retrain.
**Wall-clock:** replay+disassembly harness ~4 h; collect ~30×6.5 ≈ 3.3 h; retrain 17 min; **full gate 5 h**. → **~1.5 days.**
**Gate:** full 15-config, n=3. **PASS = ≥12/15 configs seat (≥2/3 reps); stratified mean ≥85.**
**Expected:** stratified mean 68→**~85–93** (target). Officials held.
**KILL:** if moderate-yaw configs reach proximity but *still* won't seat after coverage+corrective → failure is **perception at those poses**, not coverage → targeted encoder upgrade (DINOv2/VGGT features, RGB-only, Architecture §5) — flagged as thesis-boundary, last resort.

## Phase 4 — Hardening & submission
**Goal:** lock >90 avg. Reps at exact official yaws {3.1,−3.1,−1.8} + neighborhood; pick checkpoint by IQM; **re-validate on the `phase_1` branch** (multiple SC ports/rail, contiguous-force-penalty #595 — *more* forgiving, in our favor). **Wall-clock ~1 day** (full gate + officials reps). Gate: **official /300 ≥ 270 and stratified mean ≥90 at n=3.**

---

## Expected score trajectory
| Phase | Official /trial (/300) | Stratified mean | Driver |
|---|---|---|---|
| Current | ~40 (119.4) | 5–8 | last-inch attractor |
| 0 relabel+wrench | ~55 (~165) | ~18 | de-alias → push-in mode |
| 1 DAgger | ~85–90 (~255–270) | ~32 | reached configs seat |
| 2 SC repair | ~93 (~279–285) | ~50–68 | trial_3 seats, −23 gone |
| 3 coverage | ~93 (held) | ~85–93 | miss-floor lifted |
| 4 harden | **≥90 (≥270)** | **≥90** | reps + phase_1 validate |

## Risks & mitigations
- **Sim noise (sd 3–18, prox↔miss flips):** n≥3 IQM+CI, matched seeds, ≥1-insertion overturn bar, drop harness hangs. Never trust single-trial per-config deltas or 60 s screens (truncate insertions 119.4→71.3).
- **GPU↔Gazebo contention:** strict serialize; orchestrator train-block/collect-block scheduling; budget full gates (5 h) explicitly.
- **DAgger corrupts approach** (ensembling-ON precedent): aggregate not replace, cap corrective ≤35%, screen approach quality pre-gate.
- **Wrench sim fidelity unknown:** validate non-degeneracy Phase 0; fallback = axial-distance scalar.
- **Razor-thin official envelope:** always gate the official-*family* (|yaw|≥1.8 neighbors), not the 3 exact poses.
- **Teacher caps student:** corrective labels must be decisive monotonic push (RLDG), not the decelerating oracle tail.
- **Standing/branch (load-bearing):** repo pinned to May-30 `main`; live comp is `phase_1`. Confirm standing + port before submission; #595 helps us.

## Total calendar
Serialized on one GB10: **Phase 0 ~1 d · 1 ~2 d · 2 ~1.5 d · 3 ~1.5 d · 4 ~1 d ≈ 7 working days (~1 calendar week + buffer to 9 days).** Official-/300-only path (skip Phase 3) reaches ~279 in **~4.5 days**. Kill-gates cap downside: if Phase 0 falsifies the thesis, we've spent one day before pivoting to the head swap — the cheapest possible test of a data-first bet.