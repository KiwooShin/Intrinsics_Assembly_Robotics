# EXECUTION PLAN — Close the Last Inch as a Control Loop (competing draft)

**Thesis (opinionated).** The scored bottleneck is a *contact-control* failure, not perception or head-expressiveness. The learned approach already parks the plug at 0.05–0.08 m on all three officials; the deterministic head then commands ~zero velocity and the arm sits. But the fix the analysts rank #1 (multimodal head / DAgger) is the *slow* path to the same place. The fast path, verified in code and by the two strongest public teams (Datameister #1 = decomposed primitives; MacCody = force-guarded state machine), is: **splice a wrench-driven, impedance-controlled insertion primitive onto the learned approach, and rebuild the oracle to be force-guided so everything downstream inherits "push until seated."** The interface already exists: `set_pose_target(..., stiffness, damping)` (DeployACT.py:191) is impedance control today; `wrist_wrench` is in every `Observation` and ignored. CheatCode (CheatCode.py:255–268) descends **open-loop to a fixed floor (−0.015 SFP)** with zero force feedback — that is the attractor's *source*, reproduced in the demos. We attack the control loop directly; learned-head expressiveness and residual RL are held as ceiling-raisers, not preconditions.

---

## Phase 0 — Instrument the contact loop (½ day, ~4 h)
**Goal:** prove the two signals the whole plan rests on are non-degenerate in Gazebo.
**Tasks:** (1) Log `/fts_broadcaster/wrench` (+ `fts_tare_offset`, toolkit PR #596) through a scripted CheatCode descent on official_1/2/3 and 2 SC poses; confirm a clean, monotone Fz rise (>5 N) at seating vs ~0 during approach. (2) Sweep Z-axis stiffness (90→15 N/m) in `set_pose_target` to find the compliant band that seats without ramming. (3) Build the **privileged reset harness** (seat plug via GT, retract N mm along port axis) — reused in Phases 1–3. (4) Extend DeployACT's 30 s `budget` (DeployACT.py:156) to the full trial window so a search primitive has time.
**Gate (n≥3):** Fz spike ≥5 N at contact on ≥3/3 seeds AND a compliant descent seats official_1 ≥2/3. **Kill:** if Gazebo wrench is flat/degenerate at contact → force-guarding is impossible; pivot immediately to Phase 3 (residual RL with engine reward as the only closed-loop signal) and geometric floor micro-tune. This is the single highest-risk assumption — settle it first.

---

## Phase 1 — Force-guarded insertion primitive + hybrid deploy (1.5 days)
**Goal:** convert the officials' proximity stalls into seats — the +160/300 prize.
**Tasks:** Author `GuardedInsert` primitive (RGB/wrench only, **no GT** — eval-legal): trigger on approach-stall detection (TCP-velocity plateau <0.01 m/s within proximity), then run compliant guarded descent with low Z-stiffness + a bounded **XY spiral/Lissajous search** (radius ≤ port half-width to avoid the −24 off-limit contact), pushing until Fz threshold *and* depth-consistent stall = seated, backing off on hard contact (>15 N) to re-search. Deploy hybrid: `DeployACT` approach → hand off to `GuardedInsert`. Keep the frozen v2_wide approach untouched.
**Gate (n≥3, 180 s):** official 3-config **≥2/3 configs seat** (full-insertion event fires), n≥3 reps each, mean /300 **≥220**. Screen on the 3 officials + 2 reached strata (cfg_005, off_2) first at 60 s for reached-rate, confirm inserts only at 180 s.
**Kill:** if the primitive cannot seat any official at n≥3 (handoff pose too misaligned for a bounded search, or search trips −24 penalties net-negative) → abandon pure-scripted insert, carry the approach forward, jump to Phase 3 residual RL for the last inch. Do **not** iterate the primitive past ~1 day.

---

## Phase 2 — Force-guided oracle + coverage: make the approach reach everywhere (2 days)
**Goal:** the primitive only fires where the approach reaches proximity; 10/15 stratified configs never approach (moderate-|yaw|∈[1.2,1.5] SFP-port_0 dead-miss + SC collisions). Fix the *approach* and the *teacher*.
**Tasks:** (1) **Rebuild CheatCode's descent as force-guided** (replace the fixed-floor `while` loop with the Phase-1 guarded logic) — this fixes the SC keep-rate 3/8 (descent-floor partials + the −0.005/−0.007 floor guesswork in `cheatcode_targeting`) at the *source*, so regenerated demos teach seating, not stopping. Re-collect ~16 SC (incl. rail2, currently 0 demos) + ~16 moderate-yaw SFP-port_0 at eval-band, ~6.5 min each (~3.5 h collect, serialized). (2) **MimicGen-style privileged-pose replay** of existing 93 eps into the empty |yaw|∈[0.5,1.5] band (reuses GT poses, near-free). (3) **Add 6-D wrench to policy state (7→13 D)** + random-shift image aug (RAD/DrQ; free L1 win already measured). Retrain approach (17 min, sim idle).
**Gate (n≥3, 180 s, 15-config):** reached-rate (proximity-or-better) **≥12/15**, zero net-negative SC configs, stratified mean **≥45**. **Kill:** if force-oracle + coverage retrain still floors moderate-yaw (reached <9/15) → the approach is perception-limited, not data-limited; swap the 0.75 M CNN for a DINOv2/pretrained encoder (RGB-only, rules-legal w/ disclosure) before spending more collection.

---

## Phase 3 — Residual-RL / impedance polish to oracle level (2–3 days, ceiling-raiser)
**Goal:** close the gap from "seats most places, sometimes slow/jammed" to **>90 avg** (fast, penalty-free, near-universal).
**Tasks:** Off-policy residual RL (2509.19301 variant — **not** ResiP's 500 M-step PPO, infeasible at single-env 0.05× RTF; off-policy ~200× more sample-efficient, real-world 14→64% in ~15 min) on the **frozen** approach+primitive base, action = small residual on the impedance target, reward = **engine full-insertion event** (ground-truth, free — no learned classifier needed, unlike HIL-SERL). Small α, action clamp, DTW/imitation regularizer (AutoMate) to protect the clean approach. Serialize RL updates vs Gazebo rollouts per the RTF rule. Reserve for last — only run if Phase 2 plateaus <90.
**Gate (n≥3, 180 s, 15-config):** mean **≥90**, ≥13/15 seat. **Kill:** if RL regresses the approach or cannot converge within ~2 days of serialized rollouts → freeze and ship the best Phase-2 hybrid (expected ~68–80).

---

## Expected score trajectory (reasoning from evidence)

| Phase | Official /300 | Stratified mean/100 | Why (evidence) |
|---|---|---|---|
| Current (v2_wide) | 119.4 (exact) / 97.7 suite | 5–8 | Last-inch attractor; 0/45 insertions |
| **P1** primitive on officials | **220–270** | 8–12 | All 3 officials reach proximity → +75 each once seated (Forensics #1, +160 prize) |
| **P2** oracle+coverage+wrench | 250–280 | **45–68** | 11 reachable strata seat (Forensics: 11×93+4×1≈1027→~68); wrench disambiguates seated-vs-pushing (FMB 2→11/25) |
| **P3** residual RL | **270–290** | **85–93** | RL insertion reaches ~100% on connectors (RLDG/HIL-SERL); IndustReal SBC 83–99% |

Datameister's 293/300 proves near-perfect insertion is achievable in this sim with a primitive-based stack — our P1+P3 target sits just under it.

---

## Risks + mitigations
- **Wrench degenerate in sim (kills P1/P2/P3-reward-shaping).** → Phase 0 gates it first; fallback = TCP-motion-stall proxy for the seated signal + geometric search.
- **Spiral search trips −24 off-limit contact** (net-negative, esp. SC w/ distractor mounts). → radius ≤ port half-width, low force, back-off-on-contact; PR #595 makes force penalty *contiguous-time* (more forgiving). Screen SC configs explicitly.
- **Sim noise (sd 3–18) masks real deltas.** → every gate n≥3 reps, 180 s, IQM+bootstrap CI; 60 s only for reached-rate screening (it truncates insertions). No adoption on <4–5 pt effects.
- **RTF serialization stalls the loop.** → detached resumable scripts (`eval_batch.sh`/`collect_campaign.sh` pattern, CLAUDE.md §6 agent-waiter ban); never train and sim concurrently; watchdog on log growth.
- **Toolkit branch drift.** Repo pinned to May-30 `main`; live Phase-1 = `phase_1` branch (multiple SC ports/rail, scoring PRs #594/#595). → **verify competition standing/branch before Phase 2 collection** (load-bearing per Challenge Intel); if `phase_1`, port the primitive against its scoring first.
- **Pretrained-encoder disclosure.** DINOv2/π0 usage must be disclosed (rules require "developed during Challenge Period" + disclosure) — document if P2-kill triggers.

---

## Calendar
Phase 0: 0.5 d · Phase 1: 1.5 d · Phase 2: 2 d · Phase 3: 2–3 d, run serially with n≥3 gates between (each full 15-config × 3-rep 180 s eval ≈ 5 h wall; budget one per phase-end plus screens). **Total ≈ 6–7 calendar days** to the >90 gate; a shippable **P1 hybrid at ~240/300 lands in ~2 days** and is the fallback submission if later phases stall. Sequence is strictly gated — kill-criteria abort a branch rather than iterate it, preserving budget for the residual-RL ceiling move.

**Bottom line:** lead with control (primitive + force-guided oracle), not head architecture. P1 alone recovers the entire officials prize in 2 days; P2 makes the approach universal; P3 is the only move with a demonstrated path to oracle-level everywhere — and it's uniquely cheap here because the engine hands us a ground-truth insertion reward for free.