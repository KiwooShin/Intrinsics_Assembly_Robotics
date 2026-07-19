# Execution Plan — Break the Last-Inch Attractor via Policy-Class Swap (Competing Draft)

**Stake:** The deterministic 0.75M head is the binding constraint, and the cheapest decisive test of that thesis is **one 17-min retrain** that swaps the head *and* de-aliases the near-port observation — no new data. I diverge from the method-analyst's demotion of the head: their ACT-scripted-ablation caveat is real but misapplied. Our bimodality is **observation aliasing** (near-port RGB ≈ seated RGB → mean-regressed twist ≈ 0), not human action noise — so a flow head *plus* wrench-in-obs attacks the exact mechanism. Lead with policy class; add data only where forensics prove coverage holes (Phase 2). RL is the ceiling-raiser, not the opener.

Files: trainer `opt/train_v3.py`, deploy `DeployACT.py` (receding-horizon MODE_POSITION, 4-of-K @ 18 Hz), prep `training/prepare_dataset.py`, oracle CheatCode, eval suite `eval_suite_smoke/`.

**Wall-clock primitives (this hardware):** 1 retrain = 17 min (Gazebo unusable at 0.05× RTF → **serialize**). 1 trial = 6.5 min wall. **Officials screen** n=3 = 9 trials ≈ **1 h**. **Full 15-cfg gate** n=3 = 45 trials ≈ **5 h** (run backgrounded, no concurrent train). All gates: 180 s trials, matched seed, IQM+CI, **≥1 genuine insertion is the hard overturn bar**; sub-5-pt deltas are noise.

---

## Phase 0 — Harness + signal hardening (0.5 day, must precede all gates)
**Goal:** trustworthy gates + confirm the two levers are live.
**Tasks (sub-agents in parallel, GPU-free):** (1) fix the ~1070 s `completed=False` harness hangs that poisoned prior reps tables; (2) one-command orchestrated n≥3 matched-seed eval (officials-only + full-15) emitting IQM/CI/insertion-count; (3) **validate `/fts_broadcaster/wrench` fidelity in Gazebo** — log wrench during a CheatCode seating: must show non-degenerate contact spike vs ~0 in free space; (4) stall-state logger (dump TCP+GT-port when plug-port distance plateaus 0.05–0.08 m) to seed Phase-2 DAgger.
**Gate:** eval harness reproduces v2_wide 119.4±noise on officials at n=3; wrench spike SNR ≥ ~3× at contact.
**Kill:** wrench is degenerate/flat at contact → drop wrench-in-obs from Phase 1 (head + tail-curation only), re-rank residual-RL up.

## Phase 1 — Expressive head + wrench/height obs + tail curation (~2 days) — THE THESIS
**Goal:** flip **0 → nonzero insertions** on the officials; convert last-inch stalls to partial/full seats.
**Tasks (one integrated retrain, then iterate):**
1. **Head swap:** replace deterministic twist regressor with a **flow-matching action head** (primary; 1–10-step sampling fits the 4-of-K@18 Hz deploy budget with huge margin) on the *same* 3×128 CNN encoder, K=16. Diffusion (DDIM 10-step) is the equal-cost fallback; CVAE is the low-risk-but-likely-weak fallback (ACT scripted caveat). Modestly widen the decoder (0.75M is tiny). arXiv:2303.04137 / 2304.13705.
2. **Wrench + axial-height obs:** concat recorded 6D wrench + scalar TCP-to-port axial distance → state 7D→14D; re-normalize. Collapses the aliasing at its source (FMB 11/25 vs 2/25; FILIC +10–20 pp).
3. **Tail curation (near-free relabel):** in `prepare_dataset.py`, drop/down-weight the terminal |v|≈0 seated frames; magnitude-weight L1 so the ~30–50 push-in frames stop being averaged to zero.
**Wall-clock:** ~4–6 head/hparam variants × (17 min train + 1 h officials screen) ≈ 1.25 h each → ~1 day; then one 5 h full-15 gate (backgrounded).
**Gate (n=3):** officials **≥1 full insertion** AND official per-trial median **≥60** (partial-band 38–50 + emerging seats), no new −24 collisions. Adopt only if it clears v2 by >5 pt with an insertion.
**Kill:** flow **and** diffusion heads, with wrench + curated tail, yield **0 insertions and 0 partials** (still dead-stall at 5–8 cm) at n=3 on officials → the head class is *not* the bottleneck; the limit is perception (sub-pixel port at 128 px) or control. Pivot: DINOv2/VGGT encoder upgrade (Sec-5 briefs) or jump to Phase 3 residual-RL on the frozen v2 base.

## Phase 2 — Last-inch DAgger + coverage replay + SC oracle fix (~2–3 days)
**Goal:** the stratified target — lift the 10/15 miss-floor + SC −23 configs so the Phase-1 head can seat everywhere.
**Tasks:** (1) **Auto-DAgger:** privileged-reset to the Phase-0-logged 5–8 cm stall states, CheatCode labels decisive closing velocity, aggregate ~30–50 demos, retrain (DAgger 1011.0686; JUICER: 10+corrective beat 50). (2) **MimicGen-style pose-replay** existing demos into the empty |yaw|∈[0.5,1.5] + SFP-port_0 + rail0 cells (GT poses in hand; no new oracle rollouts). (3) **SC oracle fix FIRST** (teacher bug, not model): descent-floor ≈−0.007 + pose-conditioned entrance waypoint, re-validate zero-contact under eval-band yaw+distractors, *then* collect ~12–16 SC demos incl. rail2 (currently 0). Class-weight SC in loss.
**Wall-clock:** ~40 DAgger + ~16 SC collects × 6.5 min ≈ 6 h collection + 3–4 retrains + screens; one 5 h full-15 gate. Serialized against training.
**Gate (n=3, full-15):** DAgger'd + coverage configs move miss-floor(+1)→proximity/seat; **stratified mean ≥45**, officials ≥82, **zero clean −23 on SC**.
**Kill:** DAgger corrective demos + retrain fail to move their own target configs above proximity (n=3) → covariate-shift model is wrong for these cells; the barrier is approach-coverage capacity of the encoder, not the last inch → escalate those cells to Phase 3, ship Phase-1+partial-Phase-2 as interim.

## Phase 3 — Off-policy residual RL (contingent, ~3–4 days) — ceiling to >90
**Goal:** close the last gap to oracle-level everywhere.
**Trigger:** run only if post-P2 stratified mean <90.
**Tasks:** freeze the Phase-1 expressive BC base; additive residual (small α, action-magnitude clamp) trained **off-policy** (TD3+RLPD/REDQ — the *only* sample budget that fits single-env Gazebo; ResiP's 500M-step PPO is infeasible here) on the **engine insertion event** as sparse reward (ground-truth, free). Imitation/DTW regularizer (AutoMate) guards the clean approach. arXiv:2509.19301 (~200× vs PPO; 14→64% in ~15 min real); HIL-SERL ~100% insertion.
**Wall-clock:** RL interaction is throughput-bound at 6.5 min/rollout; expect ~10²–10³ rollouts → serialize hard, run backgrounded overnight, gate on full-15 at n=3.
**Kill:** residual regresses the frozen-base approach (α too large / reward hacking) or no insertion-rate gain after ~1 day of interaction → revert to Phase-2 checkpoint and ship it.

---

## Expected score trajectory (per-trial engine score)

| Stage | Official /trial | Official /300 | Stratified mean/100 | Basis |
|---|---|---|---|---|
| **Now** | ~40 | 119.4 exact / 97.7 suite | 5–8 | measured |
| **+P1** head+wrench+tail | **55–75** | 165–225 | 12–25 | last-inch *is* the whole officials prize (forensics §1b, §3): all 3 officials already at proximity; head lets push-in mode fire; partial→full band |
| **+P2** DAgger+coverage+SC | **82–93** | 250–279 | **45–68** | forensics reachable-config arithmetic (11×93 stratified; +60 on SC official_3) |
| **+P3** residual RL | **90–97** | 270–291 | **75–90** | RL insertion lit: HIL-SERL/ResiP ~99–100%, RLDG 96→100% |

Reasoning: P1 carries the **official /300** almost entirely (+50–160), because tier-2 is gated on tier-3>0 and all officials already reach proximity — only the seat is missing. The **stratified >90 target** is P2-gated (coverage) and P3-topped (reliability); P1 alone caps stratified ~25 (10/15 still miss).

## Risks + mitigations
- **Multimodal head samples "stop" 50/50 near port** → pair with wrench + tail-curation (P1 removes the stop mode from data and gives a signal to condition on). This is *why* the three P1 tasks ship together, not sequentially.
- **Sim wrench degenerate** → Phase-0 gate catches it; fall back to height-scalar + tail-curation.
- **Noise floor masks gains** → n≥3, IQM+CI, 180 s (never 60 s — truncates v2 119→71), ≥1-insertion overturn bar.
- **GPU–Gazebo contention** → strict serialize (CLAUDE.md rule); backgrounded 5 h gates when not training; sub-agents parallelize only GPU-free work (CSV forensics, harness, collection queueing).
- **Diffusion inference at 18 Hz** → use flow/DDIM few-step; ~900× headroom.
- **Live-comp branch drift (load-bearing):** repo is May-30 `main`; live Phase-1 is the `phase_1` branch (multiple SC ports/rail, PR #595 *contiguous* force penalty). **Confirm standing/branch before P2 SC work** — port the plan onto `phase_1` if competing.
- **Over-collecting rail0** (confound: failure tracks yaw, not rail count) → target the empty yaw band, not rail0.

## Calendar
P0 0.5 d · P1 2 d · P2 2–3 d → **strong submission ~5–6 days** (officials ~250–279, stratified ~45–68). +P3 3–4 d → **~9–10 days to >90 everywhere**. From 2026-07-12, both fit inside the ~Jul-28 Phase-1 submit anchor with buffer; P3 is the tight item — start P0–P1 immediately, decide P3 by day 6 against the P2 gate.