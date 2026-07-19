# PLAN_SCORE90 — Ladder to >90/100 average trial score

**Adopted 2026-07-19 12:40** from a 9-agent analysis workflow (5 evidence briefs,
3 competing plans, adversarial judge — full artifacts in
[docs/research_2026-07-19/](./docs/research_2026-07-19/)). User directives bound
into this plan: (1) two-policy architecture — a camera approach specialist plus
a force/proprioception insertion specialist; (2) step-by-step score ladder
23 → 40 → 50 → 60 → 70 → 80 → 90; (3) 4-hourly progress entries with avg score,
gap analysis, and next-4h plan, each preceded by multi-sub-agent analysis.

## Measurement (binds every rung)

- **Gauge**: `eval_suite_ab5` (3 officials + cfg_001 + cfg_005), 180 s, n=3 reps
  = 15 trials ≈ 1.7 h. Current baseline **23.1 ± 2.1** (v2_wide, n=15 trials).
- **Milestone confirmation**: full 15-config `eval_suite_smoke`, n=3 (~5 h).
- Continuous-score adoptions need IQM +≥5 (noise floor sd 3–18); insertion-rate
  claims need the **discrete ≥1-insertion event** (noise-immune) and n≥5–8 for
  rate comparisons. Discard `completed=False` (~1070 s) harness hangs and rerun.
- GPU serialization: never train while sim rollouts run (RTF 0.05×).

## Two tracks (judge §0b — schedule fact)

Live competition: `phase_1` branch, submit ~Jul 28. The >90-stratified goal
needs residual RL whose rollout wall-clock exceeds that window. Therefore:
- **Track S (submission spine)**: officials prize — 119.4 → ~279/300 via
  seat-on-reached-configs. Shippable in ~2–4 days; port to `phase_1` before submit.
- **Track L (ladder)**: the >90-average research line; continues past Jul 28.

## Phases (best-of-breed composition, judge §3)

**P0 — Instrument + de-alias (day 1, RUNNING as of 12:40 07-19)**
- ✅ Wrench validated offline from the 93 recorded episodes (6-D, peak |Fz|
  21–25 N at seating, ~19 N gravity baseline → baseline-relative deltas).
- Agent A (in flight): dataloader/trainer changes — terminal zero-velocity tail
  trim, wrench into state 7→13-D, push-in loss ramp (W=4), random-shift aug.
  One 17-min retrain per K.
- Agent B (in flight): **judge-selected first experiment** — `guarded_descent.py`
  (stall detector + approach-axis estimator + wrench-guarded compliant descent,
  eval-legal RGB/TCP/wrench only) env-gated into DeployACT (`AIC_GUARDED=1`) +
  `probe_batch.sh` (3 officials × 3 reps ≈ 1 h sim).
- **Gate**: probe seats ≥1 official (insertion event fires) → Track S is a
  control problem, proceed P1-S. Probe wrench flat/degenerate → kill force
  levers, pivot to residual-RL-only closed loop. Retrain shows tier-3 deepening
  on gauge → aliasing thesis confirmed, keep 13-D state.

**P1-S — Insertion specialist v1 (scripted, force-guarded) [rungs 40–60]**
Harden GuardedDescent into `GuardedInsert` (bounded search ≤ port half-width,
back-off on hard contact; −24 exposure measured in P0). Hybrid deploy: learned
approach → stall handoff → guarded insert. Gate: ≥2/3 officials seat at n≥3 →
official /300 ≥ 220.

**P1-L — De-aliased approach retrain [rung 40]**
Adopt the P0 retrain if gauge IQM +≥5 or any new insertion. Feeds the approach
that P1-S hands off from.

**P2 — Teacher repair + coverage [rungs 60–75]**
SC oracle FIRST (floor ≈ −0.007, pose-conditioned entrance waypoint, zero-contact
re-validation; Phase-2 proved bad SC demos convert misses into −23 collisions).
Then: force-guided oracle descent (replace the open-loop fixed-floor loop — the
attractor's source in the demos), last-inch auto-DAgger at logged stall states,
MimicGen-style privileged replay into the dead |yaw|∈[1.2,1.5] SFP-port_0 band
(cfg_000/004/008: reached 0/9) and rail0/rail2 holes.

**P3 — Insertion specialist v2 (learned) [rungs 75–90+, post-Jul-28]**
Distill primitive + corrective demos into a wrench+joints+gripper specialist
head (user's two-policy proposal, learned form). If BC saturates: off-policy
residual RL (arXiv:2509.19301 variant, NOT ResiP's on-policy PPO) on the frozen
base, reward = engine insertion event, small α, action clamp, DTW regularizer.

## Kill-criteria (inherited from the competing drafts)
- P0 retrain moves nothing in tier-3 → aliasing demoted; escalate to
  flow/diffusion head (contingency #1, not the lead — ACT ablation: CVAE ≈
  deterministic on scripted data).
- Primitive can't seat any official within ~1 day of iteration → stop iterating
  it; escalate to P3.
- Repaired SC oracle keep-rate still <5/8 → do NOT collect SC demos; bank SFP.

## Score trajectory (evidence-derived, forensics §5)
| Milestone | Officials /300 | Gauge avg | Stratified mean |
|---|---|---|---|
| Now | 119.4 (exact poses) | 23.1 | 7.75 |
| P0+P1 | ~220–270 | 40–60 | ~12–32 |
| P2 | ~279 | 60–75 | ~50–68 |
| P3 | held | 90+ | ~85–93 |
