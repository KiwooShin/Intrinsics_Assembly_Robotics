# Progress Log — autonomous research run (started 2026-07-12)

Rule: one entry per 2 hours, max 3 sentences + next-2h plan. Newest on top.
Latest demo video: `demo/` (none yet — pipeline video pending first milestone).

---

## 2026-07-18 12:50 — Fratricide solved: the eval "freezes" were peer-kills, not deadlocks

Live forensics on a "frozen" sim proved there was never a hang: trials share
one global ROS graph, every bringup's cleanup did global name-matched kills,
and orphaned bringup sessions from earlier incomplete kills detonated their
EXIT traps into peer trials — one victim had actually COMPLETED and was
killed during post-score homing; 321 leaked orphan nodes (aic_adapter/tf
publishers) were burning ~3 cores and corrupting /tf. Fix (committed, 165
tests): process-group-scoped teardown with zero name matching, reap-on-
timeout in the runner, one-time preflight orphan sweep, sequential-only
invariant documented. System purged to zero residuals; the clean 3-ckpt
batch on the 60-sim-s suite launched 12:12 (p1_k8→v2_wide→p1_k16, ~5-7 h).

**Next 2 h:** batch grinds with per-trial verification; on EVALBATCHDONE the
paired comparison + analysis + dashboard + demo video finally land.

## 2026-07-18 10:50 — Harness hardened after cascade postmortem; scoring bug hunt

Three harness defects were found and fixed this morning: the agent-waiter
stall pattern (banned; detached scripts mandated), a torch-less policy
launcher that zeroed every trial (venv pinned in runner + batch, tests
updated), and a two-batch collision where an overly-broad cleanup pkill
fratricided the sibling (flock single-instance lock + narrowed pattern,
verified live). p1_k8 is confirmed HEALTHY offline — its 0/100 was artifact.
Now chasing the last validity bug: policy trials that fail to insert
frequently end with NO scoring.yaml (engine torn down pre-scoring; suspected
success-only completion regex in the runner) — SCORE-FIX agent verifying and
patching while the locked batch grinds v2_wide (3/15).

**Next 2 h:** SCORE-FIX verdict → likely batch restart with fixed runner →
finally-valid paired scores for v2_wide/p1_k8/p1_k16.

## 2026-07-18 08:50 — Overnight stall root-caused; self-driving eval batch launched

The suite evaluation stalled after run 1 of 3 (~01:23): the eval agent parked
on a self-armed completion monitor that never fired — the third such failure —
and the orchestrator heartbeat was also silent overnight, so nothing caught
it. Fix (user-directed): multi-stage pipelines now run as detached resumable
scripts (`eval_batch.sh`, mirroring collect_campaign.sh) with progress logs +
DONE markers; agent-waiter pattern banned in CLAUDE.md §6. Batch relaunched:
p1_k16 evaluating now, v2_wide control next (~3 h). ⚠️ First result is
alarming: p1_k8 scored ~0/100 (0/15 insertions, 14 miss + 1 collision) — the
v2_wide control through the same harness will show whether it's a harness
wiring bug (checkpoint not loading) or a real P1 training regression.

**Next 2 h:** batch grinds runs (b)+(c); on EVALBATCHDONE → paired compare +
analysis agent + root-cause of the p1_k8 collapse.

## 2026-07-17 21:45 — SC oracle fixed (19→94) + SC data collected; retrain launched

The CheatCode SC fix validated at 94.1–94.3 with zero contacts on both test
configs including official trial_3 (one floor tune to −0.005, committed) —
trial 3's ~100 pts are now reachable — and the SC collection pass kept 5/8
demos (3 drops: 59.0 sub-threshold + two ~65 partials; floor micro-tune noted
as future work). Dataset finalized at **60 episodes** (39 SFP + 5 SC Phase-0
with wrench/joints, 16 legacy); RETRAIN-P1 agent now owns the GPU training
p1_k8 (shift8) and p1_k16 (shift4) per the W1 winners.

**Next 2 h:** retrain completes → paired suite smoke eval of all 6 checkpoints
(first-ever SC-capable models) → analysis agent + policy rollout video +
dashboard refresh → overnight failure-driven Phase-2 collection.

## 2026-07-17 19:50 — Phase-0 collection COMPLETE: 39/40 KEEPs, 55 episodes total

The Phase-0 campaign finished with a 97.5% keep-rate (39 KEEPs, oracle scores
92.6–94.0, one config-specific drop) — the dataset is now 55 episodes (39
Phase-0 stratified with wrench/joints + 16 legacy), 3.4× this morning's size.
SC-VAL agent now owns the sim to validate the CheatCode SC-entrance-frame fix
(gates trial 3's ~100 pts; SC collection follows if ≥85); GPU stays idle until
sim work completes per the RTF-contention rule. Then: retrain (shift-aug, both
Ks) → paired suite eval of all 6 checkpoints → analysis + policy video +
dashboard refresh.

**Next 2 h:** SC validation verdict + SC collection or SFP-only fallback;
retrain launch; first suite scores of the new-data models.

## 2026-07-17 17:15 — ▶ RESUMED for a 48 h autonomous run (until ~2026-07-19 17:00)

Environment verified identical to the pause state (GPU idle, data intact:
22 KEEPs + 16 legacy episodes, repo clean at 8380d2c) and the Phase-0
campaign is relaunched — resumability worked, ~18 demos remain (~2 h). Plan
unchanged from the runbook: at CAMPAIGNDONE run CheatCode SC validation
(15 min, gates trial 3's ~100 pts) then the retrain (shift-aug, both Ks),
then the paired suite evaluation of all checkpoints + analysis + policy
video + dashboard refresh.

**Next 2 h:** finish the campaign to 40 demos; prep retrain/SC-validation
launches. Standing user items: Phase-1 deadline (portal check) and the
193 GB stale-bag deletion approval.

## 2026-07-12 20:40 — ⏸ PAUSED by user (GPU needed for another project)

Paused mid-campaign at **22/40 Phase-0 KEEPs** (plus 16 legacy episodes = 38
total); all processes stopped cleanly (GPU 0%, sim down, partial bag of the
interrupted demo deleted), all completed work committed and pushed through
`898ad05`+. Best score on record: **119.4/300** (position-mode DeployACT,
`v2_wide.pt`); demo video in `~/demo/`, live dashboard published.

**TO RESUME (in order):**
1. Collection: `cd ~/work/Intrinsics_Assembly_Robotics && PLUG=sfp nohup bash
   collect_campaign.sh >> ~/training/ds_phase0/campaign.log 2>&1 &` — resumable,
   skips the 22 converted episodes, ~18 demos ≈ 2 h remain.
2. At CAMPAIGNDONE: retrain (opt/train_v3 + shift-aug per opt/results/w1_report.md
   winner, both K∈{8,16}) on ds_phase0+ds_wide+smoke, AND run the 15-min
   CheatCode SC validation (plan in YAWFIX agent report / SESSION_REPORT §SC-DIAG;
   success = ≥85 & contacts 0 → then collect the 8 SC configs).
3. Then: paired eval on eval_suite_smoke (15 cfgs) of v2_wide/v3_wide/w1_best_k8/
   w1_best_k16/new ckpts via eval_suite.py; analysis agent; policy rollout video;
   dashboard republish.
Open user items: Phase-1 deadline conflict (Jul 14 vs Aug 4 — check portal);
193 GB stale bags in ~/aic_results awaiting deletion approval.

## 2026-07-12 ~19:00 — Campaign grinding; SC root-caused; dashboard live

Phase-0 collection is at 10/40 KEEPs (oracle 92.6–93.9, 100% keep-rate) after
a watchdog save (leaked recorder + un-launched campaign caught and fixed) and
a mid-flight config swap that fixed a board-yaw sampling bug (77% of configs
were out-of-distribution). SC failure root-caused: CheatCode has no SC branch
and rams the rotated SC port frame — retarget to the `_entrance` frame is
coded/tested, awaiting the post-campaign sim window. Scoreboard dashboard
published as a live artifact; oracle demo video in ~/demo/.

**Next 2 h:** campaign reaches the ~30-KEEP retrain trigger → retrain agent
(shift-aug, both Ks) + first paired suite evaluation of all five checkpoints;
then SC oracle validation at campaign end.

## 2026-07-12 ~17:00 — Arm unfrozen: 36 → 119/300 (+230%)

Root cause was open-loop reference drift in the controller's velocity mode
(not a deadband); DeployACT now does receding-horizon MODE_POSITION with
per-inference re-anchoring to measured TCP, taking the official eval config
from 36.1 to 119.4/300 with clean directed approaches to 5–6 cm of the port
in all three trials. New bottleneck identified: last-inch stall (stationary-
heavy training endings make near-port views predict ~zero velocity) — exactly
the DAgger/data-lane target. W1 regularization matrix still training at 96%
GPU.

**Next 2 h:** Phase-0 stratified collection campaign starts on the freed sim
(48 configs over 12 strata, storage-light, first-bag wrench validation +
oracle demo video); W1 matrix completes and reports; first retrain planned at
+40 demos.

## 2026-07-12 ~16:30 — Deployment unblocked; arm-freeze root cause isolated

The torch/rclpy blocker is FIXED (venv overlay, zero system pollution) and
`v2_wide.pt` scored its first real trials — 73.36/300 on the official eval
config — but diagnostically the arm barely moves (EE path ~0.00 m; mm/s
velocity commands at 4 Hz die below the impedance controller's response,
while the oracle trained in MODE_POSITION). Eval harness (53 pinned configs,
121 tests green) and pipeline upgrade (wrench+joints, stratified strata mode
with distractors) both landed on main; analysis agent flagged the sweep's K=8
"win" as a proxy-metric artifact — adoption now gates on suite score.

**Next 2 h:** UNFREEZE agent diagnoses aic_controller and converts the policy
to position-mode chunk integration (time-boxed sim use), W1 agent screens
shift-aug + proprio-dropout × K on GPU; then Phase-0 stratified collection
takes the sim. Flag for user: 193 GB of stale May/June scoring bags in
~/aic_results await a deletion decision (permission-gated).

## 2026-07-12 ~15:30 — Recon done, execution running

All 5 recon agents reported and were synthesized into ResearchPlan.md (11
evidence→decision findings; headline: scale demos 16→150+ stratified, val-L1 is
not the metric, F/T+dropout+shift-aug are cheap wins); 50 unittests + refactors
landed on main (`df856df`). Four implementation agents are running: runtime-fix
has VNC/Gazebo/eval-config up and is scoring `v2_wide.pt` in-sim, optimization
agent is benchmarking at 96% GPU, A0 (wrench/joints + stratified configs with
distractors) and B1 (matched-seed eval harness) are building. Liveness sweep
15:31: no hung processes; GPU busy as intended.

**Next 2 h:** first real engine score for `v2_wide.pt` (baseline table row),
commit A0/B1/opt outputs as they land, start Phase-0 stratified demo collection
on the freed sim, spawn the analysis sub-agent on the first result batch.

## 2026-07-12 ~13:00 — Session start

Orchestration kicked off: 5 Opus recon sub-agents launched in parallel (Phase-1
requirements, code/environment audit, and 3 literature lanes: IL/Transformer
architectures, RL/refinement, data strategy), and repo-wide engineering rules
were codified in `CLAUDE.md` (Google style + unittests, git workflow, 2-hourly
reporting, public checkpoints allowed). Environment state from 24 days ago
(deployment-runtime blocker, 16-episode dataset, `v2_wide.pt` checkpoint) is
being re-verified by the audit agent before any execution decisions.

**Next 2 h:** collect recon reports → synthesize the research + execution plan
(`ResearchPlan-2026-07.md`), decide the deployment-runtime fix, launch first
implementation sub-agents (deploy-and-score the existing checkpoint; restart
data expansion), and start the 48-hour heartbeat.
