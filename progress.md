# Progress Log — autonomous research run (started 2026-07-12)

Rule: one entry per 2 hours, max 3 sentences + next-2h plan. Newest on top.
Latest demo video: `demo/` (none yet — pipeline video pending first milestone).

---

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
