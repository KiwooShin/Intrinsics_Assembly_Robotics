# INSERTION_PLAN — learned insertion specialist (two-policy), insertion-first

**Adopted 2026-07-20 00:25** at user direction ("prioritize the insertion task …
one model for detecting and aiming, the other specialized for insertion only when
proximal … use all available sensors, joint and policy-learning mechanism … start
easy then harder"). Derived from a 4-agent research workflow (3 web-literature
sweeps + 1 codebase feasibility audit; full artifacts in the session transcript
`wf_16427555-af7`). This supersedes the 04:00 dead-band retrain as the **lead
thrust**; the dead-band/approach retrain becomes complementary follow-on work
(it fixes *reach*; this fixes *seat*).

## Why insertion-first is also the highest-ROI move

- The officials and cfg_005 already **reach** clean proximity (~0.05 m) and only
  fail the last inch — a seat is tier-3 **+53/trial (≈+3.5 ab5 composite/trial)**,
  the single biggest lever on the board. cfg_001/dead-band configs don't reach at
  all, so the approach retrain can't score until *something* also seats.
- We own a **seating teacher**: CheatCode seats officials and 3/7 SC poses. The
  decisive research finding (all 4 agents) is that owning a successful teacher
  turns insertion from a slow-RL problem into a **supervised** one — BC the
  teacher's last-inch, RL optional and deferred. Sim at 0.05× RTF makes
  PPO-from-scratch (IndustReal/AutoMate: 1e5–1e6 trials) infeasible; BC needs zero
  rollouts.

## Architecture — two policies, geometric stall handoff (already scaffolded)

```
[learned approach policy]  --reaches port vicinity, then STALLS (speed<0.01 m/s)-->
        │  handoff trigger = StallDetector (eval-legal: RGB/TCP/wrench only;
        │  NO port TF at scored eval — see Blocker below)
        ▼
[learned INSERTION SPECIALIST]  --all-sensor, force-aware, seats the plug-->
```

- The seam **already exists** in `DeployACT.insert_cable`: `GuardedDescentController
  .cycle()` returns `gstep.targets` once `StallDetector` latches; today those
  targets come from the *scripted* `GuardedDescent.advance`. We **repoint** that to
  the specialist's `_predict_chunk`→`_command_targets` via a provider that clones
  the existing `_AuxBearingProvider` pattern. New env gate `AIC_SPECIALIST=1`.
- Handoff is a **deterministic geometric/stall gate, not a learned classifier** —
  matches the strongest recent two-stage system, "From Reach to Insert"
  (arXiv:2605.04649), which uses an AABB proximity gate. At deploy we can only use
  the **speed-stall** boundary (the port TF that CheatCode uses needs
  `ground_truth:=true`, absent at scored eval).

## Observation vector — ALL deploy-legal sensors (per user directive)

Specialist input, assembled from the `Observation` msg (all present at train+deploy):

| Channel | Dim | Notes |
|---|---|---|
| 6-D wrench, **baseline-subtracted** Δf=f−f̄ | 6 | f̄ = mean over a short stable window at handoff. **The single biggest lever** — FILIC (2509.17053): +22 pp from EE-force, +12 pp from joint torque. |
| wrench short history / rate (≈4–8 frames) | — | exposes contact transients; raw compensated wrench beats spectrograms in sim (ManipForce 2509.19047). |
| TCP pose **relative to port anchor** | 7 | anchor = stall pose; keeps it port-location-invariant. |
| joint states | 7 | proprio backbone. |
| gripper state | — | *fixed weld in this sim (cable plugin); no gripper channel exists* — carried as constant/omitted. |
| RGB — wrist/closest camera | 128² | vision fixes lateral alignment; the plug body occludes the port at the last inch, so force carries the seat (SI-Diff 2605.12247, InsertionNet). Force+proprio-only variant is a valid fast first cut. |

Action: **K×6 TCP-velocity chunk, small K (4–8)** on the existing MODE_POSITION
impedance inner loop. Upgrade path: hybrid force/position action (push-to-seat axis
force-controlled) — PhaForce 2603.08342 / Force Policy 2602.22088.

## Data — last-inch segments of the teacher's successes (no new sim needed for v1)

- Corpus we already have on disk: **`ds_phase0` (44 valid) + `ds_phase2` (33 valid)
  = 77 oracle-success episodes, ~40.6k frames, full 3×RGB + 13-D proprio + wrench**.
  Terminal-window proxy ≈ **8.1k last-inch frames**.
- Extract the segment from approach→seat: keep frames from `(last_moving − lookback)`
  through the seat, i.e. the **inverse** of today's `--tail-trim`. The seat frame is
  re-derivable via `port_offset.robust_terminal` (the seat *timestamp* isn't
  persisted today — code change #3 fixes that for exactness).
- SC eval bags (`sc_oracle_reval*`) are **camera-less** → usable only for a
  force/proprio-only head, SC-only, thin. Not in v1.

## Curriculum — easy→hard, realized over the CONFIG distribution

**Blocker (feasibility audit):** there is **no config field to spawn the plug
proximal** — the grasp is a rigid gripper weld, plug pose = FK(joints)∘grasp_offset.
So a classic reverse-curriculum via plug state-reset (RFCL 2405.03379) is **not
expressible**. We realize easy→hard two ways instead:

1. **Deployment/eval curriculum:** first prove the specialist seats the *easy*
   configs — officials + cfg_005, which the approach already reaches well-aligned.
   Then SC, then offset/angled/dead-band once approach is fixed.
2. **Data curriculum (for DAgger rounds):** vary board pose (x,y,yaw) and
   grasp_offset in collection configs so the oracle produces aligned (easy) →
   lateral-offset → angular-offset seatings; auto-advance difficulty when success
   >80%, regress <10% (IndustReal SBC, 2305.17110).

## Phases

**P-INSERT-1 — BC specialist + handoff (fast, mostly offline). LEAD, now.**
1. Code change #1: inverse "keep-last-inch" selector in `opt/episode_prep.py`
   (mirror `terminal_tail_trim_length`; reuse `frame_speed`+`episode_bounds`) +
   `--last-inch` flag in `train_v3`/`TrainConfig`. Pure logic, unit-tested, no sim.
2. Code change #3: persist seat frame/time in `prepare_dataset.py` from
   `insertion_times` (exact window end). Pure logic, unit-tested.
3. Train specialist BC: ACT backbone (Zhao 2023, arXiv:2304.13705 — chunking is the
   biggest anti-compounding-error lever), `--last-inch --wrench --k 8`, small MLP
   head `state(13)+wrench-hist → K×6`, on the 77-episode last-inch corpus. GPU only,
   no sim → `ckpt/specialist_k8.pt`.
4. Code change #2: learned-specialist provider at the `_engage_descent` seam
   (`guarded_descent.py`+`DeployACT.py`), env gate `AIC_SPECIALIST=1`. Unit-test the
   provider logic off-sim.
5. Eval (sim window): approach→stall→specialist, **officials n=3**. GATE (noise-
   immune): **≥1 insertion_event** (baseline 0/9) AND no new collision; then SC/cfg_005.

**P-INSERT-2 — oracle-relabeled DAgger at force-detected stalls.**
Roll out the specialist; when observed wrench diverges from expected (jam/plateau →
TER-DAgger 2603.04038), reset the oracle from that state, relabel, aggregate,
retrain. Expert is algorithmic → free, no human. Fixes covariate shift.

**P-INSERT-3 — thin residual RL, asymmetric privileged critic (post-Jul-28).**
Only if BC+DAgger plateaus: small bounded residual on the frozen specialist, critic
fed CheatCode's privileged state, off-policy (RLPD 2302.02948 / ResiP 2407.16677 idea,
NOT on-policy PPO). Densify sparse `insertion_event` with an **SDF-to-seated** shaping
term (IndustReal — SDF *geometry* only, computable in Gazebo) + **SAPU** filter that
drops >1 mm interpenetration frames so the policy can't exploit fake Gazebo contacts.

## The 3 smallest code changes (feasibility audit)

1. `episode_prep.py` inverse last-inch selector + `--last-inch` flag (train_v3/TrainConfig).
2. Learned-specialist provider at `_engage_descent` seam (guarded_descent.py + DeployACT.py), `AIC_SPECIALIST` gate.
3. Persist `seat_time`/`insertion_frame` in `prepare_dataset.py`.

Each ships a sibling `tests/test_*.py` (pure-numpy, CPU-only) per CLAUDE.md §2. #1/#3
unit-test without ROS/GPU.

## Biggest blocker & the design constraint it forces

**No clean proximal reset.** Cannot spawn the plug at the port; the port TF (true
proximity) is only available under `ground_truth:=true`, i.e. during data collection,
never at scored eval. Therefore both (a) training-data proximal states and (b) the
deploy-time handoff must go through the **speed-stall boundary** (eval-legal
RGB/TCP/wrench), not a true port-distance signal. The specialist is trained and
triggered around "the approach policy has stalled near the port," not "the plug is at
the port." Everything else is reuse of existing infra.

## Risk & mitigation

- **Specialist regresses the shippable approach.** Mitigated: it only activates
  *after* the stall handoff, so pre-stall behavior is byte-identical; gate on
  officials n=3 no-regression before adoption; env-gated (`AIC_SPECIALIST=0` = today).
- **Gazebo contact fidelity** (penetration/jitter/jamming; Beltran-Hernandez 2020).
  Mitigate with the impedance inner loop, generous seat threshold, wrench-noise
  domain randomization if we add RL.
- **Last-inch corpus thin / open-loop teacher.** The oracle's last-inch *does*
  contain the successful descent-to-seat; isolating it + adding the wrench channel is
  the bet. If BC underfits, DAgger (P-INSERT-2) grows the set for free.

## Immediate next actions (sim idle → do offline now)

1. Implement code changes #1 and #3 + tests (offline). 2. Train `specialist_k8.pt`
on last-inch corpus (GPU, no sim). 3. Implement change #2 + tests. 4. First sim
window: officials n=3 with `AIC_SPECIALIST=1`, gate ≥1 insertion.

## Methods considered / referenced (traceability, CLAUDE §4)

Backbone: ACT (Zhao 2023, 2304.13705); alt Diffusion Policy (Chi 2023, 2303.04137).
Two-stage handoff: From Reach to Insert (2605.04649). Force representation: FILIC
(2509.17053), ManipForce (2509.19047), Making Sense of Vision & Touch (1810.10191).
Teacher→specialist: InsertionNet (2104.14223), AutoMate assembly-by-disassembly
(2407.08028), Compliant Residual DAgger (2506.16685), TER-DAgger (2603.04038).
Curriculum/reward: Reverse Curriculum (1707.05300), RFCL (2405.03379), IndustReal
SBC+SDF+SAPU (2305.17110). Sample-efficient RL (deferred): RLPD (2302.02948), SERL/
HIL-SERL (2401.16013 / 2410.21845), ResiP residual (2407.16677). Rejected for our
budget: PPO-from-scratch (Factory 2205.03532 substrate assumes GPU-parallel SDF sim).
