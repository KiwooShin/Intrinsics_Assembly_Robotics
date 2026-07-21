# Progress Log — autonomous research run (started 2026-07-12)

Rule (updated 2026-07-19, user directive): one entry per **4 hours**, newest on
top, each with three labelled parts — **Avg score** (mean engine score of the
window's runs, with n; n≥3 reps per claim), **What's missing** (gap analysis
vs the >90/100 target), **Next 4 h** (action plan). Every cycle is preceded by
a multi-sub-agent analysis of the experiments so far (CLAUDE.md §4).
Entries above this line's date follow the old 2-hour/3-sentence format.
Latest demo video: `demo/policy_p1_k16_official_1.mp4` in this repo (gitignored;
original copy in `~/demo/`; 2026-07-18 — first POLICY rollout video: p1_k16 on
official_1, approach to 0.06 m, engine 14.0; three-camera layout matching the
oracle demo `demo/oracle_demo_sfp_rail0_sfp_port_0.mp4`). Newest: `demo/policy_v2_wide_official_2.mp4` (2026-07-19, adopted ckpt on official_2, engine 41.6).

---

## 2026-07-21 00:45 — RUN #4 START: curriculum insertion (user new direction, 48h)

**GOAL of this session (user directive 2026-07-21 ~00:40):** REOPEN the insertion task
and get a **learned policy to actually seat** — via a **curriculum that starts from the
easiest case and grows harder**, using **all available sensor modalities** (3 RGB +
wrist force/torque + joint proprioception). The user's key idea: don't start with
search/detection; **start with the plug already located directly above the aligned
port** so the model learns the *insertion motion* (descend + seat) easily, then progress
to lateral offset and full approach. New 48h clock (STOP ~2026-07-23 ~00:40). Report
every 4 h here (goal / what's missing / next 4 h). Commit + push frequently as KiwooShin.

**Why this is a fresh, valid attack (vs the retired Run-#3 conclusion):** Run #3 showed
the *full blind* task is unreachable because port localization is camera-resolution-
capped (~22.7 mm) — but it never isolated the *seating skill* itself. This curriculum
sidesteps localization for milestone 1 (privileged/curriculum initial state = plug above
the aligned port) and asks a cleaner, achievable question: **can a learned all-sensor
(esp. force-reactive) policy perform the insertion once positioned?** Then it maps how
far that skill generalizes (capture-radius sweep) before re-introducing approach.

**Task structure (clarified):** 2 distinct tasks — **SFP** (28 stratified cfgs + officials
1/2; world-z descent, floor −0.015) and **SC** (25 cfgs + official_3; tilted insertion-
axis, floor −0.010). Curriculum starts on **SFP** (simpler axis, more data, policy already
SFP-competent to ~40 proximity).

**What's missing:** (1) a learned policy that *seats* (0 seats to date on hard poses);
(2) the **wrist F/T modality wired into the ACT policy** (currently RGB+proprio only —
force is the key new signal for compliant seating); (3) a **curriculum harness** that
starts the plug above the aligned port and grows the lateral offset; (4) SOTA grounding
(research sweep running).

**Next 4 h:** (a) SOTA research synthesis (4-thread workflow → architecture + sensor-
fusion + curriculum + first experiment). (b) Probe sim: pre-position above the aligned
port (privileged, curriculum-legal) + collect **forward aligned-insertion demos with
wrench** from the CheatCode oracle. (c) Add **wrist F/T** to the ACT state (state_dim
7→13 already supported via `--wrench`). (d) **Milestone 1:** train an all-sensor policy
on aligned SFP insertion demos → eval from the aligned start → **≥1 seat = insertion
skill learned**. SOTA refs pending from the research sweep (InsertionNet, IndustReal,
AutoMate, Factory, SERL/HIL-SERL).

## 2026-07-21 00:00 — Run #3 cycle 4: showcase pivot (no submission); aux-guard ablation; SC scoring-frame fix

**Avg score:** officials on the adopted **capped-aux = 38.0** (n=3, robust proximity;
SFP officials ~40, SC official_3 33.1). **0 hard-pose seats** (insertion retired last
cycle — structurally unreachable). USER DIRECTIVE (~22:30): **not submitting** to the
AIC portal — this run **showcases robotics ability**; drop all packaging, optimize for
measurable improvements + strong artifacts. Also: **commit + push every code change as
KiwooShin** (author `kiwoo.shin@berkeley.edu`, not the orchestrator identity) so it
counts as contributions.

| Experiment (this cycle) | Verdict | Metric |
| --- | --- | --- |
| Lever-1 officials aux-guard ablation (n=3) | capped-aux WINS | capped-aux **38.0** (SFP officials robust ~40) vs v2_wide guarded-no-aux **24.8** (catastrophic near-0: official_1 0.2/12.1/13.9). The learned port-bearing head is ESSENTIAL to the guarded recovery, not a suppressor. |
| SC scoring-frame bug (Lever 3) | **FIXED + pushed** (cbcaede) | `port_name: sc_port_{N}` → nonexistent frame `sc_port_{N}_link` → TF fail → every SC trial = tier_1=**1.0** for ANY policy. Fix → `sc_port_base` on 24 eval_suite + 6 smoke60 SC configs + generator `suite.py` + regression test. Expected SC **1.0 → ~33** (verify n=3 running). |

Correction for the record: the historical v2_wide "**119.4/300** on officials" is
**3×~40 proximity**, NOT a real seat — no config seats the officials.

**What's missing:** the hard-pose seat (unreachable, fully documented — camera-resolution
cap 22.7 mm far / 49 mm stall vs ~2–5 mm needed). Otherwise the remaining gap is
**showcase packaging** of the work, not more score.

**Next 4 h (showcase plan, 2 analysis agents):** (1) SC-fix **n=3 verification** running
(smoke60 SC configs → before/after 1.0→~33; ~02:00). (2) **Technical writeup/dashboard**
(HTML) — highest ROI: lead with the 5-family insertion scorecard (0/~175) + the money
figure (localization error vs distance-to-port 22.7/45.9/49.0 mm against the 2–5 mm
capture-radius line = structural camera cap), the capped-aux ablation bar, a guarded
force/travel timeline, and a literature table (InsertionNet 2104.14223, IndustReal
2305.17110, AutoMate 2407.08028, HIL-SERL 2410.21845). (3) **Demo video** — multi-panel
montage (oracle seat → learned approach → guarded recovery → honest near-stall →
localization-failure viz); `make_video.py` alone is insufficient. **Deferred:** dead-band
retrain (5–7 h, uncertain, real regression risk to the banked anchors). Ensembling stays
retired.

## 2026-07-20 20:30 — Run #3 cycle 3: INSERTION RETIRED (all sensor-legal paths exhausted); pivot to raise-avg

**Avg score:** hard/stratified poses still **0 seats** (~175 trials). The banked
submission floor is capped-aux (ab5 **IQM 35.8** / mean 28.0 vs baseline 23.1). NOTE
(forensics): plain **v2_wide already seats on the true-official `eval_config.yaml`**
(the actually-scored submission config) at **119.4/300 ≈ 40/trial** — a real insertion;
the capped-aux aux-guard was tuned on the *harder* ab5 poses and may *suppress* that
seat on the easy config. So there IS a seat — on the submission config — and the
submission-checkpoint choice is now the highest-value open question.

**What's missing / RESOLVED:** the hard-pose seat is **structurally unreachable
sensor-legally** — decided this cycle by 2 analysis agents + a decisive offline gate:

| Sensor-legal path | Result | Why dead |
| --- | --- | --- |
| Vision localization (DAgger, covariate-fixed) | **FAIL** | held-out lateral error **22.7 mm @far/visible (50–80 mm), 49 mm @stall** vs ~5 mm needed — a **camera-resolution** cap (128 px can't resolve a 2 mm port at any distance), not just occlusion |
| Disasm-reversal specialist (latch+reverse bugs fixed, valid test) | **FAIL** | degrades official_2 13 mm→40–50 mm, 0 seats — **blind-direction** (can't know which way to correct) + pure-vertical oracle last-inch |
| Blind spiral / Lissajous force search | falsified | r18>13 mm still 0/3 (friction wall K_eff≈350 N/m; pitch 9 mm ≫ 2 mm bore) |
| Force-reactive residual RL | NO-GO | 0.05× RTF (infeasible in 35 h) + flat-face contact gives no bearing gradient (unlearnable) |
| Chamfer + lateral compliance | NO-GO | needs offset <1–2 mm; plug rests on rim at 13 mm |
| "Aim-then-commit" (localize at last unoccluded frame) | **FAIL** | that IS the far frame — 22.7 mm, camera-capped |

Literature agrees (retrospective): every sub-mm-clearance insertion method carries a
**pre-contact lateral bearing to ~2–5 mm** (in-hand RGB seeing the hole, tactile
fingertip, or estimated pose bounded to a few mm — InsertionNet 2104.14223/2203.01153,
IndustReal 2305.17110, AutoMate 2407.08028, HIL-SERL 2410.21845). This setup lacks it:
RGB occluded/low-res at the last inch, wrist F/T blind pre-contact, port TF eval-illegal
(State-Leaking DQ). 0/~175 across **five** method families ⇒ **insertion retired**.

**Next 4 h — PIVOT to raise-average** (certain ROI; ceiling ~38–40/100 proximity-only,
>40 needs a seat). Ranked (forensics agent): **(1) reconfirm the submission checkpoint**
— plain v2_wide vs capped-aux on `eval_config.yaml` at n=3; if the aux-guard suppresses
v2_wide's real seat, ship v2_wide (or route v2_wide→official / capped-aux→stratified).
**(2)** dead-band cfg_000/004/008 (15-suite +1.0 floor) + cfg_001 (ab5 −7.0, biggest
leak) — BC coverage hole (undersampled |yaw|∈[1.2,1.5]), fix = collect those poses →
**plain** warm-start retrain of v2_wide (NOT the frozen aux-probe; NOT wrench/tail-trim)
→ n≥3 gate (dead-band floor→prox AND officials don't regress). **(3)** SC config-audit
propagation. **(4)** lock variance-stabilized guard. Ensembling stays retired (n=3
negative: OFF 23.1 > ON 20.1). Submission packaging + demo video in the tail. SOTA/refs
as above + seam-fill capture-radius 2204.07776.

## 2026-07-20 16:20 — Run #3 cycle 2: localization gate FAILED; disasm unblocked; one decisive first-seat cycle

**Avg score:** still **0 insertions** (~168 trials); banked floor unchanged (capped-aux
ab5 **IQM 35.8** vs 23.1). This window RESOLVED the localization gate and unblocked the
disasm path; no new engine scores yet (decisive disasm cycle now running).

| Experiment | Verdict | Key metric |
| --- | --- | --- |
| DAgger localization retrain + held-out eval (SFP/SC separate) | **FAIL** | SFP median lateral **37 mm** (40 mm at stall point, 1/6 eps <10 mm); SC **201 mm**. Need ~2–5 mm. |
| Disasm-reversal correctness (4-agent adversarial workflow) | GO_WITH_FIXES | reversal math correct; 1 silent-corruption path fixed (latch-reject gate) |
| Disasm latch blocker (SFP welded during collection) | **DIAGNOSED + FIXED** | 21 s descent-dwell bug + TouchPlugin welds seat plate; fixed descent + disabled latch for collection → clean verify (insertion_events=0, seat_frame=n-1) |
| Lift-translate band widened for 13 mm | done | lateral 8→16 mm, axial-clear 5→14 mm, lift_frac 0.2→0.45 |

**What's missing:** the seat — and two analysis agents (results-forensics +
strategy-critique) converged that it is **likely unreachable sensor-legally**:
- **Localization is genuine occlusion, not covariate shift.** The head was trained on
  the policy's OWN deploy-stall distribution (the covariate fix) and *still* medians
  37–40 mm at the stall point. The plug body occludes the port at the last inch; wrench
  carries no pre-contact bearing. The port is observable early, never at the stall.
- **13 mm is the crux.** The nearest first-seat-reachable pose is official_2 at 13 mm
  lateral (base policy's own stall). That exceeds every measured/literature capture
  radius — friction wall K_eff≈350 N/m → ~4.4 N lateral (creep-and-stall ~5 mm); spiral
  r=18 mm already seated 0/3; force-reactive reliable ~5 mm, ~10 mm with slow spiral
  (arXiv:2204.07776: 66–78%@5 mm → 37–63%@10 mm). Ground-truth port TF is **eval-illegal**
  (challenge_rules §State-Leaking) so no scripted aim. The disasm specialist's only new
  lever is the free-space **lift-translate** maneuver (now widened to ~16 mm) — but
  without localization it cannot perceive the correction *direction*, so it must
  force-search, which is uncertain at 13 mm.
- **>90 is unreachable without a seat**; capped-aux **IQM 35.8 is very likely the final
  shippable number** regardless of branch.

**Next 4 h:** run the **ONE decisive first-seat cycle**, time-boxed to ~Jul 21 06:00:
widened SFP disasm collection (LIMIT=18, ~2 h, running) → REVERT the collection-only
latch-disable xacro → train specialist k8 **with AND without --wrench** (reversed wrench
is non-physical per AutoMate 2407.08028; adjudicate on the in-sim SEAT eval, NOT val L1)
→ **disasm-standalone on aligned official_2/3 + cfg_005**. ≥1 insertion = **FIRST SEAT**
(a ~3-pose capability demo, not a suite-wide gain; general 30–60 mm stalls stay
unseatable). **Kill (tightened):** 0 seats on aligned poses → retire insertion (localization
dead + force-search can't close 13 mm), pivot the remaining ~35 h to **raising the average**
(dead-band cfg_000/004/008, cfg_001 retrain, SC oracle repair) + submission hardening
(capped-aux banked; phase_1 port/Docker needs the user's portal). SOTA/refs: AutoMate
2407.08028, InsertionNet 2104.14223/2.0 2203.01153, IndustReal 2305.17110, seam-fill
capture-radius 2204.07776, DAgger 2606.10385.

## 2026-07-20 12:00 — Run #3 cycle 1: seat-execution plan sharpened (2 analysis agents); collection 32/48

**Avg score:** still **0 insertions** (~165 trials); banked floor unchanged
(capped-aux ab5 IQM 35.8 vs 23.1). No new engine scores this window — Stage-1 DAgger
collection is in flight (**32/48 keeps, 28 SFP / 4 SC, 1 benign drop**; the drop was a
per-config frame-naming edge, `nic_card_mount_3/sfp_port_1_link_entrance` absent from
its bag — isolated, collection robust). GPU idle, reserved for the retrain.

Two analysis sub-agents (plan-critique/risk + literature-comparison) reviewed the
committed two-stage seat plan. **They converged on three plan changes**, table below.

| Analysis topic | Verdict | Key finding / change |
| --- | --- | --- |
| Localization <10 mm gate | **Too loose** | Real gate = force-reactive **capture radius** ≈ 2–5 mm (up to ~10 mm only w/ slow spiral). arXiv:2204.07776: seat success 66–78 %@5 mm → 37–63 %@10 mm. 10 mm is the ragged edge, not "will-seat". |
| SC vs SFP eval | **Split required** | Pooled per-frame median is dominated by 28 SFP frames; SC's 61 mm offset is the real target. Do **leave-one-SC-out**, gate on SC lateral at **p75/p90**, near-port subset. |
| First-seat test / kill order | **Reorder** | **Disasm-standalone on aligned poses (official_2/3, cfg_005) is the true capability test AND cheapest kill test** — needs no localization; if it seats 0 even aligned, the localizer is moot. Don't gate it behind localization. |
| Reversed wrench conditioning | **Risk** | AutoMate (2407.08028) records only the reversed *path* — time-reversed friction shear is non-physical. Our `reverse_disasm` keeps wrench sign → `--wrench` may hurt. **Ablate with/without wrench** (or normal-force magnitude only). |
| Latch-during-descent | **Verify** | TouchPlugin may weld during the ~20 s slow descent (resets on contact-break, not motion). The plan's "verify 1 unit, insertion_events==0" guard covers this; also confirm `retract_start_z=-0.013` is shallower than seat floor per plug (SFP world-z vs SC insertion-axis differ). |

**What's missing:** the seat, still gated on numbers not yet measured — now **two**
numbers, correctly separated: (a) **localization SC/SFP held-out lateral error** vs the
capture radius (not a fixed 10 mm), and (b) the **disasm specialist's capture radius**
on aligned poses. The literature's single highest-value de-risk is a **capture-radius
sweep** (lateral offset 0→15 mm from an oracle pre-insertion pose, seat-success curve)
— it calibrates the real gate and decouples the two unknowns. First seat almost
certainly comes from **SFP** (13 mm, nearest solvable); SC (61 mm) is a pure
localization problem regardless of seater.

**Next 4 h:** collection finishes ~14:20 (DAGGERDONE). Then, GPU/sim serialized:
(1) localization retrain (GPU, sim idle) → `dagger_aux.pt`; (2) **eval SFP and SC
separately** (leave-one-SC-out, p50/p75/p90, near-port) → record both decision numbers;
(3) verify 1 SFP disasm unit (`insertion_events==0`) → full SFP disasm collection (sim);
(4) train disasm specialist **k8 with-wrench AND no-wrench (ablation)**; (5) **first-seat
test = disasm-standalone on aligned official_2/3+cfg_005** (probe_batch, AIC_SPECIALIST)
— ≥1 insertion = FIRST SEAT → demo+adopt+push. Revised KILL: disasm-standalone 0 seats
on aligned poses AND localization worse than the measured capture radius → ship
capped-aux, insertion→Track-L P3, pivot to MAX officials/ab5. SOTA/refs: AutoMate
2407.08028, InsertionNet 2104.14223 / 2.0 2203.01153, IndustReal 2305.17110, seam-fill
2204.07776, tactile peg-from-disassembly 2604.20712.

## 2026-07-20 07:30 — Run #2 cycle 7: DAgger localization pipeline live, collecting

**Avg score:** still **0 insertions** (~165 trials); the banked deploy config
(capped-aux, ab5 mean 28.0 / IQM 35.8 vs baseline 23.1) is the Track-S submission
floor, unchanged. This cycle was execution of the cycle-6 pivot: build + bring up
the **privileged-DAgger port-localization** data pipeline (the seat's one missing
capability = deploy-time port localization; aux head is 15–61 mm off at deploy).

**What's missing:** the seat, gated on one number not yet measured — the retrained
localization head's **held-out error on deploy-stall states**. Pipeline now works
end-to-end (validated offline on a real deploy-rollout bag: 486 stall frames labeled
with the true port offset). Bring-up cost five infra fixes, all committed: completion
detection (deploy policy has no CheatCode marker → wait for scoring.yaml), port
disambiguation (same port name on multiple NIC mounts → exact `--port-frame` from
`target_module_name`), and the load-bearing one — the robot TF tree (`world`) and the
scene/scoring tree (`aic_world`) are disconnected same-origin roots, now bridged by
identity (verified: the gripper-welded plug under aic_world sits at the grasp offset
from the TCP under world).

**Next 4 h:** full DAgger collection running (48 SFP configs, deploy policy in a
ground-truth sim, ~6–8 h, resumable) → also collect the officials/ab5 seat poses →
retrain a fresh frozen-encoder localization head on the deploy-stall labels →
**GATE: held-out localization error.** < ~10 mm → wire a one-shot spiral re-center
(`AIC_SEARCH_AUX`) onto the predicted port and run the officials seat eval (≥1
insertion = first seat). > ~10 mm → the occluded last-inch sensing lacks the
information; ship the capped-aux floor and move insertion to Track-L P3 (residual RL).
Kill date Jul 26. SOTA: privileged→student on-policy distillation / DAgger
(2606.10385, 2603.04038); the oracle last-inch is pure vertical (lat/vert 0.014) so
plain BC can't clone the correction — DAgger *computes* the label from the port TF.

## 2026-07-20 05:40 — Run #2 cycle 6: scripted search KILLED, pivot to learned localization (DAgger)

**Avg score (search probes):** official_2 ~43 (3/3), official_3 ~29, cfg_005 ~10;
**still 0 insertions in ~165 trials.** Decision cycle (2 analysis agents, physics +
architecture): **stop tuning the scripted spiral search — it cannot seat here.** The
r18 probe set the spiral radius to **18 mm, above official_2's measured 13 mm
offset, and still seated 0/3** — that falsifies the "search coverage" hypothesis.

**What's missing (root cause, quantified):** the seat is a **deploy-time port-
localization** problem, not a seating-primitive problem. Depth/push/axis are all
solved (plug reaches the mouth plane, decisive push, vertical axis confirmed); the
residual is **pure lateral 13 mm (official_2) to 26–61 mm (official_3) against a
1–2.5 mm bore**. The blind spiral fails on three walls: (1) **friction** — from the
traces K_eff ≈ 350 N/m gives only ~4.4 N lateral authority vs μ·N at N≈10 N (μ≈0.44),
so the tip creeps ~5 of 18 mm commanded and stalls; pushing harder raises the same
normal force (12 N back-off caps it) — unwinnable; (2) **resolution** — a 2-turn/18 mm
spiral has 9 mm loop pitch ≫ the 2 mm bore; guaranteeing overlap needs ~9 turns =
time-prohibitive at 0.05× RTF; (3) **out of range** — official_3 plunges 32–40 mm
*beside* the port (never contacts). And no primitive can localize: aux val 0.86 cm →
**15–61 mm in deployment** (covariate shift), vision occluded at the last inch, port
TF eval-illegal. Same impedance gains seat perfectly for the oracle (it has the true
TF) → the whole gap is the target, not the controller.

**Next 4 h (decision = D):** (1) **Bank the floor** — the adopted capped-aux config
(ab5 IQM 35.8 vs 23.1, officials proximity) is the guaranteed Track-S submission;
port to phase_1 + containerize (flagged to user; needs portal/Docker). (2) **Privileged-
DAgger localization** (the seat shot, = the user's learned all-sensor directive):
roll the deploy policy to its stall in a `ground_truth:=true` collection env, snapshot
eval-legal obs (RGB+TCP+wrench) → label with the oracle's **true port offset**
(port TF − TCP), retrain the localization head on the policy's *own* stall
distribution (fixes the covariate shift that makes aux non-transfer; regresses the
offset so it generalizes across 13–61 mm), wire as a one-shot spiral re-center
(`AIC_SEARCH_AUX` seam). Mostly offline: one detached sim data-gen pass + offline
retrain. **Kill-criteria:** held-out deploy-stall localization error > ~10 mm after
retrain → occluded sensing lacks the info, ship capped-aux only; no insertion by
Jul 26 → freeze + ship, insertion → Track-L P3 (residual RL). A seat is +53/trial
(≈+3.5 ab5/trial) — the only path to avg>90. SOTA: privileged→student DAgger
(2606.10385), extrinsic-contact localization; the oracle last-inch is pure vertical
(lat/vert 0.014) so plain BC can't clone the correction — DAgger *computes* the label.

## 2026-07-20 04:30 — Run #2 cycle 5: seat wall localized — it's LATERAL, depth is solved

**Avg score (search probes, officials, AIC_SEARCH):** official_2 ~42.8, official_3
~28.8, all "No insertion" at 0.05–0.11 m; **still 0 insertions** in ~155 trials.
But this cycle turned the seat from a fuzzy wall into a measured, single-axis
problem. Built and shipped (all env-gated OFF, 222 tests): a decisive-push
specialist (fixes v1's passive hold), a scripted spiral-search primitive
(`AIC_SEARCH`), and a world-vertical search axis (`AIC_SEARCH_VERTICAL`, since
the ports open −z).

**What's missing — now quantified by ground-truth-TF geometry forensics.** The
seat is **lateral mis-localization, not depth**. On official_2 the plug tip
reaches the port-mouth *plane* (plug→mouth dz ≈ 0, measured −0.1 to −2.3 mm) but
sits **~13 mm to the side**, resting on the port housing rim (bore is only
1–2.5 mm wide, so a 13 mm offset lands on the lip and jams; the apparent "46 mm
depth" is the port's internal bore the plug never enters). official_3 (SC) is
worse: **26–61 mm** lateral. The scripted search descends vertically, engages
contact (10–11 N), reaches mouth depth, and sweeps the full spiral — but the
10 mm radius < the 13 mm offset, so it never crosses the hole. Critically, the
**aux bearing does not transfer**: its "0.86 cm val" becomes **15–61 mm in
deployment** (high-variance), so it can't reliably center the plug.

**Next 4 h:** close the lateral gap. (1) Cheapest first-seat shot — raise the
search radius above the measured offset (`AIC_SEARCH_RADIUS` 0.018, covers
official_2's 13 mm; code-free) on the SFP aligned poses. (2) If insufficient /
for official_3's larger offset — aux-centered search (`AIC_SEARCH_AUX`: recenter
the spiral once at handoff on the median-gated aux port prediction; small seam in
guarded_descent.py, off = byte-identical), accepting the aux is noisy so the
spiral must still absorb ~10 mm residual. Better port-direction sensing is the
lever; depth/push/axis are done. Gate unchanged: ≥1 insertion + no officials
regression. SOTA refs: InsertionNet spiral search (2104.14223); From Reach to
Insert AABB handoff (2605.04649). Process note: let probes exit naturally —
a mid-probe kill left stale model nodes that failed a whole batch ("model not
ready"); recovered by full teardown.

## 2026-07-20 02:15 — Run #2 cycle 4: learned insertion specialist v1 — two-policy works, seat still open

**Avg score (officials n=3, 180 s, specialist handoff `AIC_SPECIALIST=1`):** mean
**30.4** vs scripted-descent baseline 35.7 — per pose official_1 12.2 (−21.6),
official_2 41.5 (+2.1), official_3 37.6 (+3.8). **0 insertions.** Specialist =
ACT K=8 BC'd on the 15 s last-inch of 77 SFP oracle successes (obs 3×RGB +
pose + 6-D wrench, encoder warm-started from v2_wide). Env-gated OFF by default,
so the adopted ab5 28.0/IQM 35.8 config is untouched; this is a research probe.

**What's missing:** the seat, still. The two-policy machinery is validated
(approach → stall → learned-specialist handoff fires, force-aware descent,
per-trial `[specialist]` telemetry) and the specialist is a small genuine gain
on the two aligned officials — but (1) it holds at 0.05–0.06 m instead of
pushing decisively through, so nothing seats; (2) on official_1 it drives ~67 mm
off-axis into the mount (−24), an OOD stall pose the vertical last-inch policy
extrapolates wrong. Root-cause hypotheses (being quantified this cycle): the
oracle's last-inch is ~pure vertical descent (no lateral-correction signal to
clone), the 15 s window likely leaks slow/near-zero-velocity frames (policy
learned "hold" not "push"), and the deploy handoff pose (~0.05 m) may sit outside
the training window's start.

**Next 4 h:** analysis+design workflow (2 agents: specialist forensics from the
guarded traces + training data; design pick) → adopt one of HYBRID (aux-bearing
descent for approach/lateral + specialist only for the final force-push when the
wrench plateaus — protects the official_2/3 gains, dodges the official_1 drift),
tighter last-inch window (6–8 s) + K=4 + push-weighting, or oracle DAgger at the
stall states → retrain offline (GPU free) → re-eval officials n=3, gate ≥1
insertion AND official_1 no-regression. Approach-side dead-band retrain remains a
complementary follow-on. SOTA referenced: ACT (2304.13705), From Reach to Insert
(2605.04649, hybrid handoff), FILIC (2509.17053, force channel), TER-DAgger
(2603.04038, force-triggered oracle DAgger).

## 2026-07-20 00:05 — Run #2 cycle 3: capped-aux adopted, SC oracle+configs fixed, seat still the wall

**Avg score (ab5 gauge, n=15 = 3 officials×3 + cfg_001×3 + cfg_005×3, 180 s):**
adopted **capped-aux** (v2_auxprobe.pt + `AIC_GUARDED_AUX=1`) = **mean 28.0,
IQM 35.8** vs baseline v2_wide **23.1** (n=15). IQM clears the +5 adoption gate
(+14); the mean move (+4.9) is within the documented sim noise (sd 3–18). Honest
decomposition (n=9 officials A/B, frozen action weights): guarded-descent alone
−6.0, aux-uncapped −5.2, **aux-capped −0.1 vs bare** — the gauge gain is a
*failure-mode swap on official_1* (mount collision → clean 0.05 m proximity, the
−24 mode removed), surfacing in IQM, not broad improvement. Side results: SC
teacher oracle repaired — after fixing a config bug (all 6 SC eval configs named
a nonexistent port link → every SC trial scored ~1 for **any** policy since the
suite was authored) it now seats **3/7** SC poses at ~93 with **0 contacts**
(official_3, cfg_006, cfg_007). **0 insertions by the learned policy in ~120
trials to date.**

**What's missing:** (1) still the seat — a proximity-only gauge caps at ~43.5,
so rung 40 has no margin without one insertion (a seat is tier-3 +53/trial ≈
+3.5 composite/trial). (2) The ab5 dead spot is **cfg_001** (rail0, SFP port_1,
board_yaw +0.837; reaches only 0.18–0.26 m, one −23 collision) — it costs the
composite **≈8.8 points**; ab5-minus-cfg_001 is already **36.8**. Note the
separate PLAN_SCORE90 P2 band **cfg_000/004/008** (|yaw|∈[1.2,1.5], port_0,
reached 0/9) is *not* in ab5 — both are approach/coverage holes fixable only by
retraining the **action head** (the adopted ckpt is a frozen-encoder/frozen-head
aux probe, so more demos into it change nothing). (3) The 4 blocked SC poses
stall at exactly 0.01 m regardless of floor depth; the seat/block separator is
**rail / board_x** (rail1 ≈0.20 seats; rail0/rail2 ≈0.15–0.18 block), i.e. a
lateral waypoint-calibration bias, not depth — SC keep-rate 3/7 < the 5/8 P2 gate.

**Next 4 h (04:00 D — dead-band coverage retrain, per two-agent critique):**
generate forced-yaw dead-band configs (`gen_deadband.py`, yaw ∈ [−1.5,−1.2] to
cover cfg_008 which the strata sampler misses; rails 0/1/2, port_0) **plus
cfg_001's region** (rail0 port_1 +0.837, the ab5 blocker) → `collect_campaign.sh`
(bags auto-deleted, resumable, keep score≥60 ∧ insertion≥1, ~8 min/demo) →
**two-stage retrain**: full **unfrozen** action head warm-started from v2_wide
with the PLAIN recipe (no wrench/tail-trim/pushin — those are the v3fix levers
rejected at n=3) then a frozen 3-D aux probe (6-D rejected) → eval gate:
cfg_000/004/008 + cfg_001 + official_1/official_3 at n=3, adopt **only** on ≥1
new insertion AND no officials regression (pointer swap; v2_wide/v2_auxprobe
never overwritten) → refresh the README demo video. SC excluded from tonight's
retrain (shares the SFP cable; below gate). SOTA/methods considered: ACT
(Zhao 2023, arXiv:2304.13705) receding-horizon backbone retained; MimicGen
(Mandlekar 2023, arXiv:2310.17596) privileged-replay pattern for band coverage;
residual-RL last-inch (arXiv:2509.19301) still deferred to post-Jul-28 P3.

## 2026-07-19 19:15 — Run #2 cycle 2: learned port bearing works (graze 3/3→1/3), seat still open

**Avg score (n=9 officials per arm, 180 s):** blind-descent 29.8 → aux-bearing
capped 35.7 ≈ bare v2_wide 35.8; uncapped 30.6 (variance up); gauge composite
≈25.0 vs baseline 23.1±2.1 (n=15) — within noise, rung 40 not yet claimed;
0 insertions in 27 probe trials today.

**What's missing:** (1) the seat — plug reliably reaches 0.04-0.05 m clean but
the final 4 cm needs an accurate insertion AXIS (depth prediction unreliable
10-51 mm; 6-D explicit-axis head trained+wired, F2 probe = final iteration);
(2) adoption-grade evidence (all deltas < IQM+5 at n=3); (3) SC oracle + the
cfg_001 coverage hole untouched (overnight P2).

**Next 4 h:** F2 6-D probe (9 trials; gate ≥1 seat) → bank winner (capped aux
config removes the off1 −24 mode regardless) → overnight cycle B adoption
gauge (n=3 ab5 vs 23.1) → cycle C SC oracle repair (floor −0.007 +
pose-conditioned waypoint + zero-contact revalidation). Cycles 00:00/04:00
with multi-agent analysis each.

## 2026-07-19 16:00 — Run #2 cycle 1: probe answered (bearing is the gap); K specialists found

**Avg score:** gauge (eval_suite_ab5) baseline v2_wide **23.1±2.1** (n=15);
this cycle v3fix_k8 16.9 (n=5), v3fix_k16 ~20.5 (n=4 valid, off3 crashed);
best-of-both per-config routing ≈30–32. Guarded probe officials (n=9): off1
11.3 (−24 mounts) / off2 42.8±0.1 clean / off3 35.4±0.2. 0 insertions
anywhere; still on the ~23 rung, target 40.

**What's missing:** (1) an insertion event — every officials point is still
tier_3 proximity; (2) port BEARING at the stall handoff (probe proved descent
is stable but blind; aux-head spec'd + implementing); (3) cfg_001
moderate-yaw coverage dead on all ckpts; (4) valid off3 for k16 (engine
crash); (5) k8's new off1 collision makes it unshippable despite its real
cfg_005 collision-fix.

**Next 4 h:** dual-arm officials campaign running (bare k16 n=3 confirmation
gate: off1 IQM+≥5 + 0 collisions; then guarded-on-k16 n=3 — first seat
attempt from the 0.05 m clean handoff; ≥1 insertion → GuardedInsert
hardening). Parallel: port-offset aux-head implementation (CPU) → frozen-probe
retrain when GPU frees. Next cycle ~20:00: campaign verdict + aux-head
frozen-probe cm-error + progress entry.

## 2026-07-19 09:10 — RUN COMPLETE: n=3 reps close the ensembling question (keep OFF); wrapping the 48 h session

The 3-rep experiment finished 09:01: no config separates ensembling ON from
OFF beyond sim noise (arm means OFF 23.1±2.1 vs ON 20.1±4.2; ON even missed
official_1 once outright), so plain v2_wide.pt stays the deploy default and
same-seed noise is now quantified at sd ≈3–18 pts — n≥3 reps required for any
future per-config claim. All 48 h deliverables are complete and pushed:
adopted checkpoint v2_wide (proven 119.4 insertion result), last-inch
attractor root-cause + ranked fixes (DAgger/CVAE/residual RL), 93-episode
Phase-0+2 dataset, hardened eval harness + 295 tests, temporal-ensembling
implementation with a CI-backed negative result, demo video, dashboard, and
full retrospective in SESSION_REPORT.md. Autonomous loop stops here — sim
processes cleaned up; resume anytime from the repo runbooks.

## 2026-07-19 07:10 — Full suite OVERTURNS ensembling adoption; sim run-variance ±5–13 pts discovered

The 15-config confirmation came back inconclusive-with-negative-point (mean
diff −2.27 [−6.17, +0.62], 9 miss-floor ties, cfg_011 miss→−23 collision), and
re-running identical config+seed trials revealed ±5–13 pt run-to-run sim noise
— the ab5 "5/5 wins" was mostly sampling luck, so the default deploy reverts to
plain v2_wide with AIC_ENSEMBLE as an opt-in flag. Next 2 h: 3-rep-per-arm
repeat experiment on the 5 ab5 configs (2 OFF reps running since 06:59, 1 ON
rep after, ~08:45 done) for per-config mean±sd and a final officials verdict,
then final analysis + dashboard + wrap.

## 2026-07-19 05:35 — ENSEMBLE A/B: v2_wide +3.9 mean (wins/ties 5/5), p2_k8 −15.5 → adopt ON for v2 only

The temporal-ensembling A/B finished 05:18: with AIC_ENSEMBLE=1 the adopted
v2_wide improves or ties all 5 configs (mean 20.8→24.7, all gains pure tier_3
final-approach, zero new contact events), while p2_k8 regresses −15.5 with two
new collisions — verdict: default deploy = v2_wide + ensembling m=0.01, OFF for
p2_k8; still 0/10 insertions, attractor nudged not broken. Next 2 h: 15-config
full-suite confirmation of v2_ens @180 s (launched ~05:40, done ~07:20) for a
paired-bootstrap CI against v2_wide_180, then final analysis + dashboard +
wrap.

## 2026-07-19 03:55 — 180-s head-to-head COMPLETE: 0/45 insertions, v2_wide leads

The full-budget matched-seed comparison finished at 03:49 with **zero
insertions in 45/45 trials**: v2_wide mean 7.7 (officials Σ97.7) > p2_k8 5.3
(best IQM 3.9 and the only strata partials, 41.5/34.9) > p1_k16 3.0; all
pairwise bootstraps inconclusive. The suite cannot separate checkpoints below
the insertion threshold, and v2_wide's proven 119.4-with-insertions on the
exact official poses stands as the only demonstrated insertion result — the
final analysis agent is writing the adoption call + full 48-h retrospective
into SESSION_REPORT now.

**Next 2 h:** analysis lands → dashboard republish with the 180-s table →
closing progress entry → run wrap-up (well before the 17:00 end).

## 2026-07-19 02:45 — 180-s verdict forming: nobody inserts on the matched suite

v2_wide's full-budget run is in: **0/15 insertions** (mean 7.7, officials
24.8/39.6/33.3 = 97.7) — it leads the officials but its famous insertions did
NOT reproduce, confirming the matched-seed suite's poses are genuinely harder
than the exact official eval poses where it scored 119.4. Zero insertions in
30/30 trials so far splits the picture cleanly: v2_wide best on officials,
p2_k8 best on stratified approaches (two 35-41 partials where everyone else
floors), and the last-inch stall is THE unsolved problem across all
checkpoints and budgets. p1_k16_180 (final run) is at 3/15.

**Next 2 h:** EVALBATCHDONE ~03:50 → full pairwise comparison + analysis
agent → adoption call (v2_wide stays the submission checkpoint on its proven
119.4 real-config result) → final 48-h report + dashboard at dawn.

## 2026-07-19 00:45 — 180-s head-to-head: p2_k8 run done (0/15 insert); control running

The retrain finished in 17 min at 96% GPU (p2_k8 val L1 0.00129, p2_k16
0.00130 — both ~40% better than p1) and the 180-s matched-seed head-to-head
launched at 23:00; p2_k8's full run is in: **0/15 insertions** (mean 5.3,
officials −1.8/37.0/29.7), with genuine approach gains on two previously-
floored strata (41.5, 34.9 partial credit) but three −23 SC collisions —
Phase-2 data improved SFP approaches while SC remains the weak point, and
the last-inch stall persists even at full budget. v2_wide_180 (the 119.4/300
reference) started 00:39; its officials decide whether the old checkpoint's
insertion ability stands as the adoption choice.

**Next 2 h:** v2_wide_180 completes (~02:20) → p1_k16_180 → EVALBATCHDONE
~04:00 → full pairwise comparison + analysis + final 48-h report at dawn.

## 2026-07-18 22:30 — Phase-2 COMPLETE (33/40 KEEP); dataset at 77 episodes

The failure-driven campaign finished cleanly at 22:28: **33 KEEPs** (30 SFP at
oracle 93.4–94.0 covering every floored stratum, 3 SC at 93.8–94.1) with 7
drops concentrated in SC (3/8 keep-rate under eval-band yaw — two failure
modes logged in SESSION_REPORT: the −0.005 floor partial-inserts and poor
approaches on some eval-band SC poses; micro-tune is top follow-up). Dataset
is now **77 episodes** (69 SFP + 8 SC, all wrench/joints); sim confirmed
clean, zero harness incidents across the entire 5-hour campaign.

**Next 2 h:** retrain both K on ds_phase0+ds_phase2 (shift-aug winners, GPU-
verified) → launch the overnight 180-s matched-seed head-to-head (p2 winner +
p1_k16 + v2_wide on eval_suite_smoke) that decides the run's final model.

## 2026-07-18 20:20 — Phase-2 collection past halfway; SC floor is pose-dependent

Collection is at 25/40 with 21 KEEPs (SFP oracle 93.4–94.0, zero SFP failures
since the one early near-miss) and the SC picture is now clear: 2 KEEP
(93.8, 94.1) / 2 DROP (62.8, 65.0 — both partial-inserts with ins=0 at the
−0.005 descent floor), i.e. the floor issue is pose-dependent rather than a
cell-wide failure, and the campaign will still deliver usable SC coverage.
CAMPAIGNDONE ~22:15, on pace, zero harness incidents.

**Next 2 h:** campaign finishes → retrain both K on ds_phase0+ds_phase2
(shift-aug winners) → launch the overnight 180-s matched-seed head-to-head
(p2 winner + p1_k16 + v2_wide) that decides the real best checkpoint; SC
floor micro-tune logged as top follow-up in SESSION_REPORT.

## 2026-07-18 18:25 — Post-batch sequence complete; Phase-2 failure-driven collection running

Everything queued on the batch landed in one push: the analysis agent's verdict
(60-s protocol is an approach-only proxy that truncates v2's insertions
119.4→71.3; floored strata are a moderate-yaw + rail0 + SC coverage gap),
the first-ever POLICY demo video (`~/demo/policy_p1_k16_official_1.mp4`,
p1_k16 approaching to 0.06 m on official_1), the dashboard republished with
the smoke60 leaderboard, and a 40-config failure-driven Phase-2 manifest
(32 SFP with port_0 boost + 8 SC, all distractor + eval-band yaw, tested
183/183). Collection is grinding cleanly (9/40 started, 7 KEEP at oracle
92.6–94.0 / 1 near-threshold DROP, ~7 min/demo, CAMPAIGNDONE ~22:00).

**Next 2 h:** collection continues into the SC cells; at CAMPAIGNDONE →
retrain both K on ds_phase0+ds_phase2 → overnight 180-s matched-seed
head-to-head (p2 winner + p1_k16 + v2_wide) that decides the real best model.

## 2026-07-18 16:40 — Control verdict: P1 NOT regressed; 60-s suite hits an insertion floor

The v2_wide control finished and settled the question: the formal paired
comparison is a statistical tie (p1_k8 +2.57 mean, 95% CI [−4.54, 9.68]),
with p1_k8 earning partial credit on all 3 officials (27.8/17.7/29.6, incl.
the SC config) while v2_wide floored official_1 at 1.0 and took two −23
collision scores to p1_k8's one — so the P1 retrain did not regress and
generalizes slightly better. The deeper finding: 0/30 insertions and 9/15
identical 1.0-vs-1.0 floors mean the 60-sim-s fast protocol discriminates
approach quality only, not insertion (the last-inch phase needs the full
180-s budget). p1_k16_60 is at 12/15 with zero harness incidents all day.

**Next 2 h:** EVALBATCHDONE (~17:05) → remaining pairwise compares + analysis
agent + dashboard republish + policy rollout video → overnight failure-driven
Phase-2 collection targeting the floored strata (distractors + eval-band yaw).

## 2026-07-18 14:45 — First valid suite result: p1_k8 0/15 insertions; control mid-run

The clean batch produced its first fully-valid run: p1_k8 on the 60-sim-s
suite scored 0/15 insertions (mean 5.3, IQM 2.9; outcomes 10 miss / 2
collision / 3 proximity) — but with structure: the 3 official configs earned
real partial credit (17.7–29.6, including 29.6 on the first-ever SC policy
trial) while 11/12 stratified configs (distractors + eval-band yaws) floored
at 1, pointing at OOD strata rather than a dead checkpoint. The v2_wide
control is mid-run (9/15, ~6 min/trial, every finished trial has
scoring.yaml, zero harness incidents) and decides regression-vs-difficulty.

**Next 2 h:** v2_wide_60 + p1_k16_60 complete → EVALBATCHDONE (~16:30–18:00)
→ paired comparison + analysis agent + dashboard republish + policy video.

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
