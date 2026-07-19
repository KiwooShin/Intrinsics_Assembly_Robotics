# AIC Competitive-Intelligence Report — Intrinsic "AI for Industry Challenge"
*Researched 2026-07-19. Sources: live event/rules pages, `intrinsic-dev/aic` GitHub (commits/PRs + fetched file contents), Open Robotics Discourse (incl. organizer posts pulled via JSON API), participant blogs, and the local toolkit docs in this repo.*

---

## 0. Headline resolutions (the two things you asked me to settle)

1. **The "Phase-1 = Jul 14" date in local docs is STALE.** It is from the *original* (pre-delay) schedule baked into an early toolkit snapshot. On **2026-05-29 the organizer (Yadunund) publicly confirmed a ~2-week slip**: Phase 1 now **starts Mon June 15, 2026**, and **Phase 2 starts August 4, 2026**. The live event page corroborates ("Phase #1 results by **August 4, 2026**", "Phase #2 results by **September 8, 2026**"). A participant (KimMcG) references a **~July 28 Phase-1 submission deadline** (one week before Aug 4). So "Jul 14" maps to *no current milestone*; the real upcoming anchors are **~Jul 28 (Phase-1 submit)** and **Aug 4 (Phase-1 results / Phase-2 start)**. Source: [Discourse results thread](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138), [event page](https://www.intrinsic.ai/events/ai-for-industry-challenge).

2. **⚠️ Important standing check:** the **Qualification phase (the exact 3-trial /300 Gazebo eval this repo targets) CLOSED May 15, 2026**; results announced May 28; **31 teams (of ~160) advanced**. Phase 1 runs on a **diverged `phase_1` toolkit branch** (Flowstate + Intrinsic Vision Model, harder task board with *multiple SC ports per rail*, updated scoring). If you are competing *live*, work must be on `phase_1`, not the May-30 `main` checkout this repo is pinned to. If you're not among the 31, the qualification window is long closed. I can't determine your standing from public data — **flagging as load-bearing.**

---

## 1. Official schedule (reconciled, most-authoritative first)

| Milestone | Date | Source |
|---|---|---|
| Toolkit launch / kickoff | Mar 2, 2026 | event page |
| Team-member registration | Apr 17, 2026 | event page |
| Registration deadline | May 8, 2026 | event page |
| **Qualification submission close** | **May 15, 2026** (portal 16:59 PT; T&C said 23:59 PT — organizers manually ran all images uploaded before 23:59) | [Discourse 55138](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138) |
| Qualification results | May 28, 2026 (**31 teams advance**, expanded from 30) | Discourse 55138 |
| **Phase 1 start** (Flowstate + IVM) | **Mon Jun 15, 2026** | organizer Yadunund, Discourse 55138 |
| Phase 1 submission deadline | ~**Jul 28, 2026** (participant-stated, unconfirmed by organizer) | KimMcG, Discourse 55138 |
| **Phase 1 results** | **by Aug 4, 2026** | event page |
| **Phase 2 start** (physical workcell @ Intrinsic HQ) | **Aug 4, 2026** | organizer, Discourse 55138 |
| Phase 2 results / winners | **by Sep 8, 2026** (KimMcG notes hard anchor is "before ROSCon Toronto") | event page |

Advancement funnel: **~160 qualifying teams → 31 → Phase 1 → top 10 → Phase 2.** Prizes $180k: **$100k / $40k / $20k / $10k / $10k.** Note the [rules page](https://www.intrinsic.ai/ai-for-industry-challenge/rules) legal doc still frames the "Challenge Period" as Mar 2–Jul 31, 2026 (the software window); it lags the operational schedule above.

---

## 2. Toolkit repo (`intrinsic-dev/aic`) — recent changes that affect scoring/sensors/rules

Repo is **public**, last push 2026-07-17. There is **no separate public `aic-phase-1` repo** (404) — Phase-1 work lives on the **`phase_1` branch** of `aic`, and there's a private `aic-phase-1` issue tracker (referenced but inaccessible). Recent, strategy-relevant activity:

**Scoring changes (Phase 1):**
- **#577 "Phase 1 scoring"** (merged Jun 12) — large engine/scoring refactor: removed the standalone Tier-1 scorer binary, rewrote `aic_engine.cpp` (−277/+101), reworked `ScoringTier2`.
- **#594 "Scoring updates for phase 1"** (merged **Jul 9**) — modifies `ScoringTier2.cc/.hh` (+37/−14). Most recent merged scoring change.
- **#595 "Scoring phase 1 tier 2 updates"** (**OPEN**, branch `luca/wrench_updates`) — two live, not-yet-merged tweaks: (a) **force penalty applies only when the threshold is exceeded for a *contiguous* time over threshold, not total time** (materially more forgiving of brief force spikes); (b) **widen min/max thresholds for trajectory-efficiency scoring**. Watch this — it changes penalty exposure.

**Sensor / interface changes:**
- **#596 "subscribe to F/T sensor topic and add tare offset"** (**OPEN**) — publishes `fts_tare_offset` on `/aic_controller/controller_state` so policies can track the last F/T tare. Confirms **force/torque is a first-class, consumable observation** (see §6). No depth topic is added anywhere.
- **#561/#582 "Support multiple SC ports per rail for phase 1"** (merged) — Phase-1 task board has **multiple SC ports per rail** (harder disambiguation; relevant to your SC weakness).
- **#585 "Add Task Board CAD and Assembly Guide"** (Jun 24) — physical task-board CAD (Phase-2 sim-to-real prep).
- **#586 "Add ROS 2 Jazzy compatibility"** (OPEN); **#573** fixes nondeterministic trial-to-trial homing races (relevant to your measured run-to-run variance).
- Earlier fixes tuning contact physics: **#489 restore SC-plug↔port contact**, **#454 lower cable friction**, **#578 fix SFP port contact-collision pose**, **#590/#589 SC-end collisions**.

**Rule/scoring clarification commits:** **#478 "explicitly say max score is 300"** (Apr 13) confirms the /300 ceiling; **#500** troubleshooting for "policies failing only on the portal" (local-vs-cloud eval drift — a known gotcha).

---

## 3. Exact scoring rubric (verified from `docs/scoring.md`, identical on `main` and `phase_1`)

**Per trial = Tier1 + Tier2 + Tier3, max 100. Three trials → max 300.**

- **Tier 1 — Validity: {0, 1}.** Model loads, responds to `InsertCable`, sends valid `MotionUpdate`/`JointMotionUpdate`. Fail ⇒ not scored.
- **Tier 2 — Motion quality (gated: all positive components award 0 unless Tier 3 > 0, i.e. plug within proximity):**
  - Smoothness **0–6** — time-weighted linear jerk (Savitzky–Golay, 15-sample window); jerk 0→6 pts, ≥50 m/s³→0. Jerk only accrues at speed >0.01 m/s (stalls don't dilute it).
  - Duration **0–12** — ≤5 s→12, ≥60 s→0, linear.
  - Efficiency **0–6** — EE path length; ≤ initial plug-port distance→6, ≥ +1 m→0.
  - Force penalty **0 to −12** — >**20 N** for >**1 s** (soon: *contiguous*, PR #595). Baseline tared to ~0 N.
  - Off-limit contact penalty **0 to −24** — any contact with enclosure/task board.
- **Tier 3 — Task success:**
  - **Full insertion, correct port = +75.** Wrong port = **−12**.
  - **Partial insertion = 38–50** (plug inside entrance→bottom bounding box, 5 mm x-y tol; proportional to depth).
  - **Proximity = 0–25** (25 at port entrance → 0 at half the initial plug-port distance).

**What ">90/trial" actually requires (do-the-arithmetic):**
Full insertion is **mandatory** — without the +75, the hard ceiling is `1 + 24(Tier2) + 50(max partial) = 75`, and realistically far less (Tier-2 duration ≤5 s and tight efficiency are near-impossible while merely "partially" inserted). To clear 90 you need: **Tier1 (1) + full insertion (75) + ≥14 of the 24 Tier-2 positive points (fast, smooth, direct) + zero force/contact penalties.** A perfect trial is `1+75+6+12+6 = 100`.

**Why your policy sits at 5–25/trial is now fully explained by the rubric:** the last-inch stall at 0.05–0.08 m lands in the **proximity band only** (≤25, often ≤ a few points), and because **Tier 2 is gated on Tier3>0**, a stall *outside* proximity zeroes the entire Tier-2 block. There is no partial-credit ramp that rewards "clean approach that doesn't seat" beyond ~25 — the rubric is deliberately steep at the seating step, so the insertion problem *is* the whole game.

---

## 4. Public participant writeups & scores (real numbers)

| Team | Result | Approach | Source |
|---|---|---|---|
| **Datameister** | **#1, 293.38/300** (won by ~1 pt; ~160 teams) | *Decomposed*, step-specific policies over "production-grade perception + 3D-AI primitives from the DM Library" — **not** a monolithic learned policy; picked best tool per sub-step. GT poses unavailable at scored eval. | [datameister.ai blog](https://datameister.ai/blog/intrinsic-ai-for-industry-challenge-qualifying-first/) |
| **MacCody (Comsysto Reply)** | **65th/160, public 112.75** (peak internal 151.71; early leaderboard 6th @ 47.34) — did **not** advance | **Classical staged visuotactile controller** built by human + coding-agent: visual acquisition → prealign → guarded descent → contact classification → local search → compliant insertion → verify/recover. OpenCV edge/Hough/chamfer vision, explicit state machines over pure force thresholding. Bottleneck: "offline gains didn't translate to online robustness"; "contact-rich perception-interaction entanglement" broke isolated fixes; "bottleneck moved from generation to selection." | [Discourse 55120](https://discourse.openrobotics.org/t/post-qualification-contribution-coding-a-classical-robot-controller-in-the-age-of-coding-agents-an-honest-assessment/55120) |
| **Tommy Ly** (@lymytom20) | (continued post-qual for learning; no score stated) | "Plug fiber cable into a 6 mm port using only wrist cameras + force feedback; 6-DOF arm, Robotiq gripper, 3 wrist RGB cams, no overhead view, no markers." | X thread via [Discourse 55138](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138) |

**Signal:** the **#1 team used a modular, primitive-based (non-end-to-end) pipeline**, and a strong classical controller only reached mid-pack (112.75). Both converge on the same lesson you've hit — **the contact-rich last inch, not the approach, is where scores are made or lost**, and monolithic imitation struggles there. The 293.38 winner implies **near-perfect insertion on all 3 trials with clean, fast, penalty-free motion** is achievable in this sim.

Other named participants: **Comsysto/MacCody** and **Tommy Ly** above; general community discussion (Docker/registry, ROS 2 Kilted, lighting randomization) on the [Discourse category](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/129). No public GitHub *solution* repos surfaced (see §7).

---

## 5. Baseline scores

**No official numeric baseline was published.** The only organizer-provided "baseline" is the trivial example policy **`WaveArm.py`** (waves the arm; does not insert → effectively Tier-1-only, near-zero). Your own **119.4/300 with real insertions** would have been a *strong mid-to-upper qualification result* by the leaderboard distribution implied above (winner 293, mid-pack ~112, 30th-place cutoff not published but below 112 given MacCody's 112.75 finished 65th — **the advance cutoff was likely well above ~112**, i.e. qualification was very competitive).

---

## 6. Rule clarifications that change strategy

- **Ground-truth is ALLOWED during training, forbidden at eval.** `docs/challenge_rules.md` §2c: *"During training, participants may use all internal state information, including ground truth data available over the `/tf` topic."* State-leaking is only prohibited *during evaluation*. ⇒ **Your CheatCode oracle reading GT poses to generate training demos is fully compliant**; your deployed ACT policy (RGB + state only) is compliant. No concern here.
- **Pretrained/pre-existing assets: allowed *with disclosure*, but not a blank check.** The rules require submissions be *"original work of the Team AND developed during the Challenge Period"* and mandate disclosure of *"pre-existing code or datasets."* The winner (Datameister) leaned on a **pre-existing** internal library and disclosed it. ⚠️ **Your `CLAUDE.md §5` ("any publicly released checkpoint may be used") overstates the written rule** — public checkpoints are tolerated *if disclosed*, and "developed during the Challenge Period" language could be read against heavy reliance on a frozen foundation model. If you adopt GR00T/π0/DINOv2/SAM, **document and disclose it explicitly** in the submission.
- **Sensors — RGB + F/T + proprio only; NO depth exists (by design, not a bridging gap).** Confirmed from `docs/policy.md` + `docs/scene_description.md`: the `Observation` message carries **`left/center/right_image` + `*_camera_info`** (three wrist **Basler acA2440-20gc mono RGB**, 1152×1024 @ 20 FPS) and **`wrist_wrench`** (ATI **Axia80-M20**, 3-axis force + 3-axis torque), plus joint states / controller state. **There is no depth sensor in the cell** — so "no depth topics bridged" isn't a config gap you can close; depth is simply not part of the hardware. Don't spend effort trying to surface it.
- **The wrench IS a consumable observation and you're ignoring it.** `wrist_wrench` is in every `Observation`; PR #596 adds `fts_tare_offset` to `controller_state`. Given the **Tier-2 force penalty** and that the *entire scoring bottleneck is the contact-rich seating step*, feeding force/torque into the policy (or a force-guarded insertion primitive) is the single most under-exploited lever your notebook flags — and it's exactly what both the winner (primitive-based) and MacCody (contact classification over force thresholds) relied on for the last inch.
- **Local-vs-cloud eval drift is a known, documented gotcha** (commit #500: "policies failing only on the portal") — plus per-trial homing nondeterminism (#573). Consistent with your measured sd 3–18 pt run-to-run noise; treat n≥3 as mandatory (you already do).
- **Determinism / randomization:** organizers were asked whether final eval randomizes lighting/shadows/sun position (open question in Discourse); task board spawns with **randomized pose + yaw**, and *"the specific target port will always be within view of the robot cameras."*

---

## 7. Explicitly NOT found (gaps / negative results)

- **No published per-team leaderboard numbers beyond Datameister (293.38), MacCody (112.75), and the 31-team advance count.** The advancing-teams roster is an *image* in Discourse 55138 (not machine-readable); individual scores and the exact **30th-place advance cutoff** were **not disclosed**.
- **No public GitHub *solution* repositories** from participants (only the official `intrinsic-dev/aic` toolkit). No ACT/diffusion-policy participant code surfaced.
- **No official numeric baseline score** from organizers (only the trivial `WaveArm` example).
- **No Discord** located; the community hub is **Open Robotics Discourse** ([category 129](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/129)). LinkedIn hits are Intrinsic's own announcements, not participant technical writeups.
- **Phase-1-specific rubric weights not fully re-published** — `phase_1/docs/scoring.md` still shows the qualification numbers (75/50/25, −12/−24), but code changed (#577/#594 merged, #595 open). The **contiguous-force-penalty and widened-efficiency changes (#595) are not yet merged**, so the exact Phase-1 Tier-2 thresholds are in flux; read `phase_1:aic_scoring/src/ScoringTier2.cc` directly if you need current constants.
- **Whether final eval randomizes lighting** — asked by participants, **no organizer answer found**.

---

## 8. Net strategic takeaways for this team

1. **The rubric proves your diagnosis is the whole ballgame:** Tier-2 is gated on proximity, and there's no credit ramp between "clean approach" and "seated" beyond ≤25 — so **every point above ~25/trial requires actual insertion.** No amount of approach polish moves the needle.
2. **The two strongest public data points are non-monolithic:** #1 (Datameister) = decomposed primitives; best classical (MacCody) = staged force-guarded state machine. Both **use force feedback at the seating step.** Your deterministic-head ACT ignoring `wrist_wrench` is the mode-averaging trap *and* leaves the one signal that disambiguates contact unused. A **force-guarded insertion primitive spliced onto your learned approach** (hybrid, à la Datameister/MacCody) is the highest-EV direction and is rules-legal.
3. **Verify your competition standing and branch.** Qualification closed May 15 on the `main` toolkit; live competition is Phase 1 on the `phase_1` branch (multiple SC ports per rail, updated scoring, Flowstate + IVM). Confirm which you're actually evaluating against — your repo is pinned to a May-30 `main` snapshot.
4. **Calendar:** if live, the operative deadline is **~Jul 28 (Phase-1 submit) / Aug 4 (results)** — *not* the "Jul 14" in local notes.

**Primary sources:** [event page](https://www.intrinsic.ai/events/ai-for-industry-challenge) · [rules](https://www.intrinsic.ai/ai-for-industry-challenge/rules) · [toolkit repo](https://github.com/intrinsic-dev/aic) · [Discourse results/schedule 55138](https://discourse.openrobotics.org/t/aic-qualification-phase-results-advancing-teams/55138) · [MacCody writeup 55120](https://discourse.openrobotics.org/t/post-qualification-contribution-coding-a-classical-robot-controller-in-the-age-of-coding-agents-an-honest-assessment/55120) · [Datameister blog](https://datameister.ai/blog/intrinsic-ai-for-industry-challenge-qualifying-first/) · local `docs/scoring.md`, `docs/challenge_rules.md`, `docs/policy.md`, `docs/scene_description.md`.