I have the full experiment record and every per-trial CSV. Compiling the forensic inventory now.

I've cross-checked every number against the raw CSVs. Here is the report.

---

# Failure-Mode Forensics — Intrinsic AIC UR5e Insertion

**Sources:** `SESSION_REPORT.md` (709 lines); the 180 s matched-seed head-to-head `results/{v2_wide,p2_k8,p1_k16}_180_smoke/results.csv`; the 60 s screen `results/{v2_wide,p1_k16,p1_k8}_60_smoke/`; ensembling/reps `results/{v2_ens,v2_ens_full,v2_ens_r3,v2_off_r2,v2_off_r3,p2k8_ens}_smoke/`; pairwise `results/cmp_*/compare.md`; pose map `eval_suite_smoke/manifest.csv`. All per-config totals below are read directly from the CSVs and reconcile exactly with the `summary.json` means (v2 7.75, p2 5.29, p1 3.03).

The canonical evidence base is the **180 s, 15-config matched-seed suite** (12 stratified + 3 official), run once per checkpoint. **0/45 insertions.** Every point earned is tier_1 (+1) + tier_2 process bonus (≤17.3) + tier_3 *proximity* partial (≤25). No trial ever fired the tier_3 insertion event.

---

## 1. Where score is lost — config-by-config bucket inventory

### 1a. The full 15-config matrix (180 s, total score; yaw from manifest)

| cfg | rail/plug/port | board_yaw | v2_wide | p2_k8 | p1_k16 | bucket |
|---|---|---|---|---|---|---|
| cfg_000 | 0 SFP p0 | −1.49 | **1.0** miss | **1.0** miss | **1.0** miss | dead-miss |
| cfg_001 | 0 SFP p1 | +0.84 | 1.0 miss | **41.49** prox | 1.0 miss | stall (p2) |
| cfg_002 | 0 SC p0 | +1.32 | 1.0 miss | 1.0 miss | **−23** coll | SC-coll |
| cfg_003 | 0 SC p1 | +0.63 | 1.0 miss | 1.0 miss | **−23** coll | SC-coll |
| cfg_004 | 1 SFP p0 | −1.45 | **1.0** miss | **1.0** miss | **1.0** miss | dead-miss |
| cfg_005 | 1 SFP p1 | +2.67 | 5.26 graze | **34.92** prox | **37.67** prox | stall(+graze) |
| cfg_006 | 1 SC p0 | −0.76 | 1.0 miss | **−23** coll | 1.0 miss | SC-coll |
| cfg_007 | 1 SC p1 | +2.84 | **1.0** miss | **1.0** miss | **1.0** miss | dead-miss |
| cfg_008 | 2 SFP p0 | −1.23 | **1.0** miss | **1.0** miss | **1.0** miss | dead-miss |
| cfg_009 | 2 SFP p1 | +1.29 | 3.24 graze | 1.0 miss | 1.0 miss | stall/graze |
| cfg_010 | 2 SC p0 | −0.63 | 1.0 miss | **−23** coll | **−23** coll | SC-coll |
| cfg_011 | 2 SC p1 | +0.91 | 1.0 miss | **−23** coll | 1.0 miss | SC-coll |
| official_1 | 2 SFP p0 | +3.10 | **24.82** prox | −1.84 coll | 1.0 miss | stall (v2) |
| official_2 | 4 SFP p1 | −3.10 | **39.57** prox | **36.98** prox | **42.97** prox | stall (all) |
| official_3 | 1 SC p0 | −1.80 | **33.33** prox | **29.73** prox | **24.84** prox | SC-stall (all) |

### 1b. Bucket counts and score at stake

Three discrete failure buckets absorb 100% of lost points (no config is time-limited — every trial runs the full ~340–450 s wall budget and stalls *positionally*, confirming a fixed point, not a timeout):

**Bucket A — MISS-FLOOR (+1.0, tier_2=tier_3=0).** Plug never enters the port bounding radius; tier_3 logs *"Distance computation failed, tf between cable and port not found."* Counts: **v2 10/15, p2 7/15, p1 9/15.** Four configs — **cfg_000, cfg_004, cfg_007, cfg_008** — are *dead* (miss on all three checkpoints, 0/9 checkpoint-trials reach the port). Three of those four are **SFP port_0 at moderate yaw −1.2 to −1.5**; SFP port_0 stratified is reached in **0 of 9** checkpoint-trials vs SFP port_1 in **5 of 9**.
- *Ceiling if fixed:* miss → clean proximity is worth **+24 to +40** per config (the observed proximity band). But a rescued miss is still not an insertion — it then inherits Bucket B's residual gap. Two fixes required to score fully.

**Bucket B — LAST-INCH STALL (proximity, positive, no insert).** Clean directed approach, then stall at **0.05–0.08 m** for the remaining ~150 s. Counts (clean proximity, no −24): **v2 3 (all officials), p2 4 (cfg_001, cfg_005, off_2, off_3), p1 3 (cfg_005, off_2, off_3).** Score composition is diagnostic — on a reached config **tier_1≈1 and tier_2≈17.2 are already near-max; every missing point is in tier_3.** Best proximity tier_3 seen = 25.0 (p1 official_2) vs oracle tier_3 ≈ 75.
- *Ceiling if fixed:* each reached config jumps from ~25–42 to **oracle ~93**, i.e. **+50 to +68 per config**, all of it tier_3. This is the largest per-config prize and the configs are *already at the port*.

**Bucket C — OFF-LIMIT COLLISION (−23; contacts_score −24 inside tier_2, tier_3=0).** Wrist/tool strikes a **distractor NIC-card mount** (`nic_card_mount_*`), not the target port. Counts: **p2 3 clean −23 + official_1 −1.84; p1 3 clean −23; v2 ZERO.** Every one of the **6 clean −23 events is an SC config** (p2: cfg_006/010/011; p1: cfg_002/003/010). SFP never produces a clean −23 — v2's two "collision"-flagged configs (cfg_005 +5.26, cfg_009 +3.24) are *positive grazes* during a proximity approach.
- *Ceiling if fixed:* −23 → clean proximity = **+48 to +63**; → insertion = **+116**. Also removes catastrophic tail risk.

> **Forensic correction:** `SESSION_REPORT.md`'s final 180 s table lists v2_wide "Off-limit −23 = 2." The CSV shows v2's two collision-flagged configs are **positive** (+5.26, +3.24) — v2 has **zero** true −23 at 180 s. The catastrophic-collision risk lives entirely in the SC-trained challengers, **not** the submitted checkpoint. (Also: `force_score = 0` on *every* trial across all files — the policy is too gentle to ever incur a force penalty, i.e. it never applies seating force. The only tier_2 penalty mode that ever fires is the −24 contact.)

---

## 2. The 119.4-official vs 0-insertions-stratified gap → the pose-generalization envelope

**The gap is a pose-*coverage* cliff, not an officials-are-harder-OOD effect — and it is the reverse of intuition.** On the same 180 s suite the 3 official configs average ~26 (v2 32.6) while the 12 stratified average ~1.4 (v2). Officials score *higher* because the training set densely covers the canonical extreme-yaw modes where officials live (|yaw| ≈ 1.8–3.1), and undersamples the moderate-yaw band the stratified suite probes.

Mapping outcome onto `board_yaw`:
- **Reached (proximity or collision — plug got to the port):** cfg_005 (2.67), cfg_009 (1.29), off_1 (3.10), off_2 (−3.10), off_3 (−1.80), plus the reached SC cells (|yaw| 0.6–1.3). The three officials all sit at **|yaw| ≥ 1.8**.
- **Dead-miss:** cfg_000 (−1.49), cfg_004 (−1.45), cfg_008 (−1.23) — the **moderate-|yaw| SFP-port_0 band**.

So the envelope has **two axes, not one:** (i) a yaw-coverage hole around |yaw| ≈ 1.2–1.5, and (ii) a **port_0 penalty** — SFP port_0 is reached 0/9 while port_1 is 5/9, independent of yaw within the moderate band. `official_1` (SFP port_0) is reached only because its yaw 3.10 is a densely-demonstrated canonical mode; the *identical* port at moderate yaw (cfg_000/004/008) is dead.

**What the 119.4 vs 0 implies:** v2's 119.4/300 was earned on the **exact official poses**, where the last-inch stall lands *inside* the insertion capture radius so the plug seats. On the harder official-*family* suite poses the identical stall lands 5–8 cm out (97.7/300, proximity-only). The capability is therefore **razor-thin and pose-brittle**: v2 does not "insert" as a robust skill — it approaches to a fixed offset and *occasionally the offset happens to be inside the hole*. The generalization envelope where insertion actually completes is essentially the 3 exact official poses plus a small neighborhood; everywhere else the same policy stalls or (for SC-trained nets) collides. This is a textbook behavior-cloning support failure (compounding error off the demonstrated manifold; DAgger, arXiv:1011.0686; ACT's dense-coverage dependence, arXiv:2304.13705). 93 episodes over a U(−π,π)×wide-x/y domain is far too sparse.

---

## 3. The attractor — evidence on the mode-averaging explanation, and reachable score

### Evidence FOR mode-averaging (deterministic head collapses near-port bimodal action to ~0)

1. **Signature is positional, not temporal.** Run-logs: EE path length ≈ full initial plug-port distance (v2/official_2: path 0.18 m vs initial 0.18 m; p2/cfg_001: 0.22 vs 0.21; p2/cfg_005: 0.25 vs 0.20) → a *complete* directed approach, then a stall at 0.05–0.08 m. The 60→180 s budget change unlocked **0 insertions** — extra time does not help, proving a fixed point.
2. **Checkpoint-independent and architecture-linked.** All three checkpoints (different data, K, epochs) stall identically at the same 5–8 cm; the one shared factor is the deterministic L1 head + demos that decelerate to **zero velocity at seating**. Near-port observations resemble the seated terminal frames → predicted twist collapses toward the demos' zero-velocity mean → receding-horizon integrator sets target ≈ current TCP → no motion. Matches the a-priori ACT/ALOHA multimodality argument (arXiv:2304.13705, the paper's own motivation for CVAE + ensembling).
3. **Partial rescue by chunk-blending is mechanism-consistent.** ENSEMBLE-AB (05:30): every v2 gain was **pure tier_3** (official_1 6.9→12.6, official_3 15.1→20.5, cfg_005 11.0→19.4) with tier_2 unchanged — exactly the predicted closing-velocity injection from older chunks (predicted at farther poses, still commanding approach velocity) overriding the newest collapsed chunk. `force_score=0` everywhere corroborates: the policy never commands seating force because its near-port velocity command is ~0.

### Evidence AGAINST / complicating it

1. **The blend does not break the attractor.** n=3 REPS-FINAL: ensembling ON vs OFF arm mean 20.10±4.21 vs 23.08±2.09 → **−3.0, worse**, with a *new* failure mode (ON corrupts the *approach*: official_1 rep-3 became an incomplete/miss). Still 0/30 insertions. So mode-averaging is necessary-but-not-sufficient as the *whole* story — a naive stale-chunk blend can perturb the approach, not just the last inch.
2. **Sim noise obscures the effect.** Same config/seed/checkpoint 180 s trials vary sd ≈ 1.7–6.6 (median ~3.1, max range ~13: official_2 ON 39.7/26.7/32.7; cfg_005 OFF 5.3/7.0/9.8). The ensembling effect (~±3–5) sits **at or below** the noise floor. Two "0.0" reps (v2_off_r3/cfg_001, v2_ens_r3/official_1) are `completed=False`, ~1070 s **harness hangs, not real misses** — they inflate ON-arm variance in the SESSION_REPORT reps table.

**Verdict on the hypothesis:** strongly supported as the *dominant* mechanism for the SFP last-inch stall (positional fixed point, zero-force, tier_3-only deficit, chunk-blend injects exactly the missing tier_3). The blend's failure to *close* it is expected — a ~0.06–0.1 m transient nudge cannot span a 0.05–0.08 m gap that needs sustained closing velocity *and* seating force. The structural fix is a head that *represents* the "push-in" mode (CVAE latent + temporal ensemble as intended inference mode; or residual RL rewarded by the insertion event).

### Reachable score — last-inch-only vs approach-must-also-improve

Using oracle full-insertion ≈ 93/config and the observed bucket counts:

| Scenario | Official 3-config /300 | Stratified 15-config mean/100 |
|---|---|---|
| **Current** | v2 **119.4** (exact poses, some inserts) / 97.7 (suite officials) | v2 **7.75**; best-of-3 oracle-select **12.8** |
| **Fix last inch ONLY** (already-reached configs seat; misses/−23 unchanged) | all 3 officials seat → **~279/300 (+160)** | only the ~5 reached SFP configs seat → 5×93+10×1 ≈ 475 → **mean ~32** |
| **+ Fix SC collision & reach-the-reachable** (all 11 reachable configs seat) | (officials already covered) | 11×93+4×1 ≈ 1027 → **mean ~68** |
| **+ Close the coverage hole** (4 dead configs too) | — | 15×93 → **mean ~93 = target** |

**The decisive asymmetry:** on the **score that counts** (official /300) the last inch is nearly the *whole* prize — all three officials already reach proximity, so fixing only the last inch takes 119.4 → ~279 (+160). But the **stratified target of >90 avg cannot be reached by the last inch alone** (caps at mean ~32) because 10/15 configs never approach — the moderate-yaw + port_0 coverage hole and the SC collisions must *also* be fixed. Last inch is necessary for both; approach coverage is required only for the broad target.

---

## 4. SC-specific losses

SC is a **structurally different and net-negative** failure profile from SFP. Across the 7 SC configs (6 stratified + official_3):

| Checkpoint | SC outcomes | SC total (7 cfgs) | SC mean |
|---|---|---|---|
| v2_wide (no SC training) | 6 miss + 1 prox (official_3 33.3) | +39.3 | **+5.6** |
| p2_k8 (8 SC eps) | 3 miss + **3 coll(−23)** + 1 prox | **−36.3** | **−5.2** |
| p1_k16 (5 SC eps) | 3 miss + **3 coll(−23)** + 1 prox | **−41.2** | **−5.9** |

Findings:
- **Adding SC data made SC *worse* at eval.** v2 (zero SC demos) plays SC safe — it *misses* (+1) rather than colliding. The Phase-0/2 SC oracle demos taught the net to *drive into* SC poses under eval-band yaw + distractors, converting +1 misses into **−23 wrist-into-distractor-mount collisions**. All 6 clean −23 events in the entire suite are SC. This is the eval-time expression of the logged "SC keep-rate collapsed to 3/8" (`SESSION_REPORT` Phase-2).
- **Exactly one SC pose works — official_3** (SC port_0, yaw −1.80), reached as clean proximity by all three (24.8–33.3). It is the one SC pose the oracle covers cleanly; every other SC config misses or collides. SC has **no generalization** beyond its single demonstrated official mode.
- **Root cause (from the oracle diagnostics):** CheatCode originally had no SC branch (rammed the rotated SC frame, 19.07); the `_entrance`-frame retarget + descent-floor −0.005 fixed the *official* pose (94.1 full) but under eval-band yaw + distractors two modes remain: **(a)** descent-floor partial-insert, ins=0 (62.8/65.0) → micro-tune floor to **≈−0.007** + re-validate zero-contact; **(b)** poor approaches (28.1/42.9/58.6) → **pose-conditioned approach waypoint**. Both are *data-generation* fixes (the teacher itself fails), so no amount of retraining on the current SC demos will fix eval SC.
- **On the official /300:** trial 3 is SC (official_3). Current suite proximity ~33; oracle-seated ~94 → **~+60** recoverable on the official from the SC last inch — but only if the SC oracle is first made robust so the demos teach seating, not collision.

---

## 5. Which buckets to attack first, by expected points

Ranked by expected points **on the score that counts (official 3-config /300)**, then by the stratified target:

**#1 — Last-inch stall on the 3 officials (Bucket B). Highest by a wide margin: +160 on the official (119.4 → ~279).** All three official configs already reach clean proximity; only tier_3 insertion (~+50–68 each, ×3) is missing, from a single shared root cause (mode-averaging). No new coverage needed. Levers, in order: **(1) last-inch DAgger** — CheatCode corrective demos seeded at the observed 0.05–0.08 m stall states teaching non-zero closing velocity + seating force (directly attacks the fixed point; arXiv:1011.0686, arXiv:1810.02890); **(2) CVAE head + temporal ensemble** so "push-in" is a selectable mode not averaged to zero (arXiv:2304.13705); **(3) residual RL** rewarded by the engine insertion event for the contact-rich last inch (arXiv:2407.16677, arXiv:2412.09858). Note the plain deterministic-head ensemble is a *dead end* here (n=3: −3.0, adds variance) — the blend needs a multimodal head under it.

**#2 — SC oracle robustness (feeds Buckets B & C).** ~+60 on the official (official_3 SC last inch, 33 → ~94) *and* eliminates the −23 tail risk that makes any SC-trained checkpoint unshippable. Must be fixed *before* more SC training or retraining — it is a teacher bug (floor ≈−0.007 + pose-conditioned waypoint + zero-contact re-validation under eval-band yaw + distractors), not a model bug. Cheap relative to payoff.

**#3 — Moderate-yaw + SFP-port_0 coverage hole (Bucket A).** Only matters for the *stratified* target, not the official (v2 already reaches all 3 officials). It is the largest bucket (10/15 for v2) but the **lowest expected value per unit effort** because each rescued config needs *two* fixes to score — approach (+24–40 to proximity) **then** the last inch (+50). Target it with DAgger at the exact dead poses: **cfg_000/004/008 (SFP port_0, yaw −1.2 to −1.5)** and the rail0 cells, ±small neighborhood. Do this only after #1/#2 land, and re-screen for regression on the 60 s suite before spending a 180 s eval.

**Methodological guardrail (binds all of the above):** per-config single-180 s-trial deltas carry ±3–18 pt noise with prox↔miss outcome flips; no adoption decision is valid below ~4–5 pts even at n=3. Gate every change on the matched-seed suite IQM+CI at **180 s** (never 60 s — it truncates v2's insertions 119.4→71.3) with **≥1 genuine insertion** as the overturn bar for the submission, and n≥3 reps.

**Bottom line:** the submitted v2_wide loses ~180/300 on the official eval almost entirely to **one root cause in three already-reached configs** (the last-inch zero-velocity attractor) — a ~+160 prize behind a single structural fix. The stratified 0/75 story is a *separate*, larger problem (pose coverage + SC collision) that gates the >90-everywhere goal but not the immediate official score.