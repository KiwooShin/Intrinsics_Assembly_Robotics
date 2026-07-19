I now have a complete, grounded understanding. Here is the design spec.

---

# DESIGN SPEC — Port-Bearing Auxiliary Regression Head (`port_aux`)

**Author:** design sub-agent · **Date:** 2026-07-19 · **Status:** ready for implementation
**Scope:** written spec + risk analysis only. No implementation, no sim/GPU, no commit.
**Problem restated:** the insertion policy parks 5–8 cm short of the port; the eval-legal guarded-descent fallback (`guarded_descent.py`) then descends *blind* along the pre-stall motion axis (`ApproachAxisEstimator`), never closing the 0.05–0.07 m plug↔port gap and occasionally clipping distractor NIC mounts (−24). We add an **eval-legal, train-time-privileged** learned estimate of the vector from the current TCP to the target port entrance, and wire it into the guarded-descent handoff to replace the motion-axis guess.

---

## 0. Key findings from the codebase (these shape every decision below)

1. **The raw bags are gone.** `~/data/demos/` is empty (§6 "delete raw bags immediately after conversion"). `prepare_dataset.py` never extracted `/tf`, so **no port pose exists in any converted episode**, and it cannot be recovered from bags. On-disk per episode: `center/left/right_images.npy`, `tcp_poses.npy (N,7)=[x,y,z,qx,qy,qz,qw]` in **base_link**, `tcp_velocities.npy (N,6)`, `timestamps.npy`, `wrenches.npy (N,6)`, `joint_positions.npy (N,7)`.
2. **Episode→config traceability is solid.** Two independent joins exist:
   - `~/training/<ds>/campaign_log.csv`: columns `timestamp,config,stratum,plug,rep,episode_dir,score,frames,insertion_events,wall_clock_s,status`. Directly maps `episode_dir` → `config` yaml → `status`(KEEP/DROP/SKIP) → `insertion_events`.
   - `~/data/configs_<phase>/manifest.csv`: columns include `config,stratum,port_name,target_module_name,board_x,board_y,board_yaw,grasp_z,target_translation,target_yaw,cable_type`.
   - KEEP episodes with usable `tcp_poses.npy`: **44 (ds_phase0) + 33 (ds_phase2) = 77** (~40k frames).
3. **The config yaml does NOT contain a port pose in base_link.** It has the board pose in **world** (`z:1.14` → table height) and per-rail `entity_pose.translation`(1-D rail slide)+`yaw`. Reconstructing the port-entrance pose in base_link needs the full static FK chain: `world→base_link` (robot mount, set by launch, **not in the yaml**) → `task_board`(pose) → `nic_card_mount_i`(rail offset) → `sfp_port_j_link` → `_entrance` offset. The entrance offsets are known (`NIC Card Mount/model.sdf`: SFP `0 0 -0.0458`; `SC Port/model.sdf`: SC `-0.01564`) but the chain spans `task_board.urdf.xacro` + 3 SDFs + the unrecorded robot base. **FK-from-yaml is fragile and is NOT the recommended label source** (see §1).
4. **`Task` is fully available at deploy** (`insert_cable(task, …)`): `cable_type, plug_type, plug_name, port_type, port_name, target_module_name`. The manifest carries the same fields at train time. → **Target-port identity can condition the aux head at both train and deploy with zero privileged info** — this is the resolution to multi-port ambiguity (§5).
5. **Two copies of the arch exist and must stay in lockstep:** `train_v2.Policy` (training, imported by `opt/train_v3.py`) and `DeployACT._Policy` (inference). Both are `_Encoder(3×128px→128-d) ×3 cams → cat[3·128 + state_dim] → MLP head → K·6`.
6. **The guarded-descent seam is clean.** `GuardedDescentController.cycle(...)` is the single per-cycle entry DeployACT calls; at stall it builds a `GuardedDescent(axis, anchor_pos, anchor_quat, baseline_force, …)`. The `axis` argument is exactly the injection point for the learned bearing. Reusable quaternion helpers already live in `pose_integration.py` (`quaternion_multiply`).
7. **Opt-in config pattern to mirror** (`tail_trim`/`use_wrench`): flag defaults off → byte-identical legacy path; checkpoint records the new keys; DeployACT reads keys, not env, to decide arch (`state_dim` precedent). Env flag gates *behavioral* use at deploy (`AIC_GUARDED`/`AIC_ENSEMBLE` precedent).

---

## 1. LABEL SOURCE

### 1.1 Recommendation: **hindsight terminal-TCP relabeling** (primary, works on existing data)

Use the **seated terminal TCP pose of each successful (KEEP, `insertion_events≥1`) demo** as the per-episode static target. For episode with poses `P = tcp_poses.npy (N,7)`:

- `target_pose_base = robust_terminal(P)` = component-wise **median of the last `M=5` frames'** position (+ terminal quaternion), computed on the **full (untrimmed)** episode.
- Per-frame label (TCP frame of frame *t*):
  `offset_tcp[t] = R(q_t)ᵀ · (target_pos_base − pos_t)`   where `q_t, pos_t` are frame *t*'s TCP quaternion/position.
- Optional 6-D axis label: `axis_tcp[t] = R(q_t)ᵀ · â`, `â = normalize(mean of last-M moving displacements before seating)` (base_link). See §2.4 — start 3-D, add axis later.

**Why this is the right target, not a compromise:**
- **Zero re-collection**, uses only `tcp_poses.npy` + `campaign_log.csv` (for the KEEP/insertion filter). The bags being gone makes this the *only* option for the 77 existing episodes.
- It automatically **absorbs the plug-tip→gripper offset and the ±2 mm/±0.04 rad per-config grasp variation** (`gen_config.py` randomizes `gripper_offset`), because the target is *the TCP pose from which THIS plug/grasp actually inserted*. A raw port-entrance pose would still need the (varying) grasp offset added back. For driving the TCP, terminal-TCP is *more* correct than the literal port pose.
- The target is already in **base_link**, the same frame as `tcp_poses.npy` and as the live `obs.controller_state.tcp_pose` at deploy → no FK, no static-offset bookkeeping, no `world→base_link` guess.
- At the stall (5–8 cm short) the label at that frame is a 5–8 cm vector — exactly the query the handoff needs; along the approach it shrinks to 0, so the head learns "remaining vector to insertion" at every frame.

**Caveats to encode:** (a) only KEEP+inserted episodes carry a valid label (mask others out — DROP_SCORE/failed demos never reached the port); (b) terminal pose ≈ port entrance + descent-floor offset (SFP −0.015, SC −0.005) + residual tracking error — all sub-cm to ~1.5 cm, acceptable for a ~1–2 cm target; (c) compute the target **before** `tail_trim`, then apply to the kept frames (the seated tail *is* the target — do not let tail-trim hide it).

### 1.2 Recommended upgrade for future collection: **true privileged port-entrance TF** (secondary)

CheatCode already looks up `base_link→{port}_link_entrance` (`CheatCode.py:222`). Add a one-line dump so **new** campaigns get the true port pose for free, including on *failed* demos (which hindsight cannot label):
- Modify the collection path to save `port_entrance_pose.npy (7,)` per episode (a `MoveRobot`-free static transform), or log it and have `prepare_dataset.py` extract `base_link→{port}_link_entrance` from `/tf`/`/tf_static` when present.
- This becomes (i) a **cross-check** on the hindsight labels (expect agreement to ≤2 cm modulo grasp/floor), and (ii) the label source for failure-mode coverage. Not required for the first working probe; list as a fast-follow.

### 1.3 Rejected: FK reconstruction from yaml (cross-check only)
Documented for completeness. Needs `world→base_link` (unrecorded) + board frame + `task_board.urdf.xacro` mount offsets + per-model port-link offsets + entrance offsets. High bug surface, and it reproduces the *port* pose not the *insertion* pose (still needs grasp offset). Use at most as a coarse sanity bound during label validation, never as the training label.

### 1.4 Label validity mask
A frame contributes to the aux loss iff its episode is `status==KEEP AND insertion_events≥1`. Build the mask by joining `epid` (already produced by `load_all`) → episode dir → `campaign_log.csv` row. Action-head training is unchanged (all frames).

---

## 2. ARCHITECTURE

### 2.1 Aux head
Second small MLP branching off the **same concatenated feature vector** the action head consumes (`cat[3·feat + state_dim]` — it must see the TCP quaternion in `state` to reason about the TCP frame):

```
aux_head = Sequential(
    Linear(feat*3 + state_dim, 256), ReLU,
    Linear(256, aux_dim)            # aux_dim = 3 (offset) or 6 (offset + axis)
)
```
~0.23 M params (negligible vs. encoder). Encoder + `cat[f,state]` are **shared** with the action head.

### 2.2 Joint vs frozen — recommend **joint multi-task, with a frozen-probe ablation gate**
- **Primary: joint** (shared encoder, multi-task loss). Port localization is a strong auxiliary signal that should *sharpen the encoder's port features and help the action head* (this is the whole diagnosis — the encoder currently lacks port bearing). Start with a **small** `aux_weight` so the aux gradient shapes but does not dominate the action objective.
- **Ablation/fallback: frozen-encoder probe** (`aux_freeze_encoder=True`): train only `aux_head` on a frozen pretrained encoder. Use it to (a) measure how much port bearing is *already* in the current features, and (b) ship an aux head **without touching the deployed action weights** if joint training regresses the primary in-sim score. Both selectable by flag.

### 2.3 Frame convention — **TCP frame** (primary), base_link alternative behind the same flag
Predict `offset_tcp` (per the proposal). Deploy converts once via the live TCP quaternion: `target_base = tcp_pos + R(q_live)·offset_tcp` (single rotation, reuse `pose_integration.quaternion_multiply`). TCP-frame gives lower-variance, position-invariant targets. Expose `aux_frame ∈ {'tcp','base'}`; if TCP-frame underperforms in val, `'base'` (predict directly in base_link, no deploy rotation) is a one-flag switch. Store `aux_frame` in the checkpoint so deploy math matches training.

### 2.4 Output width — **start 3-D**, 6-D optional
3-D offset alone yields *both* the target (`target_base`) and the descent axis (`normalize(target_base − anchor)`), which for a pre-aligned SFP/SC plug *is* the insertion axis. Add the explicit 3-D axis (→6-D) only if val shows the straight-line TCP→entrance vector diverges from the needed approach direction. Gate by `aux_dim`.

### 2.5 Accuracy target
Port capture region is cm-scale (SFP entrance sits 4.58 cm inside the mouth; usable capture radius ~0.5–1 cm). To beat the blind axis the head needs the **direction** right and the **magnitude** within the capture funnel:
- **Adoption gate:** holdout **median Euclidean error < 2 cm** at the near-port frames (last ~1 s before seating), **and** along-approach-axis sign correct > 98%, **and** lateral (perpendicular) error < 1 cm median (lateral error is what causes distractor clipping).
- Report per-axis cm error + Euclidean, split near-port vs. whole-episode.

---

## 3. TRAINING

### 3.1 Loss
- **L1** on the offset (robust to the terminal-pose outliers from tracking error), on **normalized** labels. Compute `omean, ostd` per-dim over the *valid* train frames; store both in the checkpoint. Normalizing puts the aux loss on ~unit scale (comparable to the normalized-twist action L1), so `aux_weight ~ O(0.1–1)` is interpretable.
- Total: `loss = action_L1 + aux_weight * masked_L1(aux_pred, offset_norm)`, where the aux term averages over **valid** frames only (`w_aux` peak default **0.5**, tunable; if push-in weighting is on, the aux loss uses the plain valid-mask mean, independent of push-in weights).
- Masked mean: `aux_loss = (|pred−tgt|.mean(-1) * valid).sum() / valid.sum().clamp(min=1)`.

### 3.2 Normalization & ordering
Label pipeline order per episode: compute `target_pose_base` on full episode → per-frame `offset` → **then** apply `tail_trim` keep-mask (labels ride along with kept frames) → concat → normalize with train-split `omean/ostd`. Val labels use the **train** `omean/ostd`.

### 3.3 Validation protocol
Reuse train_v3's **held-out whole-episode** split (`val_globs`). Add `val_offset_cm` (de-normalized Euclidean, cm) and `val_offset_axis_cm`/`val_offset_lateral_cm` diagnostics, reported every eval epoch alongside the existing action L1. Report the near-port subset (frames with speed already low / last 1 s) separately — that is the operating point. Keep the action metrics unchanged so joint-training regressions are visible.

### 3.4 Integration into `opt/train_v3.py` (mirror the `tail_trim`/`use_wrench` opt-in pattern)
New `TrainConfig` fields (all default off/neutral → **byte-identical legacy path**):
`port_aux: bool=False`, `aux_dim: int=3`, `aux_weight: float=0.5`, `aux_frame: str='tcp'`, `aux_freeze_encoder: bool=False`, `aux_label_glob: str=''` (campaign_log/manifest locator; default auto-derive `<ds>/campaign_log.csv` next to each episode), plus property `port_aux_enabled`. Validate in `__post_init__` (`aux_dim∈{3,6}`, `aux_weight≥0`, `aux_frame∈{'tcp','base'}`).
`train()` changes: when `cfg.port_aux`, `_load_split` also returns `(offset_labels, valid_mask)`; build `Policy(..., aux_dim=cfg.aux_dim)`; in `run_epoch` add the masked aux term; if `aux_freeze_encoder`, put encoder in eval + exclude its params from the optimizer (or `requires_grad_(False)`); extend the saved checkpoint dict (§4.1); add the val metric. With `port_aux=False`, none of this executes.

---

## 4. DEPLOY WIRING

### 4.1 Checkpoint keys (added by train_v3; read opt-in by DeployACT)
`aux_dim:int`, `aux_frame:str`, `omean, ostd` (cpu tensors), `has_aux:bool=True`. Absent on old checkpoints → DeployACT builds a plain `_Policy` (no aux head), byte-identical.

### 4.2 `DeployACT` changes
- Build `_Policy(K, state_dim, aux_dim=ck.get('aux_dim',0))`; when `aux_dim>0`, `forward` returns `(action, aux)` (or a separate `forward_aux`). Load `omean/ostd`.
- New `_predict_offset(obs) -> PortOffsetPrediction`: run the aux head, de-normalize (`aux*ostd+omean`), if `aux_frame=='tcp'` rotate into base_link with the **live** TCP quaternion → `offset_base` (+ world target `tcp_pos+offset_base`); attach a plausibility result (§4.4).
- Provide the offset to the guarded controller via a small **callable seam** (below), only when `AIC_GUARDED_AUX=1` **and** `has_aux`. Otherwise the controller keeps today's motion-axis behavior exactly.

### 4.3 `guarded_descent.py` changes — bearing source seam
Introduce a `PortBearingProvider` protocol: `predict(pos_base, quat_base) -> (target_base | None, magnitude, ok:bool)`. `GuardedDescentController.__init__` gains an optional `bearing_provider=None` and new config fields `use_aux_bearing`, plausibility bounds (`aux_min_mag=0.005`, `aux_max_mag=0.12` m), `aux_consistency_std=0.01` m, `reaim` (bool). At handoff (`cycle` stall branch):
1. **Query strategy:** query the provider over the **last few approach frames** (e.g. buffer predictions during the low-speed pre-stall window) and take the **median target** — steadier than a single query at the exact stall frame.
2. Build the guarded descent from the aux target when plausible: `axis = normalize(target_base − anchor)`, and **`travel_cap = min(config.travel_cap, |target_base − anchor| + margin)`** so the descent aims *at* the port and stops there instead of overrunning into a distractor.
3. **Fallback to `ApproachAxisEstimator`** (today's behavior) when any check fails (§4.4). Keep the existing contact/back-off/travel-cap guards unchanged as the safety net.
- Optional `reaim`: re-query each descent cycle and blend the axis slowly (rate-limited) toward the fresh target; default **off** (query-once-at-stall) to avoid jitter — recommend enabling only after the static version validates.

### 4.4 Plausibility / consistency gate (fallback triggers)
Fall back to motion-axis if: no aux head in checkpoint; `AIC_GUARDED_AUX` unset; `|offset|` outside `[aux_min_mag, aux_max_mag]`; cross-frame prediction `std > aux_consistency_std`; predicted target's along-motion component is negative (points *away* from the established approach — likely a wrong-port lock); or NaN/inf. Log which source won (`[guarded] bearing=aux target=… |d|=…mm` vs `bearing=motion-axis (reason=…)`).

### 4.5 New env flags (mirror `AIC_GUARDED_*`)
`AIC_GUARDED_AUX` (enable), `AIC_GUARDED_AUX_MINMAG`, `AIC_GUARDED_AUX_MAXMAG`, `AIC_GUARDED_AUX_STD`, `AIC_GUARDED_REAIM`. Parsed in `GuardedDescentConfig.from_env`.

---

## 5. TESTS + RISKS

### 5.1 Unit-testable seams (all CPU-only, no ROS/torch/GPU per CLAUDE.md §2)
- `port_offset.py` (pure numpy): `rotate_vector_by_quat` / `_inv` round-trip; `tcp_frame_offset` ∘ `base_target_from_tcp_offset` == identity; `robust_terminal`; `per_frame_tcp_offsets` shape/zero-at-terminal; plausibility gate boundaries.
- `opt/port_labels.py`: `campaign_log`→episode join; KEEP/insertion mask; label-before-trim ordering; missing-file / failed-episode handling; normalization stats over valid frames only.
- `guarded_descent.py`: handoff picks aux target when plausible; each fallback trigger fires and reverts to motion-axis; `travel_cap` clamps to `|offset|+margin`; byte-identical when `bearing_provider=None`.
- `opt/config.py` / `opt/train_v3.py`: new-flag validation; aux path shapes; **`port_aux=False` produces the exact legacy tensors/checkpoint** (regression guard).
- `DeployACT` offline test (extend `test_deploy_act_offline.py`): old checkpoint → no aux head, unchanged; aux checkpoint → `_predict_offset` returns a base_link target of plausible magnitude.

### 5.2 Top-3 failure modes
1. **Regression accuracy at unseen poses / distribution shift.** 77 KEEP episodes, ~12 strata; the head may interpolate board poses poorly and be worst exactly near the port (small residuals, largest leverage). *Mitigations:* held-out-episode val with the near-port cm gate (§2.5); keep `aux_weight` small so a weak head cannot corrupt the action policy; the deploy plausibility+consistency gate degrades to today's motion-axis rather than steering into a wall; frozen-probe ablation to quantify headroom before trusting joint training.
2. **Frame-convention bugs (silent).** TCP-frame ↔ base_link rotation, quaternion order (`[x,y,z,w]` in poses vs `pose_integration` conventions), label computed post-trim, or train/deploy frame mismatch — any of these yields a confidently-wrong vector. *Mitigations:* store `aux_frame` in the checkpoint and assert-match at deploy; round-trip unit tests; a one-episode numeric fixture asserting `offset_tcp[terminal]≈0` and `base_target_from_tcp_offset(...)≈target`; log the resolved target and gate on its magnitude.
3. **Multi-port ambiguity — the label/net picking THE target port.** Configs place 1–2 **distractor** NIC cards (and, for SC, the other SC port) on non-target rails (`gen_config._present_nic/_present_sc`); two same-type ports can both be visible. The *label* is unambiguous (hindsight terminal-TCP = where the cable actually inserted into the task's `target_module_name`), but the *net* must choose from images. *Mitigations:* **condition the aux head on the target-port identity from `Task`** (eval-legal at deploy, present in the manifest at train) — minimally `[is_sc, port_index]` (fully determined by `plug_type`/`port_name`), which disambiguates SC-vs-SFP and the two SFP ports; residual rail-level ambiguity between two same-type distractor rails is mitigated because the learned approach has already parked the arm *near the target rail*, so the near-port viewpoint + TCP state bias the head correctly; the deploy along-motion-consistency check rejects a wrong-rail lock (points away from the established approach) and falls back to motion-axis. Note the plug/cable is grasped and in-frame, so plug type is directly visible — but do not rely on that alone; add the explicit conditioning.

---

## 6. FILE-BY-FILE CHANGE LIST

| # | File | Change | New? |
|---|------|--------|------|
| 1 | `aic_example_policies/aic_example_policies/ros/port_offset.py` | Pure-numpy: quat vector rotate/inv, `tcp_frame_offset`/`base_target_from_tcp_offset`, `robust_terminal`, `per_frame_tcp_offsets`, `approach_axis_label`, `PortOffsetPrediction` + plausibility. Reuse `pose_integration.quaternion_multiply`. | NEW |
| 2 | `.../ros/tests/test_port_offset.py` | Round-trip, identity, boundary tests. | NEW |
| 3 | `opt/port_labels.py` | `campaign_log`/manifest join → per-episode target → per-frame offset labels + valid mask + norm stats; label-before-trim ordering. | NEW |
| 4 | `opt/tests/test_port_labels.py` | Join, mask, ordering, missing-file tests (tmp fixtures). | NEW |
| 5 | `train_v2.py` | `Policy.__init__(..., aux_dim=0)`; when >0 add `aux_head`; `forward` returns `(act, aux)` under aux; **gated so `aux_dim=0` is byte-identical**. | MODIFY |
| 6 | `aic_example_policies/.../ros/DeployACT.py` | `_Policy` gains `aux_dim`; load `aux_dim/aux_frame/omean/ostd/has_aux`; `_predict_offset`; wire `PortBearingProvider` into `GuardedDescentController` when `AIC_GUARDED_AUX` + `has_aux`. | MODIFY |
| 7 | `.../ros/guarded_descent.py` | `PortBearingProvider` seam; controller uses aux target for axis + `travel_cap`; plausibility/consistency fallback to `ApproachAxisEstimator`; new `AIC_GUARDED_AUX*` env in `from_env`. | MODIFY |
| 8 | `.../ros/tests/test_guarded_descent.py` | Extend: aux-target handoff, every fallback trigger, cap clamp, None-provider byte-identity. | MODIFY |
| 9 | `opt/config.py` | `TrainConfig`: `port_aux, aux_dim, aux_weight, aux_frame, aux_freeze_encoder, aux_label_glob` + `port_aux_enabled` + validation. Extend `TrainResult` with `val_offset_cm` fields. | MODIFY |
| 10 | `opt/train_v3.py` | Load offset labels/mask in `_load_split`; build aux `Policy`; masked aux loss in `run_epoch`; frozen-encoder branch; val cm metric; extended checkpoint dict; new CLI args. | MODIFY |
| 11 | `opt/tests/test_config.py`, `opt/tests/test_train_v3.py` | New-flag validation + `port_aux=False` legacy-identity regression + aux-path shapes. | MODIFY |
| 12 | `aic_example_policies/.../ros/tests/test_deploy_act_offline.py` | Old-ckpt unchanged; aux-ckpt `_predict_offset` plausibility. | MODIFY |
| 13 *(fast-follow, optional)* | `prepare_dataset.py` (+ collection dump) | Extract/save `port_entrance_pose.npy` from `/tf` for **future** campaigns (true privileged label + failed-episode coverage). Not needed for first probe. | MODIFY |

**Load-bearing exact facts for the implementer:** entrance offsets — SFP `0 0 -0.0458` (`aic_assets/models/NIC Card Mount/model.sdf:374`), SC `-0.01564` (`SC Port/model.sdf`, per `cheatcode_targeting.py:13`); descent floors SFP `-0.015`, SC `-0.005` (`cheatcode_targeting.py:38,47`); pose order `[x,y,z,qx,qy,qz,qw]` (`prepare_dataset.EpisodeFrame`); `DT_FRAME=0.275`; KEEP filter = `status==KEEP AND insertion_events≥1` from `campaign_log.csv`.

---

## 7. ESTIMATED IMPLEMENTATION TIME

| Work unit | Est. |
|---|---|
| `port_offset.py` + tests | 2–3 h |
| `opt/port_labels.py` + tests (join + label pipeline) | 2–3 h |
| `Policy` aux head (both copies) + `config.py` flags + `train_v3` integration + tests | 4–5 h |
| `DeployACT` + `guarded_descent.py` wiring + tests | 3–4 h |
| **Coding + unit tests total** | **~1.5–2 dev-days (11–15 h)** |
| GPU: label build, joint vs frozen train, cm-error tuning of `aux_weight` (out of this design's no-GPU scope) | +0.5–1 day iterate |

**Recommended build order (each a committable unit per CLAUDE.md §3):** (1) `port_offset.py`+tests → (2) `port_labels.py`+tests → (3) `Policy`/`config`/`train_v3` aux path + regression tests (train a checkpoint) → (4) `DeployACT`/`guarded_descent` wiring + tests → (5) in-sim guarded-descent A/B (aux bearing vs motion-axis) on the same seed set, primary metric = matched-seed eval-suite IQM per CLAUDE.md §6.

**First experiment to gate everything:** frozen-encoder probe on the current deployed encoder → measures how much port bearing is already present and gives the baseline cm error before any joint training risk. If the frozen probe already hits < 2 cm near-port, ship it as a pure add-on (no action-weight change); if not, escalate to joint training with small `aux_weight`.