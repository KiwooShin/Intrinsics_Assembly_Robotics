# Autonomous Session Report — 2026-06-18

Running log of the unattended session (user away ~10h). See [Plan.md](./Plan.md) for the full plan.

## TL;DR
End-to-end imitation pipeline is **verified**: generate randomized scene → CheatCode demo → trim → train ACT-style policy → validate. Overfit confirmed; training is GPU-efficient (95% util); proper held-out validation shows few near-identical demos don't generalize → now expanding **diversity** and designing smart/efficient data generation.

## Verified results
| Check | Result |
|---|---|
| Dataset generation | 5 near-eval demos collected, all CheatCode ~93/100 |
| Trim correctness | task-window trim drops idle+reset (≈530 clean frames/ep, ends seated) |
| Overfit (val==train) | first-action err **0.0005 m/s** (~1% range), uniform across 16-step chunk → model fits data |
| Held-out (train 4 / val 1) | train 0.075 vs **val 0.34** (norm L1); val first-action **0.0024 m/s** → limited generalization |
| GPU efficiency | util **0→95%**, throughput **1,326 → 5,608 fr/s** (~4.2×) |

## Efficiency analysis (Phase 2)
- Baseline was both CPU-loader-bound (~2,600 fr/s ceiling) AND launch-overhead-bound (24.7 ms/step for a 0.75M model in fp32/bs64).
- Fixes (in `train_v2.py`): **entire dataset on GPU** (0.78 GB fp16 → no CPU loader), **bf16 autocast**, **channels_last**, **torch.compile**, **bs256**, on-GPU loss accumulation (one sync/epoch).
- Result: 95% GPU util, 5,608 fr/s. For datasets too big for VRAM later, fall back to a cached/streaming loader.

## Pipeline (scripts in repo + ~/training)
- `gen_config.py --mode {near,wide}` — randomized single-trial SFP configs (near eval, or full SFP eval distribution: all rails, both ports, wide board pose).
- `collect_one.sh` / `collect_set.sh` — CheatCode demo collection (correct scoring detection).
- `collect_convert.sh` — **storage-light**: per-demo collect → score-filter (≥60) → trim-convert → **delete bag** (never holds >1 of the ~8 GB bags).
- `prepare_dataset.py` — bag→.npy + task-window trim.
- `train_v2.py` — efficient ACT-style trainer (all-GPU, bf16, compile, proper val split).
- `verify_dataset.py`, `make_video.py`, `profile_loader.py`, `profile2.py`.

## Generalization experiment (Phase 3 result)
Collected **11 wide** demos (full SFP eval distribution: all rails, both ports, wide board pose) via the storage-light pipeline; 1 of 12 dropped (partial insert, score 65, no insertion_event — filtered correctly). Same held-out wide val set (`ds_wide/ep_8,9,10`), three training conditions:

| Train set | Val L1 (16-step chunk) | Val first-action |err| |
|---|---|---|
| NEAR only (5) | ~0.58 | 0.00527 m/s |
| WIDE (8) | ~0.37 | 0.00376 m/s |
| NEAR+WIDE (13) | ~0.30 | **0.00296 m/s** |

**Findings:**
- Diversity + count improve held-out generalization **monotonically** (first-action err −44% near→near+wide). Near-eval-only data does NOT cover the wide distribution.
- The full 16-step chunk generalizes worse than the first action (val 0.30 vs train 0.057) → the multimodal far-future a deterministic head can't capture; motivates (a) more data and (b) full LeRobot ACT with the CVAE latent + temporal ensembling.
- Pipeline scales cleanly; GPU-efficient throughout.

Final model trained on all 16 episodes → `~/training/ckpt/v2_wide.pt` (for in-sim deployment via `DeployACT`).

## Next
- Test `DeployACT` in-sim (validate the submission/eval path), then build the **failure-driven loop**: run the learned policy across configs, score with the engine, generate CheatCode demos where it fails, retrain.
- Recommend moving to **LeRobot ACT** for the competitive model (better chunk/multimodal generalization), reusing this data pipeline.

## Strategy: generating additional data + training (Phase 4 design)
1. **Coverage-first, step-wise.** Diversity matters more than count with few demos. Expand 5 → ~15 → ~40, sampling the full eval distribution; measure the val curve at each step and stop when it plateaus / hits target.
2. **Storage-light by default.** Always convert-and-delete bags (done). Optional next: a live recorder node that writes the trimmed episode directly from topics, skipping the 8 GB bag entirely.
3. **Failure-driven (DAgger-style).** Wrap the trained model as an `aic_model` Policy, run it (not CheatCode) across many configs, score with the engine, find where it fails (wrong port / collision / no insertion), then generate CheatCode demos AT those failing configs and retrain. Focuses data where the policy is weak. (Building the policy-deployment wrapper also serves the final submission.)
4. **On-the-fly training loop (advanced).** Interleave a generator process (CheatCode rollouts → episodes) with training via a bounded replay buffer; discard old episodes once learned. Keeps storage flat regardless of total demos seen.

## Open decisions for the user (when back)
- Stay with the standalone ACT-lite trainer, or invest in full LeRobot ACT (CVAE latent + temporal ensemble) for better multimodal/far-future prediction?
- Target dataset size / compute budget for the first generalizing model.
- Priority: breadth (cover SFP+SC, clutter) vs depth (nail SFP trials 1–2 first).
