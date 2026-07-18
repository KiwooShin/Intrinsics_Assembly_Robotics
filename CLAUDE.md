# Engineering Rules — binding for ALL agents working in this repo

Set by the orchestrator agent on 2026-07-12 at the user's direction. Every agent
(main or sub-agent) that writes code in this repository MUST follow these rules.
Sub-agent prompts reference this file; deviations require explicit user approval.

## 1. Python coding standards (Google Python Style Guide)

All NEW or MODIFIED Python code must comply with the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):

- **Type hints** on every function/method signature — parameters AND return type
  (`-> None` when nothing is returned). Use `from __future__ import annotations`
  where helpful.
- **Docstrings** in Google style (`Args:` / `Returns:` / `Raises:` / `Yields:`)
  on every module, public class, and public function. One-line summary first.
- **Errors**: raise specific exceptions (`ValueError`, `FileNotFoundError`,
  `RuntimeError`, custom exceptions) with actionable messages. Never bare
  `except:`; never silently swallow exceptions. Validate inputs at public
  boundaries and fail fast.
- **Dataclasses**: use `@dataclasses.dataclass` (frozen where practical) for
  structured records — configs, episode metadata, experiment results — instead
  of dicts/tuples passed around.
- Naming: `module_name.py`, `ClassName`, `function_name`, `CONSTANT_NAME`.
  4-space indent, max line length 100, imports grouped stdlib/third-party/local
  (`.isort.cfg` at repo root applies).
- Prefer `pathlib.Path` over string paths; `logging` over bare `print` in
  library code (CLI progress output may print).

Existing legacy scripts are grandfathered; bring a file up to standard whenever
you touch it substantially.

## 2. Unit tests — required

- Every Python module gets a unittest file in a **sibling `tests/` directory**:
  `foo/bar.py` → `foo/tests/test_bar.py`. Root-level `baz.py` → `tests/test_baz.py`.
- Framework: stdlib `unittest`. Everything must run via
  `python -m unittest discover -s <dir> -v`.
- Tests must run **without ROS, Gazebo, or a GPU**: factor pure logic
  (config generation, trimming math, dataset shapes, normalization, scoring
  parsers) apart from I/O; guard hardware-dependent tests with
  `@unittest.skipUnless(torch.cuda.is_available(), ...)` or equivalent.
- A change is not "done" until its tests pass locally. Run the tests and record
  the result in your report/commit message.

## 3. Git workflow

- Remote: `origin` → https://github.com/KiwooShin/Intrinsics_Assembly_Robotics
  (branch `main`). `upstream` is the official toolkit — never push to it.
- **Only the orchestrator (main agent) commits and pushes.** Implementation
  sub-agents leave changes in the working tree and report what they changed and
  how it was verified. This prevents races between parallel agents.
- Commit granularity: one verified unit of work per commit (a script + its
  passing tests; an experiment + its recorded result).
- Message format: `<type>: <imperative subject>` where type ∈
  {feat, fix, exp, test, doc, refactor, perf}. Body states what was verified
  (tests run, scores measured). Example:
  `exp: wide-distribution ACT run (16 eps) — val first-action 0.0030 m/s`.
- **Never commit**: rosbags, `.npy`/dataset files, model checkpoints, videos,
  anything >10 MB, credentials/tokens. Keep `.gitignore` current.
- Push to `origin main` at every completed milestone (at least once per
  heartbeat cycle during autonomous runs).

## 4. Progress reporting (main agent duty)

- The orchestrator writes to `progress.md` **every 2 hours** during autonomous
  operation: a timestamped entry of **max 3 sentences** on what happened, plus
  a 1-2 line plan for the next 2 hours. Newest entry on top.
- Research/experiment reports (progress.md entries at milestones, SESSION_REPORT.md,
  plan documents) must state **which SOTA models were considered** (with checkpoint
  ids where applicable) and **which research papers were referenced** (title +
  arXiv id), so every technical decision is traceable to its sources.
- Experiment results are reported as a **summary table**: one row per experiment,
  with a title naming the experiment's main topic, a success/failure verdict,
  and the key metric(s). Detailed prose follows the table, not the other way
  around.
- After each experiment batch, a dedicated **analysis sub-agent** reviews the
  results against the research literature and past experiments (SESSION_REPORT.md
  history) and reports: why the result occurred, whether it matches published
  findings, and what to try next. Its analysis is appended to the experiment
  record.
- A **demo video** must stay available and current: after each significant
  policy/checkpoint milestone, render an insertion video (`make_video.py` or
  policy-rollout capture) into `demo/` and reference the latest one at the top
  of `progress.md`. Large videos are NOT committed to git (see §3); keep them
  on disk and reference by path.

## 5. Pretrained checkpoints — allowed

- Any publicly released model checkpoint may be used (NVIDIA, Google/DeepMind,
  OpenAI, Meta, HuggingFace community, etc.) — e.g. GR00T, pi0/SmolVLA, DINOv2,
  SAM — subject to the competition rules in `docs/challenge_rules.md`.
  Record the exact checkpoint id/URL and license in the experiment log when one
  is adopted.

## 6. Operational rules (autonomous runs)

- Storage is scarce: delete raw bags immediately after conversion
  (`collect_convert.sh` pattern). Never hold more than one ~8 GB bag.
- Do not kill or restart the simulator/VNC unless the task requires it and the
  restart procedure in Plan.md is followed.
- Record experiment outcomes (config, dataset size, val metrics, engine scores)
  in `SESSION_REPORT.md` as you go — that file is the running lab notebook.
- **Canonical metric protocol** (2026-07-12 analysis): the PRIMARY metric is the
  matched-seed in-sim eval-suite score (IQM + bootstrap CI); val first-action L1
  is a secondary diagnostic only. Every leaderboard row must pin: train-episode
  manifest (hash), config incl. chunk K, fixed epoch budget, seed set. Report
  both K=8 and K=16 where relevant; prefer mean of 3 seeds for adoption
  decisions.
- **Agent-waiter ban (2026-07-18, after 3 documented stalls):** sub-agents must
  NOT pause themselves waiting on self-armed monitors/waiters to sequence
  multi-stage pipelines — those wake-ups repeatedly failed to fire (collection
  17:26, retrain 21:59, suite-eval overnight). Every multi-stage job runs as a
  DETACHED, RESUMABLE script (`collect_campaign.sh`, `eval_batch.sh` pattern):
  nohup + progress log + skip-if-done resume + a DONE marker line. Agents
  launch it, verify the first unit completes, then report and exit. The
  orchestrator's watchdog monitors the log file, which survives heartbeat gaps.
- **5-minute liveness watchdog:** the orchestrator checks overall status every
  ~5 minutes: running processes (sim, training, collection), sub-agent activity
  (output growth), and GPU/CPU utilization. A process/agent with no observable
  progress (static logs/output AND no relevant CPU/GPU activity) across two
  consecutive checks is considered HUNG: kill it and restart the task (for
  sub-agents: stop + relaunch with a note about where it hung). Never wait
  passively on a halting process.
- **GPU watchdog:** this is a GPU-heavy project. Whenever a training/inference
  task is running, verify it is actually using the GPU (`nvidia-smi` util +
  process list, or `torch.cuda.is_available()` / device placement in code).
  If a supposedly-GPU task is CPU-bound or the GPU sits idle unintentionally,
  treat it as a bug: diagnose (wrong interpreter, missing CUDA build, tensor on
  CPU, dataloader bottleneck) and fix before letting the task continue. The
  orchestrator checks GPU utilization at every heartbeat; long-running agents
  must self-check after launching any compute job.
