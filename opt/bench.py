"""Reproducible benchmark CLI for the ACT-lite policy.

Two subcommands:

  train  -- measure training throughput (frames/s, ms/step) and GPU utilization
            (sampled from nvidia-smi during the run) for the train_v2 model.
  infer  -- measure single-observation inference latency (bs=1) across a matrix
            of {eager, compiled} x {fp32, bf16}, including the host->device copy
            of the three 128x128 camera images, plus compute-only references.

Results print as a table and are written to opt/results/*.json.

CLI examples:
    ~/miniconda3/bin/python -m opt.bench train --eps '~/training/ds_wide/ep_*' --steps 300
    ~/miniconda3/bin/python -m opt.bench infer --ckpt ~/training/ckpt/v2_wide.pt --iters 300
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import pathlib
import statistics
import subprocess
import sys
import threading
import time

# Only the repo root is added to sys.path (for train_v2); torch/numpy come from
# the active interpreter (run with ~/miniconda3/bin/python).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_v2 as tv2  # noqa: E402
from opt.config import InferBenchResult, TrainBenchResult  # noqa: E402
from opt.train_v3 import setup_backend  # noqa: E402

_LOG = logging.getLogger("opt.bench")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _REPO_ROOT / "opt" / "results"


class GpuSampler:
    """Background sampler of GPU SM activity via ``nvidia-smi dmon``.

    Polls ``nvidia-smi dmon`` in a daemon thread between :meth:`start` and
    :meth:`stop`, reading the ``sm`` (streaming-multiprocessor activity) and
    ``mem`` (memory-controller activity) percentage columns. On GB10/Grace-
    Blackwell the ``--query-gpu=utilization.gpu`` field is not populated (it
    returns 0 and ``memory.used`` is ``[N/A]`` under unified memory), so the
    ``dmon`` interface is used instead for a meaningful utilization signal.
    """

    def __init__(self, period_s: float = 0.0, gpu_index: int = 0) -> None:
        self.period_s = period_s
        self.gpu_index = gpu_index
        self._util: list[float] = []
        self._mem: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> tuple[float, float] | None:
        """Read one dmon sample; return (sm%, mem-controller%) or None."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "dmon", "-i", str(self.gpu_index), "-c", "1",
                 "-s", "u"],
                capture_output=True, text=True, timeout=6, check=True,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            # dmon "-s u" layout: gpu sm mem enc dec (jpg ofa on newer builds).
            if len(cols) >= 3 and cols[0].isdigit():
                try:
                    return float(cols[1]), float(cols[2])
                except ValueError:
                    return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._poll_once()  # dmon -c 1 blocks ~1 s (its sample period)
            if sample is not None:
                self._util.append(sample[0])
                self._mem.append(sample[1])
            if self.period_s > 0:
                self._stop.wait(self.period_s)

    def start(self) -> None:
        """Start sampling in the background."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def util_mean(self) -> float:
        return statistics.mean(self._util) if self._util else 0.0

    @property
    def util_max(self) -> float:
        return max(self._util) if self._util else 0.0

    @property
    def mem_max(self) -> float:
        return max(self._mem) if self._mem else 0.0


def _percentile(values: list[float], q: float) -> float:
    """Return the ``q`` percentile (0-100) via linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - frac) + s[hi] * frac


def bench_train(
    ep_dirs: list[str], img: int, k: int, bs: int, steps: int, compile_mode: str
) -> TrainBenchResult:
    """Benchmark training throughput and GPU utilization.

    Args:
        ep_dirs: Episode directories to load onto the GPU.
        img: Image side length.
        k: Action-chunk length.
        bs: Batch size.
        steps: Number of optimizer steps to time (after warmup).
        compile_mode: torch.compile mode ('none' to disable).

    Returns:
        A :class:`TrainBenchResult`.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("bench_train requires a CUDA device")
    setup_backend()
    imgs, state, act, _ = tv2.load_all(ep_dirs, img, k)
    smean, sstd = state.mean(0), state.std(0) + 1e-6
    amean, astd = act.reshape(-1, 6).mean(0), act.reshape(-1, 6).std(0) + 1e-6
    state_n = (state - smean) / sstd
    act_n = (act - amean) / astd
    n = imgs.shape[0]

    model = tv2.Policy(k).to(DEVICE)
    runnable = model if compile_mode == "none" else torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    def one_step() -> None:
        b = torch.randint(0, n, (bs,), device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.l1_loss(runnable(imgs[b], state_n[b]), act_n[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(20):  # warmup (compile + cudnn autotune)
        one_step()
    torch.cuda.synchronize()

    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sampler.stop()

    return TrainBenchResult(
        label=f"train[{compile_mode}]",
        frames=steps * bs,
        steps=steps,
        throughput_fps=steps * bs / dt,
        ms_per_step=1000.0 * dt / steps,
        gpu_util_mean=sampler.util_mean,
        gpu_util_max=sampler.util_max,
        mem_activity_pct=sampler.mem_max,
    )


def _load_policy(ckpt_path: str) -> tuple[torch.nn.Module, int, int]:
    """Load the deployed ACT-lite policy from a checkpoint.

    Returns:
        Tuple of (eval-mode model, K, img).
    """
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    k, img = int(ck["K"]), int(ck["img"])
    model = tv2.Policy(k).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, k, img


def _time_infer(
    fn, iters: int, warmup: int = 30
) -> tuple[float, float, float, float]:
    """Time a no-arg inference closure, returning (mean, p50, p90, std) ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    mean = statistics.mean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, _percentile(samples, 50), _percentile(samples, 90), std


def bench_infer(ckpt_path: str, iters: int) -> list[InferBenchResult]:
    """Benchmark single-observation inference latency across a config matrix.

    Measures {eager, compiled(reduce-overhead)} x {fp32, bf16} including the
    host->device copy of three 128x128 RGB camera frames into a reusable pinned
    buffer (the realistic on-robot path), plus eager compute-only references
    that isolate the H2D cost.

    Args:
        ckpt_path: Path to a train_v2/v3 checkpoint.
        iters: Timed iterations per configuration.

    Returns:
        A list of :class:`InferBenchResult`, one per configuration.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("bench_infer requires a CUDA device")
    setup_backend()
    model, _k, img = _load_policy(ckpt_path)

    # Realistic single observation: uint8 HxWx3 per camera on the host.
    rng = np.random.default_rng(0)
    cpu_imgs = torch.empty((1, 3, 3, img, img), dtype=torch.float32, pin_memory=True)
    cpu_imgs.copy_(torch.from_numpy(rng.random((1, 3, 3, img, img), dtype=np.float32)))
    cpu_state = torch.zeros((1, 7), dtype=torch.float32, pin_memory=True)
    gpu_imgs = torch.empty_like(cpu_imgs, device=DEVICE)
    gpu_state = torch.empty_like(cpu_state, device=DEVICE)

    results: list[InferBenchResult] = []

    def make_closure(runnable, dtype, include_h2d: bool):
        def closure() -> None:
            if include_h2d:
                gpu_imgs.copy_(cpu_imgs, non_blocking=True)
                gpu_state.copy_(cpu_state, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ):
                _ = runnable(gpu_imgs, gpu_state)

        return closure

    # Compiled variant: reduce-overhead captures CUDA graphs (best for latency).
    compiled = torch.compile(model, mode="reduce-overhead")

    matrix = [
        ("eager", model, False, torch.float32, True),
        ("eager", model, False, torch.bfloat16, True),
        ("compiled", compiled, True, torch.float32, True),
        ("compiled", compiled, True, torch.bfloat16, True),
        ("eager-compute-only", model, False, torch.float32, False),
        ("eager-compute-only", model, False, torch.bfloat16, False),
    ]
    for name, runnable, is_compiled, dtype, include_h2d in matrix:
        dt_name = "fp32" if dtype == torch.float32 else "bf16"
        mean, p50, p90, std = _time_infer(
            make_closure(runnable, dtype, include_h2d), iters
        )
        label = f"{name}-{dt_name}" + ("+h2d" if include_h2d else "")
        results.append(
            InferBenchResult(
                label=label, batch=1, dtype=dt_name, compiled=is_compiled,
                includes_h2d=include_h2d, iters=iters, mean_ms=mean, p50_ms=p50,
                p90_ms=p90, std_ms=std,
            )
        )
        _LOG.info("%-26s mean=%.3f ms  p50=%.3f  p90=%.3f  (%.0f Hz)",
                  label, mean, p50, p90, 1000.0 / mean if mean else 0.0)
    return results


def _expand(spec: str) -> list[str]:
    """Expand comma-separated globs into a sorted unique dir list."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.extend(sorted(glob.glob(str(pathlib.Path(part).expanduser()))))
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def _write_json(name: str, payload: object) -> pathlib.Path:
    """Write ``payload`` as JSON under opt/results and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="ACT-lite training/inference benchmark.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="training throughput + GPU util")
    ap_tr.add_argument(
        "--eps",
        default="/home/kiwoos/training/ds_wide/ep_0,/home/kiwoos/training/ds_wide/ep_1",
    )
    ap_tr.add_argument("--img", type=int, default=128)
    ap_tr.add_argument("--k", type=int, default=16)
    ap_tr.add_argument("--bs", type=int, default=256)
    ap_tr.add_argument("--steps", type=int, default=300)
    ap_tr.add_argument("--compile-mode", default="none",
                       choices=["none", "default", "reduce-overhead", "max-autotune",
                                "max-autotune-no-cudagraphs"])

    ap_in = sub.add_parser("infer", help="single-obs inference latency matrix")
    ap_in.add_argument("--ckpt", default="/home/kiwoos/training/ckpt/v2_wide.pt")
    ap_in.add_argument("--iters", type=int, default=300)

    a = ap.parse_args(argv)
    if a.cmd == "train":
        res = bench_train(_expand(a.eps), a.img, a.k, a.bs, a.steps, a.compile_mode)
        print(f"[bench.train] {res.label}: {res.throughput_fps:.0f} fr/s  "
              f"{res.ms_per_step:.2f} ms/step  sm mean={res.gpu_util_mean:.0f}% "
              f"max={res.gpu_util_max:.0f}%  mem-ctrl={res.mem_activity_pct:.0f}%")
        path = _write_json("bench_train.json", dataclasses.asdict(res))
        print(f"[bench.train] wrote {path}")
    else:
        results = bench_infer(a.ckpt, a.iters)
        print("\n[bench.infer] single-observation latency (bs=1):")
        print(f"  {'config':<26} {'mean_ms':>9} {'p50':>8} {'p90':>8} {'Hz':>8}")
        for r in results:
            print(f"  {r.label:<26} {r.mean_ms:>9.3f} {r.p50_ms:>8.3f} "
                  f"{r.p90_ms:>8.3f} {r.hz:>8.0f}")
        path = _write_json("bench_infer.json", [dataclasses.asdict(r) for r in results])
        print(f"[bench.infer] wrote {path}")


if __name__ == "__main__":
    main()
