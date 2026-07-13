"""Frozen dataclass configs and result records for the opt/ tooling.

These carry structured records between the trainer, the benchmark, and the
sweep harness instead of loose dicts/tuples (per repo engineering rules).
Only the standard library is imported so this module is testable CPU-only.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

# Baseline hyperparameters, mirroring train_v2.py defaults, so every result is
# comparable to the existing ACT-lite trainer.
DEFAULT_TRAIN_EPS: str = "/home/kiwoos/training/ds_wide/ep_*"
DEFAULT_VAL_GLOBS: tuple[str, ...] = (
    "/home/kiwoos/training/ds_wide/ep_8",
    "/home/kiwoos/training/ds_wide/ep_9",
    "/home/kiwoos/training/ds_wide/ep_10",
)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters + runtime knobs for a single train_v3 run.

    Attributes:
        train_globs: Comma-separated episode globs for the training set.
        val_globs: Comma-separated episode globs held out for validation. When
            empty the run overfits (val == train), matching train_v2.
        epochs: Number of epochs to train.
        bs: Mini-batch size (frames).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (decoupled).
        img: Square image side the cameras are resized to.
        k: Action-chunk length (number of future 6-D twists predicted).
        ema_decay: Exponential-moving-average decay for eval weights; 0 disables.
        compile_mode: torch.compile mode ('none','default','reduce-overhead',
            'max-autotune','max-autotune-no-cudagraphs').
        fused_adam: Use the fused CUDA AdamW kernel when on GPU.
        use_fp8: Attempt torchao float8 training (degrades to bf16 if absent).
        seed: RNG seed for deterministic runs.
        out: Checkpoint output path ('' to skip saving).
        shift_pad: DrQ random-shift radius in pixels applied to training images
            only (0 disables). See opt.augment.random_shift (arXiv:2004.13649).
        proprio_dropout: Per-sample probability of zeroing the normalized state
            during training only (0 disables). See opt.augment.proprio_dropout
            (arXiv:2509.18644); eval/deploy always keeps the true state.
    """

    train_globs: str = DEFAULT_TRAIN_EPS
    val_globs: str = ",".join(DEFAULT_VAL_GLOBS)
    epochs: int = 60
    bs: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img: int = 128
    k: int = 16
    ema_decay: float = 0.0
    compile_mode: str = "max-autotune"
    fused_adam: bool = True
    use_fp8: bool = False
    seed: int = 0
    out: str = ""
    shift_pad: int = 0
    proprio_dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate hyperparameters at the public boundary.

        Raises:
            ValueError: If any numeric knob is out of its valid range.
        """
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.bs <= 0:
            raise ValueError(f"bs must be > 0, got {self.bs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.img <= 0 or self.img % 8 != 0:
            raise ValueError(f"img must be a positive multiple of 8, got {self.img}")
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.shift_pad < 0:
            raise ValueError(f"shift_pad must be >= 0, got {self.shift_pad}")
        if not 0.0 <= self.proprio_dropout <= 1.0:
            raise ValueError(
                f"proprio_dropout must be in [0, 1], got {self.proprio_dropout}"
            )

    @property
    def ema_enabled(self) -> bool:
        """Whether EMA-of-weights evaluation is active."""
        return self.ema_decay > 0.0


@dataclasses.dataclass(frozen=True)
class TrainResult:
    """Outcome of a train_v3 run."""

    frames: int
    train_frames: int
    val_frames: int
    epochs: int
    throughput_fps: float
    ms_per_epoch: float
    final_train_loss: float
    final_val_loss: float
    best_val_first_action: float
    wall_s: float
    fp8_active: bool
    compile_mode: str
    ckpt_path: str = ""
    # De-normalized full-chunk L1 (m/s), averaged over all K steps and 6 dims;
    # ``best`` is the minimum over eval checkpoints. Defaults keep older callers
    # that do not populate them working.
    best_val_full_chunk: float = float("inf")
    final_val_full_chunk: float = 0.0


@dataclasses.dataclass(frozen=True)
class TrainBenchResult:
    """Training-throughput benchmark measurement."""

    label: str
    frames: int
    steps: int
    throughput_fps: float
    ms_per_step: float
    gpu_util_mean: float  # dmon sm% (streaming-multiprocessor activity), mean
    gpu_util_max: float  # dmon sm%, peak
    mem_activity_pct: float  # dmon mem% (memory-controller activity), peak


@dataclasses.dataclass(frozen=True)
class InferBenchResult:
    """Single-observation inference-latency measurement (bs=1 by default)."""

    label: str
    batch: int
    dtype: str
    compiled: bool
    includes_h2d: bool
    iters: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    std_ms: float

    @property
    def hz(self) -> float:
        """Achievable inference rate (1/mean latency)."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float("inf")


@dataclasses.dataclass(frozen=True)
class TrialConfig:
    """One hyperparameter point in a sweep."""

    trial_id: int
    lr: float
    k: int
    img: int
    weight_decay: float
    ema_decay: float
    seed: int


@dataclasses.dataclass
class TrialResult:
    """Mutable running record for a sweep trial (updated at each rung)."""

    config: TrialConfig
    epochs_done: int = 0
    val_first_action: float = float("inf")
    val_l1: float = float("inf")
    train_l1: float = float("inf")
    alive: bool = True
    wall_s: float = 0.0
    history: Optional[list[tuple[int, float]]] = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
