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
        tail_trim: Drop each demo's seated zero-velocity tail from the training
            sequence (non-destructive; the ``.npy`` files are never rewritten).
            Removes frames that alias the near-port approach. Off = legacy path.
        tail_trim_threshold: Linear-speed threshold (m/s) below which a trailing
            frame is treated as "stopped". Shared by both
            :func:`terminal_tail_trim_length` (tail trim) and
            :func:`last_inch_window` (the speed-derived last-inch window end).
        tail_trim_margin_s: Seconds of frames kept after the last moving frame,
            so a short "arrived" cue survives the trim.
        last_inch_s: Last-inch lookback window in seconds. When > 0 the trainer
            keeps ONLY each demo's terminal approach->seat window (the inverse of
            ``tail_trim``) and drops the long lead-in, for an insertion
            specialist (INSERTION_PLAN.md P-INSERT-1). ``0.0`` = off (legacy
            path). Mutually exclusive with ``tail_trim``. Uses the exact
            ``insertion_frame.npy`` seat marker per episode when present, else the
            speed-derived end.
        last_inch_min_frames: Minimum frames kept in each last-inch window (>= 1),
            so short demos still yield a usable window.
        use_wrench: Concatenate the 6-D wrist wrench onto the 7-D TCP state,
            training a 13-D-state policy. The checkpoint records ``state_dim`` so
            DeployACT can append the live wrench. Off = 7-D legacy state.
        pushin_weight: Peak per-frame loss weight for the final push-in phase
            (``1.0`` disables weighting = uniform L1, the legacy path).
        pushin_ramp_s: Seconds of frames over which the loss weight ramps from
            ``1.0`` up to ``pushin_weight`` before each (trimmed) episode end.
        dt_frame: Recording frame period (s) used to convert ``tail_trim_margin_s``
            / ``pushin_ramp_s`` to frame counts. Matches DeployACT's ``DT_FRAME``
            (~3.64 Hz, ds_wide median 0.275 s).
        port_aux: Train the port-bearing auxiliary head (predict the TCP->port
            offset) jointly with the action head. Off = legacy single-head path,
            byte-identical checkpoint (no aux keys). See
            ``docs/design_port_aux_head.md``.
        aux_dim: Auxiliary-head output width: 3 (offset) or 6 (offset + axis).
        aux_weight: Weight on the masked aux L1 term in the total loss
            (``loss = action_L1 + aux_weight * aux_L1``). Must be >= 0.
        aux_frame: Frame the offset labels/predictions live in: ``'tcp'`` (lower
            variance; deploy rotates once by the live TCP quaternion) or
            ``'base'`` (predicted directly in base_link).
        aux_freeze_encoder: Frozen-encoder probe -- freeze the encoder AND action
            head and train only the aux head (measures port bearing already in
            the features; ships an aux head without touching deployed action
            weights). Requires ``init_ckpt`` to load a pretrained encoder.
        aux_label_glob: Explicit ``campaign_log.csv`` locator for the labels;
            empty auto-derives ``<ds>/campaign_log.csv`` beside each episode.
        init_ckpt: Optional checkpoint to initialize weights from before training
            (encoder + action head; aux head stays freshly initialized). Loaded
            non-strict. Required for a meaningful ``aux_freeze_encoder`` probe.
        near_port_m: TCP->port distance (m) below which a valid frame counts as
            "near-port" for the ``val_offset_nearport_cm`` operating-point metric.
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
    tail_trim: bool = False
    tail_trim_threshold: float = 0.003
    tail_trim_margin_s: float = 0.3
    last_inch_s: float = 0.0
    last_inch_min_frames: int = 8
    use_wrench: bool = False
    pushin_weight: float = 1.0
    pushin_ramp_s: float = 2.0
    dt_frame: float = 0.275
    port_aux: bool = False
    aux_dim: int = 3
    aux_weight: float = 0.5
    aux_frame: str = "tcp"
    aux_freeze_encoder: bool = False
    aux_label_glob: str = ""
    init_ckpt: str = ""
    near_port_m: float = 0.03

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
        if self.tail_trim_threshold < 0.0:
            raise ValueError(
                f"tail_trim_threshold must be >= 0, got {self.tail_trim_threshold}"
            )
        if self.tail_trim_margin_s < 0.0:
            raise ValueError(
                f"tail_trim_margin_s must be >= 0, got {self.tail_trim_margin_s}"
            )
        if self.last_inch_s < 0.0:
            raise ValueError(f"last_inch_s must be >= 0, got {self.last_inch_s}")
        if self.last_inch_min_frames < 1:
            raise ValueError(
                f"last_inch_min_frames must be >= 1, got {self.last_inch_min_frames}"
            )
        if self.last_inch_s > 0.0 and self.tail_trim:
            raise ValueError(
                "last_inch_s and tail_trim are mutually exclusive (last-inch keeps "
                "the terminal window; tail_trim drops it)"
            )
        if self.pushin_weight < 1.0:
            raise ValueError(f"pushin_weight must be >= 1, got {self.pushin_weight}")
        if self.pushin_ramp_s < 0.0:
            raise ValueError(f"pushin_ramp_s must be >= 0, got {self.pushin_ramp_s}")
        if self.dt_frame <= 0.0:
            raise ValueError(f"dt_frame must be > 0, got {self.dt_frame}")
        if self.aux_dim not in (3, 6):
            raise ValueError(f"aux_dim must be 3 or 6, got {self.aux_dim}")
        if self.aux_weight < 0.0:
            raise ValueError(f"aux_weight must be >= 0, got {self.aux_weight}")
        if self.aux_frame not in ("tcp", "base"):
            raise ValueError(f"aux_frame must be 'tcp' or 'base', got {self.aux_frame!r}")
        if self.near_port_m <= 0.0:
            raise ValueError(f"near_port_m must be > 0, got {self.near_port_m}")

    @property
    def ema_enabled(self) -> bool:
        """Whether EMA-of-weights evaluation is active."""
        return self.ema_decay > 0.0

    @property
    def state_dim(self) -> int:
        """Proprioceptive state width: 7 (TCP pose) or 13 (pose + 6-D wrench)."""
        return 13 if self.use_wrench else 7

    @property
    def pushin_enabled(self) -> bool:
        """Whether push-in loss weighting is active (peak weight > 1)."""
        return self.pushin_weight > 1.0

    @property
    def last_inch_enabled(self) -> bool:
        """Whether last-inch terminal-window selection is active (lookback > 0)."""
        return self.last_inch_s > 0.0

    @property
    def port_aux_enabled(self) -> bool:
        """Whether the port-bearing auxiliary head is trained this run."""
        return self.port_aux


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
    # Port-bearing auxiliary-head validation metrics (cm), populated only when
    # ``port_aux`` is on; defaults keep legacy single-head callers working.
    # ``val_offset_cm`` is the median Euclidean TCP->port error over valid val
    # frames; ``nearport`` restricts to the near-port operating point (design
    # section 2.5 adoption gate < 2 cm); ``lateral`` is the perpendicular error
    # (the component that clips distractors). ``best`` is the min over evals.
    best_val_offset_cm: float = float("inf")
    final_val_offset_cm: float = 0.0
    final_val_offset_nearport_cm: float = 0.0
    final_val_offset_lateral_cm: float = 0.0


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
