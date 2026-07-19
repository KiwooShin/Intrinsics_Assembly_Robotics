"""Hindsight terminal-TCP port-offset labels for the ``port_aux`` head.

Builds the per-frame regression targets the auxiliary head trains on, purely
from the on-disk ``tcp_poses.npy`` and the collection ``campaign_log.csv`` -- no
raw bags, no privileged port pose (see ``docs/design_port_aux_head.md`` sections
1-3). For each episode:

  1. Join the episode directory to its ``campaign_log.csv`` row and keep it as a
     *valid* label source only when ``status == KEEP`` and
     ``insertion_events >= 1`` (a demo that never inserted never reached the
     port, so it carries no usable target).
  2. Take the robust terminal TCP pose (median of the last ``M`` frames'
     position + the final quaternion) of the *full, untrimmed* episode as the
     static base_link target.
  3. Emit a per-frame offset label (TCP-frame or base-frame) from every frame's
     TCP to that target -- ~0 at the seated terminal, growing along the
     approach.

Labels are computed **before** any tail-trim so the seated tail (which *is* the
target) is not hidden; ``opt.train_v3`` applies the same trim keep-mask to the
labels as to the images/state/actions afterward. Normalization statistics are
computed over the *valid* training frames only and stored in the checkpoint so
deploy de-normalizes with the exact training scale.

The heavy geometry is reused from
``aic_example_policies.ros.port_offset`` (the same math ``DeployACT`` uses at
inference), keeping the label frame convention and the deploy frame convention
identical by construction. This module is pure numpy + csv (no torch/ROS/GPU) so
it unit-tests CPU-only.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import sys

import numpy as np

# Reuse the canonical (ROS-free) offset geometry from the deploy package so the
# training label frame and the deploy resolution frame can never diverge. The
# package root is ``<repo>/aic_example_policies`` (mirrors the ros unit-test
# PYTHONPATH and train_v3's repo-root insert).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PKG_ROOT = _REPO_ROOT / "aic_example_policies"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from aic_example_policies.ros import port_offset  # noqa: E402

_LOG = logging.getLogger("opt.port_labels")

# Default trailing-frame counts (mirror the port_offset defaults / design 1.1).
DEFAULT_TERMINAL_FRAMES: int = port_offset.DEFAULT_TERMINAL_FRAMES
DEFAULT_AXIS_FRAMES: int = port_offset.DEFAULT_AXIS_FRAMES
# Small constant added to every per-dim std so normalization never divides by 0
# (mirrors train_v2/train_v3's ``+ 1e-6`` on state/action stats).
NORM_EPS: float = 1e-6
CAMPAIGN_LOG_NAME: str = "campaign_log.csv"
# Campaign status for "the output directory already existed, so collection was
# skipped": such rows re-affirm an existing episode and carry no fresh verdict,
# so they are ignored when reducing duplicate rows to the effective status.
SKIP_STATUS: str = "SKIP_EXISTS"


@dataclasses.dataclass(frozen=True)
class EpisodeStatus:
    """One ``campaign_log.csv`` row's KEEP/insertion verdict for an episode.

    Attributes:
        episode_dir: Episode directory basename (e.g. ``ep_sfp_rail0_..._r0``).
        status: Campaign status string (``KEEP`` / ``DROP_SCORE`` / ...).
        insertion_events: Parsed ``insertion_events`` count, or None if the
            column was blank (non-KEEP rows leave it empty).
        score: Parsed engine score, or None if blank.
    """

    episode_dir: str
    status: str
    insertion_events: int | None
    score: float | None

    @property
    def valid(self) -> bool:
        """Whether this episode is a usable hindsight label source.

        Returns:
            True iff ``status == 'KEEP'`` and at least one insertion event was
            recorded (the demo actually seated the plug in the port).
        """
        return (
            self.status == "KEEP"
            and self.insertion_events is not None
            and self.insertion_events >= 1
        )


@dataclasses.dataclass(frozen=True)
class EpisodeLabel:
    """Per-episode hindsight target and per-frame offset labels.

    Attributes:
        episode_dir: Absolute episode directory the labels were built from.
        n_frames: Number of frames (rows of ``tcp_poses.npy``).
        valid: Whether the episode is a usable label source (KEEP + inserted).
        target_position: Robust terminal target ``[x, y, z]`` in base_link (m).
        target_quaternion: Terminal TCP orientation ``[x, y, z, w]``.
        offsets: Per-frame labels shaped ``(n_frames, aux_dim)`` (offset, plus
            axis when ``aux_dim == 6``), in the requested frame.
    """

    episode_dir: str
    n_frames: int
    valid: bool
    target_position: np.ndarray
    target_quaternion: np.ndarray
    offsets: np.ndarray


@dataclasses.dataclass(frozen=True)
class LabelSet:
    """Concatenated per-frame labels + validity for a set of episodes.

    Attributes:
        offsets: All per-frame labels shaped ``(n_total, aux_dim)`` (float32), in
            the same episode order as the loader concatenates frames.
        valid: Per-frame validity mask shaped ``(n_total,)`` (bool); True only
            where the episode is KEEP + inserted.
        frame_counts: Per-episode frame counts (for alignment checks).
        episodes: The per-episode :class:`EpisodeLabel` records.
    """

    offsets: np.ndarray
    valid: np.ndarray
    frame_counts: tuple[int, ...]
    episodes: tuple[EpisodeLabel, ...]


def parse_campaign_log(csv_path: str | pathlib.Path) -> dict[str, EpisodeStatus]:
    """Parse a ``campaign_log.csv`` into ``episode_dir -> EpisodeStatus``.

    A resumable campaign logs several rows for the same ``episode_dir`` (a
    re-collection appends a fresh row; a resumed run appends a
    :data:`SKIP_STATUS` row that only re-affirms the existing directory). The
    *effective* verdict for an episode is therefore the last row that actually
    (re)wrote the directory -- i.e. the last **non-**:data:`SKIP_STATUS` row, or
    the last row if every row is a skip. This recovers the full KEEP+inserted
    set (44 for ds_phase0, 33 for ds_phase2) that a naive last-row-wins reduction
    would undercount whenever a KEEP is followed by resume skips.

    Args:
        csv_path: Path to a ``campaign_log.csv`` written by the collection
            campaign (columns include ``episode_dir``, ``status``,
            ``insertion_events``, ``score``).

    Returns:
        A dict keyed by the episode-directory basename, holding each episode's
        effective status.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If a required column is missing.
    """
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"campaign log not found: {path}")
    rows_by_ep: dict[str, list[EpisodeStatus]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                episode_dir = row["episode_dir"]
                status = row["status"]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(f"campaign log row missing column {exc}") from exc
            rows_by_ep.setdefault(episode_dir, []).append(
                EpisodeStatus(
                    episode_dir=episode_dir,
                    status=status,
                    insertion_events=_parse_int(row.get("insertion_events")),
                    score=_parse_float(row.get("score")),
                )
            )
    return {ep: _effective_status(rows) for ep, rows in rows_by_ep.items()}


def _effective_status(rows: list[EpisodeStatus]) -> EpisodeStatus:
    """Return the effective status among an episode's duplicate log rows.

    Args:
        rows: All rows for one ``episode_dir``, in file order (non-empty).

    Returns:
        The last non-:data:`SKIP_STATUS` row, or the last row if all are skips.
    """
    non_skip = [r for r in rows if r.status != SKIP_STATUS]
    return non_skip[-1] if non_skip else rows[-1]


def _parse_int(raw: str | None) -> int | None:
    """Parse an int from a possibly-blank CSV cell, returning None when blank."""
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _parse_float(raw: str | None) -> float | None:
    """Parse a float from a possibly-blank CSV cell, returning None when blank."""
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def resolve_campaign_log(episode_dir: str | pathlib.Path) -> pathlib.Path:
    """Return the ``campaign_log.csv`` path sitting beside an episode directory.

    Args:
        episode_dir: An episode directory (e.g. ``.../ds_phase0/ep_..._r0``).

    Returns:
        ``<episode_dir_parent>/campaign_log.csv``.
    """
    return pathlib.Path(episode_dir).resolve().parent / CAMPAIGN_LOG_NAME


def load_status_map(
    episode_dirs: list[str], campaign_log: str = ""
) -> dict[str, EpisodeStatus]:
    """Load and merge the campaign status for a set of episode directories.

    When ``campaign_log`` is empty each episode is joined to the
    ``campaign_log.csv`` beside it (episodes from several datasets are merged);
    otherwise the single explicit CSV is used for all episodes.

    Args:
        episode_dirs: Episode directories to resolve status for.
        campaign_log: Optional explicit ``campaign_log.csv`` path; empty to
            auto-derive one per episode from its parent directory.

    Returns:
        A dict keyed by episode-directory basename.

    Raises:
        FileNotFoundError: If a required ``campaign_log.csv`` does not exist.
    """
    status: dict[str, EpisodeStatus] = {}
    if campaign_log:
        return parse_campaign_log(campaign_log)
    seen_logs: set[pathlib.Path] = set()
    for ep in episode_dirs:
        log_path = resolve_campaign_log(ep)
        if log_path in seen_logs:
            continue
        seen_logs.add(log_path)
        status.update(parse_campaign_log(log_path))
    return status


def load_episode_poses(episode_dir: str | pathlib.Path) -> np.ndarray:
    """Load an episode's ``tcp_poses.npy`` as a float64 ``(n, 7)`` array.

    Args:
        episode_dir: Episode directory containing ``tcp_poses.npy``.

    Returns:
        The pose array ``[x, y, z, qx, qy, qz, qw]`` shaped ``(n, 7)``.

    Raises:
        FileNotFoundError: If ``tcp_poses.npy`` is absent.
        ValueError: If the array is not ``(n, 7)``.
    """
    path = pathlib.Path(episode_dir) / "tcp_poses.npy"
    if not path.exists():
        raise FileNotFoundError(f"tcp_poses.npy not found in {episode_dir}")
    poses = np.load(path).astype(np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"{path}: expected (n, 7) poses, got {poses.shape}")
    return poses


def build_episode_label(
    episode_dir: str,
    status_map: dict[str, EpisodeStatus],
    *,
    aux_dim: int = 3,
    aux_frame: str = "tcp",
    terminal_frames: int = DEFAULT_TERMINAL_FRAMES,
    axis_frames: int = DEFAULT_AXIS_FRAMES,
) -> EpisodeLabel:
    """Build per-frame offset labels for a single episode.

    The target is the robust terminal TCP pose of the *full* episode; the
    per-frame label is the TCP->target offset (in ``aux_frame``), optionally
    augmented with the per-frame approach-axis label when ``aux_dim == 6``. The
    validity flag comes from the campaign status join.

    Args:
        episode_dir: Episode directory (its basename joins to ``status_map``).
        status_map: ``episode_dir_basename -> EpisodeStatus`` from
            :func:`load_status_map`.
        aux_dim: Label width: 3 (offset) or 6 (offset + axis).
        aux_frame: ``'tcp'`` or ``'base'`` -- the label frame.
        terminal_frames: Trailing frames medianed for the target position.
        axis_frames: Trailing displacements averaged for the approach axis
            (only when ``aux_dim == 6``).

    Returns:
        The populated :class:`EpisodeLabel`.

    Raises:
        ValueError: If ``aux_dim`` is not 3 or 6, or ``aux_frame`` is unknown.
        FileNotFoundError: If ``tcp_poses.npy`` is absent.
    """
    if aux_dim not in (3, 6):
        raise ValueError(f"aux_dim must be 3 or 6, got {aux_dim}")
    if aux_frame not in ("tcp", "base"):
        raise ValueError(f"aux_frame must be 'tcp' or 'base', got {aux_frame!r}")
    poses = load_episode_poses(episode_dir)
    n = poses.shape[0]
    basename = pathlib.Path(episode_dir).name
    st = status_map.get(basename)
    if st is None:
        _LOG.warning(
            "episode %s has no campaign_log row; marking labels invalid.", basename
        )
    valid = bool(st.valid) if st is not None else False

    target_pos, target_quat = port_offset.robust_terminal(
        poses, terminal_frames=terminal_frames
    )
    offsets = port_offset.per_frame_tcp_offsets(poses, target_pos, frame=aux_frame)
    if aux_dim == 6:
        axis_base = port_offset.base_approach_axis(poses, axis_frames=axis_frames)
        axis_labels = port_offset.per_frame_axis_labels(
            poses, axis_base, frame=aux_frame
        )
        offsets = np.concatenate([offsets, axis_labels], axis=1)
    return EpisodeLabel(
        episode_dir=str(episode_dir),
        n_frames=n,
        valid=valid,
        target_position=target_pos,
        target_quaternion=target_quat,
        offsets=offsets.astype(np.float32),
    )


def build_labels(
    episode_dirs: list[str],
    *,
    aux_dim: int = 3,
    aux_frame: str = "tcp",
    campaign_log: str = "",
    terminal_frames: int = DEFAULT_TERMINAL_FRAMES,
    axis_frames: int = DEFAULT_AXIS_FRAMES,
) -> LabelSet:
    """Build concatenated per-frame labels + validity for a set of episodes.

    Episodes are processed in the given order and their per-frame labels are
    concatenated in that order, matching ``train_v2.load_all``'s frame
    concatenation so the returned arrays align index-for-index with the loader's
    images/state/actions (before any tail-trim).

    Args:
        episode_dirs: Episode directories, in the loader's order.
        aux_dim: Label width: 3 (offset) or 6 (offset + axis).
        aux_frame: ``'tcp'`` or ``'base'``.
        campaign_log: Optional explicit ``campaign_log.csv`` (empty to auto-derive
            per episode).
        terminal_frames: Trailing frames medianed for the target position.
        axis_frames: Trailing displacements averaged for the approach axis.

    Returns:
        The concatenated :class:`LabelSet`.

    Raises:
        ValueError: If ``episode_dirs`` is empty.
        FileNotFoundError: If a required ``campaign_log.csv`` / ``tcp_poses.npy``
            is absent.
    """
    if not episode_dirs:
        raise ValueError("episode_dirs must be non-empty")
    status_map = load_status_map(episode_dirs, campaign_log)
    episodes: list[EpisodeLabel] = []
    offsets_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    counts: list[int] = []
    for ep in episode_dirs:
        label = build_episode_label(
            ep,
            status_map,
            aux_dim=aux_dim,
            aux_frame=aux_frame,
            terminal_frames=terminal_frames,
            axis_frames=axis_frames,
        )
        episodes.append(label)
        offsets_parts.append(label.offsets)
        valid_parts.append(
            np.full(label.n_frames, label.valid, dtype=bool)
        )
        counts.append(label.n_frames)
    return LabelSet(
        offsets=np.concatenate(offsets_parts, axis=0),
        valid=np.concatenate(valid_parts, axis=0),
        frame_counts=tuple(counts),
        episodes=tuple(episodes),
    )


def normalization_stats(
    offsets: np.ndarray, valid: np.ndarray, eps: float = NORM_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-dim mean/std over the *valid* frames only.

    Normalizing the offset labels puts the aux loss on a ~unit scale, comparable
    to the normalized-twist action L1, so ``aux_weight`` is interpretable
    (design section 3.1). Statistics use only the valid (KEEP + inserted) frames
    -- the invalid frames carry meaningless targets and are masked out of the
    loss.

    Args:
        offsets: Per-frame labels shaped ``(n, aux_dim)``.
        valid: Per-frame validity mask shaped ``(n,)``.
        eps: Constant added to every std to avoid divide-by-zero.

    Returns:
        A tuple ``(omean, ostd)`` of ``(aux_dim,)`` float32 arrays.

    Raises:
        ValueError: If shapes disagree or no frame is valid.
    """
    off = np.asarray(offsets, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if off.ndim != 2:
        raise ValueError(f"offsets must be 2-D (n, aux_dim), got {off.shape}")
    if mask.shape != (off.shape[0],):
        raise ValueError(
            f"valid must have shape ({off.shape[0]},), got {mask.shape}"
        )
    if not mask.any():
        raise ValueError("no valid frames to compute normalization stats")
    sel = off[mask]
    omean = sel.mean(axis=0)
    ostd = sel.std(axis=0) + eps
    return omean.astype(np.float32), ostd.astype(np.float32)
