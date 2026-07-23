"""Render the '≥5 demo videos per milestone' gallery plus a combined SFP+SC demo.

Consumes the seated rollout episodes captured into ``~/training/ds_demos/m{1..4}/ep_*``
(one per scene) and produces, in ``docs/media``:

* up to five individual explained milestone GIFs per milestone
  (``milestone<N>_<slug>_ex<k>_<date>_<HH>h.gif``), and
* one side-by-side "both connectors together" GIF pairing an SFP insertion (Milestone 1)
  with an SC rotated-port insertion (Milestone 4).

Scores are read back from ``results/demos5/<m>_<label>/scoring.yaml``. Pure GIF-composition
logic lives in :mod:`opt.make_milestone_gif`; this module only does the numpy I/O and
wiring, so it needs no ROS/GPU (just numpy + PIL).

Usage:
    python opt/render_demos5.py [--only m1,m4] [--duo-only]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import pathlib
import re

import numpy as np
from PIL import Image

from opt.make_milestone_gif import (
    build_duo_gif,
    build_milestone_gif,
    montage_lcr,
    select_rollout_window,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
DS = pathlib.Path.home() / "training" / "ds_demos"
RESULTS = REPO / "results" / "demos5"
MEDIA = REPO / "docs" / "media"
SEAT = (67, 196, 131)
ACCENT = (49, 207, 192)


@dataclasses.dataclass(frozen=True)
class MilestoneSpec:
    """Presentation metadata for one milestone's demo gallery."""

    number: int
    slug: str
    eyebrow: str
    title: str
    caption: str
    scene_word: str  # e.g. "pose", "location"


SPECS: dict[str, MilestoneSpec] = {
    "m1": MilestoneSpec(
        number=1, slug="aligned_insertion",
        eyebrow="Milestone 1 · aligned insertion",
        title="SFP plug seated from a nominal pose",
        caption=("Learned insertion specialist integrates predicted twist chunks into a "
                 "virtual plug-tip target and drives the SFP fiber plug straight into the "
                 "aligned NIC port."),
        scene_word="pose"),
    "m2": MilestoneSpec(
        number=2, slug="offcenter_recovery",
        eyebrow="Milestone 2 · off-center start",
        title="Recovering from a lateral offset",
        caption=("The plug starts laterally offset from the port. The chunk-integrating "
                 "policy accumulates a corrective x-y bias during descent and still seats "
                 "the connector."),
        scene_word="offset"),
    "m3": MilestoneSpec(
        number=3, slug="ten_random_locations",
        eyebrow="Milestone 3 · robustness",
        title="Seating from a randomized board location",
        caption=("Robustified model trained on a wider pose distribution. Every run starts "
                 "at a different randomized port location on the board — the policy "
                 "generalizes across them."),
        scene_word="location"),
    "m4": MilestoneSpec(
        number=4, slug="sc_rotated_port",
        eyebrow="Milestone 4 · SC rotated port",
        title="SC bayonet into a physically rotated port",
        caption=("A separate SC model with a pose-conditioned insertion axis: the SC port "
                 "is physically rotated, so the approach follows the port's own z-axis "
                 "(not world-z) into the receptacle."),
        scene_word="pose"),
}


def _read_total(label: str, milestone: str) -> str:
    """Return the engine ``total`` score string for a captured trial, or '' if absent."""
    p = RESULTS / f"{milestone}_{label}" / "scoring.yaml"
    if not p.exists():
        return ""
    m = re.search(r"^total:\s*([0-9.]+)", p.read_text(), re.MULTILINE)
    return m.group(1) if m else ""


def _episode_indices(epdir: pathlib.Path, *, pre: int, target: int) -> list[int]:
    n = int(np.load(epdir / "center_images.npy", mmap_mode="r").shape[0])
    seat = int(np.load(epdir / "insertion_frame.npy"))
    return select_rollout_window(n, seat, pre=pre, post=14, target=target)


def _montage_frames(epdir: pathlib.Path, idxs: list[int], panel_w: int) -> list[Image.Image]:
    left = np.load(epdir / "left_images.npy", mmap_mode="r")
    center = np.load(epdir / "center_images.npy", mmap_mode="r")
    right = np.load(epdir / "right_images.npy", mmap_mode="r")
    return [montage_lcr(np.asarray(left[i]), np.asarray(center[i]), np.asarray(right[i]),
                        panel_w=panel_w) for i in idxs]


def _center_frames(epdir: pathlib.Path, idxs: list[int]) -> list[Image.Image]:
    center = np.load(epdir / "center_images.npy", mmap_mode="r")
    return [Image.fromarray(np.asarray(center[i])) for i in idxs]


def _sorted_eps(milestone: str) -> list[pathlib.Path]:
    d = DS / milestone
    if not d.is_dir():
        return []
    eps = [p for p in sorted(d.glob("ep_*")) if (p / "center_images.npy").exists()]
    return eps


def render_milestone(milestone: str, when: datetime.datetime, *, panel_w: int = 200,
                     limit: int = 5) -> list[str]:
    """Render up to ``limit`` individual explained GIFs for one milestone's episodes."""
    spec = SPECS[milestone]
    eps = _sorted_eps(milestone)[:limit]
    out: list[str] = []
    for k, epdir in enumerate(eps, start=1):
        label = epdir.name[len("ep_"):]
        total = _read_total(label, milestone)
        badge = f"SEATED · {total}/100" if total else "SEATED"
        idxs = _episode_indices(epdir, pre=90, target=44)
        frames = _montage_frames(epdir, idxs, panel_w)
        stamp = when.strftime("%Y-%m-%d_%H")
        path = MEDIA / f"milestone{spec.number}_{spec.slug}_ex{k}_{stamp}h.gif"
        build_milestone_gif(
            frames, str(path), eyebrow=spec.eyebrow,
            title=f"{spec.title}",
            caption=spec.caption,
            badge=badge, badge_color=SEAT,
            subtitle=f"Example {k} of {len(eps)} · left · center · right cameras")
        print(f"[render] {path.name}  ({total or '?'}/100, {len(frames)} frames)")
        out.append(str(path))
    return out


def render_duo(when: datetime.datetime) -> str | None:
    """Render the combined SFP (Milestone 1) + SC (Milestone 4) side-by-side demo."""
    sfp_eps, sc_eps = _sorted_eps("m1"), _sorted_eps("m4")
    if not sfp_eps or not sc_eps:
        print("[duo] need at least one seated m1 and one seated m4 episode; skipping")
        return None
    sfp, sc = sfp_eps[0], sc_eps[0]
    sfp_total = _read_total(sfp.name[len("ep_"):], "m1")
    sc_total = _read_total(sc.name[len("ep_"):], "m4")
    lf = _center_frames(sfp, _episode_indices(sfp, pre=88, target=48))
    rf = _center_frames(sc, _episode_indices(sc, pre=120, target=48))
    stamp = when.strftime("%Y-%m-%d_%H")
    path = MEDIA / f"combined_sfp_sc_together_{stamp}h.gif"
    build_duo_gif(
        lf, rf, str(path),
        eyebrow="Combined demo · two connector types",
        title="One system, two connectors",
        left_label="SFP fiber plug · aligned port",
        right_label="SC bayonet · rotated port",
        left_badge=f"SEATED · {sfp_total}/100" if sfp_total else "SEATED",
        right_badge=f"SEATED · {sc_total}/100" if sc_total else "SEATED",
        left_badge_color=SEAT, right_badge_color=SEAT,
        subtitle="Left: SFP insertion (world-z).  Right: SC into a physically rotated port.",
        caption=("Same UR5e, same curriculum framework — two very different connectors. The "
                 "SFP plug drops straight down into an aligned port; the SC plug follows the "
                 "rotated port's own insertion axis. Both seat."),
        panel_w=328)
    print(f"[duo] {path.name}  (SFP {sfp_total or '?'} | SC {sc_total or '?'})")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma list of milestones e.g. m1,m4")
    ap.add_argument("--duo-only", action="store_true", help="render only the combined demo")
    args = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    when = datetime.datetime.now()
    if not args.duo_only:
        which = args.only.split(",") if args.only else list(SPECS)
        for m in which:
            m = m.strip()
            if m in SPECS:
                render_milestone(m, when)
    render_duo(when)


if __name__ == "__main__":
    main()
