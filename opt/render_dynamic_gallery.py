"""Render a tight gallery of exciting, DYNAMIC insertion demos in the liked style.

The earlier per-milestone gallery leaned on aligned / tiny-offset / straight-down scenes
where almost nothing moves. This renders the *dynamic* footage instead — big lateral
offsets and high start-drops where the plug visibly travels and makes a large correction —
in the format the project's favourite GIFs use: 232-px 3-camera panels (704 px wide), a
long ~72-frame window so motion reads fluidly, 75 ms/frame, and a caption carrying the
exact scenario and engine score. Failures are shown honestly with a red badge.

Sources (footage kept regardless of seat):
* ``/tmp/faildemos5.manifest`` — edge offsets 1.2/2.0/3.0 mm + a hard SC pose
  (``~/training/ds_faildemos``).
* ``/tmp/dyndemos5.manifest`` — big offsets from 25-40 mm drops
  (``~/training/ds_dyndemos``).
* ``~/training/ds_demos/m4`` — SC angled insertions (scores in results/{demos5,topup5}).

Pure numpy/PIL, no ROS/GPU. Run with ``PYTHONPATH=. python opt/render_dynamic_gallery.py``.
"""
from __future__ import annotations

import dataclasses
import datetime
import pathlib
import re

import numpy as np
from PIL import Image

from opt.make_milestone_gif import (
    build_milestone_gif,
    montage_lcr,
    select_rollout_window,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
HOME = pathlib.Path.home()
MEDIA = REPO / "docs" / "media"
SEAT = (67, 196, 131)
MISS = (236, 122, 97)

PANEL_W = 232       # liked style: 3 x 232 = 704 px wide
PRE, POST, TARGET = 230, 18, 72   # long window so the descent + correction is fluid


@dataclasses.dataclass(frozen=True)
class Clip:
    """One dynamic rollout to render."""

    epdir: pathlib.Path
    kind: str            # 'sfp' | 'sc'
    offset_mm: float
    azimuth: float
    standoff_mm: float
    total: str
    seated: bool
    note: str


def _round(total: str) -> str:
    try:
        return f"{float(total):.0f}"
    except ValueError:
        return ""


def _load_fail() -> list[Clip]:
    man = pathlib.Path("/tmp/faildemos5.manifest")
    out: list[Clip] = []
    if not man.exists():
        return out
    for line in man.read_text().splitlines():
        p = line.split("|")
        if len(p) != 7:
            continue
        label, ms, off, az, total, seat, note = p
        ep = HOME / "training" / "ds_faildemos" / f"ep_{label}"
        if not (ep / "center_images.npy").exists():
            continue
        out.append(Clip(ep, "sc" if "sc" in label else "sfp", float(off), float(az),
                        20.0, _round(total), seat.strip() == "SEAT", note.strip()))
    return out


def _load_dyn() -> list[Clip]:
    man = pathlib.Path("/tmp/dyndemos5.manifest")
    out: list[Clip] = []
    if not man.exists():
        return out
    for line in man.read_text().splitlines():
        p = line.split("|")
        if len(p) != 7:
            continue
        label, off, az, sto, total, seat, note = p
        ep = HOME / "training" / "ds_dyndemos" / f"ep_{label}"
        if not (ep / "center_images.npy").exists():
            continue
        out.append(Clip(ep, "sfp", float(off), float(az), float(sto) * 1000.0,
                        _round(total), seat.strip() == "SEAT", note.strip()))
    return out


def _load_sc_from_demos(limit: int = 2) -> list[Clip]:
    """A couple of SC angled insertions from the main capture, scored from results/."""
    out: list[Clip] = []
    d = HOME / "training" / "ds_demos" / "m4"
    for ep in sorted(d.glob("ep_*"))[:limit] if d.is_dir() else []:
        label = ep.name[len("ep_"):]
        total, seated = "", False
        for base in ("demos5", "topup5"):
            sp = REPO / "results" / base / f"m4_{label}" / "scoring.yaml"
            if sp.exists():
                t = sp.read_text()
                m = re.search(r"^total:\s*([0-9.]+)", t, re.MULTILINE)
                total = _round(m.group(1)) if m else ""
                seated = "Cable insertion successful" in t
                break
        out.append(Clip(ep, "sc", 0.0, 0.0, 20.0, total, seated, "SC rotated port"))
    return out


def _frames(clip: Clip) -> list[Image.Image]:
    left = np.load(clip.epdir / "left_images.npy", mmap_mode="r")
    center = np.load(clip.epdir / "center_images.npy", mmap_mode="r")
    right = np.load(clip.epdir / "right_images.npy", mmap_mode="r")
    n = int(center.shape[0])
    seat_p = clip.epdir / "insertion_frame.npy"
    seat = int(np.load(seat_p)) if (clip.seated and seat_p.exists()) else -1
    if seat > 0:
        idxs = select_rollout_window(n, seat, pre=PRE, post=POST, target=TARGET)
    else:  # miss: span the whole attempt so the drift is visible
        idxs = [int(round(x)) for x in np.linspace(0, n - 1, TARGET)]
    return [montage_lcr(np.asarray(left[i]), np.asarray(center[i]), np.asarray(right[i]),
                        panel_w=PANEL_W) for i in idxs]


def _sfp_caption(c: Clip) -> str:
    drop = f", staged {c.standoff_mm:.0f} mm above the port" if c.standoff_mm > 21 else ""
    lead = f"Plug commanded {c.offset_mm:.1f} mm off-center{drop}. "
    if c.seated:
        return lead + ("The camera-only policy corrects the offset during descent and "
                       "seats the plug — no ground-truth pose at run time.")
    return lead + ("The plug drifts past the port before it can feel it, so it never "
                   "seats — an honest failure, shown not hidden.")


def render_clip(c: Clip, idx: int, when: datetime.datetime) -> str:
    frames = _frames(c)
    badge = ((f"SEATED · {c.total}/100" if c.total else "SEATED") if c.seated
             else (f"NOT SEATED · {c.total}/100" if c.total else "NOT SEATED"))
    stamp = when.strftime("%Y-%m-%d_%H")
    if c.kind == "sc":
        path = MEDIA / f"dynamic_sc_angled_{c.epdir.name[len('ep_'):]}_{stamp}h.gif"
        eyebrow, title = "Dynamic demo · SC angled insertion", "Angled insertion into a rotated port"
        caption = ("The SC/LC port is physically ROTATED, so a top-down descent rams it. A "
                   "separate sc_insert model inserts along the pose-conditioned axis — a long, "
                   f"visibly tilted descent. Engine {c.total or '?'}/100.")
    else:
        path = MEDIA / f"dynamic_offset_{c.epdir.name[len('ep_'):]}_{stamp}h.gif"
        eyebrow = "Dynamic demo · large-offset recovery"
        title = (f"Correcting a {c.offset_mm:.1f} mm offset" if c.seated
                 else f"When a {c.offset_mm:.1f} mm offset drifts past")
        caption = _sfp_caption(c)
    build_milestone_gif(
        frames, str(path), eyebrow=eyebrow, title=title, caption=caption,
        badge=badge, badge_color=SEAT if c.seated else MISS,
        subtitle="Left / center / right cameras · camera-only policy", duration_ms=75)
    tag = "SEAT" if c.seated else "MISS"
    print(f"[dyn] {path.name}  ({tag} {c.total or '?'}/100, off {c.offset_mm}mm, "
          f"standoff {c.standoff_mm:.0f}mm, {len(frames)} frames)")
    return str(path)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    clips = _load_dyn() + _load_fail() + _load_sc_from_demos()
    when = datetime.datetime.now()
    print(f"[dyn] rendering {len(clips)} dynamic clips")
    for i, c in enumerate(clips):
        render_clip(c, i, when)


if __name__ == "__main__":
    main()
