"""Render a 'fancy' explained milestone demo GIF: title card + annotated rollout.

Standing directive (2026-07-22): every milestone gets a demo video with an on-screen
explanation, converted to GIF and put in the README. Given rollout frames (or an
existing GIF) plus milestone metadata, this produces a captioned GIF with a title card,
a header bar (eyebrow + title + result badge), and a footer explanation strip, in the
same dark/teal theme as the project showcase. Pure `wrap_text` is unit-tested; the
rendering needs only PIL (no ROS/GPU).
"""
from __future__ import annotations

import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Theme — matches dashboard/showcase.html.
BG = (12, 16, 21)
PANEL = (20, 27, 35)
INK = (233, 238, 245)
MUTED = (148, 161, 177)
ACCENT = (49, 207, 192)
SEAT = (67, 196, 131)
MISS = (236, 122, 97)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

HEADER_H = 54
FOOTER_H = 78
PAD = 16


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONT_DIR}/{name}", size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy word-wrap ``text`` into lines that each fit within ``max_w`` pixels.

    Args:
        text: The string to wrap.
        font: A PIL font exposing ``getlength``.
        max_w: Maximum line width in pixels (> 0).

    Returns:
        A list of line strings; a single over-long word is kept on its own line.

    Raises:
        ValueError: If ``max_w <= 0``.
    """
    if max_w <= 0:
        raise ValueError(f"max_w must be > 0, got {max_w}")
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if not cur or font.getlength(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _badge(draw: ImageDraw.ImageDraw, xy_right: float, cy: float, text: str,
           color: tuple[int, int, int]) -> None:
    bf = _font("DejaVuSans-Bold.ttf", 13)
    w = bf.getlength(text) + 22
    x = xy_right - w
    draw.rounded_rectangle([x, cy - 14, x + w, cy + 14], radius=14, fill=color)
    draw.text((x + 11, cy - 8), text, font=bf, fill=BG)


def _annotate(frame: Image.Image, eyebrow: str, title: str, caption: str,
              badge: str, badge_color: tuple[int, int, int]) -> Image.Image:
    w = frame.width
    h = HEADER_H + frame.height + FOOTER_H
    canvas = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(canvas)
    # header
    d.text((PAD, 10), eyebrow.upper(), font=_font("DejaVuSansMono-Bold.ttf", 11), fill=ACCENT)
    d.text((PAD, 26), title, font=_font("DejaVuSans-Bold.ttf", 19), fill=INK)
    if badge:
        _badge(d, w - PAD, HEADER_H / 2, badge, badge_color)
    # body
    canvas.paste(frame, (0, HEADER_H))
    # footer
    fy = HEADER_H + frame.height
    d.rectangle([0, fy, w, h], fill=PANEL)
    cap_font = _font("DejaVuSans.ttf", 12)
    for i, line in enumerate(wrap_text(caption, cap_font, w - 2 * PAD)[:3]):
        d.text((PAD, fy + 12 + i * 17), line, font=cap_font, fill=MUTED)
    return canvas


def _title_card(w: int, h: int, eyebrow: str, title: str, subtitle: str) -> Image.Image:
    card = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(card)
    d.rectangle([0, h - 5, w, h], fill=ACCENT)  # accent underline
    eb = _font("DejaVuSansMono-Bold.ttf", 13)
    tt = _font("DejaVuSans-Bold.ttf", 26)
    st = _font("DejaVuSans.ttf", 14)
    d.text((PAD + 6, h * 0.30), eyebrow.upper(), font=eb, fill=ACCENT)
    ty = h * 0.30 + 24
    for line in wrap_text(title, tt, w - 2 * (PAD + 6)):
        d.text((PAD + 6, ty), line, font=tt, fill=INK)
        ty += 32
    if subtitle:
        ty += 4
        for line in wrap_text(subtitle, st, w - 2 * (PAD + 6)):
            d.text((PAD + 6, ty), line, font=st, fill=MUTED)
            ty += 19
    return card


def build_milestone_gif(frames: list[Image.Image], out_path: str, *, eyebrow: str,
                        title: str, caption: str, badge: str = "",
                        badge_color: tuple[int, int, int] = SEAT, subtitle: str = "",
                        title_card_frames: int = 14, duration_ms: int = 80) -> str:
    """Compose a title card + annotated rollout into an explained milestone GIF.

    Args:
        frames: Rollout frames (RGB ``PIL.Image``), all the same size.
        out_path: Destination ``.gif`` path.
        eyebrow: Small uppercase kicker (e.g. "Milestone 1 · aligned insertion").
        title: Headline shown on the title card and header bar.
        caption: One- to three-line explanation shown in the footer.
        badge: Short result pill (e.g. "SEATED · 93/100"); "" hides it.
        badge_color: Badge fill (SEAT / MISS / ACCENT).
        subtitle: Optional second line on the title card.
        title_card_frames: How many frames to hold the title card.
        duration_ms: Per-frame duration.

    Returns:
        ``out_path``.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    annotated = [_annotate(f.convert("RGB"), eyebrow, title, caption, badge, badge_color)
                 for f in frames]
    w, h = annotated[0].size
    card = _title_card(w, h, eyebrow, title, subtitle)
    seq = [card] * title_card_frames + annotated + [annotated[-1]] * 8  # hold last
    seq = [im.quantize(colors=128, method=Image.FASTOCTREE, dither=Image.Dither.NONE)
           for im in seq]
    durs = [duration_ms] * len(seq)
    durs[:title_card_frames] = [110] * title_card_frames  # linger on title
    durs[-8:] = [220] * 8                                  # linger on result
    seq[0].save(out_path, save_all=True, append_images=seq[1:], duration=durs,
                loop=0, optimize=True, disposal=2)
    return out_path


def milestone_gif_path(dirpath: str, number: int, slug: str,
                       when: datetime.datetime | None = None) -> str:
    """Build the standard milestone-GIF path with milestone number, date, and hour.

    Naming convention (user directive 2026-07-22): every milestone demo filename carries
    the milestone NUMBER, the DATE, and the HOUR, e.g.
    ``docs/media/milestone1_first_seat_2026-07-22_08h.gif``.

    Args:
        dirpath: Output directory (e.g. ``"docs/media"``).
        number: Milestone number (1-based).
        slug: Short kebab/snake description (spaces become ``_``).
        when: Timestamp to stamp; defaults to now.

    Returns:
        The full path ``<dir>/milestone<N>_<slug>_<YYYY-MM-DD>_<HH>h.gif``.

    Raises:
        ValueError: If ``number < 1`` or ``slug`` is empty.
    """
    if number < 1:
        raise ValueError(f"milestone number must be >= 1, got {number}")
    slug = slug.strip().replace(" ", "_")
    if not slug:
        raise ValueError("slug must be non-empty")
    when = when or datetime.datetime.now()
    return f"{dirpath.rstrip('/')}/milestone{number}_{slug}_{when:%Y-%m-%d_%H}h.gif"


def frames_from_gif(path: str) -> list[Image.Image]:
    """Load all frames of an existing GIF as RGB images."""
    im = Image.open(path)
    out = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        out.append(im.convert("RGB"))
    return out


def montage_lcr(left: np.ndarray, center: np.ndarray, right: np.ndarray,
                panel_w: int = 232, gap: int = 4) -> Image.Image:
    """Build a left|center|right 3-camera montage frame from H×W×3 uint8 arrays."""
    h, w = center.shape[:2]
    ph = round(panel_w * h / w)
    panels = [Image.fromarray(a).resize((panel_w, ph), Image.BILINEAR)
              for a in (left, center, right)]
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, ph), (17, 20, 26))
    for k, p in enumerate(panels):
        canvas.paste(p, (k * (panel_w + gap), 0))
    return canvas
