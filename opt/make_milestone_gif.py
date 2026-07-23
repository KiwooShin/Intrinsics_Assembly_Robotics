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


def _tally_card(w: int, h: int, eyebrow: str, big: str, lines: list[str]) -> Image.Image:
    card = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(card)
    d.rectangle([0, h - 5, w, h], fill=SEAT)
    d.text((PAD + 6, h * 0.20), eyebrow.upper(), font=_font("DejaVuSansMono-Bold.ttf", 13), fill=SEAT)
    d.text((PAD + 6, h * 0.20 + 24), big, font=_font("DejaVuSans-Bold.ttf", 40), fill=INK)
    ty = h * 0.20 + 74
    for line in lines:
        for wrapped in wrap_text(line, _font("DejaVuSans.ttf", 14), w - 2 * (PAD + 6)):
            d.text((PAD + 6, ty), wrapped, font=_font("DejaVuSans.ttf", 14), fill=MUTED)
            ty += 20
    return card


def build_montage_gif(clips: list[dict], out_path: str, *, eyebrow: str, title: str,
                      subtitle: str, tally_big: str, tally_lines: list[str],
                      duration_ms: int = 75) -> str:
    """Compose several labeled rollout clips + a tally card into one montage GIF.

    Each clip is ``{"frames": [PIL RGB...], "title": str, "caption": str,
    "badge": str, "badge_color": tuple}`` and is annotated with the shared ``eyebrow``
    plus its own per-clip title/caption/badge, then concatenated. A title card leads and
    a tally card closes. Used for the "N random locations" robustness demo.

    Raises:
        ValueError: If ``clips`` is empty or a clip has no frames.
    """
    if not clips:
        raise ValueError("clips must be non-empty")
    seq: list[Image.Image] = []
    w = h = None
    for c in clips:
        if not c.get("frames"):
            raise ValueError("each clip needs non-empty frames")
        for f in c["frames"]:
            a = _annotate(f.convert("RGB"), eyebrow, c["title"], c["caption"],
                          c.get("badge", ""), c.get("badge_color", SEAT))
            if w is None:
                w, h = a.size
            seq.append(a)
    card = _title_card(w, h, eyebrow, title, subtitle)
    tally = _tally_card(w, h, eyebrow, tally_big, tally_lines)
    full = [card] * 14 + seq + [tally] * 22
    durs = [110] * 14 + [duration_ms] * len(seq) + [200] * 22
    full = [im.quantize(colors=128, method=Image.FASTOCTREE, dither=Image.Dither.NONE) for im in full]
    full[0].save(out_path, save_all=True, append_images=full[1:], duration=durs,
                 loop=0, optimize=True, disposal=2)
    return out_path


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


def select_rollout_window(n_frames: int, insertion_frame: int, *, pre: int = 90,
                          post: int = 14, target: int = 46) -> list[int]:
    """Pick a subsampled frame-index window around the seating moment.

    Trims a long rollout (mostly a static approach) down to the descent-and-seat
    action: a window ending shortly after ``insertion_frame``, uniformly subsampled to
    at most ``target`` frames so the resulting GIF is punchy rather than 500 frames long.

    Args:
        n_frames: Total frames available in the episode (> 0).
        insertion_frame: Frame index where seating occurs; values ``<= 0`` (no
            recorded seat) fall back to a window ending at the last frame.
        pre: Frames to include before the seat.
        post: Frames to include after the seat.
        target: Maximum number of frames to return.

    Returns:
        A sorted list of frame indices into ``[0, n_frames)``.

    Raises:
        ValueError: If ``n_frames <= 0`` or ``target <= 0``.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    if target <= 0:
        raise ValueError(f"target must be > 0, got {target}")
    seat = insertion_frame if 0 < insertion_frame < n_frames else n_frames - 1
    start = max(0, seat - pre)
    end = min(n_frames, seat + post + 1)
    idxs = list(range(start, end))
    if len(idxs) > target:
        step = len(idxs) / target
        idxs = [idxs[min(len(idxs) - 1, int(i * step))] for i in range(target)]
    return idxs


def build_duo_gif(left_frames: list[Image.Image], right_frames: list[Image.Image],
                  out_path: str, *, eyebrow: str, title: str, left_label: str,
                  right_label: str, caption: str, left_badge: str = "",
                  right_badge: str = "", left_badge_color: tuple[int, int, int] = SEAT,
                  right_badge_color: tuple[int, int, int] = SEAT, subtitle: str = "",
                  panel_w: int = 320, gap: int = 8, duration_ms: int = 80,
                  title_card_frames: int = 16) -> str:
    """Compose two rollouts side by side into one 'both together' demo GIF.

    Plays a left rollout (e.g. an SFP fiber-plug insertion) and a right rollout (e.g. an
    SC bayonet insertion into the rotated port) simultaneously under a shared header and
    footer, each panel carrying its own connector label and result badge. The shorter
    clip is padded by holding its final (seated) frame so both end together.

    Args:
        left_frames: Left rollout frames (RGB ``PIL.Image``), non-empty.
        right_frames: Right rollout frames (RGB ``PIL.Image``), non-empty.
        out_path: Destination ``.gif`` path.
        eyebrow: Small uppercase kicker.
        title: Headline shown on the title card and header bar.
        left_label / right_label: Per-panel connector labels.
        caption: One- to three-line explanation shown in the footer.
        left_badge / right_badge: Per-panel result pills ("" hides).
        left_badge_color / right_badge_color: Badge fills (SEAT / MISS / ACCENT).
        subtitle: Optional second line on the title card.
        panel_w: Per-panel width in pixels.
        gap: Gap between the two panels.
        duration_ms: Per-frame duration.
        title_card_frames: How many frames to hold the title card.

    Returns:
        ``out_path``.

    Raises:
        ValueError: If either frame list is empty.
    """
    if not left_frames or not right_frames:
        raise ValueError("both left_frames and right_frames must be non-empty")
    n = max(len(left_frames), len(right_frames))

    def _pad(frs: list[Image.Image]) -> list[Image.Image]:
        return list(frs) + [frs[-1]] * (n - len(frs))

    lf, rf = _pad(left_frames), _pad(right_frames)

    def _panel(img: Image.Image) -> Image.Image:
        w0, h0 = img.width, img.height
        return img.resize((panel_w, round(panel_w * h0 / w0)), Image.BILINEAR)

    ph = _panel(lf[0].convert("RGB")).height
    label_h = 26
    w = panel_w * 2 + gap
    h = HEADER_H + label_h + ph + FOOTER_H
    eb_font = _font("DejaVuSansMono-Bold.ttf", 11)
    ti_font = _font("DejaVuSans-Bold.ttf", 19)
    lab_font = _font("DejaVuSans-Bold.ttf", 13)
    cap_font = _font("DejaVuSans.ttf", 12)
    seq: list[Image.Image] = []
    for i in range(n):
        canvas = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 10), eyebrow.upper(), font=eb_font, fill=ACCENT)
        d.text((PAD, 26), title, font=ti_font, fill=INK)
        ly = HEADER_H
        d.rectangle([0, ly, w, ly + label_h], fill=PANEL)
        d.text((PAD, ly + 6), left_label, font=lab_font, fill=ACCENT)
        d.text((panel_w + gap + PAD, ly + 6), right_label, font=lab_font, fill=ACCENT)
        py = ly + label_h
        canvas.paste(_panel(lf[i].convert("RGB")), (0, py))
        canvas.paste(_panel(rf[i].convert("RGB")), (panel_w + gap, py))
        if left_badge:
            _badge(d, panel_w - 8, py + ph - 18, left_badge, left_badge_color)
        if right_badge:
            _badge(d, w - 8, py + ph - 18, right_badge, right_badge_color)
        fy = py + ph
        d.rectangle([0, fy, w, h], fill=PANEL)
        for j, line in enumerate(wrap_text(caption, cap_font, w - 2 * PAD)[:3]):
            d.text((PAD, fy + 12 + j * 17), line, font=cap_font, fill=MUTED)
        seq.append(canvas)
    card = _title_card(w, h, eyebrow, title, subtitle)
    full = [card] * title_card_frames + seq + [seq[-1]] * 10
    full = [im.quantize(colors=128, method=Image.FASTOCTREE, dither=Image.Dither.NONE)
            for im in full]
    durs = [110] * title_card_frames + [duration_ms] * len(seq) + [220] * 10
    full[0].save(out_path, save_all=True, append_images=full[1:], duration=durs,
                 loop=0, optimize=True, disposal=2)
    return out_path
