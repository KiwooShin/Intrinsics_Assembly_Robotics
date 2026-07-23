"""Unit tests for opt.make_milestone_gif pure logic (no ROS/GPU; PIL only)."""
from __future__ import annotations

import unittest

import datetime
import os
import tempfile

from PIL import Image, ImageFont

from opt.make_milestone_gif import (
    FONT_DIR,
    build_duo_gif,
    milestone_gif_path,
    select_rollout_window,
    wrap_text,
)


class MilestoneGifPathTest(unittest.TestCase):
    def test_carries_number_date_and_hour(self) -> None:
        when = datetime.datetime(2026, 7, 22, 8, 53, 0)
        p = milestone_gif_path("docs/media", 1, "first_seat", when)
        self.assertEqual(p, "docs/media/milestone1_first_seat_2026-07-22_08h.gif")

    def test_spaces_become_underscores_and_trailing_slash_ok(self) -> None:
        when = datetime.datetime(2026, 7, 22, 9, 0, 0)
        p = milestone_gif_path("docs/media/", 2, "offset 2mm", when)
        self.assertEqual(p, "docs/media/milestone2_offset_2mm_2026-07-22_09h.gif")

    def test_rejects_bad_number_or_empty_slug(self) -> None:
        with self.assertRaises(ValueError):
            milestone_gif_path("d", 0, "x")
        with self.assertRaises(ValueError):
            milestone_gif_path("d", 1, "   ")


class WrapTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.font = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 12)

    def test_short_text_is_one_line(self) -> None:
        self.assertEqual(wrap_text("hello world", self.font, 1000), ["hello world"])

    def test_wraps_to_multiple_lines(self) -> None:
        text = "the quick brown fox jumps over the lazy dog several times over"
        lines = wrap_text(text, self.font, 80)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(self.font.getlength(line), 80 + self.font.getlength(" x"))

    def test_reconstructs_all_words_in_order(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta"
        self.assertEqual(" ".join(wrap_text(text, self.font, 60)).split(), text.split())

    def test_overlong_word_kept_on_own_line(self) -> None:
        lines = wrap_text("supercalifragilisticexpialidocious ok", self.font, 20)
        self.assertEqual(lines[0], "supercalifragilisticexpialidocious")

    def test_bad_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            wrap_text("x", self.font, 0)


class SelectRolloutWindowTest(unittest.TestCase):
    def test_indices_stay_in_range_and_sorted(self) -> None:
        idxs = select_rollout_window(520, 480)
        self.assertTrue(all(0 <= i < 520 for i in idxs))
        self.assertEqual(idxs, sorted(idxs))

    def test_caps_at_target(self) -> None:
        idxs = select_rollout_window(520, 480, pre=200, post=40, target=46)
        self.assertLessEqual(len(idxs), 46)

    def test_window_brackets_the_seat(self) -> None:
        idxs = select_rollout_window(300, 200, pre=50, post=10, target=200)
        self.assertGreaterEqual(idxs[0], 150)
        self.assertLessEqual(idxs[-1], 210)

    def test_no_recorded_seat_falls_back_to_end(self) -> None:
        idxs = select_rollout_window(100, 0, pre=30, post=10, target=200)
        self.assertEqual(idxs[-1], 99)

    def test_rejects_bad_sizes(self) -> None:
        with self.assertRaises(ValueError):
            select_rollout_window(0, 5)
        with self.assertRaises(ValueError):
            select_rollout_window(100, 5, target=0)


class BuildDuoGifTest(unittest.TestCase):
    def _frames(self, n: int, color: tuple[int, int, int]) -> list[Image.Image]:
        # Distinct per-frame content (a moving bright pixel) so GIF frame-dedup
        # optimization doesn't collapse identical frames, mirroring real rollouts.
        frames = []
        for i in range(n):
            im = Image.new("RGB", (48, 40), color)
            im.putpixel((i % 48, i % 40), (255, 255, 255))
            frames.append(im)
        return frames

    def test_writes_multiframe_gif_and_pads_shorter_clip(self) -> None:
        left = self._frames(6, (40, 90, 90))   # shorter
        right = self._frames(10, (90, 40, 40))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "duo.gif")
            ret = build_duo_gif(left, right, out, eyebrow="e", title="t",
                                left_label="SFP", right_label="SC", caption="c",
                                left_badge="SEATED", right_badge="SEATED")
            self.assertEqual(ret, out)
            self.assertTrue(os.path.exists(out))
            with Image.open(out) as im:
                # title-card frames + max(len) rollout frames + hold, all present
                self.assertGreater(getattr(im, "n_frames", 1), 10)

    def test_rejects_empty_frames(self) -> None:
        with self.assertRaises(ValueError):
            build_duo_gif([], self._frames(3, (1, 2, 3)), "x.gif", eyebrow="e",
                          title="t", left_label="a", right_label="b", caption="c")


if __name__ == "__main__":
    unittest.main()
