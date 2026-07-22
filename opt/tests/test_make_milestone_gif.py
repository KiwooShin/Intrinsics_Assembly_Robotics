"""Unit tests for opt.make_milestone_gif pure logic (no ROS/GPU; PIL only)."""
from __future__ import annotations

import unittest

import datetime

from PIL import ImageFont

from opt.make_milestone_gif import FONT_DIR, milestone_gif_path, wrap_text


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


if __name__ == "__main__":
    unittest.main()
