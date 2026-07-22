"""Unit tests for opt.make_milestone_gif pure logic (no ROS/GPU; PIL only)."""
from __future__ import annotations

import unittest

from PIL import ImageFont

from opt.make_milestone_gif import FONT_DIR, wrap_text


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
