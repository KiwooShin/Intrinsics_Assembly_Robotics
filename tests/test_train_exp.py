"""Unit tests for train_exp.py train/val split logic.

The split helpers are torch-free, so these run on any interpreter with stdlib only.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import train_exp


class TestExpandAndSplit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for i in range(5):
            (self.root / f"ep_{i}").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_expand_sorted(self) -> None:
        dirs = train_exp.expand(str(self.root / "ep_*"))
        self.assertEqual(
            [os.path.basename(d) for d in dirs],
            ["ep_0", "ep_1", "ep_2", "ep_3", "ep_4"],
        )

    def test_split_is_disjoint(self) -> None:
        train_dirs, val_dirs = train_exp.split_train_val(
            str(self.root / "ep_*"), str(self.root / "ep_3"))
        self.assertEqual([os.path.basename(d) for d in val_dirs], ["ep_3"])
        self.assertNotIn(str(self.root / "ep_3"), train_dirs)
        self.assertEqual(set(train_dirs) & set(val_dirs), set())
        self.assertEqual(len(train_dirs), 4)

    def test_split_multiple_val(self) -> None:
        val_spec = f"{self.root}/ep_0, {self.root}/ep_4"
        train_dirs, val_dirs = train_exp.split_train_val(
            str(self.root / "ep_*"), val_spec)
        self.assertEqual(len(val_dirs), 2)
        self.assertEqual(set(train_dirs) & set(val_dirs), set())
        self.assertEqual(
            sorted(os.path.basename(d) for d in train_dirs),
            ["ep_1", "ep_2", "ep_3"],
        )

    def test_split_specs_already_disjoint(self) -> None:
        train_dirs, val_dirs = train_exp.split_train_val(
            f"{self.root}/ep_0, {self.root}/ep_1", str(self.root / "ep_4"))
        self.assertEqual(set(train_dirs) & set(val_dirs), set())
        self.assertEqual(len(train_dirs), 2)

    def test_split_empty_val(self) -> None:
        train_dirs, val_dirs = train_exp.split_train_val(
            str(self.root / "ep_*"), str(self.root / "nomatch_*"))
        self.assertEqual(val_dirs, [])
        self.assertEqual(len(train_dirs), 5)


if __name__ == "__main__":
    unittest.main()
