"""Ensure the repo root is importable so ``import opt.*`` works under discover.

Unit tests import this module first; it inserts the repository root (two levels
up from this file) onto ``sys.path``. This lets the tests run via
``python -m unittest discover -s opt/tests`` from anywhere.
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
