"""Fixed matched-seed evaluation harness for AIC policy checkpoints.

Modules:
    * :mod:`eval_lib.suite` — deterministic stratified suite generation.
    * :mod:`eval_lib.scoring` — ``scoring.yaml`` parsing + outcome classification.
    * :mod:`eval_lib.stats` — pure-numpy IQM / bootstrap / Wilson estimators.
    * :mod:`eval_lib.runner` — sim orchestration (mockable) + dry-run.
    * :mod:`eval_lib.report` — aggregation, markdown/CSV reports, paired compare.

The public CLI lives in the repo-root ``eval_suite.py``.
"""

from __future__ import annotations

__all__ = ["scoring", "stats", "suite", "runner", "report"]
