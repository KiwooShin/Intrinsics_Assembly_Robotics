"""Parse the AIC engine's ``scoring.yaml`` and classify trial outcomes.

The schema is fixed by the C++ engine/scoring packages in this repo:

* ``aic_engine/src/aic_engine.cpp`` -> ``Score::serialize`` writes a top-level
  ``total`` plus one block per trial id, each with ``tier_1``/``tier_2``/
  ``tier_3`` sub-nodes (``TierScore::to_yaml``).
* ``aic_scoring/include/aic_scoring/TierScore.hh`` defines every tier's YAML:
  each tier has ``score`` and ``message``; ``tier_2`` additionally has a
  ``categories`` map keyed by human-readable category names.
* ``aic_scoring/src/ScoringTier2.cc`` populates exactly these five categories:
  ``insertion force`` (0 or -12), ``contacts`` (0 or -24), ``duration`` (0..12),
  ``trajectory smoothness`` (0..6), ``trajectory efficiency`` (0..6); and
  ``tier_3`` is 75 (full insertion), -12 (wrong port), 38..50 (partial),
  0..25 (proximity/distance), or 0 (task not completed).

Concretely::

    total: 96.0
    trial_1:
      tier_1: {score: 1.0, message: "Model validation succeeded."}
      tier_2:
        score: 20.0
        message: "Scoring succeeded."
        categories:
          insertion force: {score: 0.0, message: "No excessive force detected"}
          contacts: {score: 0.0, message: "No contact detected."}
          duration: {score: 9.0, message: "Task duration: 12.00 seconds."}
          trajectory smoothness: {score: 5.0, message: "..."}
          trajectory efficiency: {score: 6.0, message: "..."}
      tier_3: {score: 75.0, message: "Cable insertion successful."}
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

import yaml

# Category keys emitted by ScoringTier2::ComputeScore (exact strings).
CAT_INSERTION_FORCE = "insertion force"
CAT_CONTACTS = "contacts"
CAT_DURATION = "duration"
CAT_SMOOTHNESS = "trajectory smoothness"
CAT_EFFICIENCY = "trajectory efficiency"

# Tier-3 landmark scores (aic_scoring/src/ScoringTier2.cc).
TIER3_FULL_INSERTION = 75.0
TIER3_WRONG_PORT = -12.0
TIER3_PARTIAL_MIN = 38.0  # kMinInsertionScore
TIER3_PARTIAL_MAX = 50.0  # kMaxInsertionScore


class Outcome(enum.Enum):
    """Mutually exclusive qualitative outcome of a single scored trial.

    The precedence (evaluated top-to-bottom in :func:`classify_outcome`) is:
    a meaningful insertion result (``FULL``/``PARTIAL``) is reported first
    because it is the headline success; otherwise a hard safety/force failure
    (``COLLISION``/``FORCE``) is surfaced as the cause; otherwise the plug's
    proximity to the port (``PROXIMITY``) or a plain ``MISS``.
    """

    FULL = "full"
    PARTIAL = "partial"
    COLLISION = "collision"
    FORCE = "force"
    PROXIMITY = "proximity"
    MISS = "miss"


@dataclasses.dataclass(frozen=True)
class TrialScoreBreakdown:
    """Structured view of one trial's entry in ``scoring.yaml``.

    Attributes:
        trial_id: Trial key in the YAML (e.g. ``"trial_1"``).
        tier_1: Tier-1 model-validation score (0 or 1).
        tier_2: Tier-2 aggregate score (sum of its categories).
        tier_3: Tier-3 task/insertion score.
        categories: Mapping of category name to numeric score for the five
            tier-2 categories (missing categories are absent from the map).
        tier_1_message: Human-readable tier-1 message.
        tier_2_message: Human-readable tier-2 message.
        tier_3_message: Human-readable tier-3 message.
    """

    trial_id: str
    tier_1: float
    tier_2: float
    tier_3: float
    categories: dict[str, float]
    tier_1_message: str = ""
    tier_2_message: str = ""
    tier_3_message: str = ""

    @property
    def total(self) -> float:
        """Total trial score (``tier_1 + tier_2 + tier_3``)."""
        return self.tier_1 + self.tier_2 + self.tier_3

    def category(self, name: str) -> float:
        """Return a tier-2 category score, or ``0.0`` if it is absent.

        Args:
            name: One of the ``CAT_*`` constants.

        Returns:
            The category score, defaulting to ``0.0`` when not reported.
        """
        return float(self.categories.get(name, 0.0))

    @property
    def insertion_force_score(self) -> float:
        """Tier-2 insertion-force score (0 or the -12 penalty)."""
        return self.category(CAT_INSERTION_FORCE)

    @property
    def contacts_score(self) -> float:
        """Tier-2 contacts score (0 or the -24 off-limit-contact penalty)."""
        return self.category(CAT_CONTACTS)

    @property
    def duration_score(self) -> float:
        """Tier-2 duration score (0..12)."""
        return self.category(CAT_DURATION)

    @property
    def smoothness_score(self) -> float:
        """Tier-2 trajectory-smoothness (jerk) score (0..6)."""
        return self.category(CAT_SMOOTHNESS)

    @property
    def efficiency_score(self) -> float:
        """Tier-2 trajectory-efficiency (path length) score (0..6)."""
        return self.category(CAT_EFFICIENCY)

    @property
    def inserted(self) -> bool:
        """Whether a full, correct-port insertion was scored (tier_3 == 75)."""
        return self.tier_3 >= TIER3_FULL_INSERTION

    @property
    def has_force_penalty(self) -> bool:
        """Whether the excessive-force penalty was applied."""
        return self.insertion_force_score < 0.0

    @property
    def has_collision_penalty(self) -> bool:
        """Whether an off-limit-contact penalty was applied."""
        return self.contacts_score < 0.0

    @property
    def outcome(self) -> Outcome:
        """The classified :class:`Outcome` for this trial."""
        return classify_outcome(self)


@dataclasses.dataclass(frozen=True)
class ScoringResult:
    """Parsed contents of a full ``scoring.yaml`` file.

    Attributes:
        total: The engine's top-level ``total`` score across all trials.
        trials: Mapping of trial id to its :class:`TrialScoreBreakdown`.
        path: Source file the result was parsed from (if any).
    """

    total: float
    trials: dict[str, TrialScoreBreakdown]
    path: Path | None = None

    def single_trial(self) -> TrialScoreBreakdown:
        """Return the sole trial when the file contains exactly one.

        Returns:
            The single :class:`TrialScoreBreakdown`.

        Raises:
            ValueError: If the file does not contain exactly one trial.
        """
        if len(self.trials) != 1:
            raise ValueError(
                f"expected exactly one trial, found {sorted(self.trials)}"
            )
        return next(iter(self.trials.values()))


def classify_outcome(breakdown: TrialScoreBreakdown) -> Outcome:
    """Classify a trial into a single mutually-exclusive outcome class.

    Precedence (first match wins):

    1. ``tier_3 >= 75`` -> ``FULL`` (correct-port insertion; the headline win).
    2. ``38 <= tier_3 <= 50`` -> ``PARTIAL`` (plug partially seated).
    3. off-limit-contact penalty present -> ``COLLISION``.
    4. excessive-force penalty present -> ``FORCE``.
    5. ``0 < tier_3 < 38`` -> ``PROXIMITY`` (distance-based partial credit).
    6. otherwise -> ``MISS`` (includes ``tier_3 == 0`` and wrong-port ``-12``).

    Args:
        breakdown: The parsed trial score.

    Returns:
        The classified :class:`Outcome`.
    """
    t3 = breakdown.tier_3
    if t3 >= TIER3_FULL_INSERTION:
        return Outcome.FULL
    if TIER3_PARTIAL_MIN <= t3 <= TIER3_PARTIAL_MAX:
        return Outcome.PARTIAL
    if breakdown.has_collision_penalty:
        return Outcome.COLLISION
    if breakdown.has_force_penalty:
        return Outcome.FORCE
    if 0.0 < t3 < TIER3_PARTIAL_MIN:
        return Outcome.PROXIMITY
    return Outcome.MISS


def _tier_score(node: Any, default_msg: str = "") -> tuple[float, str]:
    """Extract ``(score, message)`` from a tier sub-node.

    Args:
        node: The mapping under a ``tier_*`` key.
        default_msg: Message to use when the node has none.

    Returns:
        A ``(score, message)`` tuple.

    Raises:
        ValueError: If ``node`` is not a mapping with a numeric ``score``.
    """
    if not isinstance(node, dict) or "score" not in node:
        raise ValueError(f"malformed tier node (missing 'score'): {node!r}")
    score = float(node["score"])
    message = str(node.get("message", default_msg))
    return score, message


def parse_scoring_dict(data: dict[str, Any], path: Path | None = None) -> ScoringResult:
    """Parse an already-loaded ``scoring.yaml`` mapping.

    Args:
        data: The mapping loaded from a scoring YAML document.
        path: Optional source path for provenance.

    Returns:
        A :class:`ScoringResult`.

    Raises:
        ValueError: If the mapping is missing required structure.
    """
    if not isinstance(data, dict):
        raise ValueError(f"scoring document must be a mapping, got {type(data)}")
    if "total" not in data:
        raise ValueError("scoring document missing top-level 'total'")
    total = float(data["total"])
    trials: dict[str, TrialScoreBreakdown] = {}
    for key, value in data.items():
        if key == "total":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"trial '{key}' is not a mapping: {value!r}")
        t1_score, t1_msg = _tier_score(value.get("tier_1", {}))
        t2_node = value.get("tier_2", {})
        t2_score, t2_msg = _tier_score(t2_node)
        t3_score, t3_msg = _tier_score(value.get("tier_3", {}))
        categories: dict[str, float] = {}
        for cat_name, cat_node in (t2_node.get("categories", {}) or {}).items():
            if isinstance(cat_node, dict) and "score" in cat_node:
                categories[cat_name] = float(cat_node["score"])
        trials[key] = TrialScoreBreakdown(
            trial_id=key,
            tier_1=t1_score,
            tier_2=t2_score,
            tier_3=t3_score,
            categories=categories,
            tier_1_message=t1_msg,
            tier_2_message=t2_msg,
            tier_3_message=t3_msg,
        )
    if not trials:
        raise ValueError("scoring document has a 'total' but no trials")
    return ScoringResult(total=total, trials=trials, path=path)


def parse_scoring_file(path: str | Path) -> ScoringResult:
    """Load and parse a ``scoring.yaml`` file.

    Args:
        path: Path to the YAML file written by the engine.

    Returns:
        A :class:`ScoringResult`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file content is malformed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scoring file not found: {p}")
    with p.open() as fh:
        data = yaml.safe_load(fh)
    return parse_scoring_dict(data, path=p)
