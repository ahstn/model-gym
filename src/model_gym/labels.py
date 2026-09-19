"""Five-level command-risk taxonomy.

Levels are ordinal and monotonic: 1 is the most severe, 5 the least. Classifier
label indices are ``level - 1``, so the head's index order matches severity order
and neighbouring indices are neighbouring severities.

The taxonomy follows the guardrail scheme:

| Level | Slug        | Decision | Meaning                                                    |
|------:|-------------|----------|------------------------------------------------------------|
|     1 | `critical`  | block    | Catastrophic/irreversible. Needs explicit human approval.   |
|     2 | `dangerous` | approve  | Destructive, broad blast radius, usually recoverable.       |
|     3 | `high_risk` | approve  | Destroys local state or interrupts a service.               |
|     4 | `caution`   | flag     | Disruptive or security-weakening, not destructive alone.    |
|     5 | `low_risk`  | allow    | Normally non-destructive / read-only.                       |

Level 5 is deliberately named ``low_risk`` rather than ``safe``: arbitrary
execution is never categorically safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Final


class Decision(StrEnum):
    """Action a guardrail takes for a predicted risk level."""

    BLOCK = "block"
    APPROVE = "approve"
    FLAG = "flag"
    ALLOW = "allow"


class Level(IntEnum):
    """Command risk level: 1 critical, 5 low risk."""

    CRITICAL = 1
    DANGEROUS = 2
    HIGH_RISK = 3
    CAUTION = 4
    LOW_RISK = 5

    @property
    def index(self) -> int:
        """Zero-based classifier label index, in severity order."""
        return int(self) - 1

    @property
    def slug(self) -> str:
        """Stable identifier used in JSON artifacts and reports."""
        return LEVEL_SLUGS[self]

    @property
    def decision(self) -> Decision:
        """Default guardrail decision for this level."""
        return LEVEL_DECISIONS[self]

    @property
    def description(self) -> str:
        return LEVEL_DESCRIPTIONS[self]

    @classmethod
    def from_index(cls, index: int) -> Level:
        """Build a level from a zero-based classifier label index."""
        return cls(coerce_index(index) + 1)

    @classmethod
    def from_slug(cls, slug: str) -> Level:
        try:
            return Level(LEVEL_IDS[slug.strip().lower()])
        except KeyError as exc:
            raise ValueError(f"unknown level slug {slug!r}; expected one of {sorted(LEVEL_IDS)}") from exc


LEVEL_SLUGS: Final[dict[Level, str]] = {
    Level.CRITICAL: "critical",
    Level.DANGEROUS: "dangerous",
    Level.HIGH_RISK: "high_risk",
    Level.CAUTION: "caution",
    Level.LOW_RISK: "low_risk",
}

LEVEL_DESCRIPTIONS: Final[dict[Level, str]] = {
    Level.CRITICAL: "Potentially catastrophic or irreversible; broad production or data destruction.",
    Level.DANGEROUS: "Destructive or access-impacting with significant blast radius; usually recoverable.",
    Level.HIGH_RISK: "Can destroy local state, interrupt services, or cause meaningful operational damage.",
    Level.CAUTION: "Potentially disruptive or security-weakening, but not normally destructive by itself.",
    Level.LOW_RISK: "Normally non-destructive or read-only.",
}

LEVEL_DECISIONS: Final[dict[Level, Decision]] = {
    Level.CRITICAL: Decision.BLOCK,
    Level.DANGEROUS: Decision.APPROVE,
    Level.HIGH_RISK: Decision.APPROVE,
    Level.CAUTION: Decision.FLAG,
    Level.LOW_RISK: Decision.ALLOW,
}

LEVEL_IDS: Final[dict[str, int]] = {slug: int(level) for level, slug in LEVEL_SLUGS.items()}
LEVELS: Final[tuple[Level, ...]] = tuple(Level)
NUM_LABELS: Final[int] = len(LEVELS)

# Hugging Face label maps, index-ordered by severity.
LABELS: Final[tuple[str, ...]] = tuple(level.slug for level in LEVELS)
ID2LABEL: Final[dict[int, str]] = {level.index: level.slug for level in LEVELS}
LABEL2ID: Final[dict[str, int]] = {slug: index for index, slug in ID2LABEL.items()}

# Inclusive level bound for each decision, used by the runtime guardrail.
DECISION_LEVEL_RANGES: Final[dict[Decision, tuple[int, int]]] = {
    decision: (
        min(level for level in LEVELS if LEVEL_DECISIONS[level] is decision),
        max(level for level in LEVELS if LEVEL_DECISIONS[level] is decision),
    )
    for decision in Decision
}

# Expected-level score band (inclusive lower, exclusive upper) per decision.
SCORE_BANDS: Final[dict[Decision, tuple[float, float]]] = {
    decision: (low - 0.5, high + 0.5) for decision, (low, high) in DECISION_LEVEL_RANGES.items()
}
SCORE_MIN: Final[float] = 1.0
SCORE_MAX: Final[float] = 5.0


def coerce_level(value: Any) -> Level:
    """Coerce an int, numeric string, slug, or level into a :class:`Level`."""
    if isinstance(value, Level):
        return value
    if isinstance(value, bool):
        raise TypeError(f"invalid level {value!r}; expected 1-5 or a level slug")
    if isinstance(value, int):
        return Level(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in LEVEL_IDS:
            return Level(LEVEL_IDS[text])
        try:
            return Level(int(text))
        except ValueError as exc:
            raise ValueError(f"invalid level {value!r}; expected 1-5 or a level slug") from exc
    raise TypeError(f"invalid level {value!r}; expected 1-5 or a level slug")


def coerce_index(index: Any) -> int:
    """Coerce a zero-based label index, rejecting out-of-range values."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"invalid label index {index!r}; expected an int in [0, {NUM_LABELS})")
    if not 0 <= index < NUM_LABELS:
        raise ValueError(f"label index {index} out of range [0, {NUM_LABELS})")
    return index


def index_level(index: int) -> Level:
    """Map a zero-based label index to its severity level."""
    return Level(coerce_index(index) + 1)


def level_of(index: int) -> int:
    """Map a zero-based label index to its integer severity (1-5)."""
    return int(index_level(index))


def label_of(index: int) -> str:
    """Map a zero-based label index to its label slug."""
    return LABELS[coerce_index(index)]


def decision_for_level(level: Any) -> Decision:
    """Guardrail decision for a level (1-5) or level slug."""
    return coerce_level(level).decision


def decision_for_index(index: int) -> Decision:
    """Guardrail decision for a zero-based label index."""
    return index_level(index).decision


def risk_score(probabilities: Sequence[float]) -> float:
    """Expected severity level from a probability vector: 1.0 critical, 5.0 low risk."""
    if len(probabilities) != NUM_LABELS:
        raise ValueError(f"expected {NUM_LABELS} probabilities, got {len(probabilities)}")
    total = sum(float(p) for p in probabilities)
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return sum(float(p) * level for p, level in zip(probabilities, LEVELS, strict=True)) / total


def decision_for_score(score: float) -> Decision:
    """Map an expected-severity score in [1, 5] to a guardrail decision."""
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"risk score {score} outside [{SCORE_MIN}, {SCORE_MAX}]")
    # Bands are half-open: an exact upper boundary belongs to the next,
    # less-severe band. Export metadata uses the same [low, high) rule.
    for decision in (Decision.BLOCK, Decision.APPROVE, Decision.FLAG, Decision.ALLOW):
        low, high = SCORE_BANDS[decision]
        if low <= score < high:
            return decision
    return Decision.ALLOW


@dataclass(frozen=True, slots=True)
class Band:
    """Half-open score band and the decision it maps to."""

    low: float
    high: float
    decision: Decision


def bands() -> tuple[Band, ...]:
    """Ordered score bands for the default decision policy."""
    return tuple(Band(low=low, high=high, decision=decision) for decision, (low, high) in SCORE_BANDS.items())


def is_severe(level: Any) -> bool:
    """True when a level requires blocking or approval (1-2)."""
    return coerce_level(level) <= Level.DANGEROUS


def label_table() -> list[dict[str, Any]]:
    """Serializable label table for runtime artifacts."""
    return [
        {
            "index": level.index,
            "level": int(level),
            "slug": level.slug,
            "decision": str(level.decision),
            "description": level.description,
        }
        for level in LEVELS
    ]
