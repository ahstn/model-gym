"""Taxonomy invariants: index mapping, severity order, decisions, and bands."""

from __future__ import annotations

from itertools import pairwise

import pytest

from model_gym.labels import (
    ID2LABEL,
    LABELS,
    LEVELS,
    NUM_LABELS,
    Decision,
    Level,
    bands,
    coerce_level,
    decision_for_index,
    decision_for_level,
    decision_for_score,
    index_level,
    label_of,
    risk_score,
)


def test_index_and_level_roundtrip() -> None:
    for level in LEVELS:
        assert index_level(level.index) is level
        assert Level.from_index(level.index) is level
        assert Level.from_slug(level.slug) is level
    assert [level.index for level in LEVELS] == [0, 1, 2, 3, 4]


def test_index_order_matches_severity_order() -> None:
    # Lower indices must be more severe, otherwise the ordinal metrics are meaningless.
    assert LABELS == ("critical", "dangerous", "high_risk", "caution", "low_risk")
    assert ID2LABEL == dict(enumerate(LABELS))
    assert NUM_LABELS == len(LEVELS) == 5
    assert [int(level) for level in LEVELS] == sorted(int(level) for level in LEVELS)
    assert label_of(0) == "critical"
    assert label_of(4) == "low_risk"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, Level.CRITICAL),
        ("1", Level.CRITICAL),
        ("CRITICAL", Level.CRITICAL),
        ("HIGH_RISK", Level.HIGH_RISK),
        (5, Level.LOW_RISK),
    ],
)
def test_coerce_level_accepts_int_and_slug(value: object, expected: Level) -> None:
    assert coerce_level(value) is expected


@pytest.mark.parametrize("value", [0, 6, "nope", "", None, True])
def test_coerce_level_rejects_bad_values(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        coerce_level(value)


@pytest.mark.parametrize("index", [-1, 5, 99])
def test_index_out_of_range_rejected(index: int) -> None:
    with pytest.raises(ValueError):
        index_level(index)


def test_decisions_follow_the_policy() -> None:
    assert decision_for_level(1) is Decision.BLOCK
    assert decision_for_level(2) is Decision.APPROVE
    assert decision_for_level(3) is Decision.APPROVE
    assert decision_for_level(4) is Decision.FLAG
    assert decision_for_level(5) is Decision.ALLOW
    assert decision_for_index(0) is Decision.BLOCK
    assert decision_for_index(4) is Decision.ALLOW


def test_risk_score_is_the_expected_level() -> None:
    assert risk_score([1.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert risk_score([0.0, 0.0, 0.0, 0.0, 1.0]) == pytest.approx(5.0)
    assert risk_score([0.2] * 5) == pytest.approx(3.0)
    assert risk_score([0.5, 0.5, 0.0, 0.0, 0.0]) == pytest.approx(1.5)


def test_risk_score_rejects_bad_vectors() -> None:
    with pytest.raises(ValueError):
        risk_score([0.5, 0.5])
    with pytest.raises(ValueError):
        risk_score([0.0] * 5)


def test_bands_cover_the_score_range_without_gaps() -> None:
    ordered = bands()
    assert [band.decision for band in ordered] == [Decision.BLOCK, Decision.APPROVE, Decision.FLAG, Decision.ALLOW]
    # Bands are half a level wide either side of the levels they cover, so the
    # block band starts at 0.5 even though scores below 1.0 are rejected.
    assert ordered[0].low == 0.5
    assert ordered[-1].high == 5.5
    for left, right in pairwise(ordered):
        assert left.high == right.low


def test_decision_never_becomes_less_permissive_as_score_rises() -> None:
    permissiveness = {Decision.BLOCK: 0, Decision.APPROVE: 1, Decision.FLAG: 2, Decision.ALLOW: 3}
    scores = [1.0 + index * 0.05 for index in range(81)]
    decisions = [decision_for_score(score) for score in scores]
    ranked = [permissiveness[decision] for decision in decisions]
    assert ranked == sorted(ranked)
    assert decisions[0] is Decision.BLOCK
    assert decision_for_score(5.0) is Decision.ALLOW


@pytest.mark.parametrize("score", [0.99, 5.01])
def test_decision_for_score_rejects_out_of_range(score: float) -> None:
    with pytest.raises(ValueError):
        decision_for_score(score)
