"""Ordinal metrics: the behaviour a guardrail is judged on."""

from __future__ import annotations

import numpy as np
import pytest

from model_gym.labels import NUM_LABELS
from model_gym.metrics import compute_metrics, expected_calibration_error, softmax


def one_hot(indices: list[int], confidence: float = 1.0) -> np.ndarray:
    """Confident probability rows for the given label indices."""
    rows = np.full((len(indices), NUM_LABELS), (1.0 - confidence) / (NUM_LABELS - 1))
    for row, index in zip(rows, indices, strict=True):
        row[index] = confidence
    return rows


def test_perfect_predictions_score_perfectly() -> None:
    labels = [0, 1, 2, 3, 4, 0, 4]
    metrics = compute_metrics(one_hot(labels), labels)
    assert metrics.n == len(labels)
    assert metrics.accuracy == 1.0
    assert metrics.qwk == pytest.approx(1.0)
    assert metrics.mae == 0.0
    assert metrics.severe_miss_rate == 0.0
    assert metrics.over_block_rate == 0.0
    assert metrics.decision_accuracy == 1.0
    assert metrics.ece == pytest.approx(0.0, abs=1e-9)
    assert metrics.per_level_recall == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_critical_commands_called_safe_are_counted_as_severe_misses() -> None:
    # Two critical commands predicted low risk, one dangerous predicted critical.
    probabilities = one_hot([4, 4, 0])
    metrics = compute_metrics(probabilities, [0, 0, 1])
    # Level 1 and level 2 both count as severe, so all three rows are considered.
    assert metrics.severe_miss_support == 3
    assert metrics.severe_miss_rate == pytest.approx(2 / 3)
    assert metrics.over_block_support == 0
    assert metrics.confusion[0][4] == 2
    assert metrics.confusion[1][0] == 1


def test_blocking_safe_commands_are_counted_as_over_blocks() -> None:
    probabilities = one_hot([0, 0, 4])
    metrics = compute_metrics(probabilities, [4, 4, 4])
    assert metrics.over_block_support == 3
    assert metrics.over_block_rate == pytest.approx(2 / 3)
    assert metrics.severe_miss_rate == 0.0


def test_decision_accuracy_ignores_neighbouring_levels_in_the_same_band() -> None:
    # Level 3 and level 2 both map to "approve", so the decision is unchanged.
    metrics = compute_metrics(one_hot([1, 4]), [2, 4])
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.decision_accuracy == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(0.5)


def test_ordinal_metrics_penalize_distant_errors_more() -> None:
    labels = [0, 1, 2, 3, 4] * 4
    near = compute_metrics(one_hot([0, 1, 2, 3, 3] * 4), labels)
    far = compute_metrics(one_hot([0, 1, 2, 3, 0] * 4), labels)
    assert near.qwk > far.qwk
    assert near.mae < far.mae


def test_per_level_recall_and_confusion_are_positioned_by_severity() -> None:
    labels = [0, 1, 1, 2]
    probabilities = one_hot([0, 1, 2, 1])
    metrics = compute_metrics(probabilities, labels)
    # Row 1 (level 2): one correct, one predicted level 3.
    assert metrics.confusion[1] == (0, 1, 1, 0, 0)
    assert metrics.per_level_recall[1] == pytest.approx(0.5)
    assert metrics.per_level_recall[0] == pytest.approx(1.0)
    assert metrics.per_level_support == (1, 2, 1, 0, 0)
    # Level 2 is predicted twice (once correctly, once for a level 3 command).
    assert metrics.per_level_precision[1] == pytest.approx(0.5)


def test_expected_calibration_error_detects_overconfidence() -> None:
    # Half right at full confidence: the gap is 0.5.
    probabilities = one_hot([0, 0, 1, 1])
    labels = np.asarray([0, 1, 0, 1])
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.5)


def test_softmax_is_stable_for_large_logits() -> None:
    logits = np.asarray([[1000.0, -1000.0, 0.0, 5.0, 5.0], [-1000.0, 0.0, 0.0, 0.0, 0.0]])
    probabilities = softmax(logits)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probabilities[0][0] > 0.999


def test_expected_level_tracks_the_distribution() -> None:
    metrics = compute_metrics(one_hot([0, 4]), [0, 4])
    assert metrics.mean_expected_level == pytest.approx(3.0)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="labels"):
        compute_metrics(one_hot([0, 1]), [0])
    with pytest.raises(ValueError, match="probabilities shaped"):
        compute_metrics(np.zeros((3, 2)), [0, 1, 2])
    with pytest.raises(ValueError, match="no rows"):
        compute_metrics(np.zeros((0, NUM_LABELS)), [])


def test_report_lists_every_level_and_the_asymmetric_rates() -> None:
    metrics = compute_metrics(one_hot([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
    report = metrics.format(title="unit test split")
    assert "# unit test split" in report
    assert "severe miss rate" in report
    assert "over block rate" in report
    for slug in ("critical", "dangerous", "high_risk", "caution", "low_risk"):
        assert slug in report
    assert "confusion matrix" in report
