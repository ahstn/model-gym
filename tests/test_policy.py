"""Runtime action decisions, fitting constraints, and persisted policy failures."""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest

from model_gym.metrics import compute_metrics
from model_gym.policy import (
    DecisionPolicy,
    assess_policy,
    decision_metrics,
    fit_threshold_policy,
    load_policy,
    save_policy,
)


def test_tail_risk_distinguishes_equal_expected_scores_and_survives_reload(tmp_path: Path) -> None:
    probabilities = np.asarray([[0, 0, 1, 0, 0], [0.5, 0, 0, 0, 0.5]])
    assert DecisionPolicy().decide(probabilities).tolist() == ["approve", "approve"]
    policy = DecisionPolicy("risk_thresholds", 0.5, 0.5, 0.5)
    path = tmp_path / "decision-policy.json"
    save_policy(path, policy)
    restored = load_policy(path)
    assert restored.decide(probabilities).tolist() == ["approve", "block"]
    metrics = compute_metrics(probabilities, [2, 0], policy=restored)
    assert metrics.score_policy["critical_miss_count"] == 1
    assert metrics.runtime_policy["critical_miss_count"] == 0


def test_policy_priority_and_threshold_equality() -> None:
    policy = DecisionPolicy("risk_thresholds", 0.3, 0.6, 0.4)
    probabilities = np.asarray(
        [
            [0.3, 0.3, 0, 0, 0.4],
            [0.1, 0.5, 0, 0, 0.4],
            [0, 0.2, 0, 0.4, 0.4],
            [0, 0.2, 0, 0.5, 0.3],
        ]
    )
    assert policy.decide(probabilities).tolist() == ["block", "approve", "allow", "flag"]


def test_policy_fit_cannot_make_indistinguishable_risks_satisfy_conflicting_gates() -> None:
    labels = np.tile(np.arange(5), 4)
    probabilities = np.full((len(labels), 5), 0.2)
    policy, report = fit_threshold_policy(probabilities, labels, grid_size=5)
    assert not report["eligible"]
    assert report["normalized_violation"] > 0
    assert assess_policy(decision_metrics(probabilities, labels, policy)) == {
        key: report[key] for key in ("eligible", "supported", "limits", "violations", "normalized_violation")
    }
    with pytest.raises(ValueError, match="support"):
        fit_threshold_policy(probabilities[:4], [4, 4, 4, 4])


def test_policy_fit_preserves_separable_critical_and_benign_actions() -> None:
    labels = np.arange(5)
    probabilities = np.eye(5) * 0.7 + 0.3 / 5
    policy, report = fit_threshold_policy(probabilities, labels, grid_size=11)
    assert report["eligible"]
    assert policy.decide(probabilities).tolist() == ["block", "approve", "approve", "flag", "allow"]


def test_malformed_saved_policy_never_becomes_default_score_bands(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": 1, "policy": {"kind": "risk_thresholds"}}))
    with pytest.raises(ValueError, match="every threshold"):
        load_policy(path)
    with pytest.raises(ValueError, match="finite"):
        DecisionPolicy("risk_thresholds", float("nan"))
    with pytest.raises(ValueError, match="probabilities"):
        DecisionPolicy().decide(np.asarray([[np.nan, 0, 0, 0, 1]]))
