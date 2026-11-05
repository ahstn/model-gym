"""The operational postpass cannot become a score-selection mechanism."""

import api_missing_postpass as postpass
import pytest


@pytest.mark.parametrize("status", ["failed", "retryable_failure", "unknown", "invalid"])
def test_missing_outcome_is_eligible(status):
    assert postpass.eligible({"status": status})


@pytest.mark.parametrize(
    "field", ["technical_score", "aesthetic_score", "technical_score_native", "aesthetic_score_native"]
)
def test_any_numeric_outcome_prevents_replay(field):
    assert not postpass.eligible({"status": "invalid", field: 1.0})


def test_valid_unscorable_is_not_retried():
    assert not postpass.eligible({"status": "valid", "technical_score": None, "aesthetic_score": None})


def test_observations_alone_do_not_admit_postpass(tmp_path):
    root = tmp_path / "results/api/luna"
    postpass.runner.atomic(root / "observations/R0001.json", {})
    postpass.runner.atomic(root / "plans/main.json", {"phase": "main", "cases": [{"case_id": "R0001"}]})
    postpass.runner.atomic(root / "main-status.json", {"status": "running"})
    assert not postpass.main_phases_complete("luna", tmp_path, {"R0001"})
    postpass.runner.atomic(root / "main-status.json", {"status": "complete"})
    assert postpass.main_phases_complete("luna", tmp_path, {"R0001"})


def test_postpass_cannot_supply_its_own_main_admission(tmp_path):
    root = tmp_path / "results/api/luna"
    postpass.runner.atomic(root / "observations/R0001.json", {})
    postpass.runner.atomic(
        root / "plans/missing-output-postpass.json",
        {"phase": "missing-output-postpass", "cases": [{"case_id": "R0001"}]},
    )
    postpass.runner.atomic(root / "missing-output-postpass-status.json", {"status": "complete"})
    assert not postpass.main_phases_complete("luna", tmp_path, {"R0001"})


def test_unexpected_observations_fail_closed(tmp_path):
    root = tmp_path / "results/api/luna"
    postpass.runner.atomic(root / "observations/R0002.json", {})
    with pytest.raises(ValueError, match="outside the frozen corpus"):
        postpass.main_phases_complete("luna", tmp_path, {"R0001"})
