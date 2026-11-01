"""Recovery must neither replay failed requests nor accept changed receipts."""

import copy
import json

import api_recovery as recovery
import pytest


def test_recovery_selects_only_never_attempted_jobs():
    plan = {"jobs": [{"job_id": key} for key in ("valid", "invalid", "failed", "unknown", "fresh")]}
    records = [{"job_id": key, "status": key} for key in ("valid", "invalid", "failed", "unknown")]
    assert recovery.unattempted(plan, records) == [{"job_id": "fresh"}]


@pytest.mark.parametrize("change", [None, "receipt", "request", "model", "status"])
def test_approval_is_bound_to_exact_reviewed_failure(tmp_path, change):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"body": json.dumps(recovery.ERROR_BODY), "transport_error": None}))
    row = {"attempt_id": "rehearsal/baseline--R1234/01", "model": "jev", "status": "failed",
           "http_status": 503, "price_usd": None, "raw_response_path": str(path),
           "raw_response_sha256": recovery.contract.file_sha(path),
           "request_journal_sha256": "journal-hash", "request_sha256": "request-hash"}
    approvals = {row["attempt_id"]: copy.deepcopy(row)}
    if change == "receipt":
        path.write_text(path.read_text() + "\n")
    elif change == "request":
        row["request_sha256"] = "changed"
    elif change == "model":
        row["model"] = "liquid"
    elif change == "status":
        row["status"] = "unknown"
    assert recovery.approved_failure(row, approvals) is (change is None)


def test_quarantine_circuit_breakers_include_existing_attempts_and_exact_boundaries():
    policy = {"maximum_consecutive_failures": 5, "rolling_window_requests": 100,
              "maximum_failures_in_window": 20}
    assert recovery.circuit_breaker([False] * 42 + [True] * 4, policy) is None
    assert recovery.circuit_breaker([False] * 42 + [True] * 5, policy) == "consecutive_known_provider_failures"
    high = ([True, False, False, False, False] * 20)
    assert recovery.circuit_breaker(high[:99], policy) is None
    assert recovery.circuit_breaker(high, policy) == "known_provider_failure_rate"
    assert recovery.circuit_breaker(high + [False], policy) is None


@pytest.mark.parametrize("body", [
    {"error": {"message": "Generic server error", "code": 503}},
    {**recovery.ERROR_BODY, "usage": {"cost": 0}},
    {**recovery.ERROR_BODY, "answers": {}},
])
def test_quarantine_never_accepts_other_503_classes_or_scored_bodies(tmp_path, body):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"body": json.dumps(body), "transport_error": None}))
    row = {"model": "jev", "status": "failed", "http_status": 503, "price_usd": None,
           "raw_response_path": str(path), "raw_response_sha256": recovery.contract.file_sha(path)}
    assert not recovery.known_image_failure(row)
