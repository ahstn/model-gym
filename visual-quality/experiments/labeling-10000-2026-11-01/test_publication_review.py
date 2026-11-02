"""Regression coverage for auditing reference admission across all attempts."""

import json
from pathlib import Path

import publish_run as publish
import pytest

from visual_quality.publication import sha


def group(started):
    return {model: {"request_started_at": started} for model in publish.MODELS}


def test_reference_chronology_checks_prior_failed_attempts():
    reference = {"review_completed_at": "2026-11-01T12:00:00+00:00"}
    rows = group("2026-11-01T12:01:00+00:00")
    rows["qrealign_lite"]["attempt_provenance"] = [
        {"request_started_at": "2026-11-01T11:59:00+00:00", "status": "http_error"},
        {"request_started_at": "2026-11-01T12:01:00+00:00", "status": "valid"},
    ]

    with pytest.raises(ValueError, match="preceded"):
        publish.validate_reference_order(reference, rows)


def test_reference_chronology_requires_packet_freeze_before_admission(tmp_path):
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"frozen_at": "2026-11-01T12:02:00+00:00"}))
    reference = {
        "review_completed_at": "2026-11-01T12:00:00+00:00",
        "review_freeze_path": str(receipt),
        "review_freeze_sha256": sha(receipt),
    }

    with pytest.raises(ValueError, match="preceded"):
        publish.validate_reference_order(reference, group("2026-11-01T12:01:00+00:00"))

    publish.validate_reference_order(reference, group("2026-11-01T12:03:00+00:00"))


def test_packet_freeze_requires_a_timestamp_for_every_recorded_attempt(tmp_path):
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"frozen_at": "2026-11-01T12:02:00+00:00"}))
    reference = {
        "review_completed_at": "2026-11-01T12:00:00+00:00",
        "review_freeze_path": str(receipt),
        "review_freeze_sha256": sha(receipt),
    }
    rows = group("2026-11-01T12:03:00+00:00")
    rows["luna"].update(attempt_count=2, attempt_provenance=[{"request_started_at": None}])
    with pytest.raises(ValueError, match="missing its measured request start"):
        publish.validate_reference_order(reference, rows)
    rows["luna"] = {"status": "missing", "attempt_count": 0}
    publish.validate_reference_order(reference, rows)


def test_packet_freeze_must_remain_pinned_and_follow_review_completion(tmp_path):
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"frozen_at": "2026-11-01T11:59:00+00:00"}))
    reference = {
        "review_completed_at": "2026-11-01T12:00:00+00:00",
        "review_freeze_path": str(receipt),
        "review_freeze_sha256": sha(receipt),
    }
    with pytest.raises(ValueError, match="freeze must follow completed review"):
        publish.validate_reference_order(reference, group("2026-11-01T12:03:00+00:00"))
    receipt.write_text(json.dumps({"frozen_at": "2026-11-01T12:02:00+00:00"}))
    with pytest.raises(ValueError, match="Pinned reference freeze changed"):
        publish.validate_reference_order(reference, group("2026-11-01T12:03:00+00:00"))


def test_api_export_preserves_timestamps_for_every_finalized_attempt(tmp_path):
    root = tmp_path / "luna"
    observed = root / "observations/R9000.json"
    observed.parent.mkdir(parents=True)
    observed.write_text(json.dumps({"case_id": "R9000", "attempt_count": 2, "status": "valid"}))
    started = ["2026-11-01T12:00:00+00:00", "2026-11-01T12:01:00+00:00"]
    for number, timestamp in enumerate(started, 1):
        request = root / f"attempts/R9000/{number:02}/request.json"
        request.parent.mkdir(parents=True)
        request.write_text(json.dumps({"case_id": "R9000", "plan_id": "frozen-plan"}))
        request.with_name("result.json").write_text(
            json.dumps({"request_started_at": timestamp, "price_usd": "0.001"})
        )

    observations, _ = publish.load_api(tmp_path)
    attempts = observations["luna"][0]["attempt_provenance"]
    assert [attempt["request_started_at"] for attempt in attempts] == started
    for attempt in attempts:
        assert attempt["request_record_sha256"] == sha(Path(attempt["request_path"]))
        assert attempt["raw_response_sha256"] == sha(Path(attempt["raw_response_path"]))
        assert "parsed" not in attempt and "body" not in attempt


@pytest.mark.parametrize("reused", [False, True])
def test_single_lite_observation_counts_its_original_request(reused):
    case = {"case_id": "R9000", "view_sha256": "delivered-view"}
    observation = {
        "model": "qrealign_lite",
        "status": "valid",
        "technical_score": 3.0,
        "aesthetic_score": 3.0,
        "task_valid": {"quality": True, "aesthetics": True},
        "price_usd": "0.00",
    }
    if reused:
        observation["reuse"] = {"new_request_sent": False}
    assert publish.normalized_model(case, "qrealign_lite", observation)["attempt_count"] == 1
    observation["attempt_count"] = 2
    assert publish.normalized_model(case, "qrealign_lite", observation)["attempt_count"] == 2
    assert publish.normalized_model(case, "qrealign_lite", None)["attempt_count"] == 0
