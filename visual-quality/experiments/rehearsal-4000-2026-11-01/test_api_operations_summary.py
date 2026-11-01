"""Operational reporting preserves failed latencies, unknown bills and segment identity."""

import copy
from pathlib import Path

import api_operations_summary as ops
import pytest


def rows():
    return [{"attempt_id": f"rehearsal/baseline--R{i:04}/01", "model": "jev", "phase": "rehearsal",
             "status": "valid" if i < 2 else "failed", "http_status": 200 if i < 2 else 503,
             "price_usd": "0.001" if i < 2 else None, "latency_seconds": float(i + 1),
             "request_started_at": f"2026-11-01T14:00:0{i}+00:00", "raw_response_sha256": f"raw-{i}"}
            for i in range(3)]


def segment():
    return (Path("serial.json"), {"status": "all_jobs_attempted_with_reviewed_missing_results",
            "started_at": "2026-11-01T14:00:01+00:00", "finished_at": "2026-11-01T14:00:04+00:00",
            "effective_concurrency": 1, "attempt_ids": [rows()[2]["attempt_id"]],
            "quarantined_attempt_ids": [rows()[2]["attempt_id"]]})


def test_smoke_initial_and_serial_segments_preserve_unknown_costs(monkeypatch):
    original = rows()
    untouched = copy.deepcopy(original)
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: row["status"] == "failed")
    bound = ops.bind_segments(original, [segment()], {"raw_response_sha256": "raw-0"}, 4)
    assert original == untouched
    assert [row["effective_concurrency_limit"] for row in bound] == [1, 4, 1]
    assert [row["operational_segment_id"] for row in bound] == ["initial_single_request_smoke", "original_frozen_plan", "serial"]
    summary = ops.summarize(bound)
    failed = next(row for row in summary if row["operational_segment_id"] == "serial")
    assert failed["total_price_usd"] is None and failed["known_price_usd"] == "0"
    assert failed["unpriced_attempts"] == 1 and failed["known_image_processing_503"] == 1
    assert failed["all_attempt_latency"]["p50_seconds"] == 3
    assert failed["completed_response_latency"]["p50_seconds"] is None


@pytest.mark.parametrize("kind", ["overlap", "ongoing", "outside_time", "wrong_quarantine"])
def test_bad_segment_membership_is_rejected(monkeypatch, kind):
    s = segment()
    segments = [s]
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: kind != "wrong_quarantine")
    if kind == "overlap":
        segments.append(copy.deepcopy(s))
    elif kind == "ongoing":
        s[1]["status"] = "running"
    elif kind == "outside_time":
        s[1]["started_at"] = s[1]["finished_at"]
    with pytest.raises(ValueError):
        ops.bind_segments(rows(), segments, {"raw_response_sha256": "raw-0"}, 4)
