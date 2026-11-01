"""A missing response is never converted into a known bill or replayable request."""

import json

import interruption_resume as recovery
import pytest


@pytest.mark.parametrize("change", [None, "reply_exists", "journal", "request_hash", "priced", "measured", "new_case"])
def test_interruption_exception_requires_exact_unanswered_unmeasured_journal(tmp_path, change):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"request_sha256": "request"}))
    row = {"attempt_id": "rehearsal/baseline--R1234/01", "status": "unknown", "reason": "interrupted_request",
           "price_usd": None, "latency_seconds": None, "raw_response_path": None,
           "request_journal_sha256": recovery.contract.file_sha(path), "request_sha256": "request"}
    audited = {row["attempt_id"]: {**row, "request_journal_path": str(path)}}
    if change == "reply_exists": (tmp_path / "result.json").write_text('{}')
    elif change == "journal": path.write_text(path.read_text() + "\n")
    elif change == "request_hash": row["request_sha256"] = "changed"
    elif change == "priced": row["price_usd"] = "0.00"
    elif change == "measured": row["latency_seconds"] = 1.0
    elif change == "new_case": row["attempt_id"] = "unreviewed"
    assert recovery.approved_interruption(row, audited) is (change is None)
