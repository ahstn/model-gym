"""Only billed no-answer exhaustion may bypass original fatal admission."""

import json
from decimal import Decimal

import deepseek_recovery as recovery
import pytest


def fixture(tmp_path):
    body = {"model": "deepseek-v4-1-flash-1017", "choices": [{"finish_reason": "length",
            "message": {"content": None}, "runinfra": {"output_status": {"code": "generation_limit_reached_before_answer"}}}],
            "usage": {"completion_tokens": 16384, "completion_tokens_details": {"reasoning_tokens": 16384},
                      "cost": 0.0099, "runinfra": {"cost_microcents": 990000, "output_token_accounting": {"visible_answer_tokens": 0}}}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"body": json.dumps(body), "transport_error": None}))
    row = {"model": "deepseek", "status": "failed", "http_status": 200, "reason": "Incomplete completion",
           "price_usd": "0.0099", "raw_response_path": str(path), "raw_response_sha256": recovery.contract.file_sha(path),
           "attempt_id": "rehearsal/baseline--R1830/01", "request_sha256": "request", "request_journal_sha256": "journal",
           "request_started_at": "2026-11-01T13:00:00+00:00"}
    return row, body, path


@pytest.mark.parametrize("change", [None, "content", "finish", "model", "tokens", "price", "raw"])
def test_only_exact_billed_no_answer_limit_is_recognized(tmp_path, change):
    row, body, path = fixture(tmp_path)
    if change == "content": body["choices"][0]["message"]["content"] = '{"score":3}'
    elif change == "finish": body["choices"][0]["finish_reason"] = "stop"
    elif change == "model": body["model"] = "another-model"
    elif change == "tokens": body["usage"]["completion_tokens"] = 100
    elif change == "price": row["price_usd"] = None
    if change in {"content", "finish", "model", "tokens"}:
        path.write_text(json.dumps({"body": json.dumps(body), "transport_error": None}))
        row["raw_response_sha256"] = recovery.contract.file_sha(path)
    if change == "raw": path.write_text(path.read_text() + "\n")
    assert recovery.no_answer_exhaustion(row) is (change is None)


def test_admission_preserves_other_failures_and_limits_unique_exhaustions(monkeypatch):
    monkeypatch.setattr(recovery, "no_answer_exhaustion", lambda row: row.get("known", False))
    approval = {"reviewed_attempts": [], "created_at": "2026-11-01T14:00:00+00:00", "maximum_total_exhaustions": 10}
    gate = recovery.ExhaustionAdmission(lambda row, reserve: row["fatal"], approval)
    assert gate({"known": False, "fatal": True}, Decimal(1))
    assert not gate({"known": False, "fatal": False}, Decimal(1))
    row = {"known": True, "fatal": True, "price_usd": "0.01", "request_started_at": "2026-11-01T15:00:00+00:00"}
    for i in range(9):
        assert not gate({**row, "attempt_id": str(i)}, Decimal(1))
    assert not gate({**row, "attempt_id": "8"}, Decimal(1))
    assert len(gate.seen) == 9
    assert gate({**row, "attempt_id": "9"}, Decimal(1))
    assert gate({**row, "attempt_id": "old", "request_started_at": "2026-11-01T13:00:00+00:00"}, Decimal(1))


def test_old_audited_exception_is_receipt_bound_and_budgeted(monkeypatch):
    monkeypatch.setattr(recovery, "no_answer_exhaustion", lambda row: True)
    row = {"attempt_id": "old", "price_usd": "0.01", "raw_response_sha256": "raw", "request_sha256": "req",
           "request_journal_sha256": "journal", "request_started_at": "2026-11-01T13:00:00+00:00"}
    approval = {"reviewed_attempts": [row], "created_at": "2026-11-01T14:00:00+00:00", "maximum_total_exhaustions": 10}
    gate = recovery.ExhaustionAdmission(lambda row, reserve: True, approval)
    assert not gate(row, Decimal(1))
    assert gate({**row, "request_sha256": "changed"}, Decimal(1))
    assert gate(row, Decimal("0.001"))
