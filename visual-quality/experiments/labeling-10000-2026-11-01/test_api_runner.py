"""Protect request identity, retry semantics, scope validation and cost accounting."""

import hashlib
import json
import threading
from types import SimpleNamespace

import api_runner as runner
import pytest


def chat_body(scores=("3.00", "4.00"), unsupported=(False, False)):
    tasks = {}
    for task, score, excluded in zip(runner.contract.TASKS, scores, unsupported, strict=True):
        tasks[task] = {
            "task": task,
            "status": "ok",
            "score": score,
            "evidence": "Visible subject detail is clear.",
            "limitations": ["unsupported_domain"] if excluded else [],
        }
    return {
        "model": "openai/gpt-6-luna",
        "provider": "OpenAI",
        "usage": {"cost": 0.0008},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(tasks)}}],
    }


@pytest.fixture
def profile():
    return runner.load_profile(runner.PACKAGE / "labeling")


def test_v2_profile_keeps_dual_schema_and_photo_precedence(profile):
    text = profile["teacher"]["user"]
    assert "Strong noise can damage detail while object boundaries stay sharp." in text
    assert "The survival of broad colors or outlines alone does not make an image fair." in text
    assert "Judge graphics and paintings within their visible medium" not in text
    assert "PHOTOGRAPHY ELIGIBILITY RULE" in text and "AESTHETICS RUBRIC" in text
    assert "score to a JSON string with exactly two decimal places" in text
    assert profile["teacher"]["version"] == "photo-labeling-v2-2026-11-01"
    assert set(profile["teacher"]["response_format"]["json_schema"]["schema"]["properties"]) == set(
        runner.contract.TASKS
    )


@pytest.mark.parametrize(
    "http,transport,retry",
    [
        (503, None, True),
        (429, None, True),
        (None, "ReadTimeout", True),
        (400, None, False),
        (200, None, False),
    ],
)
def test_retries_depend_on_transport_not_score(http, transport, retry, profile):
    result = {"http_status": http, "transport_error": transport, "body": "{}", "headers": {}}
    assert runner.parsed_result(result, profile["routes"]["routes"]["luna"])["retryable"] is retry


def test_inconsistent_chat_scope_floor_is_invalid(profile):
    result = {
        "http_status": 200,
        "body": json.dumps(chat_body(("1.00", "4.00"), (True, False))),
        "headers": {},
    }
    parsed = runner.parsed_result(result, profile["routes"]["routes"]["luna"])
    assert parsed["status"] == "invalid"
    assert parsed["technical_score"] is None and parsed["aesthetic_score"] is None
    assert not parsed["retryable"]


def test_valid_two_score_floor_is_retained(profile):
    result = {
        "http_status": 200,
        "body": json.dumps(chat_body(("1.00", "1.00"), (True, True))),
        "headers": {},
    }
    parsed = runner.parsed_result(result, profile["routes"]["routes"]["luna"])
    assert parsed["status"] == "valid" and parsed["scope_floor_applied"]
    assert parsed["price_usd"] == "0.0008"


def test_reference_admission_checks_exact_image(tmp_path):
    refs = tmp_path / "references.jsonl"
    refs.write_text(json.dumps({"case_id": "R1", "view_sha256": "abc"}) + "\n")
    freeze = tmp_path / "freeze.json"
    runner.atomic(
        freeze,
        {
            "references_path": str(refs),
            "references_sha256": runner.sha(refs),
            "frozen_at": "2026-01-01T00:00:00+00:00",
        },
    )
    assert runner.reference_pins([{"case_id": "R1", "view_sha256": "abc"}], freeze)
    with pytest.raises(ValueError, match="mismatched"):
        runner.reference_pins([{"case_id": "R1", "view_sha256": "def"}], freeze)
    refs.write_text(refs.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        runner.reference_pins([], freeze)


def test_unknown_charge_is_reserved_not_zero(tmp_path):
    request = tmp_path / "attempts/R1/01/request.json"
    runner.atomic(request, {})
    ledger = runner.Ledger(tmp_path, {"reserve_usd": "0.25", "budget_usd": "0.50"})
    assert ledger.summary()["known_price_usd"] == "0"
    assert ledger.summary()["unpriced_attempts"] == 1
    ledger.acquire()
    with pytest.raises(RuntimeError, match="budget"):
        ledger.acquire()
    ledger.settle("0.01")
    assert ledger.summary()["admission_liability_usd"] == "0.26"


def test_resume_preserves_valid_response_and_prevents_changed_image(tmp_path, monkeypatch, profile):
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ntest fixture")
    case = {
        "case_id": "R1",
        "view_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "image_path": str(image),
    }
    plan = {"model": "luna", "profile": profile, "policy": runner.POLICY["luna"], "plan_id": "frozen"}
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, text=json.dumps(chat_body()), headers={})

    monkeypatch.setattr(runner.requests, "post", post)
    ledger = runner.Ledger(tmp_path, plan["policy"])
    row = runner.job(plan, case, tmp_path, "test-secret", ledger, threading.Event())
    assert row["technical_score"] == 3 and row["aesthetic_score"] == 4
    assert row["latency_seconds"] >= 0 and row["price_usd"] == "0.0008"
    assert row["total_attempt_price_usd"] == "0.0008"
    assert row["technical_score_valid"] and row["aesthetic_score_valid"]
    repeated = runner.job(plan, case, tmp_path, "test-secret", ledger, threading.Event())
    assert repeated == row and len(calls) == 1
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.job(plan, case, tmp_path, "test-secret", ledger, threading.Event())


def test_decisions_scope_probability_keeps_native_score(profile):
    route = profile["routes"]["routes"]["liquid"]
    body = {
        "model": "liquid/d1.1-20261027",
        "provider": "Liquid",
        "usage": {"cost": 0.00006},
        "answers": {
            "excluded": {"type": "noul", "noul": 0.8},
            "technical_quality": {"type": "score", "score": 2.12345},
            "aesthetics": {"type": "score", "score": 3.23456},
        },
    }
    parsed = runner.parsed_result({"http_status": 200, "body": json.dumps(body), "headers": {}}, route)
    assert parsed["status"] == "valid"
    assert parsed["technical_score"] == parsed["aesthetic_score"] == 1.0
    assert parsed["technical_score_native"] == 3.12345


def test_no_statistical_inputs_in_payload(profile):
    for model in runner.MODELS:
        payload = runner.request_payload(profile, model, b"\x89PNG\r\n\x1a\ntest fixture")
        text = json.dumps(payload)
        assert "sharpness_peak" not in text and "luma_entropy" not in text


def test_transient_retry_preserves_unpriced_attempt_and_stops_after_valid(tmp_path, monkeypatch, profile):
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nretry fixture")
    case = {"case_id": "R2", "view_sha256": runner.sha(image), "image_path": str(image)}
    plan = {"model": "luna", "profile": profile, "policy": runner.POLICY["luna"], "plan_id": "frozen"}
    responses = iter(
        [
            SimpleNamespace(status_code=503, text='{"error":{"message":"Unavailable"}}', headers={}),
            SimpleNamespace(status_code=200, text=json.dumps(chat_body()), headers={}),
        ]
    )
    monkeypatch.setattr(runner.requests, "post", lambda *a, **kw: next(responses))
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda delay: None)
    row = runner.job(plan, case, tmp_path, "test-secret", runner.Ledger(tmp_path, plan["policy"]), stop)
    assert row["status"] == "valid" and row["attempt_count"] == 2
    assert row["unpriced_attempts"] == 1 and row["total_attempt_price_usd"] is None
    assert row["total_attempt_known_price_usd"] == "0.0008"
