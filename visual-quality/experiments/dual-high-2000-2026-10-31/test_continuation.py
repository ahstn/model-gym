"""Offline continuation checks: exact rejected requests only, all-attempt costs, no mutations."""

import importlib.util
import json
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rejected_body(code="hosted_shared_model_concurrency_limit"):
    return {"error": {"type": "rate_limit_error", "code": code, "limit": 60}, "request_id": "req_offline"}


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    base = module("continuation_original_test_fixture", HERE / "test_runner.py")
    fixture = base.trial.__wrapped__(tmp_path, monkeypatch, SimpleNamespace(param=8))
    trial = next(fixture)
    c = module("continuation_under_test", HERE / "continuation.py")
    monkeypatch.setattr(c, "runner", trial.module)
    monkeypatch.setattr(c, "HERE", trial.method)
    monkeypatch.setattr(c, "ORIGINAL", trial.data / "deepseek/images")
    monkeypatch.setattr(c, "DATA", trial.data / "deepseek/continuation-01")
    for name in (
        "continuation.py",
        "test_continuation.py",
        "continuation-protocol.md",
        "provider-recovery-notes.md",
        "provider-recovery-notes.json",
    ):
        (trial.method / name).write_bytes((HERE / name).read_bytes())
    original_post = trial.module.transport.requests.post

    def post(*args, **kwargs):
        response = original_post(*args, **kwargs)
        body = json.loads(response.text)
        if "error" in body:
            response.status_code = 429
            response.headers = {"retry-after": "1", "x-request-id": body["request_id"]}
        return response

    monkeypatch.setattr(trial.module.transport.requests, "post", post)
    trial.module.execute(trial.run, {"T0001"})
    trial.responses.append(rejected_body())
    assert trial.module.execute(trial.run, {"T0002"})["status"] == "blocked"
    original_pins = c.artifact_pins(c.ORIGINAL)
    offset = [3600]
    monkeypatch.setattr(
        c,
        "time",
        SimpleNamespace(
            time=lambda: time.time() + offset[0],
            sleep=lambda seconds: offset.__setitem__(0, offset[0] + seconds),
        ),
    )
    plan = c.freeze()
    yield SimpleNamespace(c=c, trial=trial, run=trial.run, plan=plan, base=base)
    assert c.artifact_pins(c.ORIGINAL) == original_pins
    with pytest.raises(StopIteration):
        next(fixture)


def test_complete_continuation_preserves_original_and_identical_payloads(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    assert plan["retained_response_case_ids"] == ["T0001"]
    assert plan["original_rejected_case_ids"] == ["T0002"]
    assert c.execute(plan, run, {"T0002"})["selected_response_calls"] == 2
    report = c.execute(plan, run)
    assert report["status"] == "complete" and report["original_run_status"] == "blocked"
    assert report["selected_response_calls"] == report["attempted"] == len(report["attempts"]) == 8
    assert report["http_attempts"] == len(t.calls) == len(report["all_attempts"]) == 9
    assert report["unpriced_rejected_attempts"] == 1
    assert Decimal(report["known_cost_usd"]) == Decimal("0.008")
    assert Decimal(report["unpriced_liability_reserve_usd"]) == Decimal("0.1671168")
    assert Decimal(report["admission_debit_usd"]) == Decimal("0.1751168")
    assert c.verify(plan, run) == report
    assert c.execute(plan, run)["http_attempts"] == len(t.calls) == 9
    loaded_run, loaded_report, rows, pins = c.load_verified_route("deepseek")
    assert loaded_run == run and loaded_report == report and len(rows) == 8 and pins
    selected = {a["attempt_id"]: a for a in report["all_attempts"] if a["selected"]}
    for row in rows:
        actual = selected[row["attempt_id"]]
        assert actual["source_run_id"] == row["provenance"]["run_id"] == run["run_id"]
        assert actual["result_path"] and actual["raw_result_sha256"]
    wire_hashes = {case["request_sha256"] for case in run["cases"]}
    assert all(t.module.digest(json.loads(kwargs["data"])) in wire_hashes for _, kwargs in t.calls)


def test_exact_rejection_retried_new_id_and_unknown_cost_remains_reserved(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    t.responses.append(rejected_body())
    report = c.execute(plan, run, {"T0002"})
    assert report["selected_response_calls"] == 2 and report["http_attempts"] == 4
    assert report["unpriced_rejected_attempts"] == 2
    assert Decimal(report["unpriced_liability_reserve_usd"]) == Decimal("0.3342336")
    attempts = [a for a in report["all_attempts"] if a["case_id"] == "T0002"]
    assert [a["sequence"] for a in attempts] == [0, 1, 2]
    assert len({a["attempt_id"] for a in attempts}) == 3
    assert [a["selected"] for a in attempts] == [False, False, True]
    assert all(a["cost_usd"] is None for a in attempts[:2])


@pytest.mark.parametrize("change", ("wrong_code", "usage", "choices", "transport", "request_id"))
def test_only_exact_capacity_envelope_allows_retry(continuation, change):
    c = continuation.c
    result = {
        "http_status": 429,
        "body": json.dumps(rejected_body()),
        "transport_error": None,
        "headers": {"retry-after": "1", "x-request-id": "req_offline"},
    }
    assert c.rejection(result) is not None
    body = json.loads(result["body"])
    if change == "wrong_code":
        body["error"]["code"] = "unrelated_rate_limit"
    elif change == "usage":
        body["usage"] = {"prompt_tokens": 20}
    elif change == "choices":
        body["choices"] = []
    elif change == "transport":
        result["transport_error"] = "ReadTimeout"
    else:
        result["headers"]["x-request-id"] = "req_other"
    result["body"] = json.dumps(body)
    assert c.rejection(result) is None


def test_unknown_continuation_attempt_never_retried(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    case = next(case for case in run["cases"] if case["case_id"] == "T0002")
    attempt_id = case["attempt_id"] + "--continuation-01--001"
    request = {key: case[key] for key in ("case_id", "arm_id", "request_sha256", "request_bytes")}
    request.update(
        attempt_id=attempt_id,
        sequence=1,
        run_id=run["run_id"],
        continuation_id=plan["continuation_id"],
        created_at=t.module.now(),
    )
    t.module.durable_json(c.DATA / "attempts" / attempt_id / "request.json", request)
    with pytest.raises(ValueError, match="Existing uncertainty"):
        c.execute(plan, run)
    assert len(t.calls) == 2


def test_priced_invalid_rating_is_terminal_and_sibling_is_preserved(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    t.responses.append(continuation.base.envelope(invalid_aesthetic=True))
    report = c.execute(plan, run, {"T0002"})
    assert report["invalid_observations"] == 1
    assert report["selected_response_calls"] == 2 and report["http_attempts"] == 3
    assert c.execute(plan, run, {"T0002"})["http_attempts"] == len(t.calls) == 3
    rows = t.module.read_jsonl(c.DATA / "observations.jsonl")
    row = next(row for row in rows if row["case_id"] == "T0002")
    assert row["technical_quality"]["score"] == 1 and row["aesthetics"]["score"] is None


def test_five_retries_limit_keeps_every_rejection(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    t.responses.extend(rejected_body() for _ in range(5))
    report = c.execute(plan, run, {"T0002"})
    assert report["status"] == "blocked" and "per_case_retry_limit" in report["blocking_reasons"]
    assert report["explicit_capacity_rejections"] == 6
    assert report["http_attempts"] == len(t.calls) == 7
    assert c.verify(plan, run) == report


def test_budget_counts_original_cost_and_all_unknown_rejection_reserves(continuation):
    c, t, run, plan = continuation.c, continuation.trial, continuation.run, continuation.plan
    plan["budget"]["max_cost_usd"] = "0.1681168"
    with pytest.raises(ValueError, match="combined admission budget"):
        c.execute(plan, run, {"T0002"})
    assert len(t.calls) == 2
