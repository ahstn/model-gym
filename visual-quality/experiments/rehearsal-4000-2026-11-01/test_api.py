"""Contract, accounting and no-duplicate-request regression tests; no live calls."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import api_contract as c
import api_runner as r
import pytest

HERE = Path(__file__).resolve().parent
ROUTES = json.loads((HERE / "api-routes.json").read_text())


def joint():
    return {
        t: {
            "task": t,
            "status": "ok",
            "score": "3.20",
            "evidence": "Visible detail is adequately preserved.",
            "limitations": [],
        }
        for t in c.TASKS
    }


def envelope(route="luna", content=None):
    body = {
        "model": ROUTES[route]["accepted_models"][0],
        "provider": "OpenAI",
        "usage": {"cost": 0.001, "is_byok": False},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content or joint())}}],
    }
    if route == "deepseek":
        body["usage"]["runinfra"] = {"cost_microcents": 100000}
    return body


def result(body, status=200):
    return {"body": json.dumps(body), "http_status": status, "headers": {}, "transport_error": None}


def small_plan(tmp_path, route="luna"):
    image = tmp_path / "image.png"
    image.write_bytes(b"test-only-image-bytes")
    case = {
        "case_id": "R1001",
        "image_path": str(image),
        "view_sha256": c.file_sha(image),
        "family_id": "test-family",
        "benchmark_partition": "rehearsal",
    }
    job = {
        "case": case,
        "arm": "baseline",
        "job_id": "baseline--R1001",
        "stats_case_id": None,
        "stats_text": None,
    }
    plan = {
        "schema": r.SCHEMA,
        "route": route,
        "phase": "rehearsal",
        "route_config": copy.deepcopy(ROUTES[route]),
        "teacher_prompt": json.loads((HERE / "api-teacher-prompt.json").read_text()),
        "decision_prompt": json.loads((HERE / "api-decision-prompts.json").read_text()),
        "reserve_per_attempt_usd": ".23",
        "jobs": [job],
        "inputs": {},
        "runtime": r.runtime(),
    }
    wire = c.encoded(c.payload(plan, job))
    job.update(request_sha256=hashlib.sha256(wire).hexdigest(), request_bytes=len(wire))
    plan["plan_id"] = c.digest(plan)
    return plan, job


def test_baseline_preserves_selected_prompt_and_scope(tmp_path):
    plan, job = small_plan(tmp_path)
    payload = c.payload(plan, job)
    assert payload["reasoning"] == {"effort": "high", "exclude": True}
    assert payload["provider"]["allow_fallbacks"] is False
    assert payload["messages"][0]["content"] == plan["teacher_prompt"]["system"]
    assert payload["messages"][1]["content"][0]["text"] == plan["teacher_prompt"]["user"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "case_id" not in str(payload)
    own = {**job, "arm": "own_stats", "stats_text": "Detail in the sharpest region: low"}
    swapped = {**own, "arm": "swapped_stats", "stats_case_id": "R1002"}
    assert c.payload(plan, own) == c.payload(plan, swapped)
    assert c.payload(plan, own)["messages"][1]["content"][1] == payload["messages"][1]["content"][1]
    assert "TECHNICAL QUALITY CHECK ONLY" in c.payload(plan, own)["messages"][1]["content"][0]["text"]


@pytest.mark.parametrize("route", ["jev", "liquid"])
def test_decisions_uses_previous_typed_contract(tmp_path, route):
    plan, job = small_plan(tmp_path, route)
    payload = c.payload(plan, job)
    assert set(payload) == {"model", "state", "questions"}
    assert payload["state"][0] == plan["decision_prompt"]["common"]
    assert (
        payload["questions"]
        == plan["decision_prompt"]["prompts"][ROUTES[route]["requested_model"]]["questions"]
    )


def test_wire_valid_sibling_survives_invalid_score():
    content = joint()
    content["aesthetics"]["score"] = 3.2
    row = c.interpret(result(envelope(content=content)), ROUTES["luna"])
    assert row["status"] == "invalid"
    assert row["technical_score"] == 3.2 and row["aesthetic_score"] is None
    assert row["task_valid"] == {"technical_quality": True, "aesthetics": False}


def test_unscorable_is_null_and_not_zero():
    content = joint()
    content["technical_quality"].update(status="unscorable", score=None)
    row = c.interpret(result(envelope(content=content)), ROUTES["luna"])
    assert row["status"] == "valid" and row["technical_score"] is None
    assert row["task_valid"]["technical_quality"] is True


def test_duplicate_json_and_wrong_model_are_rejected():
    duplicate = result(envelope())
    duplicate["body"] = '{"model":"allowed", "model":"other"}'
    assert c.interpret(duplicate, ROUTES["luna"])["status"] == "failed"
    wrong = envelope()
    wrong["model"] = "openai/other"
    assert c.interpret(result(wrong), ROUTES["luna"])["status"] == "failed"


def test_runinfra_reasoning_header_mismatch_fails():
    response = result(envelope("deepseek"))
    response["headers"]["x-runinfra-reasoning-effort"] = "none"
    assert c.interpret(response, ROUTES["deepseek"])["status"] == "failed"


def test_historical_decision_outputs_reproduce_without_rounding_scores():
    source = HERE.parent / "photo-scoring-2026-10-31/selected-responses-72.jsonl"
    checked = Counter()
    for line in source.read_text().splitlines():
        historical = json.loads(line)
        route = {"typesafe/jev-1.14": "jev", "liquid/d1.1": "liquid"}.get(historical["model_requested"])
        if not route:
            continue
        row = c.interpret(result(historical["response"]), ROUTES[route])
        assert row["status"] == "valid", (historical["case_id"], row)
        for task, output in [("technical_quality", "technical"), ("aesthetics", "aesthetic")]:
            assert row[output + "_score"] == pytest.approx(historical["normalized"][task], abs=1e-12)
            assert row[output + "_score_native"] == pytest.approx(
                historical["normalized"][task + "_native"], abs=1e-12
            )
        checked[route] += 1
    assert checked == {"jev": 72, "liquid": 72}


def test_rounded_distribution_bound_is_not_an_unlimited_tolerance():
    answer = {"score": 1.02, "probabilities": {"0": 0.32, "1": 0.33, "2": 0.34, "3": 0, "4": 0}}
    assert c.distribution(answer, allow_rounded=True)
    with pytest.raises(ValueError):
        c.distribution(answer, allow_rounded=False)
    answer["score"] = 1.2
    with pytest.raises(ValueError):
        c.distribution(answer, allow_rounded=True)


def test_cost_accounting_preserves_unknown_and_byok():
    assert c.reported_cost({}, ROUTES["luna"])["price_usd"] is None
    body = {"usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.0012}}}
    assert c.reported_cost(body, ROUTES["luna"])["price_usd"] == "0.0012"
    body["usage"]["cost"] = 0.00001
    assert c.reported_cost(body, ROUTES["luna"])["price_usd"] == "0.00121"
    del body["usage"]["cost_details"]
    assert c.reported_cost(body, ROUTES["luna"])["price_usd"] is None
    body = {"usage": {"cost": 0.001, "runinfra": {"cost_microcents": 100000}}}
    assert c.reported_cost(body, ROUTES["deepseek"])["price_usd"] == "0.001"
    body["usage"]["runinfra"]["cost_microcents"] = 999
    assert c.reported_cost(body, ROUTES["deepseek"])["price_usd"] is None


def test_only_exact_preserve_capacity_error_is_retryable():
    body = {
        "error": {"type": "rate_limit_error", "code": "hosted_shared_model_concurrency_limit"},
        "request_id": "req_test",
    }
    response = result(body, 429)
    response["headers"] = {"retry-after": "2", "x-request-id": "req_test"}
    row = c.interpret(response, ROUTES["deepseek"])
    assert row["status"] == "rejected_retryable" and row["price_usd"] is None
    assert r.liability([row], Decimal(".17")) == Decimal(".17")
    for field, value in [("transport_error", "ReadTimeout"), ("http_status", 503)]:
        bad = {**response, field: value}
        assert c.interpret(bad, ROUTES["deepseek"])["status"] != "rejected_retryable"
    assert c.interpret(response, ROUTES["luna"])["status"] == "failed"


def test_send_measures_http_interval_and_redacts_reflected_secret(tmp_path, monkeypatch):
    plan, job = small_plan(tmp_path)
    monkeypatch.setattr(r, "DATA", tmp_path)
    times = iter([10.0, 11.0, 13.75, 14.0])
    monkeypatch.setattr(r.time, "perf_counter", lambda: next(times))
    body = envelope()
    body["reflected"] = "secret-test-key"

    response = SimpleNamespace(
        status_code=200,
        text=json.dumps(body),
        headers={"x-request-id": "safe", "Authorization": "secret-test-key"},
    )

    sent = []

    def post(*args, **kwargs):
        sent.append((args, kwargs))
        return response

    monkeypatch.setattr(r.requests, "post", post)
    row = r.send(plan, job, 1, "secret-test-key", 9.0)
    assert row["latency_seconds"] == 2.75 and row["scheduler_wait_seconds"] == 1.0
    assert row["preparation_seconds"] == 1.0
    assert row["status"] == "valid"
    raw = Path(row["raw_response_path"]).read_text()
    assert "secret-test-key" not in raw and "Authorization" not in raw
    assert sent[0][1]["allow_redirects"] is False
    assert r.reconstruct(plan) == [row]
    with pytest.raises(FileExistsError):
        r.send(plan, job, 1, "secret-test-key", 9)


def test_interrupted_attempt_blocks_resend(tmp_path, monkeypatch):
    plan, job = small_plan(tmp_path)
    monkeypatch.setattr(r, "DATA", tmp_path)
    folder = r.directory("luna", "rehearsal") / "attempts" / job["job_id"] / "01"
    r.write(
        folder / "request.json",
        {
            "plan_id": plan["plan_id"],
            "job_id": job["job_id"],
            "sequence": 1,
            "request_sha256": job["request_sha256"],
            "request_bytes": job["request_bytes"],
        },
        exclusive=True,
    )
    records = r.reconstruct(plan)
    assert records[0]["status"] == "unknown" and records[0]["latency_seconds"] is None
    with pytest.raises(ValueError, match="Unknown/nonretryable"):
        r.execute(plan)


def test_no_automatic_retry_of_valid_or_invalid_response(tmp_path, monkeypatch):
    plan, job = small_plan(tmp_path)
    monkeypatch.setattr(r, "DATA", tmp_path)
    parent = r.directory("luna", "rehearsal") / "attempts" / job["job_id"]
    for sequence in (1, 2):
        folder = parent / f"{sequence:02d}"
        r.write(
            folder / "request.json",
            {
                "plan_id": plan["plan_id"],
                "job_id": job["job_id"],
                "sequence": sequence,
                "request_sha256": job["request_sha256"],
                "request_bytes": job["request_bytes"],
            },
            exclusive=True,
        )
        r.write(folder / "result.json", result(envelope()), exclusive=True)
    with pytest.raises(ValueError, match="Unexpected retry"):
        r.reconstruct(plan)


def test_stats_donor_cannot_be_same_family_and_focus_line_is_forbidden(tmp_path):
    cases = [
        {"case_id": "a", "family_id": "one", "benchmark_partition": "development"},
        {"case_id": "b", "family_id": "two", "benchmark_partition": "rehearsal"},
    ]
    stats = tmp_path / "stats.jsonl"
    stats.write_text("\n".join(json.dumps({"case_id": x, "text": "Sharpness: low"}) for x in ("a", "b")))
    donors = tmp_path / "donors.json"
    donors.write_text('{"a":"b"}')
    assert r.stats_inputs(cases, stats, donors)[1] == {"a": "b"}
    donors.write_text('{"a":"a"}')
    with pytest.raises(ValueError):
        r.stats_inputs(cases, stats, donors)
    donors.write_text('{"a":"b"}')
    stats.write_text(stats.read_text().replace("Sharpness: low", "Frame area close to peak detail: 89%"))
    with pytest.raises(ValueError):
        r.stats_inputs(cases, stats, donors)


def test_shared_budget_accounts_for_other_phase_and_unknown_receipts(tmp_path, monkeypatch):
    plan, job = small_plan(tmp_path)
    plan["route_config"]["budget_usd"] = ".30"
    sent = []
    monkeypatch.setattr(r, "send", lambda *args: sent.append(args))
    with pytest.raises(ValueError, match="Budget admission limit"):
        r._stream(plan, deque([job]), "key", [], [{"price_usd": ".10"}], {})
    assert not sent
    with pytest.raises(ValueError, match="Budget admission limit"):
        r._stream(plan, deque([job]), "key", [], [{"price_usd": None}], {})
    assert not sent


@pytest.mark.parametrize("rejections,expected_sequences", [(4, [5]), (5, [])])
def test_capacity_rejection_bound_does_not_dispatch_a_sixth_attempt(
    tmp_path, monkeypatch, rejections, expected_sequences
):
    plan, job = small_plan(tmp_path, route="deepseek")
    sent = []

    def send(plan, current, sequence, secret, queued_at):
        sent.append(sequence)
        return {"job_id": current["job_id"], "sequence": sequence, "status": "valid", "price_usd": ".001"}

    monkeypatch.setattr(r, "send", send)
    previous = {
        "job_id": job["job_id"],
        "sequence": rejections,
        "status": "rejected_retryable",
        "price_usd": "0",
    }
    r._stream(plan, deque([job]), "synthetic-key", [previous], [], {job["job_id"]: previous})
    assert sent == expected_sequences


def test_successful_resume_sends_nothing_and_preserves_observation(tmp_path, monkeypatch):
    plan, job = small_plan(tmp_path)
    monkeypatch.setattr(r, "DATA", tmp_path)
    folder = r.directory("luna", "rehearsal") / "attempts" / job["job_id"] / "01"
    r.write(
        folder / "request.json",
        {
            "plan_id": plan["plan_id"],
            "job_id": job["job_id"],
            "sequence": 1,
            "request_sha256": job["request_sha256"],
            "request_bytes": job["request_bytes"],
        },
        exclusive=True,
    )
    r.write(folder / "result.json", result(envelope()), exclusive=True)
    monkeypatch.setattr(r, "send", lambda *args: pytest.fail("Completed image was re-requested"))
    assert r.execute(plan)["selected_responses"] == 1
    assert r.execute(plan)["http_attempts"] == 1


def test_reference_freeze_requires_exact_manifest_and_all_500_cases(tmp_path):
    manifest = {
        "cases": [{"case_id": f"R{i:04}", "benchmark_partition": "representative"} for i in range(500)]
    }
    manifest_path = tmp_path / "manifest.json"
    references = tmp_path / "references.jsonl"
    freeze = tmp_path / "freeze.json"
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("Frozen reference rubric")
    r.write(manifest_path, manifest)
    references.write_text("\n".join(json.dumps({"case_id": row["case_id"]}) for row in manifest["cases"]))
    frozen = {
        "references_path": str(references),
        "references_sha256": c.file_sha(references),
        "manifest_sha256": c.file_sha(manifest_path),
        "case_ids": [row["case_id"] for row in manifest["cases"]],
        "frozen_at": "2026-01-01T00:00:00+00:00",
        "rubric_path": str(rubric),
        "rubric_sha256": c.file_sha(rubric),
        "teacher_prompt_sha256": c.file_sha(HERE / "api-teacher-prompt.json"),
    }
    r.write(freeze, frozen)
    assert len(r.frozen_references(manifest_path, manifest, freeze)) == 3
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="500 matching"):
        r.frozen_references(manifest_path, manifest, freeze)
    r.write(manifest_path, manifest)
    frozen["case_ids"][-1] = "different"
    r.write(freeze, frozen)
    with pytest.raises(ValueError, match="500 matching"):
        r.frozen_references(manifest_path, manifest, freeze)


def test_plan_refuses_modified_inputs_or_runtime(tmp_path):
    plan, _ = small_plan(tmp_path)
    r.validate_plan(plan, check_payloads=True)
    plan["runtime"]["requests"] = "different"
    plan["plan_id"] = c.digest({k: v for k, v in plan.items() if k != "plan_id"})
    with pytest.raises(ValueError, match="runtime changed"):
        r.validate_plan(plan)
