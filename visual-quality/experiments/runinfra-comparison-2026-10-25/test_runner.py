"""Offline request, cost and recovery checks; all provider calls are mocked."""

import base64
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path(__file__).with_name("runner.py")
KEY = "offline-secret-never-persist"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jr1sAAAAASUVORK5CYII="
)


def envelope(model, *, amount="0.001", score=3.2):
    rating = {"task": "technical_quality", "status": "ok", "score": score,
              "evidence": "Visible fine detail is slightly soft.", "limitations": []}
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(rating)}}], "usage": {
            "cost": amount, "runinfra": {"cost_microcents": int(Decimal(amount) * 100_000_000)}}}


@pytest.fixture
def study(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("runinfra_runner_offline_test", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "DATA", tmp_path / "data")
    monkeypatch.setattr(runner, "VIEWS", tmp_path / "views")
    monkeypatch.setenv("RUN_INFRA_API_KEY", KEY)
    runner.DATA.mkdir()
    runner.VIEWS.mkdir()
    image = runner.VIEWS / "frozen.png"
    image.write_bytes(PNG)
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("immutable input")
    calls, responses = [], []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        # A missing fixture response fails closed. No real HTTP function is used.
        assert responses, "Unexpected provider request"
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(runner.requests, "post", post)

    def response(body=None, status=200, headers=None):
        return SimpleNamespace(status_code=status, headers=headers or {},
                               text=json.dumps(body if body is not None else envelope(runner.MODEL)))

    def make_run(eligible=2):
        run = {"prompt": {"system": "Grade technical quality.", "user": "Rate this image.",
                          "response_format": {"type": "json_object"},
                          "version": "test-v1", "sha256": "1" * 64},
               "settings": {"model": runner.MODEL, "max_tokens": 1024,
                            "temperature": 0, "reasoning_effort": "none", "stream": False},
               "budget": {"max_cost_usd": "1.00", "reserve_per_request_usd": "0.1579008"},
               "inputs": {"pinned.txt": runner.file_sha256(pinned)}, "cases": []}
        for i in range(50):
            case = {"case_id": f"case-{i:02d}", "family_id": f"family-{i // 2}",
                    "view": {"image_id": f"image-{i:02d}", "view_id": f"view-{i:02d}",
                             "image_path": image.name, "source_sha256": "2" * 64,
                             "view_sha256": runner.file_sha256(image),
                             "photo_domain_eligible": [True, False, None][i % 3]}}
            body = runner.payload(run, case)
            case.update(request_sha256=runner.digest(body), request_bytes=len(runner.encoded(body)),
                        skip_reason=None if i < eligible else "request_exceeds_provider_limit")
            run["cases"].append(case)
        run["run_id"] = runner.digest(run)
        runner.atomic_json(runner.DATA / "run.json", run)
        return run

    return SimpleNamespace(runner=runner, calls=calls, responses=responses, response=response,
                           make_run=make_run, image=image, pinned=pinned)


def test_success_preserves_views_and_reuses_complete_run(study):
    r, run = study.runner, study.make_run()
    study.responses.extend([study.response(), study.response(envelope(r.SLUG))])
    report = r.execute(r.read_run())
    assert report["status"] == "complete_with_size_exclusions"
    assert report["attempted"] == report["scored"] == 2
    assert Decimal(report["known_cost_usd"]) == Decimal("0.002")
    assert len(report["blocked_case_ids"]) == 48
    assert report["training_ready"] is False
    assert study.image.read_bytes() == PNG
    for _, kwargs in study.calls:
        body = json.loads(kwargs["data"])
        data_url = body["messages"][1]["content"][1]["image_url"]["url"]
        assert base64.b64decode(data_url.split(",", 1)[1]) == PNG
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"]["Authorization"] == "Bearer " + KEY
    observations = r.read_jsonl(r.DATA / "observations.jsonl")
    assert [x["photo_domain_eligible"] for x in observations] == [True, False]
    assert all(x["aesthetics"]["score"] is None for x in observations)
    assert all(all(value is False for value in x["task_masks"].values()) for x in observations)
    assert all(x["review_status"] == "pending_human_review" for x in observations)
    assert r.verify(run) == report
    assert r.execute(run) == report
    assert len(study.calls) == 2


def test_transport_timeout_persists_uncertain_cost_and_blocks_resume(study):
    r, run = study.runner, study.make_run()
    study.responses.append(r.requests.Timeout(f"Connection failed with secret {KEY}"))
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["unpriced_attempts"] == 1
    assert report["attempts"][0]["reason"] == "transport_outcome_unknown"
    assert r.read_jsonl(r.DATA / "observations.jsonl") == []
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(study.calls) == 1
    assert KEY not in (r.DATA / "attempts/case-00/result.json").read_text()
    assert r.verify(run) == report


@pytest.mark.parametrize("failure", ["http", "score", "model", "incomplete", "refusal", "null_message"])
def test_billed_failure_keeps_cost_stops_and_cannot_retry(study, failure):
    r, run = study.runner, study.make_run()
    body = envelope(r.MODEL)
    status = 200
    if failure == "http":
        status = 429
        body["error"] = {"message": "Unavailable"}
    elif failure == "score":
        body = envelope(r.MODEL, score=5.1)
    elif failure == "model":
        body["model"] = "different-model"
    elif failure == "incomplete":
        body["choices"][0]["finish_reason"] = "length"
    elif failure == "refusal":
        body["choices"][0]["message"]["refusal"] = "Cannot rate"
    else:
        body["choices"][0]["message"] = None
    study.responses.append(study.response(body, status=status))
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["valid_observations"] == 0
    assert Decimal(report["known_cost_usd"]) == Decimal("0.001")
    assert report["unpriced_attempts"] == 0
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(study.calls) == 1
    assert r.verify(run) == report


@pytest.mark.parametrize("usage", [{}, {"cost": True}, {"cost": "NaN"},
                                   {"cost": "0.001", "runinfra": {"cost_microcents": 1}},
                                   {"cost": "0.001", "runinfra": None}])
def test_unknown_or_inconsistent_cost_preserves_score_and_stops(study, usage):
    r, run = study.runner, study.make_run()
    body = envelope(r.MODEL)
    body["usage"] = usage
    study.responses.append(study.response(body))
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["scored"] == 1
    assert report["unpriced_attempts"] == 1
    assert report["attempts"][0]["cost_usd"] is None
    assert report["attempts"][0]["reason"] == "cost_unavailable"
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(study.calls) == 1


def test_interruption_after_request_journal_blocks_without_call(study):
    r, run = study.runner, study.make_run()
    case = run["cases"][0]
    folder = r.DATA / "attempts" / case["case_id"]
    folder.mkdir(parents=True)
    request = {k: case[k] for k in ("case_id", "request_sha256", "request_bytes")}
    request.update(run_id=run["run_id"], created_at=r.now())
    r.atomic_json(folder / "request.json", request)
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    report = r.verify(run)
    assert report["attempts"][0]["reason"] == "interrupted_request"
    assert study.calls == []


def test_interruption_before_request_journal_blocks_without_call(study):
    r, run = study.runner, study.make_run()
    (r.DATA / "attempts/case-00").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete attempt directory"):
        r.execute(run)
    assert study.calls == []


def test_budget_uses_billed_spend_before_next_request(study):
    r, run = study.runner, study.make_run()
    study.responses.append(study.response(envelope(r.MODEL, amount="0.90")))
    with pytest.raises(ValueError, match="admission budget"):
        r.execute(run)
    report = r.verify(run)
    assert report["attempted"] == report["scored"] == 1
    assert Decimal(report["known_cost_usd"]) == Decimal("0.90")
    with pytest.raises(ValueError, match="admission budget"):
        r.execute(run)
    assert len(study.calls) == 1


def test_secret_echo_is_redacted_from_saved_body_and_headers(study):
    r, run = study.runner, study.make_run(eligible=1)
    body = envelope(r.MODEL)
    body["provider_debug"] = KEY
    study.responses.append(study.response(body, headers={"X-Request-ID": KEY, "X-Unrelated": KEY}))
    r.execute(run)
    for path in r.DATA.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text()
    result = json.loads((r.DATA / "attempts/case-00/result.json").read_text())
    assert result["headers"] == {"x-request-id": "[REDACTED]"}
    assert json.loads(result["body"])["provider_debug"] == "[REDACTED]"


@pytest.mark.parametrize("changed", ["view", "input", "metadata"])
def test_modified_frozen_inputs_fail_before_any_call(study, changed):
    r = study.runner
    study.make_run()
    if changed == "view":
        study.image.write_bytes(b"changed bytes")
    elif changed == "input":
        study.pinned.write_text("changed input")
    else:
        run = json.loads((r.DATA / "run.json").read_text())
        run["settings"]["max_tokens"] = 2048
        r.atomic_json(r.DATA / "run.json", run)
    with pytest.raises(ValueError, match="changed"):
        r.execute(r.read_run())
    assert study.calls == []


def test_unscorable_is_retained_as_null_and_never_promoted(study):
    r, run = study.runner, study.make_run(eligible=1)
    body = envelope(r.MODEL)
    body["choices"][0]["message"]["content"] = json.dumps({
        "task": "technical_quality", "status": "unscorable", "score": None,
        "evidence": "The view does not contain enough visible detail.", "limitations": ["limited_detail"]})
    study.responses.append(study.response(body))
    report = r.execute(run)
    assert report["scored"] == 0
    observation = r.read_jsonl(r.DATA / "observations.jsonl")[0]
    assert observation["technical_quality"]["score"] is None
    assert observation["technical_quality"]["evidence"]
    assert all(value is False for value in observation["task_masks"].values())


def test_verification_rejects_changed_export(study):
    r, run = study.runner, study.make_run(eligible=1)
    study.responses.append(study.response())
    r.execute(run)
    observations = r.read_jsonl(r.DATA / "observations.jsonl")
    observations[0]["technical_quality"]["score"] = 5.0
    r.write_jsonl(r.DATA / "observations.jsonl", observations)
    with pytest.raises(ValueError, match="Saved exports differ"):
        r.verify(run)
    assert len(study.calls) == 1
