"""Offline checks for an isolated continuation of previously unattempted views."""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_base_spec = importlib.util.spec_from_file_location("runinfra_test_fixtures", HERE / "test_runner.py")
_base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)
study = _base.study
envelope = _base.envelope


def invalid_envelope(model, *, amount="0.001"):
    body = envelope(model, amount=amount)
    message = body["choices"][0]["message"]
    # Preserve the lexical decimal; constructing a float here would erase the defect.
    message["content"] = message["content"].replace('"score": 3.2', '"score": 3.8999999999999999')
    return body


@pytest.fixture
def continuation(study, monkeypatch):
    spec = importlib.util.spec_from_file_location("runinfra_continuation_offline", HERE / "continuation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = study.runner
    run = study.make_run(eligible=3)
    # Preserve an original priced strict failure, as in the real initial run.
    study.responses.append(study.response(invalid_envelope(r.SLUG, amount="0.00030555")))
    original = r.execute(run)
    assert original["attempts"][0]["reason"] == module.STRICT_FAILURE
    study.calls.clear()
    monkeypatch.setattr(module, "runner", r)
    monkeypatch.setattr(module, "DATA", r.DATA / "continuation-01")
    return module, study


def test_freeze_excludes_original_attempt_and_pins_inputs(continuation):
    c, s = continuation
    before = c.original_evidence()
    plan = c.freeze()
    assert [case["case_id"] for case in plan["cases"]] == ["case-01", "case-02"]
    assert plan["excluded_attempted_case_ids"] == ["case-00"]
    assert plan["budget"]["prior_cost_usd"] == "0.00030555"
    assert plan["budget"]["max_cost_usd"] == "1.00"
    assert {Path(name).name for name in plan["source_pins"]} == {
        "runner.py",
        "continuation.py",
        "continuation-protocol.md",
    }
    assert plan["policy"]["rating_repair"] is False
    assert plan["training_ready"] is False
    assert c.original_evidence() == before
    assert c.read_plan()[0] == plan
    assert s.calls == []
    with pytest.raises(ValueError, match="already exists"):
        c.freeze()


def test_priced_strict_failure_advances_without_repair_or_retry(continuation):
    c, s = continuation
    before = c.original_evidence()
    c.freeze()
    plan, run = c.read_plan()
    invalid = invalid_envelope(s.runner.SLUG)
    s.responses.extend([s.response(invalid), s.response(envelope(s.runner.SLUG))])
    report = c.execute(plan, run)
    assert report["status"] == "complete_with_invalid_ratings"
    assert report["attempted"] == 2
    assert report["scored"] == report["failed_attempts"] == 1
    assert report["attempts"][0]["score"] is None
    assert report["attempts"][0]["reason"] == c.STRICT_FAILURE
    assert Decimal(report["total_known_cost_usd"]) == Decimal("0.00230555")
    saved = json.loads((c.DATA / "attempts/case-01/result.json").read_text())
    assert "3.8999999999999999" in json.loads(saved["body"])["choices"][0]["message"]["content"]
    observations = s.runner.read_jsonl(c.DATA / "observations.jsonl")
    assert [record["case_id"] for record in observations] == ["case-02"]
    assert all(value is False for value in observations[0]["task_masks"].values())
    assert observations[0]["provenance"]["run_id"] == run["run_id"]
    assert observations[0]["technical_quality"]["score"] == 3.2
    for case, (_, kwargs) in zip(plan["cases"], s.calls, strict=True):
        assert kwargs["data"] == s.runner.encoded(s.runner.payload(run, case))
    assert c.verify(*c.read_plan()) == report
    assert c.execute(*c.read_plan()) == report
    assert len(s.calls) == 2
    assert c.original_evidence() == before


@pytest.mark.parametrize("failure", ["transport", "http", "model", "envelope", "incomplete", "cost"])
def test_other_failures_stop_without_retry(continuation, failure):
    c, s = continuation
    c.freeze()
    body = envelope(s.runner.SLUG)
    status = 200
    if failure == "transport":
        response = s.runner.requests.Timeout("secret offline-secret-never-persist")
    else:
        if failure == "http":
            status = 503
        elif failure == "model":
            body["model"] = "other-model"
        elif failure == "envelope":
            body["choices"][0]["message"] = None
        elif failure == "incomplete":
            body["choices"][0]["finish_reason"] = "length"
        else:
            body["usage"] = {}
        response = s.response(body, status=status)
    s.responses.append(response)
    report = c.execute(*c.read_plan())
    assert report["status"] == "blocked"
    assert report["attempted"] == 1
    assert report["attempts"][0]["reason"] != c.STRICT_FAILURE
    with pytest.raises(ValueError, match="non-rating failure"):
        c.execute(*c.read_plan())
    assert len(s.calls) == 1
    assert c.verify(*c.read_plan()) == report
    assert not (c.DATA / "attempts/case-02").exists()


def test_combined_budget_includes_original_spend(continuation):
    c, s = continuation
    c.freeze()
    s.responses.append(s.response(envelope(s.runner.SLUG, amount="0.8420")))
    with pytest.raises(ValueError, match="combined admission budget"):
        c.execute(*c.read_plan())
    report = c.verify(*c.read_plan())
    assert Decimal(report["total_known_cost_usd"]) == Decimal("0.84230555")
    assert report["attempted"] == 1
    with pytest.raises(ValueError, match="combined admission budget"):
        c.execute(*c.read_plan())
    assert len(s.calls) == 1


def test_interrupted_attempt_blocks_without_call(continuation):
    c, s = continuation
    c.freeze()
    plan, run = c.read_plan()
    case = plan["cases"][0]
    request = {key: case[key] for key in ("case_id", "request_sha256", "request_bytes")}
    request.update(run_id=run["run_id"], continuation_id=plan["continuation_id"], created_at=s.runner.now())
    s.runner.atomic_json(c.DATA / "attempts/case-01/request.json", request)
    with pytest.raises(ValueError, match="non-rating failure"):
        c.execute(plan, run)
    report = c.verify(plan, run)
    assert report["attempts"][0]["reason"] == "interrupted_request"
    assert report["attempts"][0]["score"] is None
    assert s.calls == []


def test_changed_original_evidence_blocks_before_call(continuation):
    c, s = continuation
    c.freeze()
    path = s.runner.DATA / "attempts/case-00/result.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Original frozen evidence changed"):
        c.read_plan()
    assert s.calls == []


def test_tampered_validated_export_is_detected(continuation):
    c, s = continuation
    c.freeze()
    s.responses.extend([s.response(), s.response()])
    c.execute(*c.read_plan())
    observations = s.runner.read_jsonl(c.DATA / "observations.jsonl")
    observations[0]["technical_quality"]["score"] = 1.0
    s.runner.write_jsonl(c.DATA / "observations.jsonl", observations)
    with pytest.raises(ValueError, match="exports differ"):
        c.verify(*c.read_plan())
    assert len(s.calls) == 2
