"""One-retry eligibility, immutable evidence and crash-safe admission checks."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("expansion_40000_recover_empty", HERE / "recover_empty.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
adapter = recovery.adapter


def response(plan, *, finish="length", content="", http=200, refusal=None):
    body = {
        "model": "deepseek-v4-1-flash-1017",
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {
            "cost": 0.01,
            "runinfra": {"cost_microcents": 1000000},
            "completion_tokens": 16384,
            "completion_tokens_details": {"reasoning_tokens": 16384},
        },
    }
    if refusal:
        body["choices"][0]["message"]["refusal"] = refusal
    result = {
        "http_status": http,
        "body": json.dumps(body),
        "headers": {},
        "transport_error": None,
        "request_started_at": "2026-11-17T00:00:00+00:00",
        "response_received_at": "2026-11-17T00:00:01+00:00",
        "latency_seconds": 1.0,
    }
    result["parsed"] = adapter.parsed_result(result, plan["profile"]["routes"]["routes"]["deepseek"])
    result["price_usd"] = result["parsed"]["price_usd"]
    return result


def score_content(score="3.00"):
    return json.dumps(
        {
            task: {
                "task": task,
                "status": "ok",
                "score": score,
                "evidence": "Visible detail",
                "limitations": [],
            }
            for task in ("technical_quality", "aesthetics")
        }
    )


def add_attempt(root, plan, case, result, number=1):
    body, sha = adapter.prepare(plan, case)
    folder = root / "attempts" / case["case_id"] / f"{number:02d}"
    adapter.atomic(
        folder / "request.json",
        {
            "plan_id": plan["plan_id"],
            "model": plan["model"],
            "case_id": case["case_id"],
            "attempt": number,
            "created_at": adapter.now(),
            "request_sha256": sha,
            "request_bytes": len(body),
            "view_sha256": case["view_sha256"],
            "profile_sha256": plan["profile"]["sha256"],
        },
    )
    adapter.atomic(folder / "result.json", result)
    return adapter.observation(plan, case, root, sha)


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    cases = []
    for index, identity in enumerate(("EMPTY", "VALID")):
        image = tmp_path / f"{identity}.png"
        Image.new("RGB", (4, 4), (index * 60, 30, 20)).save(image)
        cases.append(
            {
                "case_id": identity,
                "image_path": str(image),
                "view_sha256": adapter.sha(image),
                "benchmark_partition": "rehearsal",
            }
        )
    adapter.atomic(tmp_path / "manifest.json", {"cases": cases})
    adapter.atomic(tmp_path / "api-provider-snapshot.json", {"test": True})
    args = SimpleNamespace(
        data=tmp_path,
        model="deepseek",
        phase="expansion",
        manifest=tmp_path / "manifest.json",
        profile_directory=recovery.score.PACKAGE / "labeling",
    )
    recovery.score.batch.freeze(
        args, adapter, recovery.score.POLICY["deepseek"], [Path(recovery.score.__file__)]
    )
    root = tmp_path / "results/api/deepseek"
    plan = adapter.read(root / "plans/expansion.json")
    by_id = {case["case_id"]: case for case in plan["cases"]}
    add_attempt(root, plan, by_id["EMPTY"], response(plan))
    add_attempt(root, plan, by_id["VALID"], response(plan, finish="stop", content=score_content()))
    adapter.atomic(
        root / "expansion-status.json",
        {
            "status": "complete",
            "plan_id": plan["plan_id"],
            "planned_cases": 2,
            "selected_cases": 2,
            "completed_cases": 2,
            "outcomes": {"valid": 1, "failed": 1},
        },
    )
    monkeypatch.setenv("RUN_INFRA_API_KEY", "fixture-secret")
    return tmp_path, root, plan, by_id


def fake_http(monkeypatch, raw, *, transport_failure=False):
    calls = []

    def post(endpoint, **kwargs):
        calls.append(kwargs["data"])
        if transport_failure:
            raise adapter.requests.ConnectionError("fixture outage")
        return SimpleNamespace(status_code=raw["http_status"], text=raw["body"], headers=raw["headers"])

    monkeypatch.setattr(adapter.requests, "post", post)
    return calls


def test_freeze_preserves_profile_full_manifest_and_noneligible_observations(baseline):
    data, root, baseline_plan, _ = baseline
    result = recovery.freeze(data, "deepseek")
    plan = adapter.read(result["plan_path"])
    assert result["cases"] == 1 and [c["case_id"] for c in plan["cases"]] == ["EMPTY"]
    assert plan["profile"] == baseline_plan["profile"]
    assert plan["manifest_sha256"] == baseline_plan["manifest_sha256"]
    assert plan["recovery"]["attempt_limits"] == {"EMPTY": 2}
    assert str(root / "observations/VALID.json") in plan["reference_pins"]
    assert str(root / "observations/EMPTY.json") not in plan["reference_pins"]
    assert str(root / "attempts/EMPTY/01/result.json") in plan["reference_pins"]
    assert str(Path(recovery.__file__).resolve()) in plan["code_pins"]
    with pytest.raises(ValueError, match="already frozen"):
        recovery.freeze(data, "deepseek")


@pytest.mark.parametrize("state", ["running", "partial", "stopped"])
def test_freeze_requires_completed_baseline(baseline, state):
    data, root, _, _ = baseline
    path = root / "expansion-status.json"
    value = adapter.read(path)
    value["status"] = state
    adapter.atomic(path, value)
    with pytest.raises(ValueError, match="fully complete"):
        recovery.freeze(data, "deepseek")
    assert not (root / f"plans/{recovery.PHASE}.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "refusal",
        "content_denial",
        "partial_text",
        "whitespace",
        "wrong_model",
        "not_exhausted",
        "mixed_tokens",
        "invalid_numeric",
        "valid",
        "unscorable",
    ],
)
def test_empty_length_excludes_assessments_refusals_and_unverified_exhaustion(baseline, change):
    _, _, plan, _ = baseline
    raw = response(plan)
    body = json.loads(raw["body"])
    choice = body["choices"][0]
    if change == "refusal":
        choice["message"]["refusal"] = "Cannot assess"
    elif change == "content_denial":
        raw["http_status"] = 403
        body = {"error": {"message": "Request blocked by content filter: fixture"}}
    elif change in ("partial_text", "whitespace"):
        choice["message"]["content"] = '{"technical_quality":' if change == "partial_text" else " "
    elif change == "wrong_model":
        body["model"] = "different-model"
    elif change == "not_exhausted":
        body["usage"]["completion_tokens"] = 100
    elif change == "mixed_tokens":
        body["usage"]["completion_tokens_details"]["reasoning_tokens"] = 16000
    else:
        choice["finish_reason"] = "stop"
        choice["message"]["content"] = score_content("9.99" if change == "invalid_numeric" else "3.00")
        if change == "unscorable":
            ratings = json.loads(choice["message"]["content"])
            for rating in ratings.values():
                rating.update(status="unscorable", score=None)
            choice["message"]["content"] = json.dumps(ratings)
    raw["body"] = json.dumps(body)
    raw["parsed"] = adapter.parsed_result(raw, plan["profile"]["routes"]["routes"]["deepseek"])
    assert not recovery.empty_length(raw, plan["profile"]["routes"]["routes"]["deepseek"])


@pytest.mark.parametrize("prior", ["valid", "invalid_numeric", "refusal", "content_denial"])
def test_any_prior_assessment_or_denial_excludes_case(baseline, prior):
    data, root, plan, cases = baseline
    if prior == "content_denial":
        raw = response(plan, http=403)
        raw["body"] = json.dumps({"error": {"message": "Request blocked by content filter: fixture"}})
        raw["parsed"] = adapter.parsed_result(raw, plan["profile"]["routes"]["routes"]["deepseek"])
    else:
        raw = response(
            plan,
            finish="stop",
            content=score_content("9.99" if prior == "invalid_numeric" else "3.00"),
            refusal="Cannot assess" if prior == "refusal" else None,
        )
    add_attempt(root, plan, cases["EMPTY"], raw, 1)
    add_attempt(root, plan, cases["EMPTY"], response(plan), 2)
    assert recovery.freeze(data, "deepseek")["cases"] == 0


@pytest.mark.parametrize("provider,expected", [("OpenAI", True), ("Unexpected", False)])
def test_luna_cutoff_requires_its_pinned_provider(baseline, provider, expected):
    _, _, plan, _ = baseline
    raw = response(plan)
    body = json.loads(raw["body"])
    body.update(model="openai/gpt-6-luna-20260922", provider=provider)
    raw["body"] = json.dumps(body)
    route = plan["profile"]["routes"]["routes"]["luna"]
    raw["parsed"] = adapter.parsed_result(raw, route)
    assert recovery.empty_length(raw, route) is expected


def test_one_identical_retry_preserves_originals_and_resumes_without_http(baseline, monkeypatch):
    data, root, baseline_plan, _ = baseline
    preserved = {path: path.read_bytes() for path in (root / "attempts").rglob("*.json")}
    preserved[root / "observations/VALID.json"] = (root / "observations/VALID.json").read_bytes()
    recovery.freeze(data, "deepseek")
    calls = fake_http(monkeypatch, response(baseline_plan, finish="stop", content=score_content("2.50")))
    state = recovery.run(data, "deepseek")
    assert state["status"] == "complete" and state["outcomes"] == {"valid": 1}
    assert state["active_worker_limit"] == 1 and len(calls) == 1
    original = adapter.read(root / "attempts/EMPTY/01/request.json")
    new = adapter.read(root / "attempts/EMPTY/02/request.json")
    assert original["request_sha256"] == new["request_sha256"]
    assert original["profile_sha256"] == new["profile_sha256"]
    observed = adapter.read(root / "observations/EMPTY.json")
    assert observed["attempt_count"] == 2 and observed["technical_score"] == 2.5
    assert observed["total_attempt_known_price_usd"] == "0.02"
    recovery.run(data, "deepseek")
    assert len(calls) == 1
    assert all(path.read_bytes() == content for path, content in preserved.items())
    export_spec = importlib.util.spec_from_file_location("recovery_export_contract", HERE / "export.py")
    export = importlib.util.module_from_spec(export_spec)
    export_spec.loader.exec_module(export)
    recovery_plan = adapter.read(root / f"plans/{recovery.PHASE}.json")
    export.verify_recovery_evidence(
        root,
        recovery_plan,
        baseline_plan,
        {case["case_id"]: case["view_sha256"] for case in baseline_plan["cases"]},
    )


def test_recorded_transport_failure_consumes_the_single_retry(baseline, monkeypatch):
    data, root, plan, _ = baseline
    recovery.freeze(data, "deepseek")
    calls = fake_http(monkeypatch, response(plan), transport_failure=True)
    state = recovery.run(data, "deepseek")
    assert state["status"] == "complete" and state["outcomes"] == {"retryable_failure": 1}
    assert state["unpriced_attempts"] == 1
    recovery.run(data, "deepseek")
    assert len(calls) == 1
    row = adapter.read(root / "observations/EMPTY.json")
    assert row["total_attempt_price_usd"] is None and row["attempt_count"] == 2


def test_ambiguous_admitted_request_is_never_resent(baseline, monkeypatch):
    data, root, _, cases = baseline
    frozen = recovery.freeze(data, "deepseek")
    plan = adapter.read(frozen["plan_path"])
    add_attempt(root, plan, cases["EMPTY"], response(plan), 2)
    (root / "attempts/EMPTY/02/result.json").unlink()
    calls = fake_http(monkeypatch, response(plan))
    with pytest.raises(ValueError, match="manual reconciliation required, never resend"):
        recovery.run(data, "deepseek")
    assert not calls
    assert adapter.read(root / f"{recovery.PHASE}-status.json")["unpriced_attempts"] == 1


@pytest.mark.parametrize("target", ["prior_result", "valid_observation", "image", "profile"])
def test_changed_evidence_or_request_is_rejected_before_http(baseline, monkeypatch, target):
    data, root, plan, cases = baseline
    frozen = recovery.freeze(data, "deepseek")
    if target == "prior_result":
        (root / "attempts/EMPTY/01/result.json").write_text("{}")
    elif target == "valid_observation":
        (root / "observations/VALID.json").write_text("{}")
    elif target == "image":
        Path(cases["EMPTY"]["image_path"]).write_bytes(b"changed")
    else:
        altered = adapter.read(frozen["plan_path"])
        altered["profile"]["routes"]["routes"]["deepseek"]["settings"]["max_tokens"] = 32768
        altered["plan_id"] = adapter.contract.digest({k: v for k, v in altered.items() if k != "plan_id"})
        adapter.atomic(Path(frozen["plan_path"]), altered)
    calls = fake_http(monkeypatch, response(plan))
    with pytest.raises((ValueError, KeyError)):
        recovery.run(data, "deepseek")
    assert not calls


def test_prior_transient_uses_per_case_total_attempt_limit(baseline, monkeypatch):
    data, root, plan, cases = baseline
    raw = response(plan, http=503)
    raw["body"] = json.dumps({"error": {"message": "Unavailable"}})
    raw["parsed"] = adapter.parsed_result(raw, plan["profile"]["routes"]["routes"]["deepseek"])
    raw["price_usd"] = None
    add_attempt(root, plan, cases["EMPTY"], raw, 1)
    add_attempt(root, plan, cases["EMPTY"], response(plan), 2)
    frozen = recovery.freeze(data, "deepseek")
    recovery_plan = adapter.read(frozen["plan_path"])
    assert recovery_plan["recovery"]["attempt_limits"] == {"EMPTY": 3}
    calls = fake_http(monkeypatch, response(plan, finish="stop", content=score_content()))
    recovery.run(data, "deepseek")
    recovery.run(data, "deepseek")
    assert len(calls) == 1
    assert adapter.read(root / "observations/EMPTY.json")["attempt_count"] == 3


@pytest.mark.parametrize("outcome", ["length", "invalid_numeric", "refusal"])
def test_unsuccessful_recovery_is_terminal_without_another_call(baseline, monkeypatch, outcome):
    data, root, plan, _ = baseline
    recovery.freeze(data, "deepseek")
    raw = (
        response(plan)
        if outcome == "length"
        else response(
            plan,
            finish="stop",
            content=score_content("9.99"),
            refusal="Cannot assess" if outcome == "refusal" else None,
        )
    )
    calls = fake_http(monkeypatch, raw)
    recovery.run(data, "deepseek")
    recovery.run(data, "deepseek")
    assert len(calls) == 1
    row = adapter.read(root / "observations/EMPTY.json")
    assert row["technical_score"] is None and row["aesthetic_score"] is None
    assert row["attempt_count"] == 2


@pytest.mark.parametrize("omission", ["noneligible_observation", "source_result"])
def test_recovery_requires_complete_evidence_pin_coverage(baseline, monkeypatch, omission):
    data, root, plan, _ = baseline
    frozen = recovery.freeze(data, "deepseek")
    altered = adapter.read(frozen["plan_path"])
    path = root / (
        "observations/VALID.json"
        if omission == "noneligible_observation"
        else "attempts/EMPTY/01/result.json"
    )
    altered["reference_pins"].pop(str(path))
    altered["plan_id"] = adapter.contract.digest({k: v for k, v in altered.items() if k != "plan_id"})
    adapter.atomic(Path(frozen["plan_path"]), altered)
    calls = fake_http(monkeypatch, response(plan))
    with pytest.raises(ValueError, match="pin"):
        recovery.run(data, "deepseek")
    assert not calls


def test_extra_attempt_is_rejected_without_further_http(baseline, monkeypatch):
    data, root, plan, cases = baseline
    frozen = recovery.freeze(data, "deepseek")
    recovery_plan = adapter.read(frozen["plan_path"])
    calls = fake_http(monkeypatch, response(plan))
    recovery.run(data, "deepseek")
    add_attempt(root, recovery_plan, cases["EMPTY"], response(plan), 3)
    with pytest.raises(ValueError, match="attempt count"):
        recovery.run(data, "deepseek")
    assert len(calls) == 1
