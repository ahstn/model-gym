"""Offline checks of bounded concurrency, scaled resume, immutable inputs, and billing."""

import base64
import hashlib
import importlib.util
import json
import threading
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jr1sAAAAASUVORK5CYII="
)
KEY = "offline-secret-do-not-persist"


def envelope(model="deepseek-v4-1-flash-1017", *, invalid_aesthetic=False, priced=True):
    ratings = {
        task: {
            "task": task,
            "status": "ok",
            "score": "1.00",
            "evidence": "Visible poster reproduction is outside the intended photography domain.",
            "limitations": ["unsupported_domain"],
        }
        for task in ("technical_quality", "aesthetics")
    }
    if invalid_aesthetic:
        ratings["aesthetics"]["score"] = None
    usage = {
        "prompt_tokens": 300,
        "completion_tokens": 100,
        "completion_tokens_details": {"reasoning_tokens": 50},
    }
    if priced:
        usage.update(cost="0.001", runinfra={"cost_microcents": 100000})
    reply = {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ratings)}}],
        "usage": usage,
    }
    if model == "openai/gpt-6-luna":
        reply["provider"] = "OpenAI"
    return reply


@pytest.fixture
def trial(tmp_path, monkeypatch, request):
    spec = importlib.util.spec_from_file_location("dual_high_2000_test_runner", HERE / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = getattr(request, "param", 32)
    monkeypatch.setattr(module, "CASE_COUNT", count)
    method, data = tmp_path / "method", tmp_path / "data"
    method.mkdir()
    for name in (
        "runner.py",
        "test_runner.py",
        "routes.json",
        "protocol.md",
        "photo-scope.txt",
        "prompt.json",
        "prompt-provenance.json",
    ):
        (method / name).write_bytes((HERE / name).read_bytes())
    cases = []
    for index in range(1, count + 1):
        image = tmp_path / f"image-{index:04}.png"
        image.write_bytes(PNG + str(index).encode())
        cases.append(
            {
                "case_id": f"T{index:04}",
                "image_path": str(image),
                "view_sha256": module.file_sha256(image),
                "mime_type": "image/png",
            }
        )
    manifest = {"cases": cases}
    manifest["manifest_id"] = module.manifest_digest(manifest)
    (method / "manifest.json").write_text(json.dumps(manifest))
    (method / "preparation-validation.json").write_text(
        json.dumps(
            {
                "verified": True,
                "images": count,
                "manifest_id": manifest["manifest_id"],
            }
        )
    )
    monkeypatch.setattr(module, "HERE", method)
    monkeypatch.setattr(module.transport, "HERE", method)
    monkeypatch.setattr(module.transport, "DATA", data)
    monkeypatch.setenv("RUN_INFRA_API_KEY", KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    calls, responses, guard = [], [], threading.Lock()

    def post(url, **kwargs):
        body = json.loads(kwargs["data"])
        with guard:
            calls.append((url, kwargs))
            response = responses.pop(0) if responses else envelope(body["model"])
        return SimpleNamespace(
            status_code=200,
            text=json.dumps(response),
            headers={"x-runinfra-reasoning-effort": "high", "x-request-id": "test-request-id"},
        )

    monkeypatch.setattr(module.transport.requests, "post", post)
    old_paths = [
        module.SOURCE / "runner.py",
        module.SOURCE / "routes.json",
        module.HIGH_SOURCE / "routes.json",
    ]
    old_hashes = {path: module.file_sha256(path) for path in old_paths}
    run = module.freeze("deepseek", method / "manifest.json", method / "prompt.json")
    module.save_report(run, check_payloads=False)
    fixture = SimpleNamespace(
        module=module, run=run, method=method, data=data, calls=calls, responses=responses
    )
    yield fixture
    assert {path: module.file_sha256(path) for path in old_paths} == old_hashes


def test_manifest_identity_matches_frozen_preparation_with_non_ascii(trial):
    r = trial.module
    value = {"cases": [{"case_id": "T0001"}], "caption": "Zürich — façades"}
    canonical = b'{"caption":"Z\\u00fcrich \\u2014 fa\\u00e7ades","cases":[{"case_id":"T0001"}]}'
    assert r.manifest_digest(value) == hashlib.sha256(canonical).hexdigest()
    assert r.manifest_digest(value) != r.digest(value)
    manifest = json.loads((HERE / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "manifest_id"}
    assert r.manifest_digest(content) == manifest["manifest_id"]
    assert r.digest(content) != manifest["manifest_id"]


@pytest.mark.parametrize("trial", [2000], indirect=True)
def test_2000_probe_resume_is_linear_and_sends_each_case_once(trial, monkeypatch):
    r, run = trial.module, trial.run
    assert run["settings"]["reasoning_effort"] == "high"
    assert run["policy"]["max_concurrency"] == 16
    assert r.execute(run, {"T2000"})["attempted"] == 1
    assert r.execute(run, {"T2000"})["attempted"] == 1
    reads = []
    original_read = r.read_attempt

    def tracked_read(run, case):
        reads.append(case["attempt_id"])
        return original_read(run, case)

    monkeypatch.setattr(r, "read_attempt", tracked_read)
    report = r.execute(run)
    # One prior attempt read at startup, followed by one read per new completion.
    assert len(reads) == 2000
    assert report["status"] == "complete"
    assert report["attempted"] == report["valid_observations"] == len(trial.calls) == 2000
    assert report["scored"] == {"technical_quality": 2000, "aesthetics": 2000}
    assert report["usage_totals"]["reasoning_tokens"] == 100000
    assert Decimal(report["total_known_cost_usd"]) == Decimal("2.000")
    assert r.execute(run)["attempted"] == len(trial.calls) == 2000
    assert r.verify(run) == report
    for path in trial.data.rglob("*.json"):
        assert KEY not in path.read_text()


def test_journal_before_network_and_concurrency_never_exceeds_16(trial, monkeypatch):
    r, run = trial.module, trial.run
    original_send = r.transport.send
    active, peak, guard = 0, 0, threading.Lock()
    first_wave = threading.Barrier(16, timeout=10)

    def checked_send(run, case, key):
        nonlocal active, peak
        folder = r.transport.directory(run) / "attempts" / case["attempt_id"]
        request = json.loads((folder / "request.json").read_text())
        assert request["request_sha256"] == case["request_sha256"]
        assert folder.is_relative_to(trial.data)
        with guard:
            active += 1
            peak = max(peak, active)
        first_wave.wait()
        try:
            original_send(run, case, key)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(r.transport, "send", checked_send)
    report = r.execute(run)
    assert report["status"] == "complete"
    assert peak == 16 and active == 0


def test_same_prompt_and_high_reasoning_on_both_routes(trial):
    r = trial.module
    luna = r.freeze("luna", trial.method / "manifest.json", trial.method / "prompt.json")
    assert luna["prompts"] == trial.run["prompts"]
    assert luna["settings"]["reasoning"]["effort"] == "high"
    assert luna["settings"]["provider"]["allow_fallbacks"] is False
    assert "temperature" not in luna["settings"]
    assert r.execute(luna, {"T0001"})["valid_observations"] == 1
    sent = json.loads(trial.calls[0][1]["data"])
    assert set(sent["messages"][1]["content"][0]) == {"type", "text"}
    assert sent["messages"][1]["content"][0]["text"] == luna["prompts"]["photo_scope"]["user"]
    assert "references" not in sent and "case_id" not in sent


def test_interrupted_attempt_stops_without_resending(trial):
    r, run = trial.module, trial.run
    case = run["cases"][0]
    request = {
        key: case[key] for key in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")
    }
    r.durable_json(
        r.transport.directory(run) / "attempts" / case["attempt_id"] / "request.json",
        {**request, "run_id": run["run_id"], "created_at": r.now()},
    )
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert not trial.calls
    assert r.verify(run)["attempts"][0]["status"] == "unknown"


def test_inflight_failure_drains_admitted_requests_without_admitting_more(trial, monkeypatch):
    r, run = trial.module, trial.run
    original = r.transport.requests.post
    first_wave = threading.Barrier(16, timeout=10)
    failure_interpreted = threading.Event()
    trial.responses.append(envelope(priced=False))

    def post(*args, **kwargs):
        first_wave.wait()
        response = original(*args, **kwargs)
        if "cost" in json.loads(response.text)["usage"]:
            assert failure_interpreted.wait(timeout=10)
        return response

    original_read = r.read_attempt

    def read_attempt(run, case):
        detail, row = original_read(run, case)
        if detail["cost_usd"] is None:
            failure_interpreted.set()
        return detail, row

    monkeypatch.setattr(r.transport.requests, "post", post)
    monkeypatch.setattr(r, "read_attempt", read_attempt)
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["attempted"] == len(trial.calls) == 16
    assert report["unpriced_attempts"] == 1
    assert r.verify(run) == report


def test_invalid_sibling_preserved_and_not_retried(trial):
    r, run = trial.module, trial.run
    trial.responses.append(envelope(invalid_aesthetic=True))
    report = r.execute(run, {"T0001"})
    assert report["invalid_observations"] == 1 and report["status"] == "partial"
    row = r.read_jsonl(r.transport.directory(run) / "observations.jsonl")[0]
    assert row["technical_quality"]["score"] == 1
    assert row["aesthetics"] == {"status": "invalid", "score": None}
    assert r.execute(run, {"T0001"})["attempted"] == len(trial.calls) == 1


@pytest.mark.parametrize("bad", ("missing_cost", "wrong_model", "wrong_reasoning", "excess_cost"))
def test_failure_stops_route_and_is_never_resent(trial, monkeypatch, bad):
    r, run = trial.module, trial.run
    reply = envelope(
        model="wrong-model" if bad == "wrong_model" else "deepseek-v4-1-flash-1017",
        priced=bad != "missing_cost",
    )
    if bad == "excess_cost":
        reply["usage"]["runinfra"]["cost_microcents"] = 100000000
    trial.responses.append(reply)
    if bad == "wrong_reasoning":
        original = r.transport.requests.post

        def post(*args, **kwargs):
            result = original(*args, **kwargs)
            result.headers["x-runinfra-reasoning-effort"] = "none"
            return result

        monkeypatch.setattr(r.transport.requests, "post", post)
    assert r.execute(run, {"T0001"})["status"] == "blocked"
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(trial.calls) == 1


@pytest.mark.parametrize("changed", ("image", "prompt", "routes", "protocol", "runner"))
def test_mutated_inputs_block_before_network(trial, changed):
    path = (
        Path(trial.run["cases"][0]["image_path"])
        if changed == "image"
        else trial.method
        / {
            "prompt": "prompt.json",
            "routes": "routes.json",
            "protocol": "protocol.md",
            "runner": "runner.py",
        }[changed]
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        trial.module.execute(trial.run)
    assert not trial.calls


def test_full_context_reserves_cover_all_inflight_requests(trial):
    r, run = trial.module, trial.run
    config = run["route_config"]
    reserve = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"]["max_tokens"]
    ) / 1000000
    assert Decimal(run["budget"]["reserve_per_request_usd"]) == reserve
    assert r.admission_count(run, Decimal(0), 0, 2000) == 16
    assert r.admission_count(run, Decimal(10) - reserve * 3, 2, 2000) == 1
    assert r.admission_count(run, Decimal(10) - reserve * 2, 2, 2000) == 0
    run["budget"]["max_cost_usd"] = "0.001"
    run["run_id"] = r.digest({key: value for key, value in run.items() if key != "run_id"})
    with pytest.raises(ValueError, match="admission budget"):
        r.execute(run)
    assert not trial.calls


def test_saved_export_tampering_is_detected_and_rebuilt_without_calls(trial):
    r, run = trial.module, trial.run
    report = r.execute(run, {"T0001"})
    report_path = r.transport.directory(run) / "report.json"
    value = json.loads(report_path.read_text())
    value["known_cost_usd"] = "9.99"
    report_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Saved exports differ"):
        r.verify(run)
    assert r.execute(run, {"T0001"}) == report
    assert len(trial.calls) == 1 and r.verify(run) == report
