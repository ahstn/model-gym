"""Offline checks for exact repeat identity and durable, isolated high-reasoning calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent


@pytest.fixture
def trial(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("deepseek_high_runner_tests", HERE / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method = tmp_path / "method"
    method.mkdir()
    for name in ("runner.py", "test_runner.py", "routes.json", "protocol.md", "runinfra-catalog.json"):
        (method / name).write_bytes((HERE / name).read_bytes())
    monkeypatch.setattr(module, "HERE", method)
    monkeypatch.setattr(module.transport, "HERE", method)
    monkeypatch.setattr(module.transport, "DATA", tmp_path / "data")
    monkeypatch.setenv("RUN_INFRA_API_KEY", "offline-secret-not-for-storage")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        reply = {
            "model": "deepseek-v4-1-flash-1017",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                task: {"task": task, "status": "ok", "score": "1.00",
                       "evidence": "The supplied image is a flat poster reproduction.",
                       "limitations": ["unsupported_domain"]}
                for task in ("technical_quality", "aesthetics")
            })}}],
            "usage": {
                "cost": "0.001", "runinfra": {"cost_microcents": 100000},
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        }
        return SimpleNamespace(
            status_code=200, text=json.dumps(reply), headers={"x-runinfra-reasoning-effort": "high"}
        )

    monkeypatch.setattr(module.transport.requests, "post", post)
    run = module.freeze()
    module.save_report(run)
    return SimpleNamespace(module=module, run=run, method=method, calls=calls)


def test_frozen_repeat_preserves_all_images_and_prompt(trial):
    r, run = trial.module, trial.run
    baseline = json.loads((r.BASELINE_DATA / "run.json").read_text())
    assert run["prompts"] == baseline["prompts"]
    assert len(run["cases"]) == 72
    for case, old in zip(run["cases"], baseline["cases"], strict=True):
        assert {k: v for k, v in case.items() if k not in {"request_sha256", "request_bytes"}} == {
            k: v for k, v in old.items() if k not in {"request_sha256", "request_bytes"}
        }
        body, old_body = r.payload(run, case), r.payload(baseline, old)
        assert {key for key in body if body[key] != old_body[key]} == {"reasoning_effort", "max_tokens"}
        assert body["reasoning_effort"] == "high" and body["max_tokens"] == 16384
    assert run["baseline"]["run_id"] == baseline["run_id"]
    for path in (r.BASELINE / "references.jsonl", r.BASELINE_DATA / "observations.jsonl"):
        assert run["inputs"][str(path)] == r.file_sha256(path)
    assert not any(Path(path).name in {"README.md", "report.json"} for path in run["inputs"])


def test_probe_resume_and_verify_use_isolated_single_attempts(trial, monkeypatch):
    r, run = trial.module, trial.run
    original_send = r.transport.send

    def journaled_send(run, case, key):
        directory = r.transport.directory(run)
        assert directory.is_relative_to(trial.method.parent / "data")
        assert (directory / "attempts" / case["attempt_id"] / "request.json").is_file()
        original_send(run, case, key)

    monkeypatch.setattr(r.transport, "send", journaled_send)
    assert r.execute(run, {"P048"})["attempted"] == 1
    assert r.execute(run, {"P048"})["attempted"] == 1
    report = r.execute(run)
    assert report["status"] == "complete"
    assert report["attempted"] == report["valid_observations"] == len(trial.calls) == 72
    assert r.execute(run)["attempted"] == len(trial.calls)
    assert r.verify(run) == report
    preserved = json.loads((r.transport.directory(run) / "attempts/photo_scope--P048/result.json").read_text())
    assert preserved["headers"]["x-runinfra-reasoning-effort"] == "high"
    assert json.loads(preserved["body"])["usage"]["completion_tokens_details"]["reasoning_tokens"] == 50


def test_changed_configuration_blocks_before_any_request(trial):
    path = trial.method / "routes.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Pinned input changed"):
        trial.module.execute(trial.run)
    assert not trial.calls


def test_uncertain_probe_is_not_resent(trial):
    r, run = trial.module, trial.run
    case = next(case for case in run["cases"] if case["case_id"] == "P048")
    request = {key: case[key] for key in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")}
    r.durable_json(
        r.transport.directory(run) / "attempts" / case["attempt_id"] / "request.json",
        {**request, "run_id": run["run_id"], "created_at": r.now()},
    )
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert not trial.calls
