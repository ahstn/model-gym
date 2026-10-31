"""Offline checks for trial isolation, immutable inputs, and single-attempt billing."""

import base64
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


def envelope(model, invalid_aesthetic=False):
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
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ratings)}}],
        "usage": {"cost": "0.001", "runinfra": {"cost_microcents": 100000}},
    }


@pytest.fixture
def trial(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("photo_scope_runner_tests", HERE / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, data = tmp_path / "method", tmp_path / "data"
    source.mkdir()
    for name in (
        "runner.py",
        "test_runner.py",
        "routes.json",
        "photo-scope.txt",
        "build_prompt.py",
        "prompt-provenance.json",
    ):
        (source / name).write_bytes((HERE / name).read_bytes())
    prompt = module.transport.prompt_bundle(
        module.PROJECT / "prompts/visual-rating-v1", {"id": "combined", "addendum": "Photo scope test."}
    )
    (source / "prompt.json").write_text(json.dumps(prompt))
    cases = []
    for index in range(1, 73):
        image = tmp_path / f"image-{index:03}.png"
        image.write_bytes(PNG + str(index).encode())
        cases.append(
            {
                "case_id": f"P{index:03}",
                "image_path": str(image),
                "view_sha256": module.file_sha256(image),
                "mime_type": "image/png",
            }
        )
    (source / "manifest.json").write_text(json.dumps({"cases": cases}))
    monkeypatch.setattr(module, "HERE", source)
    monkeypatch.setattr(module.transport, "HERE", source)
    monkeypatch.setattr(module.transport, "DATA", data)
    monkeypatch.setenv("RUN_INFRA_API_KEY", KEY)
    calls, responses, guard = [], [], threading.Lock()

    def post(url, **kwargs):
        with guard:
            calls.append((url, kwargs))
            response = responses.pop(0) if responses else envelope("deepseek-v4-1-flash-1017")
        return SimpleNamespace(status_code=200, text=json.dumps(response), headers={})

    monkeypatch.setattr(module.transport.requests, "post", post)
    run = module.freeze("deepseek", source / "manifest.json", source / "prompt.json")
    module.save_report(run)
    return SimpleNamespace(module=module, run=run, source=source, data=data, calls=calls, responses=responses)


def test_probe_then_resume_sends_each_of_72_once(trial):
    r = trial.module
    assert r.execute(trial.run, {"P048"})["attempted"] == 1
    assert r.execute(trial.run, {"P048"})["attempted"] == 1
    report = r.execute(trial.run)
    assert report["status"] == "complete"
    assert report["attempted"] == report["valid_observations"] == len(trial.calls) == 72
    assert r.execute(trial.run)["attempted"] == len(trial.calls) == 72
    assert r.verify(trial.run) == report
    for path in trial.data.rglob("*.json"):
        assert KEY not in path.read_text()


def test_request_journal_exists_before_network(trial, monkeypatch):
    r = trial.module
    original_send = r.transport.send

    def inspect_send(run, case, key):
        request = r.transport.directory(run) / "attempts" / case["attempt_id"] / "request.json"
        assert request.is_file()
        assert json.loads(request.read_text())["request_sha256"] == case["request_sha256"]
        original_send(run, case, key)

    monkeypatch.setattr(r.transport, "send", inspect_send)
    r.execute(trial.run, {"P048"})


def test_interrupted_attempt_is_not_resent(trial):
    r = trial.module
    case = next(case for case in trial.run["cases"] if case["case_id"] == "P048")
    request = {
        name: case[name] for name in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")
    }
    r.durable_json(
        r.transport.directory(trial.run) / "attempts" / case["attempt_id"] / "request.json",
        {
            **request,
            "run_id": trial.run["run_id"],
            "created_at": r.now(),
        },
    )
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(trial.run)
    assert not trial.calls
    assert r.verify(trial.run)["attempts"][0]["status"] == "unknown"


def test_invalid_sibling_preserved_without_repair(trial):
    r = trial.module
    trial.responses.append(envelope("deepseek-v4-1-flash-1017", invalid_aesthetic=True))
    report = r.execute(trial.run, {"P048"})
    assert report["invalid_observations"] == 1
    rows = r.transport.read_jsonl(r.transport.directory(trial.run) / "observations.jsonl")
    assert rows[0]["technical_quality"]["score"] == 1
    assert rows[0]["aesthetics"] == {"status": "invalid", "score": None}
    assert rows[0]["validation_errors"]["aesthetics"] == "Score and status disagree"
    r.execute(trial.run, {"P048"})
    assert len(trial.calls) == 1


@pytest.mark.parametrize("change", ("image", "prompt", "routes", "policy"))
def test_changed_inputs_stop_before_network(trial, change):
    case = trial.run["cases"][0]
    path = (
        Path(case["image_path"])
        if change == "image"
        else trial.source
        / {
            "prompt": "prompt.json",
            "routes": "routes.json",
            "policy": "photo-scope.txt",
        }[change]
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        trial.module.execute(trial.run)
    assert not trial.calls


def test_budget_reserves_unknown_full_context_cost(trial):
    run = trial.run
    config = run["route_config"]
    expected = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"]["max_tokens"]
    ) / 1_000_000
    assert Decimal(run["budget"]["reserve_per_request_usd"]) == expected
    run["budget"]["max_cost_usd"] = "0.001"
    run["run_id"] = trial.module.digest({key: value for key, value in run.items() if key != "run_id"})
    with pytest.raises(ValueError, match="admission budget"):
        trial.module.execute(run)
    assert not trial.calls


def test_wrong_model_blocks_continuation(trial):
    trial.responses.append(envelope("some-other-model"))
    report = trial.module.execute(trial.run, {"P048"})
    assert report["status"] == "blocked"
    with pytest.raises(ValueError, match="failed or uncertain"):
        trial.module.execute(trial.run)
    assert len(trial.calls) == 1
