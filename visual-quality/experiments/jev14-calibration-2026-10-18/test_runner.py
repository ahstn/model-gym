"""Offline checks for request isolation, durable resume and bounded dispatch."""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

RUNNER = Path(__file__).with_name("runner.py").resolve()


@pytest.fixture
def study(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("jev14_calibration_runner_offline_test", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "HERE", tmp_path / "study")
    monkeypatch.setattr(runner, "DATA", tmp_path / "data")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    runner.HERE.mkdir()
    runner.DATA.mkdir()
    (runner.DATA / "images").mkdir()
    image = runner.DATA / "images/test.jpg"
    Image.new("RGB", (256, 320), (100, 120, 140)).save(image)
    prompt = {
        "id": "offline-ai-prompt",
        "task": "ai",
        "context": "Assess the supplied image.",
        "questions": {
            "ai_probability": {"type": "noul", "instructions": "Estimate AI origin."},
            "medium": {
                "type": "choice",
                "instructions": "Select the visible medium.",
                "criteria": {"photo": "Photograph", "render": "Rendered image"},
            },
            "uncertainty": {
                "type": "choice",
                "instructions": "Select uncertainty.",
                "criteria": {"low": "Low", "high": "High"},
            },
        },
    }
    source = runner.HERE / "source.txt"
    source.write_text("frozen source")
    calls, active = [], {"current": 0, "maximum": 0}
    options = {"fail_first": False, "unknown_first": False, "delay": 0.0}
    lock = threading.Lock()
    original_post = runner.post

    def prepare(calibration=3, validation=2, budget=0.10):
        cases = [
            {
                "row_id": f"{split}-{index}",
                "split": split,
                "image": "images/test.jpg",
                "image_width": 256,
                "image_height": 320,
                "image_bytes": image.stat().st_size,
                "delivered_sha256": runner.sha(image),
            }
            for split, count in (("calibration", calibration), ("validation", validation))
            for index in range(count)
        ]
        (runner.HERE / "manifest.json").write_text(json.dumps({"cases": cases}))
        (runner.HERE / "prompt.json").write_text(json.dumps(prompt))
        protocol = {
            "maximum_attempts": len(cases),
            "manifest_sha256": runner.sha(runner.HERE / "manifest.json"),
            "prompt_sha256": runner.sha(runner.HERE / "prompt.json"),
            "execution_sha256": {str(RUNNER): runner.sha(RUNNER)},
            "source_inputs": {str(source): runner.sha(source)},
            "budget_usd": budget,
            "reserve_per_request_usd": 0.012,
            "concurrency": 4,
            "minimum_interval_seconds": 0.25,
        }
        (runner.HERE / "protocol.json").write_text(json.dumps(protocol))

    def fake_post(payload, questions):
        with lock:
            index = len(calls)
            calls.append(time.monotonic())
            active["current"] += 1
            active["maximum"] = max(active["maximum"], active["current"])

        def transport(*_args, **_kwargs):
            time.sleep((0.65 if index == 0 else 0.8) if options["fail_first"] else options["delay"])
            answers = {
                key: {"type": "noul", "noul": 0.73}
                if question["type"] == "noul"
                else {"type": "choice", "choice": next(iter(question["criteria"]))}
                for key, question in questions["questions"].items()
            }
            response = {
                "model": runner.REVISION,
                "answers": answers,
                "usage": {} if index == 0 and options["unknown_first"] else {"cost": 0.00001},
            }
            return SimpleNamespace(
                status_code=503 if index == 0 and options["fail_first"] else 200,
                headers={},
                text=json.dumps(response),
            )

        try:
            return original_post(payload, questions, transport=transport)
        finally:
            with lock:
                active["current"] -= 1

    def select():
        selection = {
            "protocol_sha256": runner.sha(runner.HERE / "protocol.json"),
            "calibration_ledger_sha256": runner.sha(runner.ledger_path("calibration")),
            "rule": {"threshold": 0.8, "gate": "none"},
        }
        (runner.HERE / "selection.json").write_text(json.dumps(selection))
        (runner.HERE / "selection-lock.json").write_text(
            json.dumps(
                {
                    "selection_sha256": runner.sha(runner.HERE / "selection.json"),
                }
            )
        )

    monkeypatch.setattr(runner, "post", fake_post)
    prepare()
    return SimpleNamespace(
        runner=runner,
        prepare=prepare,
        select=select,
        calls=calls,
        active=active,
        options=options,
        source=source,
    )


def test_validation_requires_selection_and_lock_before_any_call(study):
    runner = study.runner
    with pytest.raises(FileNotFoundError):
        runner.run("validation", 1)
    assert not study.calls
    runner.run("calibration", 3)
    study.select()
    (runner.HERE / "selection-lock.json").unlink()
    with pytest.raises(FileNotFoundError):
        runner.run("validation", 1)
    assert len(study.calls) == 3


def test_resume_skips_completed_cases_and_preserves_selection(study):
    runner = study.runner
    assert runner.run("calibration", 2)["completed"] == 2
    assert runner.run("calibration", 3)["completed"] == 3
    assert len(study.calls) == 3
    study.select()
    assert runner.run("validation", 2)["completed"] == 2
    assert len(study.calls) == 5
    _, _, _, rows = runner.load_study()
    assert all(
        row["selection_sha256"] == runner.sha(runner.HERE / "selection.json") for row in rows["validation"]
    )
    assert len({row["attempt_id"] for group in rows.values() for row in group}) == 5
    assert all(right - left >= 0.24 for left, right in zip(study.calls, study.calls[1:]))
    assert runner.verify()["verified"] == {"calibration": 3, "validation": 2}


def test_selection_and_calibration_changes_block_validation(study):
    runner = study.runner
    runner.run("calibration", 3)
    study.select()
    selection_path = runner.HERE / "selection.json"
    original = selection_path.read_bytes()
    selection = json.loads(original)
    selection["rule"]["threshold"] = 0.9
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="frozen lock"):
        runner.run("validation", 1)
    selection_path.write_bytes(original)
    with runner.ledger_path("calibration").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="calibration ledger changed"):
        runner.run("validation", 1)
    assert len(study.calls) == 3


def test_failure_stops_dispatch_drains_and_never_retries(study):
    runner = study.runner
    study.prepare(calibration=7)
    study.options["fail_first"] = True
    result = runner.run("calibration", 7)
    assert result["status"] == "stopped"
    assert 1 <= result["new_attempts"] <= 4
    assert len(runner.ledger_path("calibration").read_text().splitlines()) == len(study.calls)
    assert not list((runner.DATA / "pending").glob("*.json"))
    before = len(study.calls)
    with pytest.raises(ValueError, match="Prior failed or unpriced"):
        runner.run("calibration", 7)
    assert len(study.calls) == before


def test_unpriced_success_stops_new_dispatch(study):
    study.options["unknown_first"] = True
    result = study.runner.run("calibration", 3)
    assert result["status"] == "stopped"
    assert result["new_attempts"] == 1
    with pytest.raises(ValueError, match="unpriced"):
        study.runner.verify()
    assert len(study.calls) == 1


def test_budget_reserves_each_inflight_request(study):
    study.prepare(calibration=4, budget=0.025)
    study.options["delay"] = 0.65
    result = study.runner.run("calibration", 4)
    assert result["completed"] == 4
    assert study.active["maximum"] <= 2
    assert result["known_cost_usd"] == pytest.approx(0.00004)


def test_frozen_source_change_blocks_requests(study):
    study.source.write_text("changed")
    with pytest.raises(ValueError, match="source file changed"):
        study.runner.run("calibration", 1)
    assert not study.calls


def test_pending_intent_blocks_requests(study):
    pending = study.runner.DATA / "pending"
    pending.mkdir()
    (pending / "unresolved.json").write_text("{}")
    with pytest.raises(ValueError, match="Unreconciled"):
        study.runner.run("calibration", 1)
    assert not study.calls
