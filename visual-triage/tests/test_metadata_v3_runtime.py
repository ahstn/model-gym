"""Offline proof of metadata request identity, cost limits and final gates."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import MODELS, digest, image_part
from tracks.metadata_v3 import runtime
from tracks.metadata_v3.features import extract_features


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    image = tmp_path / "view.jpg"
    Image.new("RGB", (12, 8), (20, 40, 60)).save(image)
    features = extract_features(image, image)
    monkeypatch.setattr(runtime, "image_part", lambda sample: image_part(sample, root=tmp_path))
    sample = {
        "id": "one",
        "group_id": "one",
        "split": "train",
        "image": str(image),
        "delivered_sha256": __import__("hashlib").sha256(image.read_bytes()).hexdigest(),
        "gold": 0,
        "features": features,
        "features_sha256": digest(features),
    }
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())
    return sample, prompt, runtime.StudyBudget(tmp_path / "runs")


def response(probability=0.2, cost=0.001):
    return {
        "status": 200,
        "seconds": 0.1,
        "response": {
            "model": MODELS["decider"],
            "usage": {"cost": cost},
            "answers": {
                "ai_probability": {"type": "noul", "noul": probability},
                "medium": {
                    "type": "choice",
                    "choice": "photo",
                    "probabilities": {"photo": 1, "render": 0, "graphic": 0, "unknown": 0},
                },
                "uncertainty": {
                    "type": "choice",
                    "choice": "high",
                    "probabilities": {"low": 0, "medium": 0, "high": 1},
                },
            },
        },
    }


def test_decider_features_resume_and_old_metric_contract(inputs):
    sample, prompt, budget = inputs
    output = budget.root / "camera/candidates/seed/train.jsonl"
    calls = []

    def send(endpoint, payload):
        calls.append(payload)
        return response()

    result = runtime.evaluate([sample], prompt, "camera", output, budget, "protocol", "train", send)
    raw = output.read_bytes()
    repeated = runtime.evaluate([sample], prompt, "camera", output, budget, "protocol", "train", send)
    assert len(calls) == 1 and output.read_bytes() == raw
    assert calls[0]["model"] == MODELS["decider"]
    assert "camera" in calls[0]["state"][0] and "tone" not in calls[0]["state"][0]
    assert "source" not in calls[0]["state"][0]
    assert result["per_image"][0]["reward"] == pytest.approx(0.96)
    assert repeated["summary"]["metrics"]["brier_valid"] == pytest.approx(0.04)
    assert runtime.read_ledger(output)[0]["prompt_sha256"] == digest(prompt)
    changed = {**prompt, "context": prompt["context"] + " Changed."}
    with pytest.raises(ValueError, match="prompt"):
        runtime.evaluate([sample], changed, "camera", output, budget, "protocol", "train", send)


def test_unknown_cost_retained_and_stops_all_arms_without_retry(inputs):
    sample, prompt, budget = inputs
    output = budget.root / "image_only/candidates/seed/train.jsonl"
    calls = []

    def send(endpoint, payload):
        calls.append(payload)
        return {"transport_error": "ConnectionError", "seconds": 0.01}

    with pytest.raises(runtime.BudgetStop, match="Unknown request cost"):
        runtime.evaluate([sample], prompt, "image_only", output, budget, "protocol", "train", send)
    raw = output.read_bytes()
    for arm in ("image_only", "tone"):
        path = output if arm == "image_only" else budget.root / arm / "candidates/seed/train.jsonl"
        with pytest.raises(runtime.BudgetStop, match="Unknown request cost"):
            runtime.evaluate([sample], prompt, arm, path, budget, "protocol", "train", send)
    assert len(calls) == 1 and output.read_bytes() == raw
    assert budget.totals()["effective_cost_usd"] is None
    assert runtime.read_ledger(output)[0]["normalized"] is None


def test_guard_counts_reflection_and_other_arm_requests(inputs):
    _sample, _prompt, budget = inputs
    for arm, kind, cost in (("image_only", "task", 0.3), ("tone", "reflection", 0.3)):
        runtime.append_record(
            budget.root / arm / "cost.jsonl",
            {"record_type": kind, "metadata_arm": arm, "effective_cost_usd": cost},
        )
    assert budget.totals()["known_reported_usd"] == pytest.approx(0.6)
    limited = runtime.StudyBudget(budget.root, global_usd=0.6, arm_usd=0.5)
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        limited.check("camera")
    with pytest.raises(ValueError, match="inside"):
        budget.ledger_path(budget.root.parent / "outside.jsonl", "camera")


def test_features_hash_split_and_final_selection_gate(inputs):
    sample, prompt, budget = inputs
    path = budget.root / "resolution/candidates/seed/train.jsonl"
    with pytest.raises(ValueError, match="feature hash"):
        runtime.evaluate(
            [{**sample, "features_sha256": "changed"}],
            prompt,
            "resolution",
            path,
            budget,
            "protocol",
            "train",
        )
    with pytest.raises(ValueError, match="Wrong split"):
        runtime.evaluate(
            [{**sample, "split": "test"}], prompt, "resolution", path, budget, "protocol", "train"
        )
    with pytest.raises(FileNotFoundError):
        runtime.evaluate(
            [{**sample, "split": "test"}],
            prompt,
            "resolution",
            budget.root / "resolution/final/test.jsonl",
            budget,
            "protocol",
            "test",
        )


def test_four_workers_reserve_before_dispatch_and_drain_unknown(inputs):
    import threading
    import time

    sample, prompt, budget = inputs
    samples = [{**sample, "id": str(index), "group_id": str(index)} for index in range(8)]
    barrier = threading.Barrier(4)
    seen = []

    def send(endpoint, payload):
        seen.append(len(budget.pending()))
        barrier.wait(timeout=5)
        time.sleep(0.01)
        return {"transport_error": "ConnectionError", "seconds": 0.01}

    output = budget.root / "image_only/candidates/seed/train.jsonl"
    with pytest.raises(runtime.BudgetStop, match="Unknown"):
        runtime.evaluate(
            samples, prompt, "image_only", output, budget, "protocol", "train", send, concurrency=4
        )
    assert len(seen) == len(runtime.read_ledger(output)) == 4
    assert max(seen) == 4 and budget.pending() == {}
    assert budget.totals()["unknown_cost_count"] == 4


def test_pending_reservations_count_in_budget_and_crash_blocks_resume(inputs):
    sample, prompt, _budget = inputs
    budget = runtime.StudyBudget(_budget.root, global_usd=0.02, arm_usd=0.02, reserve_usd=0.01)
    output = budget.root / "camera/candidates/seed/train.jsonl"
    budget.root.mkdir()
    budget.begin("first", "camera", "task", output)
    budget.begin("second", "camera", "task", output)
    assert budget.totals()["reserved_inflight_usd"] == 0.02
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        budget.check("camera", 1)
    with pytest.raises(runtime.BudgetStop, match="reconciliation"):
        runtime.evaluate([sample], prompt, "camera", output, budget, "protocol", "train")


def test_all_selections_before_final_and_controls(inputs):
    sample, prompt, budget = inputs
    for arm in runtime.ARMS:
        path = budget.root / arm / "candidates/seed/prompt.json"
        runtime.write_frozen(path, prompt)
        runtime.write_frozen(
            budget.root / arm / "selection.json",
            {
                "metadata_arm": arm,
                "protocol_sha256": "protocol",
                "selected_prompt_path": str(path),
                "selected_prompt_sha256": digest(prompt),
            },
        )
    test_sample = {**sample, "split": "test"}
    for directory in ("final", "controls"):
        result = runtime.evaluate(
            [test_sample],
            prompt,
            "image_only",
            budget.root / f"image_only/{directory}/selected.jsonl",
            budget,
            "protocol",
            "test",
            lambda endpoint, payload: response(),
        )
        assert result["summary"]["expected"] == 1
    assert budget.totals()["known_reported_usd"] == pytest.approx(0.002)


def test_totals_cache_reuses_unchanged_files_and_audits_appends_and_restarts(inputs, monkeypatch):
    _sample, _prompt, budget = inputs
    first = budget.root / "camera/first.jsonl"
    second = budget.root / "tone/second.jsonl"
    row = {"record_type": "task", "metadata_arm": "camera", "effective_cost_usd": 0.001}
    runtime.append_record(first, row)
    runtime.append_record(second, {**row, "metadata_arm": "tone"})
    reads = []
    original = runtime.read_ledger

    def tracked(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(runtime, "read_ledger", tracked)
    assert budget.totals()["known_reported_usd"] == 0.002
    assert len(reads) == 2
    budget.totals()
    assert len(reads) == 2
    runtime.append_record(first, row)
    assert budget.totals()["known_reported_usd"] == 0.003
    assert reads.count(first) == 2 and reads.count(second) == 1
    fresh = runtime.StudyBudget(budget.root)
    assert fresh.totals()["known_reported_usd"] == 0.003
    assert reads.count(first) == 3 and reads.count(second) == 2
    runtime.append_record(second, {**row, "metadata_arm": "tone", "effective_cost_usd": None})
    with pytest.raises(runtime.BudgetStop, match="Unknown"):
        budget.check("camera")
    assert budget.totals()["effective_cost_usd"] is None


def test_same_size_rewrite_with_restored_mtime_invalidates_cache(inputs):
    import os

    _sample, _prompt, budget = inputs
    path = budget.root / "camera/cost.jsonl"
    runtime.append_record(
        path, {"record_type": "task", "metadata_arm": "camera", "effective_cost_usd": 0.001}
    )
    assert budget.totals()["known_reported_usd"] == 0.001
    previous = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"0.001", b"0.009"))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert budget.totals()["known_reported_usd"] == 0.009
