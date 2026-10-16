"""Operational recovery preserves raw unknown costs, rewards and original execution pins."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_metadata_v3_runtime import inputs as inputs  # noqa: PLC0414 - shared pytest fixture

from bench_protocol import MODELS
from tracks.metadata_v3 import cost_recovery as recovery
from tracks.metadata_v3 import runtime


@pytest.fixture
def failed(inputs):
    sample, prompt, original = inputs
    protocol = original.root.parent / "protocol.json"
    runtime.write_frozen(
        protocol,
        {
            "global_budget_usd": 5.0,
            "arm_budget_usd": 0.65,
            "max_task_calls_per_arm": 4000,
            "reserve_per_inflight_usd": 0.01,
        },
    )
    pricing = protocol.with_name("pricing.json")
    runtime.write_frozen(
        pricing,
        {
            "url": "https://openrouter.ai/api/v1/models?output_modalities=decisions",
            "models": [
                {
                    "id": MODELS["decider"],
                    "context_length": 262144,
                    "pricing": {"prompt": "0.00000002", "completion": "0"},
                }
            ],
        },
    )
    protocol_hash = recovery.sha(protocol)
    path = original.root / "image_only/candidates/seed/train.jsonl"
    calls = []

    def send(endpoint, payload):
        calls.append(payload)
        return {
            "endpoint": endpoint,
            "requested_model": payload["model"],
            "transport_error": "ConnectionError",
            "seconds": 0.01,
        }

    with pytest.raises(runtime.BudgetStop, match="Unknown request cost"):
        runtime.evaluate([sample], prompt, "image_only", path, original, protocol_hash, "train", send)
    raw = path.read_bytes()
    amendment = recovery.freeze_amendment(
        protocol, original.root, pricing, "Reviewed fixture; actual charge unknown"
    )
    budget = recovery.RecoveryBudget(original.root, amendment_path=amendment)
    budget.approve_failures()
    return sample, prompt, budget, protocol_hash, path, raw, calls, send


def test_cached_failure_metrics_bypass_keeps_bytes_actual_unknown_and_zero_reward(failed):
    sample, prompt, budget, protocol_hash, path, raw, calls, send = failed
    result = recovery.evaluate_recovered(
        [sample], prompt, "image_only", path, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 1 and path.read_bytes() == raw
    assert result["per_image"][0]["reward"] == 0 and result["per_image"][0]["valid"] is False
    assert result["ledger_records"][0]["effective_cost_usd"] is None
    assert result["summary"]["operations"]["effective_provider_inference_cost_usd"] is None
    totals = budget.totals()
    assert totals["unknown_cost_count"] == 1 and totals["effective_cost_usd"] is None
    assert totals["known_reported_usd"] == 0 and totals["reserved_unverified_usd"] == 0.05
    assert totals["budget_accounted_usd"] == 0.05
    assert recovery.allowance_summary(budget.root, budget.amendment_path)["costs"] == totals


def test_prospective_failure_new_id_is_not_retried_and_records_separate_receipt(failed):
    sample, prompt, budget, protocol_hash, path, raw, calls, send = failed
    members = [sample, {**sample, "id": "second", "group_id": "second"}]
    result = recovery.evaluate_recovered(
        members, prompt, "image_only", path, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2 and path.read_bytes().startswith(raw)
    assert [row["reward"] for row in result["per_image"]] == [0, 0]
    assert budget.totals()["reserved_unverified_usd"] == 0.1
    assert len(list((budget.root / "recovery-receipts").glob("*.json"))) == 2
    recovery.evaluate_recovered(
        members, prompt, "image_only", path, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "changed",
    [
        {"record_type": "reflection"},
        {"requested_model": MODELS["luna"]},
        {"status": 500},
        {"generation_id": "unknown-id"},
        {"response": {"error": "failure"}},
        {"usage": {"cost": None}},
        {"transport_error": "Timeout"},
        {"protocol_sha256": "changed"},
    ],
)
def test_other_unknown_failures_do_not_receive_allowances(failed, changed):
    _sample, _prompt, budget, _protocol_hash, path, _raw, _calls, _send = failed
    row = runtime.read_ledger(path)[0]
    runtime.append_record(budget.root / "camera/other.jsonl", {**row, "metadata_arm": "camera", **changed})
    with pytest.raises(runtime.BudgetStop, match="Unapproved"):
        budget.approve_failures()
    assert len(list((budget.root / "recovery-receipts").glob("*.json"))) == 1


def test_global_arm_caps_and_prospective_inflight_allowance(failed):
    _sample, _prompt, budget, _protocol_hash, path, _raw, _calls, _send = failed
    budget.begin("active", "image_only", "task", path)
    assert budget.pending()["active"]["reserved_usd"] == 0.05
    budget.finish("active")
    runtime.append_record(
        budget.root / "image_only/known.jsonl",
        {"metadata_arm": "image_only", "record_type": "task", "effective_cost_usd": 0.6},
    )
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        budget.check("image_only", 1)
    runtime.append_record(
        budget.root / "tone/known.jsonl",
        {"metadata_arm": "tone", "record_type": "task", "effective_cost_usd": 4.3},
    )
    assert budget.totals()["budget_accounted_usd"] == pytest.approx(4.95)
    runtime.append_record(
        budget.root / "camera/known.jsonl",
        {"metadata_arm": "camera", "record_type": "task", "effective_cost_usd": 0.02},
    )
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        budget.check("camera", 1)


@pytest.mark.parametrize("change", ["receipt", "prefix", "protocol", "pricing"])
def test_frozen_recovery_evidence_changes_fail_closed(failed, change):
    _sample, _prompt, budget, _protocol_hash, path, raw, _calls, _send = failed
    if change == "receipt":
        receipt_path = next((budget.root / "recovery-receipts").glob("*.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt["allowance_usd"] = 0.01
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="receipt changed"):
            budget.totals()
    elif change == "prefix":
        path.write_bytes(b" " + raw)
        with pytest.raises(ValueError, match="prefix changed"):
            budget.totals()
    else:
        key = "original_protocol_path" if change == "protocol" else "pricing_capture_path"
        Path(budget.amendment[key]).write_text("{}")
        with pytest.raises(ValueError, match="changed"):
            recovery.validate_amendment(budget.amendment_path, budget.root)


def test_freeze_refuses_any_held_out_attempt_and_keeps_original_files(failed):
    _sample, _prompt, budget, _protocol_hash, path, raw, _calls, _send = failed
    record = copy.deepcopy(runtime.read_ledger(path)[0])
    record["split"] = "test"
    runtime.append_record(budget.root / "image_only/final/test.jsonl", record)
    with pytest.raises(ValueError, match="before any held-out"):
        recovery.freeze_amendment(
            budget.amendment["original_protocol_path"],
            budget.root,
            budget.amendment["pricing_capture_path"],
            "Second review",
        )
    assert path.read_bytes() == raw


def test_driver_overrides_are_scoped_and_preserve_original_reserve_field(failed, monkeypatch):
    from tracks.metadata_v3 import recover_study, study

    _sample, prompt, budget, _protocol_hash, _path, _raw, _calls, _send = failed
    protocol = json.loads(Path(budget.amendment["original_protocol_path"]).read_text())
    monkeypatch.setattr(study, "HERE", budget.amendment_path.parent)
    monkeypatch.setattr(study, "RUNS", budget.root)
    original_load = lambda: (protocol, {"samples": []}, prompt, runtime.StudyBudget(budget.root))
    monkeypatch.setattr(study, "load_study", original_load)
    original_evaluate, original_search = study.evaluate, study.optimise_arm
    received = []

    def search(*args, **kwargs):
        received.append(kwargs["evaluate_fn"])
        return {"test": True}

    def run(mode):
        assert mode == "optimise"
        _protocol, _dataset, _baseline, active_budget = study.load_study()
        assert isinstance(active_budget, recovery.RecoveryBudget)
        assert active_budget.reserve_usd == 0.01
        assert study.evaluate is recovery.evaluate_recovered
        study.optimise_arm()

    monkeypatch.setattr(recover_study, "optimise_arm", search)
    monkeypatch.setattr(study, "run", run)
    recover_study.run("optimise")
    assert received == [recovery.evaluate_recovered]
    assert study.load_study is original_load
    assert study.evaluate is original_evaluate and study.optimise_arm is original_search
