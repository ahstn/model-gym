"""ReadTimeout extension preserves previous receipts and every failed raw attempt."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_metadata_v3_reallocation import reallocated as reallocated  # noqa: PLC0414 - shared fixture
from test_metadata_v3_recovery import failed as failed  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_runtime import inputs as inputs  # noqa: PLC0414 - fixture dependency

from bench_protocol import MODELS
from tracks.metadata_v3 import runtime, timeout_recovery


@pytest.fixture
def extended(reallocated):
    sample, prompt, prior, protocol_hash, old_ledger, old_raw, old_calls, _send, _old, receipts = reallocated
    members = [{**sample, "id": f"timeout-{index}", "group_id": f"timeout-{index}"} for index in range(2)]
    ledger = prior.root / "resolution/candidates/seed/train.jsonl"
    calls = []

    def send(endpoint, payload):
        calls.append(payload)
        return {
            "endpoint": endpoint,
            "requested_model": payload["model"],
            "transport_error": "ReadTimeout",
            "seconds": 180.1,
        }

    with pytest.raises(runtime.BudgetStop, match="Unknown request cost"):
        runtime.evaluate(
            members, prompt, "resolution", ledger, prior, protocol_hash, "train", send, concurrency=2
        )
    raw = ledger.read_bytes()
    path = timeout_recovery.freeze_extension(
        prior.reallocation_path, prior.root, "Explicit timeout review; no inference about host sleep"
    )
    budget = timeout_recovery.TimeoutBudget(
        prior.root,
        amendment_path=prior.amendment_path,
        reallocation_path=prior.reallocation_path,
        extension_path=path,
    )
    budget.approve_failures()
    return (
        members,
        prompt,
        budget,
        protocol_hash,
        ledger,
        raw,
        calls,
        send,
        old_ledger,
        old_raw,
        old_calls,
        receipts,
    )


def test_old_connections_and_two_new_timeouts_remain_raw_unknown_zero_reward(extended):
    (
        members,
        prompt,
        budget,
        protocol_hash,
        ledger,
        raw,
        calls,
        send,
        old_ledger,
        old_raw,
        old_calls,
        receipts,
    ) = extended
    assert all(path.read_bytes() == value for path, value in receipts.items())
    assert old_ledger.read_bytes() == old_raw and len(old_calls) == 1
    result = timeout_recovery.evaluate_timeouts(
        members, prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2 and ledger.read_bytes() == raw
    assert [row["reward"] for row in result["per_image"]] == [0, 0]
    assert all(row["effective_cost_usd"] is None for row in result["ledger_records"])
    costs = timeout_recovery.timeout_summary(budget.root, budget.extension_path)["costs"]
    assert costs["unknown_cost_count"] == 3 and costs["effective_cost_usd"] is None
    assert costs["reserved_unverified_usd"] == pytest.approx(0.15)
    assert costs["guard_budget_by_arm"]["image_only"] == 1.10
    assert costs["guard_budget_by_arm"]["resolution"] == 0.65 and costs["global_budget_usd"] == 5
    new_receipts = [
        json.loads(path.read_text())
        for path in (budget.root / "recovery-receipts").glob("*.json")
        if path not in receipts
    ]
    assert len(new_receipts) == 2
    assert all(
        receipt["version"] == 3 and receipt["transport_error"] == "ReadTimeout" for receipt in new_receipts
    )
    assert all(receipt["timeout_extension_sha256"] == budget.extension_sha256 for receipt in new_receipts)
    assert budget.arm_usd == 0.65 and budget.reserve_usd == 0.01


@pytest.mark.parametrize(
    "changes",
    [
        {"transport_error": "ConnectTimeout"},
        {"transport_error": "Timeout"},
        {"requested_model": MODELS["luna"]},
        {"status": 500},
        {"generation_id": "id"},
        {"response": {"error": "failure"}},
        {"usage": {"cost": None}},
        {"cost_usd": 0.01},
    ],
)
def test_no_other_unknown_failure_receives_timeout_allowance(extended, changes):
    (
        _members,
        _prompt,
        budget,
        _protocol_hash,
        ledger,
        _raw,
        _calls,
        _send,
        _oldledger,
        _oldraw,
        _oldcalls,
        _receipts,
    ) = extended
    row = runtime.read_ledger(ledger)[0]
    runtime.append_record(
        budget.root / "camera/other.jsonl", {**row, "sample": "other", "metadata_arm": "camera", **changes}
    )
    before = len(list((budget.root / "recovery-receipts").glob("*.json")))
    with pytest.raises(runtime.BudgetStop, match="Unsupported unknown"):
        budget.approve_failures()
    assert len(list((budget.root / "recovery-receipts").glob("*.json"))) == before


def test_new_timeout_circuit_stops_then_manual_cached_resume(extended):
    members, prompt, budget, protocol_hash, ledger, raw, calls, send, *_rest = extended
    future = {**members[0], "id": "future", "group_id": "future"}
    with pytest.raises(runtime.BudgetStop, match="verify network"):
        timeout_recovery.evaluate_timeouts(
            [*members, future], prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
        )
    assert len(calls) == 3 and ledger.read_bytes().startswith(raw)
    assert budget.totals()["unknown_cost_count"] == 4
    result = timeout_recovery.evaluate_timeouts(
        [*members, future], prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 3 and len(result["per_image"]) == 3
    assert budget.totals()["reserved_unverified_usd"] == 0.2


def test_third_receipt_cannot_claim_earlier_amendment_approved_timeout(extended):
    _members, _prompt, budget, _protocol_hash, ledger, _raw, _calls, _send, *_rest = extended
    row = runtime.read_ledger(ledger)[0]
    path = budget.receipt_path(ledger, row)
    receipt = json.loads(path.read_text())
    receipt["timeout_extension_sha256"] = budget.amendment_sha256
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt changed"):
        budget.totals()
