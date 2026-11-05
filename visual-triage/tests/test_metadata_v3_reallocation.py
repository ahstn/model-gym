"""Second guard amendment preserves raw failures and the original resume contract."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_metadata_v3_recovery import failed as failed  # noqa: PLC0414 - shared pytest fixture
from test_metadata_v3_runtime import inputs as inputs  # noqa: PLC0414 - fixture dependency

from tracks.metadata_v3 import budget_reallocation as second
from tracks.metadata_v3 import runtime


@pytest.fixture
def reallocated(failed):
    sample, prompt, original, protocol_hash, ledger, raw, calls, send = failed
    old_amendment = original.amendment_path.read_bytes()
    receipt_raw = {p: p.read_bytes() for p in (original.root / "recovery-receipts").glob("*.json")}
    path = second.freeze_reallocation(
        original.amendment_path, original.root, "Reviewed headroom shift before held-out calls"
    )
    budget = second.ReallocatedBudget(
        original.root, amendment_path=original.amendment_path, reallocation_path=path
    )
    return sample, prompt, budget, protocol_hash, ledger, raw, calls, send, old_amendment, receipt_raw


def test_original_contract_and_raw_failures_preserved_above_old_cap(reallocated):
    sample, prompt, budget, protocol_hash, ledger, raw, calls, send, old_amendment, receipt_raw = reallocated
    runtime.append_record(
        budget.root / "image_only/known.jsonl",
        {"metadata_arm": "image_only", "record_type": "task", "effective_cost_usd": 0.7},
    )
    assert budget.arm_usd == 0.65 and budget.reserve_usd == 0.01 and budget.global_usd == 5
    assert budget.check("image_only", 1)["arms"]["image_only"]["guard_budget_usd"] == 1.10
    result = second.evaluate_reallocated(
        [sample], prompt, "image_only", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert result["per_image"][0]["reward"] == 0 and len(calls) == 1
    assert ledger.read_bytes() == raw and budget.amendment_path.read_bytes() == old_amendment
    assert all(path.read_bytes() == expected for path, expected in receipt_raw.items())
    costs = second.reallocation_summary(budget.root, budget.reallocation_path)["costs"]
    assert costs["effective_cost_usd"] is None and costs["unknown_cost_count"] == 1
    assert costs["budget_accounted_usd"] == pytest.approx(0.75)


def test_other_arms_caps_and_global_cap_unchanged(reallocated):
    _sample, _prompt, budget, _protocol_hash, _ledger, _raw, _calls, _send, _old, _receipts = reallocated
    runtime.append_record(
        budget.root / "camera/known.jsonl",
        {"metadata_arm": "camera", "record_type": "task", "effective_cost_usd": 0.61},
    )
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        budget.check("camera", 1)
    for arm in runtime.ARMS:
        if arm in ("camera", "image_only"):
            continue
        runtime.append_record(
            budget.root / arm / "known.jsonl",
            {"metadata_arm": arm, "record_type": "task", "effective_cost_usd": 0.65},
        )
    runtime.append_record(
        budget.root / "image_only/known.jsonl",
        {"metadata_arm": "image_only", "record_type": "task", "effective_cost_usd": 1.03},
    )
    assert budget.totals()["budget_accounted_usd"] == pytest.approx(4.94)
    budget.begin("other-active", "camera", "task", budget.root / "camera/known.jsonl")
    with pytest.raises(runtime.BudgetStop, match="spend guard"):
        budget.check("image_only", 1)
    assert budget.totals()["global_budget_usd"] == 5


def test_new_unknown_batch_stops_once_then_cached_manual_resume(reallocated):
    sample, prompt, budget, protocol_hash, ledger, raw, calls, send, _old, _receipts = reallocated
    samples = [
        sample,
        {**sample, "id": "new", "group_id": "new"},
        {**sample, "id": "later", "group_id": "later"},
    ]
    with pytest.raises(runtime.BudgetStop, match="verify network"):
        second.evaluate_reallocated(
            samples, prompt, "image_only", ledger, budget, protocol_hash, "train", post_fn=send
        )
    assert len(calls) == 2 and ledger.read_bytes().startswith(raw)
    assert budget.totals()["unknown_cost_count"] == 2
    assert budget.totals()["reserved_unverified_usd"] == 0.1
    # A manual cached resume returns completed rows without retrying either failed ID.
    result = second.evaluate_reallocated(
        samples[:2], prompt, "image_only", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2 and [row["reward"] for row in result["per_image"]] == [0, 0]


def test_second_amendment_or_original_receipt_mutation_fails_closed(reallocated):
    _sample, _prompt, budget, _protocol_hash, _ledger, _raw, _calls, _send, _old, receipts = reallocated
    receipt = next(iter(receipts))
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Original failure receipt changed"):
        second.validate_reallocation(budget.reallocation_path, budget.root)
    receipt.write_bytes(receipts[receipt])
    amendment = json.loads(budget.reallocation_path.read_text())
    amendment["guard_budget_by_arm"]["camera"] = 1.10
    budget.reallocation_path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="second operational amendment"):
        second.validate_reallocation(budget.reallocation_path, budget.root)


def test_second_freeze_rejects_held_out_attempt(reallocated):
    _sample, _prompt, budget, _protocol_hash, ledger, _raw, _calls, _send, _old, _receipts = reallocated
    row = runtime.read_ledger(ledger)[0]
    runtime.append_record(budget.root / "image_only/final/test.jsonl", {**row, "split": "test"})
    with pytest.raises(ValueError, match="before held-out"):
        second.freeze_reallocation(budget.amendment_path, budget.root, "Late review forbidden")
