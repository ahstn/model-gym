"""HTTP error allowances stay narrow and preserve cached failures and prior receipts."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_metadata_v3_reallocation import reallocated as reallocated  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_recovery import failed as failed  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_runtime import inputs as inputs  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_shared_pool import shared as shared  # noqa: PLC0414 - shared fixture
from test_metadata_v3_timeouts import extended as extended  # noqa: PLC0414 - fixture dependency

from bench_protocol import MODELS
from tracks.metadata_v3 import http_error_recovery, runtime


@pytest.fixture
def http_failure(shared):
    members, prompt, previous, protocol_hash, _oldledger, _oldraw, _oldcalls, _oldsend, *_rest = shared
    member = {**members[0], "id": "http-error", "group_id": "http-error"}
    path = previous.root / "texture/candidates/seed/train.jsonl"
    calls = []

    def send(endpoint, payload):
        calls.append(payload)
        return {
            "endpoint": endpoint,
            "requested_model": payload["model"],
            "status": 504,
            "seconds": 55.8,
            "response": {"error": {"message": "Gateway error", "code": 504}},
        }

    with pytest.raises(runtime.BudgetStop, match="Unknown request cost"):
        runtime.evaluate([member], prompt, "texture", path, previous, protocol_hash, "train", send)
    raw = path.read_bytes()
    prior_receipts = {p: p.read_bytes() for p in (previous.root / "recovery-receipts").glob("*.json")}
    amendment = http_error_recovery.freeze_http_extension(
        previous.shared_pool_path, previous.root, "Explicit HTTP504 fixture review"
    )
    budget = http_error_recovery.HTTPBudget(
        previous.root,
        amendment_path=previous.amendment_path,
        reallocation_path=previous.reallocation_path,
        extension_path=previous.extension_path,
        shared_pool_path=previous.shared_pool_path,
        http_extension_path=amendment,
    )
    budget.approve_failures()
    return member, prompt, budget, protocol_hash, path, raw, calls, send, prior_receipts


def test_http504_cached_reward_zero_no_retry_receipts_unchanged(http_failure):
    member, prompt, budget, protocol_hash, path, raw, calls, send, old = http_failure
    result = http_error_recovery.evaluate_http(
        [member], prompt, "texture", path, budget, protocol_hash, "train", post_fn=send
    )
    assert path.read_bytes() == raw and len(calls) == 1
    assert result["per_image"][0]["reward"] == 0 and result["per_image"][0]["seconds"] == 55.8
    assert all(p.read_bytes() == value for p, value in old.items())
    receipt = json.loads(budget.receipt_path(path, runtime.read_ledger(path)[0]).read_text())
    assert receipt["version"] == 5 and receipt["status"] == 504
    assert receipt["http_extension_sha256"] == budget.http_extension_sha256
    summary = http_error_recovery.http_summary(budget.root, budget.http_extension_path)
    assert summary["costs"]["unknown_cost_count"] == 4 and summary["costs"]["effective_cost_usd"] is None
    assert summary["costs"]["reserved_unverified_usd"] == 0.2
    assert summary["costs"]["known_spend_budget_by_arm"] == {arm: 0.65 for arm in runtime.ARMS}
    assert summary["costs"]["global_budget_usd"] == 5


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_only_supported_error_codes_are_eligible(http_failure, status):
    _member, _prompt, budget, protocol_hash, path, _raw, _calls, _send, _old = http_failure
    row = runtime.read_ledger(path)[0]
    row.update(status=status, response={"error": {"message": "error", "code": status}})
    assert http_error_recovery.http_eligible(row, protocol_hash)
    assert budget.arm_usd == 0.65 and budget.reserve_usd == 0.01


@pytest.mark.parametrize(
    "changes",
    [
        {"status": 429},
        {"status": 501},
        {"requested_model": MODELS["luna"]},
        {"generation_id": "id"},
        {"usage": {"cost": None}},
        {"cost_usd": 0},
        {"response": {"error": {"message": "error", "code": 504}, "id": "id"}},
        {"response": {"error": {"message": "error", "code": 504, "usage": {}}}},
        {"response": {"error": {"message": "error", "code": 500}}},
        {"response": {"error": "error"}},
    ],
)
def test_other_unknowns_receive_no_http_allowance(http_failure, changes):
    _member, _prompt, budget, protocol_hash, path, _raw, _calls, _send, _old = http_failure
    row = {**runtime.read_ledger(path)[0], **changes}
    assert not http_error_recovery.http_eligible(row, protocol_hash)
    runtime.append_record(
        budget.root / "camera/unsupported.jsonl", {**row, "metadata_arm": "camera", "sample": "unsupported"}
    )
    with pytest.raises(runtime.BudgetStop, match="Unsupported unknown"):
        budget.approve_failures()


def test_new_http_error_circuit_stops_before_manual_cached_resume(http_failure):
    member, prompt, budget, protocol_hash, path, raw, calls, send, _old = http_failure
    next_member = {**member, "id": "next", "group_id": "next"}
    with pytest.raises(runtime.BudgetStop, match="verify network"):
        http_error_recovery.evaluate_http(
            [member, next_member], prompt, "texture", path, budget, protocol_hash, "train", post_fn=send
        )
    assert len(calls) == 2 and path.read_bytes().startswith(raw)
    result = http_error_recovery.evaluate_http(
        [member, next_member], prompt, "texture", path, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2 and [row["reward"] for row in result["per_image"]] == [0, 0]


def test_error_metadata_cannot_hide_generation_or_usage(http_failure):
    _member, _prompt, _budget, protocol_hash, path, _raw, _calls, _send, _old = http_failure
    row = runtime.read_ledger(path)[0]
    for metadata in ({"generation_id": "id"}, {"usage": {}}, {"other": "extra"}):
        changed = {**row, "response": {"error": {"message": "error", "code": 504, "metadata": metadata}}}
        assert not http_error_recovery.http_eligible(changed, protocol_hash)


def test_future_transport_after_http_keeps_all_allowances_without_scope_patch(http_failure):
    member, prompt, budget, protocol_hash, path, _raw, _calls, _send, _old = http_failure
    future = {**member, "id": "future-timeout", "group_id": "future-timeout"}
    calls = []
    original_unknown_method = budget.unknown_rows.__func__

    def timeout(endpoint, payload):
        calls.append(payload)
        return {
            "endpoint": endpoint,
            "requested_model": payload["model"],
            "transport_error": "ReadTimeout",
            "seconds": 180.1,
        }

    with pytest.raises(runtime.BudgetStop, match="verify network"):
        http_error_recovery.evaluate_http(
            [member, future], prompt, "texture", path, budget, protocol_hash, "train", post_fn=timeout
        )
    assert len(calls) == 1
    totals = budget.approve_failures()
    assert len(calls) == 1 and budget.unknown_rows.__func__ is original_unknown_method
    assert totals["unknown_cost_count"] == 5
    assert totals["reserved_unverified_usd"] == 0.25
    assert totals["effective_cost_usd"] is None
    receipt = json.loads(budget.receipt_path(path, runtime.read_ledger(path)[1]).read_text())
    assert (
        receipt["version"] == 5 and receipt["transport_error"] == "ReadTimeout" and receipt["status"] is None
    )
    result = http_error_recovery.evaluate_http(
        [member, future], prompt, "texture", path, budget, protocol_hash, "train", post_fn=timeout
    )
    assert len(calls) == 1 and [row["reward"] for row in result["per_image"]] == [0, 0]
