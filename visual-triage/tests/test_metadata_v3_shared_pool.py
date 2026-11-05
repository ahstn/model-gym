"""Shared allowances preserve unknown costs while known spend remains capped per arm."""

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_metadata_v3_reallocation import reallocated as reallocated  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_recovery import failed as failed  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_runtime import inputs as inputs  # noqa: PLC0414 - fixture dependency
from test_metadata_v3_timeouts import extended as extended  # noqa: PLC0414 - shared fixture

from bench_protocol import digest
from tracks.metadata_v3 import resume_shared_pool, runtime, shared_allowance_pool


@pytest.fixture
def shared(extended):
    (
        members,
        prompt,
        original,
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
    path = shared_allowance_pool.freeze_shared_pool(
        original.extension_path, original.root, "Reviewed shared global allowance funding"
    )
    budget = shared_allowance_pool.SharedPoolBudget(
        original.root,
        amendment_path=original.amendment_path,
        reallocation_path=original.reallocation_path,
        extension_path=original.extension_path,
        shared_pool_path=path,
    )
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


def test_allowances_use_global_pool_and_known_arm_cap_keeps_headroom(shared):
    _members, _prompt, budget, _protocol_hash, ledger, _raw, _calls, _send, *_rest = shared
    row = runtime.read_ledger(ledger)[0]
    path = budget.root / "texture/candidates/seed/train.jsonl"
    for index in range(16):
        runtime.append_record(path, {**row, "metadata_arm": "texture", "sample": f"texture-{index}"})
    runtime.append_record(
        budget.root / "texture/known.jsonl",
        {"record_type": "task", "metadata_arm": "texture", "effective_cost_usd": 0.127},
    )
    budget.approve_failures()
    totals = budget.check("texture", 1)
    assert totals["arms"]["texture"]["budget_accounted_usd"] == pytest.approx(0.927)
    assert totals["arms"]["texture"]["known_spend_budget_usd"] == 0.65
    assert "guard_budget_by_arm" not in totals and "guard_budget_usd" not in totals["arms"]["texture"]
    for index in range(4):
        budget.begin(str(index), "texture", "task", path)
    assert budget.check("texture", 1)["reserved_inflight_usd"] == 0.2
    for index in range(4):
        budget.finish(str(index))
    runtime.append_record(
        budget.root / "camera/known.jsonl",
        {"record_type": "task", "metadata_arm": "camera", "effective_cost_usd": 3.90},
    )
    with pytest.raises(runtime.BudgetStop, match="Shared global"):
        budget.check("texture", 1)
    assert budget.totals()["effective_cost_usd"] is None


def test_known_arm_cap_blocks_with_global_headroom(shared):
    _members, _prompt, budget, _protocol_hash, _ledger, _raw, _calls, _send, *_rest = shared
    runtime.append_record(
        budget.root / "texture/known.jsonl",
        {"record_type": "task", "metadata_arm": "texture", "effective_cost_usd": 0.645},
    )
    assert budget.totals()["budget_accounted_usd"] < 5
    with pytest.raises(runtime.BudgetStop, match="known-spend"):
        budget.check("texture", 1)


def test_cached_failures_above_old_arm_guard_preserve_bytes_and_receipts(shared):
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
    ) = shared
    row = runtime.read_ledger(ledger)[0]
    for index in range(12):
        runtime.append_record(ledger, {**row, "sample": f"other-{index}"})
    budget.approve_failures()
    current = ledger.read_bytes()
    assert current.startswith(raw)
    result = shared_allowance_pool.evaluate_shared_pool(
        members, prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 2 and ledger.read_bytes() == current
    assert [row["reward"] for row in result["per_image"]] == [0, 0]
    assert old_ledger.read_bytes() == old_raw and len(old_calls) == 1
    assert all(path.read_bytes() == value for path, value in receipts.items())
    summary = shared_allowance_pool.shared_pool_summary(budget.root, budget.shared_pool_path)
    assert summary["costs"]["unknown_cost_count"] == 15
    assert summary["costs"]["effective_cost_usd"] is None
    assert summary["costs"]["known_spend_budget_by_arm"] == {arm: 0.65 for arm in runtime.ARMS}


def test_diagnostics_log_no_strings_payloads_headers_or_secrets(tmp_path, monkeypatch):
    endpoint = "https://openrouter.ai/api/alpha/decisions"
    payload = {"model": "example", "secret_payload": "DO_NOT_LOG"}
    headers = {"Authorization": "Bearer DO_NOT_LOG"}
    captured = []

    def original(url, **kwargs):
        captured.append((url, kwargs))
        try:
            raise OSError(8, "DO_NOT_LOG_EXCEPTION")
        except OSError as cause:
            raise requests.ConnectionError("DO_NOT_LOG_CONNECTION") from cause

    monkeypatch.setattr(requests, "post", original)
    path = tmp_path / "diagnostics/transport.jsonl"
    with (
        resume_shared_pool.diagnostic_transport(path, tmp_path / "runs"),
        pytest.raises(requests.ConnectionError),
    ):
        requests.post(endpoint, json=payload, headers=headers, timeout=(15, 180), allow_redirects=False)
    assert requests.post is original
    assert captured[0][1]["json"] is payload and captured[0][1]["headers"] is headers
    raw = path.read_text()
    assert "DO_NOT_LOG" not in raw and "Authorization" not in raw and "secret_payload" not in raw
    row = json.loads(raw)
    assert set(row) == {"timestamp", "endpoint_path", "request_sha256", "exception_chain"}
    assert row["request_sha256"] == digest({"endpoint": endpoint, "payload": payload})
    assert row["endpoint_path"] == "/api/alpha/decisions"
    assert row["exception_chain"] == [
        {"class": "ConnectionError", "errno": None},
        {"class": "OSError", "errno": 8},
    ]
    with (
        pytest.raises(ValueError, match="outside raw runs"),
        resume_shared_pool.diagnostic_transport(tmp_path / "runs/diag.jsonl", tmp_path / "runs"),
    ):
        pass


def test_shared_pool_preserves_new_failure_circuit_stop(shared):
    members, prompt, budget, protocol_hash, ledger, raw, calls, send, *_rest = shared
    future = {**members[0], "id": "new-after-fourth", "group_id": "new-after-fourth"}
    with pytest.raises(runtime.BudgetStop, match="verify network"):
        shared_allowance_pool.evaluate_shared_pool(
            [*members, future], prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
        )
    assert len(calls) == 3 and ledger.read_bytes().startswith(raw)
    assert budget.totals()["unknown_cost_count"] == 4
    result = shared_allowance_pool.evaluate_shared_pool(
        [*members, future], prompt, "resolution", ledger, budget, protocol_hash, "train", post_fn=send
    )
    assert len(calls) == 3 and all(row["reward"] == 0 for row in result["per_image"])
