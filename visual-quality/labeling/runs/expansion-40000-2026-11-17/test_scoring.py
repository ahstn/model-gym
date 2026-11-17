"""Offline checks for new-batch provenance, resumability and stopping rules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("expansion_40000_score", Path(__file__).with_name("score.py"))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

adapter = score.adapter
batch = score.batch


@pytest.fixture
def frozen(tmp_path):
    cases = [
        {
            "case_id": f"NEW{i:04}",
            "benchmark_partition": "rehearsal",
            "image_path": str(tmp_path / f"{i}.png"),
            "view_sha256": "0" * 64,
        }
        for i in range(3)
    ]
    manifest = tmp_path / "manifest.json"
    adapter.atomic(manifest, {"cases": cases})
    adapter.atomic(tmp_path / "api-provider-snapshot.json", {"checked_at": "fixture"})
    args = SimpleNamespace(
        data=tmp_path,
        model="luna",
        phase=score.PHASE,
        manifest=manifest,
        profile_directory=score.PACKAGE / "labeling",
        limit=None,
        worker_limit=1,
    )
    batch.freeze(args, adapter, score.POLICY["luna"], [Path(score.__file__)])
    return args


def fake_job(monkeypatch, statuses=None):
    calls = []

    def run_job(plan, case, root, secret, ledger, stop):
        calls.append(case["case_id"])
        status, http_status, body = (statuses or {}).get(case["case_id"], ("valid", 200, "{}"))
        result = root / "fake" / f"{case['case_id']}.json"
        adapter.atomic(result, {"http_status": http_status, "body": body})
        return {"status": status, "raw_response_path": str(result)}

    monkeypatch.setattr(adapter, "job", run_job)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    return calls


def test_new_batch_uses_evaluated_models_and_bounded_concurrency():
    assert score.MODELS == ("luna", "deepseek", "liquid", "jev")
    assert all(policy["workers"] == 16 for policy in score.POLICY.values())
    assert adapter.POLICY["liquid"]["workers"] == 4  # Historical module was not mutated.
    profile = adapter.load_profile(score.PACKAGE / "labeling")
    assert profile["teacher"]["version"] == "photo-labeling-v2-2026-11-01"
    assert profile["routes"]["routes"]["luna"]["settings"]["reasoning"]["effort"] == "high"
    assert profile["routes"]["routes"]["deepseek"]["settings"]["reasoning_effort"] == "high"


def test_freeze_pins_execution_and_snapshot(frozen):
    _, plan, workers, _ = batch.validate_execution(frozen, adapter)
    assert workers == 1
    assert plan["manifest_sha256"] == adapter.sha(frozen.manifest)
    assert str(Path(batch.__file__).resolve()) in plan["code_pins"]
    assert str(Path(adapter.__file__).resolve()) in plan["code_pins"]
    assert str(Path(score.__file__).resolve()) in plan["code_pins"]
    assert not plan["stats_sent_to_models"]
    assert not plan["reference_scores_sent_to_models"]
    with pytest.raises(ValueError, match="already frozen"):
        batch.freeze(frozen, adapter, score.POLICY["luna"], [])
    adapter.atomic(frozen.data / "api-provider-snapshot.json", {"changed": True})
    with pytest.raises(ValueError, match="Frozen input changed"):
        batch.validate_execution(frozen, adapter)


@pytest.mark.parametrize("field,value", [("limit", 0), ("limit", -1), ("worker_limit", 0)])
def test_invalid_runtime_limits_are_rejected_before_http(frozen, field, value):
    setattr(frozen, field, value)
    with pytest.raises(ValueError, match="positive integer"):
        batch.validate_execution(frozen, adapter)


def test_smoke_subset_is_partial_and_full_run_has_exact_counts(frozen, monkeypatch):
    calls = fake_job(monkeypatch)
    frozen.limit = 1
    partial = batch.run(frozen, adapter)
    assert partial["status"] == "partial"
    assert partial["planned_cases"] == 3
    assert partial["selected_cases"] == partial["completed_cases"] == 1
    frozen.limit = None
    complete = batch.run(frozen, adapter)
    assert complete["status"] == "complete"
    assert complete["planned_cases"] == complete["selected_cases"] == complete["completed_cases"] == 3
    assert complete["outcomes"] == {"valid": 3}
    assert len(calls) == 4
    saved = adapter.read(frozen.data / "results/api/luna/expansion-status.json")
    assert saved == complete
    assert saved["plan_id"] == partial["plan_id"]


def test_content_denial_is_terminal_missing_and_does_not_trip_outage_circuit():
    circuit = batch.Circuit()
    raw = {
        "http_status": 403,
        "body": json.dumps({"error": {"message": "Request blocked by content filter: X"}}),
    }
    for _ in range(100):
        assert circuit.observe({"status": "http_failure"}, raw, adapter) is None
    assert circuit.denials == 100
    assert not circuit.recent
    assert not adapter.parsed_result(raw, {"kind": "decisions", "provider": "openrouter"})["retryable"]


@pytest.mark.parametrize("http_status", [401, 402, 403])
def test_credentials_billing_and_unrecognized_denials_stop(frozen, monkeypatch, http_status):
    _, plan, _, _ = batch.validate_execution(frozen, adapter)
    first = plan["cases"][0]["case_id"]
    calls = fake_job(monkeypatch, {first: ("http_failure", http_status, "{}")})
    state = batch.run(frozen, adapter)
    assert state["status"] == "stopped"
    assert state["completed_cases"] == 1
    assert calls == [first]
    assert "Credential" in state["stop_reason"]


def test_transient_outage_circuit_preserved():
    circuit = batch.Circuit()
    for _ in range(19):
        assert circuit.observe({"status": "retryable_failure"}, {"http_status": 503}, adapter) is None
    assert "25%" in circuit.observe({"status": "retryable_failure"}, {"http_status": 503}, adapter)


def test_resume_never_resends_prior_content_denial(tmp_path, monkeypatch):
    case = {"case_id": "NEW1", "view_sha256": "0" * 64}
    root = tmp_path / "route"
    path = root / "attempts/NEW1/01/request.json"
    adapter.atomic(path, {"request_sha256": "frozen", "view_sha256": case["view_sha256"]})
    adapter.atomic(path.with_name("result.json"), {"parsed": {"retryable": False}})
    monkeypatch.setattr(adapter, "prepare", lambda *_: (b"payload", "frozen"))
    monkeypatch.setattr(adapter, "observation", lambda *_: {"status": "http_failure"})

    def forbidden(*_):
        pytest.fail("Terminal denied payload must not be resent")

    monkeypatch.setattr(adapter, "request_once", forbidden)
    assert adapter.job({}, case, root, "test", None, None) == {"status": "http_failure"}


def test_budget_ledger_keeps_unknown_prior_attempt_liability(tmp_path):
    adapter.atomic(tmp_path / "attempts/A/01/request.json", {})
    ledger = adapter.Ledger(tmp_path, {"reserve_usd": "1.00", "budget_usd": "1.50"})
    assert ledger.summary()["unpriced_attempts"] == 1
    assert ledger.summary()["known_price_usd"] == "0"
    with pytest.raises(RuntimeError, match="budget guard"):
        ledger.acquire()


def test_recovery_policy_requires_new_phase_and_preserves_defaults():
    args = SimpleNamespace(model="luna", phase=score.PHASE, max_attempts=5, budget_usd="120.00")
    with pytest.raises(ValueError, match="separately named phase"):
        score.freeze_policy(args)
    args.phase = "transient-recovery-v1"
    policy = score.freeze_policy(args)
    assert policy["max_attempts"] == 5
    assert policy["budget_usd"] == "120.00"
    assert score.POLICY["luna"]["max_attempts"] == 3
    assert score.POLICY["luna"]["budget_usd"] == "80.00"


@pytest.mark.parametrize("budget", ["NaN", "Infinity", "-1", "0", "invalid"])
def test_recovery_rejects_invalid_budgets(budget):
    args = SimpleNamespace(model="luna", phase="recovery", max_attempts=None, budget_usd=budget)
    with pytest.raises(ValueError, match="finite and positive"):
        score.freeze_policy(args)
