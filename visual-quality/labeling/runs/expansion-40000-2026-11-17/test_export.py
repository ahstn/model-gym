"""Exercise export boundaries, accounting and finalization without paid calls."""

import fcntl
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load("expansion_40000_export", "export.py")
finalize = load("expansion_40000_finalize", "finalize.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def ledger(price, missing=0):
    return {
        model: {
            "known_price_usd": price,
            "unpriced_attempts": missing,
            "total_price_usd": None if missing else price,
        }
        for model in export.MODELS
    }


def completed_run(tmp_path):
    cases = [{"case_id": "N1", "view_sha256": "hash1"}, {"case_id": "N2", "view_sha256": "hash2"}]
    write(tmp_path / "manifest.json", {"cases": cases})
    manifest_sha256 = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    for model in export.MODELS:
        root = tmp_path / "results/api" / model
        plan = {
            "model": model,
            "phase": "expansion",
            "plan_id": model + "-plan",
            "created_at": "2026-11-17T12:00:00Z",
            "profile": {"sha256": "profile"},
            "stats_sent_to_models": False,
            "reference_scores_sent_to_models": False,
            "cases": cases,
            "policy": {"max_attempts": 3},
            "manifest_sha256": manifest_sha256,
        }
        write(root / "plans/expansion.json", plan)
        write(
            root / "expansion-status.json",
            {
                "status": "complete",
                "plan_id": plan["plan_id"],
                "planned_cases": 2,
                "selected_cases": 2,
                "completed_cases": 2,
                "outcomes": {"valid": 2},
            },
        )
        for case in cases:
            write(root / "observations" / (case["case_id"] + ".json"), case)
    return {"cases": cases}, dict.fromkeys(export.MODELS, "expansion")


def test_append_preserves_frozen_bytes_and_native_precision(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_bytes(b'{ "case_id": "old", "score": 3.0000 }\n')
    target = tmp_path / "combined.jsonl"
    export.append_export(old, [{"case_id": "new", "score": 3.141592}], target)
    assert target.read_bytes().startswith(old.read_bytes())
    assert list(export.records(target))[-1]["score"] == 3.141592
    assert old.read_bytes() == b'{ "case_id": "old", "score": 3.0000 }\n'


def test_append_rejects_unterminated_old_record(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_text('{"old":1}')
    with pytest.raises(ValueError, match="newline"):
        export.append_export(old, [{"new": 2}], tmp_path / "out.jsonl")


def test_combined_cost_carries_prior_combined_ledger_once():
    combined = export.combine_costs(ledger("13.04645922", 329), ledger("4.001"), ledger("0.012"))
    assert combined["deepseek"] == {
        "known_price_usd": "17.05945922",
        "unpriced_attempts": 329,
        "total_price_usd": None,
    }


def test_known_zero_and_unknown_are_distinct():
    assert export.combine_costs(ledger("0"))["luna"]["total_price_usd"] == "0"
    assert export.combine_costs(ledger("0", 1))["luna"]["total_price_usd"] is None
    with pytest.raises(KeyError):
        export.combine_costs({})
    with pytest.raises(ValueError, match="accounting"):
        export.combine_costs(ledger("-1"), ledger("2"))


def test_long_row_contract_includes_only_historical_lite_and_references():
    assert export.LONG_ROWS == 211500
    assert export.NEW_IMAGES * (len(export.MODELS) + 1) == 100000


def test_partial_outcome_is_preserved_in_coverage():
    rows = [
        {"model": "deepseek", "technical_score": 2.8, "aesthetic_score": None, "status": "partial"},
        {"model": "deepseek", "technical_score": 3.0, "aesthetic_score": 3.5, "status": "valid"},
    ]
    assert export.coverage(rows)["deepseek"] == {
        "expected": 2,
        "numeric_pairs": 1,
        "status_counts": {"partial": 1, "valid": 1},
    }


def test_retryable_failures_require_exhaustion_and_known_plan():
    plans = {"p": {"policy": {"max_attempts": 3}}}
    row = {"status": "retryable_failure", "source_request_plan_id": "p", "attempt_count": 2}
    assert not export.finalized_observation(row, plans)
    assert export.finalized_observation({**row, "attempt_count": 3}, plans)
    assert not export.finalized_observation(
        {**row, "status": "valid", "source_request_plan_id": "unknown"}, plans
    )
    assert export.finalized_observation({**row, "status": "failed"}, plans)
    assert not export.finalized_observation({**row, "status": "observation_not_finalized"}, plans)


def test_current_completed_plans_are_admitted(tmp_path):
    manifest, phases = completed_run(tmp_path)
    validated = []
    plans, selected = export.validate_completed_runs(tmp_path, manifest, phases, validated.append)
    assert len(validated) == len(plans) == 4
    assert selected["luna"]["plan_id"] == "luna-plan"


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_id", "old-plan"),
        ("status", "partial"),
        ("planned_cases", 1),
        ("selected_cases", 1),
        ("completed_cases", 1),
        ("outcomes", {"valid": 1}),
    ],
)
def test_stale_or_partial_status_never_exports(tmp_path, field, value):
    manifest, phases = completed_run(tmp_path)
    path = tmp_path / "results/api/luna/expansion-status.json"
    state = json.loads(path.read_text())
    write(path, {**state, field: value})
    with pytest.raises(ValueError, match="stopped, partial or incomplete"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


@pytest.mark.parametrize(
    "replacement",
    [{"case_id": "wrong-id", "view_sha256": "hash1"}, {"case_id": "N1", "view_sha256": "wrong-view"}],
)
def test_equal_count_does_not_hide_wrong_observation_join(tmp_path, replacement):
    manifest, phases = completed_run(tmp_path)
    write(tmp_path / "results/api/luna/observations/N1.json", replacement)
    with pytest.raises(ValueError, match="exactly cover"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_new_recovery_plan_cannot_be_hidden_by_old_complete_state(tmp_path):
    manifest, phases = completed_run(tmp_path)
    root = tmp_path / "results/api/luna"
    baseline = json.loads((root / "plans/expansion.json").read_text())
    write(
        root / "plans/recovery.json",
        {**baseline, "phase": "recovery", "plan_id": "new-plan", "created_at": "2026-11-17T13:00:00Z"},
    )
    with pytest.raises(ValueError, match="later plan exists"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_changed_plan_view_is_rejected_even_with_complete_state(tmp_path):
    manifest, phases = completed_run(tmp_path)
    path = tmp_path / "results/api/luna/plans/expansion.json"
    plan = json.loads(path.read_text())
    plan["cases"][0]["view_sha256"] = "different"
    write(path, plan)
    with pytest.raises(ValueError, match="plan cases"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_phases_require_explicit_safe_unique_names():
    phases = finalize.selected_phases(["luna=transport-recovery"], export.MODELS, "expansion")
    assert phases["luna"] == "transport-recovery" and phases["jev"] == "expansion"
    for values in (["qrealign_lite=expansion"], ["luna=../x"], ["luna=a", "luna=b"]):
        with pytest.raises(ValueError):
            finalize.selected_phases(values, export.MODELS, "expansion")


def test_export_and_audit_hold_every_worker_lock(tmp_path, monkeypatch):
    phases = dict.fromkeys(export.MODELS, "expansion")
    for model in phases:
        (tmp_path / "results/api" / model).mkdir(parents=True)
    calls = []

    def locked_action(*args):
        for model in phases:
            with (
                (tmp_path / "results/api" / model / ".lock").open("a") as stream,
                pytest.raises(BlockingIOError),
            ):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append("locked")

    monkeypatch.setitem(sys.modules, "export", SimpleNamespace(export_run=locked_action))
    monkeypatch.setitem(sys.modules, "audit", SimpleNamespace(audit_run=locked_action))
    monkeypatch.setitem(sys.modules, "corpus", SimpleNamespace(write_json=write))
    assert finalize.finalize(tmp_path, tmp_path / "old", phases)
    assert calls == ["locked", "locked"]
    for model in phases:
        with (tmp_path / "results/api" / model / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_busy_worker_prevents_all_exports_and_releases_earlier_locks(tmp_path, monkeypatch):
    phases = dict.fromkeys(export.MODELS, "expansion")
    for model in phases:
        (tmp_path / "results/api" / model).mkdir(parents=True)

    def forbidden(*args):
        pytest.fail("A busy worker must prevent finalization")

    monkeypatch.setitem(sys.modules, "export", SimpleNamespace(export_run=forbidden))
    monkeypatch.setitem(sys.modules, "audit", SimpleNamespace(audit_run=forbidden))
    monkeypatch.setitem(sys.modules, "corpus", SimpleNamespace(write_json=write))
    with (tmp_path / "results/api/deepseek/.lock").open("a") as active:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert finalize.finalize(tmp_path, tmp_path / "old", phases) is False
        with (tmp_path / "results/api/luna/.lock").open("a") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
