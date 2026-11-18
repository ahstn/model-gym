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


def test_one_empty_completion_retry_uses_exact_per_case_limit():
    plans = {
        "recovery": {
            "policy": {"max_attempts": 5},
            "recovery": {"kind": "empty_length_once_v1", "attempt_limits": {"N1": 2, "N2": 5}},
        }
    }
    row = {
        "case_id": "N1",
        "source_request_plan_id": "recovery",
        "attempt_count": 2,
        "status": "retryable_failure",
    }
    assert export.finalized_observation(row, plans)
    assert export.finalized_observation({**row, "status": "invalid"}, plans)
    assert not export.finalized_observation({**row, "attempt_count": 1}, plans)
    assert not export.finalized_observation({**row, "attempt_count": 3, "status": "valid"}, plans)
    assert not export.finalized_observation({**row, "case_id": "unlisted", "status": "valid"}, plans)
    assert not export.finalized_observation({**row, "status": "observation_not_finalized"}, plans)


def completed_subset_recovery(tmp_path):
    manifest, phases = completed_run(tmp_path)
    root = tmp_path / "results/api/deepseek"
    baseline = json.loads((root / "plans/expansion.json").read_text())
    original_request = root / "attempts/N1/01/request.json"
    original_result = original_request.with_name("result.json")
    request = {
        "plan_id": baseline["plan_id"],
        "model": "deepseek",
        "case_id": "N1",
        "attempt": 1,
        "view_sha256": "hash1",
        "profile_sha256": baseline["profile"]["sha256"],
        "request_sha256": hashlib.sha256(b"unchanged payload").hexdigest(),
    }
    write(original_request, request)
    write(original_result, {"http_status": 200, "body": "empty token-limit completion"})
    archive = root / "recovery-evidence/empty-length-once/N1.json"
    write(
        archive,
        {
            **manifest["cases"][0],
            "model": "deepseek",
            "plan_id": baseline["plan_id"],
            "attempt_count": 1,
            "status": "failed",
            "reason": "Incomplete completion",
            "technical_score": None,
            "aesthetic_score": None,
            "task_valid": {"technical_quality": False, "aesthetics": False},
            "raw_response_path": str(original_result),
            "raw_response_sha256": hashlib.sha256(original_result.read_bytes()).hexdigest(),
        },
    )
    source_pins = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (original_request, original_result)
    }
    pinned_paths = [
        root / "plans/expansion.json",
        root / "expansion-status.json",
        root / "observations/N2.json",
        archive,
    ]
    recovery = {
        **baseline,
        "phase": "empty-length-once",
        "plan_id": "deepseek-recovery",
        "created_at": "2026-11-17T13:00:00Z",
        "cases": baseline["cases"][:1],
        "reference_pins": {
            **source_pins,
            **{str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in pinned_paths},
        },
        "recovery": {
            "kind": "empty_length_once_v1",
            "attempt_limits": {"N1": 2},
            "baseline_phase": "expansion",
            "baseline_plan_id": baseline["plan_id"],
            "eligibility": {
                "N1": {
                    "previous_attempt_count": 1,
                    "request_sha256": request["request_sha256"],
                    "source_pins": source_pins,
                    "baseline_observation_path": str(archive),
                }
            },
        },
    }
    write(root / "attempts/N1/02/request.json", {**request, "plan_id": recovery["plan_id"], "attempt": 2})
    write(
        root / "attempts/N1/02/result.json",
        {"transport_error": "ConnectionError", "price_usd": None, "latency_seconds": 1.0},
    )
    write(root / "plans/empty-length-once.json", recovery)
    write(
        root / "empty-length-once-status.json",
        {
            "status": "complete",
            "plan_id": recovery["plan_id"],
            "planned_cases": 1,
            "selected_cases": 1,
            "completed_cases": 1,
            "outcomes": {"retryable_failure": 1},
        },
    )
    write(
        root / "observations/N1.json",
        {**manifest["cases"][0], "plan_id": recovery["plan_id"], "attempt_count": 2},
    )
    phases["deepseek"] = "empty-length-once"
    return manifest, phases, root


def test_latest_subset_recovery_keeps_full_manifest_coverage_and_prior_plan(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    unchanged = (root / "observations/N2.json").read_bytes()
    plans, selected = export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)
    assert len(plans) == 5 and "deepseek-plan" in plans and "deepseek-recovery" in plans
    assert selected["deepseek"] == {"phase": "empty-length-once", "plan_id": "deepseek-recovery", "cases": 1}
    assert (root / "observations/N2.json").read_bytes() == unchanged


@pytest.mark.parametrize(
    "field,value", [("plan_id", "deepseek-plan"), ("attempt_count", 1), ("attempt_count", 3)]
)
def test_recovery_complete_status_cannot_hide_stale_or_extra_attempt_observations(tmp_path, field, value):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "observations/N1.json"
    write(path, {**json.loads(path.read_text()), field: value})
    with pytest.raises(ValueError, match="recovery observations"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_recovery_reference_pins_are_validated_before_export(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    observation = root / "observations/N2.json"
    plan["reference_pins"][str(observation)] = hashlib.sha256(observation.read_bytes()).hexdigest()
    write(path, plan)

    def validate_pins(frozen):
        for filename, expected in frozen.get("reference_pins", {}).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Pinned original observation changed")

    export.validate_completed_runs(tmp_path, manifest, phases, validate_pins)
    write(observation, {**manifest["cases"][1], "unexpected_score": 4.0})
    with pytest.raises(ValueError, match="Pinned original observation"):
        export.validate_completed_runs(tmp_path, manifest, phases, validate_pins)


@pytest.mark.parametrize(
    "relative",
    [
        "observations/N2.json",
        "attempts/N1/01/request.json",
        "attempts/N1/01/result.json",
        "recovery-evidence/empty-length-once/N1.json",
    ],
)
def test_independent_recovery_audit_requires_every_original_reference_pin(tmp_path, relative):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    plan["reference_pins"].pop(str(root / relative))
    write(path, plan)
    with pytest.raises(ValueError, match="recovery evidence pin"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


@pytest.mark.parametrize(
    "relative", ["observations/N2.json", "attempts/N1/01/request.json", "attempts/N1/01/result.json"]
)
def test_independent_recovery_audit_rehashes_original_evidence(tmp_path, relative):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / relative
    write(path, {**json.loads(path.read_text()), "unexpected_change": True})
    with pytest.raises(ValueError, match="recovery evidence pin"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_recovery_source_allowlist_must_cover_every_prior_request_and_response(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    plan["recovery"]["eligibility"]["N1"]["source_pins"].pop(str(root / "attempts/N1/01/result.json"))
    write(path, plan)
    with pytest.raises(ValueError, match="source receipt pins"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_frozen_source_allowlist_digest_must_match_both_file_and_reference_pin(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    plan["recovery"]["eligibility"]["N1"]["source_pins"][str(root / "attempts/N1/01/result.json")] = "0" * 64
    write(path, plan)
    with pytest.raises(ValueError, match="recovery evidence pin"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_recovery_cap_cannot_silently_allow_two_additional_attempts(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    plan["recovery"]["attempt_limits"]["N1"] = 3
    write(path, plan)
    observation = root / "observations/N1.json"
    write(observation, {**json.loads(observation.read_text()), "attempt_count": 3})
    with pytest.raises(ValueError, match="exactly one additional"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_recovery_appended_payload_hash_must_match_the_frozen_original(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "attempts/N1/02/request.json"
    write(path, {**json.loads(path.read_text()), "request_sha256": "0" * 64})
    with pytest.raises(ValueError, match="original payload"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


def test_unreceipted_recovery_request_blocks_export_even_if_status_claims_complete(tmp_path):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    (root / "attempts/N1/02/result.json").unlink()
    with pytest.raises(ValueError, match="no response receipt"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


@pytest.mark.parametrize("limits", [{"unknown": 2}, {"N1": 4}, {"N1": 0}])
def test_recovery_limits_match_the_frozen_subset_and_global_ceiling(tmp_path, limits):
    manifest, phases, root = completed_subset_recovery(tmp_path)
    path = root / "plans/empty-length-once.json"
    plan = json.loads(path.read_text())
    plan["recovery"]["attempt_limits"] = limits
    write(path, plan)
    with pytest.raises(ValueError, match="recovery attempt limits"):
        export.validate_completed_runs(tmp_path, manifest, phases, lambda plan: None)


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
