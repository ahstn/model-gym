"""Baseline and recovery reporting must preserve outcomes and every charge once."""

import copy
import json
from collections import Counter
from decimal import Decimal

import pytest

from visual_quality.scoring.phase_reporting import (
    build_phase_summary,
    phase_report_lines,
    phase_summary_digest,
    project_phase_summary,
)

MODELS = ("deepseek",)


def ledger(attempts):
    known = sum((Decimal(row["price_usd"]) for row in attempts if row["price_usd"] is not None), Decimal(0))
    missing = sum(row["price_usd"] is None for row in attempts)
    return {
        "attempts": len(attempts),
        "known_price_usd": str(known),
        "unpriced_attempts": missing,
        "total_price_usd": None if missing else str(known),
    }


def fixture(tmp_path, recovery=True, recovery_status="valid", recovery_price="0.02"):
    baseline = {
        "plan_id": "base",
        "phase": "expansion",
        "model": "deepseek",
        "cases": [{"case_id": "N1"}, {"case_id": "N2"}],
    }
    plans = {"base": baseline}
    rows = []
    # The first case needed a transient retry before its empty baseline answer.
    histories = [
        [("base", "retryable_failure", None), ("base", "failed", "0.01")],
        [("base", "valid", "0.03")],
    ]
    if recovery:
        plans["retry"] = {
            "plan_id": "retry",
            "phase": "empty-length-once",
            "model": "deepseek",
            "cases": [{"case_id": "N1"}],
            "recovery": {
                "kind": "empty_length_once_v1",
                "baseline_plan_id": "base",
                "attempt_limits": {"N1": 3},
            },
        }
        histories[0].append(("retry", recovery_status, recovery_price))
    for identity, entries in zip(("N1", "N2"), histories, strict=True):
        attempts = [
            {
                "attempt_id": f"attempts/{identity}/{n:02d}",
                "source_request_plan_id": plan,
                "status": status,
                "price_usd": price,
            }
            for n, (plan, status, price) in enumerate(entries, 1)
        ]
        rows.append(
            {
                "case_id": identity,
                "status": entries[-1][1],
                "attempt_count": len(entries),
                "attempt_provenance": attempts,
            }
        )
    accounting = {"deepseek": ledger([item for row in rows for item in row["attempt_provenance"]])}
    root = tmp_path / "deepseek"
    root.mkdir()
    for key, plan in plans.items():
        outcomes, phase_attempts = Counter(), []
        for row in rows:
            attempts = [item for item in row["attempt_provenance"] if item["source_request_plan_id"] == key]
            if attempts:
                outcomes[attempts[-1]["status"]] += 1
                phase_attempts.extend(attempts)
        fees = ledger(phase_attempts) if key == "base" else accounting["deepseek"]
        state = {
            "plan_id": key,
            "status": "complete",
            "planned_cases": len(plan["cases"]),
            "selected_cases": len(plan["cases"]),
            "completed_cases": len(plan["cases"]),
            "outcomes": dict(outcomes),
            "http_attempts": fees["attempts"],
            **{field: value for field, value in fees.items() if field != "attempts"},
        }
        (root / f"{plan['phase']}-status.json").write_text(json.dumps(state))
    observations = {"deepseek": rows}
    return plans, observations, accounting


def build(tmp_path, inputs):
    return build_phase_summary(tmp_path, *inputs, models=MODELS)


def test_baseline_and_recovery_reconcile_without_using_cumulative_recovery_fee(tmp_path):
    inputs = fixture(tmp_path)
    result = build(tmp_path, inputs)["deepseek"]
    assert result["baseline"] == {
        "cases": 2,
        "attempts": 3,
        "outcomes": {"failed": 1, "valid": 1},
        "known_price_usd": "0.04",
        "unpriced_attempts": 1,
        "total_price_usd": None,
    }
    assert result["recovery"] == {
        "phases": 1,
        "cases": 1,
        "attempts": 1,
        "outcomes": {"valid": 1},
        "known_price_usd": "0.02",
        "unpriced_attempts": 0,
        "total_price_usd": "0.02",
    }
    assert result["final_outcomes"] == {"valid": 2}
    assert inputs[2]["deepseek"]["known_price_usd"] == "0.06"
    rendered = "\n".join(phase_report_lines({"deepseek": result}))
    assert "already included" in rendered and "failed: 1, valid: 1" in rendered


def test_no_recovery_has_explicit_zero_subtotal_and_preserves_missing_baseline(tmp_path):
    result = build(tmp_path, fixture(tmp_path, recovery=False))["deepseek"]
    assert result["recovery"] == {
        "phases": 0,
        "cases": 0,
        "attempts": 0,
        "outcomes": {},
        "known_price_usd": "0",
        "unpriced_attempts": 0,
        "total_price_usd": "0",
    }
    assert result["final_outcomes"] == result["baseline"]["outcomes"]


def test_one_recorded_transport_failure_consumes_recovery_with_unknown_charge(tmp_path):
    result = build(tmp_path, fixture(tmp_path, recovery_status="retryable_failure", recovery_price=None))
    recovery = result["deepseek"]["recovery"]
    assert recovery["outcomes"] == {"retryable_failure": 1}
    assert recovery["attempts"] == recovery["unpriced_attempts"] == 1
    assert recovery["known_price_usd"] == "0" and recovery["total_price_usd"] is None
    assert "unknown" in "\n".join(phase_report_lines(result))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("plan_id", "stale"),
        ("completed_cases", 0),
        ("outcomes", {"valid": 2}),
        ("http_attempts", 4),
        ("known_price_usd", "0.06"),
    ],
)
def test_malformed_baseline_status_cannot_become_aggregate_evidence(tmp_path, field, value):
    inputs = fixture(tmp_path)
    path = tmp_path / "deepseek/expansion-status.json"
    state = json.loads(path.read_text())
    state[field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="status|ledger"):
        build(tmp_path, inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_plan",
        "duplicate_attempt",
        "wrong_case",
        "wrong_limit",
        "wrong_final",
        "duplicate_case",
        "missing_phase_case",
        "wrong_total",
        "negative_price",
        "nan_price",
    ],
)
def test_inconsistent_receipt_provenance_is_rejected(tmp_path, mutation):
    plans, observations, accounting = fixture(tmp_path)
    first = observations["deepseek"][0]
    if mutation == "unknown_plan":
        first["attempt_provenance"][0]["source_request_plan_id"] = "unknown"
    elif mutation == "duplicate_attempt":
        first["attempt_provenance"][1]["attempt_id"] = first["attempt_provenance"][0]["attempt_id"]
    elif mutation == "wrong_case":
        plans["retry"]["cases"] = [{"case_id": "N2"}]
    elif mutation == "wrong_limit":
        plans["retry"]["recovery"]["attempt_limits"]["N1"] = 4
    elif mutation == "wrong_final":
        first["status"] = "failed"
    elif mutation == "duplicate_case":
        observations["deepseek"][1] = copy.deepcopy(first)
    elif mutation == "missing_phase_case":
        plans["retry"]["cases"].append({"case_id": "N2"})
    elif mutation == "wrong_total":
        accounting["deepseek"]["known_price_usd"] = "0.08"
    else:
        first["attempt_provenance"][-1]["price_usd"] = "-1" if mutation == "negative_price" else "NaN"
    with pytest.raises(ValueError):
        build(tmp_path, (plans, observations, accounting))


def test_git_projection_drops_nested_evidence_and_binds_aggregate_digest(tmp_path):
    inputs = fixture(tmp_path)
    summary = build(tmp_path, inputs)
    summary["deepseek"]["recovery"]["case_ids"] = ["R19999"]
    summary["deepseek"]["baseline"]["requests"] = [{"case_id": "R19999"}]
    result = project_phase_summary(
        summary, inputs[2], {"deepseek": {"status_counts": {"valid": 2}}}, models=MODELS, baseline_cases=2
    )
    assert "R19999" not in json.dumps(result)
    assert phase_summary_digest(result) != phase_summary_digest(summary)
    assert phase_summary_digest(result) == phase_summary_digest(json.loads(json.dumps(result)))


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_cases",
        "wrong_fee",
        "fabricated_total",
        "extra_attempt",
        "wrong_outcomes",
        "hidden_id",
        "fake_recovery",
    ],
)
def test_git_projection_rejects_nonreconciling_phase_data(tmp_path, mutation):
    inputs = fixture(tmp_path)
    summary = build(tmp_path, inputs)
    recovery = summary["deepseek"]["recovery"]
    if mutation == "wrong_cases":
        summary["deepseek"]["baseline"]["cases"] = 3
    elif mutation == "wrong_fee":
        recovery["known_price_usd"] = recovery["total_price_usd"] = "0.04"
    elif mutation == "fabricated_total":
        summary["deepseek"]["baseline"]["total_price_usd"] = "0.04"
    elif mutation == "extra_attempt":
        recovery["attempts"] = 2
    elif mutation == "wrong_outcomes":
        summary["deepseek"]["final_outcomes"] = {"valid": 1, "failed": 1}
    elif mutation == "hidden_id":
        recovery["outcomes"] = {"R19999": 1}
    else:
        recovery["phases"] = 2
    with pytest.raises(ValueError):
        project_phase_summary(
            summary, inputs[2], {"deepseek": {"status_counts": {"valid": 2}}}, models=MODELS, baseline_cases=2
        )
