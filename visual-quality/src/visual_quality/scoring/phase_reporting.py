"""Reconcile baseline and bounded recovery outcomes without duplicating charges.

Inputs are validated frozen plans and receipt-derived attempt provenance. Outputs
contain only aggregate counts and money, suitable for both local and Git reports.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation

OUTCOMES = {"valid", "partial", "invalid", "failed", "retryable_failure"}


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Phase counts must be nonnegative integers")
    return value


def _money(value):
    if type(value) not in (str, int, float):
        raise ValueError("Invalid phase monetary value")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid phase monetary value") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid phase monetary value")
    return amount


def _outcomes(value):
    if set(value) - OUTCOMES:
        raise ValueError("Unknown phase outcome status")
    return {key: _count(n) for key, n in sorted(value.items()) if _count(n)}


def _stage(attempts, outcomes):
    known = sum((_money(row["price_usd"]) for row in attempts if row["price_usd"] is not None), Decimal(0))
    missing = sum(row["price_usd"] is None for row in attempts)
    return {
        "cases": sum(outcomes.values()),
        "attempts": len(attempts),
        "outcomes": _outcomes(outcomes),
        "known_price_usd": str(known),
        "unpriced_attempts": missing,
        "total_price_usd": str(known) if not missing else None,
    }


def _check_ledger(stages, ledger, *, attempt_key="attempts"):
    if (
        sum(row["attempts"] for row in stages) != _count(ledger[attempt_key])
        or sum((_money(row["known_price_usd"]) for row in stages), Decimal(0))
        != _money(ledger["known_price_usd"])
        or sum(row["unpriced_attempts"] for row in stages) != _count(ledger["unpriced_attempts"])
    ):
        raise ValueError("Phase receipts do not reconcile with the accounting ledger")


def _check_status(root, plan, stage):
    state = json.loads((root / f"{plan['phase']}-status.json").read_text())
    if (
        state.get("status") != "complete"
        or state.get("plan_id") != plan["plan_id"]
        or any(
            state.get(key) != stage["cases"] for key in ("planned_cases", "selected_cases", "completed_cases")
        )
        or _outcomes(state.get("outcomes", {})) != stage["outcomes"]
    ):
        raise ValueError("Completed phase status does not match its receipt-derived outcomes")
    return state


def _model_summary(root, plans, rows, ledger, baseline_phase):
    baselines = [plan for plan in plans.values() if plan["phase"] == baseline_phase]
    if len(baselines) != 1:
        raise ValueError("Exactly one frozen baseline plan is required")
    baseline = baselines[0]
    recoveries = [plan for plan in plans.values() if plan["plan_id"] != baseline["plan_id"]]
    if len(recoveries) > 1 or any(
        plan.get("recovery", {}).get("kind") != "empty_length_once_v1"
        or plan["recovery"].get("baseline_plan_id") != baseline["plan_id"]
        for plan in recoveries
    ):
        raise ValueError("Phase reporting requires the baseline and at most one bounded recovery")
    case_sets = {key: {case["case_id"] for case in plan["cases"]} for key, plan in plans.items()}
    if (
        len(rows) != len(case_sets[baseline["plan_id"]])
        or {row["case_id"] for row in rows} != case_sets[baseline["plan_id"]]
    ):
        raise ValueError("Phase observations do not cover the unique baseline cases")
    attempts = {key: [] for key in plans}
    outcomes = {key: Counter() for key in plans}
    final_outcomes = Counter()
    for row in rows:
        history = row["attempt_provenance"]
        if not history or row["attempt_count"] != len(history):
            raise ValueError("Phase attempt provenance is missing or incomplete")
        by_plan = {}
        for number, item in enumerate(history, 1):
            key = item["source_request_plan_id"]
            if key not in plans or row["case_id"] not in case_sets[key]:
                raise ValueError("Attempt is absent from its frozen model plan")
            if item["attempt_id"] != f"attempts/{row['case_id']}/{number:02d}":
                raise ValueError("Phase attempt numbering is duplicated or changed")
            by_plan.setdefault(key, []).append(item)
            attempts[key].append(item)
        if next(iter(by_plan)) != baseline["plan_id"] or history[-1]["status"] != row["status"]:
            raise ValueError("Final observation or baseline attempt order differs")
        for key, group in by_plan.items():
            if group[-1]["status"] not in OUTCOMES:
                raise ValueError("A phase has an unfinished outcome")
            outcomes[key][group[-1]["status"]] += 1
            if key != baseline["plan_id"]:
                limit = plans[key]["recovery"]["attempt_limits"].get(row["case_id"])
                if len(group) != 1 or len(history) != limit or group[-1] is not history[-1]:
                    raise ValueError("Recovery must append exactly one final attempt at its frozen limit")
        final_outcomes[row["status"]] += 1
    stages = {key: _stage(attempts[key], outcomes[key]) for key in plans}
    for key, plan in plans.items():
        if stages[key]["cases"] != len(case_sets[key]):
            raise ValueError("Not every selected phase case has a recorded outcome")
        state = _check_status(root, plan, stages[key])
        if key == baseline["plan_id"]:
            _check_ledger([stages[key]], state, attempt_key="http_attempts")
        # Recovery status accounting is cumulative: never treat it as added fees.
    _check_ledger(list(stages.values()), ledger)
    recovery = stages[recoveries[0]["plan_id"]] if recoveries else _stage([], {})
    return {
        "baseline": stages[baseline["plan_id"]],
        "recovery": {"phases": len(recoveries), **recovery},
        "final_outcomes": _outcomes(final_outcomes),
    }


def build_phase_summary(api_root, plans, observations, accounting, *, models, baseline_phase="expansion"):
    """Use receipt provenance and completed statuses after plan/pin validation."""
    result = {}
    for model in models:
        model_plans = {key: plan for key, plan in plans.items() if plan["model"] == model}
        result[model] = _model_summary(
            api_root / model, model_plans, observations[model], accounting[model], baseline_phase
        )
    return result


def project_phase_summary(value, accounting, coverage, *, models, baseline_cases):
    """Whitelist aggregate fields and reject internally inconsistent stage totals."""
    if set(value) != set(models):
        raise ValueError("Phase summary must contain exactly the active models")
    result = {}
    for model in models:
        stages = {}
        for name in ("baseline", "recovery"):
            source = value[model][name]
            stage = {key: _count(source[key]) for key in ("cases", "attempts", "unpriced_attempts")}
            stage["outcomes"] = _outcomes(source["outcomes"])
            stage["known_price_usd"] = str(_money(source["known_price_usd"]))
            stage["total_price_usd"] = (
                None if source["total_price_usd"] is None else str(_money(source["total_price_usd"]))
            )
            expected_total = None if stage["unpriced_attempts"] else stage["known_price_usd"]
            if (
                sum(stage["outcomes"].values()) != stage["cases"]
                or stage["attempts"] < stage["cases"]
                or stage["unpriced_attempts"] > stage["attempts"]
                or stage["total_price_usd"] != expected_total
            ):
                raise ValueError("Inconsistent phase counts or monetary totals")
            stages[name] = stage
        base, recovery = stages["baseline"], stages["recovery"]
        recovery["phases"] = _count(value[model]["recovery"]["phases"])
        if (
            base["cases"] != baseline_cases
            or recovery["cases"] > base["cases"]
            or recovery["phases"] != int(recovery["cases"] > 0)
            or recovery["attempts"] != recovery["cases"]
            or (not recovery["cases"] and _money(recovery["known_price_usd"]) != 0)
        ):
            raise ValueError("Unexpected baseline or bounded-recovery coverage")
        final = _outcomes(value[model]["final_outcomes"])
        if final != _outcomes(coverage[model]["status_counts"]) or sum(final.values()) != baseline_cases:
            raise ValueError("Phase final outcomes differ from exported coverage")
        _check_ledger([base, recovery], accounting[model])
        result[model] = {**stages, "final_outcomes": final}
    return result


def phase_summary_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase_report_lines(summary):
    """The recovery subtotal is descriptive, never an additional ledger entry."""
    lines = [
        "",
        (
            "Baseline outcomes and bounded recovery are shown separately. Recovery fees below are already "
            "included in the new-run API totals; compatibility probes are separate."
        ),
        "",
        "| Model | Baseline outcomes | Recovery cases / attempts | Recovery outcomes | Recovery known USD | Recovery unpriced attempts | Recovery exact USD |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for model, row in summary.items():
        baseline = ", ".join(f"{key}: {n:,}" for key, n in row["baseline"]["outcomes"].items())
        recovery = row["recovery"]
        outcomes = ", ".join(f"{key}: {n:,}" for key, n in recovery["outcomes"].items()) or "none"
        total = recovery["total_price_usd"] if recovery["total_price_usd"] is not None else "unknown"
        lines.append(
            f"| {model} | {baseline} | {recovery['cases']:,} / {recovery['attempts']:,} | {outcomes} | "
            f"{recovery['known_price_usd']} | {recovery['unpriced_attempts']:,} | {total} |"
        )
    return lines
