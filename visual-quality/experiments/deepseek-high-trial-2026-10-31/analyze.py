"""Compare the frozen 72-image runs and reconcile their reported usage costs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PREVIOUS = HERE.parent / "photo-scope-trial-2026-10-31"
TASKS = ("technical_quality", "aesthetics")
RUNS = {
    "deepseek_none": (PREVIOUS.name, "deepseek"),
    "deepseek_high": (HERE.name, "deepseek"),
    "luna_high": (PREVIOUS.name, "luna"),
    "gemini_medium": (PREVIOUS.name, "gemini"),
    "haiku_low": (PREVIOUS.name, "haiku"),
}


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(row, task):
    if row is None or task in row.get("validation_errors", {}):
        return None
    rating = row.get(task, {})
    value = rating.get("score")
    if rating.get("status") != "ok" or value is None:
        return None
    return Decimal(str(value))


def mean(values):
    return float(sum(values) / len(values)) if values else None


def raw_cost(usage):
    """Use reported gateway costs plus confirmed BYOK costs, never token estimates."""
    gateway = usage.get("cost")
    details = usage.get("cost_details", {})
    upstream = details.get("upstream_inference_cost")
    byok = usage.get("is_byok", details.get("is_byok"))
    gateway = Decimal(str(gateway)) if gateway is not None else None
    upstream = Decimal(str(upstream)) if upstream is not None else None
    if gateway is None or (byok is True and upstream is None):
        effective = None
    elif byok is not False and upstream is not None:
        effective = gateway + upstream
    else:
        effective = max(gateway, upstream or Decimal(0))
    return gateway, upstream, effective, byok


def audit_usage(directory, report, settings=None):
    totals = Counter()
    applied_headers = Counter()
    requested_headers = Counter()
    gateway_total = Decimal(0)
    upstream_total = Decimal(0)
    effective_total = Decimal(0)
    unpriced = []
    attempts = []
    for item in report["attempts"]:
        raw_path = directory / "attempts" / item["attempt_id"] / "result.json"
        result = read_json(raw_path)
        headers = {key.lower(): value for key, value in result.get("headers", {}).items()}
        applied = headers.get("x-runinfra-reasoning-effort-applied")
        requested = headers.get("x-runinfra-reasoning-effort-requested")
        if applied is not None:
            applied_headers[applied] += 1
        if requested is not None:
            requested_headers[requested] += 1
        body = json.loads(result["body"], parse_float=Decimal) if result.get("body") else {}
        usage = body.get("usage", {})
        gateway, upstream, effective, byok = raw_cost(usage)
        recorded = item.get("cost_usd")
        assert effective == (Decimal(recorded) if recorded is not None else None), item["attempt_id"]
        if effective is None:
            unpriced.append(item["attempt_id"])
        else:
            effective_total += effective
        gateway_total += gateway or Decimal(0)
        upstream_total += upstream or Decimal(0)
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if usage.get(token_key) is not None:
                totals[token_key] += usage[token_key]
                totals[token_key + "_reported_calls"] += 1
        if reasoning is not None:
            totals["reasoning_tokens"] += reasoning
            totals["reasoning_tokens_reported_calls"] += 1
            totals["positive_reasoning_tokens_calls"] += reasoning > 0
        totals["byok_calls"] += byok is True
        choices = body.get("choices", [])
        finish = choices[0].get("finish_reason") if choices else None
        attempts.append({
            "case_id": item["case_id"], "attempt_id": item["attempt_id"],
            "cost_usd": str(effective) if effective is not None else None,
            "reasoning_tokens": reasoning, "finish_reason": finish,
            "reasoning_effort_applied_header": applied,
            "reasoning_effort_requested_header": requested,
            "raw_result_sha256": digest(raw_path),
        })
    assert effective_total == Decimal(report["known_cost_usd"])
    assert len(unpriced) == report["unpriced_attempts"]
    settings = settings or {}
    return {
        "calls": len(attempts), "known_cost_usd": str(effective_total),
        "gateway_reported_cost_usd": str(gateway_total),
        "upstream_reported_cost_usd": str(upstream_total),
        "unpriced_attempt_ids": unpriced,
        "cost_sources": dict(Counter(item["cost_source"] for item in report["attempts"])),
        "reasoning_evidence": {
            "requested_setting": settings.get("reasoning_effort", settings.get("reasoning", {}).get("effort")),
            "applied_header_counts": dict(applied_headers),
            "applied_header_missing_calls": len(attempts) - sum(applied_headers.values()),
            "requested_header_counts": dict(requested_headers),
            "requested_header_missing_calls": len(attempts) - sum(requested_headers.values()),
            "interpretation": "Request settings state intent. Applied headers report gateway handling; missing headers establish neither enabled nor disabled reasoning. Positive reported reasoning tokens provide evidence that reasoning ran.",
        },
        "usage": {key: totals[key] for key in (
            "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
            "prompt_tokens_reported_calls", "completion_tokens_reported_calls",
            "total_tokens_reported_calls", "reasoning_tokens_reported_calls",
            "positive_reasoning_tokens_calls", "byok_calls",
        )},
        "attempts": attempts,
    }


def load_runs(refs):
    runs = {}
    reference_ids = set(refs)
    for name, (experiment, route) in RUNS.items():
        directory = PROJECT / "data" / experiment / route / "images"
        run = read_json(directory / "run.json")
        report = read_json(directory / "report.json")
        observations = read_rows(directory / "observations.jsonl")
        observed = {row["case_id"]: row for row in observations}
        assert report["status"] == "complete" and report["attempted"] == 72, name
        assert len(report["attempts"]) == 72
        assert {item["case_id"] for item in report["attempts"]} == reference_ids
        assert len(observed) == len(observations) and set(observed) <= reference_ids
        assert {case["case_id"] for case in run["cases"]} == reference_ids
        for case in run["cases"]:
            assert case["view_sha256"] == refs[case["case_id"]]["view_sha256"]
        for case_id, row in observed.items():
            assert row["view_sha256"] == refs[case_id]["view_sha256"]
        runs[name] = {
            "run": run, "report": report, "observed": observed,
            "billing": audit_usage(directory, report, run["settings"]),
            "input_sha256": {file: digest(directory / file) for file in (
                "run.json", "report.json", "observations.jsonl",
            )},
        }
    before, after = (runs[key]["run"] for key in ("deepseek_none", "deepseek_high"))
    assert before["prompts"] == after["prompts"], "Prompt/schema changed"
    assert before["requested_model"] == after["requested_model"], "Model changed"
    return runs


def metrics(predictions, refs, ids, task):
    valid = [case_id for case_id in ids if predictions[task].get(case_id) is not None]
    errors = [predictions[task][case_id] - Decimal(str(refs[case_id][task]["score"])) for case_id in valid]
    return {
        "n": len(valid), "cohort_n": len(ids), "case_ids": valid,
        "unscored_case_ids": [case_id for case_id in ids if case_id not in valid],
        "mae": mean([abs(error) for error in errors]), "mean_signed_error": mean(errors),
        "within_0_5": sum(abs(error) <= Decimal("0.5") for error in errors),
    }


def compare_pair(before, after, refs, ids, task):
    common = [case_id for case_id in ids if before[task].get(case_id) is not None and after[task].get(case_id) is not None]
    pairs = []
    for case_id in common:
        reference = Decimal(str(refs[case_id][task]["score"]))
        old, new = before[task][case_id], after[task][case_id]
        old_error, new_error = abs(old - reference), abs(new - reference)
        pairs.append({
            "case_id": case_id, "reference": float(reference),
            "before": float(old), "after": float(new), "score_shift": float(new - old),
            "before_error": float(old_error), "after_error": float(new_error),
            "error_delta": float(new_error - old_error),
        })
    before_metrics = metrics(before, refs, common, task)
    after_metrics = metrics(after, refs, common, task)
    return {
        "n": len(common), "cohort_n": len(ids),
        "unpaired_case_ids": [case_id for case_id in ids if case_id not in common],
        "before_mae": before_metrics["mae"], "after_mae": after_metrics["mae"],
        "mae_delta": mean([Decimal(str(pair["error_delta"])) for pair in pairs]),
        "improved": sum(pair["error_delta"] < 0 for pair in pairs),
        "unchanged": sum(pair["error_delta"] == 0 for pair in pairs),
        "worsened": sum(pair["error_delta"] > 0 for pair in pairs),
        "pairs": pairs,
    }


def scope_audit(run, refs):
    audit = []
    for case_id, ref in refs.items():
        row = run["observed"].get(case_id)
        flagged = [task for task in TASKS if score(row, task) is not None and "unsupported_domain" in row[task].get("limitations", [])]
        floor = all(score(row, task) == Decimal(1) and task in flagged for task in TASKS)
        issue = (
            "invalid_or_unscored_rating" if any(score(row, task) is None for task in TASKS)
            else "missed_exclusion_or_incomplete_floor" if ref["eligibility"] == "excluded" and not floor
            else "false_exclusion" if ref["eligibility"] == "eligible" and flagged
            else "uncertain_reference_exclusion" if ref["eligibility"] == "uncertain" and flagged
            else None
        )
        audit.append({
            "case_id": case_id, "reference_eligibility": ref["eligibility"],
            "model_exclusion_tasks": flagged, "complete_policy_floor": floor,
            "issue": issue,
        })
    return {
        "flagged_case_ids": [row["case_id"] for row in audit if row["model_exclusion_tasks"]],
        "floor_case_ids": [row["case_id"] for row in audit if row["complete_policy_floor"]],
        "issues": [row for row in audit if row["issue"]],
        "poster": {task: run["observed"].get("P048", {}).get(task) for task in TASKS},
    }


def cost_summary(runs):
    per_run = {}
    models = {}
    for name, data in runs.items():
        billing = data["billing"]
        period = "incremental" if name == "deepseek_high" else "previous"
        model = data["run"]["requested_model"]
        per_run[name] = {
            "model": model, "period": period,
            **{key: value for key, value in billing.items() if key != "attempts"},
        }
        row = models.setdefault(model, {
            "previous_calls": 0, "incremental_calls": 0,
            "previous_cost_usd": Decimal(0), "incremental_cost_usd": Decimal(0),
            "unpriced_attempts": 0,
        })
        row[period + "_calls"] += billing["calls"]
        row[period + "_cost_usd"] += Decimal(billing["known_cost_usd"])
        row["unpriced_attempts"] += len(billing["unpriced_attempt_ids"])
    for row in models.values():
        row["total_calls"] = row["previous_calls"] + row["incremental_calls"]
        row["total_cost_usd"] = row["previous_cost_usd"] + row["incremental_cost_usd"]
        for field in ("previous_cost_usd", "incremental_cost_usd", "total_cost_usd"):
            row[field] = str(row[field])
    totals = {
        field: str(sum(Decimal(row[field]) for row in models.values()))
        for field in ("previous_cost_usd", "incremental_cost_usd", "total_cost_usd")
    }
    totals.update({field: sum(row[field] for row in models.values()) for field in (
        "previous_calls", "incremental_calls", "total_calls", "unpriced_attempts",
    )})
    return {
        "scope": "The prior 72-image four-model trial plus this 72-image DeepSeek high rerun; earlier 50/200-image experiments are excluded.",
        "basis": "Exact API-reported usage costs, reconciled per attempt against saved reports. Confirmed BYOK upstream costs are included; ordinary gateway/upstream costs are not double-counted. Not invoice verification.",
        "runs": per_run, "models": models, "totals": totals,
    }


def main():
    refs_path = PREVIOUS / "references.jsonl"
    assert digest(refs_path) == read_json(PREVIOUS / "references-freeze.json")["references_sha256"]
    refs = {row["case_id"]: row for row in read_rows(refs_path)}
    assert set(refs) == {f"P{number:03d}" for number in range(1, 73)}
    runs = load_runs(refs)
    cohorts = {
        "all_72": sorted(refs),
        "eligible_70": sorted(case_id for case_id, ref in refs.items() if ref["eligibility"] == "eligible"),
        "retained_eligible_49": sorted(case_id for case_id, ref in refs.items() if case_id <= "P050" and ref["eligibility"] == "eligible"),
        "new_22": sorted(case_id for case_id in refs if case_id >= "P051"),
    }
    assert [len(ids) for ids in cohorts.values()] == [72, 70, 49, 22]
    predictions = {name: {task: {case_id: score(data["observed"].get(case_id), task) for case_id in refs} for task in TASKS} for name, data in runs.items()}
    for effort in ("none", "high"):
        predictions[f"deepseek_{effort}_luna_average"] = {
            task: {case_id: (predictions[f"deepseek_{effort}"][task][case_id] + predictions["luna_high"][task][case_id]) / 2
                   if predictions[f"deepseek_{effort}"][task][case_id] is not None and predictions["luna_high"][task][case_id] is not None else None
                   for case_id in refs}
            for task in TASKS
        }
    comparisons = {}
    for cohort, ids in cohorts.items():
        comparisons[cohort] = {}
        for task in TASKS:
            common = [case_id for case_id in ids if all(predictions[name][task].get(case_id) is not None for name in RUNS)]
            comparisons[cohort][task] = {
                "individual_coverage": {name: metrics(scores, refs, ids, task) for name, scores in predictions.items()},
                "all_five_runs_common_valid_case_ids": common,
                "excluded_from_common_comparison_case_ids": [case_id for case_id in ids if case_id not in common],
                "all_five_runs_common_valid": {name: metrics(scores, refs, common, task) for name, scores in predictions.items()},
                "deepseek_none_to_high": compare_pair(predictions["deepseek_none"], predictions["deepseek_high"], refs, ids, task),
                "deepseek_luna_average_none_to_high": compare_pair(predictions["deepseek_none_luna_average"], predictions["deepseek_high_luna_average"], refs, ids, task),
            }
    analysis = {
        "interpretation": "Lower MAE means closer agreement with frozen assistant assessments, not verified accuracy. One draw per image/configuration; related image families and reused examples prevent treating 72 images as independent held-out evidence. High reasoning also raises max_tokens from 2048 to 16384, so this is a configuration comparison rather than a pure reasoning-only causal test.",
        "cohorts": cohorts, "comparisons": comparisons,
        "runs": {name: {
            "run_id": data["run"]["run_id"], "settings": data["run"]["settings"],
            "input_sha256": data["input_sha256"], "usage_and_cost": data["billing"],
            "scope": scope_audit(data, refs),
            "validation_errors": {case_id: row["validation_errors"] for case_id, row in data["observed"].items() if row["validation_errors"]},
            "missing_observation_case_ids": sorted(set(refs) - set(data["observed"])),
        } for name, data in runs.items()},
        "references_sha256": digest(refs_path), "generator_sha256": digest(Path(__file__)),
    }
    costs = cost_summary(runs)
    costs["generator_sha256"] = analysis["generator_sha256"]
    write_json(HERE / "analysis.json", analysis)
    write_json(HERE / "cost-summary.json", costs)
    print(json.dumps({
        "all_72_deepseek": {task: {key: value for key, value in comparisons["all_72"][task]["deepseek_none_to_high"].items() if key != "pairs"} for task in TASKS},
        "cost_totals": costs["totals"],
    }, indent=2))


if __name__ == "__main__":
    main()
