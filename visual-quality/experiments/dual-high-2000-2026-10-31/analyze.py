"""Audit both 2,000-image runs and compare the independently reviewed 72-case sample."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / HERE.name
TASKS = ("technical_quality", "aesthetics")
MODELS = ("deepseek", "luna")
PREDICTIONS = (*MODELS, "average")
BOOTSTRAP_SEED = 20261031200071
BOOTSTRAP_DRAWS = 10000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


helpers = module("prior_exact_billing_analysis", HERE.parent / "deepseek-high-trial-2026-10-31/analyze.py")
score, mean = helpers.score, helpers.mean
read_json, read_rows, digest, write_json = (
    helpers.read_json,
    helpers.read_rows,
    helpers.digest,
    helpers.write_json,
)


def predictions(observed, case_ids):
    result = {
        model: {
            task: {case_id: score(observed[model].get(case_id), task) for case_id in case_ids}
            for task in TASKS
        }
        for model in MODELS
    }
    result["average"] = {
        task: {
            case_id: (result["deepseek"][task][case_id] + result["luna"][task][case_id]) / 2
            if all(result[model][task][case_id] is not None for model in MODELS)
            else None
            for case_id in case_ids
        }
        for task in TASKS
    }
    return result


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = Decimal(str(fraction)) * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return float(ordered[left] + (ordered[right] - ordered[left]) * (position - left))


def distribution(values, cohort_n):
    numeric = [value for value in values if value is not None]
    return {
        "cohort_n": cohort_n,
        "numeric_n": len(numeric),
        "unscored_n": cohort_n - len(numeric),
        "mean": mean(numeric),
        "minimum": float(min(numeric)) if numeric else None,
        "maximum": float(max(numeric)) if numeric else None,
        "quantiles": {str(q): quantile(numeric, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "score_counts": dict(sorted(Counter(f"{value:.2f}" for value in numeric).items())),
    }


def pearson(left, right):
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    variance = sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    return float(covariance / variance.sqrt()) if variance else None


def student_agreement(predicted, case_ids, task):
    common = [
        case_id
        for case_id in case_ids
        if all(predicted[model][task][case_id] is not None for model in MODELS)
    ]
    left, right = ([predicted[model][task][case_id] for case_id in common] for model in MODELS)
    differences = [a - b for a, b in zip(left, right, strict=True)]
    large = [case_id for case_id, difference in zip(common, differences, strict=True) if abs(difference) >= 1]
    within = sum(abs(difference) <= Decimal("0.5") for difference in differences)
    return {
        "cohort_n": len(case_ids),
        "common_valid_n": len(common),
        "common_valid_case_ids": common,
        "unpaired_case_ids": sorted(set(case_ids) - set(common)),
        "mean_absolute_difference": mean([abs(value) for value in differences]),
        "mean_signed_difference_deepseek_minus_luna": mean(differences),
        "within_0_5_n": within,
        "within_0_5_fraction": within / len(common) if common else None,
        "pearson_correlation": pearson(left, right),
        "disagreement_at_least_1_case_ids": large,
        "interpretation": "Agreement between students is not evidence of correctness.",
    }


def common_distributions(predicted, case_ids):
    result = {}
    for task in TASKS:
        common = [
            case_id
            for case_id in case_ids
            if all(predicted[model][task][case_id] is not None for model in MODELS)
        ]
        result[task] = {
            "case_ids": common,
            "models": {
                name: distribution([predicted[name][task][case_id] for case_id in common], len(common))
                for name in PREDICTIONS
            },
        }
    return result


def reference_metrics(prediction, refs, case_ids, task):
    reference_valid = [case_id for case_id in case_ids if score(refs.get(case_id), task) is not None]
    common = [case_id for case_id in reference_valid if prediction[task].get(case_id) is not None]
    errors = [prediction[task][case_id] - score(refs[case_id], task) for case_id in common]
    within = sum(abs(value) <= Decimal("0.5") for value in errors)
    return {
        "cohort_n": len(case_ids),
        "n": len(common),
        "case_ids": common,
        "reference_unscored_case_ids": sorted(set(case_ids) - set(reference_valid)),
        "student_unscored_case_ids": sorted(set(reference_valid) - set(common)),
        "mae": mean([abs(value) for value in errors]),
        "mean_signed_error": mean(errors),
        "within_0_5_n": within,
        "within_0_5_fraction": within / len(common) if common else None,
    }


def reference_comparison(predicted, refs, case_ids, task):
    common = [
        case_id
        for case_id in case_ids
        if score(refs.get(case_id), task) is not None
        and all(predicted[model][task][case_id] is not None for model in MODELS)
    ]
    return {
        "individual_coverage": {
            name: reference_metrics(predicted[name], refs, case_ids, task) for name in PREDICTIONS
        },
        "common_valid_case_ids": common,
        "excluded_from_common_case_ids": sorted(set(case_ids) - set(common)),
        "common_valid": {
            name: reference_metrics(predicted[name], refs, common, task) for name in PREDICTIONS
        },
        "comparison_rule": "Compare the common-valid rows; individual coverage has different denominators when a task is invalid or unscored.",
    }


def paired_bootstrap(predicted, refs, case_ids, family_ids, *, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """Resample paired image errors across both tasks, preserving within-image dependence."""
    if draws < 1 or len(case_ids) != len(set(case_ids)):
        raise ValueError("Bootstrap requires positive draws and unique case IDs")
    common = [
        case_id
        for case_id in case_ids
        if all(
            score(refs.get(case_id), task) is not None
            and all(predicted[name][task].get(case_id) is not None for name in PREDICTIONS)
            for task in TASKS
        )
    ]
    if len({family_ids[case_id] for case_id in common}) != len(common):
        raise ValueError("Case bootstrap requires distinct families in the common-valid cohort")
    errors = {
        name: [
            sum(
                (abs(predicted[name][task][case_id] - score(refs[case_id], task)) for task in TASKS),
                Decimal(0),
            )
            / len(TASKS)
            for case_id in common
        ]
        for name in PREDICTIONS
    }
    differences = {
        name: [
            average - individual for average, individual in zip(errors["average"], errors[name], strict=True)
        ]
        for name in MODELS
    }
    samples = {name: [] for name in MODELS}
    generator = random.Random(seed)
    if common:
        for _ in range(draws):
            indices = [generator.randrange(len(common)) for _ in common]
            for name in MODELS:
                samples[name].append(
                    sum((differences[name][index] for index in indices), Decimal(0)) / len(common)
                )
    return {
        "cohort_n": len(case_ids),
        "common_valid_n": len(common),
        "common_valid_case_ids": common,
        "excluded_case_ids": sorted(set(case_ids) - set(common)),
        "distinct_families_n": len(common),
        "seed": seed,
        "requested_draws": draws,
        "resamples_drawn": draws if common else 0,
        "sampling_unit": "case_id; every included case belongs to a distinct family",
        "random_generator": "Python random.Random(seed), randrange with replacement; identical sampled indices for both comparisons",
        "aggregation": "Per-image mean absolute error across technical and aesthetic dimensions, then mean across images",
        "interval_method": "Paired bootstrap percentile interval, 2.5th and 97.5th percentiles with linear interpolation",
        "interval_level": 0.95,
        "negative_delta_favors": "average",
        "mean_absolute_error": {name: mean(values) for name, values in errors.items()},
        "average_minus_individual": {
            name: {
                "mean_error_delta": mean(differences[name]),
                "percentile_95_interval": [quantile(samples[name], 0.025), quantile(samples[name], 0.975)]
                if common
                else None,
            }
            for name in MODELS
        },
        "interpretation": "Descriptive paired-resampling intervals conditional on these selected images and frozen assistant judgments. They are not population-accuracy or independent-human confidence intervals. They do not capture reference uncertainty, repeated model-call variation, or sampling weights for the deliberately selected strata.",
    }


def task_flags(row, task):
    return score(row, task) is not None and "unsupported_domain" in row[task].get("limitations", [])


def scope_audit(observed, refs):
    rows = []
    for case_id, ref in sorted(refs.items()):
        row = observed.get(case_id)
        flagged = [task for task in TASKS if task_flags(row, task)]
        missing = [task for task in TASKS if score(row, task) is None]
        floor = all(score(row, task) == Decimal(1) and task in flagged for task in TASKS)
        issues = []
        if missing:
            issues.append("invalid_or_unscored_rating")
        if ref["eligibility"] == "excluded" and not floor:
            issues.append("missed_exclusion_or_incomplete_floor")
        if ref["eligibility"] == "eligible" and flagged:
            issues.append("false_exclusion")
        if ref["eligibility"] == "uncertain" and flagged:
            issues.append("uncertain_reference_exclusion")
        rows.append(
            {
                "case_id": case_id,
                "reference_eligibility": ref["eligibility"],
                "flagged_tasks": flagged,
                "unscored_tasks": missing,
                "complete_policy_floor": floor,
                "issues": issues,
            }
        )
    return {
        "reference_eligibility_counts": dict(Counter(ref["eligibility"] for ref in refs.values())),
        "issue_counts": dict(Counter(issue for row in rows for issue in row["issues"])),
        "flagged_case_ids": [row["case_id"] for row in rows if row["flagged_tasks"]],
        "complete_floor_case_ids": [row["case_id"] for row in rows if row["complete_policy_floor"]],
        "cases": rows,
        "interpretation": "A numeric 1.00 alone is a low quality score, not a scope exclusion; scope requires unsupported_domain on a valid task.",
    }


def validity(observed, case_ids):
    return {
        "expected_cases": len(case_ids),
        "observations": len(observed),
        "missing_observation_case_ids": sorted(set(case_ids) - set(observed)),
        "both_numeric_n": sum(
            all(score(observed.get(case_id), task) is not None for task in TASKS) for case_id in case_ids
        ),
        "per_task": {
            task: {
                "numeric_n": sum(score(observed.get(case_id), task) is not None for case_id in case_ids),
                "invalid_task_case_ids": [
                    case_id
                    for case_id in case_ids
                    if task in observed.get(case_id, {}).get("validation_errors", {})
                ],
                "unscored_case_ids": [
                    case_id for case_id in case_ids if score(observed.get(case_id), task) is None
                ],
                "status_counts": dict(
                    Counter(
                        observed.get(case_id, {}).get(task, {}).get("status", "missing")
                        for case_id in case_ids
                    )
                ),
            }
            for task in TASKS
        },
        "validation_errors": {
            case_id: row["validation_errors"]
            for case_id, row in sorted(observed.items())
            if row.get("validation_errors")
        },
    }


def strata(cases):
    result = defaultdict(list)
    for case in cases:
        result[case["stratum"]].append(case["case_id"])
        if case["stratum"] == "vizwiz":
            result["vizwiz_" + case["vizwiz_stratum"]].append(case["case_id"])
        if case["stratum"] == "controlled":
            result["controlled_" + case["operator"]].append(case["case_id"])
    return dict(sorted(result.items()))


def family_records(cases, prediction, task):
    groups = defaultdict(list)
    for case in cases:
        if case["stratum"] == "controlled":
            groups[case["family_id"]].append(case)
    result = []
    for family_id, members in sorted(groups.items()):
        members.sort(key=lambda case: case["severity"] or 0)
        if [case["severity"] for case in members] != [None, 1, 2, 3] or len(
            {case["operator"] for case in members}
        ) != 1:
            raise ValueError("Controlled family does not contain one parent and severities 1/2/3")
        values = [prediction[task][case["case_id"]] for case in members]
        complete = all(value is not None for value in values)
        deltas = [b - a if a is not None and b is not None else None for a, b in pairwise(values)]
        result.append(
            {
                "family_id": family_id,
                "operator": members[0]["operator"],
                "case_ids": [case["case_id"] for case in members],
                "scores": values,
                "complete": complete,
                "adjacent_deltas": deltas,
                "non_increasing": all(value <= 0 for value in deltas) if complete else None,
                "strictly_decreasing": all(value < 0 for value in deltas) if complete else None,
                "parent_to_severe_delta": values[3] - values[0]
                if values[0] is not None and values[3] is not None
                else None,
            }
        )
    return result


def family_summary(records):
    complete = [row for row in records if row["complete"]]
    return {
        "cohort_families": len(records),
        "complete_families": len(complete),
        "complete_family_ids": [row["family_id"] for row in complete],
        "incomplete_family_ids": [row["family_id"] for row in records if not row["complete"]],
        "non_increasing_families": sum(row["non_increasing"] for row in complete),
        "strictly_decreasing_families": sum(row["strictly_decreasing"] for row in complete),
        "non_increasing_fraction": sum(row["non_increasing"] for row in complete) / len(complete)
        if complete
        else None,
        "mean_parent_to_severe_delta": mean([row["parent_to_severe_delta"] for row in complete]),
        "adjacent_mean_deltas_on_complete_families": [
            mean([row["adjacent_deltas"][index] for row in complete]) for index in range(3)
        ],
        "families_with_any_increase": [row["family_id"] for row in complete if not row["non_increasing"]],
        "flat_families": [row["family_id"] for row in complete if len(set(row["scores"])) == 1],
    }


def controlled_analysis(cases, predicted, observed):
    flagged = {
        case["family_id"]
        for case in cases
        if case["stratum"] == "controlled"
        and any(task_flags(observed[model].get(case["case_id"]), task) for model in MODELS for task in TASKS)
    }
    result = {}
    for task in TASKS:
        records = {name: family_records(cases, predicted[name], task) for name in PREDICTIONS}
        common = set.intersection(
            *({row["family_id"] for row in records[name] if row["complete"]} for name in MODELS)
        )
        groupings = {
            "all_operators": None,
            **{operator: operator for operator in sorted({row["operator"] for row in records["average"]})},
        }
        result[task] = {
            "groups": {
                group: {
                    "individual_coverage": {
                        name: family_summary(
                            [row for row in records[name] if operator is None or row["operator"] == operator]
                        )
                        for name in PREDICTIONS
                    },
                    "common_valid": {
                        name: family_summary(
                            [
                                row
                                for row in records[name]
                                if row["family_id"] in common
                                and (operator is None or row["operator"] == operator)
                            ]
                        )
                        for name in PREDICTIONS
                    },
                    "common_valid_without_student_scope_flags": {
                        name: family_summary(
                            [
                                row
                                for row in records[name]
                                if row["family_id"] in common - flagged
                                and (operator is None or row["operator"] == operator)
                            ]
                        )
                        for name in PREDICTIONS
                    },
                }
                for group, operator in groupings.items()
            },
            "families": {
                name: [
                    {
                        **row,
                        "scores": [float(value) if value is not None else None for value in row["scores"]],
                        "adjacent_deltas": [
                            float(value) if value is not None else None for value in row["adjacent_deltas"]
                        ],
                        "parent_to_severe_delta": float(row["parent_to_severe_delta"])
                        if row["parent_to_severe_delta"] is not None
                        else None,
                    }
                    for row in rows
                ]
                for name, rows in records.items()
            },
        }
    return {
        "interpretation": "Negative deltas mean lower scores after stronger synthetic degradation. Technical monotonicity is a consistency diagnostic, not a human accuracy measure. Aesthetic scores need not decline monotonically; their direction is descriptive. Scope-flagged families are also separated because a policy floor can make all four scores identical.",
        "any_student_scope_exclusion_family_ids": sorted(flagged),
        "tasks": result,
    }


def audit_all_attempts(report, settings, reserve_per_request_usd):
    """Reconcile every HTTP attempt, preserving unpriced pre-serve rejections as unknown."""
    attempts, totals, applied_headers, requested_headers = [], Counter(), Counter(), Counter()
    gateway_total = upstream_total = effective_total = Decimal(0)
    unpriced, identities, selected_cases = [], set(), set()
    reserve_per_request = Decimal(reserve_per_request_usd)
    ledger = report["all_attempts"]
    for item in ledger:
        path = Path(item["result_path"])
        identity = f"{item['stage']}:{item['attempt_id']}"
        if str(path) in identities or digest(path) != item["raw_result_sha256"]:
            raise ValueError("Duplicate attempt result or changed raw reply")
        identities.add(str(path))
        raw = read_json(path)
        if raw["http_status"] != item["http_status"]:
            raise ValueError("Raw HTTP status differs from attempt ledger")
        body = json.loads(raw["body"], parse_float=Decimal) if raw.get("body") else {}
        usage = body.get("usage", {})
        gateway, upstream, effective, byok = helpers.raw_cost(usage)
        expected = Decimal(item["cost_usd"]) if item["cost_usd"] is not None else None
        if effective != expected:
            raise ValueError(f"Raw billing differs from attempt ledger: {identity}")
        selected = bool(item["selected"])
        if selected:
            if item["case_id"] in selected_cases:
                raise ValueError("Multiple selected replies for one case")
            selected_cases.add(item["case_id"])
        if effective is None:
            if item["http_status"] != 429 or item["status"] != "rejected_retryable" or selected:
                raise ValueError("Only documented unselected HTTP429 rejections may remain unpriced")
            unpriced.append(identity)
        else:
            effective_total += effective
        gateway_total += gateway or Decimal(0)
        upstream_total += upstream or Decimal(0)
        headers = {key.lower(): value for key, value in raw.get("headers", {}).items()}
        applied = headers.get("x-runinfra-reasoning-effort-applied")
        requested = headers.get("x-runinfra-reasoning-effort-requested")
        if applied is not None:
            applied_headers[applied] += 1
        if requested is not None:
            requested_headers[requested] += 1
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
        attempts.append(
            {
                "case_id": item["case_id"],
                "attempt_id": identity,
                "stage": item["stage"],
                "source_run_id": item.get("source_run_id"),
                "selected": selected,
                "http_status": item["http_status"],
                "status": item["status"],
                "cost_usd": str(effective) if effective is not None else None,
                "cost_source": item["cost_source"],
                "unpriced_liability_reserve_usd": str(
                    reserve_per_request if effective is None else Decimal(0)
                ),
                "reasoning_tokens": reasoning,
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "reasoning_effort_applied_header": applied,
                "reasoning_effort_requested_header": requested,
                "raw_result_sha256": item["raw_result_sha256"],
                "result_path": str(path),
            }
        )
    if effective_total != Decimal(report["known_cost_usd"]) or len(unpriced) != report["unpriced_attempts"]:
        raise ValueError("All-attempt billing totals differ from combined report")
    if len(attempts) != report["http_attempts"] or len(selected_cases) != len(report["attempts"]):
        raise ValueError("Selected-response or HTTP-attempt counts differ")
    reserve = reserve_per_request * len(unpriced)
    if reserve != Decimal(report["unpriced_liability_reserve_usd"]) or effective_total + reserve != Decimal(
        report["admission_debit_usd"]
    ):
        raise ValueError("Held reservations or admission debit differ from combined report")
    if (
        len(selected_cases) != report["selected_response_calls"]
        or len(unpriced) != report["unpriced_rejected_attempts"]
    ):
        raise ValueError("Selected-image or unpriced-rejection counts differ")
    return {
        "calls": len(attempts),
        "selected_response_calls": len(selected_cases),
        "known_cost_usd": str(effective_total),
        "gateway_reported_cost_usd": str(gateway_total),
        "upstream_reported_cost_usd": str(upstream_total),
        "unpriced_attempt_ids": unpriced,
        "cost_complete": not unpriced,
        "unpriced_rejected_attempts": len(unpriced),
        "unpriced_liability_reserve_usd": str(reserve),
        "admission_debit_usd": str(effective_total + reserve),
        "unpriced_cost_interpretation": "HTTP429 costs have no reported usage. They remain unknown; zero is not asserted. Conservative held reservations are budget liabilities, not billed costs.",
        "cost_sources": dict(Counter(item["cost_source"] for item in ledger)),
        "http_status_counts": dict(Counter(str(item["http_status"]) for item in ledger)),
        "reasoning_evidence": {
            "requested_setting": settings.get(
                "reasoning_effort", settings.get("reasoning", {}).get("effort")
            ),
            "applied_header_counts": dict(applied_headers),
            "applied_header_missing_calls": len(attempts) - sum(applied_headers.values()),
            "requested_header_counts": dict(requested_headers),
            "requested_header_missing_calls": len(attempts) - sum(requested_headers.values()),
            "interpretation": "Settings state intent; applied headers report gateway handling. Positive reported reasoning tokens evidence reasoning. Rejected attempts may lack usage and headers.",
        },
        "usage": {
            key: totals[key]
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "prompt_tokens_reported_calls",
                "completion_tokens_reported_calls",
                "total_tokens_reported_calls",
                "reasoning_tokens_reported_calls",
                "positive_reasoning_tokens_calls",
                "byok_calls",
            )
        },
        "attempts": attempts,
    }


def load_runs(cases):
    continuation = module("current_trial_continuation_adapter", HERE / "continuation.py")
    expected = {case["case_id"]: case for case in cases}
    runs = {}
    for model in MODELS:
        run, report, rows, artifact_pins = continuation.load_verified_route(model)
        if report["status"] != "complete" or report["attempted"] != 2000 or len(report["attempts"]) != 2000:
            raise ValueError(f"{model} must finish all 2000 attempts before final analysis")
        if {item["case_id"] for item in report["attempts"]} != set(expected):
            raise ValueError("Attempt case set differs from corpus")
        observed = {row["case_id"]: row for row in rows}
        if len(rows) != len(observed) or not set(observed) <= set(expected):
            raise ValueError("Observation IDs are duplicated or outside the corpus")
        for case_id, row in observed.items():
            if row["view_sha256"] != expected[case_id]["view_sha256"] or any(row["task_masks"].values()):
                raise ValueError("Observation view identity or training mask differs")
        runs[model] = {
            "run": run,
            "report": report,
            "observed": observed,
            "billing": audit_all_attempts(report, run["settings"], run["budget"]["reserve_per_request_usd"]),
            "input_sha256": artifact_pins,
        }
    return runs


def validate_reference_inputs(frozen):
    expected = {
        HERE / "references.jsonl": frozen["references_sha256"],
        HERE / "manifest.json": frozen["manifest_sha256"],
        HERE / "collage-selection.json": frozen["selection_sha256"],
        **{Path(path): value for path, value in frozen["inputs"].items()},
    }
    for path, expected_hash in expected.items():
        if digest(path) != expected_hash:
            raise ValueError(f"Reference freeze input changed: {path}")


def load_references(cases):
    frozen = read_json(HERE / "references-freeze.json")
    validate_reference_inputs(frozen)
    rows = read_rows(HERE / "references.jsonl")
    refs = {row["case_id"]: row for row in rows}
    selection = read_json(HERE / "collage-selection.json")
    if len(rows) != 72 or len(refs) != 72 or set(refs) != set(selection["case_ids"]):
        raise ValueError("References do not match the frozen 72-case subset")
    by_id = {case["case_id"]: case for case in cases}
    for case_id, ref in refs.items():
        if ref["view_sha256"] != by_id[case_id]["view_sha256"] or ref["eligibility"] not in {
            "eligible",
            "excluded",
            "uncertain",
        }:
            raise ValueError("Reference image identity or eligibility differs")
    return refs


def cost_summary(runs):
    models = {
        name: {
            "model": run["run"]["requested_model"],
            "images_with_both_numeric_scores": sum(
                all(score(row, task) is not None for task in TASKS) for row in run["observed"].values()
            ),
            **{key: value for key, value in run["billing"].items() if key != "attempts"},
        }
        for name, run in runs.items()
    }
    return {
        "scope": "Only this 2000-image trial: DeepSeek high and Luna high; previous experiments excluded.",
        "basis": "Exact reported usage costs reconciled against every preserved raw reply, including rejected attempts; confirmed BYOK upstream costs included without double-counting ordinary provider costs. Unpriced HTTP429 costs remain unknown. Held reservations are separate budget liabilities, not billed costs. Not invoice verification.",
        "models": models,
        "totals": {
            "calls": sum(row["calls"] for row in models.values()),
            "selected_response_calls": sum(row["selected_response_calls"] for row in models.values()),
            "known_cost_usd": str(
                sum((Decimal(row["known_cost_usd"]) for row in models.values()), Decimal(0))
            ),
            "unpriced_attempts": sum(len(row["unpriced_attempt_ids"]) for row in models.values()),
            "cost_complete": all(row["cost_complete"] for row in models.values()),
            "unpriced_liability_reserve_usd": str(
                sum((Decimal(row["unpriced_liability_reserve_usd"]) for row in models.values()), Decimal(0))
            ),
            "admission_debit_usd": str(
                sum((Decimal(row["admission_debit_usd"]) for row in models.values()), Decimal(0))
            ),
        },
        "generator_sha256": digest(Path(__file__)),
    }


def billing_by_case(attempts):
    groups = defaultdict(list)
    for attempt in attempts:
        groups[attempt["case_id"]].append(attempt)
    result = {}
    for case_id, group in groups.items():
        selected = [row for row in group if row["selected"]]
        if len(selected) != 1:
            raise ValueError("Per-image billing requires exactly one selected response")
        known = sum((Decimal(row["cost_usd"]) for row in group if row["cost_usd"] is not None), Decimal(0))
        unpriced = sum(row["cost_usd"] is None for row in group)
        reasoning = [row["reasoning_tokens"] for row in group if row["reasoning_tokens"] is not None]
        result[case_id] = {
            "cost_usd": str(known) if not unpriced else None,
            "known_cost_usd": str(known),
            "unpriced_http_attempts": unpriced,
            "http_attempts": len(group),
            "selected_response_cost_usd": selected[0]["cost_usd"],
            "unpriced_liability_reserve_usd": str(
                sum((Decimal(row["unpriced_liability_reserve_usd"]) for row in group), Decimal(0))
            ),
            "reasoning_tokens": sum(reasoning) if reasoning else None,
        }
    return result


def write_csv(cases, refs, predicted, runs):
    billing = {name: billing_by_case(data["billing"]["attempts"]) for name, data in runs.items()}
    output = []
    for case in cases:
        case_id = case["case_id"]
        ref = refs.get(case_id)
        row = {
            key: case.get(key)
            for key in (
                "case_id",
                "image_id",
                "family_id",
                "stratum",
                "operator",
                "severity",
                "vizwiz_stratum",
                "known_anchor",
                "view_sha256",
                "image_path",
            )
        }
        row.update(reference_sample=ref is not None, reference_eligibility=ref["eligibility"] if ref else "")
        for task in TASKS:
            value = score(ref, task)
            row["reference_" + task] = f"{value:.2f}" if value is not None else ""
            for name in PREDICTIONS:
                value = predicted[name][task][case_id]
                row[name + "_" + task] = f"{value:.2f}" if value is not None else ""
                if name in MODELS:
                    observed = runs[name]["observed"].get(case_id, {})
                    rating = observed.get(task, {})
                    row[name + "_" + task + "_status"] = rating.get("status", "missing")
                    row[name + "_" + task + "_validation_errors"] = json.dumps(
                        observed.get("validation_errors", {}).get(task, []), ensure_ascii=False
                    )
                    row[name + "_" + task + "_limitations"] = "|".join(rating.get("limitations", []))
                    row[name + "_" + task + "_evidence"] = rating.get("evidence", "")
        for name in MODELS:
            for key, value in billing[name][case_id].items():
                row[name + "_" + key] = value
        output.append(row)
    with (HERE / "per-image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def reference_cohorts(cases, refs):
    fresh = {case["case_id"] for case in cases if not case["known_anchor"]}
    cohorts = {
        "all_72": sorted(refs),
        "fresh_71": sorted(set(refs) & fresh),
        "eligible_reference": sorted(
            case_id for case_id, ref in refs.items() if ref["eligibility"] == "eligible"
        ),
        "fresh_eligible_reference": sorted(
            case_id for case_id, ref in refs.items() if ref["eligibility"] == "eligible" and case_id in fresh
        ),
    }
    if len(cohorts["all_72"]) != 72 or len(cohorts["fresh_71"]) != 71:
        raise ValueError("Reference sample must contain 71 fresh cases and one known anchor")
    if "T0632" not in cohorts["fresh_71"]:
        raise ValueError("The declared scope-boundary sensitivity case must be in the fresh reference sample")
    cohorts["fresh_70_excluding_scope_boundary"] = [
        case_id for case_id in cohorts["fresh_71"] if case_id != "T0632"
    ]
    return cohorts


def main():
    manifest = read_json(HERE / "manifest.json")
    cases = manifest["cases"]
    if len(cases) != 2000 or len({case["case_id"] for case in cases}) != 2000:
        raise ValueError("Expected the complete 2000-case corpus")
    refs = load_references(cases)
    runs = load_runs(cases)
    observed = {name: data["observed"] for name, data in runs.items()}
    ids = sorted(case["case_id"] for case in cases)
    predicted = predictions(observed, ids)
    cohorts = reference_cohorts(cases, refs)
    grouping = strata(cases)
    analysis = {
        "schema_version": "dual-high-2000-analysis-v1",
        "interpretation": "All-2000 distributions, student agreement and controlled consistency are descriptive, not accuracy. Only the preselected 72-case subset has independent assistant references, with 71 previously unexposed cases and one known poster anchor. Reference agreement is not verified human accuracy; strata are deliberately sampled and not population-weighted. The arithmetic mean requires both students to have a valid numeric score for that task; missing scores are never zero or copied from the other student.",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": digest(HERE / "manifest.json"),
        "references_sha256": digest(HERE / "references.jsonl"),
        "references_freeze_sha256": digest(HERE / "references-freeze.json"),
        "generator_sha256": digest(Path(__file__)),
        "billing_helpers_sha256": digest(Path(helpers.__file__)),
        "cohorts": cohorts,
        "reference_sensitivity": {
            "fresh_70_excluding_scope_boundary": {
                "primary_cohort": "fresh_71",
                "excluded_case_ids": ["T0632"],
                "pre_registered": False,
                "timing": "post_reference_freeze",
                "audit_origin": "Root assistant visual scope-boundary audit before inspecting this case's student outputs",
                "reason": "T0632 depicts a printed page with table visible at right and page angle, shadow and perspective. The frozen reviewer applied the near-full-frame page exclusion; the root reviewer considers eligibility ambiguous because physical context is visible.",
                "frozen_references_changed": False,
                "selection_informed_by_student_scores": False,
                "interpretation": "Secondary sensitivity analysis only. Primary all_72 and fresh_71 results retain the frozen judgment. This exclusion tests dependence on one disputed eligibility decision; it does not determine which judgment is correct.",
            }
        },
        "reference_comparisons": {
            cohort: {task: reference_comparison(predicted, refs, case_ids, task) for task in TASKS}
            for cohort, case_ids in cohorts.items()
        },
        "paired_bootstrap": {
            cohort: paired_bootstrap(
                predicted, refs, cohorts[cohort], {case["case_id"]: case["family_id"] for case in cases}
            )
            for cohort in ("fresh_71", "fresh_70_excluding_scope_boundary")
        },
        "all_2000": {
            "distributions": {
                name: {task: distribution(list(predicted[name][task].values()), len(ids)) for task in TASKS}
                for name in PREDICTIONS
            },
            "student_agreement": {task: student_agreement(predicted, ids, task) for task in TASKS},
            "common_valid_distributions": common_distributions(predicted, ids),
        },
        "strata": {
            stratum: {
                "case_ids": case_ids,
                "distributions": {
                    name: {
                        task: distribution(
                            [predicted[name][task][case_id] for case_id in case_ids], len(case_ids)
                        )
                        for task in TASKS
                    }
                    for name in PREDICTIONS
                },
                "student_agreement": {task: student_agreement(predicted, case_ids, task) for task in TASKS},
                "common_valid_distributions": common_distributions(predicted, case_ids),
                "reference_sample": {
                    task: reference_comparison(predicted, refs, sorted(set(case_ids) & set(refs)), task)
                    for task in TASKS
                },
            }
            for stratum, case_ids in grouping.items()
        },
        "controlled": controlled_analysis(cases, predicted, observed),
        "runs": {
            name: {
                "run_id": data["run"]["run_id"],
                "model": data["run"]["requested_model"],
                "settings": data["run"]["settings"],
                "input_sha256": data["input_sha256"],
                "validity": validity(data["observed"], ids),
                "raw_export_report_summary": {
                    key: value
                    for key, value in data["report"].items()
                    if key not in {"attempts", "all_attempts"}
                },
                "usage_and_cost": data["billing"],
                "reference_scope_audit": scope_audit(data["observed"], refs),
                "all_2000_scope_flagged_case_ids": [
                    case_id
                    for case_id in ids
                    if any(task_flags(data["observed"].get(case_id), task) for task in TASKS)
                ],
            }
            for name, data in runs.items()
        },
    }
    write_json(HERE / "analysis.json", analysis)
    costs = cost_summary(runs)
    write_json(HERE / "cost-summary.json", costs)
    write_csv(cases, refs, predicted, runs)
    print(
        json.dumps(
            {
                "costs": costs["totals"],
                "fresh_71_common_valid": {
                    task: analysis["reference_comparisons"]["fresh_71"][task]["common_valid"]
                    for task in TASKS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
