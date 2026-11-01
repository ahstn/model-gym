"""Reference agreement, paired family uncertainty and shadow-QC diagnostics.

All comparisons are conditional on assistant references, not human ground truth.
No thresholds are fit to the benchmark and no score is changed by these helpers.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from itertools import combinations

import numpy as np

TASKS = ("technical_score", "aesthetic_score")
PANELS = ("representative", "challenge", "development", "evaluation_pooled_exploratory")
BASE_MODELS = ("luna", "deepseek", "jev", "liquid", "qrealign_lite")
SEED = 20261101
INCLUSIVE_ABS_TOL = 1e-12


def at_most(value: float, boundary: float) -> bool:
    """Absorb binary arithmetic noise at inclusive boundaries only."""
    return value <= boundary or math.isclose(value, boundary, rel_tol=0, abs_tol=INCLUSIVE_ABS_TOL)


def at_least(value: float, boundary: float) -> bool:
    return value >= boundary or math.isclose(value, boundary, rel_tol=0, abs_tol=INCLUSIVE_ABS_TOL)


def in_panel(row: dict, panel: str) -> bool:
    if panel == "evaluation_pooled_exploratory":
        return row["benchmark_partition"] in {"representative", "challenge"}
    return row["benchmark_partition"] == panel


def seed_for(*parts: str) -> int:
    return SEED + int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:8], 16)


def ranks(values: list[float]) -> np.ndarray:
    """Average ranks for ties; avoids an additional statistical dependency."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    result = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    x, y = ranks(left), ranks(right)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def numeric(row: dict, task: str) -> bool:
    return bool(row.get(task + "_valid")) and row.get(task) is not None


def agreement(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n": 0, "mae": None, "bias": None, "spearman": None}
    prediction, reference = map(list, zip(*pairs, strict=True))
    error = np.asarray(prediction) - reference
    return {
        "n": len(pairs),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman": spearman(prediction, reference),
    }


def accuracy(rows: list[dict]) -> dict:
    references = {r["case_id"]: r for r in rows if r["model"] == "reference"}
    scorers = {(r["model"], r["arm"]) for r in rows if r["model"] != "reference"}
    lookup = {(r["case_id"], r["model"], r["arm"]): r for r in rows}
    output = []
    for panel in PANELS:
        expected = [identity for identity, row in references.items() if in_panel(row, panel)]
        fair_by_task = {
            task: {
                identity
                for identity in expected
                if numeric(references[identity], task)
                and all(numeric(lookup[(identity, model, "baseline")], task) for model in BASE_MODELS)
            }
            for task in TASKS
        }
        fair_ids = {
            identity
            for identity in expected
            if all(
                numeric(references[identity], task) and numeric(lookup[(identity, model, "baseline")], task)
                for task in TASKS
                for model in BASE_MODELS
            )
        }
        for model, arm in sorted(scorers):
            available = [
                (lookup[(identity, model, arm)], references[identity])
                for identity in expected
                if (identity, model, arm) in lookup
            ]
            item = {"panel": panel, "model": model, "arm": arm, "panel_images": len(expected)}
            for task in TASKS:
                item[task] = agreement(
                    [
                        (pred[task], ref[task])
                        for pred, ref in available
                        if numeric(pred, task) and numeric(ref, task)
                    ]
                )
                native = [
                    (pred[task + "_native"], ref[task])
                    for pred, ref in available
                    if ref.get("reference_eligibility") == "eligible"
                    and numeric(pred, task)
                    and numeric(ref, task)
                    and pred.get(task + "_native") is not None
                ]
                item[task + "_native_on_eligible"] = agreement(native)
                item[task + "_all_five_baselines_common"] = agreement(
                    [
                        (pred[task], ref[task])
                        for pred, ref in available
                        if pred["case_id"] in fair_by_task[task] and numeric(pred, task)
                    ]
                )
            common = [
                (pred, ref)
                for pred, ref in available
                if all(numeric(pred, task) and numeric(ref, task) for task in TASKS)
            ]
            item["combined_common_tasks"] = combined(common)
            item["all_five_baselines_common"] = combined(
                [(pred, ref) for pred, ref in common if pred["case_id"] in fair_ids]
            )
            item["fair_baseline_case_count"] = len(fair_ids)
            item["fair_baseline_task_case_counts"] = {
                task: len(identities) for task, identities in fair_by_task.items()
            }
            output.append(item)
    return {"reference_type": "assistant_judgments_not_human_ground_truth", "panels": output}


def combined(pairs: list[tuple[dict, dict]]) -> dict:
    return {
        "n": len(pairs),
        "mae": float(np.mean([sum(abs(pred[task] - ref[task]) for task in TASKS) / 2 for pred, ref in pairs]))
        if pairs
        else None,
    }


def bootstrap_families(
    values: list[float], families: list[str], *, repeats: int = 10000, seed: int = SEED
) -> dict:
    """Resample whole source families; retain every image and its paired contrast."""
    if len(values) != len(families) or repeats < 1:
        raise ValueError("Paired values/families and a positive repeat count required")
    if not values:
        return {
            "n_images": 0,
            "n_families": 0,
            "estimate": None,
            "ci95": None,
            "upper_one_sided95": None,
            "bootstrap_repeats": repeats,
            "seed": seed,
        }
    groups = defaultdict(list)
    for value, family in zip(values, families, strict=True):
        if not math.isfinite(value):
            raise ValueError("Bootstrap values must be finite")
        groups[family].append(value)
    ordered = [groups[family] for family in sorted(groups)]
    totals = np.asarray([sum(group) for group in ordered])
    counts = np.asarray([len(group) for group in ordered])
    rng, sampled = np.random.default_rng(seed), []
    for start in range(0, repeats, 250):
        index = rng.integers(0, len(groups), size=(min(250, repeats - start), len(groups)))
        sampled.extend(np.sum(totals[index], axis=1) / np.sum(counts[index], axis=1))
    return {
        "n_images": len(values),
        "n_families": len(groups),
        "estimate": float(np.mean(values)),
        "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])] if len(groups) > 1 else None,
        "upper_one_sided95": float(np.quantile(sampled, 0.95)) if len(groups) > 1 else None,
        "bootstrap_repeats": repeats,
        "seed": seed,
        "interval_limit": "Conditional family bootstrap; assistant-label uncertainty is not quantified",
    }


def baseline_comparisons(rows: list[dict], *, repeats: int = 10000) -> list[dict]:
    """Exploratory paired differences from the fixed ensemble; never select cases by score."""
    lookup = {(row["case_id"], row["model"], row["arm"]): row for row in rows}
    references = [row for row in rows if row["model"] == "reference"]
    output = []
    for panel in ("representative", "challenge"):
        panel_references = [row for row in references if in_panel(row, panel)]
        for model in BASE_MODELS:
            item = {
                "panel": panel,
                "model": model,
                "comparator": "luna_deepseek_mean",
                "arm": "baseline",
                "expected_panel_images": len(panel_references),
                "contrast": "individual_mae_minus_fixed_ensemble_mae",
                "interpretation": "Negative favors the individual point estimate; positive favors the ensemble",
                "exploratory": True,
                "multiplicity_adjustment": "none",
                "reference_basis": "Conditional on assistant judgments, not human ground truth",
            }
            for field, tasks in ((TASKS[0], (TASKS[0],)), (TASKS[1], (TASKS[1],)), ("combined_error", TASKS)):
                deltas, families = [], []
                for reference in panel_references:
                    individual = lookup[(reference["case_id"], model, "baseline")]
                    ensemble = lookup[(reference["case_id"], "luna_deepseek_mean", "baseline")]
                    if all(numeric(row, task) for row in (reference, individual, ensemble) for task in tasks):
                        deltas.append(
                            sum(
                                abs(individual[task] - reference[task])
                                - abs(ensemble[task] - reference[task])
                                for task in tasks
                            )
                            / len(tasks)
                        )
                        families.append(reference["family_id"])
                item[field] = bootstrap_families(deltas, families, repeats=repeats, seed=SEED)
            output.append(item)
    return output


def treatment_effects(rows: list[dict], *, repeats: int = 10000) -> list[dict]:
    lookup = {(r["case_id"], r["model"], r["arm"]): r for r in rows}
    references = [r for r in rows if r["model"] == "reference"]
    output = []
    for panel in PANELS:
        for model in ("luna_deepseek_mean", "luna", "deepseek"):
            for arm in ("own_stats", "swapped_stats"):
                item = {
                    "panel": panel,
                    "model": model,
                    "arm": arm,
                    "contrast": "treatment_mae_minus_baseline_mae",
                    "primary": panel == "representative"
                    and model == "luna_deepseek_mean"
                    and arm == "own_stats",
                    "secondary_status": "exploratory; no multiplicity-adjusted claim",
                }
                for task in TASKS:
                    deltas, families = [], []
                    panel_references = [ref for ref in references if in_panel(ref, panel)]
                    baseline_numeric, treatment_numeric = 0, 0
                    for ref in panel_references:
                        base = lookup[(ref["case_id"], model, "baseline")]
                        treatment = lookup[(ref["case_id"], model, arm)]
                        baseline_numeric += numeric(base, task)
                        treatment_numeric += numeric(treatment, task)
                        if all(numeric(row, task) for row in (ref, base, treatment)):
                            deltas.append(abs(treatment[task] - ref[task]) - abs(base[task] - ref[task]))
                            families.append(ref["family_id"])
                    item[task] = bootstrap_families(deltas, families, repeats=repeats, seed=SEED)
                    expected = len(panel_references)
                    coverage = len(deltas) / expected if expected else None
                    item[task].update(
                        expected_panel_images=expected,
                        numeric_reference_cases=sum(numeric(ref, task) for ref in panel_references),
                        paired_cases=len(deltas),
                        paired_coverage=coverage,
                        baseline_missing_or_invalid_scores=expected - baseline_numeric,
                        treatment_missing_or_invalid_scores=expected - treatment_numeric,
                        coverage_and_missingness_eligible=coverage is not None
                        and coverage >= 0.95
                        and treatment_numeric >= baseline_numeric,
                    )
                technical, aesthetic = (item[task] for task in TASKS)
                item["technical_practical_and_interval_target_met"] = (
                    at_most(technical["estimate"], -0.05) and technical["ci95"][1] < 0
                    if technical["ci95"] is not None
                    else None
                )
                item["aesthetic_noninferiority_upper_margin"] = 0.03
                item["aesthetic_one_sided95_noninferiority_met"] = (
                    at_most(aesthetic["upper_one_sided95"], 0.03)
                    if aesthetic["upper_one_sided95"] is not None
                    else None
                )
                item["comparison_decision_eligible"] = all(
                    item[task]["coverage_and_missingness_eligible"] for task in TASKS
                )
                item["comparison_status"] = (
                    "eligible_for_interpretation"
                    if item["comparison_decision_eligible"]
                    else "incomplete_coverage_or_degraded_validity"
                )
                item["comparison_decision_ready"] = (
                    item["comparison_decision_eligible"]
                    and item["technical_practical_and_interval_target_met"] is True
                    and item["aesthetic_one_sided95_noninferiority_met"] is True
                )
                item["minimum_paired_coverage_per_task"] = 0.95
                item["protected_slice_review_status"] = "pending_independent_review"
                item["advancement_ready"] = False
                item["protected_slice_review"] = (
                    "Requires separate scope/intentional-effects inspection; these tests alone do not approve labels"
                )
                output.append(item)
    return output


def descriptive(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["model"] != "reference":
            grouped[(row["model"], row["arm"])].append(row)
    output = []
    for (model, arm), group in sorted(grouped.items()):
        item = {
            "model": model,
            "arm": arm,
            "expected_images": len(group),
            "scope_floor_applied": sum(row.get("scope_floor_applied") is True for row in group),
        }
        for task in TASKS:
            values = [row[task] for row in group if numeric(row, task)]
            item[task] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "quantiles_0_10_50_90_100": [float(x) for x in np.quantile(values, [0, 0.1, 0.5, 0.9, 1])]
                if values
                else None,
            }
        output.append(item)
    return output


def between_models(rows: list[dict]) -> list[dict]:
    """Full-corpus agreement, without claiming correctness where no reference exists."""
    output = []
    lookup = {(row["case_id"], row["model"], row["arm"]): row for row in rows}
    for arm in ("baseline", "own_stats", "swapped_stats"):
        identities = sorted({row["case_id"] for row in rows if row["arm"] == arm})
        models = sorted({row["model"] for row in rows if row["arm"] == arm})
        for left, right in combinations(models, 2):
            item = {
                "arm": arm,
                "left": left,
                "right": right,
                "scope": "Agreement only; derived mean shares component information",
            }
            for task in TASKS:
                pairs = [
                    (lookup[(identity, left, arm)][task], lookup[(identity, right, arm)][task])
                    for identity in identities
                    if all(
                        (identity, model, arm) in lookup and numeric(lookup[(identity, model, arm)], task)
                        for model in (left, right)
                    )
                ]
                summary = agreement(pairs)
                item[task] = {
                    "n": summary["n"],
                    "mean_absolute_difference": summary["mae"],
                    "signed_left_minus_right": summary["bias"],
                    "spearman": summary["spearman"],
                }
            output.append(item)
    return output


def scope_checks(rows: list[dict]) -> list[dict]:
    references = {row["case_id"]: row for row in rows if row["model"] == "reference"}
    groups = defaultdict(list)
    for row in rows:
        ref = references.get(row["case_id"])
        if row["model"] != "reference" and ref:
            groups[
                (row["model"], row["arm"], ref["benchmark_partition"], ref["reference_eligibility"])
            ].append(row)
    return [
        {
            "model": model,
            "arm": arm,
            "panel": panel,
            "reference_eligibility": eligibility,
            "n": len(group),
            "both_numeric": sum(all(numeric(row, t) for t in TASKS) for row in group),
            "floor_declared": sum(row.get("scope_floor_applied") is True for row in group),
            "both_scores_at_1": sum(all(numeric(row, t) and row[t] == 1 for t in TASKS) for row in group),
            "either_score_above_1": sum(any(numeric(row, t) and row[t] > 1 for t in TASKS) for row in group),
        }
        for (model, arm, panel, eligibility), group in sorted(groups.items())
    ]


def ladders(cases: list[dict], rows: list[dict]) -> dict:
    families = defaultdict(dict)
    for case in cases:
        if case.get("stratum") == "controlled":
            severity = 0 if case.get("role") == "source_parent" else case.get("severity")
            if severity not in (0, 1, 2, 3) or severity in families[case["family_id"]]:
                raise ValueError("Controlled family has duplicate/invalid severity")
            families[case["family_id"]][severity] = case
    lookup = {(row["case_id"], row["model"]): row for row in rows if row["arm"] == "baseline"}
    details = []
    for family, members in sorted(families.items()):
        if set(members) != {0, 1, 2, 3}:
            raise ValueError("Controlled family must retain parent and all three variants")
        for model in (*BASE_MODELS, "luna_deepseek_mean"):
            scores = [lookup[(members[s]["case_id"], model)].get("technical_score") for s in range(4)]
            complete = all(score is not None for score in scores)
            steps = np.diff(scores) if complete else []
            details.append(
                {
                    "family_id": family,
                    "operator": members[0].get("operator"),
                    "model": model,
                    "complete": complete,
                    "technical_scores": scores,
                    "nonincreasing": bool(np.all(steps <= 1e-8)) if complete else None,
                    "adjacent_ties": int(np.sum(np.abs(steps) <= 1e-8)) if complete else None,
                    "strict_inversions": int(np.sum(steps > 1e-8)) if complete else None,
                    "severe_at_least_parent": scores[3] >= scores[0] - 1e-8 if complete else None,
                    "parent_minus_severity3": scores[0] - scores[3] if complete else None,
                }
            )
    summaries = []
    for model in (*BASE_MODELS, "luna_deepseek_mean"):
        for operator in ["all", *sorted({d["operator"] for d in details if d["operator"] is not None})]:
            group = [
                d for d in details if d["model"] == model and (operator == "all" or d["operator"] == operator)
            ]
            complete = [d for d in group if d["complete"]]
            summaries.append(
                {
                    "model": model,
                    "operator": operator,
                    "families": len(group),
                    "complete_families": len(complete),
                    "nonincreasing_families": sum(d["nonincreasing"] for d in complete),
                    "adjacent_ties": sum(d["adjacent_ties"] for d in complete),
                    "strict_inversions": sum(d["strict_inversions"] for d in complete),
                    "severe_at_least_parent": sum(d["severe_at_least_parent"] for d in complete),
                    "mean_parent_minus_severity3": float(
                        np.mean([d["parent_minus_severity3"] for d in complete])
                    )
                    if complete
                    else None,
                }
            )
    return {
        "scope": "All controlled families; no proxy-based exclusions or aesthetic monotonicity requirement",
        "pace_caveat": "Parent-to-distortion comparison is not PACE ground-truth-threshold Holistic Override Rate",
        "verified_subset_status": "No independently verified subset supplied",
        "summary": summaries,
        "families": details,
    }


def qc_review(
    rows: list[dict], stats: dict[str, dict], thresholds: dict, *, budget_fraction: float = 0.10
) -> dict:
    """Shadow review rule with a fixed budget and random unflagged audit fill."""
    references = {r["case_id"]: r for r in rows if r["model"] == "reference"}
    models = sorted({(r["model"], r["arm"]) for r in rows if r["model"] != "reference"})
    output = []
    for panel in PANELS:
        for model, arm in models:
            candidates = []
            for row in rows:
                ref, measure = references.get(row["case_id"]), stats.get(row["case_id"])
                if (
                    row["model"] != model
                    or row["arm"] != arm
                    or ref is None
                    or measure is None
                    or not in_panel(ref, panel)
                    or not numeric(row, TASKS[0])
                    or not numeric(ref, TASKS[0])
                ):
                    continue
                metrics = measure["metrics"]
                sharp, noise = metrics["sharpness_peak"], metrics["noise_sigma"]
                gate = bool(measure.get("gate"))
                percentiles = measure["percentiles"]
                soft = percentiles["sharpness_peak"] <= 25 and row[TASKS[0]] >= 2
                noisy = percentiles["noise_sigma"] >= 90 and row[TASKS[0]] >= 2
                candidates.append(
                    {
                        "case_id": row["case_id"],
                        "large_error": at_least(abs(row[TASKS[0]] - ref[TASKS[0]]), 0.75),
                        "gate": gate,
                        "soft": soft,
                        "noisy": noisy,
                        "priority": (
                            0 if gate else 1 if soft else 2 if noisy else 3,
                            sharp if soft else -noise if noisy else 0,
                            seed_for(row["case_id"], model, arm, panel),
                        ),
                    }
                )
            candidates.sort(key=lambda c: c["priority"])
            panel_images = sum(in_panel(ref, panel) for ref in references.values())
            planned_budget = math.ceil(panel_images * budget_fraction)
            budget = min(len(candidates), planned_budget)
            flagged = [c for c in candidates if c["gate"] or c["soft"] or c["noisy"]]
            unflagged = [c for c in candidates if not (c["gate"] or c["soft"] or c["noisy"])]
            audit_rng = np.random.default_rng(seed_for("qc-unflagged-audit", model, arm, panel))
            unflagged = [unflagged[index] for index in audit_rng.permutation(len(unflagged))]
            reserved_audits = math.ceil(budget * 0.20)
            audits = min(reserved_audits, len(unflagged))
            selected = flagged[: budget - audits] + unflagged[:audits]
            # Reallocate only when one pool cannot fill its assigned slots.
            selected += unflagged[audits : audits + max(0, budget - len(selected))]
            positives = sum(c["large_error"] for c in candidates)
            rng = np.random.default_rng(seed_for("qc-random", model, arm, panel))
            random_hits = []
            if budget:
                labels = np.asarray([c["large_error"] for c in candidates])
                random_hits = [
                    int(np.sum(labels[rng.choice(len(labels), budget, replace=False)])) for _ in range(1000)
                ]
            item = {
                "panel": panel,
                "model": model,
                "arm": arm,
                "valid_reference_cases_with_stats": len(candidates),
                "large_errors": positives,
                "panel_images": panel_images,
                "planned_review_budget": planned_budget,
                "review_budget": budget,
                "budget_fraction": budget_fraction,
                "random_unflagged_audit_fraction": 0.20,
                "reserved_random_audit_slots": reserved_audits,
                "available_unflagged_cases": len(unflagged),
                "flagged": hits(flagged, positives),
                "fixed_budget": hits(selected, positives),
                "unflagged_audits_in_budget": sum(
                    not (c["gate"] or c["soft"] or c["noisy"]) for c in selected
                ),
                "selected_case_ids": [c["case_id"] for c in selected],
                "random_unflagged_audit_case_ids": [
                    c["case_id"] for c in selected if not (c["gate"] or c["soft"] or c["noisy"])
                ],
                "random_review_expected_large_errors": budget * positives / len(candidates)
                if candidates
                else None,
                "random_1000_draws_hit_ci95": [float(x) for x in np.quantile(random_hits, [0.025, 0.975])]
                if random_hits
                else None,
            }
            output.append(item)
    return {
        "thresholds": thresholds,
        "scope": "Review-only flags; no label overwrite; thresholds exclude benchmark families",
        "large_error_absolute_technical_threshold": 0.75,
        "rows": output,
    }


def hits(candidates: list[dict], positives: int) -> dict:
    true_positive = sum(c["large_error"] for c in candidates)
    return {
        "reviewed": len(candidates),
        "large_errors_found": true_positive,
        "precision": true_positive / len(candidates) if candidates else None,
        "recall": true_positive / positives if positives else None,
    }
