"""Prepare protected-style and photographic-scope checks for visual adjudication.

This helper never changes a label or approves a treatment. Every output remains
pending independent visual review, including improvements and incomplete cases.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import analysis_metrics as metrics
import analyze
import api_contract
import api_runner
import reference_review

HERE = Path(__file__).resolve().parent
MODELS = ("luna", "deepseek", "luna_deepseek_mean")
ARMS = ("own_stats", "swapped_stats")
TASKS = ("technical_score", "aesthetic_score")
BENCHMARK = {"development", "representative", "challenge"}


def rebind_protected(
    original_path: Path,
    revisions_path: Path,
    freeze_path: Path,
    *,
    rebound_at: str | None = None,
) -> dict:
    """Version membership after an explicitly audited, pre-inference replacement."""
    original = api_runner.read(original_path)
    declaration = api_runner.read(revisions_path)
    freeze = api_runner.read(freeze_path)
    if original.get("revision_binding"):
        raise ValueError("Rebinding requires the original prospective selection")
    old = declaration["original_manifest"]
    if old["sha256"] != original["benchmark_manifest_sha256"]:
        raise ValueError("Revision does not start from the prospectively selected benchmark")
    old_cases = reference_review.benchmark_cases(Path(old["path"]))
    _, active, revisions = reference_review.revision_inputs(
        revisions_path, {"manifest_path": old["path"], "manifest_sha256": old["sha256"]}, old_cases
    )
    frozen_revision = freeze.get("reference_revision") or {}
    if (
        frozen_revision.get("declaration") != declaration
        or frozen_revision.get("declaration_sha256") != api_contract.file_sha(revisions_path)
        or Path(frozen_revision.get("declaration_path", "")).resolve() != revisions_path.resolve()
    ):
        raise ValueError("Reference freeze does not bind the exact declared revision")
    for path, expected in original["reference_files_sha256"].items():
        if freeze["review_files"].get(path) != expected or api_contract.file_sha(path) != expected:
            raise ValueError("Original reviewed references changed after slice selection")
    for path, expected in freeze["review_files"].items():
        if api_contract.file_sha(path) != expected:
            raise ValueError("Frozen replacement review changed")
    reference_path = Path(freeze["references_path"])
    if api_contract.file_sha(reference_path) != freeze["references_sha256"]:
        raise ValueError("Frozen references changed")
    references = analyze.read_lines(reference_path)
    ref_by_id = {row["case_id"]: row for row in references}
    if len(references) != 500 or len(ref_by_id) != 500 or set(ref_by_id) != set(active):
        raise ValueError("Rebinding requires all 500 active references")
    for identity, row in ref_by_id.items():
        reference_review.validate_assessment(row)
        if row["view_sha256"] != active[identity]["view_sha256"]:
            raise ValueError("Frozen replacement reference view mismatch")
    if set(revisions) & {row["case_id"] for row in original["style_cases"]}:
        raise ValueError("Replaced style cases require a new explicit visual attribute review")
    original_scope = {row["case_id"]: row for row in original["scope_cases"]}
    if len(original_scope) != len(original["scope_cases"]):
        raise ValueError("Duplicate prospective scope identity")
    scope = {
        identity: {
            "case_id": identity,
            "view_sha256": row["view_sha256"],
            "reference_eligibility": row["eligibility"],
        }
        for identity, row in ref_by_id.items()
        if row["eligibility"] != "eligible"
    }
    changed_scope = {
        identity
        for identity in set(scope) | set(original_scope)
        if scope.get(identity) != original_scope.get(identity)
    }
    if not changed_scope <= set(revisions):
        raise ValueError("Undeclared scope membership changed")
    created_at = rebound_at or api_runner.now()
    if timestamp(created_at) < max(timestamp(original["created_at"]), timestamp(freeze["frozen_at"])):
        raise ValueError("Protected rebind must follow primary reference freeze")
    revised = copy.deepcopy(original)
    revised.update(
        created_at=created_at,
        status="explicit benchmark-revision rebind before model calls; original selection preserved",
        benchmark_manifest_sha256=declaration["active_manifest"]["sha256"],
        reference_files_sha256=freeze["review_files"],
        scope_cases=[scope[identity] for identity in sorted(scope)],
        revision_binding={
            "original_protected_path": str(original_path.resolve()),
            "original_protected_sha256": api_contract.file_sha(original_path),
            "original_selected_at": original["created_at"],
            "declaration_path": str(revisions_path.resolve()),
            "declaration_sha256": api_contract.file_sha(revisions_path),
            "freeze_path": str(freeze_path.resolve()),
            "freeze_sha256": api_contract.file_sha(freeze_path),
            "revised_case_ids": sorted(revisions),
            "style_membership_changed": False,
            "scope_changes": [
                {"case_id": identity, "previous": original_scope.get(identity), "active": scope.get(identity)}
                for identity in sorted(changed_scope)
            ],
            "selection_uses_model_results": False,
        },
    )
    return revised


def validate_revision_binding(protected: dict, freeze: dict, freeze_path: Path) -> None:
    binding = protected.get("revision_binding")
    if freeze.get("reference_revision") and not binding:
        raise ValueError("Revised benchmark requires an explicit protected-selection rebind")
    if binding:
        original = Path(binding["original_protected_path"])
        if (
            api_contract.file_sha(original) != binding["original_protected_sha256"]
            or api_contract.file_sha(freeze_path) != binding["freeze_sha256"]
            or Path(binding["freeze_path"]).resolve() != freeze_path.resolve()
        ):
            raise ValueError("Protected rebind provenance changed")
        expected = rebind_protected(
            original, Path(binding["declaration_path"]), freeze_path, rebound_at=protected["created_at"]
        )
        if protected != expected:
            raise ValueError("Protected rebind differs from declared revision and original membership")


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed > datetime.now(UTC):
        raise ValueError("Expected a past timezone-aware timestamp")
    return parsed


def valid(row: dict, task: str) -> bool:
    return bool(row.get(task + "_valid")) and row.get(task) is not None


def reference_score(reference: dict, task: str) -> float | None:
    original = "technical_quality" if task == "technical_score" else "aesthetics"
    return reference[original]["score"] if reference[original]["status"] == "ok" else None


def style_comparison(reference: dict, baseline: dict, treatment: dict, task: str) -> dict:
    score = reference_score(reference, task)
    baseline_valid, treatment_valid = valid(baseline, task), valid(treatment, task)
    baseline_error = abs(baseline[task] - score) if baseline_valid and score is not None else None
    treatment_error = abs(treatment[task] - score) if treatment_valid and score is not None else None
    delta = (
        treatment_error - baseline_error
        if baseline_error is not None and treatment_error is not None
        else None
    )
    return {
        "task": task,
        "reference_score": score,
        "baseline_score": baseline.get(task) if baseline_valid else None,
        "treatment_score": treatment.get(task) if treatment_valid else None,
        "baseline_absolute_error": baseline_error,
        "treatment_absolute_error": treatment_error,
        "delta_absolute_error": delta,
        "large_error_increase": metrics.at_least(delta, 0.50) if delta is not None else False,
        "new_missing_task": baseline_valid and not treatment_valid,
        "recovered_missing_task": not baseline_valid and treatment_valid,
        "improvement": delta < 0 if delta is not None else False,
        "paired_comparison_complete": delta is not None,
        "adjudication_status": "pending_visual_review",
    }


def scope_state(reference_eligibility: str, row: dict) -> dict:
    both_numeric = all(valid(row, task) for task in TASKS)
    floor = row.get("scope_floor_applied")
    declared_floor = floor if type(floor) is bool else None
    both_one = both_numeric and all(row[task] == 1 for task in TASKS)
    any_above = any(valid(row, task) and row[task] > 1 for task in TASKS)
    if reference_eligibility == "eligible":
        error = (
            True if declared_floor is True else False if both_numeric and declared_floor is False else None
        )
    elif reference_eligibility == "excluded":
        error = True if any_above else False if both_one else None
    else:
        error = None
    return {
        "both_numeric": both_numeric,
        "declared_floor": declared_floor,
        "both_scores_one": both_one,
        "any_numeric_above_one": any_above,
        "scope_error": error,
        "missing_or_invalid_tasks": [task for task in TASKS if not valid(row, task)],
    }


def scope_comparison(reference: dict, baseline: dict, treatment: dict) -> dict:
    eligibility = reference["eligibility"]
    before, after = scope_state(eligibility, baseline), scope_state(eligibility, treatment)
    new_error = before["scope_error"] is False and after["scope_error"] is True
    floor_flip = (
        eligibility == "uncertain"
        and before["both_numeric"]
        and after["both_numeric"]
        and before["declared_floor"] is not None
        and after["declared_floor"] is not None
        and before["declared_floor"] != after["declared_floor"]
    )
    incomplete = (
        not before["both_numeric"]
        or not after["both_numeric"]
        or eligibility != "excluded"
        and (before["declared_floor"] is None or after["declared_floor"] is None)
    )
    return {
        "reference_eligibility": eligibility,
        "baseline": before,
        "treatment": after,
        "new_scope_error": new_error,
        "new_scope_error_type": (
            "new_declared_floor_on_eligible"
            if eligibility == "eligible"
            else "new_score_above_one_on_excluded"
        )
        if new_error
        else None,
        "scope_improvement": before["scope_error"] is True and after["scope_error"] is False,
        "incomplete_comparison": incomplete,
        "uncertain_floor_flip_descriptive_only": floor_flip,
        "adjudication_status": "pending_visual_review",
    }


def evaluate(protected: dict, references: list[dict], scores: list[dict]) -> dict:
    lookup = {(row["case_id"], row["model"], row["arm"]): row for row in scores}
    if len(lookup) != len(scores):
        raise ValueError("Duplicate score identity")
    ref_by_id = {row["case_id"]: row for row in references}
    if len(ref_by_id) != len(references):
        raise ValueError("Duplicate reference identity")
    style_by_id = {row["case_id"]: row for row in protected["style_cases"]}
    if len(style_by_id) != len(protected["style_cases"]) or not set(style_by_id) <= set(ref_by_id):
        raise ValueError("Invalid protected-style membership")
    style_rows, scope_rows, queue = [], [], {}
    for reference in references:
        identity = reference["case_id"]
        for model in MODELS:
            baseline = lookup[(identity, model, "baseline")]
            for arm in ARMS:
                treatment = lookup[(identity, model, arm)]
                common = {
                    "case_id": identity,
                    "view_sha256": reference["view_sha256"],
                    "image_path": baseline.get("image_path"),
                    "model": model,
                    "arm": arm,
                    "baseline_attempt_id": baseline.get("attempt_id"),
                    "treatment_attempt_id": treatment.get("attempt_id"),
                }
                scope = {**common, **scope_comparison(reference, baseline, treatment)}
                scope_rows.append(scope)
                reasons = []
                if identity in style_by_id:
                    reasons.append("prospectively_selected_style_case")
                    for task in TASKS:
                        item = {
                            **common,
                            "style_tags": style_by_id[identity]["tags"],
                            **style_comparison(reference, baseline, treatment, task),
                        }
                        style_rows.append(item)
                        if item["large_error_increase"]:
                            reasons.append(task + ":absolute_error_increase_at_least_0.50")
                        if item["new_missing_task"]:
                            reasons.append(task + ":new_missing_or_invalid_task")
                if scope["new_scope_error"]:
                    reasons.append(scope["new_scope_error_type"])
                if scope["incomplete_comparison"]:
                    reasons.append("scope_comparison_incomplete_not_compliant")
                if scope["uncertain_floor_flip_descriptive_only"]:
                    reasons.append("uncertain_reference_floor_flip_descriptive_only")
                if reasons:
                    queue[(identity, model, arm)] = {
                        **common,
                        "reasons": reasons,
                        "reference_eligibility": reference["eligibility"],
                        "scope_comparison": scope,
                        "adjudication_status": "pending_visual_review",
                        "adjudication": None,
                    }
    grouped = defaultdict(list)
    for row in style_rows:
        grouped[(row["model"], row["arm"], row["task"])].append(row)
    summaries = [
        {
            "model": model,
            "arm": arm,
            "task": task,
            "expected_style_cases": len(protected["style_cases"]),
            "paired_cases": sum(row["paired_comparison_complete"] for row in group),
            "large_error_increases": sum(row["large_error_increase"] for row in group),
            "new_missing_tasks": sum(row["new_missing_task"] for row in group),
            "recovered_missing_tasks": sum(row["recovered_missing_task"] for row in group),
            "improvements": sum(row["improvement"] for row in group),
        }
        for (model, arm, task), group in sorted(grouped.items())
    ]
    scope_groups = defaultdict(list)
    for row in scope_rows:
        scope_groups[(row["model"], row["arm"], row["reference_eligibility"])].append(row)
    scope_summaries = [
        {
            "model": model,
            "arm": arm,
            "reference_eligibility": eligibility,
            "cases": len(group),
            "new_scope_errors": sum(row["new_scope_error"] for row in group),
            "incomplete_comparisons": sum(row["incomplete_comparison"] for row in group),
            "scope_improvements": sum(row["scope_improvement"] for row in group),
            "uncertain_floor_flips_descriptive_only": sum(
                row["uncertain_floor_flip_descriptive_only"] for row in group
            ),
        }
        for (model, arm, eligibility), group in sorted(scope_groups.items())
    ]
    return {
        "style_rows": style_rows,
        "scope_rows": scope_rows,
        "queue": list(queue.values()),
        "summary": {
            "reference_cases": len(references),
            "selected_style_cases": len(style_by_id),
            "style_task_comparisons": len(style_rows),
            "scope_comparisons": len(scope_rows),
            "style": summaries,
            "scope": scope_summaries,
            "review_queue_items": len(queue),
            "queue_reason_counts": dict(
                Counter(reason for row in queue.values() for reason in row["reasons"])
            ),
            "adjudication_status": "pending_visual_review",
            "advancement_ready": False,
            "training_approval": False,
            "labels_changed": False,
            "interpretation": "Assistant-reference review queue; no automatic acceptance or label changes",
        },
    }


def validate_bindings(
    protected: dict,
    benchmark: dict,
    manifest: dict,
    references: list[dict],
    freeze: dict,
    scores: list[dict],
    *,
    paths: dict[str, Path],
) -> dict:
    validate_revision_binding(protected, freeze, paths["freeze"])
    expected_hashes = {
        "benchmark": protected["benchmark_manifest_sha256"],
        "manifest": freeze["manifest_sha256"],
        "references": freeze["references_sha256"],
    }
    for key, expected in expected_hashes.items():
        if api_contract.file_sha(paths[key]) != expected:
            raise ValueError("Changed protected-review input: " + key)
    if Path(freeze["references_path"]).resolve() != paths["references"].resolve():
        raise ValueError("Reference freeze points to a different artifact")
    for path, expected in protected["reference_files_sha256"].items():
        if freeze["review_files"].get(path) != expected or api_contract.file_sha(path) != expected:
            raise ValueError("Original reviewed references changed after slice selection")
    if not protected["reference_files_sha256"]:
        raise ValueError("Protected selection must bind original reviewed references")
    selected_at, frozen_at = timestamp(protected["created_at"]), timestamp(freeze["frozen_at"])
    by_id = {row["case_id"]: row for row in manifest["cases"]}
    benchmark_ids = {identity for identity, row in by_id.items() if row["benchmark_partition"] in BENCHMARK}
    if len(by_id) != 4000 or len(manifest["cases"]) != 4000 or len(benchmark_ids) != 500:
        raise ValueError("Expected the complete 4,000 corpus and 500-case benchmark")
    if manifest["inputs"]["benchmark_manifest_sha256"] != protected["benchmark_manifest_sha256"]:
        raise ValueError("Final corpus does not bind the prospective benchmark manifest")
    for rows in (benchmark["cases"], references):
        if len(rows) != 500 or {row["case_id"] for row in rows} != benchmark_ids:
            raise ValueError("Benchmark/reference coverage mismatch")
        for row in rows:
            if row["view_sha256"] != by_id[row["case_id"]]["view_sha256"]:
                raise ValueError("Benchmark/reference view mismatch")
    if len(freeze["case_ids"]) != 500 or set(freeze["case_ids"]) != benchmark_ids:
        raise ValueError("Reference freeze identities mismatch")
    ref_by_id = {row["case_id"]: row for row in references}
    for reference in references:
        reference_review.validate_assessment(reference)
    if len(protected["style_cases"]) != 12 or len({row["case_id"] for row in protected["style_cases"]}) != 12:
        raise ValueError("Expected exactly 12 prospectively selected style cases")
    for row in protected["style_cases"] + protected["scope_cases"]:
        if row["case_id"] not in benchmark_ids or row["view_sha256"] != by_id[row["case_id"]]["view_sha256"]:
            raise ValueError("Protected membership view mismatch")
    scope_ids = {row["case_id"] for row in protected["scope_cases"]}
    if (
        len(scope_ids) != len(protected["scope_cases"])
        or scope_ids != {identity for identity, row in ref_by_id.items() if row["eligibility"] != "eligible"}
        or any(
            row["reference_eligibility"] != ref_by_id[row["case_id"]]["eligibility"]
            for row in protected["scope_cases"]
        )
    ):
        raise ValueError("Prospective excluded/uncertain membership differs from frozen references")
    expected_slots = set(analyze.slots(manifest["cases"]))
    expected_slots |= {(identity, "reference", "reference") for identity in benchmark_ids}
    expected_slots |= {
        (identity, "luna_deepseek_mean", arm)
        for identity in by_id
        for arm in (
            ("baseline", "own_stats", "swapped_stats") if identity in benchmark_ids else ("baseline",)
        )
    }
    lookup = {(row["case_id"], row["model"], row["arm"]): row for row in scores}
    if len(lookup) != len(scores) or set(lookup) != expected_slots:
        raise ValueError("Scores export has missing, duplicate or unexpected identity slots")
    unverified_timing = []
    for row in scores:
        identity = row["case_id"]
        if row["view_sha256"] != by_id[identity]["view_sha256"]:
            raise ValueError("Score export view hash mismatch")
        if row.get("image_path") != by_id[identity].get("image_path"):
            raise ValueError("Score export review-image path mismatch")
        if (
            row["model"] in MODELS
            and row.get("scope_floor_applied") is not None
            and type(row["scope_floor_applied"]) is not bool
        ):
            raise ValueError("Invalid declared-scope flag")
        for task in TASKS:
            analyze.valid_score(row.get(task))
            if type(row.get(task + "_valid")) is not bool:
                raise ValueError("Missing score-validity flag")
        if identity in benchmark_ids and row["model"] not in {"reference", "luna_deepseek_mean"}:
            if row.get("request_started_at"):
                started = timestamp(row["request_started_at"])
                if started < selected_at or started < frozen_at:
                    raise ValueError("Benchmark request preceded protected selection or reference freeze")
            elif row.get("attempt_count", 0):
                unverified_timing.append(
                    {
                        "case_id": identity,
                        "model": row["model"],
                        "arm": row["arm"],
                        "reason": "attempt exists but dispatch timestamp unavailable",
                    }
                )
    for identity, reference in ref_by_id.items():
        exported = lookup[(identity, "reference", "reference")]
        if exported.get("reference_eligibility") != reference["eligibility"]:
            raise ValueError("Reference eligibility differs in score export")
        for task in TASKS:
            if exported.get(task) != reference_score(reference, task):
                raise ValueError("Reference score differs in score export")
        for arm in ("baseline", *ARMS):
            components = [lookup[(identity, model, arm)] for model in ("luna", "deepseek")]
            mean = lookup[(identity, "luna_deepseek_mean", arm)]
            for task in TASKS:
                value = (
                    analyze.decimal_mean([row[task] for row in components])
                    if all(valid(row, task) for row in components)
                    else None
                )
                if mean.get(task) != value:
                    raise ValueError("Derived mean differs from its frozen component scores")
                if mean[task + "_valid"] != all(row[task + "_valid"] for row in components):
                    raise ValueError("Derived mean validity differs from its components")
            if mean.get("scope_floor_applied") != all(
                row.get("scope_floor_applied") is True for row in components
            ):
                raise ValueError("Derived mean floor flag differs from its components")
    return {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": api_contract.file_sha(path)}
            for name, path in paths.items()
        },
        "protected_selected_at": protected["created_at"],
        "references_frozen_at": freeze["frozen_at"],
        "dispatch_timing_unverified": unverified_timing,
        "timing_status": "incomplete_requires_journal_audit"
        if unverified_timing
        else "verified_for_recorded_dispatches",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, filename in (
        ("protected", "protected-slices.json"),
        ("benchmark", "benchmark-manifest.json"),
        ("manifest", "manifest.json"),
        ("references", "references.jsonl"),
        ("freeze", "references-freeze.json"),
        ("scores", "scores-long.jsonl"),
    ):
        parser.add_argument("--" + name, type=Path, default=HERE / filename)
    parser.add_argument("--output", type=Path, default=HERE / "protected-review")
    parser.add_argument("--rebind-revisions", type=Path)
    parser.add_argument("--rebind-output", type=Path, default=HERE / "protected-slices-v2.json")
    args = parser.parse_args()
    if args.rebind_revisions:
        rebound = rebind_protected(args.protected, args.rebind_revisions, args.freeze)
        api_runner.write(args.rebind_output, rebound, exclusive=True)
        print(json.dumps({"rebound_selection": str(args.rebind_output.resolve()), "binding": rebound["revision_binding"]}))
        return
    paths = {
        name: getattr(args, name)
        for name in ("protected", "benchmark", "manifest", "references", "freeze", "scores")
    }
    protected, benchmark, manifest, freeze = (
        api_runner.read(paths[name]) for name in ("protected", "benchmark", "manifest", "freeze")
    )
    references, scores = (analyze.read_lines(paths[name]) for name in ("references", "scores"))
    bindings = validate_bindings(protected, benchmark, manifest, references, freeze, scores, paths=paths)
    result = evaluate(protected, references, scores)
    result["summary"].update(
        created_at=api_runner.now(),
        bindings=bindings,
        helper_sha256=api_contract.file_sha(__file__),
        metric_helper_sha256=api_contract.file_sha(metrics.__file__),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("style-deltas.jsonl", result["style_rows"]),
        ("scope-comparisons.jsonl", result["scope_rows"]),
        ("review-queue.jsonl", result["queue"]),
    ):
        api_runner.jsonl(args.output / name, rows)
    api_runner.write(args.output / "summary.json", result["summary"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "style_comparisons": len(result["style_rows"]),
                "scope_comparisons": len(result["scope_rows"]),
                "queue_items": len(result["queue"]),
                "adjudication_status": "pending_visual_review",
            }
        )
    )


if __name__ == "__main__":
    main()
