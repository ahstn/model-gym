"""Bind visual adjudications to frozen scores and render a separate review report.

This companion does not change primary labels, score normalization, numerical
tests or the prospectively selected protected panel. The visual verdicts are
supplied separately by an assistant who has opened the exact delivered images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import analyze
import api_contract
import api_runner
import protected_review

HERE = Path(__file__).resolve().parent
ADJUDICATION = HERE / "protected-adjudication"
VERDICTS = {"visibly_justified", "regression", "ambiguous", "not_assessable"}


def canonical_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def key(row: dict) -> str:
    return "/".join(row[field] for field in ("case_id", "model", "arm"))


def frozen_context(scores_path: Path) -> dict:
    paths = {
        "protected": HERE / "protected-slices-v2.json",
        "benchmark": HERE / "benchmark-manifest-v2.json",
        "manifest": HERE / "manifest.json",
        "references": HERE / "references.jsonl",
        "freeze": HERE / "references-freeze.json",
        "scores": scores_path,
    }
    objects = {
        name: analyze.read_lines(path) if name in {"references", "scores"} else api_runner.read(path)
        for name, path in paths.items()
    }
    bindings = protected_review.validate_bindings(
        objects["protected"],
        objects["benchmark"],
        objects["manifest"],
        objects["references"],
        objects["freeze"],
        objects["scores"],
        paths=paths,
    )
    result = protected_review.evaluate(objects["protected"], objects["references"], objects["scores"])
    return {
        "bindings": bindings,
        "references": {row["case_id"]: row for row in objects["references"]},
        "scores": {(row["case_id"], row["model"], row["arm"]): row for row in objects["scores"]},
        "queue": {key(row): row for row in result["queue"]},
        "summary": result["summary"],
    }


def verify_receipt(row: dict) -> None:
    path, expected = row.get("raw_response_path"), row.get("raw_response_sha256")
    if bool(path) != bool(expected):
        raise ValueError("Incomplete raw-response binding")
    if path and api_contract.file_sha(path) != expected:
        raise ValueError("Changed raw-response receipt")


def pair_bindings(queue_row: dict, context: dict) -> dict:
    identity, model, arm = (queue_row[field] for field in ("case_id", "model", "arm"))
    reference = context["references"][identity]
    baseline = context["scores"][(identity, model, "baseline")]
    treatment = context["scores"][(identity, model, arm)]
    if api_contract.file_sha(queue_row["image_path"]) != queue_row["view_sha256"]:
        raise ValueError("Changed image supplied to adjudicator")
    if api_contract.file_sha(reference["review_timing_path"]) != reference["review_timing_sha256"]:
        raise ValueError("Changed primary reference timing record")
    for row in (baseline, treatment):
        verify_receipt(row)
    components = {}
    if model == "luna_deepseek_mean":
        for component in ("luna", "deepseek"):
            components[component] = {}
            for current_arm in ("baseline", arm):
                row = context["scores"][(identity, component, current_arm)]
                verify_receipt(row)
                components[component][current_arm] = {
                    "record_sha256": canonical_sha(row),
                    "attempt_id": row.get("attempt_id"),
                    "raw_response_path": row.get("raw_response_path"),
                    "raw_response_sha256": row.get("raw_response_sha256"),
                }
    return {
        "case_id": identity,
        "model": model,
        "arm": arm,
        "selection_reasons": queue_row["reasons"],
        "image_path": queue_row["image_path"],
        "view_sha256": queue_row["view_sha256"],
        "reference_record_sha256": canonical_sha(reference),
        "reference_file_sha256": context["bindings"]["inputs"]["references"]["sha256"],
        "reference_timing_sha256": reference["review_timing_sha256"],
        "baseline_attempt_id": baseline.get("attempt_id"),
        "treatment_attempt_id": treatment.get("attempt_id"),
        "baseline_raw_response_sha256": baseline.get("raw_response_sha256"),
        "treatment_raw_response_sha256": treatment.get("raw_response_sha256"),
        "baseline_record_sha256": canonical_sha(baseline),
        "treatment_record_sha256": canonical_sha(treatment),
        "component_bindings": components,
    }


def prepare(scores_path: Path, output_path: Path) -> dict:
    context = frozen_context(scores_path)
    queue = []
    for identity, row in sorted(context["queue"].items()):
        if row["model"] == "luna":
            continue
        case, model, arm = (row[field] for field in ("case_id", "model", "arm"))
        queue.append(
            {
                "key": identity,
                "bindings": pair_bindings(row, context),
                "reference": context["references"][case],
                "baseline": context["scores"][(case, model, "baseline")],
                "treatment": context["scores"][(case, model, arm)],
                "scope_comparison": row["scope_comparison"],
            }
        )
    result = {
        "created_at": api_runner.now(),
        "source_scores": {"path": str(scores_path.resolve()), "sha256": api_contract.file_sha(scores_path)},
        "frozen_bindings": context["bindings"],
        "queue": queue,
        "status": "Awaiting explicit visual adjudication; no numeric labels changed",
    }
    api_runner.write(output_path, result, exclusive=True)
    return result


def validate_review(review: dict, review_input: dict, context: dict, *, luna: bool) -> dict:
    expected = {
        identity: row for identity, row in context["queue"].items() if (row["model"] == "luna") == luna
    }
    comparisons = review["comparisons"]
    frozen = context["bindings"]["inputs"]
    if (
        review["reference_freeze"]["sha256"] != frozen["freeze"]["sha256"]
        or review["reference_freeze"]["references_sha256"] != frozen["references"]["sha256"]
    ):
        raise ValueError("Visual review is bound to a different reference freeze")
    if set(comparisons) != set(expected):
        raise ValueError("Visual review does not cover its exact final comparison queue")
    input_path = Path(review["input"]["path"])
    if (
        api_contract.file_sha(input_path) != review["input"]["sha256"]
        or api_runner.read(input_path) != review_input
    ):
        raise ValueError("Changed adjudication input")
    input_rows = {key(row) if luna else row["key"]: row for row in review_input["queue"]}
    if set(input_rows) != set(expected):
        raise ValueError("Reviewed input queue differs from final protected queue")
    images = review["images"]
    if set(images) != {row["case_id"] for row in expected.values()}:
        raise ValueError("Visual image coverage mismatch")
    for identity, row in expected.items():
        record, source = comparisons[identity], input_rows[identity]
        binding = pair_bindings(row, context)
        for field, value in binding.items():
            if field == "component_bindings" and luna:
                continue
            if record.get(field) != value:
                raise ValueError("Adjudication binding differs from final records: " + identity + "/" + field)
        if record.get("input_file_sha256") != review["input"]["sha256"]:
            raise ValueError("Comparison is not bound to exact visual-review input")
        if luna:
            reviewed_scores = (source["scores"]["baseline"], source["scores"][row["arm"]])
        else:
            reviewed_scores = (source["baseline"], source["treatment"])
            if source["bindings"] != binding:
                raise ValueError("Prepared comparison bindings changed")
        for phase, scored in zip(("baseline", "treatment"), reviewed_scores, strict=True):
            if canonical_sha(scored) != binding[phase + "_record_sha256"]:
                raise ValueError("Reviewed score differs from final exported score")
        image = images[row["case_id"]]
        if (
            image.get("view_sha256") != row["view_sha256"]
            or image.get("opened_exact_image") is not True
            or image.get("primary_reference_changed") is not False
        ):
            raise ValueError("Missing explicit exact-image inspection record")
        if record.get("verdict") not in VERDICTS or not record.get("visible_evidence"):
            raise ValueError("Missing visual verdict or evidence")
        if (
            record.get("primary_reference_changed") is not False
            or record.get("model_scores_changed") is not False
        ):
            raise ValueError("Adjudication must preserve numeric labels")
        if set(record.get("tasks", {})) != {"technical_quality", "aesthetics"}:
            raise ValueError("Each adjudication needs both task opinions")
        for task_name, task in record["tasks"].items():
            if task.get("verdict") not in VERDICTS | {"unchanged"} or not task.get("evidence"):
                raise ValueError("Missing task verdict/evidence")
            score_field = "technical_score" if task_name == "technical_quality" else "aesthetic_score"
            reference_task = context["references"][row["case_id"]][task_name]
            reference_score = reference_task["score"] if reference_task["status"] == "ok" else None
            if (
                task.get("baseline_score") != reviewed_scores[0].get(score_field)
                or task.get("treatment_score") != reviewed_scores[1].get(score_field)
                or task.get("frozen_reference_score") != reference_score
            ):
                raise ValueError("Visual narrative quotes a different score")
            if task["verdict"] == "unchanged" and task["baseline_score"] != task["treatment_score"]:
                raise ValueError("Unchanged task verdict has different numeric scores")
        if (
            context["references"][row["case_id"]]["eligibility"] == "uncertain"
            and record.get("reference_eligibility") != "uncertain"
        ):
            raise ValueError("Uncertain primary eligibility was rewritten")
    return comparisons


def report_text(comparisons: dict, context: dict) -> str:
    names = {"luna": "Luna", "deepseek": "DeepSeek", "luna_deepseek_mean": "Luna + DeepSeek mean"}
    arms = {"own_stats": "Own stats", "swapped_stats": "Swapped stats"}
    groups = defaultdict(Counter)
    for row in comparisons.values():
        groups[(row["model"], row["arm"])][row["verdict"]] += 1
    rows = [
        "| Model | Arm | Visibly justified | Regression | Ambiguous | Not assessable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (model, arm), counts in sorted(groups.items()):
        rows.append(
            f"| {names[model]} | {arms[arm]} | "
            + " | ".join(
                str(counts[verdict])
                for verdict in ("visibly_justified", "regression", "ambiguous", "not_assessable")
            )
            + " |"
        )
    regressions = ["| Image | Model | Arm | Visual diagnosis |", "|---|---|---|---|"]
    for identity, row in sorted(comparisons.items()):
        if row["verdict"] == "regression":
            evidence = (
                row.get("regression_summary")
                or " ".join(
                    task["evidence"] for task in row["tasks"].values() if task["verdict"] == "regression"
                )
                or row["visible_evidence"]
            )
            regressions.append(
                f"| {row['case_id']} | {names[row['model']]} | {arms[row['arm']]} | {evidence.replace('|', '/').replace(chr(10), ' ')} |"
            )
    image_count = len({row["case_id"] for row in comparisons.values()})
    return (
        "\n\n".join(
            [
                "# Protected style and scope adjudication",
            f"Visual review is complete for all {len(comparisons)} queued comparisons across {image_count} exact delivered images. This includes all 72 model/arm comparisons on the 12 prospectively selected style cases, plus newly flagged scope changes and descriptive uncertain-scope flips. The supplied score export, raw receipts, image bytes, reference freeze and reviewed input records were checked before joining the judgments. Primary references and model scores are unchanged.",
                "These are nonblind diagnostic assistant opinions after opening the exact images. A visibly justified change has an image-based rationale; a regression has a visible quality or scope problem; ambiguous differences cannot be resolved confidently. Small score changes often remain ambiguous. The review neither provides human ground truth nor proves that a particular statistic caused a response difference. Derived-mean comparisons share their component responses and are not independent replications.",
                "\n".join(rows),
                "The following comparisons have visually supported regressions. Both arms remain in the exports; no example or label was removed because it disagreed with the reference.",
                "\n".join(regressions),
                "Reference-uncertain eligibility remains uncertain and its floor flips are descriptive, not confirmed scope errors. Review completion is separate from the numerical advancement tests and does not approve any labels for training. The numerical analysis artifact deliberately leaves its own protected-review status unassessed; this supplemental adjudication supplies the completed review.",
            "The derived mean declares a scope floor only when both component models declare it. A new floor in just one component can still lower the mean without triggering a new declared mean-floor error. Component scope mistakes and their numerical effect therefore remain relevant; the absence of a new mean-floor flag does not establish scope safety.",
            ]
        )
        + "\n"
    )


def assemble(scores_path: Path, additional_review_path: Path, output: Path) -> dict:
    context = frozen_context(scores_path)
    comparisons, inputs = {}, {}
    for path, luna in ((ADJUDICATION / "luna-reviewed.json", True), (additional_review_path, False)):
        review = api_runner.read(path)
        review_input = api_runner.read(Path(review["input"]["path"]))
        validated = validate_review(review, review_input, context, luna=luna)
        if set(comparisons) & set(validated):
            raise ValueError("Duplicate adjudication across review files")
        comparisons.update(validated)
        inputs[str(path.resolve())] = api_contract.file_sha(path)
    if set(comparisons) != set(context["queue"]):
        raise ValueError("Combined adjudication does not cover the final queue")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(report_text(comparisons, context))
    validation = {
        "created_at": api_runner.now(),
        "status": "complete_visual_review",
        "bindings": context["bindings"],
        "review_files_sha256": inputs,
        "helper_sha256": api_contract.file_sha(__file__),
        "comparison_count": len(comparisons),
        "image_count": len({row["case_id"] for row in comparisons.values()}),
        "verdict_counts": dict(Counter(row["verdict"] for row in comparisons.values())),
        "exact_final_queue_coverage": True,
        "labels_changed": False,
        "training_approval": False,
        "numeric_analysis_status_not_overwritten": True,
        "report_sha256": api_contract.file_sha(output / "report.md"),
    }
    api_runner.write(output / "adjudication-validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "assemble"))
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--input-output", type=Path, default=ADJUDICATION / "deepseek-mean-input.json")
    parser.add_argument("--review", type=Path, default=ADJUDICATION / "deepseek-mean-reviewed.json")
    parser.add_argument("--output", type=Path, default=HERE / "protected-review")
    args = parser.parse_args()
    result = (
        prepare(args.scores, args.input_output)
        if args.action == "prepare"
        else assemble(args.scores, args.review, args.output)
    )
    print(
        json.dumps(
            {
                "action": args.action,
                "comparisons": len(result["queue"])
                if args.action == "prepare"
                else result["comparison_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
