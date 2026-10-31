"""Post hoc scalar-only Haiku diagnostic; never repairs or replaces strict labels."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / HERE.name / "haiku" / "images"
EVIDENCE_ERROR = "Evidence must contain 1–30 words"


def import_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = import_local("sensitivity_report", HERE / "report.py")
wire = import_local("sensitivity_wire", HERE / "wire.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def strict_json(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def recover_scalars(observation):
    """Prove evidence length is the sole failure before reading its exact scalar."""
    errors = observation["validation_errors"]
    path = DATA / "attempts" / observation["attempt_id"] / "result.json"
    result = strict_json(path.read_text())
    if result["http_status"] != 200 or result.get("transport_error"):
        raise ValueError("Failed HTTP result cannot supply a diagnostic score")
    body = strict_json(result["body"])
    if body.get("error") or len(body["choices"]) != 1:
        raise ValueError("Invalid response envelope")
    choice = body["choices"][0]
    if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
        raise ValueError("Incomplete or refused response")
    if body["model"] != observation["provenance"]["model_returned"]:
        raise ValueError("Recorded and raw models differ")
    if body.get("provider") != observation["provenance"]["provider_returned"]:
        raise ValueError("Recorded and raw providers differ")
    text = choice["message"]["content"]
    parsed = wire.parse_joint(text)
    if parsed["errors"] != errors or parsed["wire_score_valid"] != observation["wire_score_valid"]:
        raise ValueError("Reparsed result differs from strict observation")
    raw = strict_json(text)
    validation_copy = copy.deepcopy(raw)
    recovered, unrecovered = [], []
    for task in errors:
        rating = raw[task]
        evidence = rating.get("evidence")
        if errors[task] != EVIDENCE_ERROR:
            unrecovered.append(
                {
                    "case_id": observation["case_id"],
                    "arm_id": observation["arm_id"],
                    "task": task,
                    "attempt_id": observation["attempt_id"],
                    "raw_score": rating.get("score"),
                    "raw_status": rating.get("status"),
                    "strict_status": "invalid",
                    "strict_validation_error": errors[task],
                    "raw_result_path": str(path),
                    "raw_result_sha256": digest(path),
                    "label_restored": False,
                }
            )
            continue
        if not isinstance(evidence, str) or len(evidence.split()) <= 30:
            raise ValueError("Recovery is limited to nonempty explanations exceeding 30 words")
        score = rating.get("score")
        if type(score) is not str or score not in wire.SCORES or rating.get("status") != "ok":
            raise ValueError("Recovery requires an exact permitted enum scalar and ok status")
        if observation[task] != {"score": None, "status": "invalid"}:
            raise ValueError("Expected the unchanged strict invalid task record")
        # This ephemeral copy checks constraints that the original parser reaches
        # after its length guard. It is never exported as a repaired observation.
        validation_copy[task]["evidence"] = "Diagnostic length check only."
        recovered.append(
            {
                "case_id": observation["case_id"],
                "arm_id": observation["arm_id"],
                "task": task,
                "attempt_id": observation["attempt_id"],
                "raw_score": score,
                "raw_evidence_word_count": len(evidence.split()),
                "strict_status": "invalid",
                "strict_validation_error": errors[task],
                "raw_result_path": str(path),
                "raw_result_sha256": digest(path),
                "label_restored": False,
            }
        )
    expected_remaining = {task: error for task, error in errors.items() if error != EVIDENCE_ERROR}
    if wire.parse_joint(json.dumps(validation_copy))["errors"] != expected_remaining:
        raise ValueError("Other schema or content constraints fail after removing the length violation")
    return recovered, unrecovered


def apply_diagnostic_scores(rows, recovered):
    indexed = {(r["arm_id"], r["case_id"], r["task"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate comparison row")
    for recovered_row in recovered:
        key = tuple(recovered_row[k] for k in ("arm_id", "case_id", "task"))
        row = indexed[key]
        if row["student_score"] is not None or row["student_status"] != "invalid":
            raise ValueError("Recovery would overwrite an existing valid scalar")
        value = float(Decimal(recovered_row["raw_score"]))
        low, high = row["reference_interval"]
        error = value - row["reference_score"]
        row.update(
            student_score=value,
            student_status="diagnostic_score_only",
            error=error,
            absolute_error=abs(error),
            interval_distance=max(low - value, 0, value - high),
            critical_contradiction=(value >= 4 and high <= 2) or (value <= 2 and low >= 4),
        )


def paired(rows, baseline):
    current = {row["case_id"]: row for row in rows if "error" in row}
    previous = {row["case_id"]: row for row in baseline if "error" in row}
    shared = sorted(current.keys() & previous.keys())
    result = report.paired_bootstrap([current[k] for k in shared], [previous[k] for k in shared])
    result["paired_metrics"] = {
        "arm": report.metrics([current[k] for k in shared]),
        "baseline": report.metrics([previous[k] for k in shared]),
    }
    return result


def main():
    source_paths = [
        HERE / "manifest.json",
        HERE / "references.jsonl",
        HERE / "references-freeze.json",
        HERE / "report.py",
        HERE / "wire.py",
        wire.BASE_PATH,
        DATA / "observations.jsonl",
        DATA / "report.json",
        DATA / "run.json",
        Path(__file__).resolve(),
    ]
    pins = {str(path): digest(path) for path in source_paths}
    manifest = strict_json((HERE / "manifest.json").read_text())
    freeze = strict_json((HERE / "references-freeze.json").read_text())
    if freeze["references_sha256"] != pins[str(HERE / "references.jsonl")]:
        raise ValueError("Frozen references changed")
    if freeze["manifest_sha256"] != pins[str(HERE / "manifest.json")]:
        raise ValueError("Frozen manifest changed")
    route_report = strict_json((DATA / "report.json").read_text())
    if route_report["status"] != "complete" or route_report["attempted"] != 300:
        raise ValueError("Haiku route must have completed all 300 image attempts")
    observations = report.read_rows(DATA / "observations.jsonl")
    references = report.read_rows(HERE / "references.jsonl")
    strict_rows = [
        row
        for row in report.comparison_rows(manifest["cases"], references, {"haiku": observations})
        if row["route"] == "haiku"
    ]
    if len(observations) != 300 or len(strict_rows) != 600:
        raise ValueError("Expected all 50 images across six arms and two dimensions")
    recovered, unrecovered = [], []
    for observation in observations:
        if observation["validation_errors"]:
            available, unavailable = recover_scalars(observation)
            recovered.extend(available)
            unrecovered.extend(unavailable)
    diagnostic_rows = copy.deepcopy(strict_rows)
    apply_diagnostic_scores(diagnostic_rows, recovered)
    results = {}
    for task in report.TASKS:
        baseline = [r for r in diagnostic_rows if r["task"] == task and r["arm_id"] == "baseline"]
        strict_baseline = [r for r in strict_rows if r["task"] == task and r["arm_id"] == "baseline"]
        results[task] = {}
        for arm in report.ARMS:
            current = [r for r in diagnostic_rows if r["task"] == task and r["arm_id"] == arm]
            strict_current = [r for r in strict_rows if r["task"] == task and r["arm_id"] == arm]
            if len(current) != 50:
                raise ValueError(f"Diagnostic plan must retain all 50 images: {task}/{arm}")
            eligible = [r for r in current if "error" in r]
            results[task][arm] = {
                "strict_label_validity": report.coverage(strict_current),
                "strict_valid_only_metrics": report.metrics([r for r in strict_current if "error" in r]),
                "strict_paired_baseline_comparison": paired(strict_current, strict_baseline),
                "recovered_scalars": sum(r["student_status"] == "diagnostic_score_only" for r in current),
                "score_only_coverage": {
                    "planned": len(current),
                    "available_scalars": len(eligible),
                    "remaining_invalid_case_ids": [r["case_id"] for r in current if "error" not in r],
                },
                "score_only_metrics_available": report.metrics(eligible),
                "score_only_paired_baseline_comparison": paired(current, baseline),
            }
    counts = Counter(item["task"] for item in recovered)
    output = {
        "schema_version": "post-hoc-haiku-score-only-sensitivity-v1",
        "analysis_status": "post_hoc_diagnostic_only",
        "route": "haiku",
        "human_gold": False,
        "label_status_unchanged": True,
        "eligible_for_training": False,
        "scope": "Exact enum scalar recovery only when evidence exceeds 30 words and every other validation rule passes",
        "interpretation": [
            "These scalar metrics diagnose visual score agreement separately from explanation-format reliability.",
            "Recovered scalar scores do not restore valid labels; all strict observations and collage values remain unchanged.",
            "Null scores paired with ok status cannot be recovered and remain excluded; paired comparisons use the shared available cases.",
            "Confidence intervals are descriptive post hoc sensitivity results, not additional confirmatory tests.",
            "A negative paired difference means lower error than baseline; both dimensions remain separate.",
            "Bootstrap resamples whole image families within source strata, using 10000 draws and seed 20261030, matching the main report.",
        ],
        "recovery_count_by_task": dict(counts),
        "recovered_tasks": recovered,
        "unrecovered_tasks": unrecovered,
        "unrecovered_count_by_reason": dict(Counter(row["strict_validation_error"] for row in unrecovered)),
        "source_files_sha256": pins,
        "results": results,
    }
    for path, expected in pins.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during analysis: {path}")
    for row in recovered + unrecovered:
        if digest(Path(row["raw_result_path"])) != row["raw_result_sha256"]:
            raise ValueError("Raw result changed during sensitivity analysis")
    destination = HERE / "score_only_sensitivity.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "recovered": dict(counts),
                "mae": {
                    task: {
                        arm: {
                            "n": row["score_only_metrics_available"]["n"],
                            "mae": row["score_only_metrics_available"]["mae"],
                        }
                        for arm, row in values.items()
                    }
                    for task, values in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
