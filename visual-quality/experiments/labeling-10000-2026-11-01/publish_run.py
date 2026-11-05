"""Prepare full local exports and publish only the first 1,000 benchmark images.

The default action writes ignored local artifacts. ``--publish`` is an explicit
operator action after coverage review, never a side effect of scoring. Provider
observations and reference ledgers remain immutable inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from visual_quality.ensemble import MODEL, configuration, score_ensemble, write_observations
from visual_quality.publication import jsonlines, sha, verify_publication

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")
MODELS = ("luna", "deepseek", "liquid", "jev", "qrealign_lite")
CONTEXT = (
    "case_id",
    "family_id",
    "view_sha256",
    "image_path",
    "source",
    "stratum",
    "benchmark_partition",
    "benchmark_order",
    "operator",
    "severity",
    "content_cluster",
)
STATS = ("sharpness_peak", "luma_entropy", "histogram_entropy", "focus_area", "channel_clip", "colourfulness")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value, *, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid finite numeric measurement")
    if maximum is not None and value > maximum:
        raise ValueError("Measurement exceeds allowed range")
    return value


def validate_manifest(manifest, *, expected_images=10000, expected_references=1500):
    cases = manifest["cases"]
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != expected_images or len(cases) != expected_images:
        raise ValueError("Manifest must contain the expected unique image count")
    if len({case["view_sha256"] for case in cases}) != len(cases):
        raise ValueError("Duplicate delivered views in manifest")
    benchmark = [case for case in cases if case.get("benchmark_order") is not None]
    if len(benchmark) != expected_references:
        raise ValueError("Reference benchmark size differs")
    if any(type(case["benchmark_order"]) is not int for case in benchmark):
        raise ValueError("Benchmark order must be an explicit integer")
    benchmark.sort(key=lambda case: case["benchmark_order"])
    if [case["benchmark_order"] for case in benchmark] != list(range(1, expected_references + 1)):
        raise ValueError("Benchmark order must be unique and contiguous")
    families = {case["family_id"] for case in benchmark}
    benchmark_ids = {case["case_id"] for case in benchmark}
    if len(families) != len(benchmark) or any(
        c["family_id"] in families for c in cases if c["case_id"] not in benchmark_ids
    ):
        raise ValueError("Benchmark family leaks across images or into rehearsal")
    return cases, benchmark


def index_records(records, cases, name):
    result = {}
    for record in records:
        identity = record["case_id"]
        if identity in result or identity not in cases:
            raise ValueError(f"Duplicate or unknown {name} image: {identity}")
        if record["view_sha256"] != cases[identity]["view_sha256"]:
            raise ValueError(f"Changed delivered view in {name}: {identity}")
        if record.get("family_id", cases[identity]["family_id"]) != cases[identity]["family_id"]:
            raise ValueError(f"Changed family in {name}: {identity}")
        result[identity] = record
    return result


def base_row(case, model):
    return {
        **{field: case.get(field) for field in CONTEXT},
        "model": model,
        "arm": "reference" if model == "reference" else "baseline",
        "status": "missing",
        "technical_score": None,
        "aesthetic_score": None,
        "technical_score_valid": False,
        "aesthetic_score_valid": False,
        "latency_seconds": None,
        "latency_kind": "no_request_recorded",
        "review_elapsed_seconds": None,
        "price_usd": "0.00" if model in {"reference", "qrealign_lite"} else None,
        "price_basis": "user_assigned_zero"
        if model in {"reference", "qrealign_lite"}
        else "unknown_or_no_request",
        "attempt_count": 0,
        "scope_floor_applied": False,
        "training_ready": False,
    }


def normalized_model(case, model, observation):
    row = base_row(case, model)
    if observation is not None:
        if observation.get("model") != model or observation.get("arm", "baseline") != "baseline":
            raise ValueError("Unexpected model or treatment arm")
        if "image_stats" in observation or "image_stats_full" in observation:
            raise ValueError("Image statistics must remain outside inference observations")
        row.update(observation)
        row.update({field: case.get(field) for field in CONTEXT})
    row["training_ready"] = False
    if model == "qrealign_lite":
        if observation is not None:
            row["attempt_count"] = observation.get("attempt_count", 1)
        if observation is not None and Decimal(str(observation.get("price_usd"))) != 0:
            raise ValueError("Lite observation must retain the zero-price reporting rule")
        row["price_usd"], row["price_basis"] = "0.00", "user_assigned_zero; infrastructure separate"
        for dimension, task in (("technical", "quality"), ("aesthetic", "aesthetics")):
            if "task_valid" in row:
                row[f"{dimension}_score_valid"] = row["task_valid"].get(task, False)
        if observation is not None and "latency_kind" not in observation:
            row["latency_kind"] = (
                "http_request_through_complete_response"
                if row.get("http_status") == 200
                else "incomplete_or_unmeasured_http_attempt"
            )
        raw_path = row.get("raw_response_path")
        if raw_path and not row.get("raw_response_sha256"):
            if Path(raw_path).is_file():
                row["raw_response_sha256"] = sha(Path(raw_path))
            elif row["status"] in {"valid", "partial", "invalid"}:
                raise ValueError("Completed Lite observation is missing its raw response")
            else:
                row["raw_response_missing"] = True
    for dimension in ("technical", "aesthetic"):
        field = f"{dimension}_score"
        value = row.get(field)
        if value is not None:
            number(value, minimum=1, maximum=5)
            if row.get(field + "_valid") is not True:
                raise ValueError("A numeric outcome lacks a valid assessment flag")
        row[field + "_display"] = f"{value:.2f}" if value is not None else None
    if row["latency_seconds"] is not None:
        number(row["latency_seconds"])
    if row["price_usd"] is not None:
        price = Decimal(str(row["price_usd"]))
        if not price.is_finite() or price < 0:
            raise ValueError("Invalid monetary receipt")
        row["price_usd"] = str(price)
    return row


def normalized_reference(case, reference):
    row = base_row(case, "reference")
    row["latency_kind"] = "not_applicable_assistant_review"
    if reference is None:
        return row
    if reference.get("latency_seconds") is not None or Decimal(str(reference["price_usd"])) != 0:
        raise ValueError("Reference reporting contract mismatch")
    row.update(
        status="valid",
        reference_eligibility=reference["eligibility"],
        scope_floor_applied=reference["eligibility"] == "excluded",
        review_elapsed_seconds=reference.get("review_elapsed_seconds"),
        reference_provenance={
            key: value for key, value in reference.items() if key.startswith(("review_", "reuse"))
        },
        evidence={},
        limitations={},
    )
    if row["review_elapsed_seconds"] is not None:
        number(row["review_elapsed_seconds"])
    for dimension, task in (("technical", "technical_quality"), ("aesthetic", "aesthetics")):
        rating = reference[task]
        if rating["status"] not in {"ok", "unscorable"}:
            raise ValueError("Unknown reference assessment status")
        value = rating["score"]
        if rating["status"] == "ok":
            number(value, minimum=1, maximum=5)
        elif value is not None:
            raise ValueError("Unscorable reference must have a null score")
        if reference["eligibility"] == "excluded" and value != 1:
            raise ValueError("Excluded reference must receive the agreed floor")
        row[f"{dimension}_score"] = row[f"{dimension}_score_native"] = value
        row[f"{dimension}_score_valid"] = value is not None
        row[f"{dimension}_score_display"] = f"{value:.2f}" if value is not None else None
        row["evidence"][task], row["limitations"][task] = rating["evidence"], rating["limitations"]
    return row


def validate_reference_order(reference, group):
    """Check every request against the independent review and pinned packet freeze."""
    reviewed = reference.get("review_completed_at")
    if not reviewed:
        if reference.get("review_freeze_path"):
            raise ValueError("Frozen reference is missing its review completion time")
        return
    completed = datetime.fromisoformat(reviewed)
    if completed.tzinfo is None:
        raise ValueError("Reference review time must include a timezone")
    admission_time = completed
    freeze_path = reference.get("review_freeze_path")
    if freeze_path:
        path = Path(freeze_path)
        if sha(path) != reference.get("review_freeze_sha256"):
            raise ValueError("Pinned reference freeze changed")
        frozen = datetime.fromisoformat(json.loads(path.read_text())["frozen_at"])
        if frozen.tzinfo is None or frozen < completed:
            raise ValueError("Reference freeze must follow completed review and include a timezone")
        admission_time = frozen
    for model in MODELS:
        row = group[model]
        attempts = row.get("attempt_provenance") or [row]
        if (
            freeze_path
            and row.get("attempt_count", 0) > 0
            and any(not attempt.get("request_started_at") for attempt in attempts)
        ):
            raise ValueError("Benchmark attempt is missing its measured request start time")
        for attempt in [row, *attempts]:
            started = attempt.get("request_started_at")
            if not started:
                continue
            requested = datetime.fromisoformat(started)
            if requested.tzinfo is None or requested < admission_time:
                raise ValueError("Benchmark inference preceded independent reference review or freeze")


def api_attempt_provenance(request, result, path, root):
    """Keep attempt receipts and measured timings without embedding response bodies."""
    result_path = path.with_name("result.json")
    return {
        "attempt_id": str(path.parent.relative_to(root)),
        "source_request_plan_id": request.get("plan_id"),
        "request_path": str(path),
        "request_record_sha256": sha(path),
        "request_sha256": request.get("request_sha256"),
        "raw_response_path": str(result_path),
        "raw_response_sha256": sha(result_path) if result_path.exists() else None,
        "status": result.get("parsed", {}).get("status", "observation_not_finalized"),
        "http_status": result.get("http_status"),
        "transport_error": result.get("transport_error"),
        "request_started_at": result.get("request_started_at"),
        "response_received_at": result.get("response_received_at"),
        "latency_seconds": result.get("latency_seconds"),
        "price_usd": result.get("price_usd"),
    }


def load_api(root):
    observations, audit = {}, {}
    for model in MODELS[:-1]:
        folder = root / model
        rows = [json.loads(path.read_text()) for path in sorted((folder / "observations").glob("*.json"))]
        by_id = {row["case_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate API observation files")
        history = defaultdict(list)
        for path in sorted((folder / "attempts").glob("*/*/request.json")):
            request = json.loads(path.read_text())
            result_path = path.with_name("result.json")
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            history[request["case_id"]].append((request, result, path))
        known, unknown, count = Decimal(0), 0, 0
        for identity, attempts in history.items():
            prices = [
                Decimal(result["price_usd"]) if result.get("price_usd") is not None else None
                for _, result, _ in attempts
            ]
            subtotal = sum((p for p in prices if p is not None), Decimal(0))
            missing = sum(p is None for p in prices)
            known, unknown, count = known + subtotal, unknown + missing, count + len(attempts)
            row = by_id.get(identity)
            if row is None or row.get("attempt_count") != len(attempts):
                request = attempts[-1][0]
                row = {
                    **request,
                    "arm": "baseline",
                    "status": "observation_not_finalized",
                    "technical_score": None,
                    "aesthetic_score": None,
                }
                by_id[identity] = row
            row.update(
                attempt_count=len(attempts),
                total_attempt_known_price_usd=str(subtotal),
                total_attempt_price_usd=str(subtotal) if not missing else None,
                unpriced_attempts=missing,
                source_request_plan_id=attempts[-1][0].get("plan_id"),
                attempt_provenance=[
                    api_attempt_provenance(request, result, path, folder)
                    for request, result, path in attempts
                ],
            )
            if attempts[-1][1].get("transport_error"):
                row["latency_kind"] = "http_attempt_until_transport_failure"
        observations[model] = list(by_id.values())
        audit[model] = {
            "attempts": count,
            "known_price_usd": str(known),
            "unpriced_attempts": unknown,
            "total_price_usd": str(known) if not unknown else None,
        }
    return observations, audit


def merge_lite_attempts(records):
    """Join bounded transport continuations without selecting between valid scores."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["case_id"]].append(row)
    merged = []
    for identity, attempts in grouped.items():
        if len(attempts) == 1:
            merged.append(attempts[0])
            continue
        if any(row.get("reuse") for row in attempts):
            raise ValueError("A reused historical Lite observation was rescored")
        for field in ("view_sha256", "family_id", "checkpoint", "revision", "request_sha256"):
            values = {row.get(field) for row in attempts}
            if len(values) != 1 or None in values:
                raise ValueError(f"Lite continuation changed {field}: {identity}")
        if len({row["attempt_id"] for row in attempts}) != len(attempts):
            raise ValueError("Duplicate Lite attempt supplied to publication")
        assessments = [row for row in attempts if row["status"] in {"valid", "partial", "invalid"}]
        if len(assessments) > 1:
            raise ValueError("Multiple Lite assessments would require score selection")
        ordered = sorted(attempts, key=lambda row: row["request_started_at"])
        selected = dict(assessments[0] if assessments else ordered[-1])
        latencies = [row.get("latency_seconds") for row in attempts]
        selected.update(
            attempt_count=len(attempts),
            total_attempt_latency_seconds=sum(latencies) if all(v is not None for v in latencies) else None,
            attempt_provenance=[
                {
                    key: row.get(key)
                    for key in (
                        "attempt_id",
                        "run_id",
                        "status",
                        "http_status",
                        "request_started_at",
                        "response_received_at",
                        "latency_seconds",
                        "raw_response_path",
                        "price_usd",
                    )
                }
                for row in ordered
            ],
        )
        merged.append(selected)
    return merged


def build_exports(
    manifest, references, observations, statistics, *, expected_images=10000, expected_references=1500
):
    cases, benchmark = validate_manifest(
        manifest, expected_images=expected_images, expected_references=expected_references
    )
    case_map = {case["case_id"]: case for case in cases}
    refs = index_records(references, case_map, "reference")
    if set(refs) - {case["case_id"] for case in benchmark}:
        raise ValueError("Reference appears outside benchmark")
    stats = index_records(statistics, case_map, "statistics")
    maps = {model: index_records(observations.get(model, []), case_map, model) for model in MODELS}
    rows, labelled = [], []
    for case in cases:
        identity = case["case_id"]
        group = {model: normalized_model(case, model, maps[model].get(identity)) for model in MODELS}
        group[MODEL] = score_ensemble(
            {model: group[model] for model in ("luna", "deepseek", "liquid")}, case_id=identity
        )
        if case.get("benchmark_order") is not None:
            group["reference"] = normalized_reference(case, refs.get(identity))
            if identity in refs:
                validate_reference_order(refs[identity], group)
        rows.extend(group.values())
        metadata = stats.get(identity, {}).get("image_stats")
        if metadata is not None:
            if (
                metadata.get("used_in_inference") is not False
                or metadata.get("training_features_enabled") is not False
            ):
                raise ValueError("Statistics must be disabled for inference/training features")
            for field in STATS:
                number(metadata[field])
            if metadata["histogram_entropy"] != metadata["luma_entropy"]:
                raise ValueError("Histogram entropy alias differs from luma entropy")
        labelled.append(
            {
                **case,
                "image_stats": metadata,
                "statistics_status": "available" if metadata else "missing",
                "assessments": group,
                "training_ready": False,
                "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
            }
        )
    coverage = {}
    for model in (*MODELS, MODEL, "reference"):
        selected = [row for row in rows if row["model"] == model]
        coverage[model] = {
            "expected": len(selected),
            "status_counts": dict(Counter(r["status"] for r in selected)),
            "numeric_pairs": sum(
                all(r[f"{d}_score"] is not None for d in ("technical", "aesthetic")) for r in selected
            ),
        }
    summary = {
        "images": len(cases),
        "benchmark_images": len(benchmark),
        "statistics_images": len(stats),
        "reference_records": len(refs),
        "coverage": coverage,
        "ensemble": configuration(),
        "training_ready": False,
        "stats_in_inference": False,
    }
    return rows, labelled, summary


def write_publication(
    destination, manifest, references, rows, labelled, summary, *, limit=1000, incomplete_reason=None
):
    if not 1 <= limit <= 1000:
        raise ValueError("Publication limit must be at most 1,000 images")
    if (
        summary["statistics_images"] != summary["images"]
        or summary["coverage"]["reference"]["numeric_pairs"] != summary["benchmark_images"]
    ):
        raise ValueError("Final publication requires complete statistics and benchmark references")
    incomplete = {m: v for m, v in summary["coverage"].items() if v["numeric_pairs"] != v["expected"]}
    if incomplete and not incomplete_reason:
        raise ValueError("Incomplete model coverage requires a documented operator decision")
    ordered = sorted(
        (c for c in manifest["cases"] if c.get("benchmark_order") is not None),
        key=lambda c: c["benchmark_order"],
    )
    if len(ordered) < limit:
        raise ValueError("Not enough ordered benchmark images for publication")
    cases = ordered[:limit]
    ids = [case["case_id"] for case in cases]
    allowed = set(ids)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Publication destination must be empty; validate staging before replacing active records"
        )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "publication-case-ids.json", ids)
    write_json(
        destination / "manifest.json",
        {
            "schema_version": "labeling-publication-v1",
            "cases": cases,
            "counts": {
                "published": len(cases),
                "full_corpus": summary["images"],
                "full_benchmark": summary["benchmark_images"],
            },
            "publication_policy": "first explicit benchmark orders only; remaining labels stored outside git",
        },
    )
    write_observations(destination / "references.jsonl", [r for r in references if r["case_id"] in allowed])
    selected = [row for row in rows if row["case_id"] in allowed]
    write_observations(destination / "scores.jsonl", selected)
    write_observations(destination / "scores.csv", selected)
    write_observations(
        destination / "labelled-images.jsonl", [row for row in labelled if row["case_id"] in allowed]
    )
    write_json(
        destination / "provenance.json",
        {
            "schema_version": "labeling-publication-v1",
            "status": "pending training review",
            "records": {path.name: sha(path) for path in sorted(destination.iterdir()) if path.is_file()},
            "model_rows": dict(Counter(row["model"] for row in selected)),
            "ensemble": configuration(),
            "full_run_summary": summary,
            "incomplete_models_reason": incomplete_reason,
            "input_provenance": summary.get("input_provenance", {}),
        },
    )
    return verify_publication(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, default=DATA / "results/api")
    parser.add_argument("--lite-reused", type=Path)
    parser.add_argument("--lite-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DATA / "exports")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publication-destination", type=Path, default=DATA / "publication-staging")
    parser.add_argument("--allow-incomplete-models", metavar="REASON")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Full per-image exports must remain under ignored run data")
    manifest = json.loads(args.manifest.read_text())
    references, statistics = jsonlines(args.references), jsonlines(args.stats)
    observations, costs = load_api(args.api_root)
    lite = jsonlines(args.lite_reused) if args.lite_reused else []
    lite_paths = [path for folder in args.lite_run for path in sorted((folder / "attempts").glob("*.json"))]
    lite.extend(json.loads(path.read_text()) for path in lite_paths)
    observations["qrealign_lite"] = merge_lite_attempts(lite)
    rows, labelled, summary = build_exports(manifest, references, observations, statistics)
    summary["cost_per_model"] = {
        **costs,
        "qrealign_lite": {"price_usd": "0.00", "infrastructure_separate": True},
        "reference": {"price_usd": "0.00", "no_external_request": True},
    }
    summary["input_provenance"] = {
        str(path): sha(path) for path in [args.manifest, args.references, args.stats]
    }
    if args.lite_reused:
        summary["input_provenance"][str(args.lite_reused)] = sha(args.lite_reused)
    args.output.mkdir(parents=True, exist_ok=True)
    write_observations(args.output / "scores-long.jsonl", rows)
    write_observations(args.output / "scores-long.csv", rows)
    write_observations(args.output / "labelled-images.jsonl", labelled)
    write_json(args.output / "summary.json", summary)
    if args.publish:
        result = write_publication(
            args.publication_destination,
            manifest,
            references,
            rows,
            labelled,
            summary,
            incomplete_reason=args.allow_incomplete_models,
        )
        summary["publication_validation"] = result
        write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
