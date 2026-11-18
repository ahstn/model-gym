"""Reparse all new outcomes and verify immutable history, joins and publication."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_attempts(row, case, model, plans, cases_by_plan, runner, digest):
    attempts = row["attempt_provenance"]
    require(len(attempts) == row["attempt_count"] > 0, "Attempt count differs from provenance")
    require(len({item["attempt_id"] for item in attempts}) == len(attempts), "Duplicate attempt ID")
    for item in attempts:
        path = Path(item["request_path"])
        require(digest(path) == item["request_record_sha256"], "Request receipt changed")
        request = runner.read(path)
        plan = plans.get(request["plan_id"])
        require(plan is not None and plan["model"] == model, "Attempt refers to an unknown model plan")
        require(request["model"] == model, "Request model mismatch")
        require(request["case_id"] == case["case_id"], "Attempt case mismatch")
        require(case["case_id"] in cases_by_plan[request["plan_id"]], "Attempt case absent from its plan")
        require(request["view_sha256"] == case["view_sha256"], "Attempt delivered view mismatch")
        require(request["profile_sha256"] == plan["profile"]["sha256"], "Attempt profile mismatch")
        require(request["request_sha256"] == item["request_sha256"], "Attempt payload hash mismatch")
        result_path = path.with_name("result.json")
        if result_path.exists():
            require(str(result_path) == item["raw_response_path"], "Attempt response path mismatch")
            require(digest(result_path) == item["raw_response_sha256"], "Attempt raw response changed")
            result = runner.read(result_path)
            require(result.get("latency_seconds") == item["latency_seconds"], "Attempt timing changed")
            require(result.get("price_usd") == item["price_usd"], "Attempt price changed")
        else:
            require(
                item["price_usd"] is None and item["raw_response_sha256"] is None,
                "Missing response cannot have a fabricated receipt",
            )
    require(row["source_request_plan_id"] == attempts[-1]["source_request_plan_id"], "Final plan mismatch")
    require(row["raw_response_path"] == attempts[-1]["raw_response_path"], "Final response mismatch")
    missing = sum(item["price_usd"] is None for item in attempts)
    subtotal = sum(
        (Decimal(item["price_usd"]) for item in attempts if item["price_usd"] is not None), Decimal(0)
    )
    require(row["unpriced_attempts"] == missing, "Unknown attempt charges changed")
    require(Decimal(row["total_attempt_known_price_usd"]) == subtotal, "Attempt subtotal changed")
    require(
        row["total_attempt_price_usd"] == (str(subtotal) if not missing else None), "Attempt total changed"
    )


def audit_run(data, old, phases):
    import api_runner as runner
    import corpus  # noqa: F401 -- establishes the frozen adapter import context
    import export
    from corpus import INITIAL, PACKAGE, digest, read_lines, write_json

    from visual_quality.ensemble import MODEL, score_ensemble
    from visual_quality.publication import verify_publication
    from visual_quality.scoring.phase_reporting import build_phase_summary, phase_summary_digest

    pins = json.loads((data / "initial-evidence-pins.json").read_text())
    for path, expected in pins.items():
        require(digest(path) == expected, f"Historical/publication evidence changed: {path}")
    manifest = json.loads((data / "manifest.json").read_text())
    cases = manifest["cases"]
    previous = json.loads((old / "combined-manifest.json").read_text())["cases"]
    combined = json.loads((data / "combined-manifest.json").read_text())["cases"]
    require(
        len(cases) == export.NEW_IMAGES and len(previous) + len(cases) == export.TOTAL_IMAGES,
        "Unexpected expansion size",
    )
    require(combined == previous + cases, "Combined manifest changed historical cases or order")
    for field in ("case_id", "view_sha256", "image_id"):
        require(len({case[field] for case in combined}) == export.TOTAL_IMAGES, f"Duplicate combined {field}")
    require(
        not ({case["family_id"] for case in previous} & {case["family_id"] for case in cases}),
        "Source family overlaps historical data",
    )
    labels = read_lines(data / "exports/new-labelled-images.jsonl")
    stat_rows = read_lines(data / "stats.jsonl")
    stats = {row["case_id"]: row for row in stat_rows}
    require(
        len(labels) == len(stat_rows) == len(stats) == export.NEW_IMAGES,
        "Missing/duplicate labels or statistics",
    )
    plans, selected = export.validate_completed_runs(data, manifest, phases, runner.validate_plan)
    cases_by_plan = {
        identity: {case["case_id"] for case in plan["cases"]} for identity, plan in plans.items()
    }
    manifest_hash = digest(data / "manifest.json")
    for plan in plans.values():
        require(plan["manifest_sha256"] == manifest_hash, "Plan manifest hash mismatch")
        require(plan["training_ready"] is False, "Unexpected training authorization")
        routes = plan["profile"]["routes"]["routes"]
        require(routes["luna"]["settings"]["reasoning"]["effort"] == "high", "Luna reasoning changed")
        require(routes["deepseek"]["settings"]["reasoning_effort"] == "high", "DeepSeek reasoning changed")
        for model, expected in {
            "luna": "openai/gpt-6-luna",
            "deepseek": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "liquid": "liquid/d1.1",
            "jev": "typesafe/jev-1.14",
        }.items():
            require(routes[model]["requested_model"] == expected, "Requested grader changed")
        if plan.get("recovery", {}).get("kind") == "empty_length_once_v1":
            for case in plan["cases"]:
                _, request_sha256 = runner.prepare(plan, case)
                require(
                    request_sha256 == plan["recovery"]["eligibility"][case["case_id"]]["request_sha256"],
                    "Recovery payload reconstructed from the exact view/profile differs from the original",
                )
    parsed, statuses = 0, Counter()
    for case, label in zip(cases, labels, strict=True):
        identity = case["case_id"]
        require(
            label["case_id"] == identity and label["view_sha256"] == case["view_sha256"],
            "Label join mismatch",
        )
        require(digest(case["image_path"]) == case["view_sha256"], "Delivered image bytes changed")
        require(stats[identity]["view_sha256"] == case["view_sha256"], "Statistics view mismatch")
        require(label["image_stats"] == stats[identity]["image_stats"], "Statistics join changed")
        metadata = label["image_stats"]
        require(metadata["histogram_entropy"] == metadata["luma_entropy"], "Entropy alias changed")
        require(
            metadata["used_in_inference"] is False and metadata["training_features_enabled"] is False,
            "Statistics enabled as inference/training features",
        )
        require(
            label["training_ready"] is False and not any(label["task_masks"].values()),
            "Training masks must stay disabled",
        )
        group = label["assessments"]
        require(set(group) == {*export.MODELS, MODEL}, "Unexpected new grader/reference")
        for model in export.MODELS:
            row = group[model]
            require(
                row["view_sha256"] == case["view_sha256"] and row["case_id"] == identity,
                "Assessment join mismatch",
            )
            audit_attempts(row, case, model, plans, cases_by_plan, runner, digest)
            plan = plans[row["source_request_plan_id"]]
            require(row["profile_sha256"] == plan["profile"]["sha256"], "Assessment profile mismatch")
            path = Path(row["raw_response_path"])
            require(digest(path) == row["raw_response_sha256"], "Final raw response changed")
            result = runner.read(path)
            value = runner.parsed_result(result, plan["profile"]["routes"]["routes"][model])
            require(value == result["parsed"], "Raw outcome no longer parses identically")
            for field in ("status", "technical_score", "aesthetic_score"):
                require(row[field] == value[field], f"Assessment {field} differs from raw response")
            require(
                row["latency_seconds"] == result["latency_seconds"] and row["latency_seconds"] > 0,
                "Missing or changed measured latency",
            )
            require(row["price_usd"] == value["price_usd"], "Final response price differs")
            if model in ("luna", "deepseek"):
                require(row["requested_reasoning"] == "high", "Requested reasoning changed")
            require(export.finalized_observation(row, plans), "Unfinished observation")
            statuses[model + ":" + row["status"]] += 1
            parsed += 1
        require(
            group[MODEL]
            == score_ensemble(
                {model: group[model] for model in ("luna", "deepseek", "liquid")}, case_id=identity
            ),
            "Ensemble differs from canonical weights",
        )
    for name in ("labelled-images.jsonl", "scores-long.jsonl"):
        original = old / "exports" / name
        require(
            export.prefix_digest(data / "exports" / name, original.stat().st_size) == digest(original),
            "Historical canonical bytes changed",
        )
    require(
        sum(1 for _ in export.records(data / "exports/labelled-images.jsonl")) == export.TOTAL_IMAGES,
        "Wrong combined labelled image count",
    )
    actual = export.coverage(export.records(data / "exports/scores-long.jsonl"))
    summary = json.loads((data / "exports/summary.json").read_text())
    require(
        actual == summary["coverage"] and sum(row["expected"] for row in actual.values()) == export.LONG_ROWS,
        "Combined outcome coverage differs",
    )
    require(actual["reference"]["expected"] == export.REFERENCE_IMAGES, "Reference count changed")
    require(
        actual["qrealign_lite"]["expected"] == export.HISTORICAL_LITE_IMAGES, "Historical Lite count changed"
    )
    require(summary["current_plans"] == selected, "Final summary uses stale plans")
    observations = {model: [label["assessments"][model] for label in labels] for model in export.MODELS}
    phase_summary = build_phase_summary(
        data / "results/api", plans, observations, summary["accounting"], models=export.MODELS
    )
    require(phase_summary == summary["phase_summary"], "Baseline/recovery summary differs from receipts")
    previous_summary = json.loads((old / "exports/summary.json").read_text())
    require(
        summary["historical_cost_per_model"] == previous_summary["combined_cost_per_model"],
        "Prior two batches must be counted exactly once",
    )
    require(
        summary["combined_cost_per_model"]
        == export.combine_costs(
            previous_summary["combined_cost_per_model"],
            summary["accounting"],
            summary["compatibility_probe_accounting"],
        ),
        "Combined API ledger differs",
    )
    publication = PACKAGE / "labeling/records"
    verify_publication(publication)
    published = json.loads((publication / "publication-case-ids.json").read_text())
    initial = json.loads((INITIAL / "manifest.json").read_text())["cases"]
    expected_publication = sorted(
        (case for case in initial if case.get("benchmark_order") is not None),
        key=lambda case: case["benchmark_order"],
    )[: export.PUBLICATION_IMAGES]
    require(published == [case["case_id"] for case in expected_publication], "First-1,000 allowlist changed")
    result = {
        "passed": True,
        "new_images": export.NEW_IMAGES,
        "total_images": export.TOTAL_IMAGES,
        "raw_observations_reparsed": parsed,
        "new_statuses": dict(statuses),
        "image_statistics_joins": len(stats),
        "initial_records_unchanged": len(previous),
        "new_qrealign_calls": 0,
        "reference_images": export.REFERENCE_IMAGES,
        "publication_images": len(published),
        "training_ready": False,
        "current_plans": selected,
        "exports_sha256": digest(data / "exports/labelled-images.jsonl"),
        "phase_summary_sha256": phase_summary_digest(phase_summary),
    }
    write_json(data / "audit.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    from corpus import DATA, OLD
    from score import MODELS, PHASE

    audit_run(DATA, OLD, dict.fromkeys(MODELS, PHASE))
