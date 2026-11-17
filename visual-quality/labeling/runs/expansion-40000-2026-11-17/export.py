"""Append new assessments to the frozen 20K exports without rewriting history.

Run ``finalize.py`` to acquire all worker locks, export and audit together. Direct
execution of this module uses the same locked finalization path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

MODELS = ("luna", "deepseek", "liquid", "jev")
NEW_IMAGES = 20000
TOTAL_IMAGES = 40000
REFERENCE_IMAGES = 1500
PUBLICATION_IMAGES = 1000
HISTORICAL_LITE_IMAGES = 10000
LONG_ROWS = TOTAL_IMAGES * (len(MODELS) + 1) + HISTORICAL_LITE_IMAGES + REFERENCE_IMAGES


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def append_export(old_path, new_rows, target):
    """Preserve historical serialization, precision and provenance byte-for-byte."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if old_path.stat().st_size:
        with old_path.open("rb") as source:
            source.seek(-1, 2)
            if source.read(1) != b"\n":
                raise ValueError("Historical JSONL must end with a newline")
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as out:
        with old_path.open("rb") as source:
            shutil.copyfileobj(source, out)
        for row in new_rows:
            out.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
    temporary.replace(target)


def prefix_digest(path, length):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while length:
            chunk = stream.read(min(8 * 1024 * 1024, length))
            if not chunk:
                raise ValueError("Combined export is shorter than its historical prefix")
            hasher.update(chunk)
            length -= len(chunk)
    return hasher.hexdigest()


def combine_costs(*ledgers):
    """Each disjoint ledger appears once; an unknown receipt never becomes zero."""
    combined = {}
    for model in MODELS:
        entries = [ledger[model] for ledger in ledgers]
        for entry in entries:
            price = Decimal(entry["known_price_usd"])
            if (
                not price.is_finite()
                or price < 0
                or type(entry["unpriced_attempts"]) is not int
                or entry["unpriced_attempts"] < 0
            ):
                raise ValueError("Invalid monetary accounting")
        known = sum((Decimal(entry["known_price_usd"]) for entry in entries), Decimal(0))
        missing = sum(entry["unpriced_attempts"] for entry in entries)
        if not known.is_finite() or known < 0 or missing < 0:
            raise ValueError("Invalid monetary accounting")
        combined[model] = {
            "known_price_usd": str(known),
            "unpriced_attempts": missing,
            "total_price_usd": str(known) if not missing else None,
        }
    return combined


def coverage(rows):
    result = {}
    for row in rows:
        entry = result.setdefault(row["model"], {"expected": 0, "numeric_pairs": 0, "status_counts": {}})
        entry["expected"] += 1
        entry["numeric_pairs"] += int(
            all(row.get(d + "_score") is not None for d in ("technical", "aesthetic"))
        )
        entry["status_counts"][row["status"]] = entry["status_counts"].get(row["status"], 0) + 1
    return result


def finalized_observation(row, plans):
    plan = plans.get(row.get("source_request_plan_id"))
    if plan is None:
        return False
    recovery = plan.get("recovery", {})
    if recovery.get("kind") == "empty_length_once_v1":
        limit = recovery.get("attempt_limits", {}).get(row.get("case_id"))
        return (
            type(limit) is int
            and limit > 0
            and row.get("attempt_count") == limit
            and row["status"] in ("valid", "partial", "invalid", "failed", "retryable_failure")
        )
    if row["status"] in ("valid", "partial", "invalid", "failed"):
        return True
    return (
        row["status"] == "retryable_failure" and row.get("attempt_count", 0) >= plan["policy"]["max_attempts"]
    )


def verify_recovery_evidence(root, plan, baseline, expected):
    """Independently verify the complete frozen recovery allowlist and its evidence."""
    recovery = plan["recovery"]
    eligible = recovery.get("eligibility", {})
    limits = recovery["attempt_limits"]
    if set(eligible) != set(limits):
        raise ValueError("Recovery eligibility and attempt limits differ")
    if (
        recovery.get("baseline_phase") != baseline["phase"]
        or recovery.get("baseline_plan_id") != baseline["plan_id"]
        or plan["profile"] != baseline["profile"]
        or {case["case_id"]: case["view_sha256"] for case in baseline["cases"]} != expected
    ):
        raise ValueError("Recovery baseline/profile/full-manifest evidence differs")
    pins = plan.get("reference_pins", {})

    def pinned(path, required=None):
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if pins.get(str(path)) != actual or (required is not None and actual != required):
            raise ValueError(f"Missing or changed recovery evidence pin: {path}")
        return json.loads(path.read_text())

    baseline_path = root / "plans" / f"{baseline['phase']}.json"
    if pinned(baseline_path) != baseline:
        raise ValueError("Recovery baseline plan differs from its pinned file")
    state = pinned(root / f"{baseline['phase']}-status.json")
    if (
        state.get("status") != "complete"
        or state.get("plan_id") != baseline["plan_id"]
        or any(
            state.get(key) != len(expected) for key in ("planned_cases", "selected_cases", "completed_cases")
        )
        or sum(state.get("outcomes", {}).values()) != len(expected)
    ):
        raise ValueError("Recovery baseline was not fully complete")
    for identity in set(expected) - set(eligible):
        pinned(root / "observations" / f"{identity}.json")
    for identity, item in eligible.items():
        previous = item.get("previous_attempt_count")
        if type(previous) is not int or previous < 1 or limits[identity] != previous + 1:
            raise ValueError("Recovery must permit exactly one additional attempt")
        if not isinstance(item.get("request_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", item["request_sha256"]
        ):
            raise ValueError("Recovery requires a frozen original request-body hash")
        archive = root / "recovery-evidence" / plan["phase"] / f"{identity}.json"
        if item.get("baseline_observation_path") != str(archive):
            raise ValueError("Recovery baseline observation archive path differs")
        row = pinned(archive)
        if (
            row.get("case_id") != identity
            or row.get("view_sha256") != expected[identity]
            or row.get("model") != plan["model"]
            or row.get("plan_id") != baseline["plan_id"]
            or row.get("attempt_count") != previous
            or row.get("status") != "failed"
            or row.get("reason") != "Incomplete completion"
            or any(row.get(d + "_score") is not None for d in ("technical", "aesthetic"))
            or any(row.get("task_valid", {}).values())
        ):
            raise ValueError("Archived observation does not prove a missing baseline assessment")
        attempts = sorted((root / "attempts" / identity).glob("*/request.json"))
        if len(attempts) != previous + 1:
            raise ValueError("Recovery receipt count is not exactly one additional attempt")
        expected_paths = {
            str(path)
            for request in attempts[:previous]
            for path in (request, request.with_name("result.json"))
        }
        if set(item.get("source_pins", {})) != expected_paths:
            raise ValueError("Recovery source receipt pins are incomplete or unexpected")
        for number, request_path in enumerate(attempts, 1):
            if request_path.parent.name != f"{number:02d}":
                raise ValueError("Recovery attempt numbering changed")
            prior = number <= previous
            request = (
                pinned(request_path, item["source_pins"][str(request_path)])
                if prior
                else json.loads(request_path.read_text())
            )
            if any(
                request.get(key) != value
                for key, value in {
                    "plan_id": baseline["plan_id"] if prior else plan["plan_id"],
                    "model": plan["model"],
                    "case_id": identity,
                    "attempt": number,
                    "view_sha256": expected[identity],
                    "profile_sha256": plan["profile"]["sha256"],
                    "request_sha256": item.get("request_sha256"),
                }.items()
            ):
                raise ValueError("Recovery request differs from its frozen original payload/identity")
            result_path = request_path.with_name("result.json")
            if not result_path.exists():
                raise ValueError("Recovery request has no response receipt; reconciliation required")
            if prior:
                pinned(result_path, item["source_pins"][str(result_path)])
        original_result = attempts[previous - 1].with_name("result.json")
        if (
            row.get("raw_response_path") != str(original_result)
            or row.get("raw_response_sha256") != item["source_pins"][str(original_result)]
        ):
            raise ValueError("Archived observation points to different original response evidence")


def validate_completed_runs(data, manifest, phases, validate_plan):
    """Reject stale status files, partial runs, changed views and unselected plans."""
    expected = {case["case_id"]: case["view_sha256"] for case in manifest["cases"]}
    if len(expected) != len(manifest["cases"]) or set(phases) != set(MODELS):
        raise ValueError("Unique manifest IDs and exactly four active model phases required")
    all_plans = {}
    selected = {}
    with (data / "manifest.json").open("rb") as stream:
        manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    for model, phase in phases.items():
        root = data / "results/api" / model
        plans = [json.loads(path.read_text()) for path in sorted((root / "plans").glob("*.json"))]
        by_phase = {plan["phase"]: plan for plan in plans}
        if len(by_phase) != len(plans) or phase not in by_phase:
            raise ValueError(f"{model}: missing or duplicate active phase")
        plan = by_phase[phase]
        if any(p["created_at"] > plan["created_at"] for p in plans):
            raise ValueError(f"{model}: a later plan exists; select its phase explicitly")
        for entry in plans:
            validate_plan(entry)
            if entry["model"] != model or entry["profile"]["sha256"] != plan["profile"]["sha256"]:
                raise ValueError(f"{model}: changed route/profile across plans")
            if entry["manifest_sha256"] != manifest_sha256:
                raise ValueError(f"{model}: changed manifest since plan freeze")
            if entry["stats_sent_to_models"] or entry["reference_scores_sent_to_models"]:
                raise ValueError(f"{model}: stats/reference contamination")
            case_views = {case["case_id"]: case["view_sha256"] for case in entry["cases"]}
            if (
                len(case_views) != len(entry["cases"])
                or not case_views
                or any(expected.get(identity) != view for identity, view in case_views.items())
            ):
                raise ValueError(f"{model}: plan cases do not match the delivered manifest")
            recovery = entry.get("recovery")
            if recovery is not None:
                limits = recovery.get("attempt_limits", {})
                if (
                    recovery.get("kind") != "empty_length_once_v1"
                    or set(limits) != set(case_views)
                    or any(
                        type(limit) is not int or not 0 < limit <= entry["policy"]["max_attempts"]
                        for limit in limits.values()
                    )
                ):
                    raise ValueError(f"{model}: unsupported or inconsistent recovery attempt limits")
            if entry["plan_id"] in all_plans:
                raise ValueError("Duplicate frozen plan ID")
            all_plans[entry["plan_id"]] = entry
        state = json.loads((root / f"{phase}-status.json").read_text())
        count = len(plan["cases"])
        if (
            state.get("plan_id") != plan["plan_id"]
            or state.get("status") != "complete"
            or any(
                state.get(field) != count for field in ("planned_cases", "selected_cases", "completed_cases")
            )
            or sum(state.get("outcomes", {}).values()) != count
        ):
            raise ValueError(f"{model}: current frozen plan is stopped, partial or incomplete")
        observed = [json.loads(path.read_text()) for path in (root / "observations").glob("*.json")]
        observed_views = {row["case_id"]: row["view_sha256"] for row in observed}
        if len(observed) != len(expected) or observed_views != expected:
            raise ValueError(f"{model}: observations do not exactly cover the delivered manifest")
        if plan.get("recovery", {}).get("kind") == "empty_length_once_v1":
            by_id = {row["case_id"]: row for row in observed}
            if any(
                by_id[identity].get("plan_id") != plan["plan_id"]
                or by_id[identity].get("attempt_count") != limit
                for identity, limit in plan["recovery"]["attempt_limits"].items()
            ):
                raise ValueError(f"{model}: recovery observations are stale or exceed their frozen limits")
            baseline = next(
                (item for item in plans if item["plan_id"] == plan["recovery"].get("baseline_plan_id")), None
            )
            if baseline is None:
                raise ValueError(f"{model}: recovery baseline plan is missing")
            verify_recovery_evidence(root, plan, baseline, expected)
        selected[model] = {"phase": phase, "plan_id": plan["plan_id"], "cases": count}
    return all_plans, selected


def model_metrics(rows, accounting, ensemble_model):
    metrics = []
    for model in (*MODELS, ensemble_model):
        selected = [row for row in rows if row["model"] == model]
        pairs = [
            row
            for row in selected
            if all(row.get(d + "_score") is not None for d in ("technical", "aesthetic"))
        ]
        latencies = [row["latency_seconds"] for row in selected if row.get("latency_seconds") is not None]
        metrics.append(
            {
                "model": model,
                "numeric_pairs": len(pairs),
                "expected": len(selected),
                "mean_technical": mean(row["technical_score"] for row in pairs) if pairs else None,
                "mean_aesthetic": mean(row["aesthetic_score"] for row in pairs) if pairs else None,
                "median_final_attempt_seconds": median(latencies) if latencies else None,
                "statuses": dict(Counter(row["status"] for row in selected)),
                "requested_high_reasoning": sum(row.get("requested_reasoning") == "high" for row in selected),
                "accounting": accounting.get(model),
            }
        )
    return metrics


def optional_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_report(data, old, summary):
    def formatted(value):
        return "—" if value is None else f"{value:.2f}"

    lines = [
        "# 40,000-image labeling expansion",
        "",
        (
            "This run adds 20,000 new images to the frozen 20,000-image dataset. "
            "The existing 1,500 assistant references, first-1,000 publication allowlist and initial "
            "10,000 Q-ReAlign Lite observations are retained unchanged. No new references or Lite inference were added."
        ),
        "",
        (
            "The canonical ensemble remains 37.5% Luna, 37.5% DeepSeek and 25% Liquid, independently "
            "for technical quality and aesthetics. All three valid components are required for each outcome. "
            "The unchanged v2 profile uses high reasoning for Luna and DeepSeek and revised typed Decisions "
            "questions for Liquid and Jev."
        ),
        "",
        "## New 20K coverage",
        "",
        "| Model | Numeric pairs | Mean technical | Mean aesthetic | Median final HTTP attempt (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["new_model_metrics"]:
        lines.append(
            f"| {row['model']} | {row['numeric_pairs']:,}/{row['expected']:,} | "
            f"{formatted(row['mean_technical'])} | {formatted(row['mean_aesthetic'])} | "
            f"{formatted(row['median_final_attempt_seconds'])} |"
        )
    lines += [
        "",
        (
            "Latency is measured from HTTP invocation through complete response receipt. Baseline grading "
            "ran four concurrent routes with up to 16 workers per route, so these are observed timings "
            "under shared load, not isolated provider benchmarks. Payload preparation and scheduler wait "
            "are excluded; retries retain separate timings and receipts."
        ),
        "",
        (
            "Means describe available numeric pairs, not accuracy. There are no independent "
            "reference scores for the new 20K. Missing, invalid and exhausted requests remain missing "
            "where appropriate, with original attempts, measured timings and receipts retained."
        ),
        "",
        "| Model | Combined numeric pairs | Combined statuses |",
        "|---|---:|---|",
    ]
    for model, row in summary["coverage"].items():
        statuses = ", ".join(f"{key}: {value:,}" for key, value in sorted(row["status_counts"].items()))
        lines.append(f"| {model} | {row['numeric_pairs']:,}/{row['expected']:,} | {statuses} |")
    lines += [
        "",
        "## Corpus and review metadata",
        "",
        (
            "Actual EmbeddingGemma 2 vectors organize content clusters, related images and duplicate "
            "review candidates. See `content-coverage.json` and `possible-duplicates.json` for the "
            "local detailed evidence; similarity is not a quality label."
        ),
        "",
        (
            "All new exact delivered views have local image-stats metadata: sharpness_peak, luma_entropy, "
            "focus_area, channel_clip and Hasler–Suesstrunk colourfulness. histogram_entropy is an alias "
            "of luma_entropy, not an independent measurement. focus_area is relative to the image's own "
            "sharpest region. Statistics were not sent to graders or used to overwrite scores."
        ),
        "",
        "## Final datasets and accounting",
        "",
        f"- Full canonical labels: `{data / 'exports/labelled-images.jsonl'}`.",
        f"- Combined model rows: `{data / 'exports/scores-long.jsonl'}` ({LONG_ROWS:,} rows).",
        f"- Incremental labels: `{data / 'exports/new-labelled-images.jsonl'}`.",
        f"- Prior canonical bytes: `{old / 'exports/labelled-images.jsonl'}` are unchanged.",
        "- Git publication remains the unchanged first 1,000 reference images in `labeling/records/`.",
        "- All training masks remain false; acquisition split=train is not training authorization.",
        "",
        (
            "Costs below include all retained attempts, retries and compatibility probes. The previous "
            "20K combined cost ledger is carried forward exactly once, then only this run's attempts "
            "are added. Unknown charges remain unknown. Ensemble rows are derived and do not add API "
            "charges or independent request latency. Reference and historical Lite prices remain $0.00; "
            "actual infrastructure charges are separate."
        ),
        "",
        "Embedding billing receipt: "
        + ("recorded" if summary["embedding_billing"] else "not yet available")
        + ". "
        "Cleanup evidence: "
        + ("recorded; see summary" if summary["embedding_cleanup"] else "not yet available")
        + ".",
        "",
        "| Model | Prior 20K known USD | New 20K known USD, including probes | Total known USD | Unpriced attempts | Exact total USD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fresh = combine_costs(summary["accounting"], summary["compatibility_probe_accounting"])
    for model in MODELS:
        total = summary["combined_cost_per_model"][model]
        lines.append(
            f"| {model} | {summary['historical_cost_per_model'][model]['known_price_usd']} | "
            f"{fresh[model]['known_price_usd']} | {total['known_price_usd']} | "
            f"{total['unpriced_attempts']} | {total['total_price_usd'] or 'unknown'} |"
        )
    lines += [
        "| Reference | 0.00 | 0.00 | 0.00 | 0 | 0.00 |",
        "| Q-ReAlign Lite (historical only) | 0.00 | 0.00 | 0.00 | 0 | 0.00 |",
    ]
    (data / "REPORT.md").write_text("\n".join(lines) + "\n")


def export_run(data, old, phases):
    """Called while finalizer holds every model lock; never submits inference."""
    import api_runner
    import corpus  # noqa: F401 -- establishes the frozen adapter import context
    import publish_run as publication
    from corpus import digest, read_lines, write_json, write_lines

    from visual_quality.ensemble import MODEL

    for path, expected in json.loads((data / "initial-evidence-pins.json").read_text()).items():
        if digest(path) != expected:
            raise ValueError(f"Historical/publication evidence changed: {path}")
    manifest = json.loads((data / "manifest.json").read_text())
    if len(manifest["cases"]) != NEW_IMAGES:
        raise ValueError("This expansion requires exactly 20,000 new images")
    plans, selected = validate_completed_runs(data, manifest, phases, api_runner.validate_plan)
    original_models = publication.MODELS
    try:
        publication.MODELS = (*MODELS, "qrealign_lite")  # load_api uses the four API routes preceding Lite.
        observations, accounting = publication.load_api(data / "results/api")
        _, probes = publication.load_api(data / "compatibility-probe/results/api")
        publication.MODELS = MODELS
        rows, images, summary = publication.build_exports(
            manifest,
            [],
            observations,
            read_lines(data / "stats.jsonl"),
            expected_images=NEW_IMAGES,
            expected_references=0,
        )
    finally:
        publication.MODELS = original_models
    if any(not finalized_observation(row, plans) for row in rows if row["model"] in MODELS):
        raise ValueError("Unfinished or unprovenanced observations cannot become a final export")
    if summary["statistics_images"] != NEW_IMAGES or any(
        image["statistics_status"] != "available" for image in images
    ):
        raise ValueError("Statistics incomplete")
    if any(set(image["assessments"]) != {*MODELS, MODEL} for image in images):
        raise ValueError("Exactly four graders and the canonical ensemble are required")
    exports = data / "exports"
    exports.mkdir(exist_ok=True)
    write_lines(exports / "new-labelled-images.jsonl", images)
    write_lines(exports / "new-scores-long.jsonl", rows)
    append_export(old / "exports/labelled-images.jsonl", images, exports / "labelled-images.jsonl")
    append_export(old / "exports/scores-long.jsonl", rows, exports / "scores-long.jsonl")
    fields = [
        "case_id",
        "model",
        "technical_score",
        "aesthetic_score",
        "latency_seconds",
        "latency_kind",
        "price_usd",
        "price_basis",
        "status",
        "view_sha256",
    ]
    with (exports / "scores-long.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records(exports / "scores-long.jsonl"))
    previous = json.loads((old / "exports/summary.json").read_text())
    if previous["images"] != TOTAL_IMAGES - NEW_IMAGES:
        raise ValueError("Unexpected historical image count")
    combined_coverage = coverage(records(exports / "scores-long.jsonl"))
    if sum(row["expected"] for row in combined_coverage.values()) != LONG_ROWS:
        raise ValueError("Combined long export row count differs")
    summary.update(
        images=TOTAL_IMAGES,
        total_images=TOTAL_IMAGES,
        new_images=NEW_IMAGES,
        benchmark_images=REFERENCE_IMAGES,
        reference_records=REFERENCE_IMAGES,
        publication_images=PUBLICATION_IMAGES,
        statistics_images=previous["statistics_images"] + summary["statistics_images"],
        incremental_coverage=summary["coverage"],
        coverage=combined_coverage,
        new_model_metrics=model_metrics(rows, accounting, MODEL),
        accounting=accounting,
        compatibility_probe_accounting=probes,
        historical_cost_per_model=previous["combined_cost_per_model"],
        combined_cost_per_model=combine_costs(previous["combined_cost_per_model"], accounting, probes),
        current_plans=selected,
        initial_images_unchanged=True,
        qrealign_new_inferences=0,
        corpus_coverage=json.loads((data / "content-coverage.json").read_text()),
        possible_duplicate_pairs=json.loads((data / "possible-duplicates.json").read_text()),
        embedding_billing=optional_json(data / "embedding-operations/billing-receipt.json"),
        embedding_cleanup=optional_json(data / "embedding-operations/cleanup-status.json"),
        initial_exports={
            name: digest(old / "exports" / name) for name in ("labelled-images.jsonl", "scores-long.jsonl")
        },
        exports={path.name: digest(path) for path in exports.glob("*.jsonl")},
    )
    write_json(exports / "summary.json", summary)
    write_report(data, old, summary)
    return summary


if __name__ == "__main__":
    from finalize import main

    main()
