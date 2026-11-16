# ruff: noqa: I001
# Bootstrap the frozen source adapters before importing publication.
"""Join the new observations and append them to immutable initial-10K exports."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from decimal import Decimal
from statistics import mean, median

from corpus import DATA, OLD, digest, read_lines, write_json, write_lines

import publish_run as publication

from visual_quality.ensemble import MODEL

MODELS = ("luna", "deepseek", "liquid", "jev")


def finalized_observation(row, plans):
    """Exhausted transport failures stay missing, but unfinished work cannot export."""
    if row["status"] in ("valid", "partial", "invalid", "failed"):
        return True
    if row["status"] != "retryable_failure":
        return False
    plan = plans.get(row.get("source_request_plan_id"))
    return bool(plan and row.get("attempt_count", 0) >= plan["policy"]["max_attempts"])


def append_export(old_path, new_rows, target):
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as out:
        with old_path.open("rb") as src:
            shutil.copyfileobj(src, out)
        for row in new_rows:
            out.write((json.dumps(row, sort_keys=True) + "\n").encode())
    temporary.replace(target)


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    observations, accounting = publication.load_api(DATA / "results/api")
    _, probe_accounting = publication.load_api(DATA / "compatibility-probe/results/api")
    publication.MODELS = MODELS
    rows, images, summary = publication.build_exports(
        manifest,
        [],
        observations,
        read_lines(DATA / "stats.jsonl"),
        expected_images=10000,
        expected_references=0,
    )
    plans = {
        plan["plan_id"]: plan
        for path in (DATA / "results/api").glob("*/plans/*.json")
        for plan in [json.loads(path.read_text())]
    }
    if any(not finalized_observation(r, plans) for r in rows if r["model"] in MODELS):
        raise ValueError("Unfinished or missing model observations; do not publish a final export")
    if summary["statistics_images"] != 10000:
        raise ValueError("Statistics incomplete")
    if any("qrealign_lite" in r["assessments"] or "reference" in r["assessments"] for r in images):
        raise ValueError("New inference must contain exactly four graders plus ensemble")
    exports = DATA / "exports"
    exports.mkdir(exist_ok=True)
    write_lines(exports / "new-labelled-images.jsonl", images)
    write_lines(exports / "new-scores-long.jsonl", rows)
    append_export(OLD / "exports/labelled-images.jsonl", images, exports / "labelled-images.jsonl")
    append_export(OLD / "exports/scores-long.jsonl", rows, exports / "scores-long.jsonl")
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
        with (exports / "scores-long.jsonl").open() as source:
            for line in source:
                writer.writerow(json.loads(line))
    table = []
    for model in (*MODELS, MODEL):
        selected = [r for r in rows if r["model"] == model]
        numeric = [
            r for r in selected if all(r.get(d + "_score") is not None for d in ("technical", "aesthetic"))
        ]
        times = [r["latency_seconds"] for r in selected if r.get("latency_seconds") is not None]
        table.append(
            {
                "model": model,
                "new_images": len(selected),
                "valid_score_pairs": len(numeric),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "mean_technical": mean(r["technical_score"] for r in numeric) if numeric else None,
                "mean_aesthetic": mean(r["aesthetic_score"] for r in numeric) if numeric else None,
                "median_latency_seconds": median(times) if times else None,
                "accounting": accounting.get(model),
            }
        )
    initial_summary = json.loads((OLD / "exports/summary.json").read_text())
    historical_cost = initial_summary["cost_per_model"]
    incremental_coverage = summary["coverage"]
    combined_coverage = {}
    for model in sorted(set(initial_summary["coverage"]) | set(incremental_coverage)):
        parts = [initial_summary["coverage"].get(model, {}), incremental_coverage.get(model, {})]
        statuses = Counter()
        for part in parts:
            statuses.update(part.get("status_counts", {}))
        combined_coverage[model] = {
            "expected": sum(p.get("expected", 0) for p in parts),
            "numeric_pairs": sum(p.get("numeric_pairs", 0) for p in parts),
            "status_counts": dict(statuses),
        }
    combined_cost = {}
    for model in MODELS:
        parts = [historical_cost[model], accounting[model], probe_accounting[model]]
        known = sum((Decimal(p["known_price_usd"]) for p in parts), Decimal(0))
        missing = sum(p["unpriced_attempts"] for p in parts)
        combined_cost[model] = {
            "known_price_usd": str(known),
            "unpriced_attempts": missing,
            "total_price_usd": None if missing else str(known),
        }
    summary.update(
        images=20000,
        total_images=20000,
        benchmark_images=1500,
        statistics_images=initial_summary["statistics_images"] + summary["statistics_images"],
        reference_records=1500,
        coverage=combined_coverage,
        incremental_coverage=incremental_coverage,
        new_images=10000,
        reference_images=1500,
        publication_images=1000,
        new_model_metrics=table,
        accounting=accounting,
        compatibility_probe_accounting=probe_accounting,
        combined_cost_per_model=combined_cost,
        initial_images_unchanged=True,
        qrealign_new_inferences=0,
        initial_exports={
            n: digest(OLD / "exports" / n) for n in ["labelled-images.jsonl", "scores-long.jsonl"]
        },
        exports={
            n: digest(exports / n)
            for n in [
                "labelled-images.jsonl",
                "scores-long.jsonl",
                "new-labelled-images.jsonl",
                "new-scores-long.jsonl",
            ]
        },
    )
    write_json(exports / "summary.json", summary)
    lines = [
        "# 20,000-image labeling expansion",
        "",
        "The existing ensemble remains 37.5% Luna, 37.5% DeepSeek and 25% Liquid. The new 10,000 images use the unchanged v2 profile. The initial 10K observations are appended byte-for-byte; no additional Q-ReAlign inference or reference assessments were requested. The existing 1,500 assistant references and first-1,000 publication allowlist remain unchanged.",
        "",
        "All new per-image data is local and ignored by Git. Image statistics are metadata only. Training masks remain disabled.",
        "",
        "## Incremental model coverage",
        "",
        "| Model | Valid score pairs | Mean tech | Mean aest | Median HTTP seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in table:
        latency = "—" if r["median_latency_seconds"] is None else f"{r['median_latency_seconds']:.2f}"
        technical = "—" if r["mean_technical"] is None else f"{r['mean_technical']:.3f}"
        aesthetic = "—" if r["mean_aesthetic"] is None else f"{r['mean_aesthetic']:.3f}"
        lines.append(
            f"| {r['model']} | {r['valid_score_pairs']:,} / {r['new_images']:,} | {technical} | {aesthetic} | {latency} |"
        )
    lines += [
        "",
        "These are score distributions, not accuracy measurements: no new independent reference scores were created. The original benchmark remains the comparison baseline. Missing or invalid outcomes remain missing; the ensemble never renormalizes its weights.",
        "",
        "## Total API cost by model",
        "",
        "Known subtotals include all recorded attempts and retries: initial 10K, added 10K and the 16-call compatibility probe. Unpriced attempts are excluded from these subtotals, so a total with unpriced calls is a lower bound. The ensemble has no independent charge. Embedding infrastructure is reported separately.",
        "",
        "| Model | Initial 10K known USD | Added 10K known USD | Probe known USD | Combined known USD | Unpriced attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = [
            historical_cost[model]["known_price_usd"],
            accounting[model]["known_price_usd"],
            probe_accounting[model]["known_price_usd"],
            combined_cost[model]["known_price_usd"],
        ]
        money = " | ".join(f"${Decimal(v):.6f}" for v in values)
        lines.append(f"| {model} | {money} | {combined_cost[model]['unpriced_attempts']} |")
    lines.append("| Q-ReAlign Lite (historical only) | $0.00 | — | — | $0.00 | 0 |")
    lines.append("| Reference (existing 1,500) | $0.00 | — | — | $0.00 | 0 |")
    (DATA / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"images": 20000, "new_images": 10000, "coverage": summary["coverage"], "accounting": accounting}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
