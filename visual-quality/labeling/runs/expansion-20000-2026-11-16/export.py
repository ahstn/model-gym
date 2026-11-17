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
import api_runner

from visual_quality.ensemble import MODEL

MODELS = ("luna", "deepseek", "liquid", "jev")


def finalized_observation(row, plans):
    """Exhausted transport failures stay missing, but unfinished work cannot export."""
    if row["status"] in ("valid", "partial", "invalid", "failed"):
        return True
    if row["status"] != "retryable_failure":
        return False
    plan = plans.get(row.get("source_request_plan_id"))
    if not plan:
        return False
    policy = plan["policy"]
    baseline = policy.get("authorized_denials", {}).get(row.get("case_id"))
    maximum = baseline["attempt_count"] + policy["additional_attempts"] if baseline else policy["max_attempts"]
    return row.get("attempt_count", 0) >= maximum


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
    _, historical_cost = publication.load_api(OLD / "results/api")
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
        times = [r["latency_seconds"] for r in numeric if r.get("latency_seconds") is not None]
        all_times = [r["latency_seconds"] for r in selected if r.get("latency_seconds") is not None]
        table.append(
            {
                "model": model,
                "new_images": len(selected),
                "valid_score_pairs": len(numeric),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "content_denials": sum(
                    api_runner.image_content_rejection(api_runner.read(r["raw_response_path"]))
                    for r in selected
                    if r["status"] == "failed" and r.get("raw_response_path")
                ),
                "mean_technical": mean(r["technical_score"] for r in numeric) if numeric else None,
                "mean_aesthetic": mean(r["aesthetic_score"] for r in numeric) if numeric else None,
                "median_latency_seconds": median(times) if times else None,
                "median_latency_scope": "final HTTP attempts with both outcome scores available",
                "median_all_final_attempt_latency_seconds": median(all_times) if all_times else None,
                "high_reasoning_observations": sum(r.get("requested_reasoning") == "high" for r in numeric),
                "positive_reasoning_token_receipts": sum((r.get("reasoning_tokens") or 0) > 0 for r in numeric),
                "accounting": accounting.get(model),
            }
        )
    initial_summary = json.loads((OLD / "exports/summary.json").read_text())
    content = json.loads((DATA / "content-coverage.json").read_text())
    duplicates = json.loads((DATA / "possible-duplicates.json").read_text())
    embedding_billing = json.loads((DATA / "embedding-operations/billing-receipt.json").read_text())
    embedding_cleanup = json.loads((DATA / "embedding-operations/cleanup-status.json").read_text())
    historical_coverage = {}
    for row in read_lines(OLD / "exports/scores-long.jsonl"):
        entry = historical_coverage.setdefault(
            row["model"], {"expected": 0, "numeric_pairs": 0, "status_counts": Counter()}
        )
        entry["expected"] += 1
        entry["numeric_pairs"] += int(all(row.get(d + "_score") is not None for d in ("technical", "aesthetic")))
        entry["status_counts"][row["status"]] += 1
    incremental_coverage = summary["coverage"]
    combined_coverage = {}
    for model in sorted(set(historical_coverage) | set(incremental_coverage)):
        parts = [historical_coverage.get(model, {}), incremental_coverage.get(model, {})]
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
    recovery = {}
    before_path = DATA / "snapshots/before-guardrail-recovery/exports/summary.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        for model in ("liquid", "jev"):
            state_path = DATA / "results/api" / model / "secret-guardrail-disabled-status.json"
            if state_path.exists():
                recovery[model] = {
                    **json.loads(state_path.read_text()),
                    "additional_known_price_usd": str(Decimal(accounting[model]["known_price_usd"]) - Decimal(before["accounting"][model]["known_price_usd"])),
                    "additional_http_attempts": accounting[model]["attempts"] - before["accounting"][model]["attempts"],
                }
                gateway_path = DATA / "results/api" / model / "missing-output-postpass-status.json"
                if gateway_path.exists():
                    recovery[model]["gateway_postpass"] = json.loads(gateway_path.read_text())
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
        corpus_coverage=content,
        possible_duplicate_pairs=len(duplicates["pairs"]),
        embedding_billing=embedding_billing,
        embedding_cleanup=embedding_cleanup,
        guardrail_recovery=recovery,
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
        "## Corpus and review",
        "",
        "The added set contains 7,500 Megalith images, 1,500 VizWiz images, 250 non-photographic controls and 750 controlled variants from 250 included parents. Exact delivered-image hashes are unique across all 20,000 records; new source families are disjoint from the original set and protected evaluation families were excluded before selection.",
        "",
        f"EmbeddingGemma 2 revision `{content['revision']}` assigns new candidates to the original 64 content prototypes. Square-root sampling quotas account for existing cluster coverage without reading quality labels. Of the new records, {content['embedding_modes']['exact_view']:,} use exact-view embeddings and {content['embedding_modes']['family_representative']:,} explicitly reuse a source-family embedding for a resized or distorted view. Graders and image-stats always receive the actual delivered view.",
        "",
        f"Six cross-family content neighbors are surfaced per image where available. There are {len(duplicates['pairs']):,} high-similarity pair candidates across the combined corpus; they are unconfirmed review candidates, not automatic duplicate labels. The local `review/index.html` shows content counts and the highest-similarity pairs. All candidates remain in `possible-duplicates.json`.",
        "",
        "The original 1,500 assistant references remain the benchmark. No new reference judgments were requested, and no additional Q-ReAlign calls were made. The first 1,000 committed examples remain unchanged.",
        "",
        "## Incremental model coverage",
        "",
        "| Model | Valid score pairs | Mean tech | Mean aest | Median valid HTTP seconds |",
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
        "These are score distributions, not accuracy measurements: no new independent reference scores were created. Means use each model's available score pairs, so differing missing coverage also changes the evaluated population. The original benchmark remains the comparison baseline. Missing or invalid outcomes remain missing; the ensemble never renormalizes its weights.",
        "",
        "Latency is the final successful HTTP attempt through complete response receipt. It excludes payload preparation and earlier retries. Raw records retain every attempt's latency and costs. Concurrency and service conditions changed during this run, so these measurements are operational observations rather than a controlled speed comparison.",
        "",
        "## Operational limitations",
        "",
        f"DeepSeek returned 9,998 numeric pairs, one schema-valid unscorable response, and one invalid aesthetic response retaining its technical score. The added set has {incremental_coverage[MODEL]['numeric_pairs']:,} complete ensemble pairs; the combined set has {combined_coverage[MODEL]['numeric_pairs']:,}.",
        "",
        "The initial teacher phase encountered connection errors and rate limits. A separately frozen continuation reduced concurrency to 16 and allowed bounded additional transient retries while preserving valid responses and all unknown-charge reserves. The rubric, image bytes, high reasoning and response validation were unchanged.",
        "",
        "The first pass retained 2,855 shared OpenRouter secret-scanner denials per decision model: 2,757 compressed Bitcoin-key patterns, 85 uncompressed Bitcoin-key patterns and 13 Google API-key patterns. The user subsequently disabled their secret-redaction guardrail and explicitly approved resending these flagged image payloads. The recovery uses identical images and prompts, keeps all original receipts, preserves other assessments, and stops on a new 403. Earlier exports remain in the local pre-recovery snapshot. Unknown original charges remain unknown.",
        "",
        "Recovery results: " + "; ".join(f"{m}: {v['status']}, {v.get('outcomes', {})}, additional known cost ${Decimal(v['additional_known_price_usd']):.6f}" for m, v in recovery.items()) if recovery else "No guardrail recovery has been executed.",
        "",
        "A separate, single-request HTTP520 recovery is recorded where needed: " + "; ".join(f"{m}: {v['gateway_postpass']['status']}, {v['gateway_postpass'].get('outcomes', {})}" for m, v in recovery.items() if "gateway_postpass" in v),
        "",
        "Jev's original HTTP400 `Invalid image dimensions` rejection is outside the secret-scanner recovery and remains missing. The exact view stays unchanged across graders. Current coverage and any additional failures are retained in the aggregate JSON.",
        "",
        "## Files and infrastructure",
        "",
        f"Full local data: `{DATA}`. The combined image records are `exports/labelled-images.jsonl`; `exports/new-labelled-images.jsonl` contains only the added 10K. `exports/scores-long.jsonl` and `.csv` contain assessor rows. `stats.jsonl` contains new image measurements, also joined separately into image-level exports. None of the added per-image labels is committed to Git.",
        "",
        f"The owned embedding GPU was stopped after vectors were saved. Runpod reported ${embedding_billing['receipt']['metadata']['totals']['totalAmount']:.6f} for that pod as of {embedding_billing['queried_at']}; billing may settle later. This infrastructure charge is separate from the API table below and does not change the historical Q-ReAlign/reference zero-price reporting rule.",
        "The pod's disk remains retained while permission for permanent deletion is pending; storage charges may continue. Automatic approval review rejected irreversible disk deletion without explicit authorization."
        if embedding_cleanup.get("disk_retained")
        else "The temporary embedding pod and disk have been removed after explicit authorization; all required outputs remain local.",
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
