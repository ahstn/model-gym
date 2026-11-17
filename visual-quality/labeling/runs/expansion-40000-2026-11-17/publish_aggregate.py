"""Publish only explicitly selected aggregate fields from the ignored full run.

The full summary includes duplicate-pair identities and must never be copied into
Git. This projection drops all free text, paths, observations and review records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from visual_quality.ensemble import configuration

MODELS = ("luna", "deepseek", "liquid", "jev")
ASSESSORS = (*MODELS, "photo_ensemble", "qrealign_lite", "reference")
STATUSES = {
    "valid",
    "partial",
    "invalid",
    "failed",
    "missing",
    "retryable_failure",
    "observation_not_finalized",
}
COUNTS = (
    "images",
    "total_images",
    "new_images",
    "benchmark_images",
    "reference_records",
    "publication_images",
    "statistics_images",
    "qrealign_new_inferences",
)
CATEGORIES = ("megalith", "vizwiz", "controls", "distortions")


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Aggregate counts must be nonnegative integers")
    return value


def number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid numeric aggregate")
    return value


def money(value):
    if value is None:
        return None
    if type(value) not in (str, int, float):
        raise ValueError("Invalid aggregate monetary value")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid aggregate monetary value") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid aggregate monetary value")
    return str(amount)


def costs(ledger):
    result = {}
    for model in MODELS:
        row = ledger[model]
        result[model] = {
            "known_price_usd": money(row["known_price_usd"]),
            "unpriced_attempts": count(row["unpriced_attempts"]),
            "total_price_usd": money(row["total_price_usd"]),
        }
    return result


def coverage(rows):
    if set(rows) - set(ASSESSORS):
        raise ValueError("Unknown assessor in aggregate coverage")
    result = {}
    for model, row in rows.items():
        if set(row["status_counts"]) - STATUSES:
            raise ValueError("Unknown aggregate outcome status")
        result[model] = {
            "expected": count(row["expected"]),
            "numeric_pairs": count(row["numeric_pairs"]),
            "status_counts": {status: count(n) for status, n in row["status_counts"].items()},
        }
    return result


def infrastructure(billing, cleanup):
    """A posted billing checkpoint is not a reconciled final infrastructure cost."""
    billing = billing or {}
    metadata = billing.get("receipt", {}).get("metadata", {})
    record_count = count(metadata.get("recordCount", 0))
    posted = money(metadata.get("totals", {}).get("totalAmount")) if record_count else None
    status = billing.get("accounting_status", "posted_checkpoint")
    if status not in {"posted_checkpoint", "reconciled_final"}:
        raise ValueError("Unknown infrastructure accounting status")
    if status == "reconciled_final" and posted is None:
        raise ValueError("Reconciled infrastructure billing requires a nonempty provider receipt")
    return {
        "accounting_status": status if posted is not None else "unavailable",
        "posted_charge_usd": posted,
        "runtime_estimate_usd": money(billing.get("runtime_estimate_usd")),
        "billing_records": record_count,
        "cleanup_evidence_available": cleanup is not None,
    }


def project(summary, audit):
    """Construct a new object; never recursively copy arbitrary source content."""
    if audit.get("passed") is not True or audit.get("total_images") != summary["total_images"]:
        raise ValueError("A matching completed audit is required before aggregate publication")
    export_hash = summary.get("exports", {}).get("labelled-images.jsonl")
    if (
        not isinstance(export_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", export_hash)
        or audit.get("exports_sha256") != export_hash
        or set(summary.get("current_plans", {})) != set(MODELS)
        or audit.get("current_plans") != summary["current_plans"]
    ):
        raise ValueError("Audit export hash or current plans do not match this summary")
    if summary.get("training_ready") is not False or summary.get("stats_in_inference") is not False:
        raise ValueError("Training/stats policy differs from the accepted run")
    result = {
        "schema_version": "visual-quality-40k-aggregate-v1",
        **{key: count(summary[key]) for key in COUNTS},
        "training_ready": False,
        "stats_in_inference": False,
        "coverage": coverage(summary["coverage"]),
        "incremental_coverage": coverage(summary["incremental_coverage"]),
        "ensemble_weights": configuration()["weights"]["technical"],
        "historical_cost_per_model": costs(summary["historical_cost_per_model"]),
        "combined_cost_per_model": costs(summary["combined_cost_per_model"]),
        "accounting": costs(summary["accounting"]),
        "compatibility_probe_accounting": costs(summary["compatibility_probe_accounting"]),
    }
    expected_weights = configuration()["weights"]
    if summary["ensemble"]["weights"] != expected_weights:
        raise ValueError("Ensemble weights differ from the accepted scorer")
    result["new_cost_per_model"] = {}
    for model in MODELS:
        entries = (result["accounting"][model], result["compatibility_probe_accounting"][model])
        known = sum((Decimal(row["known_price_usd"]) for row in entries), Decimal(0))
        missing = sum(row["unpriced_attempts"] for row in entries)
        result["new_cost_per_model"][model] = {
            "known_price_usd": str(known),
            "unpriced_attempts": missing,
            "total_price_usd": str(known) if not missing else None,
        }
    result["new_model_metrics"] = []
    seen_models = set()
    for row in summary["new_model_metrics"]:
        if row["model"] not in (*MODELS, "photo_ensemble") or row["model"] in seen_models:
            raise ValueError("Unknown or duplicate new grader")
        seen_models.add(row["model"])
        result["new_model_metrics"].append(
            {
                "model": row["model"],
                "numeric_pairs": count(row["numeric_pairs"]),
                "expected": count(row["expected"]),
                **{
                    key: number(row[key])
                    for key in ("mean_technical", "mean_aesthetic", "median_final_attempt_seconds")
                },
                "requested_high_reasoning": count(row["requested_high_reasoning"]),
            }
        )
    corpus = summary["corpus_coverage"]
    cluster_ids = [count(row["cluster"]) for row in corpus["clusters"]]
    if len(set(cluster_ids)) != len(cluster_ids) or any(identity >= 64 for identity in cluster_ids):
        raise ValueError("Expected unique content cluster numbers from 0 through 63")
    result["corpus_coverage"] = {
        "categories": {key: count(corpus["categories"].get(key, 0)) for key in CATEGORIES},
        "clusters": [
            {key: count(row[key]) for key in ("cluster", "candidate_count", "initial_count", "new_count")}
            for row in corpus["clusters"]
        ],
        "possible_duplicate_pairs": len(summary["possible_duplicate_pairs"]["pairs"]),
    }
    result["audit"] = {
        "passed": True,
        **{
            key: count(audit[key])
            for key in (
                "new_images",
                "total_images",
                "raw_observations_reparsed",
                "image_statistics_joins",
                "initial_records_unchanged",
                "new_qrealign_calls",
                "reference_images",
                "publication_images",
            )
        },
    }
    result["embedding_infrastructure"] = infrastructure(
        summary.get("embedding_billing"), summary.get("embedding_cleanup")
    )
    return result


def report(summary):
    def display(value):
        return "—" if value is None else f"{value:.2f}"

    infrastructure_cost = summary["embedding_infrastructure"]
    status = infrastructure_cost["accounting_status"]
    posted = infrastructure_cost["posted_charge_usd"]
    estimate = infrastructure_cost["runtime_estimate_usd"]
    if status == "reconciled_final":
        infrastructure_text = f"Embedding infrastructure: reconciled provider total ${posted}."
    elif status == "posted_checkpoint":
        infrastructure_text = (
            f"Embedding infrastructure: provider-posted billing checkpoint ${posted}; "
            "this is not a reconciled final total."
        )
    else:
        infrastructure_text = "Embedding infrastructure: provider billing is not yet available."
    if estimate is not None:
        infrastructure_text += f" The elapsed-runtime cost estimate is ${estimate}."
    infrastructure_text += (
        " Cleanup evidence is recorded."
        if infrastructure_cost["cleanup_evidence_available"]
        else " Cleanup evidence is not yet recorded."
    )

    lines = [
        "# 40K labeling aggregate report",
        "",
        (
            f"The audited corpus contains {summary['images']:,} images, including {summary['new_images']:,} "
            "new images. The original 1,500 assistant references and first-1,000 Git publication remain unchanged. "
            "No new Q-ReAlign inference was run. Training masks remain disabled."
        ),
        "",
        (
            "The ensemble remains 37.5% Luna, 37.5% DeepSeek and 25% Liquid for each outcome. "
            "Luna and DeepSeek use high reasoning. Image statistics remain separate review metadata. "
            "The new sample has no independent reference ratings; mean scores are not accuracy measures."
        ),
        "",
        "| Model | New numeric pairs | Mean technical | Mean aesthetic | Median final attempt (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["new_model_metrics"]:
        lines.append(
            f"| {row['model']} | {row['numeric_pairs']:,}/{row['expected']:,} | "
            f"{display(row['mean_technical'])} | {display(row['mean_aesthetic'])} | "
            f"{display(row['median_final_attempt_seconds'])} |"
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
            f"EmbeddingGemma 2 grouped content into {len(summary['corpus_coverage']['clusters'])} clusters. "
            f"There are {summary['corpus_coverage']['possible_duplicate_pairs']:,} potential duplicate pairs "
            "for review; similarity does not establish duplicate identity or quality."
        ),
        "",
        (
            "Full labels, observations, images, neighbors and duplicate identities remain in ignored local "
            "data. Only explicitly selected aggregate fields appear in this report and its companion summary."
        ),
        "",
        (
            "Known costs include historical and new attempts without counting the derived ensemble twice. "
            "Unknown charges remain unknown; retries and probes are included. Reference and historical Lite "
            "prices are $0.00 by the user's reporting rule. Infrastructure is separate."
        ),
        "",
        infrastructure_text,
        "",
        "| Model | New known API USD, including probes | Combined known API USD | Combined unpriced attempts | Exact combined API USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in summary["combined_cost_per_model"].items():
        lines.append(
            f"| {model} | {summary['new_cost_per_model'][model]['known_price_usd']} | "
            f"{row['known_price_usd']} | {row['unpriced_attempts']} | "
            f"{row['total_price_usd'] or 'unknown'} |"
        )
    lines += [
        "| Reference | 0.00 | 0.00 | 0 | 0.00 |",
        "| Q-ReAlign Lite (historical only) | 0.00 | 0.00 | 0 | 0.00 |",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    summary_path, audit_path = args.data / "exports/summary.json", args.data / "audit.json"
    summary = project(json.loads(summary_path.read_text()), json.loads(audit_path.read_text()))
    summary["source_sha256"] = {
        "summary": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "audit": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_directory / "REPORT.md").write_text(report(summary))


if __name__ == "__main__":
    main()
