"""Aggregate agreement, coverage, latency and cost; never export image-level labels.

Inputs are the publication pipeline's full local exports. Frozen ensemble weights
and reference labels are read-only. A report marked finished may retain terminal
failures or valid-unscorable responses, but may not hide unfinished observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean

MODELS = ("luna", "deepseek", "liquid", "jev", "qrealign_lite", "photo_ensemble")
NAMES = dict(
    zip(MODELS, ("Luna", "DeepSeek", "Liquid d1.1", "Jev 1.14", "Q-ReAlign Lite", "Ensemble"), strict=True)
)
NAMES["reference"] = "Reference"
DIMENSIONS = ("technical", "aesthetic")
GROUPS = {
    "original_500": ("Original references", 1, 500),
    "fresh_examples_500": ("Fresh reference examples", 501, 1000),
    "confirmation_500": ("Confirmation references", 1001, 1500),
}
UNFINISHED = {
    "missing",
    "in_flight",
    "observation_not_finalized",
    "retryable_failure",
    "pending",
    "interrupted",
}
FIELDS = {
    "case_id",
    "view_sha256",
    "model",
    "status",
    "benchmark_order",
    "technical_score",
    "aesthetic_score",
    "technical_score_valid",
    "aesthetic_score_valid",
    "reference_eligibility",
    "scope_excluded",
    "scope_floor_applied",
    "scope_validation_error",
    "latency_seconds",
    "latency_kind",
    "price_usd",
    "reuse",
}


def jsonlines(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def score(row, dimension):
    value = row.get(dimension + "_score")
    return (
        value
        if row.get("status") in ("valid", "partial")
        and row.get(dimension + "_score_valid") is True
        and finite(value)
        and 1 <= value <= 5
        else None
    )


def quantile(values, probability):
    """Linear quantile with endpoints included (NumPy/R type 7 convention)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def spearman(left, right):
    if len(left) < 2:
        return None
    x, y = ranks(left), ranks(right)
    xm, ym = mean(x), mean(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return numerator / denominator if denominator else None


def agreement(pairs):
    """Pairs are (reference, prediction); positive bias means overrating."""
    errors = [prediction - reference for reference, prediction in pairs]
    return {
        "n": len(pairs),
        "mae": mean(abs(value) for value in errors) if errors else None,
        "bias": mean(errors) if errors else None,
        "spearman": spearman([r for r, _ in pairs], [p for _, p in pairs]),
    }


def index_rows(rows):
    indexed, views = {}, {}
    for row in rows:
        key = (row["case_id"], row["model"])
        if key in indexed:
            raise ValueError("Duplicate model observation")
        if row["model"] not in (*MODELS, "reference"):
            raise ValueError("Unknown model observation")
        if row["case_id"] in views and views[row["case_id"]] != row["view_sha256"]:
            raise ValueError("Different delivered views joined across models")
        views[row["case_id"]] = row["view_sha256"]
        indexed[key] = row
    return indexed


def pairs_for(indexed, identities, model, dimension):
    pairs = []
    for identity in identities:
        reference = score(indexed[identity, "reference"], dimension)
        prediction = score(indexed.get((identity, model), {}), dimension)
        if reference is not None and prediction is not None:
            pairs.append((reference, prediction))
    return pairs


def common_ids(indexed, identities, models, dimension):
    return [
        identity
        for identity in identities
        if all(
            score(indexed.get((identity, model), {}), dimension) is not None
            for model in (*models, "reference")
        )
    ]


def technical_tails(pairs):
    poor = [p for r, p in pairs if r <= 2]
    strong = [p for r, p in pairs if r >= 4]
    return {
        "poor_reference_n": len(poor),
        "poor_predicted_at_least_3": sum(p >= 3 for p in poor),
        "strong_reference_n": len(strong),
        "strong_predicted_below_3": sum(p < 3 for p in strong),
    }


def scope_decision(row):
    """Read the recorded floor decision, never infer it from a score of one."""
    if row.get("status") not in {"valid", "partial"} or row.get("scope_validation_error"):
        return None
    flags = [row[key] for key in ("scope_excluded", "scope_floor_applied") if type(row.get(key)) is bool]
    return flags[0] if flags and len(set(flags)) == 1 else None


def scope_confusion(indexed, identities, model):
    counts = dict.fromkeys(
        (
            "reference_eligible_n",
            "eligible_classified_n",
            "eligible_false_floor_n",
            "eligible_unknown_n",
            "reference_excluded_n",
            "excluded_classified_n",
            "excluded_miss_n",
            "excluded_unknown_n",
            "reference_uncertain_or_missing_n",
            "classification_unknown_n",
        ),
        0,
    )
    for identity in identities:
        reference = indexed[identity, "reference"].get("reference_eligibility")
        prediction = scope_decision(indexed.get((identity, model), {}))
        counts["classification_unknown_n"] += prediction is None
        if reference not in {"eligible", "excluded"}:
            counts["reference_uncertain_or_missing_n"] += 1
            continue
        counts[f"reference_{reference}_n"] += 1
        if prediction is None:
            counts[f"{reference}_unknown_n"] += 1
        else:
            counts[f"{reference}_classified_n"] += 1
            if reference == "eligible":
                counts["eligible_false_floor_n"] += prediction
            else:
                counts["excluded_miss_n"] += not prediction
    return counts


def group_metrics(indexed, identities, stats):
    output = {"expected": len(identities), "per_model": {}, "common_coverage": {}, "paired_ensemble": {}}
    for model in MODELS:
        result = {
            dimension: agreement(pairs_for(indexed, identities, model, dimension)) for dimension in DIMENSIONS
        }
        result["technical_tails"] = technical_tails(pairs_for(indexed, identities, model, "technical"))
        output["per_model"][model] = result
    for name, models in (
        ("all_six", MODELS),
        ("four_api_and_ensemble", tuple(m for m in MODELS if m != "qrealign_lite")),
    ):
        output["common_coverage"][name] = {}
        for dimension in DIMENSIONS:
            common = common_ids(indexed, identities, models, dimension)
            output["common_coverage"][name][dimension] = {
                "n": len(common),
                "models": {
                    model: agreement(pairs_for(indexed, common, model, dimension)) for model in models
                },
            }
    for model in MODELS[:-1]:
        output["paired_ensemble"][model] = {}
        for dimension in DIMENSIONS:
            common = common_ids(indexed, identities, (model, "photo_ensemble"), dimension)
            candidate = agreement(pairs_for(indexed, common, model, dimension))
            ensemble = agreement(pairs_for(indexed, common, "photo_ensemble", dimension))
            output["paired_ensemble"][model][dimension] = {
                "n": len(common),
                "model_mae": candidate["mae"],
                "ensemble_mae": ensemble["mae"],
                "ensemble_minus_model_mae": ensemble["mae"] - candidate["mae"] if common else None,
            }
    # Scope-excluded flat content does not define the softness diagnostic threshold.
    eligible = [
        identity
        for identity in identities
        if indexed[identity, "reference"].get("reference_eligibility") == "eligible"
        and identity in stats
        and finite(stats[identity]["sharpness_peak"])
    ]
    threshold = quantile([stats[identity]["sharpness_peak"] for identity in eligible], 0.25)
    softness = {"eligible_n": len(eligible), "threshold": threshold, "models": {}}
    for model in MODELS:
        split = {}
        for name, identities_in_split in (
            ("soft", [i for i in eligible if stats[i]["sharpness_peak"] <= threshold]),
            ("other", [i for i in eligible if stats[i]["sharpness_peak"] > threshold]),
        ):
            split[name] = agreement(pairs_for(indexed, identities_in_split, model, "technical"))
        softness["models"][model] = split
    output["scope_confusion"] = {model: scope_confusion(indexed, identities, model) for model in MODELS[:4]}
    output["softness_diagnostic"] = softness
    return output


def cost_and_latency(rows, summary):
    costs, latency = {}, {}
    for model in (*MODELS[:-1], "reference"):
        selected = [row for row in rows if row["model"] == model]
        prices = [Decimal(str(row["price_usd"])) for row in selected if row.get("price_usd") is not None]
        source = summary["cost_per_model"][model]
        costs[model] = {
            **source,
            "selected_known_price_usd": str(sum(prices, Decimal(0))),
            "selected_unpriced_observations": sum(row.get("price_usd") is None for row in selected),
        }
        if model == "reference":
            continue
        completed = [
            row
            for row in selected
            if row.get("latency_kind") == "http_request_through_complete_response"
            and finite(row.get("latency_seconds"))
            and row["latency_seconds"] >= 0
        ]
        measured = [row["latency_seconds"] for row in completed]
        fresh = [row["latency_seconds"] for row in completed if not row.get("reuse")]
        latency[model] = {
            "complete_response_n": len(measured),
            "median_seconds": quantile(measured, 0.5),
            "p95_seconds": quantile(measured, 0.95),
            "reused_n": sum(bool(row.get("reuse")) for row in completed),
            "fresh_complete_response_n": len(fresh),
            "fresh_median_seconds": quantile(fresh, 0.5),
            "fresh_p95_seconds": quantile(fresh, 0.95),
        }
    return costs, latency


def build_report(rows, summary, stats, *, run_state):
    indexed = index_rows(rows)
    references = [row for row in rows if row["model"] == "reference"]
    orders = sorted(row.get("benchmark_order") for row in references)
    if orders != list(range(1, 1501)):
        raise ValueError("Expected exactly 1,500 references in explicit publication order")
    coverage = {}
    for model in (*MODELS, "reference"):
        selected = [row for row in rows if row["model"] == model]
        statuses = dict(Counter(row["status"] for row in selected))
        expected = summary["benchmark_images"] if model == "reference" else summary["images"]
        if len(selected) != expected:
            raise ValueError("Export is missing expected observation slots")
        coverage[model] = {
            "expected": expected,
            "status_counts": statuses,
            "numeric_pairs": sum(
                all(score(row, dimension) is not None for dimension in DIMENSIONS) for row in selected
            ),
        }
        if coverage[model] != summary["coverage"][model]:
            raise ValueError("Summary and score export coverage disagree")
        if run_state == "finished" and model in MODELS[:-1] and UNFINISHED.intersection(statuses):
            raise ValueError("Cannot mark a report finished while requests are unfinished")
    for reference in references:
        stat = stats.get(reference["case_id"])
        if stat and stat["view_sha256"] != reference["view_sha256"]:
            raise ValueError("Statistics and reference delivered views differ")
    groups = {}
    for name, (_, first, last) in GROUPS.items():
        identities = [row["case_id"] for row in references if first <= row["benchmark_order"] <= last]
        groups[name] = group_metrics(indexed, identities, stats)
    costs, latency = cost_and_latency(rows, summary)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_state": run_state,
        "images": summary["images"],
        "benchmark_images": len(references),
        "coverage": coverage,
        "groups": groups,
        "latency": latency,
        "cost_per_model": costs,
        "policy": {
            "reference_kind": "independent assistant judgments, not human ratings",
            "ensemble_weights": {"luna": 0.375, "deepseek": 0.375, "liquid": 0.25},
            "weights_fitted_here": False,
            "statistics_used_for_scores": False,
            "bias_definition": "prediction minus reference",
            "score_precision": "native; rounded only for presentation",
            "quantile_definition": "linear interpolation on (n-1)*p",
            "softness_threshold": "per-cohort 25th percentile among reference-eligible photographs; <= threshold is soft",
            "tail_thresholds": "reference <=2 with prediction >=3; reference >=4 with prediction <3",
        },
    }


def fmt(value, *, signed=False, digits=2):
    return "—" if value is None else format(value, f"{'+' if signed else ''}.{digits}f")


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ]


def fraction(numerator, denominator):
    return f"{numerator}/{denominator}" if denominator else "— (no paired observations)"


def dollars(value):
    return "unknown" if value is None else f"${Decimal(str(value)):.6f}"


def infrastructure_receipt(path):
    receipt = json.loads(path.read_text())
    metadata = receipt["billing"]["metadata"]
    records = receipt["billing"]["records"]
    total = Decimal(str(metadata["totals"]["totalAmount"]))
    record_total = sum((Decimal(str(row["totalAmount"])) for row in records), Decimal(0))
    if not total.is_finite() or total < 0 or abs(total - record_total) > Decimal("0.000001"):
        raise ValueError("Infrastructure receipt totals disagree")
    return {
        "posted_price_usd": str(total),
        "receipt_saved_at": receipt["receipt_saved_at"],
        "posted_through": max((row["endTime"] for row in records), default=None),
        "completeness": receipt["completeness"],
    }


def confirmation_summary(report):
    group = report["groups"]["confirmation_500"]
    common = group["common_coverage"]["four_api_and_ensemble"]
    if any(common[dimension]["n"] != group["expected"] for dimension in DIMENSIONS):
        return "The confirmation comparison is incomplete; use the matched-coverage tables below before comparing assessors."
    ensemble_t = common["technical"]["models"]["photo_ensemble"]["mae"]
    ensemble_a = common["aesthetic"]["models"]["photo_ensemble"]["mae"]
    luna_t = common["technical"]["models"]["luna"]["mae"]
    luna_a = common["aesthetic"]["models"]["luna"]["mae"]
    return (
        f"On the same {group['expected']} confirmation images, the fixed ensemble has technical MAE {fmt(ensemble_t, digits=3)} "
        f"and aesthetic MAE {fmt(ensemble_a, digits=3)}; Luna alone has {fmt(luna_t, digits=3)} and {fmt(luna_a, digits=3)}. "
        "Small differences are descriptive: no confidence intervals or human-rating validation were run. "
        "Lite is compared on its own shared coverage below."
    )


def render(report):
    partial = report["run_state"] != "finished"
    lines = [
        "# 10,000-image labeling rehearsal: agreement report",
        "",
        f"Generated {report['generated_at']}. **{'In progress; incomplete model coverage' if partial else 'Inference finished; coverage exceptions retained'}**.",
        "",
        (
            "This measures agreement with 1,500 independent assistant assessments, not accuracy against human ratings. "
            "The original 500 references were reused; 500 fresh examples and 500 confirmation references were assessed blind and frozen before admission to inference. "
            "The confirmation cohort was not used here to fit weights, prompts or score corrections."
        ),
        "",
        (
            "The canonical score remains 37.5% Luna + 37.5% DeepSeek + 25% Liquid, separately for technical quality and aesthetics. "
            "All three valid numeric components are required. Calculations use native precision; only presentation is rounded. "
            "There is no replacement score for a failed or unscorable observation."
        ),
        "",
        confirmation_summary(report),
        "",
        "**Coverage across the full corpus**",
        "",
    ]
    lines += table(
        ("Assessor", "Expected", "Numeric T/A pairs", "Observation statuses"),
        [
            (
                NAMES[model],
                values["expected"],
                values["numeric_pairs"],
                ", ".join(f"{key}: {value}" for key, value in values["status_counts"].items()),
            )
            for model, values in report["coverage"].items()
        ],
    )
    lines += [
        "",
        (
            "MAE is absolute error on the 1–5 scale; lower is better. Bias is prediction minus reference. "
            "Spearman ρ compares ranking, with average ranks for ties; a constant score series has no defined correlation. "
            "Coverage is reported separately for each outcome. Different coverage can change apparent rankings."
        ),
        "",
    ]
    for group_name, (title, _, _) in GROUPS.items():
        group = report["groups"][group_name]
        lines += [f"## {title} ({group['expected']} images)", ""]
        lines += table(
            ("Assessor", "T n", "T MAE", "T bias", "T ρ", "A n", "A MAE", "A bias", "A ρ"),
            [
                (
                    NAMES[model],
                    values["technical"]["n"],
                    fmt(values["technical"]["mae"], digits=3),
                    fmt(values["technical"]["bias"], signed=True),
                    fmt(values["technical"]["spearman"]),
                    values["aesthetic"]["n"],
                    fmt(values["aesthetic"]["mae"], digits=3),
                    fmt(values["aesthetic"]["bias"], signed=True),
                    fmt(values["aesthetic"]["spearman"]),
                )
                for model, values in group["per_model"].items()
            ],
        )
        lines += [
            "",
            (
                "**Same-image comparisons.** Each Δ compares the ensemble with that assessor on their shared numeric outcomes. "
                "Negative Δ means lower ensemble MAE; different assessor rows can have different shared samples."
            ),
            "",
        ]
        lines += table(
            ("Compared with", "Shared T n", "Δ T MAE", "Shared A n", "Δ A MAE"),
            [
                (
                    NAMES[model],
                    values["technical"]["n"],
                    fmt(values["technical"]["ensemble_minus_model_mae"], signed=True, digits=3),
                    values["aesthetic"]["n"],
                    fmt(values["aesthetic"]["ensemble_minus_model_mae"], signed=True, digits=3),
                )
                for model, values in group["paired_ensemble"].items()
            ],
        )
        for cohort_name, title_text in (
            ("all_six", "All six assessors"),
            ("four_api_and_ensemble", "Four API models and ensemble"),
        ):
            common = group["common_coverage"][cohort_name]
            lines += [
                "",
                f"{title_text}, common images only: technical n={common['technical']['n']}; aesthetic n={common['aesthetic']['n']}.",
                "",
            ]
            lines += table(
                ("Assessor", "Common T MAE", "Common A MAE"),
                [
                    (
                        NAMES[model],
                        fmt(values["mae"], digits=3),
                        fmt(common["aesthetic"]["models"][model]["mae"], digits=3),
                    )
                    for model, values in common["technical"]["models"].items()
                ],
            )
        lines += [
            "",
            "**Technical tail errors.** Denominators are reference-poor (≤2) or reference-strong (≥4) images with a numeric model outcome.",
            "",
        ]
        lines += table(
            ("Assessor", "Poor rated ≥3", "Strong rated <3"),
            [
                (
                    NAMES[model],
                    fraction(
                        values["technical_tails"]["poor_predicted_at_least_3"],
                        values["technical_tails"]["poor_reference_n"],
                    ),
                    fraction(
                        values["technical_tails"]["strong_predicted_below_3"],
                        values["technical_tails"]["strong_reference_n"],
                    ),
                )
                for model, values in group["per_model"].items()
            ],
        )
        lines += [
            "",
            (
                "**Photography eligibility decisions.** Only explicit recorded API floor decisions are compared. "
                "False-floor and missed-exclusion fractions use known decisions within their reference class. "
                "Unknown or failed decisions are counted separately, not treated as correct. "
                "Uncertain reference eligibility is excluded from both class denominators. "
                "No gate decision is inferred for Lite or the ensemble from a low score."
            ),
            "",
        ]
        lines += table(
            (
                "Model",
                "Eligible refs",
                "False floors / known",
                "Eligible unknown",
                "Excluded refs",
                "Misses / known",
                "Excluded unknown",
                "Uncertain refs",
                "All unknown decisions",
            ),
            [
                (
                    NAMES[model],
                    values["reference_eligible_n"],
                    fraction(values["eligible_false_floor_n"], values["eligible_classified_n"]),
                    values["eligible_unknown_n"],
                    values["reference_excluded_n"],
                    fraction(values["excluded_miss_n"], values["excluded_classified_n"]),
                    values["excluded_unknown_n"],
                    values["reference_uncertain_or_missing_n"],
                    values["classification_unknown_n"],
                )
                for model, values in group["scope_confusion"].items()
            ],
        )
        softness = group["softness_diagnostic"]
        lines += [
            "",
            (
                f"**Sharpness diagnostic.** Among {softness['eligible_n']} reference-eligible photographs, the within-cohort 25th percentile "
                f"of `sharpness_peak` is {fmt(softness['threshold'])}. This is a descriptive review cue, not an absolute focus threshold or score correction."
            ),
            "",
        ]
        lines += table(
            ("Assessor", "Soft n", "Soft T bias", "Other n", "Other T bias"),
            [
                (
                    NAMES[model],
                    values["soft"]["n"],
                    fmt(values["soft"]["bias"], signed=True),
                    values["other"]["n"],
                    fmt(values["other"]["bias"], signed=True),
                )
                for model, values in softness["models"].items()
            ],
        )
        lines += [""]
    lines += [
        "## Interpretation and next steps",
        "",
        (
            "Keep the accepted ensemble fixed through this rehearsal. Use the separate confirmation results to judge whether its agreement generalizes; "
            "a smaller assistant-reference error alone does not establish human preference or label suitability. "
            "Review poor-image overrating, high-quality underrating and missing outcomes before enabling training masks. "
            "Any subsequent weight or prompt change needs a new untouched confirmation set."
        ),
        "",
        (
            "Image statistics remain separate debugging metadata. The sharpness split describes where errors concentrate and does not justify overwriting labels. "
            "`focus_area` is relative to each image’s own peak detail; histogram entropy is the `luma_entropy` alias. "
            "No statistics were passed to the graders in this run."
        ),
        "",
        "## Request latency",
        "",
        (
            "These are selected-attempt end-to-end HTTP durations through complete response receipt, including completed HTTP errors. "
            "Transport failures and unmeasured requests are excluded from these quantiles. Retries are retained in cost accounting. "
            "Reference reviews and the derived ensemble have no HTTP latency. Lite reuses 4,000 compatible historical observations; "
            "its fresh-request columns avoid mixing those historical timings with this run. "
            "Luna and DeepSeek used 16 workers; Liquid and Jev increased from four to eight. "
            "Lite increased from two to three, with a separate 100-image trial at six overlapping requests. "
            "These observed network and workload timings are not an isolated model-speed benchmark."
        ),
        "",
    ]
    lines += table(
        ("Model", "Measured n", "Median s", "p95 s", "Reused n", "Fresh n", "Fresh median s", "Fresh p95 s"),
        [
            (
                NAMES[model],
                values["complete_response_n"],
                fmt(values["median_seconds"]),
                fmt(values["p95_seconds"]),
                values["reused_n"],
                values["fresh_complete_response_n"],
                fmt(values["fresh_median_seconds"]),
                fmt(values["fresh_p95_seconds"]),
            )
            for model, values in report["latency"].items()
        ],
    )
    lines += [
        "",
        "## Accounting",
        "",
        (
            "Selected receipts describe the latest retained observation per image. All-attempt totals include retries and the bounded missing-response recovery pass. "
            "An unpriced attempt has an unknown charge; it is not assumed free. The known subtotal is therefore not a complete bill where unpriced attempts remain. "
            "Lite and Reference are $0.00 under the requested scoring-cost convention; paid Runpod infrastructure is accounted for separately. "
            "The ensemble adds no request or charge and is not double-counted."
        ),
        "",
        "This report and its companion JSON contain aggregate results only. Input file hashes are recorded in the JSON for reproducibility.",
        "",
        "**Total cost per model (USD)**",
        "",
    ]
    if report.get("infrastructure"):
        lines[-2:] = []
        lines += table(
            ("Infrastructure", "Posted USD", "Posted through", "Receipt scope"),
            [
                (
                    name,
                    dollars(value["posted_price_usd"]),
                    value["posted_through"] or "—",
                    value["completeness"],
                )
                for name, value in report["infrastructure"].items()
            ],
        )
        lines += [
            "",
            "Posted infrastructure amounts are separate from inference-row prices and are not estimates of a complete invoice unless the receipt says so.",
            "",
            "**Total cost per model (USD)**",
            "",
        ]
    cost_rows = []
    for model, values in report["cost_per_model"].items():
        if model in ("qrealign_lite", "reference"):
            known, total, unknown, attempts = "0.00", "0.00", 0, "—"
        else:
            known, total = values["known_price_usd"], values["total_price_usd"]
            unknown, attempts = values["unpriced_attempts"], values["attempts"]
        cost_rows.append(
            (
                NAMES[model],
                dollars(values["selected_known_price_usd"]),
                values["selected_unpriced_observations"],
                attempts,
                dollars(known),
                unknown,
                dollars(total),
            )
        )
    lines += table(
        (
            "Model",
            "Selected known subtotal",
            "Selected unpriced",
            "All attempts",
            "All-attempt known subtotal",
            "Unpriced attempts",
            "Total",
        ),
        cost_rows,
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-state", choices=("in-progress", "finished"), default="in-progress")
    parser.add_argument("--embedding-billing", type=Path)
    parser.add_argument("--lite-billing", type=Path)
    args = parser.parse_args()
    scores_path, summary_path = args.exports / "scores-long.jsonl", args.exports / "summary.json"
    input_paths = (scores_path, summary_path, args.stats, args.embedding_billing, args.lite_billing)
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths if path is not None}
    rows = [{key: value for key, value in row.items() if key in FIELDS} for row in jsonlines(scores_path)]
    summary = json.loads(summary_path.read_text())
    stats = {}
    for row in jsonlines(args.stats):
        if row["case_id"] in stats:
            raise ValueError("Duplicate statistics record")
        stats[row["case_id"]] = {
            "view_sha256": row["view_sha256"],
            "sharpness_peak": row["image_stats"]["sharpness_peak"],
        }
    report = build_report(rows, summary, stats, run_state=args.run_state)
    report["input_provenance"] = input_hashes
    report["infrastructure"] = {}
    for name, path in (("EmbeddingGemma 2", args.embedding_billing), ("Q-ReAlign Lite", args.lite_billing)):
        if path:
            report["infrastructure"][name] = infrastructure_receipt(path)
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError("Report inputs changed while being read; regenerate from a stable export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.write_text(render(report))
    print(
        json.dumps(
            {
                "report": str(args.output),
                "state": args.run_state,
                "images": report["images"],
                "references": report["benchmark_images"],
            }
        )
    )


if __name__ == "__main__":
    main()
