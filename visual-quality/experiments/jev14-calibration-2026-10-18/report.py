"""Report a frozen calibration rule against family-disjoint validation responses.

This command only reads local evidence. It never calls a model or selects a
rule from validation data. All cases without an AI flag remain for review.
"""

import argparse
import csv
import fcntl
import hashlib
import html
import importlib.util
import io
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/jev14-calibration-2026-10-18"
SPLITS = ("calibration", "validation")
PREVALENCES = (0.01, 0.05, 0.10, 0.50)


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"jev_calibration_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def operations(records, expected):
    costs = [row.get("effective_cost_usd") for row in records]
    durations = [row.get("seconds") for row in records]
    known_costs = [cost for cost in costs if finite_number(cost) and cost >= 0]
    measured_times = [duration for duration in durations if finite_number(duration) and duration >= 0]
    usages = [row.get("usage") or {} for row in records]
    tokens = {}
    for key in ("input_tokens", "output_tokens"):
        values = [usage.get(key) for usage in usages]
        known = [value for value in values if type(value) is int and value >= 0]
        tokens[key] = {
            "reported_total": sum(known),
            "known_calls": len(known),
            "unknown_calls": len(values) - len(known),
        }
    return {
        "expected_calls": expected,
        "attempted_calls": len(records),
        "unattempted_calls": expected - len(records),
        "known_cost_usd": math.fsum(known_costs),
        "cost_known_calls": len(known_costs),
        "cost_unknown_calls": len(costs) - len(known_costs),
        "latency_measured_calls": len(measured_times),
        "latency_p50_seconds": percentile(measured_times, 0.50),
        "latency_p95_seconds": percentile(measured_times, 0.95),
        "tokens": tokens,
        "resolved_models": dict(Counter(str(row.get("resolved_model")) for row in records)),
    }


def precision_scenarios(result):
    """Apply Bayes' rule under stated prevalence assumptions, not prevalence estimates."""
    recall, fpr, upper = (result[key] for key in ("recall", "fpr", "fpr_upper_95"))

    def precision(prior, false_rate):
        if recall is None or false_rate is None:
            return None
        denominator = recall * prior + false_rate * (1 - prior)
        return recall * prior / denominator if denominator else None

    return [
        {
            "assumed_ai_prevalence": prior,
            "precision_using_observed_rates": precision(prior, fpr),
            "precision_using_fpr_upper_95_and_observed_recall": precision(prior, upper),
        }
        for prior in PREVALENCES
    ]


def metric_rows(cases, records, expected_model):
    """Retain missing and failed cases in the denominator as unresolved."""
    indexed = {record["row_id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate response row_id")
    planned = {case["row_id"] for case in cases}
    if len(planned) != len(cases) or set(indexed) - planned:
        raise ValueError("Duplicate manifest case or unplanned response")
    output = []
    for case in cases:
        record = indexed.get(case["row_id"], {})
        prediction = record.get("normalized") or {}
        valid = (
            bool(record)
            and record.get("status") == 200
            and not any(
                record.get(key)
                for key in ("validation_error", "transport_error", "cost_validation_error", "error")
            )
            and isinstance(record.get("normalized"), dict)
            and record.get("resolved_model") == expected_model
            and finite_number(record.get("effective_cost_usd"))
        )
        output.append(
            {
                **case,
                "attempted": bool(record),
                "valid": valid,
                "ai_probability": prediction.get("ai_probability"),
                "uncertainty": prediction.get("uncertainty"),
                "uncertainty_native_confidence": prediction.get("uncertainty_native_confidence"),
                "uncertainty_probabilities": prediction.get("uncertainty_probabilities"),
                "medium": prediction.get("medium"),
                "medium_native_confidence": prediction.get("medium_native_confidence"),
                "status": record.get("status"),
                "validation_error": record.get("validation_error")
                or record.get("transport_error")
                or record.get("cost_validation_error")
                or record.get("error"),
                "resolved_model": record.get("resolved_model"),
                "seconds": record.get("seconds"),
                "effective_cost_usd": record.get("effective_cost_usd"),
            }
        )
    return output


def compact_cases(rows, rule, metrics):
    output = []
    for row in rows:
        flagged = bool(metrics.evaluate([row], rule)["counts"]["flagged"])
        output.append(
            {
                key: row.get(key)
                for key in (
                    "row_id",
                    "split",
                    "family_id",
                    "generator",
                    "expected_ai",
                    "image",
                    "delivered_sha256",
                    "attempted",
                    "valid",
                    "status",
                    "resolved_model",
                    "ai_probability",
                    "uncertainty",
                    "uncertainty_native_confidence",
                    "medium",
                    "medium_native_confidence",
                    "seconds",
                    "effective_cost_usd",
                    "validation_error",
                )
            }
            | {"selected_rule_flag": flagged, "routing": "flag_candidate" if flagged else "review"}
        )
    return output


def curve_rows(candidates):
    return [
        {
            "threshold": result["rule"]["threshold"],
            "gate": result["rule"]["gate"],
            **result["counts"],
            **{key: result[key] for key in ("fpr", "fpr_upper_95", "recall", "review_share")},
            "recall_lower_95": result["recall_ci_95"][0] if result["recall_ci_95"] else None,
            "recall_upper_95": result["recall_ci_95"][1] if result["recall_ci_95"] else None,
        }
        for result in candidates
    ]


def calibration_svg(curve, selected):
    """Plot empirical calibration tradeoffs, never a validation threshold sweep."""
    width, height = 880, 590
    left, right, top, bottom = 90, 830, 110, 480
    points = [row for row in curve if finite_number(row["fpr"]) and finite_number(row["recall"])]
    maximum = max((100 * row["fpr"] for row in points), default=0)
    x_max = max(2, math.ceil(maximum / 5) * 5)

    def x(value):
        return left + (right - left) * 100 * value / x_max

    def y(value):
        return bottom - (bottom - top) * value

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Jev 1.14 calibration threshold tradeoffs</title>',
        '<desc id="desc">Empirical false-positive rate versus AI recall for the predeclared probability and uncertainty rules. Calibration data only. The vertical line is the one percent false-positive target.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#17212b">',
        '<text x="90" y="32" font-size="21" font-weight="bold">Calibration threshold tradeoffs</text>',
        '<text x="90" y="57" font-size="14">Calibration data only; observed rates, without confidence bounds</text>',
    ]
    for tick in range(6):
        value = tick / 5
        yy = y(value)
        lines += [
            f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left - 12}" y="{yy + 5:.2f}" font-size="13" text-anchor="end">{value * 100:.0f}</text>',
        ]
        xx = left + (right - left) * value
        lines += [
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#f1f3f5"/>',
            f'<text x="{xx:.2f}" y="{bottom + 24}" font-size="13" text-anchor="middle">{x_max * value:g}</text>',
        ]
    target = x(0.01)
    lines += [
        f'<line x1="{target:.2f}" y1="{top}" x2="{target:.2f}" y2="{bottom}" stroke="#64748b" stroke-dasharray="5 5"/>',
        f'<text x="{target + 5:.2f}" y="{top + 16}" font-size="12" fill="#475569">1% target</text>',
    ]
    for index, (gate, color, label) in enumerate(
        (
            ("none", "#1769aa", "Probability only"),
            ("not_high", "#c85d00", "Exclude high uncertainty"),
        )
    ):
        group = [row for row in points if row["gate"] == gate]
        coordinates = " ".join(f"{x(row['fpr']):.2f},{y(row['recall']):.2f}" for row in group)
        dash = ' stroke-dasharray="7 4"' if gate == "not_high" else ""
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>'
        )
        legend_x = 90 + index * 270
        lines += [
            f'<line x1="{legend_x}" y1="83" x2="{legend_x + 30}" y2="83" stroke="{color}" stroke-width="2.5"{dash}/>',
            f'<text x="{legend_x + 38}" y="88" font-size="14">{label}</text>',
        ]
    if selected is not None:
        matching = [
            row
            for row in points
            if row["threshold"] == selected["threshold"] and row["gate"] == selected["gate"]
        ]
        if len(matching) != 1:
            raise ValueError("Selected rule is missing from the calibration curve")
        row = matching[0]
        lines.append(
            f'<circle cx="{x(row["fpr"]):.2f}" cy="{y(row["recall"]):.2f}" r="6" fill="white" stroke="#111827" stroke-width="2.5"><title>{html.escape(rule_name(selected))}</title></circle>'
        )
        caption = f"Selected: {rule_name(selected)}. Recall {100 * row['recall']:.1f}%; FPR {100 * row['fpr']:.2f}%."
    else:
        caption = "No calibration rule selected. All cases remain unresolved for review."
    lines += [
        f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#374151"/>',
        f'<text x="{(left + right) / 2}" y="535" font-size="15" text-anchor="middle">Real-photo false-positive rate (%)</text>',
        '<text transform="translate(25 295) rotate(-90)" font-size="15" text-anchor="middle">AI recall (%)</text>',
        f'<text x="90" y="568" font-size="12">{html.escape(caption)}</text>',
        "</g></svg>",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.write_text(stream.getvalue())


def percent(value):
    return f"{100 * value:.2f}%" if finite_number(value) else "—"


def decimal(value, places=4):
    return f"{value:.{places}f}" if finite_number(value) else "—"


def rule_name(rule):
    if rule is None:
        return "No rule selected; all cases require review"
    gate = "; origin uncertainty must be low or medium" if rule["gate"] == "not_high" else ""
    return f"AI probability >= {rule['threshold']:.2f}{gate}"


def metric_table(result, label):
    counts = result["counts"]
    interval = result["recall_ci_95"]
    ci = f"{percent(interval[0])}–{percent(interval[1])}" if interval else "—"
    return f"| {label} | {counts['fp']}/{counts['real']} | {percent(result['fpr_upper_95'])} | {counts['tp']}/{counts['ai']} | {ci} | {counts['review']}/{counts['total']} | {counts['unresolved']} |"


def source_allocation(protocol):
    parts = []
    for split in SPLITS:
        source = protocol["splits"][split]
        label = "Calibration" if split == "calibration" else "held-out validation"
        parts.append(
            f"{label} uses the official `{source['official_split']}` split ({source['real']} real and {source['ai']} generated images)"
        )
    return (
        "; ".join(parts)
        + ". These are different caption families from one dataset, not a cross-source validation."
    )


def markdown(summary):
    selected = summary["selected_rule"]
    verdict = "PASS" if summary["validation_gate"]["passed"] else "FAIL"
    if summary["status"] != "complete":
        verdict = "PENDING: incomplete evidence"
    lines = [
        "# Jev 1.14 calibration and held-out validation",
        "",
        f"Status: **{summary['status']}**. Validation gate: **{verdict}**. Generated: {summary['generated_at']}.",
        "",
        f"Frozen rule: **{rule_name(selected)}**. The rule was selected from calibration responses only. Validation responses did not alter it.",
        "",
        "Flags enter a candidate queue. The Review column counts all remaining cases, which stay unresolved. Both queues require review; no case is accepted as a verified camera photo. This study does not authorize automatic rejection or deletion.",
        "",
        "## Rule performance",
        "",
        "The gate requires all validation cases to have valid responses, a one-sided exact 95% false-positive upper bound at or below 1%, and observed AI recall of at least 70%. Recall intervals are two-sided exact 95% intervals. The recall floor uses the observed rate, not its lower confidence bound.",
        "",
    ]
    for split in SPLITS:
        result = summary["splits"][split]
        lines += [
            f"### {split.capitalize()}",
            "",
            "| Rule | Real photos falsely flagged | FPR upper 95% | AI images flagged | Recall 95% interval | Review | Unresolved |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
        lines.append(metric_table(result["selected"], "Frozen selection"))
        lines.extend(
            metric_table(value, f"Fixed {threshold}") for threshold, value in result["comparators"].items()
        )
        lines += [""]
    lines += [
        "![Calibration false-positive rate versus AI recall](calibration-curve.svg)",
        "",
        "The plot shows calibration results only. The circle marks the frozen rule; the dashed vertical line marks the empirical 1% selection target. Validation uses the stricter exact upper-bound gate.",
        "",
    ]
    lines += [
        "Validation checks: "
        + "; ".join(
            f"`{name}`: {'pass' if passed else 'fail'}"
            for name, passed in summary["validation_gate"]["checks"].items()
        )
        + ".",
        "",
        "## Precision at different AI prevalence rates",
        "",
        "These are scenarios using validation recall and false-positive rates. The dataset's balanced mixture is not an estimate of production prevalence. The second precision column substitutes the exact FPR upper bound while keeping observed recall; it is a sensitivity check, not a precision confidence interval. Zero observed false flags must not be read as perfect precision.",
        "",
        "| Assumed AI prevalence | Precision from observed rates | Precision with FPR upper bound |",
        "| ---: | ---: | ---: |",
    ]
    lines += [
        f"| {percent(item['assumed_ai_prevalence'])} | {percent(item['precision_using_observed_rates'])} | {percent(item['precision_using_fpr_upper_95_and_observed_recall'])} |"
        for item in summary["precision_scenarios"]
    ]
    lines += [
        "",
        "## Validation groups",
        "",
        "Group results are descriptive. They do not establish a separate pass for each generator. Real-photo rows have no AI recall; generated-image rows have no real-photo false-positive rate.",
        "",
        "| Generator/source class | Valid/total | AI flagged/total | Real false flags/total | Review |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["splits"]["validation"]["groups"].items():
        count = result["counts"]
        lines.append(
            f"| {name} | {count['valid']}/{count['total']} | {count['tp']}/{count['ai']} | {count['fp']}/{count['real']} | {count['review']} |"
        )
    lines += [
        "",
        "## Cost and latency",
        "",
        "| Split | Calls/expected | Reported cost (USD) | Calls without known cost | Median seconds | 95th percentile seconds | Input/output tokens reported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in (*SPLITS, "total"):
        result = summary["operations"][split]
        tokens = result["tokens"]
        lines.append(
            f"| {split} | {result['attempted_calls']}/{result['expected_calls']} | {decimal(result['known_cost_usd'], 6)} | {result['cost_unknown_calls']} | {decimal(result['latency_p50_seconds'], 3)} | {decimal(result['latency_p95_seconds'], 3)} | {tokens['input_tokens']['reported_total']}/{tokens['output_tokens']['reported_total']} |"
        )
    lines += [
        "",
        "## Scope and limits",
        "",
        f"The source is [MS COCOAI](https://huggingface.co/datasets/{summary['protocol']['dataset']}/tree/{summary['protocol']['dataset_revision']}), from the pinned Hugging Face revision recorded in the protocol and manifest. {source_allocation(summary['protocol'])} Each sampled caption family contributes one real image and one generated image. Families are kept separate between the two study splits.",
        "",
        "Pre-inference source correction: " + summary["protocol"]["source_split_correction"],
        "",
        "The [MS COCOAI paper](https://arxiv.org/html/2601.00553v1) links the dataset. The cohort covers five generators: SD2.1, SDXL, SD3, DALL-E3, and Midjourney-v6. One source, its generated-image mixture, and the delivered image resolution limit generalisation. This result does not establish performance on native camera files, other sources, newer generators, or production data. The repository card does not specify a dataset license. Paper licensing does not license the images; [COCO terms](https://cocodataset.org/dataset/termsofuse.htm) distinguish annotation licensing from the photos' source terms. Images remain in the ignored local data directory and are not redistributed with the report.",
        "",
        "Native confidence for an uncertainty choice is not a separate probability that the origin decision is correct. This procedure selects a decision threshold; it does not fit calibrated output probabilities.",
        "",
        "The cases are independent of our earlier prompt and benchmark cohorts. Overlap with Jev pretraining data is unknown, so this is not a claim that the model has never seen these public images. Calibration performance includes selection bias. Validation supplies the gate result. Exact class-level binomial bounds do not cover source shift or all forms of dependence; the paired family design must remain visible when interpreting aggregate results.",
        "",
        "## Evidence",
        "",
        "- [Machine-readable summary](summary.json), including ledger and protocol hashes.",
        "- [Per-case results](cases.csv). Missing and failed cases remain in the denominators.",
        "- [Calibration threshold curve](calibration-curve.csv). No validation threshold search was used.",
        "- [Frozen selection](../selection.json), [protocol](../protocol.json), and [manifest](../manifest.json).",
        "- [Acquisition audit](../acquisition-audit.json), including the acquired representation and source limits.",
        "",
    ]
    limitations = summary["acquisition_audit"].get("limitations", [])
    if limitations:
        lines += (
            ["Acquisition limits recorded before inference:", ""]
            + [f"- {item}" for item in limitations]
            + [""]
        )
    return "\n".join(lines)


def load_verified_evidence():
    """The runner must verify hashes and response contracts before reporting."""
    runner = load_module("runner")
    paths = {
        name: HERE / name
        for name in (
            "protocol.json",
            "manifest.json",
            "prompt.json",
            "selection.json",
            "selection-lock.json",
            "acquisition-audit.json",
            "report.py",
            "metrics.py",
        )
    }
    paths.update({f"{split}_ledger": DATA / f"{split}.jsonl" for split in SPLITS})

    def snapshot():
        return {key: path.read_bytes() if path.exists() else b"" for key, path in paths.items()}

    with (DATA / ".run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if list((DATA / "pending").glob("*.json")):
            raise ValueError("Unreconciled request intent; report after reconciliation")
        before = snapshot()
        protocol, cases, _, records = runner.load_study(require_selection=True)
        selection = read_json(HERE / "selection.json")
        after = snapshot()
        if before != after:
            raise ValueError("Evidence changed during reporting")
        for split in SPLITS:
            if len(records[split]) != sum(
                bool(line.strip()) for line in before[f"{split}_ledger"].splitlines()
            ):
                raise ValueError("Verified responses do not cover the ledger snapshot")
        return (
            protocol,
            cases,
            records,
            selection,
            {key: digest(raw) for key, raw in before.items()},
            read_json(HERE / "acquisition-audit.json"),
        )


def build_report():
    protocol, cases, records, selection, hashes, acquisition = load_verified_evidence()
    metrics = load_module("metrics")
    rule = selection["rule"]
    rows = {
        split: metric_rows(
            [case for case in cases if case["split"] == split], records[split], protocol["expected_revision"]
        )
        for split in SPLITS
    }
    if metrics.select_rule(rows["calibration"]) != rule:
        raise ValueError("Frozen selection differs from the calibration-only selection")
    if selection.get("calibration_metrics") != (
        metrics.evaluate(rows["calibration"], rule) if rule is not None else None
    ):
        raise ValueError("Frozen calibration metrics differ from verified responses")
    gate = metrics.validation_gate(rows["validation"], rule)
    complete = all(row["valid"] for split in SPLITS for row in rows[split]) and all(rows.values())
    if not complete:
        gate["passed"] = False
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if complete else "partial",
        "selected_rule": rule,
        "selection_policy": "Calibration only; frozen before validation inference; no validation retuning",
        "evidence_sha256": hashes,
        "protocol": protocol,
        "acquisition_audit": acquisition,
        "validation_gate": gate,
        "splits": {},
        "operations": {},
        "precision_scenarios": precision_scenarios(gate["metrics"]),
    }
    for split in SPLITS:
        summary["splits"][split] = {
            "selected": metrics.evaluate(rows[split], rule),
            "comparators": metrics.comparators(rows[split]),
            "groups": metrics.grouped_metrics(rows[split], rule),
        }
        summary["operations"][split] = operations(records[split], len(rows[split]))
    summary["operations"]["total"] = operations(
        [record for split in SPLITS for record in records[split]], len(cases)
    )
    all_rows = [row for split in SPLITS for row in rows[split]]
    return (
        summary,
        compact_cases(all_rows, rule, metrics),
        curve_rows(metrics.candidate_results(rows["calibration"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    summary, cases, curve = build_report()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "cases.csv", cases)
    write_csv(args.output / "calibration-curve.csv", curve)
    (args.output / "calibration-curve.svg").write_text(calibration_svg(curve, summary["selected_rule"]))
    (args.output / "README.md").write_text(markdown(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "validation_passed": summary["validation_gate"]["passed"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
