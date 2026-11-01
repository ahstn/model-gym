"""Render the rehearsal report from audited analysis artifacts, without new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("luna", "deepseek", "jev", "liquid", "qrealign_lite", "luna_deepseek_mean")
NAMES = {
    "luna": "GPT-6 Luna",
    "deepseek": "DeepSeek V4.1 Flash 1017",
    "jev": "Jev 1.14",
    "liquid": "Liquid d1.1",
    "qrealign_lite": "Q-ReAlign Lite",
    "reference": "Reference",
    "luna_deepseek_mean": "Luna + DeepSeek mean",
}
ARMS = {
    "baseline": "Baseline",
    "own_stats": "Own stats",
    "swapped_stats": "Swapped stats",
    "reference": "Reference",
}
COMPLETE = {"valid", "partial", "invalid"}


def number(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def cost(value: object) -> str:
    if value is None:
        return "Unknown"
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid price in analysis artifact")
    return "$0.00" if amount == 0 else "$" + format(amount, "f")


def table(headers: tuple[str, ...], rows: list[list[object]]) -> str:
    def cells(values: list | tuple) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"

    return "\n".join([cells(headers), cells(["---"] * len(headers)), *[cells(row) for row in rows]])


def with_n(value: dict, field: str = "mae") -> str:
    return f"{number(value[field])} ({value['n']})"


def interval(value: dict) -> str:
    bounds = value.get("ci95")
    return "Unavailable" if bounds is None else f"[{number(bounds[0])}, {number(bounds[1])}]"


def target(value: bool | None) -> str:
    return "Met" if value is True else "Not met" if value is False else "Not assessable"


def validate_counts(analysis: dict, manifest: dict) -> None:
    cases = manifest["cases"]
    benchmark = sum(case["benchmark_partition"] != "rehearsal" for case in cases)
    expected = {(model, "baseline"): len(cases) for model in MODELS if model != "luna_deepseek_mean"}
    expected.update(
        {(model, arm): benchmark for model in ("luna", "deepseek") for arm in ("own_stats", "swapped_stats")}
    )
    expected[("reference", "reference")] = benchmark
    actual = {(row["model"], row["arm"]): row for row in analysis["operations"]["model_arms"]}
    if len(actual) != len(analysis["operations"]["model_arms"]) or set(actual) != set(expected):
        raise ValueError("Model/arm completion table has missing or duplicate groups")
    for key, count in expected.items():
        row = actual[key]
        if row["expected_images"] != count or sum(row["status_counts"].values()) != count:
            raise ValueError("Completion counts disagree with the corpus")
        if row["rows_with_attempts"] > count or row["rows_with_attempts"] > row["attempts"]:
            raise ValueError("Attempt counts cannot support the claimed completed rows")
    if analysis["long_rows"] != sum(expected.values()) + len(cases) + 2 * benchmark:
        raise ValueError("Long-table coverage disagrees with expected model/reference/derived rows")
    totals = analysis["operations"]["model_totals"]
    if len(totals) != 6 or {row["model"] for row in totals} != {*MODELS[:-1], "reference"}:
        raise ValueError("Cost table must contain each actual model and Reference exactly once")
    for total in totals:
        groups = [row for row in actual.values() if row["model"] == total["model"]]
        if (
            total["attempts"] != sum(row["attempts"] for row in groups)
            or total["unpriced_attempts"] != sum(row["unpriced_attempts"] for row in groups)
            or Decimal(total["known_price_usd"])
            != sum((Decimal(row["known_price_usd"]) for row in groups), Decimal(0))
        ):
            raise ValueError("Model cost total disagrees with all-arm attempt accounting")
        if total["unpriced_attempts"] and total["total_price_usd"] is not None:
            raise ValueError("Unknown attempt prices must not be reported as a complete total")


def accuracy_section(analysis: dict, panel: str, count: int) -> str:
    lookup = {
        (row["model"], row["arm"]): row for row in analysis["accuracy"]["panels"] if row["panel"] == panel
    }
    rows, fair_rows = [], []
    for model in MODELS:
        row = lookup.get((model, "baseline"))
        if row is None:
            rows.append([NAMES[model], *(["Unavailable"] * 7)])
            continue
        technical, aesthetic = row["technical_score"], row["aesthetic_score"]
        fair_technical, fair_aesthetic = (
            row[task + "_all_five_baselines_common"] for task in ("technical_score", "aesthetic_score")
        )
        fair_rows.append(
            [
                NAMES[model],
                with_n(fair_technical),
                with_n(fair_aesthetic),
                number(fair_technical["bias"]),
                number(fair_aesthetic["bias"]),
                f"{number(fair_technical['spearman'])} / {number(fair_aesthetic['spearman'])}",
            ]
        )
        rows.append(
            [
                NAMES[model],
                with_n(technical),
                with_n(aesthetic),
                number(technical["bias"]),
                number(aesthetic["bias"]),
                f"{number(technical['spearman'])} / {number(aesthetic['spearman'])}",
                with_n(row["combined_common_tasks"]),
                with_n(row["all_five_baselines_common"]),
            ]
        )
    title = "Representative benchmark" if panel == "representative" else "Challenge benchmark"
    overall = f"## {title} ({count} images)\n\n" + table(
        (
            "Model",
            "Tech MAE (n)",
            "Aest MAE (n)",
            "Tech bias",
            "Aest bias",
            "ρ tech / aest",
            "Common-task MAE (n)",
            "Five-baseline common MAE (n)",
        ),
        rows,
    )
    return (
        overall
        + "\n\nOn the task-specific all-five common subsets, an invalid aesthetic score does not remove an otherwise valid technical score (and vice versa):\n\n"
        + table(
            (
                "Model",
                "Common tech MAE (n)",
                "Common aest MAE (n)",
                "Tech bias",
                "Aest bias",
                "ρ tech / aest",
            ),
            fair_rows,
        )
    )


def treatment_section(analysis: dict) -> str:
    selected = [
        row
        for row in analysis["treatments"]
        if row["panel"] == "representative" and row["model"] == "luna_deepseek_mean"
    ]
    rows = []
    for row in selected:
        technical, aesthetic = row["technical_score"], row["aesthetic_score"]
        rows.append(
            [
                ARMS[row["arm"]],
                technical["n_images"],
                technical["n_families"],
                number(technical["estimate"]),
                interval(technical),
                number(aesthetic["estimate"]),
                interval(aesthetic),
                number(aesthetic["upper_one_sided95"]),
            ]
        )
    parts = [
        "## Primary stats comparison",
        "The primary comparison is own stats versus baseline for the fixed 50/50 Luna/DeepSeek mean on the representative panel. Negative differences mean lower MAE against the assistant references. Each task uses its own valid paired cases; an unrelated challenger's missing label does not remove a case.",
        table(
            (
                "Treatment",
                "Tech paired n",
                "Families",
                "Δ tech MAE",
                "95% family CI",
                "Δ aest MAE",
                "95% family CI",
                "Aest one-sided 95% upper",
            ),
            rows,
        ),
    ]
    primary = next((row for row in selected if row.get("primary")), None)
    if primary:
        coverage = [
            [
                ARMS[row["arm"]],
                name,
                row[task]["expected_panel_images"],
                row[task]["numeric_reference_cases"],
                row[task]["paired_cases"],
                number(row[task]["paired_coverage"]),
                row[task]["baseline_missing_or_invalid_scores"],
                row[task]["treatment_missing_or_invalid_scores"],
            ]
            for row in selected
            for task, name in (("technical_score", "Technical"), ("aesthetic_score", "Aesthetic"))
        ]
        parts.append(
            table(
                (
                    "Treatment",
                    "Task",
                    "Expected",
                    "Numeric reference",
                    "Paired",
                    "Coverage fraction",
                    "Baseline unusable",
                    "Treatment unusable",
                ),
                coverage,
            )
        )
        parts.append(
            f"Coverage and missingness gate: **{target(primary['comparison_decision_eligible'])}**. Each task requires at least 95% paired coverage and no increase in nonnumeric or invalid scores versus baseline. Incomplete comparisons retain their numerical estimates for inspection but cannot support advancement."
        )
        technical = target(primary["technical_practical_and_interval_target_met"])
        aesthetic = target(primary["aesthetic_one_sided95_noninferiority_met"])
        parts.append(
            f"Numerical technical target: **{technical}** (MAE reduction ≥0.05 and two-sided 95% upper bound below zero). Numerical aesthetic noninferiority: **{aesthetic}** (one-sided 95% upper bound ≤+0.03). These numerical tests are separate from the coverage gate. Advancement remains unapproved pending the independent protected-slice review."
        )
        repeat_counts = {
            row[task]["bootstrap_repeats"]
            for row in selected
            for task in ("technical_score", "aesthetic_score")
        }
        seeds = {row[task]["seed"] for row in selected for task in ("technical_score", "aesthetic_score")}
        parts.append(
            f"Bootstrap resamples whole source families: repeats {', '.join(map(str, sorted(repeat_counts)))}; seed(s) {', '.join(map(str, sorted(seeds)))}. The intervals are conditional on assistant labels. Swapped stats are an exploratory sensitivity check; sensitivity does not prove the image was ignored. Individual-model, challenge and development contrasts remain secondary."
        )
    else:
        parts.append(
            "The primary contrast is unavailable in this analysis artifact; no treatment claim is made."
        )
    return "\n\n".join(parts)


def scope_section(analysis: dict) -> str:
    rows = []
    for panel in ("representative", "challenge"):
        for model in MODELS:
            eligible = next(
                (
                    row
                    for row in analysis["scope"]
                    if row["model"] == model
                    and row["arm"] == "baseline"
                    and row["panel"] == panel
                    and row["reference_eligibility"] == "eligible"
                ),
                None,
            )
            excluded = next(
                (
                    row
                    for row in analysis["scope"]
                    if row["model"] == model
                    and row["arm"] == "baseline"
                    and row["panel"] == panel
                    and row["reference_eligibility"] == "excluded"
                ),
                None,
            )
            rows.append(
                [
                    panel.title(),
                    NAMES[model],
                    f"{eligible['floor_declared']} / {eligible['n']}" if eligible else "—",
                    f"{excluded['either_score_above_1']} / {excluded['n']}" if excluded else "—",
                    f"{excluded['both_numeric']} / {excluded['n']}" if excluded else "—",
                ]
            )
    return "\n\n".join(
        [
            "## Scope diagnostics",
            "The reference photography-scope rule floors clearly excluded content at 1.00 for both dimensions. Declared floors on reference-eligible images and scores above 1.00 on reference-excluded images are shown separately. Missing scores are not evidence that the floor was respected. Lite retains its native outputs and has no added eligibility floor.",
            table(
                (
                    "Panel",
                    "Model",
                    "Declared floor / eligible",
                    "Either >1 / excluded",
                    "Both numeric / excluded",
                ),
                rows,
            ),
            "Reference-uncertain cases and native-score agreement on eligible images are preserved in analysis.json. These scope counts do not establish performance on every intentional-effects or other protected slice.",
        ]
    )


def ladder_section(analysis: dict) -> str:
    ladder = analysis["ladders"]
    rows = [
        [
            NAMES.get(row["model"], row["model"]),
            f"{row['complete_families']} / {row['families']}",
            row["nonincreasing_families"],
            row["adjacent_ties"],
            row["strict_inversions"],
            row["severe_at_least_parent"],
            number(row["mean_parent_minus_severity3"]),
        ]
        for row in ladder["summary"]
        if row["operator"] == "all"
    ]
    return "\n\n".join(
        [
            "## Controlled degradation ladders",
            "All controlled families are retained. Technical scores are expected to be non-increasing from parent through the three severities; aesthetic monotonicity is not required. Ties and strict inversions are separate counts. Positive parent-minus-severity-3 values mean the original scored higher.",
            table(
                (
                    "Model",
                    "Complete / families",
                    "Non-increasing",
                    "Adjacent ties",
                    "Strict inversions",
                    "Severe ≥ parent",
                    "Mean parent − severe",
                ),
                rows,
            ),
            ladder["verified_subset_status"] + ". " + ladder["pace_caveat"] + ".",
        ]
    )


def qc_section(analysis: dict) -> str:
    qc = analysis.get("qc", {})
    if "rows" not in qc:
        return (
            "## QC review trial\n\nNot computed: "
            + qc.get("reason", "QC results are absent from this artifact")
            + "."
        )
    rows = [
        [
            NAMES.get(row["model"], row["model"]),
            row["valid_reference_cases_with_stats"],
            f"{row['review_budget']} / {row['planned_review_budget']}",
            row["fixed_budget"]["large_errors_found"],
            number(row["fixed_budget"]["precision"]),
            number(row["fixed_budget"]["recall"]),
            number(row["random_review_expected_large_errors"]),
            f"{row['reserved_random_audit_slots']} / {row['unflagged_audits_in_budget']}",
        ]
        for row in qc["rows"]
        if row["panel"] == "representative" and row["arm"] == "baseline"
    ]
    return "\n\n".join(
        [
            "## QC review trial",
            "QC flags select images for review without changing a label. The fixed budget is 10% of the panel, capped by valid cases with stats. Twenty percent of review slots (rounded up) are reserved for random unflagged audits; slots move between pools only when a pool cannot fill its allocation. A large error is absolute technical error ≥0.75. Percentiles come from the frozen nonbenchmark calibration families. The table below is the representative baseline panel; challenge and treatment diagnostics remain in analysis.json.",
            table(
                (
                    "Model",
                    "Valid cases",
                    "Used / planned reviews",
                    "Large errors found",
                    "Precision",
                    "Recall",
                    "Random expected hits",
                    "Reserved / actual audits",
                ),
                rows,
            ),
        ]
    )


def render(analysis: dict, manifest: dict) -> str:
    validate_counts(analysis, manifest)
    cases = manifest["cases"]
    partitions = Counter(case["benchmark_partition"] for case in cases)
    groups = analysis["operations"]["model_arms"]
    models = [row for row in groups if row["model"] != "reference"]
    ref = next(row for row in groups if row["model"] == "reference")
    expected = sum(row["expected_images"] for row in models)
    completed = sum(sum(row["status_counts"].get(status, 0) for status in COMPLETE) for row in models)
    parts = [
        "# Four-thousand-image rehearsal results",
        f"Analysis generated {analysis['created_at']}. The frozen corpus has {len(cases):,} image views across {len({case['family_id'] for case in cases}):,} source families. There are {partitions['representative']} representative, {partitions['challenge']} challenge and {partitions['development']} development reference cases. Reference rows recorded as valid: {ref['status_counts'].get('valid', 0)} / {ref['expected_images']}.",
        f"Recorded completed scoring responses: **{completed:,} / {expected:,}** expected model/arm/image slots. Completion includes invalid schema responses; numeric-label coverage is shown below. {expected - completed:,} slots lack a completed scoring response. The CSV contains {analysis['long_rows']:,} rows including explicit missing slots and derived means.",
        "Reference scores are independent assistant visual judgments, not human MOS or verified human ground truth. Lower MAE means closer agreement with those judgments. All labels remain experimental. The Luna/DeepSeek mean is derived from its two components.",
        "[Full scores](scores-long.csv) · [Detailed analysis](analysis.json) · [Per-model costs](cost-per-model.csv)\n\n## Completion and validity",
    ]
    completion = []
    for row in groups:
        if row["model"] == "reference":
            continue
        done = sum(row["status_counts"].get(status, 0) for status in COMPLETE)
        completion.append(
            [
                NAMES[row["model"]],
                ARMS[row["arm"]],
                row["expected_images"],
                row["rows_with_attempts"],
                done,
                row["numeric_scores"]["technical_score"],
                row["numeric_scores"]["aesthetic_score"],
                row["expected_images"] - done,
                row["attempts"],
            ]
        )
    parts.append(
        table(
            (
                "Model",
                "Arm",
                "Expected",
                "Attempted images",
                "Completed responses",
                "Numeric tech",
                "Numeric aest",
                "Missing/error",
                "All attempts",
            ),
            completion,
        )
    )
    parts.append(
        "Each task retains its own valid cases. Combined MAE uses cases with both scores available. The additional five-baseline common subset supports a like-for-like comparison; it does not change the primary paired treatment population. Signed bias is prediction minus reference. Spearman ρ is rank agreement."
    )
    parts.extend(
        accuracy_section(analysis, panel, partitions[panel]) for panel in ("representative", "challenge")
    )
    parts.append(
        f"Development results ({partitions['development']} images), pooled evaluation results and full-corpus score agreement are retained in analysis.json. Agreement on the {partitions['rehearsal']:,} nonbenchmark images does not measure accuracy because they have no independent reference labels."
    )
    parts.extend(
        [treatment_section(analysis), scope_section(analysis), qc_section(analysis), ladder_section(analysis)]
    )
    timing = []
    for row in models:
        latency = row["selected_response_http_latency"]
        timing.append(
            [
                NAMES[row["model"]],
                ARMS[row["arm"]],
                latency["measured"],
                number(latency["mean_seconds"]),
                number(latency["p50_seconds"]),
                number(latency["p95_seconds"]),
                row.get("unmeasured_latency_attempts", "Unavailable"),
            ]
        )
    parts.extend(
        [
            "## End-to-end request latency",
            "Seconds from HTTP invocation through the complete response include upload, network/proxy, provider queue and processing. Local scheduler waiting and payload preparation are separate. This table uses completed scoring responses; all-attempt distributions and failures remain in analysis.json. Routes used different concurrency, so this is not a controlled production-speed comparison. Reference review duration is separate from HTTP latency; derived means have no new request.",
            table(
                ("Model", "Arm", "Measured responses", "Mean s", "Median s", "P95 s", "Unmeasured attempts"),
                timing,
            ),
            "## Limitations and next decisions",
            "The reference sample is selected from this corpus and represents assistant preferences. The bootstrap conditions on those labels and does not measure human-rating uncertainty. Challenge and development panels are separate from the representative primary population. Secondary comparisons are exploratory; the report does not claim that every quality issue or protected slice is unaffected. Missing or invalid labels reduce coverage and are not silently replaced. Controlled ladder results do not by themselves verify that every delivered distortion is perceptually worse. A larger labeling decision should consider the primary paired effect, aesthetic noninferiority, scope regressions, error review and coverage together.",
            "## Total cost per model",
            "These totals count every recorded attempt once across all arms, including retries and errors when a charge is reported. Unknown receipts remain unknown; the known subtotal is not a complete total when an attempt is unpriced. Derived means add no charge. Q-ReAlign Lite and Reference prices are explicitly assigned $0.00 as requested. Runpod infrastructure is separate from that reporting override; its monetary total is not supplied in this analysis artifact.",
        ]
    )
    totals = {row["model"]: row for row in analysis["operations"]["model_totals"]}
    costs = [
        [
            NAMES[model],
            totals[model]["attempts"],
            cost(totals[model]["known_price_usd"]),
            totals[model]["unpriced_attempts"],
            cost(totals[model]["total_price_usd"]),
            "Assigned zero" if model in {"qrealign_lite", "reference"} else "Provider receipts",
        ]
        for model in (*MODELS[:-1], "reference")
    ]
    parts.append(
        table(("Model", "All attempts", "Known subtotal", "Unknown-price attempts", "Total", "Basis"), costs)
    )
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=HERE / "analysis.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "report.md")
    args = parser.parse_args()
    analysis, manifest = (json.loads(path.read_text()) for path in (args.analysis, args.manifest))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    matching = [
        value for key, value in analysis["inputs"].items() if Path(key).resolve() == args.manifest.resolve()
    ]
    if matching != [digest]:
        raise ValueError("Report manifest does not match the audited analysis input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(analysis, manifest))
    print(json.dumps({"report": str(args.output.resolve()), "analysis_created_at": analysis["created_at"]}))


if __name__ == "__main__":
    main()
