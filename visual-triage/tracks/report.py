"""Write final track reports, per-image traces and paired prompt comparisons offline."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import ROOT
from tracks.metrics import paired_reward_delta, summarize
from tracks.run import (
    file_sha,
    ledger_records,
    local_path,
    read_protocol,
    select_split,
    track_spending,
    validate_records,
    validate_test_selection,
)
from tracks.settings import TRACKS, task_for


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def cost_cell(value, precision=8):
    return "—" if value is None else f"{value:.{precision}f}"


def per_image_markdown(result, output_directory):
    lines = [
        "# Per-image final results",
        "",
        "Errors and missing results remain in the expected denominator. Evidence is available in the matching JSON report; Decider evidence contains selected rubric labels only.",
        "",
        "| Prompt | Image | Group | Source | Gold | Prediction | Correct classification | Error/loss | Reported cost USD | Latency s | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, evaluation in result["evaluations"].items():
        for row in evaluation["per_image"]:
            link = os.path.relpath(ROOT / row["image"], output_directory)
            prediction = row["prediction"] or {}
            value = prediction.get(
                "technical_quality" if task_for(result["track"]) == "quality" else "ai_probability"
            )
            source = row.get("source") or {}
            source_label = " / ".join(str(source[key]) for key in ("dataset", "id") if source.get(key))
            status = (
                "valid"
                if row["valid"]
                else row.get("validation_error") or "invalid"
                if row["attempted"]
                else "unattempted"
            )
            cells = [
                name,
                f"[{row['id']}]({link})",
                row["group_id"],
                source_label,
                row["gold"],
                value,
                row["correct"],
                row["error"],
                cost_cell(row["effective_cost_usd"]),
                row["seconds"],
                status,
            ]
            lines.append("| " + " | ".join(cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def cost_summary(spending):
    if spending.get("unknown_cost_count"):
        return (
            f"Known reported track cost: ${spending['known_reported_usd']:.6f}. "
            f"Unreconciled budget reservation: ${spending['reserved_unverified_usd']:.2f}. "
            f"Guard-accounted total: ${spending['budget_accounted_usd']:.6f}. "
            "Actual total cost remains unknown; the reservation is not a provider-reported charge."
        )
    return f"Total reported track spending: ${spending['total_usd']:.6f}."


def summary_markdown(result):
    quality = task_for(result["track"]) == "quality"
    title = "Low-quality detection" if quality else "Matched photo AI detection"
    reuse = result["selected_reuses_baseline"]
    comparison_column = "Spearman" if quality else "AUROC"
    error_column = "MAE" if quality else "Brier"
    introduction = (
        "Development selection kept the baseline. It was scored once on the frozen test panel; no independent prompt gain was measured. Test labels did not guide selection."
        if reuse
        else "Baseline and selected prompts used the same frozen test views. Prompts were selected on development groups; test labels did not guide selection."
    )
    lines = [
        f"# {title}: final validation",
        "",
        introduction,
        "",
        f"| Prompt | Valid / expected | {error_column} | {comparison_column} | Balanced accuracy (95% group CI) | Recall | False-positive rate | Reported cost USD | Latency p50 / p95 s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, evaluation in result["evaluations"].items():
        summary = evaluation["summary"]
        metrics, operations = summary["metrics"], summary["operations"]
        binary = metrics["bad_detection"]["valid_only"] if quality else metrics["valid_only"]
        accuracy_interval = (
            summary["group_bootstrap"]["intervals"].get("balanced_accuracy_valid", {}).get("percentile95")
        )
        accuracy_text = cell(binary["balanced_accuracy"])
        if accuracy_interval is not None:
            accuracy_text += f" [{cell(accuracy_interval[0])}, {cell(accuracy_interval[1])}]"
        row = [
            name,
            f"{operations['valid']} / {summary['expected']}",
            metrics["mae_valid" if quality else "brier_valid"],
            metrics["spearman_valid" if quality else "auroc_valid"],
            accuracy_text,
            binary["recall"],
            binary["false_positive_rate"],
            cost_cell(operations["effective_provider_inference_cost_usd"], 6),
            f"{cell(operations['latency_p50_seconds'])} / {cell(operations['latency_p95_seconds'])}",
        ]
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    if not reuse:
        delta = result["paired_comparison"]
        interval = delta["paired_group_bootstrap95"]
        interval_text = (
            f"[{cell(interval[0])}, {cell(interval[1])}]" if interval is not None else "unavailable"
        )
        lines += [
            "",
            f"Selected minus baseline reward: **{cell(delta['selected_minus_baseline'])}** across {delta['n_groups']} source groups; paired group bootstrap 95% interval: {interval_text}. The 0.005 development selection margin does not establish statistical significance.",
        ]
    lines += [
        "",
        "Accuracy intervals resample whole source groups. The JSON report includes confusion counts, Wilson intervals for recall and false-positive rates within each class, continuous-score intervals, invalid-output penalties, costs from every attempt, and image-level feedback. A zero observed false-positive count does not imply zero future risk.",
        "",
    ]
    if quality:
        lines += [
            (
                "Low quality means a score strictly below 2.6. Native human MOS uses the original 1..5 direction: higher is better. These authentic photographs come from a bounded source panel; source coverage limits transfer claims. Human aesthetic labels are absent; aesthetic predictions are retained but not scored."
                if result["track"] == "quality_v2"
                else "Low quality means a score strictly below 2.6. Human DMOS comes from reference-aware distortion assessment; the VLM judges each image without its reference. Score agreement therefore compares related tasks. These are synthetic defects, so results do not establish performance on all authentic camera failures. Human aesthetic labels are absent; aesthetic predictions are retained but not scored."
            ),
            "",
        ]
    else:
        lines += [
            "AI probability at least 0.5 predicts generated origin. The open review band is (0.2, 0.8). Scene-caption matching controls some content differences but does not produce identical scenes or cameras. Source and generator effects can remain; pixel judgments do not prove provenance.",
            "",
        ]
    if "external_validation" in result:
        lines += [
            "External validation used the same frozen prompt selection. See [external results](external.md) and [external image results](external-per-image.md).",
            "",
        ]
    execution = (
        "Final calls used one baseline prompt."
        if reuse
        else "Final calls alternate prompt order by source group."
    )
    spending_text = (
        cost_summary(result["spending"])
        + " This includes search, reflection with training images, final inference and metadata calls."
        if result["track"].endswith("_v2")
        else (
            cost_summary(result["spending"])
            if result["spending"].get("unknown_cost_count")
            else f"Total track spending, including optimisation and text reflection: **${result['spending']['total_usd']:.6f}**."
        )
    )
    lines += [
        spending_text
        + f" {execution} BYOK provider inference and OpenRouter charges are separate fields; the spending guard conservatively counts both when separately reported.",
        "",
        "See [per-image results](per-image.md) and [complete machine-readable report](report.json).",
        "",
    ]
    return "\n".join(lines)


def build_report(track, bootstrap_replicates=500):
    protocol_path, protocol, dataset = read_protocol(track)
    directory = ROOT / "tracks" / track
    selection_path = directory / "prompt-selection.json"
    selection = json.loads(selection_path.read_bytes())
    prompts = {"baseline": local_path(protocol["baseline_prompt_path"])}
    reuse = selection["selected_prompt_sha256"] == protocol["baseline_prompt_sha256"]
    if not reuse:
        prompts["selected"] = local_path(selection["selected_prompt_path"])
    samples = select_split(dataset, "test")
    evaluations, inputs = {}, []
    for name, prompt_path in prompts.items():
        validate_test_selection(selection_path, protocol_path, protocol, prompt_path)
        prompt = json.loads(prompt_path.read_bytes())
        ledger = directory / "final" / f"{name}.jsonl"
        records = ledger_records(ledger)
        validate_records(records, protocol, dataset, prompt)
        evaluations[name] = summarize(track, samples, records, bootstrap_replicates)
        inputs.append(
            {
                "ledger": str(ledger.relative_to(ROOT)),
                "sha256": file_sha(ledger),
                "prompt_sha256": file_sha(prompt_path),
            }
        )
    baseline = evaluations["baseline"]["per_image"]
    selected = baseline if reuse else evaluations["selected"]["per_image"]
    comparison = paired_reward_delta(baseline, selected, 0 if reuse else bootstrap_replicates)
    comparison["independent_candidate_evaluated"] = not reuse
    if reuse:
        comparison["note"] = (
            "Selected baseline reuses the same final calls; no independent prompt gain was measured"
        )
    result = {
        "track": track,
        "protocol_sha256": file_sha(protocol_path),
        "selection_sha256": file_sha(selection_path),
        "dataset_sha256": protocol["dataset_sha256"],
        "inputs": inputs,
        "selected_reuses_baseline": reuse,
        "evaluations": evaluations,
        "paired_comparison": comparison,
        "spending": track_spending(track),
        "delivery": dataset.get("delivery"),
        "interval_limits": "Reported accuracy intervals use whole-group bootstrap; the separately named naive independent-image Wilson interval ignores group dependence. Recall/FPR Wilson intervals use one observation per class per parent/pair. Pair accuracy uses one outcome per pair. Intervals do not capture source bias.",
    }
    if TRACKS[track]["version"] == 2 and any(sample["split"] == "external" for sample in dataset["samples"]):
        external_samples = select_split(dataset, "external")
        external = {}
        for name, prompt_path in prompts.items():
            ledger = directory / "final" / f"external-{name}.jsonl"
            records = ledger_records(ledger)
            validate_records(records, protocol, dataset, json.loads(prompt_path.read_bytes()))
            external[name] = summarize(track, external_samples, records, bootstrap_replicates)
        external_baseline = external["baseline"]["per_image"]
        external_selected = external_baseline if reuse else external["selected"]["per_image"]
        result["external_validation"] = {
            "selection_used": False,
            "evaluations": external,
            "paired_comparison": paired_reward_delta(
                external_baseline, external_selected, 0 if reuse else bootstrap_replicates
            ),
            "inputs": [
                {
                    "ledger": str((directory / "final" / f"external-{name}.jsonl").relative_to(ROOT)),
                    "sha256": file_sha(directory / "final" / f"external-{name}.jsonl"),
                }
                for name in prompts
            ],
        }
    output = directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "per-image.md").write_text(per_image_markdown(result, output))
    (output / "README.md").write_text(summary_markdown(result))
    if "external_validation" in result:
        external_result = {**result, **result["external_validation"]}
        external_result.pop("external_validation")
        external_markdown = summary_markdown(external_result).replace(
            "(per-image.md)", "(external-per-image.md)"
        )
        (output / "external.md").write_text(external_markdown)
        (output / "external-per-image.md").write_text(per_image_markdown(external_result, output))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if not 0 <= args.bootstrap_replicates <= 10000:
        parser.error("Use 0..10000 bootstrap replicates")
    result = build_report(args.track, args.bootstrap_replicates)
    print(json.dumps(result["paired_comparison"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
