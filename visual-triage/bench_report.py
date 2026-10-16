"""Render per-image benchmark evidence without selecting models or changing scores."""

import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from bench_analyze import ai_label, human_target, load_samples, numeric

ROOT = Path(__file__).resolve().parent
MODELS = {
    "decider": "Perplexity Decider v1.1",
    "gemini": "Gemini 3.8 Flash",
    "luna": "GPT-6 Luna",
}


def text(value):
    if value is None:
        return "—"
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ")


def number(value, places=2):
    return f"{value:.{places}f}" if numeric(value) else "—"


def money(value):
    return "$" + number(value, 7) if numeric(value, 0) else "unknown"


def link_path(path, output):
    return quote(os.path.relpath(path, output.parent), safe="/.")


def model_label(row):
    label = MODELS.get(row["arm"], row["arm"])
    reasoning = row.get("requested_reasoning")
    if reasoning and reasoning.get("effort"):
        label += " (" + reasoning["effort"] + ")"
    return text(label)


def human_reference_text(sample, axis):
    target = human_target(sample, axis)
    if target["raw"] is None and target["normalized"] is None:
        return "not available"
    label = sample.get("labels", {}).get(axis)
    description = "native " + number(target["raw"], 4)
    if isinstance(label, dict) and label.get("scale"):
        description += " on " + text(label["scale"])
    if target["normalized"] is not None:
        description += "; 1–5 value " + number(target["normalized"])
    if isinstance(label, dict) and label.get("caveat"):
        description += ". " + text(label["caveat"])
    return description


def image_heading(sample, output):
    lines = ["### " + text(sample["id"]), ""]
    if sample.get("image"):
        image = (ROOT / sample["image"]).resolve()
        if not image.is_relative_to(ROOT):
            raise ValueError("Manifest image path is outside visual-triage")
        lines += [f"[Open delivered image]({link_path(image, output)}).", ""]
    return lines


def reference_lines(sample, reference, task):
    if task == "quality":
        lines = [
            "Assistant reference: quality "
            + number(reference.get("technical_quality"))
            + "/5; aesthetics "
            + number(reference.get("aesthetics"))
            + "/5.",
            "",
            "Human quality: " + human_reference_text(sample, "technical_quality") + ".",
            "Human aesthetics: " + human_reference_text(sample, "aesthetics") + ".",
        ]
    else:
        truth = ai_label(sample)
        truth_text = "AI-generated" if truth is True else "non-generative" if truth is False else "unknown"
        lines = [
            "Dataset origin label: **" + truth_text + "**.",
            "",
            "Assistant visual suspicion p(AI): " + number(reference.get("ai_probability"), 3) + ".",
        ]
        if reference.get("ai_probability_interval") is not None:
            lines.append(
                "Assistant uncertainty interval: " + text(reference["ai_probability_interval"]) + "."
            )
    if reference.get("confidence") is not None:
        lines.append("Assistant confidence: " + text(reference["confidence"]) + ".")
    if reference.get("medium") is not None:
        lines.append("Assistant visible medium: " + text(reference["medium"]) + ".")
    if reference.get("uncertainty") is not None:
        lines.append("Assistant origin uncertainty: " + text(reference["uncertainty"]) + ".")
    lines += ["", "Assistant assessment: " + text(reference.get("evidence", "Not supplied.")), ""]
    return lines


def prediction_table(rows, task):
    if task == "quality":
        lines = [
            "| Model / effort | Quality | Aesthetics | Provider inference cost | OpenRouter charge | Latency | Provider |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Model / effort | p(AI) | Medium | Uncertainty | Provider inference cost | OpenRouter charge | Latency | Provider |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    for row in sorted(
        rows,
        key=lambda item: (list(MODELS).index(item["arm"]) if item["arm"] in MODELS else 9, model_label(item)),
    ):
        prediction = row.get("prediction") if row.get("valid") else {}
        prediction = prediction or {}
        label = model_label(row) + (" — invalid output" if not row.get("valid") else "")
        values = [label]
        if task == "quality":
            values += [number(prediction.get("technical_quality")), number(prediction.get("aesthetics"))]
        else:
            values += [
                number(prediction.get("ai_probability"), 3),
                text(prediction.get("medium")),
                text(prediction.get("uncertainty")),
            ]
        values += [
            money(row.get("effective_cost_usd")),
            money(row.get("cost_usd")),
            number(row.get("seconds"), 3) + " s",
            text(row.get("provider")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    for arm, label in MODELS.items():
        if not any(row["arm"] == arm for row in rows):
            values = [label + " — no attempt recorded"] + ["—"] * (6 if task == "quality" else 7)
            lines.append("| " + " | ".join(values) + " |")
    return lines + [""]


def render_report(summary, references, samples, output):
    rows = summary["per_image"]
    grouped = defaultdict(list)
    for row in rows:
        identifier = row["sample"]
        if identifier not in samples:
            raise ValueError(f"Result image absent from manifest: {identifier}")
        hashes = {
            item["delivered_sha256"]
            for item in (row, samples[identifier], references.get(identifier, {}))
            if item.get("delivered_sha256")
        }
        if len(hashes) > 1:
            raise ValueError(f"Delivered image hash differs between inputs: {identifier}")
        grouped[row["task"], identifier].append(row)
    lines = [
        "# Per-image benchmark assessment",
        "",
        "Assistant references are model judgments, not human gold. Model probability is visual suspicion; source provenance supplies the AI target. Assistant intervals are subjective ranges, not statistical confidence intervals. All scores describe the delivered view unless the human-label note states otherwise.",
        "",
        "Quality and aesthetics use 1–5 scales. AI probability uses 0–1. A dash means unavailable; invalid outputs retain their billed cost and request latency. Each recorded model configuration is shown without selecting the most favorable response.",
        "",
        "Provider inference cost uses reported upstream inference for BYOK requests and OpenRouter usage cost otherwise. OpenRouter charges are shown separately; do not add these columns without checking the billing contract. Image links point to local, ignored files acquired by the benchmark builders.",
        "",
    ]
    for task, title in (("quality", "Quality and aesthetics"), ("ai", "AI-origin detection")):
        identifiers = sorted(
            {identifier for row_task, identifier in grouped if row_task == task}
            | {identifier for identifier, sample in samples.items() if sample.get("_benchmark") == task}
        )
        lines += ["## " + title, "", f"Images: {len(identifiers)}.", ""]
        for identifier in identifiers:
            sample, reference = samples[identifier], references.get(identifier, {})
            lines += image_heading(sample, output)
            lines += reference_lines(sample, reference, task)
            lines += prediction_table(grouped[task, identifier], task)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--references", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = render_report(
            json.loads(args.summary.read_text()),
            load_samples(args.references),
            load_samples(args.manifest),
            args.output.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote per-image report: {args.output}")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
