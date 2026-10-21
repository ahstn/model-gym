"""Compare Jev and selected Decider prompts on the latest frozen test panels."""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_analyze import operational_metrics
from bench_protocol import digest, normalize_decisions
from tracks.jev14.report import _validate_attempt, compact_case, display, sha
from tracks.metrics import paired_reward_delta, summarize

ROOT = Path(__file__).resolve().parents[2]
TRACKS = {"quality": "quality_v2", "ai": "ai_detection_v2"}


def test_samples(task, document):
    """Map source gold labels to existing metric functions without changing their meaning."""
    selected = []
    for sample in document["samples"]:
        if sample["split"] != "test":
            continue
        row = dict(sample)
        if task == "ai":
            if type(sample["gold"]) is not bool:
                raise ValueError("AI reference label must be a provenance boolean")
            row["labels"] = {"ai_generated": sample["gold"]}
        selected.append(row)
    if len({sample["id"] for sample in selected}) != len(selected):
        raise ValueError("Duplicate source test image")
    return selected


def validate_baseline(task, samples, records, prompt):
    expected = {sample["id"]: sample for sample in samples}
    if len(records) != len(expected) or {row["sample"] for row in records} != set(expected):
        raise ValueError("Decider baseline must cover the exact complete test panel")
    for row in records:
        if (
            row["delivered_sha256"] != expected[row["sample"]]["delivered_sha256"]
            or row["prompt_sha256"] != digest(prompt)
            or row["task"] != task
            or row["arm"] != "decider"
        ):
            raise ValueError("Decider input image, prompt or task differs")
        if (
            row.get("status") != 200
            or row.get("validation_error")
            or row["normalized"] != normalize_decisions(row["response"], prompt["questions"])
        ):
            raise ValueError("Decider baseline has an invalid or changed result")
        if task == "ai" and (
            row.get("metadata_arm") != "image_only" or row.get("context_sha256") != digest(None)
        ):
            raise ValueError("AI comparison requires the image-only Decider arm")


def compare_panel(task, samples, jev_records, baseline_records, *, bootstrap_replicates=500):
    """Keep missing results in the denominator and use one source-group comparison."""
    expected = {sample["id"] for sample in samples}
    if any(row["sample"] not in expected for row in jev_records):
        raise ValueError("Jev result is outside the planned panel")
    track = TRACKS[task]
    jev = summarize(track, samples, jev_records, bootstrap_replicates)
    decider = summarize(track, samples, baseline_records, bootstrap_replicates)
    valid_ids = {row["id"] for row in jev["per_image"] if row["valid"]}
    paired_samples = [sample for sample in samples if sample["id"] in valid_ids]
    paired_baseline = summarize(
        track, paired_samples, [row for row in baseline_records if row["sample"] in valid_ids], 0
    )
    comparison = paired_reward_delta(decider["per_image"], jev["per_image"], bootstrap_replicates)
    return {
        "jev14": jev["summary"],
        "decider": decider["summary"],
        "paired_valid_decider": paired_baseline["summary"],
        "paired_valid_ids": sorted(valid_ids),
        "paired_reward_delta": comparison,
    }, {"jev14": jev["per_image"], "decider": decider["per_image"]}


def fixed_gates(rows):
    """Describe the existing probability gates without fitting a new threshold."""
    result = {gate: {"real": 0, "ai": 0} for gate in ("accept", "review", "reject", "invalid")}
    for row in rows:
        if not row["valid"]:
            gate = "invalid"
        else:
            probability = row["prediction"]["ai_probability"]
            gate = "accept" if probability <= 0.2 else "reject" if probability >= 0.8 else "review"
        result[gate]["ai" if row["gold"] else "real"] += 1
    return result


def ai_sources(samples, traces):
    sources = {
        sample["id"]: (sample.get("source", {}).get("generator") or "unknown")
        for sample in samples
        if sample["labels"]["ai_generated"]
    }
    result = {}
    for source in sorted(set(sources.values())):
        identifiers = {key for key, value in sources.items() if value == source}
        result[source] = {"images": len(identifiers)}
        for arm, rows in traces.items():
            selected = [row for row in rows if row["id"] in identifiers]
            result[source][arm] = {
                "valid": sum(row["valid"] for row in selected),
                "detected": sum(
                    row["valid"] and row["prediction"]["ai_probability"] >= 0.5 for row in selected
                ),
            }
    return result


def assemble_report(cases, records, samples, baselines, context_tokens, *, bootstrap_replicates=500):
    """Produce compact evidence from validated requests on the two latest panels."""
    keys = {case["case_id"] for case in cases}
    if len(keys) != len(cases) or len({row["case_id"] for row in records}) != len(records):
        raise ValueError("Duplicate latest case")
    if {row["case_id"] for row in records} - keys:
        raise ValueError("Unexpected latest request")
    by_id, panels, rows, operation_rows = {row["case_id"]: row for row in records}, {}, [], []
    for task in TRACKS:
        selected_cases = [case for case in cases if case["task"] == task]
        if {case["sample"] for case in selected_cases} != {sample["id"] for sample in samples[task]}:
            raise ValueError("Latest case allocation differs from the source test panel")
        chosen = [row for row in records if row["task"] == task]
        panels[task], traces = compare_panel(
            task, samples[task], chosen, baselines[task], bootstrap_replicates=bootstrap_replicates
        )
        if task == "ai":
            panels[task]["by_source"] = ai_sources(samples[task], traces)
            panels[task]["fixed_gates"] = {arm: fixed_gates(rows) for arm, rows in traces.items()}
        jev_trace = {row["id"]: row for row in traces["jev14"]}
        baseline_trace = {row["id"]: row for row in traces["decider"]}
        operation_rows.extend(row for row in traces["jev14"] if row["attempted"])
        for case in selected_cases:
            record = by_id.get(case["case_id"])
            if record is not None:
                strict_valid = (
                    record.get("status") == 200
                    and not record.get("strict_validation_error")
                    and isinstance(record.get("strict_normalized"), dict)
                )
                recovered = (
                    not strict_valid
                    and not record.get("validation_error")
                    and isinstance(record.get("normalized"), dict)
                )
                record = {**record, "strict_parser_valid": strict_valid, "parser_review_applied": recovered}
            row = compact_case(case, record)
            traced, baseline = jev_trace[case["sample"]], baseline_trace[case["sample"]]
            row.update(
                gold=traced["gold"],
                gold_field=traced["gold_field"],
                group_id=traced["group_id"],
                correct=traced["correct"],
                decider_correct=baseline["correct"],
                decider_cost_usd=baseline["effective_cost_usd"],
                decider_seconds=baseline["seconds"],
            )
            for axis in ("technical_quality", "aesthetics", "ai_probability"):
                row["decider_" + axis] = baseline["prediction"].get(axis) if baseline["valid"] else None
            rows.append(row)
        tokens = [
            row["input_tokens"] for row in rows if row["task"] == task and row["input_tokens"] is not None
        ]
        panels[task]["context"] = {
            "reported_requests": len(tokens),
            "minimum_input_tokens": min(tokens) if tokens else None,
            "median_input_tokens": median(tokens) if tokens else None,
            "maximum_input_tokens": max(tokens) if tokens else None,
            "maximum_context_fraction": max(tokens) / context_tokens if tokens else None,
        }
    operations = operational_metrics(operation_rows)
    return {
        "status": "partial"
        if len(records) < len(cases)
        else "complete_with_invalid"
        if operations["invalid"]
        else "complete",
        "expected": len(cases),
        "unattempted": len(cases) - len(records),
        "operations": operations,
        "panels": panels,
        "context_tokens": context_tokens,
        "parser_columns": {
            "strict_parser_valid": "True when the original strict parser accepted the response",
            "parser_review_applied": "True only when bounded rounding recovered a strict-parser rejection; all responses use the same fixed parser",
        },
    }, rows


def build_report(root=ROOT, *, bootstrap_replicates=500):
    """Check frozen evidence and retain failures without making any network requests."""
    root = Path(root)
    data, track = root / "data/jev14-latest-v1", root / "tracks/jev14/latest"
    protocol_path = track / "protocol.json"
    protocol, protocol_hash = json.loads(protocol_path.read_text()), sha(protocol_path)
    pins = {path: value["sha256"] for path, value in protocol["source_inputs"].items()}
    pins.update(protocol["execution_sha256"])
    if any(sha(root / path) != expected for path, expected in pins.items()):
        raise ValueError("A frozen latest-trial source or execution file changed")
    if sha(data / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Latest manifest changed")
    cases = json.loads((data / "manifest.json").read_text())["cases"]
    prompts = {task: json.loads((root / path).read_text()) for task, path in protocol["prompts"].items()}
    samples, baselines = {}, {}
    for task in TRACKS:
        samples[task] = test_samples(task, json.loads((root / protocol["datasets"][task]).read_text()))
        by_id = {sample["id"]: sample for sample in samples[task]}
        chosen = [case for case in cases if case["task"] == task]
        if len(chosen) != protocol["counts"][task] or {case["sample"] for case in chosen} != set(by_id):
            raise ValueError("Latest manifest does not cover the exact test panel")
        for case in chosen:
            sample = by_id[case["sample"]]
            gold = sample["labels"]["quality_mos_1to5" if task == "quality" else "ai_generated"]
            if (
                case["delivered_sha256"] != sample["delivered_sha256"]
                or case["parent_delivered_sha256"] != sample["delivered_sha256"]
                or case["group_id"] != sample["group_id"]
                or case["gold"] != gold
                or case["phase"] != "primary"
            ):
                raise ValueError("Latest case image, gold or grouping differs")
        baselines[task] = [
            json.loads(line) for line in (root / protocol["baselines"][task]).read_text().splitlines() if line
        ]
        validate_baseline(task, samples[task], baselines[task], prompts[task])
    ledger_path = data / "attempts.jsonl"
    raw = ledger_path.read_bytes() if ledger_path.exists() else b""
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) > min(len(cases), protocol["maximum_attempts"]):
        raise ValueError("Too many latest-trial attempts")
    from tracks.jev14.latest import review

    for index, (case, row) in enumerate(zip(cases, records), 1):
        strict = {**row, "normalized": row.get("strict_normalized")}
        strict.pop("validation_error", None)
        if row.get("strict_validation_error"):
            strict["validation_error"] = row["strict_validation_error"]
        prompt = prompts[case["task"]]
        _validate_attempt(root, strict, case, protocol, protocol_hash, prompt, index)
        parsed = review(strict, prompt)
        if row.get("normalized") != parsed["normalized"] or row.get("validation_error") != parsed.get(
            "validation_error"
        ):
            raise ValueError("Latest parsed result differs from its raw response")
    summary, rows = assemble_report(
        cases,
        records,
        samples,
        baselines,
        protocol["context_tokens"],
        bootstrap_replicates=bootstrap_replicates,
    )
    summary.update(
        protocol_sha256=protocol_hash,
        manifest_sha256=protocol["manifest_sha256"],
        ledger_sha256=hashlib.sha256(raw).hexdigest(),
        source_inputs=protocol["source_inputs"],
        model=protocol["model"],
        expected_revision=protocol["expected_revision"],
        prompt_paths=protocol["prompts"],
        baseline_paths=protocol["baselines"],
        strict_parser_rejections=sum(
            row.get("status") == 200 and bool(row.get("strict_validation_error")) for row in records
        ),
        failures=[
            {"case_id": row["case_id"], "status": row.get("status"), "error": row.get("validation_error")}
            for row in records
            if row.get("validation_error")
        ],
    )
    return summary, rows


def _cost_latency(result):
    op = result["operations"]
    return (
        display(op["effective_provider_inference_cost_usd"], 6),
        f"{display(op['latency_p50_seconds'])} / {display(op['latency_p95_seconds'])}",
    )


def render_markdown(summary):
    op = summary["operations"]
    quality = summary["panels"]["quality"]
    ai = summary["panels"]["ai"]
    lead = ""
    if summary["status"] == "complete":
        jq, dq = quality["jev14"]["metrics"], quality["decider"]["metrics"]
        ja, da = ai["jev14"]["metrics"], ai["decider"]["metrics"]
        change = 100 * (1 - jq["mae_valid"] / dq["mae_valid"]) if dq["mae_valid"] else 0
        jc, dc = ja["valid_only"]["confusion"], da["valid_only"]["confusion"]
        lead = (
            f"Jev's quality error was {abs(change):.0f}% {'lower' if change >= 0 else 'higher'}. "
            f"It detected {jc['tp']} generated images versus Decider's {dc['tp']}, "
            f"with {jc['fp']} real photos falsely flagged versus {dc['fp']}. "
            f"Quality false flags were {jq['bad_detection']['valid_only']['confusion']['fp']} versus "
            f"{dq['bad_detection']['valid_only']['confusion']['fp']}. "
            f"Ranking scores remained close: quality Spearman {jq['spearman_valid']:.3f} / {dq['spearman_valid']:.3f}; "
            f"AI AUROC {ja['auroc_valid']:.3f} / {da['auroc_valid']:.3f} (Jev / Decider)."
        )
    lines = [
        "# Jev 1.14: latest quality and AI tests",
        "",
        lead,
        "",
        (
            f"**{summary['status'].replace('_', ' ').capitalize()}: {op['valid']}/{summary['expected']} valid results.** "
            "This is the main comparison: 100 KonIQ quality images and 160 AI-origin images. "
            "Jev and Decider used the same frozen pixels and selected prompts. No Jev prompt tuning was performed."
        ),
        "",
        "## Image quality",
        "",
        "| Model | Valid / planned | Human MOS MAE | Rank correlation | Low quality detected | False flags | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, name in (("jev14", "Jev 1.14"), ("decider", "Decider v1.1, selected prompt")):
        result = summary["panels"]["quality"][arm]
        metric, operation = result["metrics"], result["operations"]
        confusion = metric["bad_detection"]["valid_only"]["confusion"]
        cost, latency = _cost_latency(result)
        lines.append(
            f"| {name} | {operation['valid']}/{result['expected']} | {display(metric['mae_valid'])} "
            f"| {display(metric['spearman_valid'])} | {confusion['tp']}/{confusion['tp'] + confusion['fn']} "
            f"| {confusion['fp']}/{confusion['fp'] + confusion['tn']} | {cost} | {latency} |"
        )
    lines.extend(
        [
            "",
            (
                "MAE compares technical quality with human ratings on a 1–5 scale; lower is better. "
                "Low quality means a score below 2.6. The panel contains 50 low-quality and 50 other photographs. "
                "Aesthetic scores are saved per image, but no human aesthetic labels are available."
            ),
            "",
            "## AI detection",
            "",
            "| Model | Valid / planned | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, name in (("jev14", "Jev 1.14"), ("decider", "Decider v1.1, selected prompt")):
        result = summary["panels"]["ai"][arm]
        metric, operation = result["metrics"], result["operations"]
        confusion = metric["valid_only"]["confusion"]
        cost, latency = _cost_latency(result)
        accuracy = metric["accuracy_all_expected_invalid_as_wrong"]
        lines.append(
            f"| {name} | {operation['valid']}/{result['expected']} | {display(accuracy * 100, 1)}% "
            f"| {confusion['tp']}/{confusion['tp'] + confusion['fn']} | {confusion['fp']}/{confusion['fp'] + confusion['tn']} "
            f"| {display(metric['brier_valid'])} | {display(metric['auroc_valid'])} | {cost} | {latency} |"
        )
    lines.extend(
        [
            "",
            (
                "The fixed AI threshold is 0.5. Labels come from source provenance: 80 VISION camera photos and "
                "20 images each from DALL-E 2, DALL-E 3, Firefly and Midjourney v5. "
                "Both models receive image pixels only, with no EXIF or C2PA text. "
                "Accuracy counts missing or invalid results as wrong; the other score metrics use valid results."
            ),
            "",
        ]
    )
    sources = ai["by_source"]
    source_names = {
        "dalle2": "DALL-E 2",
        "dalle3": "DALL-E 3",
        "firefly": "Firefly",
        "midjourney-v5": "Midjourney v5",
    }
    source_counts = [
        f"{source_names.get(name, name)} {counts['jev14']['detected']}/{counts['images']} vs "
        f"{counts['decider']['detected']}/{counts['images']}"
        for name, counts in sources.items()
    ]
    lines.extend(
        [
            "AI detections by source (Jev vs Decider): " + "; ".join(source_counts) + ".",
            "",
            "| Existing probability gate | Jev: real / AI | Decider: real / AI |",
            "| --- | ---: | ---: |",
        ]
    )
    gates = ai["fixed_gates"]
    for gate, label in (
        ("accept", "Accept, p ≤ 0.2"),
        ("review", "Review, 0.2 < p < 0.8"),
        ("reject", "Reject, p ≥ 0.8"),
    ):
        left, right = gates["jev14"][gate], gates["decider"][gate]
        lines.append(f"| {label} | {left['real']} / {left['ai']} | {right['real']} / {right['ai']} |")
    lines.extend(
        [
            "",
            (
                "Jev sends more images to review under the existing gates. Its higher accuracy at 0.5 does not "
                "provide greater automatic coverage. These results do not validate automatic rejection."
            ),
            "",
            "## Image size and context",
            "",
            "| Panel | Image size | Input tokens: min / median / max | Maximum context used |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for task, size in (("quality", "1024×768 JPEG"), ("ai", "512×512 JPEG")):
        context = summary["panels"][task]["context"]
        tokens = " / ".join(
            display(context[key], 0)
            for key in ("minimum_input_tokens", "median_input_tokens", "maximum_input_tokens")
        )
        fraction = context["maximum_context_fraction"]
        lines.append(
            f"| {task} | {size} | {tokens} | {display(fraction * 100 if fraction is not None else None, 2)}% |"
        )
    lines.extend(
        [
            "",
            (
                "The catalogue lists a 262,144-token context window. Token counts include text, questions and image processing. "
                "The panels use different prompts, so their token difference does not isolate image size. "
                "The [separate size probes](../shared-results-2026-10-15/README.md) test the same two development images "
                "at 320, 512 and 1,024 pixels. Upscaling adds no source detail."
            ),
            "",
            (
                f"Latest-trial cost: ${display(op['known_effective_provider_inference_cost_usd'], 6)} known; "
                f"{op['unknown_cost_count']} unpriced attempts. The earlier compatibility failure has an unknown charge "
                "and is excluded from this trial total. Its budget reservation is not an actual charge."
            ),
            "",
            (
                f"The fixed parser allowed bounded two-decimal rounding from the start. "
                f"The original strict parser would reject {summary.get('strict_parser_rejections', 0)} responses. "
                "The CSV marks those recovered cases with `parser_review_applied`; `strict_parser_valid` records the original result. "
                "Native scores and probabilities are unchanged. Both experimental headers were sent; "
                "OpenRouter forwarding was not independently confirmed."
            ),
            "",
            (
                "Decider was tested on 11–12 October 2026; Jev was tested later. Latency is historical, not a simultaneous serving test. "
                "Source bias and the limited generator set restrict wider conclusions. "
                "The [older Luna/Gemini panel](../shared-results-2026-10-15/README.md) uses different images and is a separate comparison."
            ),
            "",
            (
                "[Per-image results](cases.csv) and [JSON summary](summary.json) contain failure counts, source-group intervals, "
                "paired comparisons and input hashes. Raw images and responses remain outside Git. "
                "See [OpenRouter multimodal Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions), "
                "[TypeSafe image limits](https://docs.typesafe.ai/images) and "
                "[Jev pricing](https://openrouter.ai/typesafe/jev-1.14)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(root, output):
    summary, cases = build_report(root)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(dict.fromkeys(key for row in cases for key in row)))
    writer.writeheader()
    writer.writerows(cases)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "cases.csv").write_text(text.getvalue())
    (output / "README.md").write_text(render_markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = write_report(ROOT, args.output)
    print(f"Saved {summary['status']} latest report with {summary['operations']['attempted']} attempts")


if __name__ == "__main__":
    main()
