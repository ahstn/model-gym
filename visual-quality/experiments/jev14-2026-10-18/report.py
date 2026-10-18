"""Compare a frozen Jev rerun with historical model calls; never call an API."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path("/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage")
LEDGER = HERE.parents[1] / "data/jev14-2026-10-18/attempts.jsonl"
NAMES = {"jev14": "Jev 1.14, 18 Oct", "decider": "Perplexity Decider v1.1", "luna": "GPT-6 Luna, low"}


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(path):
    samples = read_json(path)["samples"]
    result = {sample["id"]: sample for sample in samples}
    if len(result) != len(samples):
        raise ValueError("Duplicate source sample")
    return result


def load_verified_run():
    """Use the runner's complete validation and reject a changing ledger snapshot."""
    before = LEDGER.read_bytes() if LEDGER.exists() else b""
    spec = importlib.util.spec_from_file_location("jev14_rerun", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol, cases, records = module.load_study()
    after = LEDGER.read_bytes() if LEDGER.exists() else b""
    if before != after:
        raise ValueError("Ledger changed during reporting; retry after the run stops")
    if len(records) != len([line for line in after.splitlines() if line.strip()]):
        raise ValueError("Validated records do not cover the ledger snapshot")
    for case, row in zip(cases, records):
        copied_fields = set(case) - {"image", "eligible", "gold"}
        if any(row.get(key) != case[key] for key in copied_fields):
            raise ValueError("Ledger case metadata differs from the frozen manifest")
        if row.get("arm") != "jev14":
            raise ValueError("Ledger model arm changed")
    return protocol, cases, records, hashlib.sha256(after).hexdigest()


def baseline_metadata(rows):
    return {
        "resolved_models": dict(Counter(row.get("resolved_model") for row in rows)),
        "requested_reasoning": dict(
            Counter(json.dumps(row.get("requested_reasoning"), sort_keys=True) for row in rows)
        ),
        "first_timestamp": min((row["timestamp"] for row in rows if row.get("timestamp")), default=None),
        "last_timestamp": max((row["timestamp"] for row in rows if row.get("timestamp")), default=None),
        "cost_basis": "Reported inference cost includes BYOK upstream charges where present",
    }


def shared_panel(task, cases, records):
    from bench_protocol import digest, normalize_chat, normalize_decisions
    from tracks.jev14.report import compare_primary, evaluate

    samples = source_index(SOURCE / f"evidence/benchmarks/{task}.json")
    references = source_index(SOURCE / f"evidence/assessments/{task}-60.json")
    baselines = read_rows(SOURCE / f"evidence/final/{task}.jsonl")
    prompt = read_json(SOURCE / f"prompts/final/{task}.json")
    planned = [case for case in cases if case["panel"] == "shared" and case["task"] == task]
    fresh = [row for row in records if row["panel"] == "shared" and row["task"] == task]
    for row in baselines:
        if (
            row["prompt_sha256"] != digest(prompt)
            or row["delivered_sha256"] != samples[row["sample"]]["delivered_sha256"]
        ):
            raise ValueError("Shared baseline prompt or image hash changed")
        normalizer = normalize_decisions if row["arm"] == "decider" else normalize_chat
        if (
            row.get("status") != 200
            or row.get("validation_error")
            or row["normalized"] != normalizer(row["response"], prompt["questions"])
        ):
            raise ValueError("Shared baseline result is invalid or changed")
    summary, traces = compare_primary(task, planned, fresh, baselines, samples, references)
    summary["baselines"].pop("gemini", None)
    summary["paired_valid_baselines"].pop("gemini", None)
    chosen = {case["sample"]: samples[case["sample"]] for case in planned}
    all_traces = {"jev14": traces}
    for arm in ("decider", "luna"):
        selected = [row for row in baselines if row["arm"] == arm and row["sample"] in chosen]
        _, all_traces[arm] = evaluate(chosen, references, selected, task, arm)
    summary["baseline_metadata"] = {
        arm: baseline_metadata([row for row in baselines if row["arm"] == arm]) for arm in ("decider", "luna")
    }
    return summary, all_traces


def latest_panel(task, records):
    from tracks.jev14.latest_report import (
        compare_panel,
        test_samples,
        validate_baseline,
    )

    original = read_json(SOURCE / "tracks/jev14/latest/protocol.json")
    samples = test_samples(task, read_json(SOURCE / original["datasets"][task]))
    baselines = read_rows(SOURCE / original["baselines"][task])
    prompt = read_json(SOURCE / original["prompts"][task])
    validate_baseline(task, samples, baselines, prompt)
    fresh = [row for row in records if row["panel"] == "latest" and row["task"] == task]
    summary, traces = compare_panel(task, samples, fresh, baselines, bootstrap_replicates=2000)
    summary["baseline_metadata"] = baseline_metadata(baselines)
    return summary, traces


def gates(rows):
    result = {name: {"real": 0, "ai": 0} for name in ("accept", "review", "reject", "invalid")}
    for row in rows:
        gold = row.get("ai_generated", row.get("gold"))
        if type(gold) is not bool:
            raise ValueError("AI gate requires a provenance label")
        if not row["valid"]:
            gate = "invalid"
        else:
            probability = row["prediction"]["ai_probability"]
            gate = "accept" if probability <= 0.2 else "reject" if probability >= 0.8 else "review"
        result[gate]["ai" if gold else "real"] += 1
    return result


def build_report():
    protocol, cases, records, ledger_hash = load_verified_run()
    sys.path.insert(0, str(SOURCE))
    from bench_analyze import costs_for_comparison, operational_metrics

    counts = Counter((case["panel"], case["task"]) for case in cases)
    if counts != {
        ("shared", "quality"): 58,
        ("shared", "ai"): 60,
        ("latest", "quality"): 100,
        ("latest", "ai"): 160,
    }:
        raise ValueError("Unexpected comparison allocation")
    if len({case["row_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate planned case")
    operations = operational_metrics(
        [
            {
                **row,
                **costs_for_comparison(row),
                "valid": row.get("status") == 200
                and not row.get("validation_error")
                and isinstance(row.get("normalized"), dict),
            }
            for row in records
        ]
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if operations["valid"] == len(cases) else "partial",
        "expected": len(cases),
        "operations": operations,
        "unattempted": len(cases) - len(records),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "manifest_sha256": sha(HERE / "manifest.json"),
        "ledger_sha256": ledger_hash,
        "report_code_sha256": sha(Path(__file__)),
        "model": protocol.get("model"),
        "expected_revision": protocol.get("expected_revision"),
        "shared": {},
        "latest": {},
        "fixed_gates": {},
        "failures": [
            {
                key: row.get(key)
                for key in ("row_id", "status", "validation_error", "transport_error", "effective_cost_usd")
            }
            for row in records
            if row.get("validation_error") or row.get("transport_error") or row.get("status") != 200
        ],
    }
    traces = {}
    for task in ("quality", "ai"):
        summary["shared"][task], traces[("shared", task)] = shared_panel(task, cases, records)
        summary["latest"][task], traces[("latest", task)] = latest_panel(task, records)
    for panel in ("shared", "latest"):
        summary["fixed_gates"][panel] = {arm: gates(rows) for arm, rows in traces[(panel, "ai")].items()}
    return summary, compact_cases(cases, records, traces)


def compact_cases(cases, records, traces):
    actual = {row["row_id"]: row for row in records}
    lookup = {
        (panel, task, arm, row.get("sample", row.get("id"))): row
        for (panel, task), arms in traces.items()
        for arm, rows in arms.items()
        for row in rows
    }
    output = []
    for case in cases:
        record = actual.get(case["row_id"], {})
        item = {key: case[key] for key in ("row_id", "panel", "task", "sample", "image", "delivered_sha256")}
        item.update(
            {
                key: record.get(key)
                for key in ("status", "resolved_model", "seconds", "effective_cost_usd", "validation_error")
            }
        )
        for arm in ("jev14", "decider", "luna"):
            row = lookup.get((case["panel"], case["task"], arm, case["sample"]), {})
            prediction = row.get("prediction") or {}
            item[arm + "_valid"] = row.get("valid")
            for axis in ("technical_quality", "aesthetics", "ai_probability"):
                item[arm + "_" + axis] = prediction.get(axis)
        output.append(item)
    return output


def number(value, places=3):
    return f"{value:.{places}f}" if isinstance(value, (int, float)) else "—"


def cost_latency(result):
    operation = result["operations"]
    return f"{number(operation['effective_provider_inference_cost_usd'], 6)} | {number(operation['latency_p50_seconds'])} / {number(operation['latency_p95_seconds'])}"


def markdown(summary):
    op = summary["operations"]
    lines = [
        "# Jev 1.14 comparison, 18 October 2026",
        "",
        f"**{summary['status'].capitalize()}: {op['valid']}/{summary['expected']} valid results.** Jev used the same frozen images and prompts as the prior runs. Reported inference cost is ${number(op['effective_provider_inference_cost_usd'], 6)}; {op['unknown_cost_count']} attempts have unknown cost. Missing and invalid results remain in the planned denominator.",
        "",
        "The shared panel compares all three requested models. The latest panel gives stronger human quality evidence, but has no Luna baseline. These panels use different images and prompts, so their scores must remain separate.",
        "",
        "**Shared quality panel: 58 images**",
        "",
        "| Model | Valid | Human MOS MAE | Human rank correlation | Assistant quality / aesthetics MAE | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    shared = summary["shared"]["quality"]
    for arm, result in [("jev14", shared["jev14"]), *shared["baselines"].items()]:
        metrics = result["metrics"]
        human, assistant = metrics["human_reference"]["technical_quality"], metrics["assistant_reference"]
        lines.append(
            f"| {NAMES[arm]} | {result['operations']['valid']}/58 | {number(human['mae_valid'])} | {number(human['spearman'])} | {number(assistant['technical_quality']['mae_valid'])} / {number(assistant['aesthetics']['mae_valid'])} | {cost_latency(result)} |"
        )
    lines += [
        "",
        "Lower mean absolute error (MAE) is better. Quality uses a 1–5 scale. Human MOS is the mean opinion score. UHD-IQA MOS is mapped from 0.01–1 to 1–5 with 1 + 4 × (MOS − 0.01) / 0.99; this scale conversion does not calibrate it to our technical-quality rubric. The assistant assessments are weak references, not human labels. This UHD-IQA panel lacks examples below 2.6/5, and its human ratings used larger image views. Human aesthetic labels are absent. Two narrow panoramas are excluded because the original Jev image limits reject them.",
        "",
        "**Shared AI-origin panel: 60 images**",
        "",
        "| Model | Valid | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    shared = summary["shared"]["ai"]
    for arm, result in [("jev14", shared["jev14"]), *shared["baselines"].items()]:
        metrics = result["metrics"]
        confusion = metrics["confusion_valid_only"]
        lines.append(
            f"| {NAMES[arm]} | {result['operations']['valid']}/60 | {number(100 * metrics['accuracy_all_labeled_invalid_as_wrong'], 1)}% | {confusion['tp']}/30 | {confusion['fp']}/30 | {number(metrics['brier_valid_only'])} | {number(metrics['auroc_valid_only'])} | {cost_latency(result)} |"
        )
    lines += [
        "",
        "The fixed threshold is 0.5. Labels come from source provenance. The set includes 30 real photographs and 30 generated images; 12 generated images are graphics or renders. Source and style differences limit general claims about photo detection.",
        "",
        "**Latest quality panel: 100 KonIQ photographs**",
        "",
        "| Model | Valid | Human MOS MAE | Rank correlation | Low quality detected | False flags | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("jev14", "decider"):
        result = summary["latest"]["quality"][arm]
        metrics = result["metrics"]
        confusion = metrics["bad_detection"]["valid_only"]["confusion"]
        lines.append(
            f"| {NAMES[arm]} | {result['operations']['valid']}/100 | {number(metrics['mae_valid'])} | {number(metrics['spearman_valid'])} | {confusion['tp']}/50 | {confusion['fp']}/50 | {cost_latency(result)} |"
        )
    lines += [
        "",
        "Low quality means a score below 2.6. The set contains 50 low-quality and 50 other photographs. This is the direct human-label check for a quality scorer; it does not validate aesthetic scoring.",
        "",
        "**Latest AI-origin panel: 160 images**",
        "",
        "| Model | Valid | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("jev14", "decider"):
        result = summary["latest"]["ai"][arm]
        metrics = result["metrics"]
        confusion = metrics["valid_only"]["confusion"]
        lines.append(
            f"| {NAMES[arm]} | {result['operations']['valid']}/160 | {number(100 * metrics['accuracy_all_expected_invalid_as_wrong'], 1)}% | {confusion['tp']}/80 | {confusion['fp']}/80 | {number(metrics['brier_valid'])} | {number(metrics['auroc_valid'])} | {cost_latency(result)} |"
        )
    lines += [
        "",
        "This panel contains 80 VISION camera photographs and 20 images each from DALL-E 2, DALL-E 3, Firefly and Midjourney v5. Only image pixels were supplied. Accuracy counts missing or invalid outputs as wrong; score metrics use valid outputs.",
        "",
        "**Fixed review gates**",
        "",
        "| Panel | Model | Accept real / AI | Review real / AI | Reject real / AI | Invalid real / AI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for panel, arms in summary["fixed_gates"].items():
        for arm, gate in arms.items():
            counts = " | ".join(
                f"{gate[key]['real']} / {gate[key]['ai']}"
                for key in ("accept", "review", "reject", "invalid")
            )
            lines.append(f"| {panel} | {NAMES[arm]} | {counts} |")
    lines += [
        "",
        "The existing gates are p ≤ 0.2 for accept, 0.2 < p < 0.8 for review, and p ≥ 0.8 for reject. These are descriptive benchmark gates. The run does not authorize deletion or automatic rejection.",
        "",
        "**Uncertainty and comparison limits**",
        "",
    ]
    for task in ("quality", "ai"):
        delta = summary["latest"][task]["paired_reward_delta"]
        interval = delta.get("paired_group_bootstrap95")
        if interval is not None:
            label = (
                "quality reward (1 − absolute error / 4)"
                if task == "quality"
                else "AI reward (1 − squared error)"
            )
            lines.append(
                f"- Latest {label}: Jev minus Decider = {number(delta['selected_minus_baseline'], 4)}; paired source-group bootstrap 95% interval [{number(interval[0], 4)}, {number(interval[1], 4)}], {delta['replicates']} replicates. Higher reward is better."
            )
    lines += [
        "",
        "The JSON summary retains Wilson intervals for shared AI rates and group bootstrap intervals for the latest panels. These intervals do not cover source bias. The shared comparator calls ran on 11 October 2026; latest Decider calls ran on 11–12 October. Jev ran on 18 October. Latency includes network time and is a historical comparison, not a simultaneous serving test. Prompts were selected for Decider and transferred without Jev tuning.",
        "",
        "Historical Luna used `openai/gpt-6-luna` through chat completions, low reasoning effort and a 4,096-token output cap. It is not a test of a current Luna Decisions endpoint. Decider resolved to `perplexity/pplx-decider-v1.1-27b-20261006`. Luna reported no revision suffix. Its nonzero BYOK upstream cost is included even where the OpenRouter charge was zero. Costs are provider-reported inference costs, not invoice reconciliation.",
        "",
        "[JSON summary](summary.json) contains hashes, coverage, failures, model settings and detailed metrics. [Per-image comparison](cases.csv) retains all planned cases. Raw responses remain in the ignored data directory. Image bytes, prompts and prior evidence were preserved.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = build_report()
    text = markdown(summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "README.md").write_text(text)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "cases.csv").write_text(buffer.getvalue())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "valid": summary["operations"]["valid"],
                "expected": summary["expected"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
