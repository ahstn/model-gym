"""Freeze and evaluate pixel statistics with the dev-selected prompt; never retune."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import ROOT, digest, image_part, make_request
from bench_runner import run as run_benchmark
from tracks import image_context
from tracks.metrics import paired_reward_delta, summarize
from tracks.report import cell, cost_summary, per_image_markdown
from tracks.run import (
    _reported_spend,
    file_sha,
    ledger_records,
    local_path,
    read_protocol,
    relative_path,
    track_spending,
    validate_candidate,
    validate_records,
    validate_test_selection,
    write_frozen,
)
from tracks.settings import task_for

TRACKS = ("quality_v2", "ai_detection_v2")
FINAL_SPLITS = ("test", "external")
SELECTION_RULE = "Use the dev-selected frozen GEPA prompt without mutation, retuning or selection on test/external outcomes"


def _directory(track):
    if track not in TRACKS:
        raise ValueError("Metadata ablation requires a v2 track")
    return ROOT / "tracks" / track


def _final_paths(directory):
    return sorted(path for name in ("final", "final-metadata") for path in (directory / name).glob("*.jsonl"))


def freeze(track):
    """Derive context once before final calls; selection may be frozen later."""
    directory = _directory(track)
    protocol_path, protocol, dataset = read_protocol(track)
    if protocol.get("context_mode") != "image_only":
        raise ValueError("Primary optimisation must use image_only")
    if any(ledger_records(path) for path in _final_paths(directory)):
        raise ValueError("Freeze the ablation before any primary or metadata final calls")
    output = directory / "context-dataset.json"
    plan_path = directory / "ablation-plan.json"
    if output.exists() or plan_path.exists():
        raise ValueError("Ablation inputs already exist; frozen evidence cannot be replaced")
    contextual = copy.deepcopy(dataset)
    for sample in contextual["samples"]:
        if "model_context" in sample or "model_context_sha256" in sample:
            raise ValueError("Primary image-only dataset must not contain model context")
        if sample["split"] not in FINAL_SPLITS:
            continue
        image_part(sample)
        context = image_context.model_context(
            image_context.extract_image_context(local_path(sample["image"])), "pixel_stats"
        )
        image_context.validate_model_context(context)
        sample.update(model_context=context, model_context_sha256=digest(context))
    write_frozen(output, contextual)
    plan = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "track": track,
        "context_mode": "pixel_stats",
        "protocol_path": relative_path(protocol_path),
        "protocol_sha256": file_sha(protocol_path),
        "dataset_path": protocol["dataset_path"],
        "dataset_sha256": protocol["dataset_sha256"],
        "context_dataset_path": relative_path(output),
        "context_dataset_sha256": file_sha(output),
        "context_helper_sha256": file_sha(image_context.__file__),
        "sample_ids": {
            split: [s["id"] for s in dataset["samples"] if s["split"] == split] for split in FINAL_SPLITS
        },
        "selection_rule": SELECTION_RULE,
        "selection_path": relative_path(directory / "prompt-selection.json"),
        "track_budget_usd": 5.0,
        "cost_scope": "Optimisation, reflection, primary final, external and metadata ablation calls",
        "camera_context": False,
        "recompute_at_run": False,
    }
    write_frozen(plan_path, plan)
    return plan


def _read_frozen(track):
    directory = _directory(track)
    plan_path = directory / "ablation-plan.json"
    plan = json.loads(plan_path.read_bytes())
    protocol_path, protocol, original = read_protocol(track)
    expected = {
        "track": track,
        "context_mode": "pixel_stats",
        "selection_rule": SELECTION_RULE,
        "track_budget_usd": 5.0,
        "camera_context": False,
        "recompute_at_run": False,
        "protocol_sha256": file_sha(protocol_path),
        "dataset_sha256": protocol["dataset_sha256"],
        "context_helper_sha256": file_sha(image_context.__file__),
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen ablation plan or helper changed")
    if protocol.get("context_mode") != "image_only":
        raise ValueError("Primary protocol must use image_only")
    context_path = local_path(plan["context_dataset_path"])
    if file_sha(context_path) != plan["context_dataset_sha256"]:
        raise ValueError("Frozen context dataset bytes changed")
    contextual = json.loads(context_path.read_bytes())
    stripped = copy.deepcopy(contextual)
    for sample in stripped["samples"]:
        sample.pop("model_context", None)
        sample.pop("model_context_sha256", None)
    if stripped != original:
        raise ValueError("Context dataset changed original samples or labels")
    for split in FINAL_SPLITS:
        samples = [s for s in contextual["samples"] if s["split"] == split]
        if [s["id"] for s in samples] != plan["sample_ids"][split]:
            raise ValueError("Frozen final sample IDs changed")
        for sample in samples:
            context = sample["model_context"]
            image_context.validate_model_context(context)
            if "camera" in context or sample["model_context_sha256"] != digest(context):
                raise ValueError("Frozen pixel context hash or shape changed")
            image_part(sample)
    return directory, plan_path, plan, protocol_path, protocol, original, contextual


def spending(track):
    """Require known API costs, including unsuccessful recorded attempts."""
    _directory(track)
    return track_spending(track)


def _selected(directory, protocol_path, protocol):
    path = directory / "prompt-selection.json"
    selection = json.loads(path.read_bytes())
    prompt_path = local_path(selection["selected_prompt_path"])
    validate_test_selection(path, protocol_path, protocol, prompt_path)
    if file_sha(prompt_path) != selection["selected_prompt_sha256"]:
        raise ValueError("Ablation requires exactly the selected prompt")
    return path, selection, prompt_path, validate_candidate(prompt_path, protocol)


def _validate_metadata_records(records, protocol, dataset, prompt, split):
    validate_records(records, protocol, dataset, prompt)
    samples = {s["id"]: s for s in dataset["samples"] if s["split"] == split}
    seen = set()
    for row in records:
        if row["sample"] in seen or row["sample"] not in samples:
            raise ValueError("Metadata ledger contains duplicate or wrong-split attempts")
        seen.add(row["sample"])
        sample = samples[row["sample"]]
        endpoint, payload = make_request(
            protocol["model_alias"],
            image_part(sample),
            prompt,
            "low",
            4096,
            extra_context=sample["model_context"],
        )
        if row.get("request_sha256") != digest({"endpoint": endpoint, "payload": payload}):
            raise ValueError("Metadata ledger request changed")
        if (
            row.get("sample_context_mode") != "pixel_stats"
            or row.get("model_context_sha256") != sample["model_context_sha256"]
            or row.get("delivered_sha256") != sample["delivered_sha256"]
        ):
            raise ValueError("Metadata ledger context or delivered image changed")


def _run_receipt(track, plan_path, plan, selection_path, selection, prompt_path):
    return {
        "track": track,
        "ablation_plan_sha256": file_sha(plan_path),
        "selection_sha256": file_sha(selection_path),
        "selected_prompt_path": relative_path(prompt_path),
        "selected_prompt_sha256": selection["selected_prompt_sha256"],
        "selection_rule": SELECTION_RULE,
        "sample_ids": plan["sample_ids"],
    }


def run(track):
    """Run one attempt per final image, resuming all recorded attempts without retry."""
    directory, plan_path, plan, protocol_path, protocol, _, dataset = _read_frozen(track)
    selection_path, selection, prompt_path, prompt = _selected(directory, protocol_path, protocol)
    record_path = directory / "ablation-run.json"
    receipt = _run_receipt(track, plan_path, plan, selection_path, selection, prompt_path)
    # Check both ledgers and the combined cost before recording or starting any run.
    for split in FINAL_SPLITS:
        rows = ledger_records(directory / "final-metadata" / f"{split}.jsonl")
        _validate_metadata_records(rows, protocol, dataset, prompt, split)
        if rows and not record_path.exists():
            raise ValueError("Metadata ledger lacks its frozen run record")
    spending(track)
    if record_path.exists():
        if json.loads(record_path.read_bytes()) != receipt:
            raise ValueError("Frozen ablation run or selected prompt changed")
    else:
        write_frozen(record_path, receipt)
    groups = defaultdict(list)
    for sample in dataset["samples"]:
        if sample["split"] in FINAL_SPLITS:
            groups[(sample["split"], sample["group_id"])].append(sample["id"])
    for (split, _), identifiers in groups.items():
        output = directory / "final-metadata" / f"{split}.jsonl"
        records = ledger_records(output)
        _validate_metadata_records(records, protocol, dataset, prompt, split)
        if set(identifiers) <= {row["sample"] for row in records}:
            continue
        costs = spending(track)
        remaining = plan["track_budget_usd"] - costs.get("budget_accounted_usd", costs["total_usd"])
        if remaining < 0.10:
            raise ValueError("Combined track budget reached")
        run_benchmark(
            SimpleNamespace(
                manifest=local_path(plan["context_dataset_path"]),
                prompt=prompt_path,
                task=protocol["task"],
                models=[protocol["model_alias"]],
                output=output,
                ids=identifiers,
                limit=None,
                effort="low",
                max_tokens=4096,
                max_calls=len(plan["sample_ids"][split]),
                reserve_usd=0.10,
                budget_usd=remaining + _reported_spend(records),
                sample_context_mode="pixel_stats",
            )
        )
    return {"track": track, "spending": spending(track), "run_record": relative_path(record_path)}


def _primary_ledger(directory, split, name):
    if split == "test":
        return directory / "final" / f"{name}.jsonl"
    return directory / "final" / f"external-{name}.jsonl"


def _report_markdown(result):
    lines = [
        "# Frozen pixel statistics ablation",
        "",
        "The same development-selected prompt was evaluated with the same delivered images. Pixel statistics were frozen before final calls. Test and external labels did not guide prompt or context selection.",
        "",
    ]
    quality = task_for(result["track"]) == "quality"
    for split, panel in result["panels"].items():
        lines += [
            f"## {split.capitalize()} panel",
            "",
            "| Arm | Valid / expected | Group mean reward | Error metric | Score association | Balanced accuracy (95% group CI) | Reported arm cost USD | Client latency p50 / p95 s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for arm, evaluation in panel["evaluations"].items():
            summary = evaluation["summary"]
            metrics = summary["metrics"]
            binary = metrics["bad_detection"]["valid_only"] if quality else metrics["valid_only"]
            error = metrics["mae_valid" if quality else "brier_valid"]
            association = metrics["spearman_valid" if quality else "auroc_valid"]
            interval = (
                summary["group_bootstrap"]["intervals"].get("balanced_accuracy_valid", {}).get("percentile95")
            )
            accuracy = cell(binary["balanced_accuracy"]) + (
                f" [{cell(interval[0])}, {cell(interval[1])}]" if interval is not None else ""
            )
            operations = summary["operations"]
            reported_cost = operations["effective_provider_inference_cost_usd"]
            cost = "—" if reported_cost is None else f"{reported_cost:.6f}"
            latency = f"{cell(operations['latency_p50_seconds'])} / {cell(operations['latency_p95_seconds'])}"
            lines.append(
                f"| {arm} | {operations['valid']} / {summary['expected']} | {cell(summary['objective']['score'])} | {cell(error)} | {cell(association)} | {accuracy} | {cost} | {latency} |"
            )
        delta = panel["paired_comparison"]
        interval = delta["paired_group_bootstrap95"]
        interval_text = (
            f"[{cell(interval[0])}, {cell(interval[1])}]" if interval is not None else "unavailable"
        )
        lines += [
            "",
            f"{split}: metadata minus image-only reward is {cell(delta['selected_minus_baseline'])}; paired group bootstrap 95% interval is {interval_text}.",
            "",
        ]
    metric_note = (
        "Error metric is MAE on the native 1..5 quality scale; score association is Spearman correlation."
        if quality
        else "Error metric is Brier loss; score association is AUROC."
    )
    lines += [
        metric_note,
        "",
        "Missing and invalid attempts retain zero reward. Group bootstrap intervals do not account for source bias. Descriptive pixel measurements are not calibrated quality scores or proof of origin.",
        "",
        cost_summary(result["spending"]),
        "",
        "See [per-image evidence](metadata-per-image.md) and [JSON results](metadata-report.json).",
        "",
    ]
    return "\n".join(lines)


def report(track, bootstrap_replicates=500):
    """Build a separate offline report without changing the primary report."""
    if not 0 <= bootstrap_replicates <= 10000:
        raise ValueError("Use 0..10000 bootstrap replicates")
    directory, plan_path, plan, protocol_path, protocol, original, contextual = _read_frozen(track)
    selection_path, selection, prompt_path, prompt = _selected(directory, protocol_path, protocol)
    receipt = _run_receipt(track, plan_path, plan, selection_path, selection, prompt_path)
    if json.loads((directory / "ablation-run.json").read_bytes()) != receipt:
        raise ValueError("Frozen ablation run or selected prompt changed")
    name = (
        "baseline"
        if selection["selected_prompt_sha256"] == protocol["baseline_prompt_sha256"]
        else "selected"
    )
    result = {
        "track": track,
        "ablation_plan_sha256": file_sha(plan_path),
        "selection_sha256": file_sha(selection_path),
        "selected_prompt_sha256": file_sha(prompt_path),
        "selection_rule": SELECTION_RULE,
        "panels": {},
        "spending": spending(track),
    }
    for split in FINAL_SPLITS:
        samples = [s for s in contextual["samples"] if s["split"] == split]
        if not samples:
            continue
        primary_path = _primary_ledger(directory, split, name)
        metadata_path = directory / "final-metadata" / f"{split}.jsonl"
        primary, metadata = ledger_records(primary_path), ledger_records(metadata_path)
        validate_records(primary, protocol, original, prompt)
        if any(row.get("sample_context_mode", "image_only") != "image_only" for row in primary):
            raise ValueError("Primary comparison ledger must use image_only")
        _validate_metadata_records(metadata, protocol, contextual, prompt, split)
        evaluations = {
            "image_only": summarize(track, samples, primary, bootstrap_replicates),
            "pixel_stats": summarize(track, samples, metadata, bootstrap_replicates),
        }
        result["panels"][split] = {
            "evaluations": evaluations,
            "paired_comparison": paired_reward_delta(
                evaluations["image_only"]["per_image"],
                evaluations["pixel_stats"]["per_image"],
                bootstrap_replicates,
            ),
            "inputs": [
                {"path": relative_path(path), "sha256": file_sha(path) if path.exists() else None}
                for path in (primary_path, metadata_path)
            ],
        }
    output = directory / "report"
    # Reports are immutable snapshots. A caller must preserve each completed report.
    if any(
        (output / name).exists()
        for name in ("metadata-report.json", "metadata-report.md", "metadata-per-image.md")
    ):
        raise ValueError("Metadata report evidence already exists")
    write_frozen(output / "metadata-report.json", result)
    evaluations = {
        f"{split}/{arm}": evaluation
        for split, panel in result["panels"].items()
        for arm, evaluation in panel["evaluations"].items()
    }
    for path, text in (
        (output / "metadata-report.md", _report_markdown(result)),
        (
            output / "metadata-per-image.md",
            per_image_markdown({"track": track, "evaluations": evaluations}, output),
        ),
    ):
        with path.open("x") as handle:
            handle.write(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    result = (
        report(args.track, args.bootstrap_replicates)
        if args.command == "report"
        else globals()[args.command](args.track)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
