"""Freeze track inputs and run train/dev evaluations or a locked final test."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import MODELS, ROOT, digest, number, validate_prompt
from bench_runner import effective_cost
from bench_runner import run as run_benchmark
from cost_reservations import totals as cost_totals
from evidence_archive import require_restored
from tracks.metrics import summarize
from tracks.settings import TRACKS, task_for

SPLITS = {"train", "dev", "test", "external"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(path):
    path = Path(path)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Track artifacts must remain within visual-triage")
    return resolved


def relative_path(path):
    return str(local_path(path).relative_to(ROOT))


def write_frozen(path, value):
    """Never replace a previously frozen artifact, even after a failed run."""
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_dataset(track, dataset):
    if dataset.get("track") != track:
        raise ValueError("Dataset track does not match the requested track")
    samples = dataset["samples"]
    ids, groups, views = set(), {}, {}
    split_counts = dict.fromkeys({sample["split"] for sample in samples} | {"train", "dev", "test"}, 0)
    for sample in samples:
        identifier, group, split = sample["id"], sample["group_id"], sample["split"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Sample IDs must be nonempty and unique")
        if not isinstance(group, str) or not group or split not in SPLITS:
            raise ValueError("Samples require a group ID and train/dev/test split")
        sha = sample["delivered_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError("Every delivered view requires a SHA256")
        for owner, key in ((groups, group), (views, sha)):
            if key in owner and owner[key] != split:
                raise ValueError("Source group or exact delivered view crosses dataset splits")
            owner[key] = split
        local_path(sample["image"])
        if task_for(track) == "quality":
            number(sample["labels"]["quality_mos_1to5"], 1, 5)
        elif type(sample["labels"]["ai_generated"]) is not bool:
            raise TypeError("AI provenance labels must be boolean")
        split_counts[split] += 1
        ids.add(identifier)
    if not all(split_counts[split] for split in ("train", "dev", "test")):
        raise ValueError("Train, dev and test must each contain samples")
    return split_counts


def create_protocol(
    track,
    baseline_prompt_path,
    *,
    dataset_path=None,
    output_path=None,
    model_alias=None,
    context_mode="image_only",
    max_proposals=20,
):
    settings = dict(TRACKS[track])
    if model_alias is not None:
        if settings["version"] == 1 or model_alias not in MODELS:
            raise ValueError("Model overrides require a versioned track and known model alias")
        settings["arm"] = model_alias
    if context_mode not in {"image_only", "pixel_stats"} or not 10 <= max_proposals <= 40:
        raise ValueError("Use image_only/pixel_stats and 10..40 proposals")
    dataset_path = local_path(dataset_path or f"tracks/{track}/dataset.json")
    baseline_prompt_path = local_path(baseline_prompt_path)
    dataset = json.loads(dataset_path.read_bytes())
    counts = validate_dataset(track, dataset)
    validate_prompt(json.loads(baseline_prompt_path.read_bytes()), settings["task"])
    result = {
        "version": settings["version"],
        "created_at": datetime.now(UTC).isoformat(),
        "track": track,
        "dataset_path": relative_path(dataset_path),
        "dataset_sha256": file_sha(dataset_path),
        "baseline_prompt_path": relative_path(baseline_prompt_path),
        "baseline_prompt_sha256": file_sha(baseline_prompt_path),
        "task": settings["task"],
        "model_alias": settings["arm"],
        "model_id": MODELS[settings["arm"]],
        "reasoning_effort": "not exposed" if settings["arm"] == "decider" else "low",
        "max_output_tokens": 4096,
        "split_counts": counts,
        "quality_bad_rule": "quality < 2.6",
        "ai_rule": "ai_probability >= 0.5",
        "review_band_open": [0.2, 0.8],
        "selection": "Best dev group-balanced mean reward; require gain >= 0.005 over baseline, otherwise retain baseline",
        "minimum_gain": 0.005,
        "context_only_mutation": track != "quality_v2",
        "limits": "Track total at most $2, including optimisation, reflection and final calls; parent controls the overall experiment budget",
    }
    if settings["version"] == 2:
        result.update(
            track_budget_usd=settings["budget_usd"],
            context_mode=context_mode,
            label_direction="higher quality is better; AI probability means generated origin",
            max_proposals=max_proposals,
        )
        result["limits"] = "Track total at most $5, including optimisation, reflection and final calls"
    write_frozen(output_path or f"tracks/{track}/protocol.json", result)
    return result


def read_protocol(track, path=None):
    require_restored(ROOT, scope="tracks/" + track + "/")
    path = local_path(path or f"tracks/{track}/protocol.json")
    protocol = json.loads(path.read_bytes())
    settings = dict(TRACKS[track])
    if settings["version"] == 2:
        if protocol.get("model_alias") not in MODELS or protocol.get("version") != 2:
            raise ValueError("Unknown versioned protocol model or version")
        settings["arm"] = protocol["model_alias"]
        if protocol.get("track_budget_usd") != settings["budget_usd"] or protocol.get("context_mode") not in {
            "image_only",
            "pixel_stats",
        }:
            raise ValueError("Versioned budget or context mode differs from track settings")
        if (
            protocol.get("label_direction")
            != "higher quality is better; AI probability means generated origin"
        ):
            raise ValueError("Frozen label direction differs from the task")
        if not 10 <= protocol.get("max_proposals", 0) <= 40:
            raise ValueError("Versioned protocol requires 10..40 proposals")
    expected = {
        "version": settings["version"],
        "track": track,
        "task": settings["task"],
        "model_alias": settings["arm"],
        "model_id": MODELS[settings["arm"]],
        "max_output_tokens": 4096,
        "quality_bad_rule": "quality < 2.6",
        "ai_rule": "ai_probability >= 0.5",
        "review_band_open": [0.2, 0.8],
        "minimum_gain": 0.005,
        "context_only_mutation": track != "quality_v2",
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen protocol differs from the declared track settings")
    effort = "not exposed" if settings["arm"] == "decider" else "low"
    if protocol.get("reasoning_effort") != effort:
        raise ValueError("Frozen reasoning effort differs from the track setting")
    for prefix in ("dataset", "baseline_prompt"):
        if file_sha(local_path(protocol[prefix + "_path"])) != protocol[prefix + "_sha256"]:
            raise ValueError(f"Frozen {prefix} bytes changed")
    dataset = json.loads(local_path(protocol["dataset_path"]).read_bytes())
    validate_dataset(track, dataset)
    return path, protocol, dataset


def validate_candidate(prompt_path, protocol):
    prompt = json.loads(local_path(prompt_path).read_bytes())
    baseline = json.loads(local_path(protocol["baseline_prompt_path"]).read_bytes())
    return validate_candidate_value(prompt, baseline, protocol["task"], protocol["context_only_mutation"])


def validate_candidate_value(prompt, baseline, task, context_only_mutation):
    """Validate an in-memory candidate before writing files or making requests."""
    validate_prompt(prompt, task)

    def fixed(value):
        value = json.loads(json.dumps(value))
        if not context_only_mutation:
            for key, question in value["questions"].items():
                if key != "technical_quality":
                    continue
                question.pop("instructions", None)
                criteria = question.get("criteria", [])
                if len(criteria) != 5 or any(
                    not text.startswith(str(i)) for i, text in enumerate(criteria, 1)
                ):
                    raise ValueError("Quality rubric must preserve ascending native 1..5 anchors")
                question["criteria"] = [text.split(":", 1)[0] for text in criteria]
        return {key: item for key, item in value.items() if key not in {"context", "id"}}

    if fixed(prompt) != fixed(baseline):
        raise ValueError(
            "Only prompt context and ID may change during optimisation"
            if context_only_mutation
            else "Only context and technical quality rubric descriptions may change; aesthetics, schema and scale are fixed"
        )
    return prompt


def select_split(dataset, split, sample_ids=None):
    if split not in SPLITS:
        raise ValueError("Choose train, dev or test")
    candidates = [sample for sample in dataset["samples"] if sample["split"] == split]
    if sample_ids is None:
        return candidates
    selected = set(sample_ids)
    if (
        not selected
        or len(selected) != len(sample_ids)
        or not selected <= {sample["id"] for sample in candidates}
    ):
        raise ValueError("Sample IDs must be unique members of the requested split")
    groups = {sample["group_id"] for sample in candidates if sample["id"] in selected}
    complete = {sample["id"] for sample in candidates if sample["group_id"] in groups}
    if selected != complete:
        raise ValueError("Evaluate whole source groups or matched pairs")
    return [sample for sample in candidates if sample["id"] in selected]


def ledger_records(path):
    path = local_path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def validate_records(records, protocol, dataset, prompt):
    samples = {sample["id"]: sample for sample in dataset["samples"]}
    for record in records:
        if record["sample"] not in samples:
            raise ValueError("Ledger sample is absent from the frozen dataset")
        if record.get("manifest_sha256") != digest(dataset) or record.get("prompt_sha256") != digest(prompt):
            raise ValueError("Ledger dataset or prompt hash differs from frozen inputs")
        if record.get("task") != protocol["task"] or record.get("arm") != protocol["model_alias"]:
            raise ValueError("Ledger model or task differs from the track")


def freeze_selection(
    protocol_path, candidate_prompt_path, baseline_dev_ledger, candidate_dev_ledger, *, output_path=None
):
    protocol_path = local_path(protocol_path)
    track = json.loads(protocol_path.read_bytes())["track"]
    _, protocol, dataset = read_protocol(track, protocol_path)
    prompts = {
        "baseline": local_path(protocol["baseline_prompt_path"]),
        "candidate": local_path(candidate_prompt_path),
    }
    ledgers = {"baseline": local_path(baseline_dev_ledger), "candidate": local_path(candidate_dev_ledger)}
    evaluations = {}
    for name, prompt_path in prompts.items():
        prompt = validate_candidate(prompt_path, protocol)
        records = ledger_records(ledgers[name])
        validate_records(records, protocol, dataset, prompt)
        if any(
            sample["split"] in {"test", "external"}
            for sample in dataset["samples"]
            if sample["id"] in {record["sample"] for record in records}
        ):
            raise ValueError("Final-test outcomes cannot be used for prompt selection")
        result = summarize(track, select_split(dataset, "dev"), records)
        if result["summary"]["unattempted"]:
            raise ValueError("Prompt selection requires attempted outcomes for every dev sample")
        evaluations[name] = {
            "prompt_path": relative_path(prompt_path),
            "prompt_sha256": file_sha(prompt_path),
            "ledger_path": relative_path(ledgers[name]),
            "ledger_sha256": file_sha(ledgers[name]),
            "score": result["summary"]["objective"]["score"],
        }
    gain = evaluations["candidate"]["score"] - evaluations["baseline"]["score"]
    chosen = "candidate" if gain >= protocol["minimum_gain"] else "baseline"
    result = {
        "version": 1,
        "track": track,
        "selected_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_sha(protocol_path),
        "selection_split": "dev",
        "baseline_prompt_sha256": protocol["baseline_prompt_sha256"],
        "selected": chosen,
        "selected_prompt_path": evaluations[chosen]["prompt_path"],
        "selected_prompt_sha256": evaluations[chosen]["prompt_sha256"],
        "minimum_gain": protocol["minimum_gain"],
        "candidate_gain": gain,
        "evaluations": evaluations,
    }
    write_frozen(output_path or f"tracks/{track}/prompt-selection.json", result)
    return result


def validate_test_selection(path, protocol_path, protocol, prompt_path):
    selection = json.loads(local_path(path).read_bytes())
    if (
        selection.get("protocol_sha256") != file_sha(protocol_path)
        or selection.get("selection_split") != "dev"
    ):
        raise ValueError("Final test requires a selection frozen against this protocol using dev only")
    for evaluation in selection["evaluations"].values():
        for prefix in ("prompt", "ledger"):
            if file_sha(local_path(evaluation[prefix + "_path"])) != evaluation[prefix + "_sha256"]:
                raise ValueError("Frozen prompt-selection evidence changed")
    allowed = {protocol["baseline_prompt_sha256"], selection["selected_prompt_sha256"]}
    if file_sha(prompt_path) not in allowed:
        raise ValueError("Final test permits only baseline or the selected prompt")


def eval_prompt(
    track,
    split,
    prompt_path,
    output_path,
    *,
    protocol_path=None,
    selection_path=None,
    sample_ids=None,
    max_calls=360,
    budget_usd=2.0,
):
    """Evaluate a full split or whole-group minibatch and return scored trace feedback."""
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls < 1:
        raise ValueError("Use a positive request cap")
    protocol_path, protocol, dataset = read_protocol(track, protocol_path)
    number(budget_usd, 0.1, protocol.get("track_budget_usd", 2.0))
    prompt_path, output_path = local_path(prompt_path), local_path(output_path)
    prompt = validate_candidate(prompt_path, protocol)
    samples = select_split(dataset, split, sample_ids)
    if split in {"test", "external"}:
        validate_test_selection(
            selection_path or f"tracks/{track}/prompt-selection.json", protocol_path, protocol, prompt_path
        )
    metadata = {
        "track": track,
        "split": split,
        "protocol_sha256": file_sha(protocol_path),
        "dataset_sha256": protocol["dataset_sha256"],
        "prompt_sha256": file_sha(prompt_path),
    }
    receipt = output_path.with_suffix(output_path.suffix + ".meta.json")
    if receipt.exists():
        if json.loads(receipt.read_bytes()) != metadata:
            raise ValueError("Output ledger belongs to a different frozen evaluation")
    else:
        if output_path.exists():
            raise ValueError("Existing ledger lacks its frozen evaluation receipt")
        write_frozen(receipt, metadata)
    records = ledger_records(output_path)
    validate_records(records, protocol, dataset, prompt)
    run_benchmark(
        SimpleNamespace(
            manifest=local_path(protocol["dataset_path"]),
            prompt=prompt_path,
            task=protocol["task"],
            models=[protocol["model_alias"]],
            output=output_path,
            ids=[sample["id"] for sample in samples],
            limit=None,
            effort="low",
            max_tokens=4096,
            max_calls=max_calls,
            budget_usd=budget_usd,
            reserve_usd=0.10,
            sample_context_mode=protocol.get("context_mode", "image_only"),
        )
    )
    records = ledger_records(output_path)
    validate_records(records, protocol, dataset, prompt)
    result = summarize(track, samples, records)
    return {**result, "ledger_path": str(output_path), "ledger_records": records}


def _reported_cost(record):
    response = record.get("response")
    response = response if isinstance(response, dict) else {}
    usage = record.get("usage") or response.get("usage") or {}
    if usage:
        cost, _ = effective_cost(usage, record.get("cost_usd"))
    else:
        cost = record.get("effective_cost_usd")
    return number(cost, 0, float("inf")) if cost is not None else None


def _reported_spend(records):
    costs = [_reported_cost(record) for record in records]
    if any(cost is None for cost in costs):
        raise ValueError("Track spending is unknown; reconcile it before any final calls")
    return sum(costs)


def track_spending(track):
    """Count canonical optimisation, reflection and final ledgers without trusting estimates."""
    require_restored(ROOT, scope="tracks/" + track + "/")
    directory = ROOT / "tracks" / track
    optimisation = directory / "optimisation"
    summary = json.loads((optimisation / "summary.json").read_bytes())
    task_paths = sorted(optimisation.glob("candidates/*/*.jsonl"))
    reflection_path = optimisation / "reflection.jsonl"
    final_paths = sorted((directory / "final").glob("*.jsonl"))
    tasks = [(path, record, _reported_cost(record)) for path in task_paths for record in ledger_records(path)]
    reflection = [
        (reflection_path, record, _reported_cost(record)) for record in ledger_records(reflection_path)
    ]
    final = [
        (path, record, _reported_cost(record)) for path in final_paths for record in ledger_records(path)
    ]
    try:
        optimisation_costs = cost_totals(tasks + reflection)
        final_costs = cost_totals(final)
    except ValueError as exc:
        raise ValueError("Track spending is unknown; reconcile it before any final calls") from exc
    optimisation_spend = optimisation_costs["actual_total_usd"]
    frozen_costs = summary["costs"]
    frozen_actual = frozen_costs["effective_cost_usd"]
    actual_matches = (frozen_actual is None and optimisation_spend is None) or (
        frozen_actual is not None
        and optimisation_spend is not None
        and abs(number(frozen_actual, 0, float("inf")) - optimisation_spend) <= 1e-9
    )
    if (
        not actual_matches
        or frozen_costs["task_calls"] != len(tasks)
        or frozen_costs["reflection_calls"] != len(reflection)
    ):
        raise ValueError("Optimisation summary and recorded spending differ")
    if optimisation_costs["unknown_cost_count"]:
        for field in (
            "known_reported_usd",
            "reserved_unverified_usd",
            "budget_accounted_usd",
        ):
            if abs(number(frozen_costs.get(field), 0, float("inf")) - optimisation_costs[field]) > 1e-9:
                raise ValueError("Optimisation summary and recorded reservation accounting differ")
        if (
            type(frozen_costs.get("unknown_cost_count")) is not int
            or frozen_costs["unknown_cost_count"] != optimisation_costs["unknown_cost_count"]
        ):
            raise ValueError("Optimisation summary and recorded reservation accounting differ")
    metadata = []
    if TRACKS[track]["version"] == 2:
        metadata = [
            (path, record, _reported_cost(record))
            for path in sorted((directory / "final-metadata").glob("*.jsonl"))
            for record in ledger_records(path)
        ]
    try:
        combined = cost_totals(tasks + reflection + final + metadata)
        metadata_costs = cost_totals(metadata)
    except ValueError as exc:
        raise ValueError("Track spending is unknown; reconcile it before any final calls") from exc
    cap = read_protocol(track)[1].get("track_budget_usd", 2.0)
    # Preserve v1 report bytes: its total was the sum of the two phase totals.
    total = (
        optimisation_spend + final_costs["actual_total_usd"]
        if TRACKS[track]["version"] == 1
        else combined["actual_total_usd"]
    )
    result = {
        "optimisation_calls": len(tasks),
        "reflection_calls": len(reflection),
        "final_calls": len(final),
        "optimisation_usd": optimisation_spend,
        "final_usd": final_costs["actual_total_usd"],
        "total_usd": total,
        "track_budget_usd": cap,
    }
    if TRACKS[track]["version"] == 2:
        result.update(metadata_calls=len(metadata), metadata_usd=metadata_costs["actual_total_usd"])
    if combined["unknown_cost_count"]:
        result.update({key: value for key, value in combined.items() if key != "actual_total_usd"})

    return result


def run_final(track):
    """Alternate baseline/selected order by test group within the remaining track budget."""
    protocol_path, protocol, dataset = read_protocol(track)
    selection_path = ROOT / "tracks" / track / "prompt-selection.json"
    selection = json.loads(selection_path.read_bytes())
    configurations = [("baseline", local_path(protocol["baseline_prompt_path"]))]
    if selection["selected_prompt_sha256"] != protocol["baseline_prompt_sha256"]:
        configurations.append(("selected", local_path(selection["selected_prompt_path"])))
    for _, prompt in configurations:
        validate_test_selection(selection_path, protocol_path, protocol, prompt)
    cap = protocol.get("track_budget_usd", 2.0)
    groups = defaultdict(list)
    samples = [sample for sample in dataset["samples"] if sample["split"] in {"test", "external"}]
    for sample in samples:
        groups[(sample["split"], sample["group_id"])].append(sample["id"])
    for index, ((split, _), identifiers) in enumerate(groups.items()):
        order = configurations if index % 2 == 0 else configurations[::-1]
        for name, prompt in order:
            output = (
                ROOT
                / "tracks"
                / track
                / "final"
                / (f"{name}.jsonl" if split == "test" else f"external-{name}.jsonl")
            )
            records = ledger_records(output)
            validate_records(records, protocol, dataset, json.loads(prompt.read_bytes()))
            done = {record["sample"] for record in records}
            if set(identifiers) <= done:
                continue
            spending = track_spending(track)
            accounted = spending.get("budget_accounted_usd", spending["total_usd"])
            allowance = cap - accounted + _reported_spend(records)
            if accounted + 0.10 > cap:
                raise ValueError("Combined optimisation and final track budget reached")
            eval_prompt(
                track,
                split,
                prompt,
                output,
                selection_path=selection_path,
                sample_ids=identifiers,
                max_calls=len(samples),
                budget_usd=min(cap, allowance),
            )
    return {
        "track": track,
        "spending": track_spending(track),
        "configurations": [name for name, _ in configurations],
        "order": "Alternate baseline/selected order by source group; requests remain sequential",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("freeze")
    create.add_argument("--track", choices=TRACKS, required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--model", choices=MODELS)
    create.add_argument("--context-mode", choices=["image_only", "pixel_stats"], default="image_only")
    create.add_argument("--max-proposals", type=int, default=20)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--track", choices=TRACKS, required=True)
    evaluate.add_argument("--split", choices=sorted(SPLITS), required=True)
    evaluate.add_argument("--prompt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--ids", nargs="+")
    evaluate.add_argument("--max-calls", type=int, default=360)
    evaluate.add_argument("--budget-usd", type=float, default=2.0)
    choose = sub.add_parser("select")
    choose.add_argument("--protocol", type=Path, required=True)
    choose.add_argument("--candidate", type=Path, required=True)
    choose.add_argument("--baseline-ledger", type=Path, required=True)
    choose.add_argument("--candidate-ledger", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--track", choices=TRACKS, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = create_protocol(
            args.track,
            args.baseline,
            model_alias=args.model,
            context_mode=args.context_mode,
            max_proposals=args.max_proposals,
        )
    elif args.command == "select":
        result = freeze_selection(args.protocol, args.candidate, args.baseline_ledger, args.candidate_ledger)
    elif args.command == "final":
        result = run_final(args.track)
    else:
        result = eval_prompt(
            args.track,
            args.split,
            args.prompt,
            args.output,
            sample_ids=args.ids,
            max_calls=args.max_calls,
            budget_usd=args.budget_usd,
        )["summary"]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
