"""Run Jev on the user-selected latest 100 quality and 160 AI test cases."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import DECISIONS, digest, image_part, validate_prompt
from bench_runner import effective_cost
from tracks.jev14.client import CONTEXT_TOKENS, HEADERS, MODEL, REVISION, make_payload, post
from tracks.jev14.rounding import normalize
from tracks.jev14.run import ROOT, TRACK, case_record, sha, write_new

DATA = ROOT / "data/jev14-latest-v1"
PROTOCOL = TRACK / "latest/protocol.json"
MANIFEST = DATA / "manifest.json"
LEDGER = DATA / "attempts.jsonl"
ROUNDING_ERRORS = {
    None,
    "ValueError: Probability distribution does not sum to one",
    "ValueError: Score differs from its probability-weighted mean",
}
DATASETS = {"quality": "tracks/quality_v2/dataset.json", "ai": "tracks/metadata_v3/dataset.json"}
BASELINES = {
    "quality": "tracks/quality_v2/final/selected.jsonl",
    "ai": "data/metadata-v3/runs/image_only/final/selected.jsonl",
}


def freeze():
    if PROTOCOL.exists() or MANIFEST.exists():
        raise ValueError("Latest trial is already frozen")
    selection_path = ROOT / "data/metadata-v3/runs/image_only/selection.json"
    selection = json.loads(selection_path.read_text())
    ai_prompt = Path(selection["selected_prompt_path"])
    if not ai_prompt.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Selected AI prompt is outside the study")
    prompts = {"quality": ROOT / "tracks/quality_v2/prompts/selected.json", "ai": ai_prompt}
    if digest(json.loads(ai_prompt.read_text())) != selection["selected_prompt_sha256"]:
        raise ValueError("Selected AI prompt changed")
    cases, sources = (
        [],
        [
            selection_path,
            ROOT / "tracks/quality_v2/protocol.json",
            ROOT / "tracks/metadata_v3/protocol.json",
            ROOT / "tracks/quality_v2/selection.json",
        ],
    )
    for task, dataset_path in DATASETS.items():
        prompt = json.loads(prompts[task].read_text())
        validate_prompt(prompt, task)
        samples = [
            sample
            for sample in json.loads((ROOT / dataset_path).read_text())["samples"]
            if sample["split"] == "test"
        ]
        expected = 100 if task == "quality" else 160
        baseline = [json.loads(line) for line in (ROOT / BASELINES[task]).read_text().splitlines() if line]
        by_id = {row["sample"]: row for row in baseline}
        if len(samples) != expected or len(baseline) != expected or set(by_id) != {s["id"] for s in samples}:
            raise ValueError("Latest baseline and test cohort differ")
        for sample in samples:
            image_part(sample, ROOT)
            row = by_id[sample["id"]]
            if row["delivered_sha256"] != sample["delivered_sha256"] or row["prompt_sha256"] != digest(
                prompt
            ):
                raise ValueError("Baseline image or prompt differs")
            case = case_record(sample, task, "primary")
            if not case["eligible"]:
                raise ValueError("Latest image is outside Jev's limits")
            case.update(
                benchmark=f"jev14-latest-{task}",
                source_dataset_path=dataset_path,
                gold=sample["labels"]["quality_mos_1to5"] if task == "quality" else sample["gold"],
                group_id=sample["group_id"],
                source=sample.get("source", {}).get("generator") or sample.get("source", {}).get("dataset"),
            )
            cases.append(case)
        sources += [ROOT / dataset_path, ROOT / BASELINES[task], prompts[task]]
    cases.sort(key=lambda case: digest(case["case_id"]))
    if Counter(case["task"] for case in cases) != {"quality": 100, "ai": 160}:
        raise ValueError("Unexpected case allocation")
    prior_ledger = ROOT / "data/jev14-v1/attempts.jsonl"
    prior = [json.loads(line) for line in prior_ledger.read_text().splitlines() if line]
    unknown = [row for row in prior if row["effective_cost_usd"] is None]
    if len(unknown) != 1 or unknown[0].get("status") != 400:
        raise ValueError("Prior unpriced attempt needs a separate review")
    sources += [prior_ledger, ROOT / "data/jev14-v1/research/openrouter-decisions-models.json"]
    write_new(MANIFEST, {"cases": cases})
    code = [
        Path(__file__),
        TRACK / "client.py",
        TRACK / "rounding.py",
        TRACK / "run.py",
        ROOT / "bench_protocol.py",
        ROOT / "bench_runner.py",
    ]
    protocol = {
        "version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "expected_revision": REVISION,
        "request_headers": HEADERS,
        "context_tokens": CONTEXT_TOKENS,
        "manifest_sha256": sha(MANIFEST),
        "maximum_attempts": 260,
        "budget_usd": 2,
        "prior_known_cost_usd": sum(row["effective_cost_usd"] or 0 for row in prior),
        "prior_unknown_cost_reserve_usd": 0.012,
        "prior_cost_note": "One HTTP400 charge is unreported; reserve is a guard, not its actual cost",
        "reserve_per_request_usd": 0.012,
        "minimum_interval_seconds": 1,
        "source_inputs": {str(path.relative_to(ROOT)): {"sha256": sha(path)} for path in sources},
        "execution_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "prompts": {task: str(path.relative_to(ROOT)) for task, path in prompts.items()},
        "baselines": BASELINES,
        "datasets": DATASETS,
        "counts": {"quality": 100, "ai": 160},
        "ai_threshold": 0.5,
        "low_quality_threshold": 2.6,
        "view": "Exact frozen bytes: quality1024x768JPEG; AI512x512JPEG. No metadata text or resizing.",
        "prompt_policy": "Transfer the selected Decider prompts unchanged; no Jev-specific optimisation",
        "parsing": "Bounded two-decimal probability rounding; preserve native scores and probabilities",
        "retry_policy": "No retries; retain and stop on any failed or unpriced attempt",
        "order": "SHA256 of case ID, independent of labels and results",
    }
    write_new(PROTOCOL, protocol)
    return {"protocol_sha256": sha(PROTOCOL), "cases": len(cases)}


def review(row, prompt):
    result = dict(row)
    result["strict_normalized"] = result.pop("normalized", None)
    result["strict_validation_error"] = result.pop("validation_error", None)
    result["normalized"] = None
    try:
        if (
            result.get("status") != 200
            or result.get("resolved_model") != REVISION
            or result["strict_validation_error"] not in ROUNDING_ERRORS
        ):
            raise ValueError("HTTP, model or non-rounding validation failed")
        result["normalized"] = normalize(result["response"], prompt["questions"])
    except (TypeError, ValueError, KeyError) as exc:
        result["validation_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def load_study():
    protocol = json.loads(PROTOCOL.read_text())
    pins = {path: pin["sha256"] for path, pin in protocol["source_inputs"].items()}
    pins.update(protocol["execution_sha256"])
    if sha(MANIFEST) != protocol["manifest_sha256"] or any(sha(ROOT / p) != h for p, h in pins.items()):
        raise ValueError("A frozen latest-trial input changed")
    cases = json.loads(MANIFEST.read_text())["cases"]
    prompts = {task: json.loads((ROOT / path).read_text()) for task, path in protocol["prompts"].items()}
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line] if LEDGER.exists() else []
    if len(rows) > len(cases):
        raise ValueError("Attempt limit exceeded")
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        prompt = prompts[case["task"]]
        payload = make_payload(ROOT, case, prompt)
        request_hash = digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload})
        if (
            row["case_id"] != case["case_id"]
            or row["attempt"] != index
            or row["protocol_sha256"] != sha(PROTOCOL)
            or row["request_sha256"] != request_hash
            or row["manifest_sha256"] != protocol["manifest_sha256"]
            or row["prompt_sha256"] != digest(prompt)
            or row.get("status") != 200
            or row.get("resolved_model") != REVISION
            or row.get("validation_error")
            or row.get("transport_error")
            or row.get("cost_validation_error")
            or row.get("strict_validation_error") not in ROUNDING_ERRORS
        ):
            raise ValueError("Changed or failed prior call; review before continuing")
        if row["normalized"] != normalize(row["response"], prompt["questions"]) or row["usage"] != row[
            "response"
        ].get("usage"):
            raise ValueError("Saved prediction or usage differs from raw response")
        cost, basis = effective_cost(row["usage"])
        if cost is None or row["effective_cost_usd"] != cost or row["cost_basis"] != basis:
            raise ValueError("Unknown or changed request cost")
    return protocol, cases, prompts, rows


def run(max_new):
    protocol, cases, prompts, rows = load_study()
    if not 1 <= max_new <= protocol["maximum_attempts"]:
        raise ValueError("Invalid request count")
    if any(sha(ROOT / case["image"]) != case["delivered_sha256"] for case in cases):
        raise ValueError("Frozen image changed")
    spent = sum(row["effective_cost_usd"] for row in rows)
    allowance = protocol["prior_known_cost_usd"] + protocol["prior_unknown_cost_reserve_usd"]
    for case in cases[len(rows) : len(rows) + max_new]:
        if spent + allowance + protocol["reserve_per_request_usd"] > protocol["budget_usd"]:
            raise ValueError("Local budget guard reached")
        if rows:
            previous = datetime.fromisoformat(rows[-1]["timestamp"]).timestamp()
            time.sleep(max(0, protocol["minimum_interval_seconds"] - (time.time() - previous)))
        prompt = prompts[case["task"]]
        row = review(post(make_payload(ROOT, case, prompt), prompt), prompt)
        row.update({key: value for key, value in case.items() if key not in {"image", "eligible", "gold"}})
        row.update(
            arm="jev14",
            attempt=len(rows) + 1,
            prompt_id=prompt["id"],
            prompt_sha256=digest(prompt),
            protocol_sha256=sha(PROTOCOL),
            manifest_sha256=protocol["manifest_sha256"],
        )
        with LEDGER.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        print(
            json.dumps(
                {
                    "attempt": len(rows),
                    "case": case["case_id"],
                    "status": row.get("status"),
                    "valid": row["normalized"] is not None,
                    "seconds": row["seconds"],
                    "cost": row["effective_cost_usd"],
                    "input_tokens": row["usage"].get("input_tokens"),
                }
            ),
            flush=True,
        )
        if row.get("validation_error") or row["effective_cost_usd"] is None:
            return {"status": "stopped", "attempts": len(rows)}
        spent += row["effective_cost_usd"]
    return {
        "status": "complete" if len(rows) == len(cases) else "partial",
        "attempts": len(rows),
        "reported_cost_usd": spent,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--max-new", type=int, default=260)
    args = parser.parse_args()
    print(json.dumps(freeze() if args.command == "freeze" else run(args.max_new), indent=2))


if __name__ == "__main__":
    main()
