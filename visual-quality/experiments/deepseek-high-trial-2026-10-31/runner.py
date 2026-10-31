"""Repeat the frozen 72-image DeepSeek trial with high reasoning enabled."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

from visual_quality.storage import file_sha256

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data/deepseek-high-trial-2026-10-31"
BASELINE = HERE.parent / "photo-scope-trial-2026-10-31"
BASELINE_DATA = PROJECT / "data/photo-scope-trial-2026-10-31/deepseek/images"
SOURCE_RUNNER = BASELINE / "runner.py"


def load_source(name: str):
    spec = importlib.util.spec_from_file_location(name, SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A fresh module instance makes transport configuration local to this experiment.
source = load_source("deepseek_high_transport")
transport = source.transport
transport.HERE, transport.DATA = HERE, DATA
ROUTES = ("deepseek",)
ARM_ID = "photo_scope"
digest, encoded, now = source.digest, source.encoded, source.now
payload, durable_json = source.payload, source.durable_json
read_run, validate_run = source.read_run, source.validate_run
save_report, verify, execute = source.save_report, source.verify, source.execute


def freeze() -> dict:
    """Pin the prior evidence and change only reasoning and the output allowance."""
    target = transport.DATA / "deepseek/images"
    if (target / "run.json").exists():
        raise ValueError("Run already frozen; use probe, run or verify")
    baseline_source = load_source("deepseek_high_baseline_verification")
    baseline = baseline_source.read_run("deepseek")
    report = baseline_source.verify(baseline)
    if report["status"] != "complete" or report["attempted"] != 72:
        raise ValueError("The 72-image no-reasoning baseline must be complete")
    config = transport.route_entry("deepseek")
    expected = copy.deepcopy(baseline["route_config"])
    expected["settings"].update(reasoning_effort="high", max_tokens=16384)
    if config != expected or baseline["settings"]["reasoning_effort"] != "none":
        raise ValueError("Only none-to-high reasoning and 2048-to-16384 output changes are allowed")
    if baseline["settings"]["max_tokens"] != 2048:
        raise ValueError("Unexpected baseline output allowance")
    references = BASELINE / "references.jsonl"
    reference_freeze = json.loads((BASELINE / "references-freeze.json").read_text())
    if file_sha256(references) != reference_freeze["references_sha256"]:
        raise ValueError("Frozen reference snapshot changed")
    pinned = dict(baseline["inputs"])
    for path in (
        HERE / "runner.py", HERE / "test_runner.py", HERE / "routes.json", HERE / "protocol.md",
        HERE / "runinfra-catalog.json", SOURCE_RUNNER, BASELINE_DATA / "run.json",
        BASELINE_DATA / "observations.jsonl", references, BASELINE / "references-freeze.json",
    ):
        pinned[str(path.resolve())] = file_sha256(path)
    reserve = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"]["max_tokens"]
    ) / 1_000_000
    run = {
        "schema_version": "deepseek-high-run-v1",
        "route": "deepseek", "phase": "images", "created_at": now(),
        "endpoint": config["endpoint"], "requested_model": config["requested_model"],
        "route_config": config, "settings": config["settings"],
        "prompts": copy.deepcopy(baseline["prompts"]), "cases": copy.deepcopy(baseline["cases"]),
        "previous_cost_usd": "0", "training_ready": False,
        "baseline": {
            "run_id": baseline["run_id"], "run_path": str(BASELINE_DATA / "run.json"),
            "observations_path": str(BASELINE_DATA / "observations.jsonl"),
            "reference_path": str(references),
            "reference_sha256": reference_freeze["references_sha256"],
            "manifest_path": str(BASELINE / "manifest.json"),
            "changed_request_fields": {
                "reasoning_effort": {"before": "none", "after": "high"},
                "max_tokens": {"before": 2048, "after": 16384},
            },
        },
        "policy": {
            "automatic_retries": False, "fallback": False, "max_concurrency": 4,
            "reference_kind": "reused_frozen_hybrid_assistant_quality_with_photo_scope",
            "human_gold": False, "targeted_probe_is_included_in_72": True,
            "inference_contract": "Identical frozen photo-scope prompt; no post-hoc score replacement",
            "probe_before_batch": "Inspect P048 response headers and reasoning usage before run command",
        },
        "budget": {
            "max_cost_usd": "2.00", "total_route_caps_usd": "2.00",
            "reserve_per_request_usd": str(reserve),
            "reserve_basis": "full_context_and_max_output",
        },
        "inputs": pinned,
    }
    for case, previous in zip(run["cases"], baseline["cases"], strict=True):
        body = payload(run, case)
        old_body = payload(baseline, previous)
        changes = {key for key in body.keys() | old_body.keys() if body.get(key) != old_body.get(key)}
        if changes != {"reasoning_effort", "max_tokens"}:
            raise ValueError("Unexpected request differences: " + str(changes))
        case.update(request_sha256=digest(body), request_bytes=len(encoded(body)))
        if case["request_bytes"] > config["max_request_bytes"]:
            raise ValueError("Request exceeds provider byte limit: " + case["case_id"])
    run["run_id"] = digest(run)
    durable_json(target / "run.json", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "run", "verify"))
    parser.add_argument("--route", choices=("deepseek", "all"), default="deepseek")
    parser.add_argument("--case-id", default="P048", help="Probe counts as one of the 72 cases")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            report = save_report(freeze())
        else:
            run = read_run("deepseek")
            report = (
                verify(run) if args.command == "verify"
                else execute(run, {args.case_id} if args.command == "probe" else None)
            )
        print(json.dumps({key: value for key, value in report.items() if key != "attempts"}, indent=2))
        if report["status"] == "blocked":
            raise SystemExit(1)
    except (ValueError, OSError) as error:
        print(json.dumps({"route": "deepseek", "status": "error", "error": str(error)}), flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
