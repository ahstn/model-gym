"""Frozen 72-image photo-scope trial, reusing the previous durable transport."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from decimal import Decimal
from pathlib import Path

from visual_quality.storage import file_sha256

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data/photo-scope-trial-2026-10-31"
TRANSPORT_PATH = HERE.parent / "prompt-ablation-2026-10-30/runner.py"
_spec = importlib.util.spec_from_file_location("photo_scope_frozen_transport", TRANSPORT_PATH)
transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transport)
# This is a fresh module instance; no old files or running experiments are changed.
# Its public transport functions read these two configuration locations at runtime.
transport.HERE, transport.DATA = HERE, DATA

ROUTES, TASKS = transport.ROUTES, transport.TASKS
digest, encoded, now = transport.digest, transport.encoded, transport.now
durable_json, payload = transport.durable_json, transport.payload
read_run, validate_run = transport.read_run, transport.validate_run
save_report, verify = transport.save_report, transport.verify
parse_joint = transport.parse_joint
ARM_ID = "photo_scope"


def load_prompt(path: Path) -> dict:
    prompt = json.loads(path.read_text())
    if set(prompt) != {"version", "system", "user", "response_format", "sha256"}:
        raise ValueError("Prompt must use the existing five-field bundle format")
    unhashed = {key: value for key, value in prompt.items() if key != "sha256"}
    if hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest() != prompt["sha256"]:
        raise ValueError("Prompt bundle digest differs")
    if not all(
        isinstance(prompt[name], str) and prompt[name].strip() for name in ("version", "system", "user")
    ):
        raise ValueError("Prompt text and version must be nonempty strings")
    baseline = transport.prompt_bundle(
        PROJECT / "prompts/visual-rating-v1", {"id": "baseline", "addendum": ""}
    )
    if prompt["response_format"] != baseline["response_format"]:
        raise ValueError("This trial must retain the prior schema-constrained wire format")
    return prompt


def freeze(route: str, manifest_path: Path, prompt_path: Path) -> dict:
    target = transport.DATA / route / "images"
    if (target / "run.json").exists():
        raise ValueError("Run already frozen; use run or verify")
    config = transport.route_entry(route)
    prompt = load_prompt(prompt_path)
    manifest = json.loads(manifest_path.read_text())
    originals = manifest["cases"]
    if len(originals) != 72 or len({case["case_id"] for case in originals}) != 72:
        raise ValueError("Exactly 72 unique case identifiers are required")
    if len({case["view_sha256"] for case in originals}) != 72:
        raise ValueError("The 72 cases must have distinct delivered image bytes")
    cases = []
    for original in sorted(
        originals, key=lambda case: digest(["photo-scope-order-v1", route, case["case_id"]])
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", original["case_id"]):
            raise ValueError("Case identifier is unsafe")
        if original.get("mime_type", "image/png") != "image/png":
            raise ValueError("Frozen wire transport requires PNG views")
        cases.append({**original, "arm_id": ARM_ID, "attempt_id": ARM_ID + "--" + original["case_id"]})
    files = [
        HERE / "runner.py",
        HERE / "test_runner.py",
        HERE / "routes.json",
        manifest_path,
        prompt_path,
        HERE / "photo-scope.txt",
        HERE / "build_prompt.py",
        HERE / "prompt-provenance.json",
        TRANSPORT_PATH,
        TRANSPORT_PATH.with_name("wire.py"),
        transport.BASE_PATH,
        transport._base_path,
        PROJECT / "src/visual_quality/storage.py",
        PROJECT / "prompts/visual-rating-v1/system.txt",
        PROJECT / "prompts/visual-rating-v1/joint.txt",
    ]
    files += [
        path
        for name in ("extend_manifest.py", "extension-selection.json", "preparation-validation.json")
        if (path := HERE / name).exists()
    ]
    reserve = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"]["max_tokens"]
    ) / 1_000_000
    run = {
        "schema_version": "photo-scope-run-v1",
        "route": route,
        "phase": "images",
        "created_at": now(),
        "endpoint": config["endpoint"],
        "requested_model": config["requested_model"],
        "route_config": config,
        "prompts": {ARM_ID: prompt},
        "settings": config["settings"],
        "cases": cases,
        "previous_cost_usd": "0",
        "policy": {
            "automatic_retries": False,
            "fallback": False,
            "max_concurrency": 4,
            "reference_kind": "no_new_blind_reference",
            "human_gold": False,
            "targeted_probe_is_included_in_72": True,
            "inference_contract": "Frozen prompt applies photo-domain policy; no post-hoc score replacement",
        },
        "budget": {
            "max_cost_usd": "2.00",
            "total_route_caps_usd": "8.00",
            "reserve_per_request_usd": str(reserve),
            "reserve_basis": "full_context_and_max_output",
        },
        "inputs": {str(path.resolve()): file_sha256(path) for path in files},
        "training_ready": False,
    }
    for case in cases:
        body = payload(run, case)
        case.update(request_sha256=digest(body), request_bytes=len(encoded(body)))
        if case["request_bytes"] > config["max_request_bytes"]:
            raise ValueError("Request exceeds provider byte limit: " + case["case_id"])
    run["run_id"] = digest(run)
    durable_json(target / "run.json", run)
    return run


def execute(run: dict, case_ids: set[str] | None = None) -> dict:
    """Send only unattempted selected cases; never repeat a completed or uncertain call."""
    validate_run(run)
    known_ids = {case["case_id"] for case in run["cases"]}
    if case_ids is not None and not case_ids <= known_ids:
        raise ValueError("Unknown selected case identifiers: " + ", ".join(sorted(case_ids - known_ids)))
    key = os.environ.get(run["route_config"]["key_env"])
    if not key:
        raise ValueError(run["route_config"]["key_env"] + " is unavailable")
    directory = transport.directory(run)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = save_report(run, check_payloads=False)
        if report["status"] == "blocked":
            raise ValueError("Existing failed or uncertain attempts prevent further calls")
        pending = [
            case
            for case in run["cases"]
            if (case_ids is None or case["case_id"] in case_ids)
            and not (directory / "attempts" / case["attempt_id"]).exists()
        ]
        concurrency = min(4, run["policy"]["max_concurrency"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pending:
                remaining = Decimal(run["budget"]["max_cost_usd"]) - Decimal(report["total_known_cost_usd"])
                reserve = Decimal(run["budget"]["reserve_per_request_usd"])
                admitted = min(
                    concurrency, len(pending), int(remaining / reserve) if reserve else len(pending)
                )
                if admitted < 1:
                    raise ValueError("Next request exceeds the admission budget")
                batch = pending[:admitted]
                for case in batch:
                    folder = directory / "attempts" / case["attempt_id"]
                    folder.mkdir(parents=True)
                    request = {
                        name: case[name]
                        for name in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")
                    }
                    durable_json(
                        folder / "request.json", {**request, "run_id": run["run_id"], "created_at": now()}
                    )
                futures = [pool.submit(transport.send, run, case, key) for case in batch]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                report = save_report(run, check_payloads=False)
                print(
                    json.dumps(
                        {
                            name: report[name]
                            for name in (
                                "route",
                                "attempted",
                                "scored",
                                "invalid_observations",
                                "total_known_cost_usd",
                                "status",
                            )
                        }
                    ),
                    flush=True,
                )
                if report["status"] == "blocked":
                    break
                pending = pending[admitted:]
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "run", "verify"))
    parser.add_argument("--route", choices=(*ROUTES, "all"), required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--prompt", type=Path, default=HERE / "prompt.json")
    parser.add_argument(
        "--case-id", default="P048", help="Case to send in probe mode, included in the 72-case run"
    )
    args = parser.parse_args()

    def operate(route):
        if args.command == "freeze":
            report = save_report(freeze(route, args.manifest, args.prompt))
        else:
            run = read_run(route)
            report = (
                verify(run)
                if args.command == "verify"
                else execute(run, {args.case_id} if args.command == "probe" else None)
            )
        return {key: value for key, value in report.items() if key != "attempts"}

    routes = ROUTES if args.route == "all" else (args.route,)
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(routes)) as pool:
        futures = {pool.submit(operate, route): route for route in routes}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(json.dumps(result, indent=2), flush=True)
                failed |= result["status"] == "blocked"
            except Exception as error:  # noqa: BLE001 — isolate paid routes at the CLI boundary
                failed = True
                print(
                    json.dumps({"route": futures[future], "status": "error", "error": str(error)}), flush=True
                )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
