"""Durable 2,000-image trial with two high-reasoning routes and bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path

from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data/dual-high-2000-2026-10-31"
SOURCE = HERE.parent / "photo-scope-trial-2026-10-31"
HIGH_SOURCE = HERE.parent / "deepseek-high-trial-2026-10-31"
_spec = importlib.util.spec_from_file_location("dual_high_2000_source", SOURCE / "runner.py")
source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source)
transport = source.transport
# A fresh module instance isolates all writes from the prior frozen experiments.
transport.HERE, transport.DATA = HERE, DATA
ROUTES, TASKS = ("deepseek", "luna"), transport.TASKS
ARM_ID, CASE_COUNT, CHECKPOINT_EVERY = "photo_scope", 2000, 128
digest, encoded, now = transport.digest, transport.encoded, transport.now
payload, durable_json, parse_joint = transport.payload, transport.durable_json, transport.parse_joint
validate_run = transport.validate_run


def manifest_digest(value: object) -> str:
    """Preparation IDs use compact JSON; transport/request IDs use spaced JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_inputs(manifest_path: Path, prompt_path: Path, manifest: dict) -> dict:
    files = [
        HERE / name
        for name in (
            "runner.py",
            "test_runner.py",
            "routes.json",
            "protocol.md",
            "photo-scope.txt",
            "prompt-provenance.json",
        )
    ]
    files += [
        manifest_path,
        prompt_path,
        SOURCE / "runner.py",
        SOURCE / "prompt.json",
        SOURCE / "routes.json",
        HIGH_SOURCE / "routes.json",
        source.TRANSPORT_PATH,
        source.TRANSPORT_PATH.with_name("wire.py"),
        transport.BASE_PATH,
        transport._base_path,
        PROJECT / "src/visual_quality/storage.py",
        PROJECT / "prompts/visual-rating-v1/system.txt",
        PROJECT / "prompts/visual-rating-v1/joint.txt",
    ]
    files += [
        path
        for name in (
            "prepare.py",
            "selection.json",
            "collage-selection.json",
            "preparation-validation.json",
            "provider-research.json",
            "review-instructions.md",
        )
        if (path := HERE / name).exists()
    ]
    if (reference_freeze := HERE / "references-freeze.json").exists():
        references = HERE / "references.jsonl"
        frozen = json.loads(reference_freeze.read_text())
        if file_sha256(references) != frozen["references_sha256"]:
            raise ValueError("Frozen reference snapshot changed")
        files += [references, reference_freeze]
    files += sorted(HERE.glob("*catalog.json")) + sorted(HERE.glob("*endpoints.json"))
    inherited = {**manifest.get("inputs", {}), **manifest.get("packet_files", {})}
    for name, expected in inherited.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("Manifest source or packet changed: " + name)
    return {**inherited, **{str(path.resolve()): file_sha256(path) for path in files}}


def freeze(route: str, manifest_path: Path, prompt_path: Path) -> dict:
    """Freeze the existing prompt and provider settings against 2,000 delivered views."""
    if route not in ROUTES:
        raise ValueError("Unknown route")
    target = transport.DATA / route / "images"
    if (target / "run.json").exists():
        raise ValueError("Run already frozen; use probe, run or verify")
    config = transport.route_entry(route)
    prior = HIGH_SOURCE if route == "deepseek" else SOURCE
    if config != json.loads((prior / "routes.json").read_text())[route]:
        raise ValueError("The existing high-reasoning route settings must be unchanged")
    if prompt_path.read_bytes() != (SOURCE / "prompt.json").read_bytes():
        raise ValueError("The existing photo-scope prompt must be byte-identical")
    for name in ("photo-scope.txt", "prompt-provenance.json"):
        if (HERE / name).read_bytes() != (SOURCE / name).read_bytes():
            raise ValueError("The existing photo-scope policy must be byte-identical")
    prompt = source.load_prompt(prompt_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["manifest_id"] != manifest_digest(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    ):
        raise ValueError("Frozen manifest identity differs")
    prepared = json.loads((HERE / "preparation-validation.json").read_text())
    if (
        prepared.get("verified") is not True
        or prepared.get("images") != CASE_COUNT
        or prepared.get("manifest_id") != manifest["manifest_id"]
    ):
        raise ValueError("The completed 2000-image preparation must be verified before freezing")
    originals = manifest["cases"]
    if len(originals) != CASE_COUNT or len({c["case_id"] for c in originals}) != CASE_COUNT:
        raise ValueError("Exactly 2000 unique case identifiers are required")
    if len({c["view_sha256"] for c in originals}) != CASE_COUNT:
        raise ValueError("The 2000 cases must have distinct delivered image bytes")
    cases = []
    for original in sorted(originals, key=lambda c: digest(["dual-high-2000-order-v1", route, c["case_id"]])):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", original["case_id"]):
            raise ValueError("Unsafe case identifier")
        if original.get("mime_type", "image/png") != "image/png":
            raise ValueError("Frozen wire transport requires PNG views")
        cases.append({**original, "arm_id": ARM_ID, "attempt_id": ARM_ID + "--" + original["case_id"]})
    reserve = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"]["max_tokens"]
    ) / 1_000_000
    run = {
        "schema_version": "dual-high-2000-run-v1",
        "route": route,
        "phase": "images",
        "created_at": now(),
        "endpoint": config["endpoint"],
        "requested_model": config["requested_model"],
        "route_config": config,
        "settings": config["settings"],
        "prompts": {ARM_ID: prompt},
        "cases": cases,
        "previous_cost_usd": "0",
        "training_ready": False,
        "policy": {
            "automatic_retries": False,
            "fallback": False,
            "max_concurrency": 16,
            "reference_kind": "independent_72_case_assistant_subset_frozen_separately",
            "human_gold": False,
            "references_sent_to_models": False,
            "probe_is_included_in_2000": True,
            "inference_contract": "Unchanged frozen photo-scope prompt; no post-hoc model score replacement",
            "checkpoint_every_completions": CHECKPOINT_EVERY,
            "raw_journal_is_authoritative": True,
        },
        "budget": {
            "max_cost_usd": "10.00",
            "total_route_caps_usd": "20.00",
            "reserve_per_request_usd": str(reserve),
            "reserve_basis": "full_context_and_max_output",
        },
        "inputs": frozen_inputs(manifest_path, prompt_path, manifest),
    }
    for case in cases:
        body = payload(run, case)
        case.update(request_sha256=digest(body), request_bytes=len(encoded(body)))
        if case["request_bytes"] > config["max_request_bytes"]:
            raise ValueError("Request exceeds provider byte limit: " + case["case_id"])
    run["run_id"] = digest(run)
    durable_json(target / "run.json", run)
    return run


def read_run(route: str, phase="images") -> dict:
    run = json.loads((transport.DATA / route / phase / "run.json").read_text())
    if route not in ROUTES or run["route"] != route or phase != "images" or run["phase"] != phase:
        raise ValueError("Frozen run location differs")
    validate_run(run, check_payloads=False)
    return run


def read_attempt(run: dict, case: dict) -> tuple[dict, dict | None]:
    folder = transport.directory(run) / "attempts" / case["attempt_id"]
    if not (folder / "request.json").exists():
        raise ValueError("Incomplete attempt directory; inspect manually")
    request = json.loads((folder / "request.json").read_text())
    expected = {
        key: case[key] for key in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")
    }
    if any(request.get(key) != value for key, value in {**expected, "run_id": run["run_id"]}.items()):
        raise ValueError("Attempt request metadata differs from frozen run")
    if not (folder / "result.json").exists():
        return {
            "case_id": case["case_id"],
            "arm_id": case["arm_id"],
            "attempt_id": case["attempt_id"],
            "status": "unknown",
            "cost_usd": None,
            "reason": "interrupted_request",
            "wire_score_valid": None,
            "task_valid": dict.fromkeys(TASKS, False),
        }, None
    result = json.loads((folder / "result.json").read_text())
    detail, row = transport.interpret(run, case, request, result)
    if detail["cost_usd"] is not None and Decimal(detail["cost_usd"]) > Decimal(
        run["budget"]["reserve_per_request_usd"]
    ):
        detail.update(status="failed", reason="reported_cost_exceeds_full_context_reserve")
    effort = result.get("headers", {}).get("x-runinfra-reasoning-effort")
    if run["route"] == "deepseek" and effort is not None and effort != "high":
        detail.update(status="failed", reason="unexpected_returned_reasoning_effort")
    return detail, row


def summarize(run: dict, attempts: list[dict], observations: list[dict]) -> dict:
    stopped = any(a["status"] in {"failed", "unknown"} or a["cost_usd"] is None for a in attempts)
    spend = sum((Decimal(a["cost_usd"]) for a in attempts if a["cost_usd"] is not None), Decimal(0))
    total = spend + Decimal(run["previous_cost_usd"])
    stopped |= total > Decimal(run["budget"]["max_cost_usd"])
    usage = Counter()
    for row in observations:
        raw = row["provenance"]["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw.get(key)
            if type(value) is int and value >= 0:
                usage[key] += value
        details = raw.get("completion_tokens_details")
        value = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(value) is int and value >= 0:
            usage["reasoning_tokens"] += value
            usage["responses_reporting_reasoning_tokens"] += 1
    return {
        "schema_version": "dual-high-2000-report-v1",
        "run_id": run["run_id"],
        "route": run["route"],
        "phase": run["phase"],
        "status": "blocked" if stopped else "complete" if len(attempts) == len(run["cases"]) else "partial",
        "planned": len(run["cases"]),
        "attempted": len(attempts),
        "valid_observations": sum(a["status"] == "valid" for a in attempts),
        "invalid_observations": sum(a["status"] == "invalid" for a in attempts),
        "task_valid_observations": {t: sum(a["task_valid"][t] for a in attempts) for t in TASKS},
        "scored": {t: sum(r[t]["score"] is not None for r in observations) for t in TASKS},
        "known_cost_usd": str(spend),
        "total_known_cost_usd": str(total),
        "unpriced_attempts": sum(a["cost_usd"] is None for a in attempts),
        "usage_totals": dict(sorted(usage.items())),
        "score_distributions": {
            t: dict(
                sorted(
                    Counter(
                        format(Decimal(str(r[t]["score"])), ".2f")
                        for r in observations
                        if r[t]["score"] is not None
                    ).items()
                )
            )
            for t in TASKS
        },
        "photo_scope_flagged_cases": sorted(
            r["case_id"]
            for r in observations
            if any("unsupported_domain" in r[t].get("limitations", []) for t in TASKS)
        ),
        "attempts": attempts,
        "training_ready": False,
    }


def reconstruct(run: dict, *, check_payloads=True) -> tuple[dict, list[dict]]:
    validate_run(run, check_payloads=check_payloads)
    root = transport.directory(run) / "attempts"
    expected = {case["attempt_id"] for case in run["cases"]}
    if root.exists() and not {path.name for path in root.iterdir()}.issubset(expected):
        raise ValueError("Unexpected attempt directory")
    attempts, observations = [], []
    for case in run["cases"]:
        if not (root / case["attempt_id"]).exists():
            continue
        detail, row = read_attempt(run, case)
        attempts.append(detail)
        if row is not None:
            observations.append(row)
    return summarize(run, attempts, observations), observations


def save_report(run: dict, *, check_payloads=True, state=None) -> dict:
    if state is None:
        report, rows = reconstruct(run, check_payloads=check_payloads)
    else:
        attempts, observations = state
        ordered = [attempts[c["attempt_id"]] for c in run["cases"] if c["attempt_id"] in attempts]
        rows = [observations[c["attempt_id"]] for c in run["cases"] if c["attempt_id"] in observations]
        report = summarize(run, ordered, rows)
    write_jsonl(transport.directory(run) / "observations.jsonl", rows)
    atomic_json(transport.directory(run) / "report.json", report)
    return report


def admission_count(run: dict, known_cost: Decimal, in_flight: int, pending: int) -> int:
    remaining = Decimal(run["budget"]["max_cost_usd"]) - known_cost
    reserve = Decimal(run["budget"]["reserve_per_request_usd"])
    if reserve <= 0:
        raise ValueError("A positive full-context admission reserve is required")
    affordable = int(max(remaining, Decimal(0)) / reserve) - in_flight
    return max(0, min(min(16, run["policy"]["max_concurrency"]) - in_flight, affordable, pending))


def progress(report: dict) -> None:
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
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


def execute(run: dict, case_ids: set[str] | None = None) -> dict:
    """Stream single attempts, reserving the worst case for every request in flight."""
    validate_run(run)
    known_ids = {case["case_id"] for case in run["cases"]}
    if case_ids is not None and not case_ids <= known_ids:
        raise ValueError("Unknown selected case identifiers: " + ", ".join(sorted(case_ids - known_ids)))
    directory = transport.directory(run)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial, rows = reconstruct(run, check_payloads=False)
        attempts = {row["attempt_id"]: row for row in initial["attempts"]}
        observations = {row["attempt_id"]: row for row in rows}
        state = (attempts, observations)
        if initial["status"] == "blocked":
            save_report(run, state=state)
            raise ValueError("Existing failed or uncertain attempts prevent further calls")
        pending = deque(
            case
            for case in run["cases"]
            if case["attempt_id"] not in attempts and (case_ids is None or case["case_id"] in case_ids)
        )
        key = os.environ.get(run["route_config"]["key_env"])
        if pending and not key:
            raise ValueError(run["route_config"]["key_env"] + " is unavailable")
        # Raw result files are persisted per completion. Aggregate exports are checkpoints,
        # rebuilt from raw files on every process start and on explicit verification.
        try:
            _stream(run, pending, key, state, Decimal(initial["total_known_cost_usd"]))
        except BaseException:
            save_report(run, check_payloads=False)
            raise
        return save_report(run, state=state)


def _stream(run: dict, pending: deque, key: str | None, state: tuple, spent: Decimal) -> None:
    attempts, observations = state
    active, stopped, completed = {}, False, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, run["policy"]["max_concurrency"])) as pool:
        while pending or active:
            admitted = 0 if stopped else admission_count(run, spent, len(active), len(pending))
            for _ in range(admitted):
                case = pending.popleft()
                folder = transport.directory(run) / "attempts" / case["attempt_id"]
                folder.mkdir(parents=True)
                request = {
                    name: case[name]
                    for name in (
                        "case_id",
                        "arm_id",
                        "attempt_id",
                        "request_sha256",
                        "request_bytes",
                    )
                }
                durable_json(
                    folder / "request.json", {**request, "run_id": run["run_id"], "created_at": now()}
                )
                active[pool.submit(transport.send, run, case, key)] = case
            if not active:
                if pending and not stopped:
                    raise ValueError("Next request exceeds the admission budget")
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                case = active.pop(future)
                try:
                    future.result()
                except Exception:  # noqa: BLE001 — a journaled local failure cannot be resent.
                    stopped = True
                detail, row = read_attempt(run, case)
                attempts[case["attempt_id"]] = detail
                if row is not None:
                    observations[case["attempt_id"]] = row
                if detail["cost_usd"] is not None:
                    spent += Decimal(detail["cost_usd"])
                stopped |= detail["status"] in {"failed", "unknown"} or detail["cost_usd"] is None
                stopped |= spent > Decimal(run["budget"]["max_cost_usd"])
                completed += 1
            if completed // CHECKPOINT_EVERY > (completed - len(done)) // CHECKPOINT_EVERY:
                validate_run(run, check_payloads=False)
                progress(save_report(run, state=state))
            elif completed // 16 > (completed - len(done)) // 16:
                progress(summarize(run, list(attempts.values()), list(observations.values())))


def verify(run: dict) -> dict:
    report, rows = reconstruct(run)
    directory = transport.directory(run)
    if (
        json.loads((directory / "report.json").read_text()) != report
        or read_jsonl(directory / "observations.jsonl") != rows
    ):
        raise ValueError("Saved exports differ from preserved replies")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "run", "verify"))
    parser.add_argument("--route", choices=(*ROUTES, "all"), required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--prompt", type=Path, default=HERE / "prompt.json")
    parser.add_argument(
        "--case-id", action="append", help="Repeat to select probe cases; each counts within 2000"
    )
    args = parser.parse_args()
    if args.command == "probe" and not args.case_id:
        parser.error("probe requires at least one --case-id from the frozen manifest")
    if args.case_id and args.command != "probe":
        parser.error("--case-id only applies to probe")

    def operate(route):
        if args.command == "freeze":
            report = save_report(freeze(route, args.manifest, args.prompt), check_payloads=False)
        else:
            run = read_run(route)
            report = (
                verify(run)
                if args.command == "verify"
                else execute(run, set(args.case_id) if args.case_id else None)
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
            except Exception as error:  # noqa: BLE001 — isolate provider routes at the CLI boundary.
                failed = True
                print(
                    json.dumps({"route": futures[future], "status": "error", "error": str(error)}), flush=True
                )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
