"""Durable, bounded API execution for the 4,000-image rehearsal and stats ablation.

Freeze and run rehearsal/benchmark phases separately; each route shares one budget.
The raw per-attempt journal is authoritative. No score or ambiguous failure is retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import ssl
import sys
import time
from collections import Counter, deque
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import requests

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01")
ROUTES = ("deepseek", "luna", "jev", "liquid")
PHASES = ("rehearsal", "benchmark")
PARTITIONS = {"development", "representative", "challenge", "rehearsal"}
SCHEMA = "rehearsal-api-v1"


def runtime() -> dict:
    return {
        "python": sys.version,
        "requests": requests.__version__,
        "urllib3": requests.packages.urllib3.__version__,
        "openssl": ssl.OPENSSL_VERSION,
    }


def now() -> str:
    return datetime.now(UTC).isoformat()


def read(path: Path) -> dict:
    return contract.load_json(path.read_text())


def write(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path if exclusive else path.with_name(path.name + ".tmp")
    with temporary.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def directory(route: str, phase: str | None = None) -> Path:
    root = DATA / "results/api" / route
    return root / phase if phase else root


def frozen_references(manifest_path: Path, manifest: dict, freeze_path: Path) -> dict:
    freeze = read(freeze_path)
    expected = {r["case_id"] for r in manifest["cases"] if r["benchmark_partition"] != "rehearsal"}
    references = Path(freeze["references_path"])
    rubric = Path(freeze["rubric_path"])
    rows = [contract.load_json(line) for line in references.read_text().splitlines() if line.strip()]
    if (
        len(expected) != 500
        or len(rows) != 500
        or {r["case_id"] for r in rows} != expected
        or set(freeze["case_ids"]) != expected
        or len(freeze["case_ids"]) != 500
        or contract.file_sha(references) != freeze["references_sha256"]
        or contract.file_sha(manifest_path) != freeze["manifest_sha256"]
        or contract.file_sha(rubric) != freeze["rubric_sha256"]
        or contract.file_sha(HERE / "api-teacher-prompt.json") != freeze["teacher_prompt_sha256"]
    ):
        raise ValueError("All 500 matching benchmark references must be frozen before calls")
    timestamp = datetime.fromisoformat(freeze["frozen_at"])
    if timestamp.tzinfo is None or timestamp > datetime.now(UTC):
        raise ValueError("Reference freeze time must be a past timezone-aware timestamp")
    return {
        str(references.resolve()): contract.file_sha(references),
        str(freeze_path.resolve()): contract.file_sha(freeze_path),
        str(rubric.resolve()): contract.file_sha(rubric),
    }


def stats_inputs(cases: list[dict], path: Path, donor_path: Path) -> tuple[dict, dict]:
    rows = [contract.load_json(line) for line in path.read_text().splitlines() if line.strip()]
    stats = {r["case_id"]: r["text"] for r in rows}
    donors = read(donor_path)
    if isinstance(donors.get("donors"), dict):
        donors = donors["donors"]
    by_id = {r["case_id"]: r for r in cases}
    benchmark = {r["case_id"] for r in cases if r["benchmark_partition"] != "rehearsal"}
    if len(rows) != len(stats) or set(donors) != benchmark:
        raise ValueError("Unique stats rows and one donor for each benchmark case required")
    for case_id, donor_id in donors.items():
        if (
            donor_id not in by_id
            or by_id[case_id]["family_id"] == by_id[donor_id]["family_id"]
            or case_id not in stats
            or donor_id not in stats
        ):
            raise ValueError("Stats donor is missing or from the same family")
    for text in stats.values():
        if not isinstance(text, str) or not text.strip() or "Frame area close to peak detail" in text:
            raise ValueError("Stats blocks must be nonempty and omit the misleading focus-area line")
    return stats, donors


def freeze(
    route: str, phase: str, manifest_path: Path, references: Path, stats_path: Path, donor_path: Path
) -> dict:
    target = directory(route, phase) / "plan.json"
    if target.exists():
        raise ValueError("Phase already frozen; use run or verify")
    manifest = read(manifest_path)
    cases = manifest["cases"]
    if (
        len(cases) != 4000
        or len({r["case_id"] for r in cases}) != 4000
        or len({r["view_sha256"] for r in cases}) != 4000
        or any(r["benchmark_partition"] not in PARTITIONS for r in cases)
        or Counter(r["benchmark_partition"] for r in cases)
        != {"rehearsal": 3500, "development": 100, "representative": 300, "challenge": 100}
    ):
        raise ValueError("Manifest must contain the agreed 4,000 cases and 100/300/100 benchmark")
    config = read(HERE / "api-routes.json")[route]
    files = [
        manifest_path,
        HERE / "api_runner.py",
        HERE / "api_contract.py",
        HERE / "test_api.py",
        HERE / "api-routes.json",
        HERE / "api-teacher-prompt.json",
        HERE / "api-decision-prompts.json",
        HERE / "api-provider-snapshot.json",
        HERE / "protocol.md",
        contract.WIRE_PATH,
    ]
    references_pins = frozen_references(manifest_path, manifest, references) if phase == "benchmark" else {}
    use_stats = phase == "benchmark" and config["kind"] == "chat"
    stats, donors = stats_inputs(cases, stats_path, donor_path) if use_stats else ({}, {})
    if use_stats:
        files += [stats_path, donor_path]
    selected = [r for r in cases if (r["benchmark_partition"] == "rehearsal") == (phase == "rehearsal")]
    if len({r["family_id"] for r in cases if r["benchmark_partition"] != "rehearsal"}) != 500:
        raise ValueError("Benchmark must have 500 distinct source families")
    selected_ids = {r["case_id"] for r in selected}
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", identity) for identity in selected_ids):
        raise ValueError("Unsafe case identifier")
    reserve = (
        Decimal(config["input_price_per_million"]) * config["context_length"]
        + Decimal(config["output_price_per_million"]) * config["settings"].get("max_tokens", 0)
    ) / 1_000_000
    plan = {
        "schema": SCHEMA,
        "route": route,
        "phase": phase,
        "created_at": now(),
        "route_config": config,
        "teacher_prompt": read(HERE / "api-teacher-prompt.json"),
        "decision_prompt": read(HERE / "api-decision-prompts.json"),
        "reserve_per_attempt_usd": str(reserve),
        "training_ready": False,
        "runtime": runtime(),
        "inputs": {**{str(p.resolve()): contract.file_sha(p) for p in files}, **references_pins},
        "policy": {
            "transport_retries": False,
            "invalid_score_retries": False,
            "capacity_retry_code": "hosted_shared_model_concurrency_limit",
            "fixed_concurrency": config["concurrency"],
            "latency_excludes_scheduler_wait": True,
            "latency_includes": "HTTP invocation through complete response body, including upload/network/provider queue",
            "budget_scope": "all phases and attempts of this route",
            "reference_scores_sent": False,
        },
        "jobs": [],
    }
    for case in selected:
        for arm in contract.ARMS if use_stats else ("baseline",):
            donor = donors[case["case_id"]] if arm == "swapped_stats" else case["case_id"]
            job = {
                "case": case,
                "arm": arm,
                "job_id": arm + "--" + case["case_id"],
                "stats_case_id": donor if arm != "baseline" else None,
                "stats_text": stats[donor] if arm != "baseline" else None,
            }
            body = contract.encoded(contract.payload(plan, job))
            job.update(request_sha256=hashlib.sha256(body).hexdigest(), request_bytes=len(body))
            if len(body) > config["max_request_bytes"]:
                raise ValueError("Provider request size exceeded: " + job["job_id"])
            plan["jobs"].append(job)
    plan["jobs"].sort(
        key=lambda j: contract.digest(["rehearsal-api-order-20261101", route, phase, j["job_id"]])
    )
    plan["plan_id"] = contract.digest(plan)
    write(target, plan, exclusive=True)
    return plan


def validate_plan(plan: dict, *, check_payloads: bool = False) -> None:
    if plan["schema"] != SCHEMA or plan["plan_id"] != contract.digest(
        {k: v for k, v in plan.items() if k != "plan_id"}
    ):
        raise ValueError("Frozen plan changed")
    if plan["runtime"] != runtime():
        raise ValueError("Frozen Python/HTTP runtime changed")
    for name, expected in plan["inputs"].items():
        if contract.file_sha(name) != expected:
            raise ValueError("Pinned input changed: " + name)
    if check_payloads:
        for job in plan["jobs"]:
            body = contract.encoded(contract.payload(plan, job))
            if hashlib.sha256(body).hexdigest() != job["request_sha256"] or len(body) != job["request_bytes"]:
                raise ValueError("Frozen request changed")


def attempt_record(plan: dict, job: dict, folder: Path) -> dict:
    request = read(folder / "request.json")
    expected = {
        "plan_id": plan["plan_id"],
        "job_id": job["job_id"],
        "request_sha256": job["request_sha256"],
        "request_bytes": job["request_bytes"],
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Attempt request journal changed")
    result_path = folder / "result.json"
    if result_path.exists():
        result = read(result_path)
        row = contract.interpret(result, plan["route_config"])
    else:
        result = {}
        row = contract.interpret({"transport_error": "interrupted_request"}, plan["route_config"])
        row["reason"] = "interrupted_request"
    case = job["case"]
    row.update(
        case_id=case["case_id"],
        family_id=case["family_id"],
        benchmark_partition=case["benchmark_partition"],
        view_sha256=case["view_sha256"],
        model=plan["route"],
        arm=job["arm"],
        phase=plan["phase"],
        job_id=job["job_id"],
        stats_case_id=job["stats_case_id"],
        plan_id=plan["plan_id"],
        attempt_id=plan["phase"] + "/" + job["job_id"] + "/" + folder.name,
        sequence=request["sequence"],
        request_sha256=job["request_sha256"],
        request_started_at=result.get("request_started_at"),
        response_received_at=result.get("response_received_at"),
        latency_seconds=result.get("latency_seconds"),
        scheduler_wait_seconds=result.get("scheduler_wait_seconds"),
        preparation_seconds=result.get("preparation_seconds"),
        http_status=result.get("http_status"),
        requested_reasoning=plan["route_config"]["settings"].get(
            "reasoning", plan["route_config"]["settings"].get("reasoning_effort")
        ),
        returned_reasoning_effort=result.get("headers", {}).get("x-runinfra-reasoning-effort"),
        raw_response_path=str(result_path.resolve()) if result_path.exists() else None,
        raw_response_sha256=contract.file_sha(result_path) if result_path.exists() else None,
        request_journal_sha256=contract.file_sha(folder / "request.json"),
        training_ready=False,
    )
    return row


def reconstruct(plan: dict) -> list[dict]:
    root = directory(plan["route"], plan["phase"]) / "attempts"
    by_id = {j["job_id"]: j for j in plan["jobs"]}
    if not root.exists():
        return []
    if not {p.name for p in root.iterdir()} <= set(by_id):
        raise ValueError("Unexpected attempt job directory")
    records = []
    for job in plan["jobs"]:
        parent = root / job["job_id"]
        if not parent.exists():
            continue
        previous = None
        for sequence, folder in enumerate(sorted(parent.iterdir()), 1):
            if folder.name != f"{sequence:02d}" or previous and previous["status"] != "rejected_retryable":
                raise ValueError("Unexpected retry sequence or retry of a completed/unknown attempt")
            row = attempt_record(plan, job, folder)
            if row["sequence"] != sequence:
                raise ValueError("Attempt sequence changed")
            records.append(row)
            previous = row
    return records


def liability(records: list[dict], reserve: Decimal) -> Decimal:
    return sum(
        (Decimal(r["price_usd"]) if r.get("price_usd") is not None else reserve for r in records), Decimal(0)
    )


def fatal(row: dict, reserve: Decimal) -> bool:
    return (
        row["status"] in {"failed", "unknown"}
        or row.get("price_usd") is None
        and row["status"] != "rejected_retryable"
        or row.get("price_usd") is not None
        and Decimal(row["price_usd"]) > reserve
    )


def summarize(plan: dict, records: list[dict]) -> dict:
    selected = {r["job_id"]: r for r in records if r["status"] in {"valid", "invalid"}}
    reserve = Decimal(plan["reserve_per_attempt_usd"])
    return {
        "schema": SCHEMA,
        "route": plan["route"],
        "phase": plan["phase"],
        "plan_id": plan["plan_id"],
        "planned_jobs": len(plan["jobs"]),
        "http_attempts": len(records),
        "selected_responses": len(selected),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "numeric_scores": {
            field: sum(r.get(field) is not None for r in selected.values())
            for field in ("technical_score", "aesthetic_score")
        },
        "known_price_usd": str(
            sum((Decimal(r["price_usd"]) for r in records if r.get("price_usd") is not None), Decimal(0))
        ),
        "unpriced_attempts": sum(r.get("price_usd") is None for r in records),
        "admission_liability_usd": str(liability(records, reserve)),
        "status": "blocked"
        if any(fatal(r, reserve) for r in records)
        else "complete"
        if len(selected) == len(plan["jobs"])
        else "partial",
        "training_ready": False,
    }


def save(plan: dict, records: list[dict]) -> dict:
    root = directory(plan["route"], plan["phase"])
    selected = {r["job_id"]: r for r in records if r["status"] in {"valid", "invalid"}}
    jsonl(root / "attempts.jsonl", records)
    jsonl(
        root / "observations.jsonl", [selected[j["job_id"]] for j in plan["jobs"] if j["job_id"] in selected]
    )
    summary = summarize(plan, records)
    write(root / "summary.json", summary)
    return summary


def send(plan: dict, job: dict, sequence: int, secret: str, admitted_at: float) -> dict:
    worker_started = time.perf_counter()
    body = contract.encoded(contract.payload(plan, job))
    if hashlib.sha256(body).hexdigest() != job["request_sha256"]:
        raise ValueError("Request changed after freeze")
    folder = directory(plan["route"], plan["phase"]) / "attempts" / job["job_id"] / f"{sequence:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(
        folder / "request.json",
        {
            "plan_id": plan["plan_id"],
            "job_id": job["job_id"],
            "sequence": sequence,
            "request_sha256": job["request_sha256"],
            "request_bytes": len(body),
            "created_at": now(),
        },
        exclusive=True,
    )
    route = plan["route_config"]
    result = {
        "http_status": None,
        "body": None,
        "headers": {},
        "transport_error": None,
        "request_started_at": now(),
        "scheduler_wait_seconds": worker_started - admitted_at,
    }
    started = time.perf_counter()
    result["preparation_seconds"] = started - worker_started
    try:
        response = requests.post(
            route["endpoint"],
            data=body,
            headers={
                **route["headers"],
                "Authorization": "Bearer " + secret,
                "Content-Type": "application/json",
            },
            timeout=(30, route["read_timeout_seconds"]),
            allow_redirects=False,
        )
        # requests.post returns only after reading the complete non-streaming response body.
        result["latency_seconds"] = time.perf_counter() - started
        result["response_received_at"] = now()
        result.update(
            http_status=response.status_code,
            body=response.text.replace(secret, "[REDACTED]"),
            headers={
                k.lower(): v.replace(secret, "[REDACTED]")
                for k, v in response.headers.items()
                if k.lower().startswith(("x-runinfra-", "x-request-id", "retry-after"))
            },
        )
    except requests.RequestException as error:
        result.update(
            latency_seconds=time.perf_counter() - started,
            response_received_at=now(),
            transport_error=type(error).__name__,
        )
    write(folder / "result.json", result, exclusive=True)
    return attempt_record(plan, job, folder)


def execute(plan: dict, *, limit: int | None = None, case_ids: set[str] | None = None) -> dict:
    validate_plan(plan)
    root = directory(plan["route"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        other = []
        for phase in PHASES:
            path = directory(plan["route"], phase) / "plan.json"
            if path.exists() and phase != plan["phase"]:
                previous = read(path)
                validate_plan(previous)
                other.extend(reconstruct(previous))
        records = reconstruct(plan)
        reserve = Decimal(plan["reserve_per_attempt_usd"])
        if any(fatal(r, reserve) for r in other + records):
            raise ValueError("Unknown/nonretryable attempts require investigation before more route calls")
        if (
            sum(row["status"] == "rejected_retryable" for row in other + records)
            >= plan["route_config"]["maximum_capacity_rejections"]
        ):
            raise ValueError("Route pre-serve rejection limit reached")
        known = {j["case"]["case_id"] for j in plan["jobs"]}
        if case_ids is not None and not case_ids <= known:
            raise ValueError("Unknown case IDs")
        last = {r["job_id"]: r for r in records}
        jobs = [
            j
            for j in plan["jobs"]
            if (case_ids is None or j["case"]["case_id"] in case_ids)
            and (j["job_id"] not in last or last[j["job_id"]]["status"] == "rejected_retryable")
        ]
        if limit is not None:
            jobs = jobs[:limit]
        secret = os.environ.get(plan["route_config"]["key_env"])
        if jobs and not secret:
            raise ValueError("Required API key is unavailable")
        try:
            _stream(plan, deque(jobs), secret, records, other, last)
        except BaseException:
            save(plan, reconstruct(plan))
            raise
        return save(plan, records)


def _stream(plan: dict, pending: deque, secret: str, records: list, other: list, last: dict) -> None:
    route = plan["route_config"]
    reserve, cap = Decimal(plan["reserve_per_attempt_usd"]), Decimal(route["budget_usd"])
    active, stopped, cooldown_until = {}, False, 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=route["concurrency"]) as pool:
        while pending or active:
            available = int(max(cap - liability(records + other, reserve), Decimal(0)) / reserve) - len(
                active
            )
            if not stopped and time.monotonic() >= cooldown_until:
                for _ in range(max(0, min(route["concurrency"] - len(active), available, len(pending)))):
                    job = pending.popleft()
                    sequence = last.get(job["job_id"], {}).get("sequence", 0) + 1
                    if sequence > route["maximum_rejections_per_job"]:
                        stopped = True
                        break
                    active[pool.submit(send, plan, job, sequence, secret, time.perf_counter())] = job
            if not active:
                if pending and not stopped and time.monotonic() < cooldown_until:
                    time.sleep(min(1, cooldown_until - time.monotonic()))
                    continue
                if pending and not stopped:
                    raise ValueError("Budget admission limit reached")
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    row = future.result()
                except Exception:
                    stopped = True
                    # Any incomplete request is retained and blocks replay. Reconstruct at exit.
                    raise
                records.append(row)
                last[job["job_id"]] = row
                stopped |= fatal(row, reserve)
                if row["status"] == "rejected_retryable":
                    rejected = sum(r["status"] == "rejected_retryable" for r in records + other)
                    stopped |= rejected >= route["maximum_capacity_rejections"]
                    pending.append(job)
                    delay = max(row["retry_after_seconds"], min(60, 2 ** row["sequence"]))
                    cooldown_until = max(cooldown_until, time.monotonic() + delay)
                if len(records) % 50 == 0 or stopped:
                    summary = save(plan, records)
                    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "verify"))
    parser.add_argument("--route", required=True, choices=ROUTES)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--references", type=Path, default=HERE / "references-freeze.json")
    parser.add_argument("--stats", type=Path, default=HERE / "stats-prompts.jsonl")
    parser.add_argument("--donors", type=Path, default=HERE / "stats-donors.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze(args.route, args.phase, args.manifest, args.references, args.stats, args.donors)
        print(
            json.dumps(
                {
                    "plan_id": plan["plan_id"],
                    "jobs": len(plan["jobs"]),
                    "route": args.route,
                    "phase": args.phase,
                }
            )
        )
        return
    plan = read(directory(args.route, args.phase) / "plan.json")
    if args.command == "verify":
        validate_plan(plan, check_payloads=True)
        print(json.dumps(save(plan, reconstruct(plan))))
    else:
        print(
            json.dumps(execute(plan, limit=args.limit, case_ids=set(args.case_id) if args.case_id else None))
        )


if __name__ == "__main__":
    main()
