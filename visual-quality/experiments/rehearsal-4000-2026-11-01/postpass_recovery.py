"""Receipt-bound, serial recovery only after the complete 18,000-job initial pass.

This adapter owns new postpass metadata. Frozen runners, plans, prior journals and
initial summaries are never rewritten. Importing this module cannot dispatch work.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner
import deepseek_recovery
import interruption_resume

HERE = Path(__file__).resolve().parent
POLICY = HERE / "postpass-recovery-policy.json"
COMPLETE = {"valid", "partial", "invalid"}
CAPS = {"deepseek": "30.00", "luna": "30.00", "jev": "15.00", "liquid": "15.00"}
COUNTS = {route: {"rehearsal": 3500, "benchmark": 1500 if route in {"luna", "deepseek"} else 500}
          for route in runner.ROUTES}
LIMITS = {"exact_jev_image_processing_503": 2, "audited_client_interruption": 1}
BREAKER = {"maximum_consecutive_failures": 5, "rolling_window_requests": 100,
           "maximum_failures_in_window": 20, "error_class": "exact_jev_image_processing_503"}
HELPERS = ("postpass_recovery.py", "api_runner.py", "api_contract.py", "api_recovery.py",
           "interruption_resume.py", "deepseek_recovery.py")
RECEIPT_FIELDS = ("attempt_id", "job_id", "plan_id", "sequence", "request_sha256",
                  "request_journal_sha256", "raw_response_path", "raw_response_sha256",
                  "status", "reason", "price_usd", "request_started_at", "response_received_at")


def root() -> Path:
    return runner.DATA / "results/api/postpass"


def policy_document() -> dict:
    policy = runner.read(POLICY)
    if (policy["schema"] != "rehearsal-postpass-policy-v1"
            or policy["maximum_additional_attempts"] != LIMITS
            or policy["route_budgets_usd"] != CAPS
            or policy["expected_initial_jobs"] != COUNTS
            or policy["effective_concurrency"] != 1 or policy["circuit_breaker"] != BREAKER
            or policy["interruption_counts"] != {"luna": 16, "liquid": 4, "jev": 1}
            or policy["interruption_audit_sha256"] != contract.file_sha(HERE / "api-client-interruption-audit.json")):
        raise ValueError("Postpass policy or pinned interruption audit changed")
    return policy


def helper_pins() -> dict:
    return {str(HERE / name): contract.file_sha(HERE / name) for name in HELPERS}


def service_snapshot() -> dict:
    """Read only process state, never the unit environment or credentials."""
    snapshot = {}
    for route in sorted(runner.ROUTES):
        unit = f"modelgym-rehearsal-{route}-20261101.service"
        output = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState,MainPID,ControlGroup"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout
        state = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if set(state) != {"ActiveState", "MainPID", "ControlGroup"}:
            raise ValueError("Cannot prove initial service state: " + unit)
        if state["ActiveState"] not in {"inactive", "failed"} or state["MainPID"] != "0":
            raise ValueError("Initial service still live: " + unit)
        cgroup = Path("/sys/fs/cgroup") / state["ControlGroup"].lstrip("/")
        if (state["ControlGroup"] and cgroup.exists()
                and any(path.read_text().strip() for path in cgroup.rglob("cgroup.procs"))):
            raise ValueError("Initial service cgroup still has processes: " + unit)
        snapshot[route] = {"unit": unit, **state}
    return {"observed_at": runner.now(), "units": snapshot}


@contextmanager
def exclusive_dispatch():
    """One recovery globally; also exclude every original route driver."""
    root().mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        paths = [root() / ".lock", *(runner.directory(route) / ".lock" for route in sorted(runner.ROUTES))]
        for path in paths:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_plans(*, check_payloads: bool = True) -> dict:
    plans = {}
    for route in runner.ROUTES:
        plans[route] = {}
        for phase in runner.PHASES:
            plan = runner.read(runner.directory(route, phase) / "plan.json")
            runner.validate_plan(plan, check_payloads=check_payloads)
            config = plan["route_config"]
            reserve = (Decimal(config["input_price_per_million"]) * config["context_length"]
                       + Decimal(config["output_price_per_million"]) * config["settings"].get("max_tokens", 0)) / 1_000_000
            if (plan["route"] != route or plan["phase"] != phase
                    or len(plan["jobs"]) != COUNTS[route][phase]
                    or len({job["job_id"] for job in plan["jobs"]}) != len(plan["jobs"])
                    or Decimal(config["budget_usd"]) != Decimal(CAPS[route])
                    or Decimal(plan["reserve_per_attempt_usd"]) != reserve):
                raise ValueError("Original plan identity, coverage, cap or full-context reserve changed")
            plans[route][phase] = plan
        if plans[route]["rehearsal"]["route_config"] != plans[route]["benchmark"]["route_config"]:
            raise ValueError("Route configuration differs between phases")
    return plans


def protected(row: dict) -> bool:
    return row["status"] in COMPLETE or any(row.get(key) is not None for key in (
        "technical_score", "aesthetic_score", "technical_score_native", "aesthetic_score_native"))


def known_failure(row: dict) -> bool:
    try:
        return not protected(row) and api_recovery.known_image_failure(row)
    except (ValueError, TypeError, KeyError, OSError):
        return False


def receipt(row: dict) -> dict:
    request = runner.directory(row["model"]) / row["phase"] / "attempts" / row["job_id"] / f"{row['sequence']:02d}" / "request.json"
    return {**{key: row.get(key) for key in RECEIPT_FIELDS},
            "request_journal_path": str(request.resolve()),
            "request_journal_sha256": contract.file_sha(request)}


def group_jobs(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[(row["phase"], row["job_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["sequence"])
        if sum(protected(row) for row in rows) > 1:
            raise ValueError("Multiple completed/scored responses for one job")
    return groups


def classify_initial(rows: list[dict], audited: dict) -> str | None:
    """Choose by error receipts only, never by score, label, source or category."""
    if any(protected(row) for row in rows):
        return None
    latest = rows[-1]
    if known_failure(latest):
        return "exact_jev_image_processing_503"
    if interruption_resume.approved_interruption(latest, audited):
        return "audited_client_interruption"
    return None


def freeze_document(policy: dict, plans: dict, records: dict, services: dict) -> dict:
    """Build the complete immutable allowlist after all initial gates pass."""
    audit = runner.read(HERE / "api-client-interruption-audit.json")
    audited = {(row["route"], row["attempt_id"]): row for row in audit["interrupted_attempts"]}
    if len(audited) != 21 or Counter(route for route, _ in audited) != policy["interruption_counts"]:
        raise ValueError("Expected exactly the original 21 audited interruptions")
    document = {"schema": "rehearsal-postpass-allowlist-v1", "created_at": runner.now(),
                "policy_path": str(POLICY), "policy_sha256": contract.file_sha(POLICY),
                "helper_sha256": helper_pins(), "initial_service_snapshot": services,
                "initial_status_snapshot": {}, "plans": {}, "initial_receipts": {}, "allowlist": []}
    seen_audited = set()
    for route in runner.ROUTES:
        route_audit = {identity: item for (name, identity), item in audited.items() if name == route}
        route_rows = [row for phase in runner.PHASES for row in records[route][phase]]
        groups = group_jobs(route_rows)
        document["plans"][route], document["initial_receipts"][route] = {}, {}
        for phase, plan in plans[route].items():
            expected = {job["job_id"] for job in plan["jobs"]}
            if {row["job_id"] for row in records[route][phase]} != expected:
                raise ValueError("Initial pass incomplete: " + route + "/" + phase)
            path = runner.directory(route, phase) / "plan.json"
            document["plans"][route][phase] = {"path": str(path), "sha256": contract.file_sha(path),
                "plan_id": plan["plan_id"], "job_ids": sorted(expected)}
            document["initial_receipts"][route][phase] = [receipt(row) for row in records[route][phase]]
        for (phase, job_id), rows in groups.items():
            if rows[-1]["status"] == "rejected_retryable":
                raise ValueError("Initial capacity retry is still pending")
            for row in rows:
                if interruption_resume.approved_interruption(row, route_audit):
                    seen_audited.add((route, row["attempt_id"]))
                elif row["status"] in {"failed", "unknown"} and not (
                    known_failure(row) or deepseek_recovery.no_answer_exhaustion(row)
                ):
                    raise ValueError("Unreviewed initial failure or request-only journal: " + row["attempt_id"])
            kind = classify_initial(rows, route_audit)
            if kind:
                document["allowlist"].append({"route": route, "phase": phase, "job_id": job_id,
                    "case_id": rows[-1]["case_id"], "source_class": kind,
                    "source_attempt_id": rows[-1]["attempt_id"], "source_receipt": receipt(rows[-1]),
                    "original_history": [receipt(row) for row in rows],
                    "initial_sequence": rows[-1]["sequence"],
                    "maximum_additional_attempts": LIMITS[kind],
                    "maximum_sequence": rows[-1]["sequence"] + LIMITS[kind]})
        reserve = Decimal(plans[route]["rehearsal"]["reserve_per_attempt_usd"])
        document["initial_status_snapshot"][route] = {
            "planned_jobs": sum(len(plan["jobs"]) for plan in plans[route].values()),
            "attempted_jobs": len(groups), "attempts": len(route_rows),
            "status_counts": dict(Counter(row["status"] for row in route_rows)),
            "unpriced_attempts": sum(row.get("price_usd") is None for row in route_rows),
            "admission_liability_usd": str(runner.liability(route_rows, reserve))}
    if seen_audited != set(audited):
        raise ValueError("Audited interruption missing or changed")
    document["allowlist"].sort(key=lambda item: (item["route"], item["case_id"], item["phase"], item["job_id"]))
    document["allowlist_id"] = contract.digest(document)
    return document


def prepare(destination: Path) -> dict:
    policy = policy_document()
    with exclusive_dispatch():
        services = service_snapshot()  # Fail before reading responses while any initial route is live.
        plans = load_plans()
        records = {route: {phase: runner.reconstruct(plan) for phase, plan in phases.items()}
                   for route, phases in plans.items()}
        document = freeze_document(policy, plans, records, services)
        service_snapshot()
        runner.write(destination, document, exclusive=True)
        destination.chmod(0o444)
    return {"allowlist_path": str(destination.resolve()), "allowlist_sha256": contract.file_sha(destination),
            "eligible_jobs": len(document["allowlist"]), "initial_jobs": 18000}


def read_allowlist(path: Path, expected_sha256: str) -> dict:
    policy_document()
    if contract.file_sha(path) != expected_sha256:
        raise ValueError("Allowlist file hash changed")
    approval = runner.read(path)
    if (approval["schema"] != "rehearsal-postpass-allowlist-v1"
            or approval["allowlist_id"] != contract.digest({k: v for k, v in approval.items() if k != "allowlist_id"})
            or approval["policy_sha256"] != contract.file_sha(POLICY)
            or approval["helper_sha256"] != helper_pins()):
        raise ValueError("Allowlist, policy or helper source pins changed")
    keys = [(item["route"], item["phase"], item["job_id"]) for item in approval["allowlist"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate allowlist entry")
    for item in approval["allowlist"]:
        limit = LIMITS[item["source_class"]]
        if (item["route"] == "deepseek" or item["maximum_additional_attempts"] != limit
                or item["maximum_sequence"] != item["initial_sequence"] + limit
                or item["source_class"] == "exact_jev_image_processing_503" and item["route"] != "jev"):
            raise ValueError("Invalid allowlist attempt bound or route")
    # Runtime-only proof of the externally pinned file; never serialize this key
    # back into the immutable document or include it in its internal digest.
    return {**approval, "_verified_file_sha256": expected_sha256}


def metadata_path(route: str, attempt_id: str) -> Path:
    return root() / route / "attempt-metadata" / attempt_id / "admission.json"


def validate_additional(row: dict, previous: list[dict], entry: dict | None, approval: dict) -> None:
    if (not entry or protected(previous[-1]) or any(protected(old) for old in previous)
            or row["sequence"] > entry["maximum_sequence"]
            or row["sequence"] > entry["initial_sequence"] + 1 and (
                entry["source_class"] != "exact_jev_image_processing_503" or not known_failure(previous[-1]))):
        raise ValueError("Unauthorized retry, exceeded cap or replay of completed/scored response")
    meta = runner.read(metadata_path(row["model"], row["attempt_id"]))
    if (meta["allowlist_id"] != approval["allowlist_id"]
            or meta["approval_sha256"] != approval["_verified_file_sha256"]
            or meta["policy_sha256"] != approval["policy_sha256"]
            or meta["driver_sha256"] != approval["helper_sha256"][str(Path(__file__).resolve())]
            or meta["effective_concurrency"] != 1 or meta["sequence"] != row["sequence"]
            or meta["request_sha256"] != row["request_sha256"]
            or meta["source_attempt_id"] != entry["source_attempt_id"]
            or meta["source_class"] != entry["source_class"]
            or meta["source_receipt"] != entry["source_receipt"]):
        raise ValueError("Postpass admission metadata changed")
    request = runner.read(Path(receipt(row)["request_journal_path"]))
    timestamps = [datetime.fromisoformat(value) for value in (
        approval["created_at"], meta["admitted_at"], request["created_at"],
        *([row["request_started_at"]] if row.get("request_started_at") else []))]
    if any(value.tzinfo is None for value in timestamps) or timestamps != sorted(timestamps):
        raise ValueError("Postpass admission must precede journal/HTTP start and follow the allowlist freeze")
    outcome_path = metadata_path(row["model"], row["attempt_id"]).with_name("outcome.json")
    if outcome_path.exists() and runner.read(outcome_path)["receipt"] != receipt(row):
        raise ValueError("Postpass outcome receipt changed")


def reconstruct(plan: dict, approval: dict) -> list[dict]:
    """Validate original receipts and only the explicitly authorized extra sequences.

    Caller must first use read_allowlist to validate its externally pinned file hash.
    This loader intentionally does not call the frozen runner.reconstruct, whose
    original no-replay invariant rejects the newly authorized failed->retry path.
    """
    route, phase = plan["route"], plan["phase"]
    pinned = approval["plans"][route][phase]
    if (plan["plan_id"] != pinned["plan_id"] or contract.file_sha(pinned["path"]) != pinned["sha256"]
            or sorted(job["job_id"] for job in plan["jobs"]) != pinned["job_ids"]):
        raise ValueError("Original phase plan changed")
    initial = {item["attempt_id"]: item for item in approval["initial_receipts"][route][phase]}
    entries = {item["job_id"]: item for item in approval["allowlist"] if item["route"] == route and item["phase"] == phase}
    folder = runner.directory(route, phase) / "attempts"
    if {path.name for path in folder.iterdir()} != set(pinned["job_ids"]):
        raise ValueError("Initial job coverage changed or unexpected job directory")
    records, seen = [], set()
    for job in plan["jobs"]:
        previous = []
        for sequence, attempt in enumerate(sorted((folder / job["job_id"]).iterdir()), 1):
            if attempt.name != f"{sequence:02d}":
                raise ValueError("Attempt sequence gap or unexpected directory")
            row = runner.attempt_record(plan, job, attempt)
            if row["sequence"] != sequence:
                raise ValueError("Attempt sequence changed")
            identity = row["attempt_id"]
            if identity in initial:
                if receipt(row) != initial[identity]:
                    raise ValueError("Original request/response receipt changed")
                seen.add(identity)
            else:
                validate_additional(row, previous, entries.get(job["job_id"]), approval)
            previous.append(row)
            records.append(row)
    if seen != set(initial):
        raise ValueError("Original attempt was removed")
    group_jobs(records)
    return records


def all_records(plans: dict, approval: dict) -> dict:
    return {route: [row for plan in phases.values() for row in reconstruct(plan, approval)]
            for route, phases in plans.items()}


def recovery_rows(records: list[dict], approval: dict, route: str) -> list[dict]:
    original = {row["attempt_id"] for rows in approval["initial_receipts"][route].values() for row in rows}
    return [row for row in records if row["attempt_id"] not in original]


def stop_reason(records: list[dict], approval: dict, route: str, reserve: Decimal) -> str | None:
    for row in recovery_rows(records, approval, route):
        if known_failure(row):
            continue
        if row["status"] not in COMPLETE or runner.fatal(row, reserve):
            return "new_unreviewed_error_or_unknown_bill"
    observed = sorted((row for row in records if row.get("response_received_at") and row.get("http_status") is not None),
                      key=lambda row: (row["response_received_at"], row["attempt_id"]))
    return api_recovery.circuit_breaker([known_failure(row) for row in observed], BREAKER)


def next_job(entries: list[dict], records: list[dict]) -> tuple[dict, int] | None:
    groups = group_jobs(records)
    for entry in sorted(entries, key=lambda item: (item["case_id"], item["phase"], item["job_id"])):
        rows = groups[(entry["phase"], entry["job_id"])]
        if any(protected(row) for row in rows) or rows[-1]["sequence"] >= entry["maximum_sequence"]:
            continue
        if rows[-1]["sequence"] > entry["initial_sequence"] and not known_failure(rows[-1]):
            raise ValueError("Unexpected recovery outcome cannot acquire another attempt")
        return entry, rows[-1]["sequence"] + 1
    return None


def accounting(rows: list[dict], reserve: Decimal) -> dict:
    known = sum((Decimal(row["price_usd"]) for row in rows if row.get("price_usd") is not None), Decimal(0))
    unknown = sum(row.get("price_usd") is None for row in rows)
    return {"attempts": len(rows), "known_price_usd": str(known), "unpriced_attempts": unknown,
            "total_price_usd": None if unknown else str(known),
            "full_reserve_per_unknown_attempt_usd": str(reserve),
            "admission_liability_usd": str(runner.liability(rows, reserve))}


def route_report(route: str, plans: dict, records: list[dict], approval: dict) -> dict:
    groups = group_jobs(records)
    eligible = {(item["phase"], item["job_id"]): item for item in approval["allowlist"] if item["route"] == route}
    dimensions = defaultdict(Counter)
    totals = Counter(dict.fromkeys(("recovered", "fully_scored_recovered", "partial_or_invalid_recovered",
                                    "persistent", "ineligible", "missing_jobs"), 0))
    for phase, plan in plans.items():
        for job in plan["jobs"]:
            key, case = (phase, job["job_id"]), job["case"]
            rows = groups[key]
            completed = next((row for row in rows if row["status"] in COMPLETE), None)
            # Presence only: report missingness without ranking or comparing score values.
            missing = completed is None or any(completed.get(field) is None for field in ("technical_score", "aesthetic_score"))
            state = "recovered" if key in eligible and completed is not None else "persistent" if key in eligible else "ineligible"
            totals[state] += 1
            totals["missing_jobs"] += missing
            if state == "recovered":
                fully_scored = all(
                    completed.get("task_valid", {}).get(task) is True
                    and type(completed.get(field)) in {int, float} and math.isfinite(completed[field])
                    for task, field in (("technical_quality", "technical_score"), ("aesthetics", "aesthetic_score")))
                totals["fully_scored_recovered" if fully_scored else "partial_or_invalid_recovered"] += 1
            for dimension, value in (("source", case.get("source", "unknown")), ("partition", case["benchmark_partition"]),
                                     ("category", case.get("stratum", "unknown")), ("operator", case.get("operator", "unknown")),
                                     ("content_cluster", case.get("rehearsal_content_cluster", "unknown"))):
                bucket = dimensions[(dimension, str(value))]
                bucket["jobs"] += 1
                bucket["missing_jobs"] += missing
                bucket[state] += 1
    reserve = Decimal(plans["rehearsal"]["reserve_per_attempt_usd"])
    extra = recovery_rows(records, approval, route)
    return {"route": route, "updated_at": runner.now(), "allowlist_id": approval["allowlist_id"],
            "recovered_definition": "Eligible jobs with one completed response, including partial or invalid; fully scored counts are separate",
            "eligible_jobs": len(eligible), **dict(totals), "all_attempt_accounting": accounting(records, reserve),
            "recovery_attempt_accounting": accounting(extra, reserve),
            "recovery_status_counts": dict(Counter(row["status"] for row in extra)),
            "stop_reason": stop_reason(records, approval, route, reserve),
            "missingness": [{"dimension": key[0], "value": key[1], **dict(value)} for key, value in sorted(dimensions.items())],
            "training_ready": False}


def write_reports(plans: dict, records: dict, approval: dict) -> None:
    reports = {}
    for route in runner.ROUTES:
        reports[route] = route_report(route, plans[route], records[route], approval)
        runner.write(root() / route / "report.json", reports[route])
        runner.jsonl(root() / route / "attempts.jsonl", records[route])
    runner.write(root() / "report.json", {"allowlist_id": approval["allowlist_id"], "updated_at": runner.now(), "routes": reports})


def dispatch(route: str, plans: dict, records: dict, approval: dict, approval_path: Path,
             approval_sha256: str, limit: int | None) -> dict:
    reserve = Decimal(plans[route]["rehearsal"]["reserve_per_attempt_usd"])
    entries = [item for item in approval["allowlist"] if item["route"] == route]
    started = runner.now()
    segment_path = root() / route / "segments" / (started.replace(":", "-") + ".json")
    segment = {"route": route, "started_at": started, "status": "running", "effective_concurrency": 1,
               "original_plan_concurrency": plans[route]["rehearsal"]["route_config"]["concurrency"],
               "approval_path": str(approval_path.resolve()), "approval_sha256": approval_sha256,
               "allowlist_id": approval["allowlist_id"], "policy_sha256": approval["policy_sha256"],
               "driver_sha256": contract.file_sha(__file__), "helper_sha256": approval["helper_sha256"],
               "attempt_ids": [], "quarantined_attempt_ids": [], "latency_segment": "post_initial_pass_serial_recovery"}
    runner.write(segment_path, segment, exclusive=True)
    try:
        while True:
            for name in runner.ROUTES:
                reason = stop_reason(records[name], approval, name, Decimal(plans[name]["rehearsal"]["reserve_per_attempt_usd"]))
                if reason:
                    raise RuntimeError(name + ": " + reason)
            pending = next_job(entries, records[route])
            if not pending:
                segment["status"] = "allowed_attempts_complete"
                break
            if limit is not None and len(segment["attempt_ids"]) >= limit:
                segment["status"] = "limited_dispatch_complete"
                break
            entry, sequence = pending
            plan = plans[route][entry["phase"]]
            if runner.liability(records[route], reserve) + reserve > Decimal(CAPS[route]):
                raise RuntimeError("Conservative shared route budget admission exhausted")
            secret = os.environ.get(plan["route_config"]["key_env"])
            if not secret:
                raise ValueError("Required API key unavailable")
            # Repeat source/input checks immediately before each paid invocation.
            read_allowlist(approval_path, approval_sha256)
            runner.validate_plan(plan)
            job = next(job for job in plan["jobs"] if job["job_id"] == entry["job_id"])
            identity = entry["phase"] + "/" + job["job_id"] + f"/{sequence:02d}"
            admission_path = metadata_path(route, identity)
            runner.write(admission_path, {
                **{key: segment[key] for key in ("allowlist_id", "approval_sha256", "driver_sha256", "policy_sha256", "effective_concurrency")},
                "segment_path": str(segment_path), "admitted_at": runner.now(), "sequence": sequence,
                "request_sha256": job["request_sha256"], "source_attempt_id": entry["source_attempt_id"],
                "source_class": entry["source_class"], "source_receipt": entry["source_receipt"],
                "liability_before_usd": str(runner.liability(records[route], reserve)),
                "admitted_reserve_usd": str(reserve)}, exclusive=True)
            segment["attempt_ids"].append(identity)
            runner.write(segment_path, segment)
            row = runner.send(plan, job, sequence, secret, time.perf_counter())
            records[route].append(row)
            runner.write(admission_path.with_name("outcome.json"), {"recorded_at": runner.now(), "receipt": receipt(row),
                "request_started_at": row.get("request_started_at"), "response_received_at": row.get("response_received_at"),
                "latency_seconds": row.get("latency_seconds")}, exclusive=True)
            if known_failure(row):
                segment["quarantined_attempt_ids"].append(identity)
            runner.write(segment_path, segment)
            if len(segment["attempt_ids"]) % 25 == 0:
                write_reports(plans, records, approval)
    except BaseException as error:
        segment.update(status="stopped_requires_audit", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        segment["finished_at"] = runner.now()
        runner.write(segment_path, segment)
        # A killed send may have created a request-only journal. Preserve it and
        # its full liability, and make subsequent runs reject that new unknown.
        records = all_records(plans, approval)
        write_reports(plans, records, approval)
    return {"segment_path": str(segment_path), "status": segment["status"], "new_attempts": len(segment["attempt_ids"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare", help="Freeze only after every initial route has finished")
    prepare_parser.add_argument("--output", type=Path, default=root() / "allowlist.json")
    for command in ("run", "report"):
        command_parser = subcommands.add_parser(command)
        command_parser.add_argument("--allowlist", type=Path, required=True)
        command_parser.add_argument("--allowlist-sha256", required=True)
        if command == "run":
            command_parser.add_argument("--route", choices=("jev", "luna", "liquid"), required=True)
            command_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        if getattr(args, "limit", None) is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        with exclusive_dispatch():
            service_snapshot()
            approval = read_allowlist(args.allowlist, args.allowlist_sha256)
            plans = load_plans()
            records = all_records(plans, approval)
            if args.command == "run":
                result = dispatch(args.route, plans, records, approval, args.allowlist, args.allowlist_sha256, args.limit)
            else:
                write_reports(plans, records, approval)
                result = {"report_path": str(root() / "report.json")}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
