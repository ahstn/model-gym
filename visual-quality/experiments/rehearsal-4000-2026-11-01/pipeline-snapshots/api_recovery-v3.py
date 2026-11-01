"""Continue only unattempted Jev jobs after explicit receipt-bound failure review.

Frozen plans, prompts and original journals remain unchanged. Unknown bills retain
their full reservation. A prospective bounded policy can quarantine one exact
reviewed provider503 class. Other errors stop dispatch; nothing is retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import time
from collections import deque
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_runner as runner

ERROR_BODY = {"error": {"message": 'HTTP 503: {"detail":{"error_type":"api_usage_error","message":"Image processing is unavailable."}}', "code": 503}}


def known_image_failure(row: dict) -> bool:
    if (
        row["model"] != "jev" or row["status"] != "failed"
        or row["http_status"] != 503 or row["price_usd"] is not None
        or not row.get("raw_response_path")
    ):
        return False
    result_path = Path(row["raw_response_path"])
    result = runner.read(result_path)
    return (
        not result.get("transport_error")
        and json.loads(result["body"]) == ERROR_BODY
        and row["raw_response_sha256"] == contract.file_sha(result_path)
    )


def approved_failure(row: dict, approvals: dict) -> bool:
    approval = approvals.get(row["attempt_id"])
    if not approval:
        return False
    return (
        known_image_failure(row)
        and row["raw_response_sha256"] == approval["raw_response_sha256"]
        and row["request_journal_sha256"] == approval["request_journal_sha256"]
        and row["request_sha256"] == approval["request_sha256"]
    )


def circuit_breaker(failures: list[bool], policy: dict) -> str | None:
    consecutive = policy["maximum_consecutive_failures"]
    window = policy["rolling_window_requests"]
    if len(failures) >= consecutive and all(failures[-consecutive:]):
        return "consecutive_known_provider_failures"
    if len(failures) >= window and sum(failures[-window:]) >= policy["maximum_failures_in_window"]:
        return "known_provider_failure_rate"
    return None


def unattempted(plan: dict, records: list[dict]) -> list[dict]:
    attempted = {row["job_id"] for row in records}
    return [job for job in plan["jobs"] if job["job_id"] not in attempted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    approval_doc = runner.read(args.approval)
    if approval_doc["route"] != "jev" or approval_doc["replay_completed_attempts"] is not False:
        raise ValueError("Invalid scoped recovery declaration")
    approvals = {row["attempt_id"]: row for row in approval_doc["reviewed_attempts"]}
    if len(approvals) != len(approval_doc["reviewed_attempts"]) or not approvals:
        raise ValueError("Recovery needs unique explicitly reviewed failures")
    policy = approval_doc.get("prospective_known_failure_policy")
    if policy and (
        policy != {"maximum_consecutive_failures": 5, "rolling_window_requests": 100,
                   "maximum_failures_in_window": 20, "error_class": "exact_jev_image_processing_503"}
        or approval_doc.get("driver_sha256") != contract.file_sha(__file__)
        or args.concurrency != 1
    ):
        raise ValueError("Unsupported or changed prospective quarantine policy")
    root = runner.directory("jev")
    started_at = runner.now()
    segment_path = root / "recovery-segments" / (started_at.replace(":", "-") + ".json")
    segment = {
        "started_at": started_at, "status": "running", "route": "jev",
        "approval_path": str(args.approval.resolve()), "approval_sha256": contract.file_sha(args.approval),
        "driver_sha256": contract.file_sha(__file__), "original_plan_concurrency": 4,
        "effective_concurrency": args.concurrency, "attempt_ids": [], "phases": {},
        "latency_segment": "reviewed_provider_failure_recovery",
        "no_attempt_replayed": True,
        "prospective_known_failure_policy": policy,
        "quarantined_attempt_ids": [],
    }
    runner.write(segment_path, segment, exclusive=True)
    try:
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plans = {phase: runner.read(runner.directory("jev", phase) / "plan.json") for phase in runner.PHASES}
            records = {}
            for phase, plan in plans.items():
                runner.validate_plan(plan)
                records[phase] = runner.reconstruct(plan)
            reserve = Decimal(plans["rehearsal"]["reserve_per_attempt_usd"])
            history = [row for group in records.values() for row in group]
            fatal = [row for row in history if runner.fatal(row, reserve)]
            if not set(approvals) <= {row["attempt_id"] for row in fatal} or not all(
                approved_failure(row, approvals)
                or policy and row["request_started_at"] >= approval_doc["created_at"] and known_image_failure(row)
                for row in fatal
            ):
                raise ValueError("Unreviewed fatal attempt or changed approved receipt")
            # Interrupted journals have no observed HTTP outcome and must not be
            # counted as successful responses or assigned an invented timestamp.
            observed = [row for row in history if row.get("response_received_at")]
            failure_events = [known_image_failure(row) for row in sorted(observed, key=lambda row: row["response_received_at"])]
            if policy and circuit_breaker(failure_events, policy):
                raise ValueError("Prospective quarantine circuit breaker already open")
            secret = os.environ.get("OPENROUTER_API_KEY")
            if not secret:
                raise ValueError("Required API key unavailable")
            total_sent = 0
            for phase, plan in plans.items():
                pending = deque(unattempted(plan, records[phase]))
                if args.limit is not None:
                    pending = deque(list(pending)[:max(0, args.limit - total_sent)])
                active, stopped = {}, False
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    while pending or active:
                        history = [row for group in records.values() for row in group]
                        available = int(max(Decimal(plan["route_config"]["budget_usd"]) - runner.liability(history, reserve), Decimal(0)) / reserve) - len(active)
                        if not stopped:
                            for _ in range(max(0, min(args.concurrency - len(active), available, len(pending)))):
                                job = pending.popleft()
                                active[pool.submit(runner.send, plan, job, 1, secret, time.perf_counter())] = job
                                total_sent += 1
                        if not active:
                            if pending and not stopped:
                                raise ValueError("Conservative budget admission exhausted")
                            break
                        done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            active.pop(future)
                            row = future.result()
                            records[phase].append(row)
                            segment["attempt_ids"].append(row["attempt_id"])
                            known_failure = known_image_failure(row)
                            failure_events.append(known_failure)
                            if policy and known_failure:
                                segment["quarantined_attempt_ids"].append(row["attempt_id"])
                            else:
                                stopped |= runner.fatal(row, reserve) or row["status"] not in {"valid", "invalid"}
                            if policy:
                                segment["circuit_breaker_reason"] = circuit_breaker(failure_events, policy)
                                stopped |= segment["circuit_breaker_reason"] is not None
                        if stopped or len(records[phase]) % 25 == 0:
                            segment["phases"][phase] = runner.save(plan, records[phase])
                            runner.write(segment_path, segment)
                            print(json.dumps(segment["phases"][phase]), flush=True)
                segment["phases"][phase] = runner.save(plan, records[phase])
                runner.write(segment_path, segment)
                if stopped:
                    raise RuntimeError("Unreviewed provider failure or quarantine circuit breaker; no further dispatch or retry")
                if args.limit is not None and total_sent >= args.limit:
                    segment["status"] = "limited_dispatch_complete"
                    break
                runner.validate_plan(plan, check_payloads=True)
                if unattempted(plan, records[phase]):
                    raise ValueError("Phase still contains unattempted jobs")
            else:
                segment["status"] = "all_jobs_attempted_with_reviewed_missing_results"
        segment["finished_at"] = runner.now()
        runner.write(segment_path, segment)
        print(json.dumps({"segment_path": str(segment_path), "status": segment["status"], "new_attempts": total_sent}), flush=True)
    except BaseException as error:
        if "plans" in locals():
            for phase, plan in plans.items():
                current = runner.reconstruct(plan)
                segment["phases"][phase] = runner.save(plan, current)
                segment["attempt_ids"] = sorted(set(segment["attempt_ids"]) | {
                    row["attempt_id"] for row in current
                    if row.get("request_started_at") and row["request_started_at"] >= started_at
                })
        segment.update(status="stopped_requires_audit", finished_at=runner.now(), error_type=type(error).__name__, error=str(error))
        runner.write(segment_path, segment)
        raise


if __name__ == "__main__":
    main()
