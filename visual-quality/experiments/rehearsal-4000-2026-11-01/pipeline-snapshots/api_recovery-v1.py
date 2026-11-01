"""Continue only unattempted Jev jobs after explicit receipt-bound failure review.

Frozen plans, prompts and original journals remain unchanged. Unknown bills retain
their full reservation. A newly observed error stops dispatch; nothing is retried.
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


def approved_failure(row: dict, approvals: dict) -> bool:
    approval = approvals.get(row["attempt_id"])
    if not approval:
        return False
    result_path = Path(row["raw_response_path"])
    result = runner.read(result_path)
    return (
        row["model"] == "jev"
        and row["status"] == "failed"
        and row["http_status"] == 503
        and row["price_usd"] is None
        and not result.get("transport_error")
        and json.loads(result["body"]) == ERROR_BODY
        and row["raw_response_sha256"] == approval["raw_response_sha256"] == contract.file_sha(result_path)
        and row["request_journal_sha256"] == approval["request_journal_sha256"]
        and row["request_sha256"] == approval["request_sha256"]
    )


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
            if set(approvals) != {row["attempt_id"] for row in fatal} or not all(
                approved_failure(row, approvals) for row in fatal
            ):
                raise ValueError("Unreviewed fatal attempt or changed approved receipt")
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
                            stopped |= runner.fatal(row, reserve) or row["status"] not in {"valid", "invalid"}
                        if stopped or len(records[phase]) % 25 == 0:
                            segment["phases"][phase] = runner.save(plan, records[phase])
                            runner.write(segment_path, segment)
                            print(json.dumps(segment["phases"][phase]), flush=True)
                segment["phases"][phase] = runner.save(plan, records[phase])
                runner.write(segment_path, segment)
                if stopped:
                    raise RuntimeError("New provider failure; no further dispatch or automatic retry")
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
