"""Resume unattempted jobs after independently confirmed client-process interruption.

Interrupted journals remain immutable, unpriced and unmeasured. Their original
conservative reservations remain charged against admission; they are never replayed.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner


def approved_interruption(row: dict, audited: dict) -> bool:
    item = audited.get(row["attempt_id"])
    if not item:
        return False
    path = Path(item["request_journal_path"])
    return (
        row["status"] == "unknown" and row.get("reason") == "interrupted_request"
        and row.get("price_usd") is None and row.get("latency_seconds") is None
        and row.get("raw_response_path") is None and not (path.parent / "result.json").exists()
        and row["request_journal_sha256"] == item["request_journal_sha256"] == contract.file_sha(path)
        and row["request_sha256"] == item["request_sha256"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("luna", "liquid", "jev"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--known-policy", type=Path)
    args = parser.parse_args()
    audit = runner.read(args.audit)
    if audit["driver_sha256"] != contract.file_sha(__file__) or audit["replay_interrupted_requests"] is not False:
        raise ValueError("Changed interruption audit or driver")
    audited = {row["attempt_id"]: row for row in audit["interrupted_attempts"] if row["route"] == args.route}
    if not audited:
        raise ValueError("Route has no explicitly audited interruptions")
    plans = {phase: runner.read(runner.directory(args.route, phase) / "plan.json") for phase in runner.PHASES}
    history = []
    for plan in plans.values():
        runner.validate_plan(plan)
        history.extend(runner.reconstruct(plan))
    by_id = {row["attempt_id"]: row for row in history}
    if any(identity not in by_id or not approved_interruption(by_id[identity], audited) for identity in audited):
        raise ValueError("Audited interrupted request changed or obtained a response")
    original_fatal = runner.fatal

    def admitted_fatal(row, reserve):
        return False if approved_interruption(row, audited) else original_fatal(row, reserve)

    started = runner.now()
    name = started.replace(":", "-") + ".json"
    directory = "interruption-admissions" if args.route == "jev" else "recovery-segments"
    destination = runner.directory(args.route) / directory / name
    segment = {"route": args.route, "started_at": started, "status": "running",
               "effective_concurrency": 1 if args.route == "jev" else plans["rehearsal"]["route_config"]["concurrency"],
               "approval_path": str(args.audit.resolve()), "approval_sha256": contract.file_sha(args.audit),
               "driver_sha256": contract.file_sha(__file__), "attempt_ids": [], "quarantined_attempt_ids": [],
               "interrupted_attempt_ids": sorted(audited), "phases": {}, "no_interrupted_attempt_replayed": True}
    runner.write(destination, segment, exclusive=True)
    try:
        runner.fatal = admitted_fatal
        if args.route == "jev":
            if not args.known_policy:
                raise ValueError("Jev needs its separately frozen bounded503 policy")
            previous_argv = sys.argv
            try:
                sys.argv = ["api_recovery.py", "--approval", str(args.known_policy), "--concurrency", "1"]
                api_recovery.main()
            finally:
                sys.argv = previous_argv
        else:
            for phase, plan in plans.items():
                runner.execute(plan)
                records = runner.reconstruct(plan)
                latest = {row["job_id"]: row for row in records}
                if any(admitted_fatal(row, Decimal(plan["reserve_per_attempt_usd"])) for row in records) or any(
                    job["job_id"] not in latest or latest[job["job_id"]]["status"] == "rejected_retryable" for job in plan["jobs"]
                ):
                    raise RuntimeError("New failure or incomplete dispatch; stopped for audit")
                runner.validate_plan(plan, check_payloads=True)
        segment["status"] = "all_jobs_attempted_with_reviewed_missing_results"
    except BaseException as error:
        segment.update(status="stopped_requires_audit", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        runner.fatal = original_fatal
        for phase, plan in plans.items():
            records = runner.reconstruct(plan)
            segment["phases"][phase] = runner.save(plan, records)
            segment["attempt_ids"] = sorted(set(segment["attempt_ids"]) | {
                row["attempt_id"] for row in records if row.get("request_started_at") and row["request_started_at"] >= started
            })
        segment["finished_at"] = runner.now()
        runner.write(destination, segment)
        print(json.dumps({"route": args.route, "status": segment["status"], "segment": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
