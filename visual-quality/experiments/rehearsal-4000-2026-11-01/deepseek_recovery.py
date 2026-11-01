"""Audit and skip billed no-answer token exhaustion without changing frozen requests.

The operational adapter changes only admission of an exact completed failure class.
It uses the original scheduler, capacity429 handling, concurrency and budget. Every
attempt remains unchanged; no completed response is retried or scored from reasoning.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_runner as runner


def no_answer_exhaustion(row: dict) -> bool:
    if (row["model"] != "deepseek" or row["status"] != "failed" or row["http_status"] != 200
            or row.get("reason") != "Incomplete completion" or row.get("price_usd") is None):
        return False
    path = Path(row["raw_response_path"])
    raw = runner.read(path)
    body = json.loads(raw["body"])
    choices, usage = body.get("choices", []), body.get("usage", {})
    if len(choices) != 1:
        return False
    choice = choices[0]
    return (
        raw.get("transport_error") is None
        and contract.file_sha(path) == row["raw_response_sha256"]
        and body.get("model") in {"deepseek-v4-1-flash-1017", "deepseek-ai/DeepSeek-V4.1-Flash-1017"}
        and choice.get("finish_reason") == "length"
        and choice.get("message", {}).get("content") in (None, "")
        and choice.get("runinfra", {}).get("output_status", {}).get("code") == "generation_limit_reached_before_answer"
        and usage.get("completion_tokens") == 16384
        and usage.get("completion_tokens_details", {}).get("reasoning_tokens") == 16384
        and usage.get("runinfra", {}).get("output_token_accounting", {}).get("visible_answer_tokens") == 0
        and Decimal(str(usage.get("cost"))) == Decimal(row["price_usd"])
        and Decimal(str(usage.get("runinfra", {}).get("cost_microcents"))) / 100_000_000 == Decimal(row["price_usd"])
    )


class ExhaustionAdmission:
    def __init__(self, original, approval):
        self.original = original
        self.approval = approval
        self.reviewed = {row["attempt_id"]: row for row in approval["reviewed_attempts"]}
        self.seen = set()

    def __call__(self, row, reserve):
        if not no_answer_exhaustion(row):
            return self.original(row, reserve)
        previous = self.reviewed.get(row["attempt_id"])
        if previous:
            if any(row.get(key) != previous[key] for key in ("raw_response_sha256", "request_journal_sha256", "request_sha256")):
                return True
        elif not row.get("request_started_at") or row["request_started_at"] < self.approval["created_at"]:
            return True
        if Decimal(row["price_usd"]) > reserve:
            return True
        self.seen.add(row["attempt_id"])
        return len(self.seen) >= self.approval["maximum_total_exhaustions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    approval = runner.read(args.approval)
    if (approval["route"] != "deepseek" or approval["replay_completed_attempts"] is not False
            or approval["maximum_total_exhaustions"] != 10 or approval["driver_sha256"] != contract.file_sha(__file__)):
        raise ValueError("Invalid or changed bounded exhaustion admission")
    plans = {phase: runner.read(runner.directory("deepseek", phase) / "plan.json") for phase in runner.PHASES}
    history = []
    for plan in plans.values():
        runner.validate_plan(plan)
        history.extend(runner.reconstruct(plan))
    by_id = {row["attempt_id"]: row for row in history}
    for row in approval["reviewed_attempts"]:
        actual = by_id[row["attempt_id"]]
        if not no_answer_exhaustion(actual) or any(actual[k] != row[k] for k in ("raw_response_sha256", "request_journal_sha256", "request_sha256")):
            raise ValueError("Audited historical exhaustion receipt changed")
    started = runner.now()
    destination = runner.directory("deepseek") / "recovery-segments" / (started.replace(":", "-") + ".json")
    segment = {"route": "deepseek", "started_at": started, "status": "running", "effective_concurrency": 8,
               "original_plan_concurrency": 8, "approval_path": str(args.approval.resolve()),
               "approval_sha256": contract.file_sha(args.approval), "driver_sha256": contract.file_sha(__file__),
               "attempt_ids": [], "quarantined_attempt_ids": [], "exhausted_attempt_ids": [], "phases": {},
               "maximum_total_exhaustions": 10, "no_completed_attempt_replayed": True,
               "runtime_adaptation": "Only exact bounded exhaustion admission; scheduler/capacity retries/config unchanged"}
    runner.write(destination, segment, exclusive=True)
    original_fatal = runner.fatal
    admission = ExhaustionAdmission(original_fatal, approval)
    try:
        for phase, plan in plans.items():
            runner.fatal = admission
            try:
                runner.execute(plan)
            finally:
                runner.fatal = original_fatal
            current = runner.reconstruct(plan)
            segment["phases"][phase] = runner.save(plan, current)
            segment["attempt_ids"] = sorted(set(segment["attempt_ids"]) | {
                row["attempt_id"] for row in current if row.get("request_started_at") and row["request_started_at"] >= started
            })
            segment["exhausted_attempt_ids"] = sorted(admission.seen)
            runner.write(destination, segment)
            reserve = Decimal(plan["reserve_per_attempt_usd"])
            latest = {row["job_id"]: row for row in current}
            if any(admission(row, reserve) for row in current) or any(
                job["job_id"] not in latest or latest[job["job_id"]]["status"] == "rejected_retryable" for job in plan["jobs"]
            ):
                raise RuntimeError("Unreviewed failure, exhaustion bound, or incomplete dispatch; stopped for audit")
            runner.validate_plan(plan, check_payloads=True)
        segment["status"] = "all_jobs_attempted_with_reviewed_missing_results"
    except BaseException as error:
        segment.update(status="stopped_requires_audit", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        runner.fatal = original_fatal
        for phase, plan in plans.items():
            current = runner.reconstruct(plan)
            segment["phases"][phase] = runner.save(plan, current)
            segment["attempt_ids"] = sorted(set(segment["attempt_ids"]) | {
                row["attempt_id"] for row in current if row.get("request_started_at") and row["request_started_at"] >= started
            })
        segment.update(finished_at=runner.now(), exhausted_attempt_ids=sorted(admission.seen))
        runner.write(destination, segment)
        print(json.dumps({"segment": str(destination), "status": segment["status"], "exhaustions": len(admission.seen)}), flush=True)


if __name__ == "__main__":
    main()
