"""Later first-pass-only availability amendment; never replay an attempted job."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner
import interruption_resume
import jev_circuit_reset as prior_reset

HERE = Path(__file__).resolve().parent
COMPATIBILITY_POLICY = prior_reset.POLICY
POLICY = {"maximum_consecutive_failures": 5, "rolling_window_requests": 100,
          "maximum_failures_in_window": 30, "cumulative_minimum_responses": 100,
          "maximum_cumulative_failure_fraction": "0.20", "error_class": "exact_jev_image_processing_503"}
HELPERS = ("jev_availability.py", "api_recovery.py", "api_runner.py", "api_contract.py",
           "interruption_resume.py", "jev_circuit_reset.py")


def helper_pins():
    return {str(HERE / name): contract.file_sha(HERE / name) for name in HELPERS}


def snapshot():
    plans, rows = prior_reset.load_history()
    return {"plans": {phase: {"path": str(runner.directory("jev", phase) / "plan.json"),
                             "sha256": contract.file_sha(runner.directory("jev", phase) / "plan.json"),
                             "plan_id": plan["plan_id"]} for phase, plan in plans.items()},
            "receipts": sorted((prior_reset.receipt(row) for row in rows), key=lambda row: row["attempt_id"])}


def breaker(failures, compatibility_policy):
    if compatibility_policy != COMPATIBILITY_POLICY:
        raise ValueError("Underlying exact-error compatibility policy changed")
    if len(failures) >= 5 and all(failures[-5:]):
        return "consecutive_known_provider_failures"
    if len(failures) >= 100 and sum(failures[-100:]) >= 30:
        return "rolling_known_provider_failures_30_of_100"
    # Integer arithmetic implements >=20%, including every observed first-pass response.
    if len(failures) >= 100 and sum(failures) * 5 >= len(failures):
        return "cumulative_known_provider_failures_at_least_20_percent"
    return None


def read_approval(path, expected_sha):
    if contract.file_sha(path) != expected_sha:
        raise ValueError("Availability approval hash changed")
    doc = runner.read(path)
    if (doc["availability_schema"] != "jev-first-pass-availability-v1"
            or doc["scope"] != "remaining_initial_jobs_only" or doc["effective_availability_policy"] != POLICY
            or doc["helper_sha256"] != helper_pins() or doc["effective_concurrency"] != 4
            or doc["replay_completed_attempts"] is not False or doc["postpass_policy_changed"] is not False
            or doc["approval_id"] != contract.digest({k: v for k, v in doc.items() if k != "approval_id"})):
        raise ValueError("Availability approval scope, policy or source changed")
    for key in ("previous_stopped_segment", "interruption_audit", "evidence"):
        item = doc[key]
        if contract.file_sha(item["path"]) != item["sha256"]:
            raise ValueError("Availability evidence changed")
    return doc


def validate_start(doc):
    previous = runner.read(Path(doc["previous_stopped_segment"]["path"]))
    elapsed = (datetime.now(UTC) - prior_reset.timestamp(previous["finished_at"])).total_seconds()
    if previous["status"] != "stopped_requires_audit" or previous["route"] != "jev" or elapsed < 120:
        raise ValueError("Finished stopped segment and120-second cooldown required")
    if snapshot() != doc["initial_prefix"]:
        raise ValueError("Initial receipt prefix changed or new unowned requests appeared")


def execute(path, expected_sha):
    doc = read_approval(path, expected_sha)
    root = runner.directory("jev")
    with (root / ".availability-amendment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_start(doc)
        destination = root / "availability-amendments" / (doc["approval_id"] + ".json")
        state = {"route": "jev", "status": "running", "started_at": runner.now(), "approval_path": str(path.resolve()),
                 "approval_sha256": expected_sha, "approval_id": doc["approval_id"], "driver_sha256": contract.file_sha(__file__),
                 "effective_concurrency": 4, "effective_availability_policy": POLICY, "history_scope": "all_observed_initial_responses",
                 "attempt_ids": [], "new_receipts": []}
        runner.write(destination, state, exclusive=True)
        original_breaker, original_phases, original_argv = api_recovery.circuit_breaker, runner.PHASES, sys.argv
        initial_ids = {row["attempt_id"] for row in doc["initial_prefix"]["receipts"]}
        checked = False
        def guarded_breaker(failures, policy):
            nonlocal checked
            if not checked:
                # Called by the original driver while holding its route/budget lock.
                validate_start(doc)
                checked = True
            return breaker(failures, policy)
        try:
            api_recovery.circuit_breaker = guarded_breaker
            runner.PHASES = ("benchmark", "rehearsal")
            sys.argv = ["interruption_resume.py", "--route", "jev", "--audit", doc["interruption_audit"]["path"],
                        "--known-policy", str(path)]
            interruption_resume.main()
            state["status"] = "all_initial_jobs_attempted"
        except BaseException as error:
            state.update(status="stopped_requires_audit", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            api_recovery.circuit_breaker, runner.PHASES, sys.argv = original_breaker, original_phases, original_argv
            current = snapshot()
            by_id = {row["attempt_id"]: row for row in current["receipts"]}
            if any(by_id.get(row["attempt_id"]) != row for row in doc["initial_prefix"]["receipts"]):
                state["status"] = "historical_receipt_changed"
            state["new_receipts"] = [row for row in current["receipts"] if row["attempt_id"] not in initial_ids]
            state["attempt_ids"] = [row["attempt_id"] for row in state["new_receipts"]]
            state["finished_at"] = runner.now()
            runner.write(destination, state)
    return {"state_path": str(destination), "status": state["status"], "new_attempts": len(state["attempt_ids"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.approval, args.approval_sha256)), flush=True)


if __name__ == "__main__":
    main()
