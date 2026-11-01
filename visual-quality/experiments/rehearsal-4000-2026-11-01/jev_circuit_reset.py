"""Explicit receipt-bound circuit reset, ten serial probes, then bounded continuation.

Only never-attempted jobs are dispatched. Existing execution helpers are adapted
in memory and restored on every exit; their source and frozen plans stay intact.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner
import interruption_resume

HERE = Path(__file__).resolve().parent
PHASE_ORDER = ("benchmark", "rehearsal")
POLICY = {"maximum_consecutive_failures": 5, "rolling_window_requests": 100,
          "maximum_failures_in_window": 20, "error_class": "exact_jev_image_processing_503"}
HELPERS = ("jev_circuit_reset.py", "api_recovery.py", "interruption_resume.py", "api_runner.py", "api_contract.py")


def helper_pins():
    return {str(HERE / name): contract.file_sha(HERE / name) for name in HELPERS}


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": contract.file_sha(path)}


def verify_pin(item):
    if contract.file_sha(item["path"]) != item["sha256"]:
        raise ValueError("Pinned circuit-control evidence changed: " + item["path"])
    return runner.read(Path(item["path"]))


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Timezone-aware circuit-control timestamp required")
    return parsed


def receipt(row):
    return {key: row.get(key) for key in (
        "attempt_id", "phase", "job_id", "sequence", "plan_id", "request_sha256", "request_journal_sha256",
        "raw_response_path", "raw_response_sha256", "request_started_at", "response_received_at", "status", "price_usd")}


def load_history():
    plans, history = {}, []
    for phase in PHASE_ORDER:
        plan = runner.read(runner.directory("jev", phase) / "plan.json")
        runner.validate_plan(plan)
        plans[phase] = plan
        history.extend(runner.reconstruct(plan))
    return plans, history


def capture_prefix():
    """Read operational receipts only; no numerical scores enter the artifact."""
    plans, rows = load_history()
    attempted = {(row["phase"], row["job_id"]) for row in rows}
    observed = sorted((row for row in rows if row.get("response_received_at")), key=lambda row: row["response_received_at"])
    return {
        "plans": {phase: {**pin(runner.directory("jev", phase) / "plan.json"), "plan_id": plan["plan_id"]}
                  for phase, plan in plans.items()},
        "attempt_receipts": sorted((receipt(row) for row in rows), key=lambda row: row["attempt_id"]),
        "observed_prefix": [{**receipt(row), "known_image_503": api_recovery.known_image_failure(row)} for row in observed],
        "probe_jobs": [{"phase": phase, "job_id": job["job_id"], "request_sha256": job["request_sha256"]}
                       for phase, plan in plans.items() for job in plan["jobs"]
                       if (phase, job["job_id"]) not in attempted][:10],
    }


def freeze_control(previous_segment, audit, probe_approval, continue_approval, destination, prior_failed_controls=()):
    """Create one immutable explicitly approved reset artifact; never dispatch."""
    prior = [pin(path) for path in prior_failed_controls]
    if len(prior) >= 3:
        raise ValueError("Three consecutive failed half-opens require a new external decision")
    for item in prior:
        if verify_pin(item)["status"] != "probe_failed":
            raise ValueError("Prior half-open evidence is not a failed probe")
    stopped = runner.read(Path(previous_segment))
    if stopped["status"] != "stopped_requires_audit" or stopped.get("route") != "jev" or not stopped.get("finished_at"):
        raise ValueError("Reset must bind a finished stopped Jev segment")
    snapshot = capture_prefix()
    if len(snapshot["probe_jobs"]) != 10:
        raise ValueError("Ten never-attempted jobs are required for a half-open probe")
    for path, concurrency in ((probe_approval, 1), (continue_approval, 4)):
        approval = runner.read(Path(path))
        if (approval["route"] != "jev" or approval["effective_concurrency"] != concurrency
                or approval["prospective_known_failure_policy"] != POLICY
                or approval["replay_completed_attempts"] is not False):
            raise ValueError("Unexpected probe/continuation approval policy")
    control = {"schema": "jev-circuit-reset-v1", "created_at": runner.now(),
        "previous_stopped_segment": pin(previous_segment), "stopped_at": stopped["finished_at"],
        "audit": pin(audit), "probe_approval": pin(probe_approval), "continue_approval": pin(continue_approval),
        "cooldown_seconds": 120, "probe_requests": 10, "maximum_probe_known_503": 1,
        "probe_concurrency": 1, "continue_concurrency": 4, "phase_order": list(PHASE_ORDER),
        "circuit_policy": POLICY, "helper_sha256": helper_pins(), "prefix": snapshot,
        "prior_failed_halfopens": prior, "maximum_consecutive_failed_halfopens": 3,
        "no_automatic_probe_loop": True, "no_attempt_replayed": True}
    control["control_id"] = contract.digest(control)
    destination = Path(destination)
    runner.write(destination, control, exclusive=True)
    destination.chmod(0o444)
    return {"control_path": str(destination.resolve()), "control_sha256": contract.file_sha(destination),
            "control_id": control["control_id"], "observed_prefix_count": len(snapshot["observed_prefix"])}


def read_control(path, expected_sha):
    if contract.file_sha(path) != expected_sha:
        raise ValueError("Circuit control hash changed")
    control = runner.read(Path(path))
    if (control["schema"] != "jev-circuit-reset-v1" or control["control_id"] != contract.digest(
            {key: value for key, value in control.items() if key != "control_id"})
            or control["helper_sha256"] != helper_pins()
            or control["circuit_policy"] != POLICY or control["phase_order"] != list(PHASE_ORDER)
            or [control[key] for key in ("cooldown_seconds", "probe_requests", "maximum_probe_known_503",
                                        "probe_concurrency", "continue_concurrency", "maximum_consecutive_failed_halfopens")]
            != [120, 10, 1, 1, 4, 3] or len(control["prior_failed_halfopens"]) >= 3):
        raise ValueError("Circuit control policy or source changed")
    for key in ("previous_stopped_segment", "audit", "probe_approval", "continue_approval"):
        verify_pin(control[key])
    for item in control["prior_failed_halfopens"]:
        if verify_pin(item)["status"] != "probe_failed":
            raise ValueError("Prior failed-half-open evidence changed")
    return control


def cooldown(control, now=None):
    elapsed = ((now or datetime.now(UTC)) - timestamp(control["stopped_at"])).total_seconds()
    if elapsed < 120:
        raise ValueError("At least 120 seconds of cooldown are required before dispatch")


def validate_history(control, current, extra_receipts=()):
    frozen = control["prefix"]
    expected = sorted([*frozen["attempt_receipts"], *extra_receipts], key=lambda row: row["attempt_id"])
    if current["plans"] != frozen["plans"] or current["attempt_receipts"] != expected:
        raise ValueError("Frozen receipt prefix changed or new unowned calls appeared")
    count = len(frozen["observed_prefix"])
    if current["observed_prefix"][:count] != frozen["observed_prefix"]:
        raise ValueError("Observed receipt order or exact503 prefix changed")


class ResetBreaker:
    def __init__(self, control, stage, original, validate_first):
        self.prefix = [row["known_image_503"] for row in control["prefix"]["observed_prefix"]]
        self.stage, self.original, self.validate_first = stage, original, validate_first
        self.initialized = False

    def __call__(self, failures, policy):
        if policy != POLICY or failures[:len(self.prefix)] != self.prefix or len(failures) < len(self.prefix):
            raise ValueError("Circuit policy or observed Boolean prefix changed")
        if not self.initialized:
            # api_recovery invokes this after acquiring the original route lock.
            self.validate_first()
            self.initialized = True
        suffix = failures[len(self.prefix):]
        if self.stage == "probe":
            if len(suffix) > 10:
                raise ValueError("Half-open probe exceeded its ten-request bound")
            return "half_open_second_known_503" if sum(suffix) >= 2 else None
        return self.original(suffix, policy)


def probe_success(rows):
    if len(rows) != 10 or len({row["attempt_id"] for row in rows}) != 10:
        return False
    reserve = Decimal("0.011010048")  # Original Jev full-context reservation: 262144 * .042 / 1e6.
    failures = 0
    for row in rows:
        if not row.get("response_received_at") or row.get("http_status") is None:
            return False
        if api_recovery.known_image_failure(row):
            failures += 1
        elif row["status"] not in {"valid", "invalid"} or runner.fatal(row, reserve):
            return False
    return failures <= 1


def control_directory(control):
    return runner.directory("jev") / "circuit-resets" / control["control_id"]


def continuation_receipts(control):
    probe = runner.read(control_directory(control) / "probe.json")
    if (probe["status"] != "probe_succeeded" or probe["control_id"] != control["control_id"]
            or len(probe["attempt_ids"]) != 10 or probe["effective_concurrency"] != 1
            or len(probe["new_receipts"]) != 10 or probe["known_image_503_count"] > 1):
        raise ValueError("Continuation requires a successful complete ten-request probe")
    plans, rows = load_history()
    selected = [row for row in rows if row["attempt_id"] in set(probe["attempt_ids"])]
    if not probe_success(selected) or sorted((receipt(row) for row in selected), key=lambda row: row["attempt_id"]) != probe["new_receipts"]:
        raise ValueError("Successful probe receipts changed or are incomplete")
    return probe["new_receipts"]


@contextmanager
def adapted_execution(control, stage, state, state_path, initial_extra):
    original_breaker, original_main = api_recovery.circuit_breaker, api_recovery.main
    original_phases, original_send, original_fatal = runner.PHASES, runner.send, runner.fatal
    mutex = threading.Lock()
    attempted = {row["attempt_id"] for row in [*control["prefix"]["attempt_receipts"], *initial_extra]}
    probe_jobs = control["prefix"]["probe_jobs"]
    def validate_first():
        cooldown(control)
        validate_history(control, capture_prefix(), initial_extra)
    def bounded_main():
        previous = sys.argv
        try:
            sys.argv = [*previous, "--limit", "10"] if stage == "probe" else previous
            return original_main()
        finally:
            sys.argv = previous
    def guarded_send(plan, job, sequence, secret, admitted_at):
        identity = plan["phase"] + "/" + job["job_id"] + f"/{sequence:02d}"
        with mutex:
            if sequence != 1 or identity in attempted or identity in state["attempt_ids"]:
                raise ValueError("Circuit reset cannot replay an attempted job")
            if stage == "probe":
                index = len(state["attempt_ids"])
                if index >= 10 or probe_jobs[index] != {"phase": plan["phase"], "job_id": job["job_id"], "request_sha256": job["request_sha256"]}:
                    raise ValueError("Probe departed from frozen unattempted job order")
            state["attempt_ids"].append(identity)
            runner.write(state_path, state)
        return original_send(plan, job, sequence, secret, admitted_at)
    try:
        runner.PHASES = PHASE_ORDER
        runner.send = guarded_send
        api_recovery.main = bounded_main
        api_recovery.circuit_breaker = ResetBreaker(control, stage, original_breaker, validate_first)
        yield
    finally:
        api_recovery.circuit_breaker, api_recovery.main = original_breaker, original_main
        runner.PHASES, runner.send, runner.fatal = original_phases, original_send, original_fatal


def execute(stage, path, expected_sha):
    control = read_control(path, expected_sha)
    folder = control_directory(control)
    folder.mkdir(parents=True, exist_ok=True)
    with (runner.directory("jev") / ".circuit-reset.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cooldown(control)
        extra = continuation_receipts(control) if stage == "continue" else []
        validate_history(control, capture_prefix(), extra)
        state_path = folder / (stage + ".json")
        state = {"stage": stage, "route": "jev", "status": "running", "started_at": runner.now(),
                 "control_id": control["control_id"], "control_path": str(Path(path).resolve()), "control_sha256": expected_sha,
                 "driver_sha256": contract.file_sha(__file__), "helper_sha256": control["helper_sha256"],
                 "approval_path": control[stage + "_approval"]["path"], "approval_sha256": control[stage + "_approval"]["sha256"],
                 "effective_concurrency": 1 if stage == "probe" else 4, "phase_order": list(PHASE_ORDER),
                 "attempt_ids": [], "new_receipts": [], "reset_observed_prefix_count": len(control["prefix"]["observed_prefix"])}
        runner.write(state_path, state, exclusive=True)
        previous_argv = sys.argv
        try:
            with adapted_execution(control, stage, state, state_path, extra):
                sys.argv = ["interruption_resume.py", "--route", "jev", "--audit", control["audit"]["path"],
                            "--known-policy", control[stage + "_approval"]["path"]]
                interruption_resume.main()
            state["status"] = "continuation_complete" if stage == "continue" else "probe_pending_validation"
        except BaseException as error:
            state.update(status="probe_failed" if stage == "probe" else "stopped_requires_audit",
                         error_type=type(error).__name__, error=str(error))
            raise
        finally:
            sys.argv = previous_argv
            plans, rows = load_history()
            selected = [row for row in rows if row["attempt_id"] in set(state["attempt_ids"])]
            state["new_receipts"] = sorted((receipt(row) for row in selected), key=lambda row: row["attempt_id"])
            state["known_image_503_count"] = sum(api_recovery.known_image_failure(row) for row in selected)
            if stage == "probe" and state["status"] == "probe_pending_validation":
                state["status"] = "probe_succeeded" if probe_success(selected) else "probe_failed"
            state["finished_at"] = runner.now()
            state["consecutive_failed_halfopens"] = len(control["prior_failed_halfopens"]) + (state["status"] == "probe_failed")
            runner.write(state_path, state)
        if stage == "probe" and state["status"] != "probe_succeeded":
            raise RuntimeError("Half-open probe incomplete or unsuccessful; continuation forbidden")
    return {"state_path": str(state_path), "status": state["status"], "attempts": len(state["new_receipts"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "continue"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.stage, args.control, args.control_sha256)), flush=True)


if __name__ == "__main__":
    main()
