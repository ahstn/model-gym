"""Continue unattempted frozen views; retain strict rating failures without retries."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("runinfra_frozen_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
DATA = runner.DATA / "continuation-01"
STRICT_FAILURE = "invalid_rating: LabelValidationError"


def may_advance(attempt: dict) -> bool:
    """Only a priced valid response or strict rating failure permits another view."""
    return attempt["cost_usd"] is not None and (
        attempt["status"] == "valid"
        or (attempt["status"] == "failed" and attempt["reason"] == STRICT_FAILURE)
    )


def original_evidence() -> dict:
    paths = [runner.DATA / name for name in ("run.json", "report.json", "observations.jsonl")]
    paths += sorted((runner.DATA / "attempts").rglob("*"))
    return {str(path.relative_to(runner.DATA)): runner.file_sha256(path) for path in paths if path.is_file()}


def freeze() -> dict:
    if DATA.exists():
        raise ValueError("Continuation directory already exists; inspect or verify it")
    run = runner.read_run()
    original = runner.verify(run)
    if not all(may_advance(attempt) for attempt in original["attempts"]):
        raise ValueError("Original run has an uncertain or non-rating failure")
    attempted = {attempt["case_id"] for attempt in original["attempts"]}
    remaining = [
        case for case in run["cases"] if not case["skip_reason"] and case["case_id"] not in attempted
    ]
    if not remaining or len(remaining) > 40:
        raise ValueError("Expected at most 40 remaining original eligible views")
    plan = {
        "schema_version": "runinfra-continuation-plan-v1",
        "created_at": runner.now(),
        "original_run_id": run["run_id"],
        "original_run_sha256": runner.file_sha256(runner.DATA / "run.json"),
        "original_evidence": original_evidence(),
        "source_pins": {
            str(path): runner.file_sha256(path)
            for path in (HERE / "runner.py", Path(__file__).resolve(), HERE / "continuation-protocol.md")
        },
        "excluded_attempted_case_ids": sorted(attempted),
        "excluded_size_case_ids": [case["case_id"] for case in run["cases"] if case["skip_reason"]],
        "cases": remaining,
        "settings": run["settings"],
        "budget": {**run["budget"], "prior_cost_usd": original["known_cost_usd"]},
        "policy": {
            "automatic_retries": False,
            "fallback": False,
            "rating_repair": False,
            "continue_only_after_priced_failure": STRICT_FAILURE,
            "view_policy": "original PNG bytes and request hashes unchanged",
        },
        "training_ready": False,
    }
    plan["continuation_id"] = runner.digest(plan)
    DATA.mkdir(parents=True)
    runner.atomic_json(DATA / "plan.json", plan)
    save_report(plan, run)
    return plan


def read_plan() -> tuple[dict, dict]:
    plan = json.loads((DATA / "plan.json").read_text())
    if runner.digest({k: v for k, v in plan.items() if k != "continuation_id"}) != plan["continuation_id"]:
        raise ValueError("Continuation plan changed")
    for name, expected in plan["source_pins"].items():
        if runner.file_sha256(Path(name)) != expected:
            raise ValueError("Continuation source changed")
    if original_evidence() != plan["original_evidence"]:
        raise ValueError("Original frozen evidence changed")
    run = runner.read_run()
    original = runner.verify(run)
    if run["run_id"] != plan["original_run_id"]:
        raise ValueError("Original run identity changed")
    if runner.file_sha256(runner.DATA / "run.json") != plan["original_run_sha256"]:
        raise ValueError("Original run bytes changed")
    attempted = {attempt["case_id"] for attempt in original["attempts"]}
    expected = [case for case in run["cases"] if not case["skip_reason"] and case["case_id"] not in attempted]
    if plan["cases"] != expected or plan["excluded_attempted_case_ids"] != sorted(attempted):
        raise ValueError("Continuation selection differs from unattempted views")
    if plan["settings"] != run["settings"] or plan["budget"] != {
        **run["budget"],
        "prior_cost_usd": original["known_cost_usd"],
    }:
        raise ValueError("Continuation settings or total budget changed")
    return plan, run


def reconstruct(plan: dict, run: dict) -> tuple[dict, list[dict]]:
    observations, attempts = [], []
    attempt_root = DATA / "attempts"
    allowed = {case["case_id"] for case in plan["cases"]}
    if attempt_root.exists() and not {path.name for path in attempt_root.iterdir()}.issubset(allowed):
        raise ValueError("Unexpected continuation attempt directory")
    for case in plan["cases"]:
        folder = attempt_root / case["case_id"]
        if not folder.exists():
            continue
        if not (folder / "request.json").is_file():
            raise ValueError("Incomplete continuation attempt directory; inspect manually")
        request = json.loads((folder / "request.json").read_text())
        expected = {key: case[key] for key in ("case_id", "request_sha256", "request_bytes")}
        expected.update(run_id=run["run_id"], continuation_id=plan["continuation_id"])
        if any(request.get(key) != value for key, value in expected.items()):
            raise ValueError("Continuation request differs from frozen inputs")
        if not (folder / "result.json").is_file():
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "status": "unknown",
                    "cost_usd": None,
                    "reason": "interrupted_request",
                    "score": None,
                }
            )
            continue
        result = json.loads((folder / "result.json").read_text())
        details, record = runner.interpret(run, case, request, result)
        details["score"] = record["technical_quality"]["score"] if record is not None else None
        attempts.append(details)
        if record is not None:
            observations.append(record)
    failed = sum(attempt["status"] != "valid" for attempt in attempts)
    stopped = any(not may_advance(attempt) for attempt in attempts)
    status = "blocked" if stopped else "partial"
    if not stopped and len(attempts) == len(plan["cases"]):
        status = "complete_with_invalid_ratings" if failed else "complete"
    known = sum(
        (Decimal(attempt["cost_usd"]) for attempt in attempts if attempt["cost_usd"] is not None), Decimal(0)
    )
    report = {
        "schema_version": "runinfra-continuation-report-v1",
        "continuation_id": plan["continuation_id"],
        "original_run_id": run["run_id"],
        "status": status,
        "planned": len(plan["cases"]),
        "attempted": len(attempts),
        "valid_observations": len(observations),
        "scored": sum(record["technical_quality"]["score"] is not None for record in observations),
        "failed_attempts": failed,
        "known_cost_usd": str(known),
        "prior_cost_usd": plan["budget"]["prior_cost_usd"],
        "total_known_cost_usd": str(known + Decimal(plan["budget"]["prior_cost_usd"])),
        "unpriced_attempts": sum(attempt["cost_usd"] is None for attempt in attempts),
        "excluded_attempted_case_ids": plan["excluded_attempted_case_ids"],
        "excluded_size_case_ids": plan["excluded_size_case_ids"],
        "attempts": attempts,
        "training_ready": False,
    }
    return report, observations


def save_report(plan: dict, run: dict) -> dict:
    report, observations = reconstruct(plan, run)
    runner.write_jsonl(DATA / "observations.jsonl", observations)
    runner.atomic_json(DATA / "report.json", report)
    return report


def verify(plan: dict, run: dict) -> dict:
    report, observations = reconstruct(plan, run)
    if (
        json.loads((DATA / "report.json").read_text()) != report
        or runner.read_jsonl(DATA / "observations.jsonl") != observations
    ):
        raise ValueError("Continuation exports differ from preserved replies")
    return report


def execute(plan: dict, run: dict) -> dict:
    key = os.environ.get("RUN_INFRA_API_KEY")
    if not key:
        raise ValueError("RUN_INFRA_API_KEY is unavailable")
    with (DATA / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = save_report(plan, run)
        if report["status"] == "blocked":
            raise ValueError("An uncertain or non-rating failure prevents further calls")
        for case in plan["cases"]:
            folder = DATA / "attempts" / case["case_id"]
            if folder.exists():
                continue
            reserve = Decimal(plan["budget"]["reserve_per_request_usd"])
            if Decimal(report["total_known_cost_usd"]) + reserve > Decimal(plan["budget"]["max_cost_usd"]):
                raise ValueError("Next request exceeds the combined admission budget")
            body = runner.payload(run, case)
            wire = runner.encoded(body)
            if runner.digest(body) != case["request_sha256"] or len(wire) != case["request_bytes"]:
                raise ValueError("Continuation request changed")
            request = {key: case[key] for key in ("case_id", "request_sha256", "request_bytes")}
            request.update(
                run_id=run["run_id"], continuation_id=plan["continuation_id"], created_at=runner.now()
            )
            folder.mkdir(parents=True)
            runner.atomic_json(folder / "request.json", request)
            try:
                response = runner.requests.post(
                    runner.ENDPOINT,
                    data=wire,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    timeout=(15, 180),
                    allow_redirects=False,
                )
                headers = {
                    name.lower(): value.replace(key, "[REDACTED]")
                    for name, value in response.headers.items()
                    if name.lower().startswith(("x-runinfra-", "x-request-id", "retry-after"))
                }
                result = {
                    "http_status": response.status_code,
                    "body": response.text.replace(key, "[REDACTED]"),
                    "headers": headers,
                    "transport_error": None,
                    "received_at": runner.now(),
                }
            except runner.requests.RequestException as error:
                result = {
                    "http_status": None,
                    "body": None,
                    "headers": {},
                    "transport_error": type(error).__name__,
                    "received_at": runner.now(),
                }
            runner.atomic_json(folder / "result.json", result)
            report = save_report(plan, run)
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "attempted",
                            "scored",
                            "failed_attempts",
                            "total_known_cost_usd",
                            "status",
                        )
                    }
                ),
                flush=True,
            )
            if report["status"] == "blocked":
                break
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
        plan, run = read_plan()
        report = verify(plan, run)
    else:
        plan, run = read_plan()
        report = execute(plan, run) if args.command == "run" else verify(plan, run)
    print(json.dumps({key: value for key, value in report.items() if key != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
