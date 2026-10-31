"""Continue the unchanged DeepSeek cohort after explicit shared-capacity rejections."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import importlib.util
import json
import os
import re
import time
from collections import Counter, deque
from datetime import datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dual2000_continuation_original", HERE / "runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
ORIGINAL = runner.DATA / "deepseek/images"
DATA = runner.DATA / "deepseek/continuation-01"
RETRY_CODE = "hosted_shared_model_concurrency_limit"


def rejection(result: dict) -> dict | None:
    """Allow only the explicit pre-serve capacity rejection, never unknown outcomes."""
    if result.get("transport_error") is not None or result.get("http_status") != 429:
        return None
    try:
        body = json.loads(result["body"])
        error = body["error"]
        identifier = body["request_id"]
        delay = Decimal(result["headers"]["retry-after"])
        if (
            error.get("type") != "rate_limit_error"
            or error.get("code") != RETRY_CODE
            or "choices" in body
            or body.get("usage")
            or not isinstance(identifier, str)
            or not identifier.startswith("req_")
            or result["headers"].get("x-request-id") != identifier
            or not delay.is_finite()
            or delay < 0
        ):
            return None
        return {"provider_request_id": identifier, "retry_after_seconds": float(delay)}
    except (KeyError, ValueError, TypeError, ArithmeticError, AttributeError):
        return None


def artifact_pins(directory: Path) -> dict:
    return {
        str(path.resolve()): runner.file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != ".lock"
    }


def decorate(detail: dict, result: dict | None, folder: Path, stage: str, sequence: int) -> dict:
    detail = copy.deepcopy(detail)
    request = json.loads((folder / "request.json").read_text())
    detail.update(
        stage=stage,
        sequence=sequence,
        selected=False,
        source_run_id=request["run_id"],
        source_continuation_id=request.get("continuation_id"),
        request_path=str((folder / "request.json").resolve()),
        request_journal_sha256=runner.file_sha256(folder / "request.json"),
        result_path=str((folder / "result.json").resolve()) if result is not None else None,
        raw_result_sha256=runner.file_sha256(folder / "result.json") if result is not None else None,
        http_status=result.get("http_status") if result is not None else None,
        provider_request_id=result.get("headers", {}).get("x-request-id") if result is not None else None,
        received_at=result.get("received_at") if result is not None else None,
    )
    retry = rejection(result) if result is not None else None
    if retry is not None:
        detail.update(status="rejected_retryable", reason=RETRY_CODE, **retry)
    if detail["cost_usd"] is None:
        detail["cost_source"] = "unavailable; full-context liability reserve retained, not an assumed charge"
    return detail


def original_state(run: dict) -> dict:
    report, rows = runner.reconstruct(run, check_payloads=False)
    attempts = []
    for detail in report["attempts"]:
        folder = ORIGINAL / "attempts" / detail["attempt_id"]
        result = (
            json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else None
        )
        attempts.append(decorate(detail, result, folder, "original", 0))
    return {
        "attempts": attempts,
        "rows": {row["case_id"]: row for row in rows},
        "last": {detail["case_id"]: detail for detail in attempts},
        "original_status": report["status"],
    }


def allowed(detail: dict) -> bool:
    return detail["status"] == "rejected_retryable" or (
        detail["status"] in {"valid", "invalid"} and detail["cost_usd"] is not None
    )


def freeze() -> dict:
    if (DATA / "plan.json").exists():
        raise ValueError("Continuation already frozen; use run or verify")
    run = runner.read_run("deepseek")
    original = runner.verify(run)
    state = original_state(run)
    if not all(allowed(detail) for detail in state["attempts"]):
        raise ValueError("Original evidence contains an uncertain or non-capacity failure")
    rejected = [a["case_id"] for a in state["attempts"] if a["status"] == "rejected_retryable"]
    if original["status"] != "blocked" or not rejected:
        raise ValueError("This continuation requires a blocked explicit-capacity-rejection run")
    plan = {
        "schema_version": "dual-high-2000-continuation-plan-v1",
        "created_at": runner.now(),
        "original_run_id": run["run_id"],
        "original_evidence": artifact_pins(ORIGINAL),
        "source_pins": {
            str((HERE / name).resolve()): runner.file_sha256(HERE / name)
            for name in (
                "continuation.py",
                "test_continuation.py",
                "continuation-protocol.md",
                "provider-recovery-notes.md",
                "provider-recovery-notes.json",
            )
        },
        "retained_response_case_ids": sorted(state["rows"]),
        "original_rejected_case_ids": sorted(rejected),
        "remaining_case_ids": [c["case_id"] for c in run["cases"] if c["case_id"] not in state["rows"]],
        "budget": copy.deepcopy(run["budget"]),
        "policy": {
            "initial_concurrency": 8,
            "max_concurrency": 8,
            "minimum_concurrency": 2,
            "halve_concurrency_after_rejection": True,
            "increase_after_successes": 64,
            "maximum_retries_per_case": 5,
            "maximum_total_rejections": 40,
            "retry_only_code": RETRY_CODE,
            "retry_only_http_status": 429,
            "cooldown_initial_seconds": 2,
            "cooldown_cap_seconds": 60,
            "honor_retry_after": True,
            "reset_backoff_after_successes": 16,
            "retain_reserve_for_every_unpriced_rejection": True,
            "unknown_transport_retries": False,
            "invalid_rating_retries": False,
            "fallback": False,
            "prompt_or_image_changes": False,
        },
        "training_ready": False,
    }
    plan["continuation_id"] = runner.digest(plan)
    runner.durable_json(DATA / "plan.json", plan)
    save_report(plan, run, state)
    return plan


def read_plan() -> tuple[dict, dict]:
    plan = json.loads((DATA / "plan.json").read_text())
    if (
        runner.digest({key: value for key, value in plan.items() if key != "continuation_id"})
        != plan["continuation_id"]
    ):
        raise ValueError("Frozen continuation plan changed")
    for filename, expected in {**plan["source_pins"], **plan["original_evidence"]}.items():
        if runner.file_sha256(Path(filename)) != expected:
            raise ValueError("Frozen continuation evidence changed: " + filename)
    if artifact_pins(ORIGINAL) != plan["original_evidence"]:
        raise ValueError("Original evidence set changed")
    run = runner.read_run("deepseek")
    if run["run_id"] != plan["original_run_id"] or run["budget"] != plan["budget"]:
        raise ValueError("Original identity or budget differs")
    return plan, run


def read_attempt(plan: dict, run: dict, folder: Path, by_id: dict) -> tuple[dict, dict | None]:
    request = json.loads((folder / "request.json").read_text())
    case = by_id[request["case_id"]]
    sequence = request["sequence"]
    if type(sequence) is not int or sequence < 1:
        raise ValueError("Invalid continuation attempt sequence")
    attempt_id = f"{case['attempt_id']}--continuation-01--{sequence:03d}"
    expected = {
        "case_id": case["case_id"],
        "arm_id": case["arm_id"],
        "attempt_id": attempt_id,
        "request_sha256": case["request_sha256"],
        "request_bytes": case["request_bytes"],
        "run_id": run["run_id"],
        "continuation_id": plan["continuation_id"],
    }
    if (
        type(sequence) is not int
        or sequence < 1
        or folder.name != attempt_id
        or any(request.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Continuation request differs from the frozen request")
    variant = {**case, "attempt_id": attempt_id}
    result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else None
    if result is None:
        detail, row = (
            {
                "case_id": case["case_id"],
                "arm_id": case["arm_id"],
                "attempt_id": attempt_id,
                "status": "unknown",
                "reason": "interrupted_request",
                "cost_usd": None,
                "task_valid": dict.fromkeys(runner.TASKS, False),
                "wire_score_valid": None,
            },
            None,
        )
    else:
        detail, row = runner.transport.interpret(run, variant, request, result)
        if detail["cost_usd"] is not None and Decimal(detail["cost_usd"]) > Decimal(
            plan["budget"]["reserve_per_request_usd"]
        ):
            detail.update(status="failed", reason="reported_cost_exceeds_full_context_reserve")
        effort = result.get("headers", {}).get("x-runinfra-reasoning-effort")
        if effort is not None and effort != "high":
            detail.update(status="failed", reason="unexpected_returned_reasoning_effort")
    detail = decorate(detail, result, folder, "continuation", sequence)
    if row is not None:
        row["provenance"].update(
            continuation_id=plan["continuation_id"],
            original_attempt_id=case["attempt_id"],
            request_journal_path=detail["request_path"],
            result_path=detail["result_path"],
            raw_result_sha256=detail["raw_result_sha256"],
            attempt_sequence=sequence,
        )
    return detail, row


def consume(state: dict, detail: dict, row: dict | None) -> None:
    previous = state["last"].get(detail["case_id"])
    if previous is not None and previous["status"] != "rejected_retryable":
        raise ValueError("Only an explicit capacity rejection may have a subsequent attempt")
    if detail["sequence"] != (previous["sequence"] + 1 if previous is not None else 1):
        raise ValueError("Continuation attempt sequence is not contiguous")
    state["attempts"].append(detail)
    state["last"][detail["case_id"]] = detail
    if row is not None:
        state["rows"][detail["case_id"]] = row


def reconstruct(plan: dict, run: dict) -> dict:
    state = original_state(run)
    by_id = {case["case_id"]: case for case in run["cases"]}
    root = DATA / "attempts"
    for folder in sorted(root.iterdir()) if root.exists() else []:
        if not folder.is_dir() or not re.fullmatch(
            r"photo_scope--T\d{4}--continuation-01--\d{3}", folder.name
        ):
            raise ValueError("Unexpected continuation attempt directory")
        detail, row = read_attempt(plan, run, folder, by_id)
        if detail["case_id"] not in plan["remaining_case_ids"]:
            raise ValueError("A retained original response was repeated")
        consume(state, detail, row)
    return state


def normalized(run: dict, state: dict, plan: dict | None = None) -> tuple[dict, list[dict]]:
    order = {case["case_id"]: index for index, case in enumerate(run["cases"])}
    rows = [state["rows"][case["case_id"]] for case in run["cases"] if case["case_id"] in state["rows"]]
    selected_ids = {row["attempt_id"] for row in rows}
    ledger = sorted(
        [
            {
                **detail,
                "selected": detail["attempt_id"] in selected_ids,
                "unpriced_liability_reserve_usd": run["budget"]["reserve_per_request_usd"]
                if detail["cost_usd"] is None
                else "0",
            }
            for detail in state["attempts"]
        ],
        key=lambda detail: (order[detail["case_id"]], detail["sequence"]),
    )
    selected = [detail for detail in ledger if detail["selected"]]
    report = runner.summarize(run, selected, rows)
    known = sum((Decimal(a["cost_usd"]) for a in ledger if a["cost_usd"] is not None), Decimal(0))
    unpriced = sum(a["cost_usd"] is None for a in ledger)
    reserve = Decimal(run["budget"]["reserve_per_request_usd"]) * unpriced
    reasons = []
    if not all(allowed(detail) for detail in ledger):
        reasons.append("uncertain_or_non_capacity_failure")
    counts = Counter(a["case_id"] for a in ledger)
    rejected = sum(a["status"] == "rejected_retryable" for a in ledger)
    if plan is not None and len(rows) < len(run["cases"]):
        if any(
            a["status"] == "rejected_retryable" and counts[c] > plan["policy"]["maximum_retries_per_case"]
            for c, a in state["last"].items()
        ):
            reasons.append("per_case_retry_limit")
        if rejected >= plan["policy"]["maximum_total_rejections"]:
            reasons.append("total_rejection_limit")
    if known + reserve > Decimal(run["budget"]["max_cost_usd"]):
        reasons.append("combined_budget_exhausted")
    if reasons:
        report["status"] = "blocked"
    report.update(
        schema_version="dual-high-2000-normalized-route-report-v1",
        continuation_id=plan["continuation_id"] if plan else None,
        original_run_status=state["original_status"],
        http_attempts=len(ledger),
        selected_response_calls=len(rows),
        all_attempts=ledger,
        attempts_semantics="One selected terminal response per image; all HTTP attempts are in all_attempts",
        known_cost_usd=str(known),
        total_known_cost_usd=str(known),
        unpriced_attempts=unpriced,
        unpriced_rejected_attempts=sum(
            a["cost_usd"] is None and a["status"] == "rejected_retryable" for a in ledger
        ),
        explicit_capacity_rejections=rejected,
        unpriced_liability_reserve_usd=str(reserve),
        admission_debit_usd=str(known + reserve),
        blocking_reasons=reasons,
    )
    return report, rows


def save_report(plan: dict, run: dict, state: dict | None = None) -> dict:
    report, rows = normalized(run, state if state is not None else reconstruct(plan, run), plan)
    runner.write_jsonl(DATA / "observations.jsonl", rows)
    runner.write_jsonl(DATA / "all-attempts.jsonl", report["all_attempts"])
    runner.atomic_json(DATA / "report.json", report)
    return report


def verify(plan: dict, run: dict) -> dict:
    runner.validate_run(run)
    report, rows = normalized(run, reconstruct(plan, run), plan)
    if (
        json.loads((DATA / "report.json").read_text()) != report
        or runner.read_jsonl(DATA / "observations.jsonl") != rows
        or runner.read_jsonl(DATA / "all-attempts.jsonl") != report["all_attempts"]
    ):
        raise ValueError("Continuation exports differ from preserved replies")
    return report


def send(run: dict, case: dict, folder: Path, key: str) -> None:
    body = runner.payload(run, case)
    wire = runner.encoded(body)
    if runner.digest(body) != case["request_sha256"] or len(wire) != case["request_bytes"]:
        raise ValueError("Continuation payload changed")
    try:
        response = runner.transport.requests.post(
            run["endpoint"],
            data=wire,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            timeout=(15, 740),
            allow_redirects=False,
        )
        result = {
            "http_status": response.status_code,
            "body": response.text.replace(key, "[REDACTED]"),
            "headers": {
                name.lower(): value.replace(key, "[REDACTED]")
                for name, value in response.headers.items()
                if name.lower().startswith(("x-runinfra-", "x-request-id", "retry-after"))
            },
            "transport_error": None,
            "received_at": runner.now(),
        }
    except runner.transport.requests.RequestException as error:
        result = {
            "http_status": None,
            "body": None,
            "headers": {},
            "transport_error": type(error).__name__,
            "received_at": runner.now(),
        }
    runner.durable_json(folder / "result.json", result)


def execute(plan: dict, run: dict, case_ids: set[str] | None = None) -> dict:
    runner.validate_run(run)
    if case_ids is not None and not case_ids <= set(plan["remaining_case_ids"]):
        raise ValueError("Unknown continuation probe case")
    key = os.environ.get("RUN_INFRA_API_KEY")
    if not key:
        raise ValueError("RUN_INFRA_API_KEY is unavailable")
    with (DATA / ".lock").open("a") as lock, (ORIGINAL / ".lock").open("a") as original_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = reconstruct(plan, run)
        report = save_report(plan, run, state)
        if report["status"] == "blocked":
            raise ValueError("Existing uncertainty or continuation limit prevents further requests")
        pending = deque(
            c
            for c in run["cases"]
            if c["case_id"] not in state["rows"] and (case_ids is None or c["case_id"] in case_ids)
        )
        try:
            stream(plan, run, state, pending, key)
        except BaseException:
            save_report(plan, run)
            raise
        return save_report(plan, run, state)


def stream(plan: dict, run: dict, state: dict, pending: deque, key: str) -> None:
    policy = plan["policy"]
    concurrency = policy["initial_concurrency"]
    active, failures, successes, completions = {}, 0, 0, 0
    recent = [a for a in state["attempts"] if a["status"] == "rejected_retryable"]
    cooldown_until = max(
        (
            datetime.fromisoformat(a["received_at"]).timestamp()
            + max(policy["cooldown_initial_seconds"], a["retry_after_seconds"])
            for a in recent
        ),
        default=0,
    )
    reserve = Decimal(plan["budget"]["reserve_per_request_usd"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=policy["max_concurrency"]) as pool:
        while pending or active:
            report, _ = normalized(run, state, plan)
            stopped = report["status"] == "blocked"
            remaining = Decimal(plan["budget"]["max_cost_usd"]) - Decimal(report["admission_debit_usd"])
            affordable = max(0, int(max(remaining, Decimal(0)) / reserve) - len(active))
            admitted = (
                0
                if stopped or time.time() < cooldown_until
                else min(max(0, concurrency - len(active)), affordable, len(pending))
            )
            for _ in range(admitted):
                case = pending.popleft()
                previous = state["last"].get(case["case_id"])
                if previous is not None and previous["status"] != "rejected_retryable":
                    raise ValueError("A terminal or uncertain attempt cannot be repeated")
                sequence = previous["sequence"] + 1 if previous else 1
                attempt_id = f"{case['attempt_id']}--continuation-01--{sequence:03d}"
                folder = DATA / "attempts" / attempt_id
                folder.mkdir(parents=True)
                request = {
                    name: case[name] for name in ("case_id", "arm_id", "request_sha256", "request_bytes")
                }
                request.update(
                    attempt_id=attempt_id,
                    sequence=sequence,
                    run_id=run["run_id"],
                    continuation_id=plan["continuation_id"],
                    created_at=runner.now(),
                )
                runner.durable_json(folder / "request.json", request)
                active[pool.submit(send, run, case, folder, key)] = (case, folder)
            if not active:
                if stopped or not pending:
                    break
                if not affordable:
                    raise ValueError(
                        "Next request exceeds combined admission budget including rejection reserves"
                    )
                time.sleep(min(30, max(0.01, cooldown_until - time.time())))
                continue
            done, _ = concurrent.futures.wait(
                active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
            )
            had_rejection = False
            for future in done:
                case, folder = active.pop(future)
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 — persist uncertainty without retrying.
                    print(
                        json.dumps({"case_id": case["case_id"], "local_error_type": type(error).__name__}),
                        flush=True,
                    )
                detail, row = read_attempt(plan, run, folder, {case["case_id"]: case})
                consume(state, detail, row)
                completions += 1
                if detail["status"] == "rejected_retryable":
                    had_rejection = True
                    pending.appendleft(case)
                    failures += 1
                    successes = 0
                    concurrency = max(policy["minimum_concurrency"], concurrency // 2)
                    backoff = min(
                        policy["cooldown_cap_seconds"],
                        policy["cooldown_initial_seconds"] * 2 ** min(failures - 1, 10),
                    )
                    cooldown_until = max(
                        cooldown_until, time.time() + max(backoff, detail["retry_after_seconds"])
                    )
                elif detail["status"] in {"valid", "invalid"} and detail["cost_usd"] is not None:
                    successes += 1
                    if successes >= policy["reset_backoff_after_successes"]:
                        failures = 0
                    if successes >= policy["increase_after_successes"]:
                        concurrency = min(policy["max_concurrency"], concurrency + 1)
                        successes = 0
            if completions and (completions // 32 > (completions - len(done)) // 32 or had_rejection):
                report = save_report(plan, run, state)
                print(
                    json.dumps(
                        {
                            key: report[key]
                            for key in (
                                "selected_response_calls",
                                "http_attempts",
                                "known_cost_usd",
                                "unpriced_rejected_attempts",
                                "unpriced_liability_reserve_usd",
                                "admission_debit_usd",
                                "status",
                            )
                        }
                        | {"concurrency": concurrency}
                    ),
                    flush=True,
                )


def load_verified_route(route: str) -> tuple[dict, dict, list[dict], dict]:
    if route == "deepseek" and (DATA / "plan.json").exists():
        plan, run = read_plan()
        report = verify(plan, run)
        rows = runner.read_jsonl(DATA / "observations.jsonl")
        paths = [
            DATA / name for name in ("plan.json", "report.json", "observations.jsonl", "all-attempts.jsonl")
        ]
        paths += [ORIGINAL / name for name in ("run.json", "report.json", "observations.jsonl")]
    else:
        run = runner.read_run(route)
        runner.verify(run)
        directory = runner.transport.directory(run)
        original_report, rows = runner.reconstruct(run, check_payloads=False)
        ledger = [
            decorate(
                detail,
                json.loads((directory / "attempts" / detail["attempt_id"] / "result.json").read_text()),
                directory / "attempts" / detail["attempt_id"],
                "original",
                0,
            )
            for detail in original_report["attempts"]
        ]
        state = {
            "attempts": ledger,
            "rows": {row["case_id"]: row for row in rows},
            "last": {a["case_id"]: a for a in ledger},
            "original_status": original_report["status"],
        }
        report, rows = normalized(run, state)
        paths = [directory / name for name in ("run.json", "report.json", "observations.jsonl")]
    return run, report, rows, {str(path.resolve()): runner.file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "run", "verify"))
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    if args.command == "probe" and not args.case_id:
        parser.error("probe requires --case-id")
    if args.command == "freeze":
        freeze()
    plan, run = read_plan()
    report = (
        execute(plan, run, set(args.case_id) if args.case_id else None)
        if args.command in {"probe", "run"}
        else verify(plan, run)
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"attempts", "all_attempts"}}, indent=2
        )
    )
    if report["status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
