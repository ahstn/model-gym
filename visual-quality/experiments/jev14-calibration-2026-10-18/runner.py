"""Run the frozen calibration study with bounded concurrency and no retries."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/jev14-calibration-2026-10-18"
SOURCE = Path("/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage")
SPLITS = ("calibration", "validation")
sys.path.insert(0, str(SOURCE))

from bench_protocol import DECISIONS, digest, validate_prompt
from bench_runner import effective_cost
from tracks.jev14.client import HEADERS, MODEL, REVISION, make_payload, post
from tracks.jev14.latest import review


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_id(case):
    identifier = case.get("row_id", case.get("case_id", case.get("id")))
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Case requires a nonempty row_id, case_id or id")
    return identifier


def ledger_path(split):
    return DATA / f"{split}.jsonl"


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def failed(row):
    return (
        row.get("status") != 200
        or row.get("resolved_model") != REVISION
        or not isinstance(row.get("normalized"), dict)
        or any(row.get(key) for key in ("validation_error", "transport_error", "cost_validation_error"))
        or row.get("effective_cost_usd") is None
    )


def selection_hash(protocol_hash, cases, rows):
    """A validation call requires a frozen choice made after complete calibration."""
    selection = read(HERE / "selection.json")
    selected_hash = sha(HERE / "selection.json")
    if read(HERE / "selection-lock.json").get("selection_sha256") != selected_hash:
        raise ValueError("Selection differs from its frozen lock")
    expected = {case_id(case) for case in cases if case["split"] == "calibration"}
    calibration = rows["calibration"]
    if {row["row_id"] for row in calibration} != expected or any(failed(row) for row in calibration):
        raise ValueError("Validation requires complete, valid and priced calibration")
    if selection.get("protocol_sha256") != protocol_hash or selection.get("calibration_ledger_sha256") != sha(
        ledger_path("calibration")
    ):
        raise ValueError("Selection protocol or calibration ledger changed")
    return selected_hash


def check_record(row, case, protocol, prompt, selected_hash):
    protocol_hash = sha(HERE / "protocol.json")
    payload = make_payload(DATA, case, prompt)
    expected_id = digest([protocol_hash, case["split"], case_id(case)])
    if (
        row.get("row_id") != case_id(case)
        or row.get("split") != case["split"]
        or row.get("case_sha256") != digest(case)
        or row.get("attempt_id") != expected_id
        or row.get("protocol_sha256") != protocol_hash
        or row.get("manifest_sha256") != protocol["manifest_sha256"]
        or row.get("prompt_sha256") != digest(prompt)
        or row.get("request_sha256")
        != digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload})
        or row.get("selection_sha256") != (selected_hash if case["split"] == "validation" else None)
    ):
        raise ValueError("Saved request differs from frozen inputs")
    response = row.get("response")
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    if row.get("usage") != usage or row.get("resolved_model") != response.get("model"):
        raise ValueError("Saved usage or model differs from the response")
    strict = {**row, "normalized": row.get("strict_normalized")}
    strict.pop("validation_error", None)
    if row.get("strict_validation_error"):
        strict["validation_error"] = row["strict_validation_error"]
    parsed = review(strict, prompt)
    if row.get("normalized") != parsed["normalized"] or row.get("validation_error") != parsed.get(
        "validation_error"
    ):
        raise ValueError("Saved normalized values differ from the native response")
    try:
        cost, basis = effective_cost(usage)
    except (TypeError, ValueError):
        if not row.get("cost_validation_error") or row.get("effective_cost_usd") is not None:
            raise ValueError("Invalid cost lacks a preserved failure") from None
    else:
        if row.get("effective_cost_usd") != cost or row.get("cost_basis") != basis:
            raise ValueError("Saved cost changed")


def load_study(require_selection=False):
    """Verify inputs and saved outputs without sending a request."""
    protocol = read(HERE / "protocol.json")
    if (
        sha(HERE / "manifest.json") != protocol["manifest_sha256"]
        or sha(HERE / "prompt.json") != protocol["prompt_sha256"]
    ):
        raise ValueError("Frozen manifest or prompt changed")
    pins = protocol["execution_sha256"]
    if str(Path(__file__).resolve()) not in pins:
        raise ValueError("Protocol does not pin this runner")
    for path, expected in pins.items():
        if not Path(path).is_absolute() or sha(Path(path)) != expected:
            raise ValueError(f"Frozen execution file changed: {path}")
    for path, expected in protocol.get("source_inputs", {}).items():
        if not Path(path).is_absolute() or sha(Path(path)) != expected:
            raise ValueError(f"Frozen source file changed: {path}")
    if protocol.get("model", MODEL) != MODEL or protocol.get("expected_revision", REVISION) != REVISION:
        raise ValueError("Frozen model differs from the client")
    cases = read(HERE / "manifest.json")["cases"]
    identities = [(case["split"], case_id(case)) for case in cases]
    if (
        len(cases) != protocol["maximum_attempts"]
        or len(set(identities)) != len(cases)
        or any(split not in SPLITS for split, _ in identities)
    ):
        raise ValueError("Invalid frozen case allocation")
    prompt = read(HERE / "prompt.json")
    validate_prompt(prompt, "ai")
    for case in cases:
        make_payload(DATA, case, prompt)
    rows = {
        split: [json.loads(line) for line in ledger_path(split).read_text().splitlines() if line.strip()]
        if ledger_path(split).exists()
        else []
        for split in SPLITS
    }
    selected_hash = (
        selection_hash(sha(HERE / "protocol.json"), cases, rows)
        if require_selection or rows["validation"]
        else None
    )
    by_id = dict(zip(identities, cases))
    seen, attempts = set(), set()
    for split in SPLITS:
        for row in rows[split]:
            key = (split, row.get("row_id"))
            attempt = row.get("attempt")
            if (
                key not in by_id
                or key in seen
                or type(attempt) is not int
                or not 1 <= attempt <= len(cases)
                or attempt in attempts
            ):
                raise ValueError("Duplicate, unexpected or invalid saved attempt")
            seen.add(key)
            attempts.add(attempt)
            check_record(row, by_id[key], protocol, prompt, selected_hash)
    return protocol, cases, prompt, rows


def dispatch_record(case, payload, prompt, protocol, ordinal, selected_hash):
    protocol_hash = sha(HERE / "protocol.json")
    return {
        "row_id": case_id(case),
        "split": case["split"],
        "task": "ai",
        "arm": "jev14",
        "attempt": ordinal,
        "attempt_id": digest([protocol_hash, case["split"], case_id(case)]),
        "case_sha256": digest(case),
        "protocol_sha256": protocol_hash,
        "manifest_sha256": protocol["manifest_sha256"],
        "prompt_sha256": digest(prompt),
        "prompt_id": prompt["id"],
        "selection_sha256": selected_hash,
        "request_sha256": digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload}),
        "timestamp": datetime.now(UTC).isoformat(),
        "delivered_sha256": case["delivered_sha256"],
    }


def request(payload, prompt):
    """Catch unexpected client errors without exposing exception text or credentials."""
    started = time.perf_counter()
    try:
        return review(post(payload, prompt), prompt)
    except Exception as exc:  # noqa: BLE001 - retain every possibly billed attempt without secret text.
        return review(
            {
                "transport_error": type(exc).__name__,
                "seconds": time.perf_counter() - started,
                "normalized": None,
                "usage": {},
                "cost_usd": None,
                "effective_cost_usd": None,
                "cost_basis": "unknown_openrouter_cost",
                "resolved_model": None,
            },
            prompt,
        )


def save_result(future, metadata, pending):
    row = {**future.result(), **metadata}
    with ledger_path(row["split"]).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(DATA)
    pending.unlink()
    sync_directory(pending.parent)
    return row


def execute(protocol, cases, prompt, rows, split, max_new, selected_hash):
    completed = {row["row_id"] for row in rows[split]}
    todo = [case for case in cases if case["split"] == split and case_id(case) not in completed][:max_new]
    all_rows = [row for group in rows.values() for row in group]
    spent = sum(row["effective_cost_usd"] for row in all_rows)
    ordinal = max((row["attempt"] for row in all_rows), default=0)
    concurrency, interval = protocol["concurrency"], protocol["minimum_interval_seconds"]
    reserve, budget = protocol["reserve_per_request_usd"], protocol["budget_usd"]
    if (
        type(concurrency) is not int
        or not 1 <= concurrency <= 4
        or not 0.25 <= interval
        or not 0.012 <= reserve <= budget <= 0.25
    ):
        raise ValueError("Run limits exceed the approved concurrency, rate or budget")
    if not all(math.isfinite(value) for value in (interval, reserve, budget)):
        raise ValueError("Non-finite execution limit")
    inflight, dispatched, finished, stopped = {}, 0, 0, None
    latest = max((datetime.fromisoformat(row["timestamp"]).timestamp() for row in all_rows), default=0)
    next_start = time.monotonic() + max(0, interval - (time.time() - latest))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while inflight or (dispatched < len(todo) and stopped is None):
            for future in [item for item in inflight if item.done()]:
                metadata, pending = inflight.pop(future)
                row = save_result(future, metadata, pending)
                rows[split].append(row)
                finished += 1
                spent += row["effective_cost_usd"] or 0
                if failed(row):
                    stopped = "failed_or_unpriced_request"
                if finished % 25 == 0 or failed(row):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "completed": len(rows[split]),
                                "new_completed": finished,
                                "inflight": len(inflight),
                                "known_cost_usd": spent,
                                "stop_reason": stopped,
                            }
                        ),
                        flush=True,
                    )
            if (
                stopped is None
                and dispatched < len(todo)
                and len(inflight) < concurrency
                and time.monotonic() >= next_start
            ):
                if spent + reserve * (len(inflight) + 1) > budget:
                    if not inflight:
                        stopped = "budget_guard"
                else:
                    try:
                        case = todo[dispatched]
                        payload = make_payload(DATA, case, prompt)
                        ordinal += 1
                        metadata = dispatch_record(case, payload, prompt, protocol, ordinal, selected_hash)
                        pending = DATA / "pending" / (metadata["attempt_id"] + ".json")
                        write_new(pending, metadata)
                        inflight[pool.submit(request, payload, prompt)] = (metadata, pending)
                        dispatched += 1
                        next_start = time.monotonic() + interval
                        continue
                    except Exception as exc:  # noqa: BLE001 - stop dispatch, then drain paid requests.
                        stopped = "dispatch_preparation_" + type(exc).__name__
            if inflight:
                can_dispatch = stopped is None and dispatched < len(todo) and len(inflight) < concurrency
                timeout = (
                    max(0, next_start - time.monotonic())
                    if can_dispatch and spent + reserve * (len(inflight) + 1) <= budget
                    else None
                )
                wait(inflight, timeout=timeout, return_when=FIRST_COMPLETED)
            elif stopped is None and dispatched < len(todo):
                time.sleep(max(0, next_start - time.monotonic()))
    expected = sum(case["split"] == split for case in cases)
    return {
        "status": "stopped" if stopped else "complete" if len(rows[split]) == expected else "partial",
        "split": split,
        "completed": len(rows[split]),
        "expected": expected,
        "new_attempts": dispatched,
        "known_cost_usd": spent,
        "stop_reason": stopped,
    }


def run(split, max_new):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "pending").mkdir(exist_ok=True)
    with (DATA / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, cases, prompt, rows = load_study(require_selection=split == "validation")
        if list((DATA / "pending").glob("*.json")):
            raise ValueError("Unreconciled request intent; inspect before further calls")
        if any(failed(row) for group in rows.values() for row in group):
            raise ValueError("Prior failed or unpriced request; no automatic retries")
        if type(max_new) is not int or not 1 <= max_new <= protocol["maximum_attempts"]:
            raise ValueError("Invalid request count")
        if split == "calibration" and (HERE / "selection-lock.json").exists():
            raise ValueError("Calibration is sealed by the selection lock")
        selected_hash = (
            selection_hash(sha(HERE / "protocol.json"), cases, rows) if split == "validation" else None
        )
        return execute(protocol, cases, prompt, rows, split, max_new, selected_hash)


def verify(require_selection=False):
    """Verify a stopped run; unresolved calls prevent selection or further dispatch."""
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, _, _, rows = load_study(require_selection=require_selection)
        if list((DATA / "pending").glob("*.json")):
            raise ValueError("Unreconciled request intent; inspect before selection")
        if any(failed(row) for group in rows.values() for row in group):
            raise ValueError("Failed or unpriced request requires review")
        return {"verified": {split: len(group) for split, group in rows.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "run"))
    parser.add_argument("--split", choices=SPLITS, default="calibration")
    parser.add_argument("--max-new", type=int, default=1600)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.split, args.max_new)
    else:
        result = verify(require_selection=args.split == "validation")
    print(json.dumps(result, indent=2))
