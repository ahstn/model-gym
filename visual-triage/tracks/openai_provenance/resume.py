"""Resume after the reviewed HTTP 502 failure without changing the frozen study."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import digest
from tracks.openai_provenance.client import check_image
from tracks.openai_provenance.study import DATA, ROOT, TRACK, load_study, sha, write_new

AMENDMENT = TRACK / "resume-amendment.json"


def freeze():
    protocol, _, rows = load_study()
    failures = [row for row in rows if not row["success"]]
    if not failures or any(row["http_status"] != 502 for row in failures):
        raise ValueError("Review requires an observed HTTP 502 failure only")
    value = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha(TRACK / "protocol.json"),
        "resume_code_sha256": sha(Path(__file__)),
        "observed_failures": [
            {"case": row["case"], "attempt": row["attempt"], "record_sha256": digest(row)} for row in failures
        ],
        "reason": "The official guide allows transient server retries; preserve original failures",
        "eligible_status": 502,
        "max_retries_per_case": 1,
        "max_total_attempts": protocol["maximum_attempts"] + 16,
        "minimum_interval_seconds": protocol["minimum_interval_seconds"],
        "minimum_retry_delay_seconds": 30,
        "unchanged": "Cohort, native bytes, API request, labels, order and interpretations",
        "retry_ledger": "data/openai-provenance-v1/retries.jsonl",
        "cost": "Failed and successful request charges remain unreported, not zero",
    }
    write_new(AMENDMENT, value)
    return {"amendment_sha256": sha(AMENDMENT)}


def retry_rows():
    path = DATA / "retries.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    )


def validate(amendment, protocol, rows, retries):
    if amendment["protocol_sha256"] != sha(TRACK / "protocol.json"):
        raise ValueError("Resume amendment points to another protocol")
    if amendment["resume_code_sha256"] != sha(Path(__file__)):
        raise ValueError("Resume code changed")
    by_id = {row["case"]: row for row in rows}
    for failure in amendment["observed_failures"]:
        if digest(by_id[failure["case"]]) != failure["record_sha256"]:
            raise ValueError("Reviewed failure was changed")
    retried = set()
    for row in retries:
        parent = by_id.get(row["case"])
        if row["case"] in retried or not parent or parent["success"] or parent["http_status"] != 502:
            raise ValueError("Unexpected repeated attempt")
        if (
            row["retry_of_record_sha256"] != digest(parent)
            or row["retry_of_attempt"] != parent["attempt"]
            or row["input_sha256"] != parent["input_sha256"]
            or row["protocol_sha256"] != amendment["protocol_sha256"]
            or row["amendment_sha256"] != sha(AMENDMENT)
        ):
            raise ValueError("Retry does not match its original request")
        retried.add(row["case"])
    for row in rows + retries:
        if not row["success"] and row["http_status"] != amendment["eligible_status"]:
            raise ValueError("Another failure needs review before further calls")
    if len(rows) > protocol["maximum_attempts"] or len(rows) + len(retries) > amendment["max_total_attempts"]:
        raise ValueError("Attempt limit exceeded")


def run(max_new):
    protocol, cases, rows = load_study()
    amendment = json.loads(AMENDMENT.read_text())
    if not 1 <= max_new <= amendment["max_total_attempts"]:
        raise ValueError("max-new exceeds the frozen attempt limit")
    retries = retry_rows()
    validate(amendment, protocol, rows, retries)
    done, retried = {row["case"] for row in rows}, {row["case"] for row in retries}
    pending = [(case, None) for case in cases if case["case"] not in done]
    by_id = {case["case"]: case for case in cases}
    pending += [
        (by_id[row["case"]], row) for row in rows if not row["success"] and row["case"] not in retried
    ]
    pending = pending[:max_new]
    for case, parent in pending:
        if len(rows) + len(retries) >= amendment["max_total_attempts"]:
            return {"status": "attempt_limit", "attempts": len(rows) + len(retries)}
        previous = max(datetime.fromisoformat(row["started_at"]).timestamp() for row in rows + retries)
        delay = amendment["minimum_interval_seconds"] - (time.time() - previous)
        if parent:
            delay = max(
                delay,
                amendment["minimum_retry_delay_seconds"]
                - (
                    time.time()
                    - datetime.fromisoformat(parent["started_at"]).timestamp()
                    - parent["elapsed_seconds"]
                ),
            )
        time.sleep(max(0, delay))
        row = check_image(ROOT / case["native_image"], case["native_sha256"])
        row.update(
            case=case["case"],
            source=case["source"],
            protocol_sha256=amendment["protocol_sha256"],
            amendment_sha256=sha(AMENDMENT),
            attempt=len(rows) + len(retries) + 1,
        )
        filename = "attempts.jsonl"
        if parent:
            row.update(retry_of_record_sha256=digest(parent), retry_of_attempt=parent["attempt"])
            filename = "retries.jsonl"
        with (DATA / filename).open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        (retries if parent else rows).append(row)
        print(
            json.dumps(
                {
                    "attempts": len(rows) + len(retries),
                    "case": case["case"],
                    "retry": parent is not None,
                    "success": row["success"],
                    "http_status": row["http_status"],
                    "seconds": row["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        if not row["success"]:
            return {"status": "stopped", "attempts": len(rows) + len(retries)}
    valid = {row["case"] for row in rows + retries if row["success"]}
    return {
        "status": "complete" if len(valid) == len(cases) else "partial",
        "valid_cases": len(valid),
        "attempts": len(rows) + len(retries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--max-new", type=int, default=160)
    args = parser.parse_args()
    if not 1 <= args.max_new <= 176:
        parser.error("max-new must be between 1 and 176")
    print(json.dumps(freeze() if args.command == "freeze" else run(args.max_new), indent=2))


if __name__ == "__main__":
    main()
