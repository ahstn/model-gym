"""Continue the frozen Jev trial after reviewing probability-rounding validation."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import DECISIONS, digest
from bench_runner import effective_cost
from tracks.jev14.client import HEADERS, MODEL, REVISION, make_payload, post
from tracks.jev14.rounding import normalize
from tracks.jev14.run import LEDGER, MANIFEST, PROTOCOL, ROOT, TRACK, sha, write_new

AMENDMENT = TRACK / "parser-amendment.json"
ROUNDING_ERRORS = {
    None,
    "ValueError: Probability distribution does not sum to one",
    "ValueError: Score differs from its probability-weighted mean",
}


def read_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    if sha(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("Manifest changed")
    pins = {path: pin["sha256"] for path, pin in protocol["source_inputs"].items()}
    pins.update(protocol["execution_sha256"])
    for path, expected in pins.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    cases = json.loads(MANIFEST.read_text())["cases"]
    raw = LEDGER.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) > len(cases):
        raise ValueError("Attempt limit exceeded")
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        prompt = json.loads((ROOT / f"prompts/final/{case['task']}.json").read_text())
        payload = make_payload(ROOT, case, prompt)
        expected_request = digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload})
        if (
            row["case_id"] != case["case_id"]
            or row["attempt"] != index
            or row["protocol_sha256"] != sha(PROTOCOL)
            or row["manifest_sha256"] != protocol["manifest_sha256"]
            or row["prompt_sha256"] != digest(prompt)
            or row["request_sha256"] != expected_request
            or row["request_headers"] != HEADERS
            or row["requested_model"] != MODEL
            or row["delivered_sha256"] != case["delivered_sha256"]
        ):
            raise ValueError("Saved request differs from frozen order or inputs")
        if row.get("status") != 200 or row.get("transport_error") or row.get("cost_validation_error"):
            raise ValueError("A transport or cost error requires another review")
        if row.get("validation_error") not in ROUNDING_ERRORS:
            raise ValueError("A validation error outside probability rounding needs review")
        response = row["response"]
        if (
            row["resolved_model"] != REVISION
            or response.get("model") != REVISION
            or row["usage"] != response.get("usage")
        ):
            raise ValueError("Saved model or usage changed")
        parsed = normalize(response, prompt["questions"])
        if "reviewed_normalized" in row and row["reviewed_normalized"] != parsed:
            raise ValueError("Reviewed result differs from raw response")
        cost, basis = effective_cost(row["usage"])
        if cost is None or row["effective_cost_usd"] != cost or row["cost_basis"] != basis:
            raise ValueError("Unknown or changed request cost")
    return protocol, cases, rows, raw


def freeze():
    protocol, _, rows, raw = read_inputs()
    rejected = [row for row in rows if row.get("validation_error")]
    if len(rows) != 31 or len(rejected) != 1 or rejected[0]["case_id"] != "primary/quality/quality-012":
        raise ValueError("Unexpected reviewed prefix")
    amendment = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha(PROTOCOL),
        "prefix_rows": len(rows),
        "prefix_sha256": hashlib.sha256(raw).hexdigest(),
        "parser_sha256": sha(TRACK / "rounding.py"),
        "resume_sha256": sha(Path(__file__)),
        "maximum_attempts": protocol["maximum_attempts"],
        "reason": "Two-decimal probability rounding produced sum0.99; preserve native scores",
        "rule": "Apply bounded rounding-aware validation to every response, including saved rows",
        "unchanged": "Images, prompts, native values, classification thresholds, order and budget",
        "retries": "None; continue unattempted cases only. Original evidence remains unchanged.",
    }
    write_new(AMENDMENT, amendment)
    return {"amendment_sha256": sha(AMENDMENT), "preserved_prefix_rows": len(rows)}


def load_reviewed():
    protocol, cases, rows, raw = read_inputs()
    amendment = json.loads(AMENDMENT.read_text())
    prefix = b"".join(raw.splitlines(keepends=True)[: amendment["prefix_rows"]])
    if (
        amendment["protocol_sha256"] != sha(PROTOCOL)
        or amendment["parser_sha256"] != sha(TRACK / "rounding.py")
        or amendment["resume_sha256"] != sha(Path(__file__))
        or hashlib.sha256(prefix).hexdigest() != amendment["prefix_sha256"]
    ):
        raise ValueError("Parser amendment or preserved ledger prefix changed")
    for row in rows[amendment["prefix_rows"] :]:
        if row.get("amendment_sha256") != sha(AMENDMENT) or "reviewed_normalized" not in row:
            raise ValueError("Continued request lacks its parser amendment")
        if row.get("reviewed_validation_error"):
            raise ValueError("A new invalid response requires review")
    return protocol, cases, rows


def run(max_new):
    protocol, cases, rows = load_reviewed()
    if not 1 <= max_new <= protocol["maximum_attempts"]:
        raise ValueError("Invalid request count")
    for case in cases:
        if sha(ROOT / case["image"]) != case["delivered_sha256"]:
            raise ValueError("Frozen image changed")
    spent = sum(row["effective_cost_usd"] for row in rows)
    for case in cases[len(rows) : len(rows) + max_new]:
        if spent + protocol["reserve_per_request_usd"] > protocol["budget_usd"]:
            raise ValueError("Budget guard reached")
        previous = datetime.fromisoformat(rows[-1]["timestamp"]).timestamp()
        time.sleep(max(0, protocol["minimum_interval_seconds"] - (time.time() - previous)))
        prompt = json.loads((ROOT / f"prompts/final/{case['task']}.json").read_text())
        row = post(make_payload(ROOT, case, prompt), prompt)
        row["reviewed_normalized"] = None
        try:
            if row.get("status") != 200 or row.get("resolved_model") != REVISION:
                raise ValueError("HTTP or model validation failed")
            if row.get("validation_error") not in ROUNDING_ERRORS:
                raise ValueError("Validation failure is outside probability rounding")
            row["reviewed_normalized"] = normalize(row["response"], prompt["questions"])
        except (ValueError, TypeError, KeyError) as exc:
            row["reviewed_validation_error"] = type(exc).__name__ + ": " + str(exc)
        row.update({key: value for key, value in case.items() if key not in {"image", "eligible"}})
        row.update(
            arm="jev14",
            attempt=len(rows) + 1,
            prompt_id=prompt["id"],
            prompt_sha256=digest(prompt),
            manifest_sha256=protocol["manifest_sha256"],
            protocol_sha256=sha(PROTOCOL),
            amendment_sha256=sha(AMENDMENT),
        )
        with LEDGER.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        print(
            json.dumps(
                {
                    "attempt": len(rows),
                    "case": case["case_id"],
                    "status": row.get("status"),
                    "valid": row["reviewed_normalized"] is not None,
                    "cost": row["effective_cost_usd"],
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
        if row.get("reviewed_validation_error") or row["effective_cost_usd"] is None:
            return {"status": "stopped", "attempts": len(rows)}
        spent += row["effective_cost_usd"]
    return {
        "status": "complete" if len(rows) == len(cases) else "partial",
        "attempts": len(rows),
        "reported_cost_usd": spent,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--max-new", type=int, default=126)
    args = parser.parse_args()
    print(json.dumps(freeze() if args.command == "freeze" else run(args.max_new), indent=2))


if __name__ == "__main__":
    main()
