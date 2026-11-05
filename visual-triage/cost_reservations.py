"""Frozen budget reservations preserve unknown provider costs and failed attempts."""

import argparse
import hashlib
import json
from pathlib import Path

from bench_protocol import digest

TRACKS = {"quality_v2", "ai_detection_v2"}
RESERVATION_USD = 0.10


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def track_directory(ledger):
    for directory in Path(ledger).resolve().parents:
        if directory.name in TRACKS and directory.parent.name == "tracks":
            return directory
    return None


def eligible(record):
    return (
        record.get("transport_error") == "ConnectionError"
        and record.get("status") is None
        and not record.get("response")
        and not record.get("generation_id")
        and not record.get("usage")
        and record.get("cost_usd") is None
        and record.get("effective_cost_usd") is None
        and record.get("normalized") is None
    )


def reservation(record, ledger):
    """Return a guard allowance for an exact reviewed failure, never an actual cost."""
    directory = track_directory(ledger)
    if directory is None or not (directory / "cost-reservations.json").exists():
        return None
    plan = json.loads((directory / "cost-reservations.json").read_bytes())
    if plan.get("version") != 1 or plan.get("track") != directory.name or plan.get("no_retry") is not True:
        raise ValueError("Invalid frozen cost-reservation policy")
    for field, path in (
        ("protocol_sha256", directory / "protocol.json"),
        ("run_sha256", directory / "optimisation/run.json"),
        ("evidence_sha256", directory / plan["evidence_path"]),
    ):
        if not path.resolve().is_relative_to(directory) or sha_bytes(path.read_bytes()) != plan.get(field):
            raise ValueError("Cost-reservation evidence or frozen inputs changed")
    entries = plan.get("reservations")
    if not isinstance(entries, list) or len(entries) != 1 or not plan.get("review_reason"):
        raise ValueError("A reservation must identify one explicitly reviewed failure")
    entry = entries[0]
    if entry.get("ledger_path") != str(Path(ledger).resolve().relative_to(directory)):
        return None
    raw = Path(ledger).read_bytes()
    size = entry.get("ledger_prefix_bytes")
    if (
        type(size) is not int
        or size < 1
        or size > len(raw)
        or sha_bytes(raw[:size]) != entry.get("ledger_prefix_sha256")
    ):
        raise ValueError("Original cost-reserved ledger prefix changed")
    if entry.get("record_sha256") != digest(record):
        return None
    if (
        not eligible(record)
        or entry.get("sample") != record.get("sample")
        or entry.get("request_sha256") != record.get("request_sha256")
        or entry.get("reservation_usd") != RESERVATION_USD
        or entry.get("actual_cost_usd", "missing") is not None
    ):
        raise ValueError("Cost reservation does not match the reviewed unknown-cost failure")
    return RESERVATION_USD


def guard_cost(record, ledger, actual):
    if actual is not None:
        return actual
    amount = reservation(record, ledger)
    if amount is None:
        raise ValueError("Unknown cost has no matching reviewed reservation")
    return amount


def totals(entries):
    """Entries contain (ledger path, original record, provider-reported cost or None)."""
    known, reserved, unknown = 0.0, 0.0, 0
    for path, record, actual in entries:
        amount = guard_cost(record, path, actual)
        if actual is None:
            unknown += 1
            reserved += amount
        else:
            known += amount
    return {
        "actual_total_usd": None if unknown else known,
        "known_reported_usd": known,
        "reserved_unverified_usd": reserved,
        "budget_accounted_usd": known + reserved,
        "unknown_cost_count": unknown,
    }


def freeze(ledger, sample, request_sha256, evidence, reason):
    ledger, evidence = Path(ledger).resolve(), Path(evidence).resolve()
    directory = track_directory(ledger)
    if directory is None or not evidence.is_relative_to(directory) or not reason.strip():
        raise ValueError("Use a v2 ledger, evidence within its track and an explicit review reason")
    raw = ledger.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    matches = [
        row for row in rows if row.get("sample") == sample and row.get("request_sha256") == request_sha256
    ]
    if len(matches) != 1 or not eligible(matches[0]):
        raise ValueError("Exactly one eligible unknown-cost ConnectionError is required")
    row = matches[0]
    value = {
        "version": 1,
        "track": directory.name,
        "no_retry": True,
        "review_reason": reason,
        "protocol_sha256": sha_bytes((directory / "protocol.json").read_bytes()),
        "run_sha256": sha_bytes((directory / "optimisation/run.json").read_bytes()),
        "evidence_path": str(evidence.relative_to(directory)),
        "evidence_sha256": sha_bytes(evidence.read_bytes()),
        "reservations": [
            {
                "ledger_path": str(ledger.relative_to(directory)),
                "sample": sample,
                "request_sha256": request_sha256,
                "record_sha256": digest(row),
                "ledger_prefix_bytes": len(raw),
                "ledger_prefix_sha256": sha_bytes(raw),
                "reservation_usd": RESERVATION_USD,
                "actual_cost_usd": None,
                "basis": "reviewed_budget_reservation_not_provider_reported_cost",
            }
        ],
    }
    path = directory / "cost-reservations.json"
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(freeze(args.ledger, args.sample, args.request_sha256, args.evidence, args.reason))


if __name__ == "__main__":
    main()
