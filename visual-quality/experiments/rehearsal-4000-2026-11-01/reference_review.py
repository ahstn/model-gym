"""Anonymous review packets, measured review duration, and immutable reference freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("technical_quality", "aesthetics")
LIMITATIONS = {"limited_detail", "partial_view", "ambiguous_style", "unsupported_domain"}
BENCHMARK_PARTITIONS = {"development", "representative", "challenge"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc() -> str:
    return datetime.now(UTC).isoformat()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def packet_case(packet: Path, case_id: str) -> dict:
    cases = json.loads(packet.read_text())["cases"]
    matches = [case for case in cases if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Case must occur exactly once in assigned packet")
    return matches[0]


def begin(packet: Path, case_id: str, timing_file: Path) -> str:
    case = packet_case(packet, case_id)
    image = Path(case["image_path"])
    if digest(image) != case["view_sha256"]:
        raise ValueError("Review image changed")
    write_new(
        timing_file,
        {
            "case_id": case_id,
            "view_sha256": case["view_sha256"],
            "packet_sha256": digest(packet),
            "review_started_at": utc(),
            "monotonic_started": time.monotonic(),
        },
    )
    return str(image)


def validate_assessment(row: dict) -> None:
    if row["eligibility"] not in {"eligible", "excluded", "uncertain"}:
        raise ValueError("Invalid eligibility")
    if not row["eligibility_evidence"].strip():
        raise ValueError("Missing eligibility evidence")
    for task in TASKS:
        result = row[task]
        if result["task"] != task or result["status"] not in {"ok", "unscorable"}:
            raise ValueError("Invalid task/status")
        score = result["score"]
        if result["status"] == "unscorable":
            if score is not None:
                raise ValueError("Unscorable task must have null score")
        elif (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 1 <= score <= 5
            or not math.isclose(score * 10, round(score * 10), abs_tol=1e-8)
        ):
            raise ValueError("Reference scores must be 1–5 in tenths")
        if not 1 <= len(result["evidence"].split()) <= 30:
            raise ValueError("Evidence must contain 1–30 words")
        if not isinstance(result["limitations"], list) or not set(result["limitations"]) <= LIMITATIONS:
            raise ValueError("Invalid limitations")
        if row["eligibility"] == "excluded":
            if result["status"] != "ok" or score != 1 or result["limitations"] != ["unsupported_domain"]:
                raise ValueError("Excluded content requires both policy floors")
        elif "unsupported_domain" in result["limitations"]:
            raise ValueError("Nonexcluded reference cannot assert unsupported domain")


def complete(packet: Path, case_id: str, timing_file: Path, output: Path, assessment: dict) -> dict:
    completed_at, completed_monotonic = utc(), time.monotonic()
    case = packet_case(packet, case_id)
    timing = json.loads(timing_file.read_text())
    if (
        timing["case_id"] != case_id
        or timing["view_sha256"] != case["view_sha256"]
        or timing["packet_sha256"] != digest(packet)
    ):
        raise ValueError("Review timing belongs to different input")
    if digest(Path(case["image_path"])) != case["view_sha256"]:
        raise ValueError("Review image changed during assessment")
    if set(assessment) != {"eligibility", "eligibility_evidence", *TASKS}:
        raise ValueError("Unexpected assessment fields")
    validate_assessment(assessment)
    elapsed = completed_monotonic - timing["monotonic_started"]
    if elapsed < 0:
        raise ValueError("Monotonic clock reset; cannot report duration")
    row = {
        "case_id": case_id,
        "view_sha256": case["view_sha256"],
        **assessment,
        "review_started_at": timing["review_started_at"],
        "review_completed_at": completed_at,
        "review_elapsed_seconds": round(elapsed, 6),
        "review_timing_path": str(timing_file.resolve()),
        "review_timing_sha256": digest(timing_file),
        "review_completed_monotonic": completed_monotonic,
        "review_timing_basis": "Image-open request preparation through structured assessment; includes tool overhead",
        "latency_seconds": None,
        "latency_kind": "not_applicable_assistant_review",
        "price_usd": 0.0,
        "price_basis": "user_assigned_zero",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if any(item["case_id"] == case_id for item in previous):
        raise ValueError("Refusing to overwrite a saved reference")
    with output.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row


def validate_timing(row: dict, expected_packet_sha256: str | None = None) -> None:
    if row["latency_seconds"] is not None or row["latency_kind"] != "not_applicable_assistant_review":
        raise ValueError("Assistant review has no HTTP latency")
    elapsed = row["review_elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("Review duration must be measured and nonnegative")
    timing_path = Path(row["review_timing_path"])
    if digest(timing_path) != row["review_timing_sha256"]:
        raise ValueError("Review timing record changed")
    timing = json.loads(timing_path.read_text())
    if expected_packet_sha256 is not None and timing["packet_sha256"] != expected_packet_sha256:
        raise ValueError("Review timing belongs to a different frozen packet")
    if (
        timing["case_id"] != row["case_id"]
        or timing["view_sha256"] != row["view_sha256"]
        or timing["review_started_at"] != row["review_started_at"]
    ):
        raise ValueError("Review timing belongs to different image")
    measured = row["review_completed_monotonic"] - timing["monotonic_started"]
    if not math.isfinite(measured) or not math.isclose(measured, elapsed, abs_tol=1e-5):
        raise ValueError("Review duration disagrees with monotonic record")
    start, end = (datetime.fromisoformat(row[key]) for key in ("review_started_at", "review_completed_at"))
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError("Review timestamps must be ordered and timezone-aware")
    if end > datetime.now(UTC):
        raise ValueError("Review completion cannot be in the future")


def make_packets(manifest: Path, packet_size: int) -> dict:
    cases = [
        case
        for case in json.loads(manifest.read_text())["cases"]
        if case.get("benchmark_partition") in BENCHMARK_PARTITIONS
    ]
    if (
        len(cases) != 500
        or len({case["family_id"] for case in cases}) != 500
        or len({case["case_id"] for case in cases}) != 500
    ):
        raise ValueError("Expected 500 distinct benchmark families")
    expected = {"development": 100, "representative": 300, "challenge": 100}
    actual = {part: sum(case["benchmark_partition"] == part for case in cases) for part in expected}
    if actual != expected:
        raise ValueError(f"Unexpected partitions: {actual}")
    ordered = sorted(
        cases, key=lambda case: hashlib.sha256(("reference-20261101:" + case["case_id"]).encode()).hexdigest()
    )
    packets = []
    for offset in range(0, len(ordered), packet_size):
        destination = HERE / "references" / f"packet-{offset // packet_size + 1:02}.json"
        payload = {
            "cases": [
                {key: case[key] for key in ("case_id", "image_path", "view_sha256")}
                for case in ordered[offset : offset + packet_size]
            ]
        }
        write_new(destination, payload)
        packets.append(
            {"path": str(destination), "sha256": digest(destination), "count": len(payload["cases"])}
        )
    record = {
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": digest(manifest),
        "created_at": utc(),
        "partitions": actual,
        "packets": packets,
        "rubric_sha256": digest(HERE / "references/rubric.txt"),
        "instructions_sha256": digest(HERE / "references/review-instructions.md"),
        "case_ids": sorted(case["case_id"] for case in cases),
        "review_case_contract": {
            case["case_id"]: {
                key: case[key]
                for key in (
                    "image_path",
                    "view_sha256",
                    "family_id",
                    "benchmark_partition",
                )
            }
            for case in cases
        },
    }
    write_new(HERE / "references/packet-index.json", record)
    return record


def freeze(manifest: Path) -> dict:
    index = json.loads((HERE / "references/packet-index.json").read_text())
    if digest(Path(index["manifest_path"])) != index["manifest_sha256"]:
        raise ValueError("Review input manifest changed after anonymous packets")
    if (
        digest(HERE / "references/rubric.txt") != index["rubric_sha256"]
        or digest(HERE / "references/review-instructions.md") != index["instructions_sha256"]
    ):
        raise ValueError("Review instructions changed after packet creation")
    packet_by_case = {}
    for packet in index["packets"]:
        if digest(Path(packet["path"])) != packet["sha256"]:
            raise ValueError("Anonymous packet changed")
        for case in json.loads(Path(packet["path"]).read_text())["cases"]:
            if case["case_id"] in packet_by_case:
                raise ValueError("Case appears in multiple anonymous packets")
            packet_by_case[case["case_id"]] = packet["sha256"]
    rows = []
    review_files = sorted((HERE / "references").glob("review-*.jsonl"))
    for path in review_files:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected = {
        case["case_id"]: case
        for case in json.loads(manifest.read_text())["cases"]
        if case.get("benchmark_partition") in BENCHMARK_PARTITIONS
    }
    contract = {
        identity: {
            key: case[key] for key in ("image_path", "view_sha256", "family_id", "benchmark_partition")
        }
        for identity, case in expected.items()
    }
    if contract != index["review_case_contract"]:
        raise ValueError("Final corpus changed a frozen benchmark case")
    if len(rows) != 500 or {row["case_id"] for row in rows} != set(expected):
        raise ValueError("Reference coverage must be exactly 500 unique cases")
    for row in rows:
        validate_assessment(row)
        validate_timing(row, packet_by_case[row["case_id"]])
        if row["view_sha256"] != expected[row["case_id"]]["view_sha256"]:
            raise ValueError("Reference image identity mismatch")
        if row["price_usd"] != 0 or row["latency_seconds"] is not None:
            raise ValueError("Reference price/latency contract mismatch")
    destination = HERE / "references.jsonl"
    with destination.open("x") as stream:
        for row in sorted(rows, key=lambda row: row["case_id"]):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    record = {
        "references_path": str(destination),
        "references_sha256": digest(destination),
        "frozen_at": utc(),
        "case_ids": sorted(expected),
        "manifest_sha256": digest(manifest),
        "review_input_manifest_sha256": index["manifest_sha256"],
        "rubric_sha256": digest(HERE / "references/rubric.txt"),
        "rubric_path": str(HERE / "references/rubric.txt"),
        "teacher_prompt_sha256": digest(HERE / "api-teacher-prompt.json"),
        "instructions_sha256": digest(HERE / "references/review-instructions.md"),
        "instructions_path": str(HERE / "references/review-instructions.md"),
        "protocol_sha256": digest(HERE / "protocol.md"),
        "protocol_path": str(HERE / "protocol.md"),
        "review_files": {str(path): digest(path) for path in review_files},
        "reference_type": "independent_assistant_visual_assessment_not_human_gold",
    }
    write_new(HERE / "references-freeze.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packets", "freeze"))
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--packet-size", type=int, default=25)
    args = parser.parse_args()
    if args.packet_size < 1:
        parser.error("packet size must be positive")
    result = (
        make_packets(args.manifest, args.packet_size) if args.command == "packets" else freeze(args.manifest)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
