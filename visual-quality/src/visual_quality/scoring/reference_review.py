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

from .history import rehearsal_root

HERE = rehearsal_root()
TASKS = ("technical_quality", "aesthetics")
LIMITATIONS = {"limited_detail", "partial_view", "ambiguous_style", "unsupported_domain"}
BENCHMARK_PARTITIONS = {"development", "representative", "challenge"}
CASE_FIELDS = ("image_path", "view_sha256", "family_id", "benchmark_partition")
REVISION_FIELDS = ("candidate_id", "image_id", *CASE_FIELDS, "benchmark_stratum")


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


def benchmark_cases(manifest: Path) -> dict:
    cases = [
        case
        for case in json.loads(manifest.read_text())["cases"]
        if case.get("benchmark_partition") in BENCHMARK_PARTITIONS
    ]
    if (
        len(cases) != 500
        or len({case["case_id"] for case in cases}) != 500
        or len({case["family_id"] for case in cases}) != 500
        or len({case["view_sha256"] for case in cases}) != 500
        or {part: sum(c["benchmark_partition"] == part for c in cases) for part in BENCHMARK_PARTITIONS}
        != {"development": 100, "representative": 300, "challenge": 100}
    ):
        raise ValueError("Expected 500 distinct benchmark cases and unchanged 100/300/100 partitions")
    return {case["case_id"]: case for case in cases}


def case_contract(cases: dict, fields: tuple = CASE_FIELDS) -> dict:
    return {identity: {key: case[key] for key in fields} for identity, case in cases.items()}


def original_review_inputs() -> tuple[dict, dict, dict]:
    index = json.loads((HERE / "references/packet-index.json").read_text())
    if digest(Path(index["manifest_path"])) != index["manifest_sha256"]:
        raise ValueError("Review input manifest changed after anonymous packets")
    if (
        digest(HERE / "references/rubric.txt") != index["rubric_sha256"]
        or digest(HERE / "references/review-instructions.md") != index["instructions_sha256"]
    ):
        raise ValueError("Review instructions changed after packet creation")
    cases = benchmark_cases(Path(index["manifest_path"]))
    if case_contract(cases) != index["review_case_contract"] or sorted(cases) != index["case_ids"]:
        raise ValueError("Original packet index disagrees with its unchanged manifest")
    packet_by_case = {}
    for packet in index["packets"]:
        if digest(Path(packet["path"])) != packet["sha256"]:
            raise ValueError("Anonymous packet changed")
        payload = json.loads(Path(packet["path"]).read_text())
        if set(payload) != {"cases"} or len(payload["cases"]) != packet["count"]:
            raise ValueError("Invalid anonymous packet envelope")
        for case in payload["cases"]:
            if case["case_id"] in packet_by_case:
                raise ValueError("Case appears in multiple anonymous packets")
            expected = cases.get(case["case_id"], {})
            if case != {key: expected.get(key) for key in ("case_id", "image_path", "view_sha256")}:
                raise ValueError("Anonymous packet disagrees with original image contract")
            packet_by_case[case["case_id"]] = packet
    if set(packet_by_case) != set(cases):
        raise ValueError("Original anonymous packets do not cover all 500 cases")
    return index, cases, packet_by_case


def revision_inputs(path: Path, index: dict, original: dict) -> tuple[dict, dict, dict]:
    declaration = json.loads(path.read_text())
    if declaration["schema_version"] != 1:
        raise ValueError("Unsupported benchmark revision schema")
    old, active = declaration["original_manifest"], declaration["active_manifest"]
    if (
        Path(old["path"]).resolve() != Path(index["manifest_path"]).resolve()
        or old["sha256"] != index["manifest_sha256"]
        or digest(Path(old["path"])) != old["sha256"]
        or digest(Path(active["path"])) != active["sha256"]
    ):
        raise ValueError("Benchmark revision manifest identity changed")
    revised = benchmark_cases(Path(active["path"]))
    entries = declaration["revisions"]
    revisions = {entry["case_id"]: entry for entry in entries}
    if not entries or len(entries) != len(revisions) or set(revised) != set(original):
        raise ValueError("Revision cases must be unique and preserve all benchmark identities")
    old_contract, new_contract = (case_contract(cases, REVISION_FIELDS) for cases in (original, revised))
    changed = {identity for identity in original if old_contract[identity] != new_contract[identity]}
    if changed != set(revisions):
        raise ValueError("Benchmark changes differ from explicitly declared revisions")
    for identity, entry in revisions.items():
        previous, replacement = entry["previous"], entry["replacement"]
        evidence = entry["visual_review"]
        excluded = set(entry["excluded_from_final_candidates"])
        if (
            previous != old_contract[identity]
            or replacement != new_contract[identity]
            or entry["reason_code"] != "confirmed_historical_near_duplicate"
            or not entry["selection"]
            or not evidence["decision"]
            or digest(Path(evidence["path"])) != evidence["artifact_sha256"]
            or not {previous["candidate_id"], entry["historical_peer"]["candidate_id"]} <= excluded
            or replacement["candidate_id"] in excluded
            or replacement["view_sha256"] == previous["view_sha256"]
            or replacement["family_id"] == previous["family_id"]
            or any(
                previous[field] != replacement[field]
                for field in ("benchmark_partition", "benchmark_stratum")
            )
        ):
            raise ValueError("Revision lacks an objective, matching replacement contract and provenance")
    return declaration, revised, revisions


def make_revision_packets(revisions_path: Path) -> dict:
    """Create a separate blind packet without parsing any existing reference labels."""
    index, original, _ = original_review_inputs()
    _, revised, revisions = revision_inputs(revisions_path, index, original)
    directory = HERE / "references/revisions"
    destination = directory / "packet-index.json"
    if directory.exists():
        raise ValueError("Revision review directory already exists; preserve its immutable artifacts")
    directory.mkdir(parents=True)
    instructions = directory / "review-instructions.md"
    with instructions.open("xb") as stream:
        stream.write((HERE / "references/review-instructions.md").read_bytes())
    rubric = directory / "rubric.txt"
    with rubric.open("xb") as stream:
        stream.write((HERE / "references/rubric.txt").read_bytes())
    packets = []
    for position, identity in enumerate(sorted(revisions), 1):
        case = revised[identity]
        if digest(Path(case["image_path"])) != case["view_sha256"]:
            raise ValueError("Replacement review image changed")
        path = directory / f"packet-{position:02}.json"
        payload = {"cases": [{key: case[key] for key in ("case_id", "image_path", "view_sha256")}]}
        write_new(path, payload)
        packets.append(
            {
                "path": str(path),
                "sha256": digest(path),
                "case_id": identity,
                "review_path": str(directory / f"review-{position:02}.jsonl"),
                "timing_path": str(directory / "timings" / f"{identity}.json"),
            }
        )
    record = {
        "schema_version": 1,
        "created_at": utc(),
        "revisions_path": str(revisions_path.resolve()),
        "revisions_sha256": digest(revisions_path),
        "original_packet_index_sha256": digest(HERE / "references/packet-index.json"),
        "original_review_files": {
            str(path): digest(path) for path in sorted((HERE / "references").glob("review-*.jsonl"))
        },
        "rubric_sha256": index["rubric_sha256"],
        "rubric_path": str(rubric),
        "instructions_path": str(instructions),
        "instructions_sha256": digest(instructions),
        "review_case_contract": case_contract({identity: revised[identity] for identity in revisions}),
        "packets": packets,
    }
    write_new(destination, record)
    return record


def load_reviews(files: list[Path], cases: dict, packets: dict) -> tuple[dict, dict]:
    rows, sources = {}, {}
    for path in files:
        file_sha = digest(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row["case_id"]
            if identity not in cases or identity in rows:
                raise ValueError("Reference coverage contains duplicate or undeclared cases")
            validate_assessment(row)
            validate_timing(row, packets[identity]["sha256"])
            if row["view_sha256"] != cases[identity]["view_sha256"]:
                raise ValueError("Reference image identity mismatch")
            if row["price_usd"] != 0 or row["latency_seconds"] is not None:
                raise ValueError("Reference price/latency contract mismatch")
            rows[identity] = row
            sources[identity] = {
                "review_path": str(path),
                "review_sha256": file_sha,
                "view_sha256": row["view_sha256"],
                "packet_path": packets[identity]["path"],
                "packet_sha256": packets[identity]["sha256"],
                "timing_path": row["review_timing_path"],
                "timing_sha256": row["review_timing_sha256"],
                "row_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            }
    if set(rows) != set(cases):
        raise ValueError("Reference coverage must contain every expected case exactly once")
    return rows, sources


def replacement_reviews(path: Path, index: dict, original: dict) -> tuple[dict, dict, dict, dict]:
    declaration, revised, revisions = revision_inputs(path, index, original)
    packet_index_path = HERE / "references/revisions/packet-index.json"
    packet_index = json.loads(packet_index_path.read_text())
    original_files = {str(p): digest(p) for p in sorted((HERE / "references").glob("review-*.jsonl"))}
    if (
        packet_index["schema_version"] != 1
        or packet_index["revisions_sha256"] != digest(path)
        or Path(packet_index["revisions_path"]).resolve() != path.resolve()
        or packet_index["original_packet_index_sha256"] != digest(HERE / "references/packet-index.json")
        or packet_index["original_review_files"] != original_files
        or packet_index["rubric_sha256"] != index["rubric_sha256"]
        or digest(Path(packet_index["rubric_path"])) != index["rubric_sha256"]
        or packet_index["instructions_sha256"] != index["instructions_sha256"]
        or digest(Path(packet_index["instructions_path"])) != index["instructions_sha256"]
        or packet_index["review_case_contract"] != case_contract({k: revised[k] for k in revisions})
    ):
        raise ValueError("Frozen revision packet metadata or original reviews changed")
    packets = {packet["case_id"]: packet for packet in packet_index["packets"]}
    if set(packets) != set(revisions) or len(packets) != len(packet_index["packets"]):
        raise ValueError("Revision packets must cover each declared replacement once")
    files = [Path(packet["review_path"]) for packet in packets.values()]
    if len(set(files)) != len(files) or set(files) != set(packet_index_path.parent.glob("review-*.jsonl")):
        raise ValueError("Missing or undeclared replacement review files")
    for identity, packet in packets.items():
        expected = {key: revised[identity][key] for key in ("case_id", "image_path", "view_sha256")}
        if digest(Path(packet["path"])) != packet["sha256"] or json.loads(
            Path(packet["path"]).read_text()
        ) != {"cases": [expected]}:
            raise ValueError("Replacement anonymous packet changed or exposes undeclared fields")
    rows, sources = load_reviews(files, {k: revised[k] for k in revisions}, packets)
    if any(
        Path(row["review_timing_path"]).resolve() != Path(packets[identity]["timing_path"]).resolve()
        for identity, row in rows.items()
    ):
        raise ValueError("Replacement review must use its separate revision timing record")
    created = datetime.fromisoformat(packet_index["created_at"])
    if created.tzinfo is None or created > datetime.now(UTC):
        raise ValueError("Invalid revision packet creation time")
    if any(datetime.fromisoformat(row["review_started_at"]) < created for row in rows.values()):
        raise ValueError("Replacement requires a fresh review after its revision packet freeze")
    provenance = {
        "declaration_path": str(path.resolve()),
        "declaration_sha256": digest(path),
        "packet_index_path": str(packet_index_path),
        "packet_index_sha256": digest(packet_index_path),
        "declaration": declaration,
    }
    return revised, rows, sources, provenance


def freeze(manifest: Path, revisions: Path | None = None) -> dict:
    index, original, packet_by_case = original_review_inputs()
    expected = benchmark_cases(manifest)
    review_files = sorted((HERE / "references").glob("review-*.jsonl"))
    rows, sources = load_reviews(review_files, original, packet_by_case)
    revision_record, superseded = None, {}
    if revisions is None:
        if case_contract(expected) != index["review_case_contract"]:
            raise ValueError("Final corpus changed a frozen benchmark case")
    else:
        revised, replacements, replacement_sources, revision_record = replacement_reviews(
            revisions, index, original
        )
        if case_contract(expected, REVISION_FIELDS) != case_contract(revised, REVISION_FIELDS):
            raise ValueError("Final corpus differs from the explicitly revised benchmark contract")
        superseded = {identity: sources[identity] for identity in replacements}
        rows.update(replacements)
        sources.update(replacement_sources)
        review_files += [Path(value["review_path"]) for value in replacement_sources.values()]
    destination = HERE / "references.jsonl"
    if destination.exists() or (HERE / "references-freeze.json").exists():
        raise ValueError("References already frozen; preserve the existing artifacts")
    with destination.open("x") as stream:
        for row in (rows[identity] for identity in sorted(rows)):
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
        "review_case_contract": case_contract(expected),
        "original_review_case_contract": index["review_case_contract"],
        "original_packet_index_path": str(HERE / "references/packet-index.json"),
        "original_packet_index_sha256": digest(HERE / "references/packet-index.json"),
        "review_sources": sources,
        "superseded_reviews": superseded,
        "reference_revision": revision_record,
        "reference_type": "independent_assistant_visual_assessment_not_human_gold",
    }
    write_new(HERE / "references-freeze.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packets", "revision-packets", "freeze"))
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--packet-size", type=int, default=25)
    parser.add_argument("--revisions", type=Path, help="Explicit objective benchmark revision declaration")
    args = parser.parse_args()
    if args.packet_size < 1:
        parser.error("packet size must be positive")
    if args.command == "revision-packets":
        if args.revisions is None:
            parser.error("revision-packets requires --revisions")
        result = make_revision_packets(args.revisions)
    elif args.command == "packets":
        if args.revisions is not None:
            parser.error("Use revision-packets for an explicitly revised benchmark")
        result = make_packets(args.manifest, args.packet_size)
    else:
        result = freeze(args.manifest, revisions=args.revisions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
