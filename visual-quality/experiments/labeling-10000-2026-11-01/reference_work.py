"""Blind reference packets and immutable per-packet freezes for the expanded benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "rehearsal-4000-2026-11-01"
SPEC = importlib.util.spec_from_file_location("prior_reference_review", PRIOR / "reference_review.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def prepare(manifest: Path, output: Path, packet_size: int, expected_count: int = 1000) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    old_refs = {json.loads(line)["case_id"] for line in (PRIOR / "references.jsonl").read_text().splitlines()}
    selected = [c for c in cases if c.get("benchmark_order") and c["case_id"] not in old_refs]
    selected.sort(key=lambda c: c["benchmark_order"])
    if len(selected) != expected_count or len({c["family_id"] for c in selected}) != expected_count:
        raise ValueError(f"Expected {expected_count} new benchmark families")
    if len({c["view_sha256"] for c in selected}) != expected_count:
        raise ValueError("Benchmark contains identical delivered views")
    output.mkdir(parents=True, exist_ok=True)
    packets = []
    for offset in range(0, len(selected), packet_size):
        path = output / f"packet-{offset // packet_size + 1:03}.json"
        payload = {
            "cases": [
                {k: case[k] for k in ("case_id", "image_path", "view_sha256")}
                for case in selected[offset : offset + packet_size]
            ],
            "rubric": "visual-rating-v2 technical; independent aesthetics; photography eligibility override",
            "source_manifest_sha256": review.digest(manifest),
            "blind": "No model scores, stats, cluster labels or source defect votes supplied",
        }
        review.write_new(path, payload)
        packets.append(
            {"path": str(path.resolve()), "sha256": review.digest(path), "count": len(payload["cases"])}
        )
    index = {"created_at": datetime.now(UTC).isoformat(), "cases": len(selected), "packets": packets}
    review.write_new(output / "index.json", index)
    return index


def freeze(packet: Path, records: Path, destination: Path) -> dict:
    expected = {c["case_id"]: c for c in json.loads(packet.read_text())["cases"]}
    rows = [json.loads(line) for line in records.read_text().splitlines() if line]
    if len(rows) != len(expected) or {r["case_id"] for r in rows} != set(expected):
        raise ValueError("Reference packet is incomplete or contains duplicate/unexpected cases")
    for row in rows:
        review.validate_assessment(row)
        review.validate_timing(row, review.digest(packet))
        if row["view_sha256"] != expected[row["case_id"]]["view_sha256"]:
            raise ValueError("Reference view differs from assigned packet")
    receipt = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "references_path": str(records.resolve()),
        "references_sha256": review.digest(records),
        "packet_path": str(packet.resolve()),
        "packet_sha256": review.digest(packet),
        "case_ids": sorted(expected),
        "reference_kind": "independent_assistant_visual_judgments_not_human_ratings",
    }
    review.write_new(destination, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--packet-size", type=int, default=50)
    prep.add_argument("--expected-count", type=int, choices=(500, 1000), default=1000)
    for name in ("begin", "complete"):
        action = commands.add_parser(name)
        action.add_argument("--packet", type=Path, required=True)
        action.add_argument("--case-id", required=True)
        action.add_argument("--timing", type=Path, required=True)
        if name == "complete":
            action.add_argument("--output", type=Path, required=True)
    finish = commands.add_parser("freeze")
    finish.add_argument("--packet", type=Path, required=True)
    finish.add_argument("--records", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.output, args.packet_size, args.expected_count)
    elif args.command == "begin":
        result = review.begin(args.packet, args.case_id, args.timing)
    elif args.command == "complete":
        result = review.complete(args.packet, args.case_id, args.timing, args.output, json.load(sys.stdin))
    else:
        result = freeze(args.packet, args.records, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
