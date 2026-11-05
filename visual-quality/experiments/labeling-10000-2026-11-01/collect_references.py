"""Consolidate immutable review packets without changing their judgments."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from reference_work import PRIOR, review

DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")


def collect(data: Path, *, allow_partial: bool = False) -> dict:
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark = {c["case_id"]: c for c in manifest["cases"] if c.get("benchmark_order")}
    old_path = PRIOR / "references.jsonl"
    old_sha = review.digest(old_path)
    rows = {}
    for line in old_path.read_text().splitlines():
        row = json.loads(line)
        rows[row["case_id"]] = {
            **row,
            "reuse": {
                "source": str(old_path),
                "source_sha256": old_sha,
                "kind": "original_500_assistant_reference_judgments_unchanged",
                "rubric_provenance": "rehearsal-4000-2026-11-01; historically exposed cohort",
            },
        }
    if len(rows) != 500:
        raise ValueError("Expected exactly 500 original reference records")
    receipts = []
    for path in sorted((data / "references").glob("*/*/packet-???-freeze.json")):
        freeze = json.loads(path.read_text())
        records = Path(freeze["references_path"])
        packet = Path(freeze["packet_path"])
        if (
            review.digest(records) != freeze["references_sha256"]
            or review.digest(packet) != freeze["packet_sha256"]
        ):
            raise ValueError(f"Frozen review artifact changed: {path}")
        frozen_at = datetime.fromisoformat(freeze["frozen_at"])
        if frozen_at.tzinfo is None or frozen_at > datetime.now(UTC):
            raise ValueError("Invalid reference freeze timestamp")
        expected = {c["case_id"]: c for c in json.loads(packet.read_text())["cases"]}
        batch = [json.loads(line) for line in records.read_text().splitlines()]
        ids = [row["case_id"] for row in batch]
        if len(ids) != len(set(ids)) or set(ids) != set(expected) or set(ids) != set(freeze["case_ids"]):
            raise ValueError("Packet reference membership changed")
        for row in batch:
            identity = row["case_id"]
            if identity in rows:
                raise ValueError(f"Duplicate independent reference: {identity}")
            review.validate_assessment(row)
            review.validate_timing(row, freeze["packet_sha256"])
            if row["view_sha256"] != expected[identity]["view_sha256"]:
                raise ValueError("Reference differs from its assigned image")
            rows[identity] = {
                **row,
                "review_freeze_path": str(path),
                "review_freeze_sha256": review.digest(path),
            }
        receipts.append({"path": str(path), "sha256": review.digest(path), "cases": len(batch)})
    if set(rows) - set(benchmark):
        raise ValueError("Reference outside the frozen benchmark")
    for identity, row in rows.items():
        review.validate_assessment(row)
        if row["view_sha256"] != benchmark[identity]["view_sha256"]:
            raise ValueError(f"Reference/manifest image mismatch: {identity}")
    complete = len(rows) == len(benchmark) == 1500
    if not complete and not allow_partial:
        raise ValueError(f"Only {len(rows)}/1500 references frozen")
    ordered = sorted(rows.values(), key=lambda row: benchmark[row["case_id"]]["benchmark_order"])
    destination = data / ("references-all.jsonl" if complete else "references-partial.jsonl")
    serialized = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in ordered)
    if complete and destination.exists() and destination.read_text() != serialized:
        raise ValueError("Refusing to change the complete reference set")
    destination.write_text(serialized)
    receipt = {
        "references_path": str(destination),
        "references_sha256": review.digest(destination),
        "manifest_sha256": review.digest(manifest_path),
        "case_ids": [row["case_id"] for row in ordered],
        "packets": receipts,
        "reference_kind": "independent_assistant_visual_judgments_not_human_ratings",
        "complete": complete,
        "frozen_at": datetime.now(UTC).isoformat() if complete else None,
    }
    receipt_path = data / ("references-freeze.json" if complete else "references-partial-provenance.json")
    if complete and receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["references_sha256"] != receipt["references_sha256"]:
            raise ValueError("Complete reference freeze changed")
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "complete": complete,
        "references": len(rows),
        "path": str(destination),
        "receipt": str(receipt_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.data, allow_partial=args.allow_partial)))
