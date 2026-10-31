"""Validate the nine blind packets and freeze references before image requests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/runinfra-string-trial-2026-10-26"
DEFECTS = {"blur", "noise", "compression", "exposure", "obstruction", "contrast", "other", "none"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    references_path = HERE / "references.jsonl"
    freeze_path = HERE / "references-freeze.json"
    if references_path.exists() or freeze_path.exists():
        raise ValueError("References already frozen; do not overwrite")
    if (DATA / "images/run.json").exists() or (DATA / "images/attempts").exists():
        raise ValueError("Reference freeze must precede the image run")
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records, reviewer_files, packets = {}, {}, {}
    for number in range(1, 10):
        packet_path = DATA / f"review-packets/assignment-{number:02}.json"
        review_path = HERE / f"references/review-{number:02}.jsonl"
        packet = json.loads(packet_path.read_text())
        reviews = [json.loads(line) for line in review_path.read_text().splitlines() if line.strip()]
        if [r["case_id"] for r in reviews] != [r["case_id"] for r in packet]:
            raise ValueError(f"Packet coverage or order differs: {number}")
        for row, image in zip(reviews, packet, strict=True):
            if set(image) != {"case_id", "path", "view_sha256", "width", "height"}:
                raise ValueError("Review packet contains unexpected metadata")
            if sha(Path(image["path"])) != image["view_sha256"]:
                raise ValueError("Reviewed image bytes differ from packet")
            if row["case_id"] in records or row["view_sha256"] != image["view_sha256"]:
                raise ValueError("Duplicate case or mismatched image")
            if row["reviewer_id"] != f"blind_review_{number:02}":
                raise ValueError("Incorrect reviewer identifier")
            evidence = row["evidence"]
            if not isinstance(evidence, str) or not 1 <= len(evidence.split()) <= 40:
                raise ValueError("Reference evidence must contain 1–40 words")
            if not isinstance(row["defects"], list) or not row["defects"] or set(row["defects"]) - DEFECTS:
                raise ValueError("Invalid defect categories")
            score, interval = row["score"], row["interval"]
            if score is None:
                if interval is not None or not row.get("unscorable_reason"):
                    raise ValueError("Null reference requires a reason")
            else:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError("Reference interval requires two endpoints")
                for value in (score, *interval):
                    if type(value) not in {int, float}:
                        raise ValueError("Reference scores must be numeric")
                    value = Decimal(str(value))
                    if not value.is_finite() or not 1 <= value <= 5 or value % Decimal("0.1"):
                        raise ValueError("Reference scores must use tenths from 1 to 5")
                if not interval[0] <= score <= interval[1]:
                    raise ValueError("Reference interval must contain the point score")
            records[row["case_id"]] = row
        reviewer_files[str(review_path)] = sha(review_path)
        packets[str(packet_path)] = sha(packet_path)
    if len(records) != 200 or set(records) != {case["case_id"] for case in manifest["cases"]}:
        raise ValueError("Expected exactly 200 cohort references")
    for case in manifest["cases"]:
        if records[case["case_id"]]["view_sha256"] != case["view_sha256"]:
            raise ValueError("Reference differs from manifest")
    references_path.write_text(
        "".join(json.dumps(records[case["case_id"]], allow_nan=False) + "\n" for case in manifest["cases"])
    )
    freeze = {
        "schema_version": "runinfra-string-trial-references-freeze-v1",
        "references_sha256": sha(references_path),
        "manifest_sha256": sha(manifest_path),
        "count": 200,
        "human_gold": False,
        "reference_kind": "frozen_assistant_judgments",
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_files": reviewer_files,
        "assignment_packets": packets,
        "review_instructions_sha256": sha(HERE / "review-instructions.md"),
        "freeze_script_sha256": sha(Path(__file__)),
        "reference_design": "One assistant judgment per image, across nine separate review contexts; no human ratings.",
        "reviewer_context_disclosure": {
            "blind_review_03": "Previously implemented the analysis code; confirmed exposure only to aggregate counts and generic schemas, not actual case-linked metadata, images, outputs, or reviews.",
            "other_reviewers": "Started in fresh contexts with only instructions, anonymous packet, and image paths.",
        },
        "blinding": {
            "provider_image_requests_before_freeze": 0,
            "reviewers_received_provider_outputs": False,
        },
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frozen": len(records), "references_sha256": freeze["references_sha256"]}))


if __name__ == "__main__":
    main()
