"""Validate and seal the three blind reviewer packets before any image inference."""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/dual-score-trial-2026-10-27"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "references-freeze.json").exists():
        raise ValueError("References are already frozen")
    if list(DATA.glob("*/images/attempts/*")):
        raise ValueError("Image requests predate reference freeze")
    cases = json.loads((HERE / "manifest.json").read_text())["cases"]
    rows, files = [], {}
    for index in range(1, 4):
        source = HERE / f"references-reviewer-{index}.jsonl"
        files[str(source)] = sha(source)
        packet = json.loads((DATA / f"review-packets/packet-{index}.json").read_text())
        batch = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(batch) != len(packet) or {r["case_id"] for r in batch} != {r["case_id"] for r in packet}:
            raise ValueError("Review packet coverage mismatch")
        rows.extend(batch)
    by_id = {r["case_id"]: r for r in rows}
    if len(rows) != 50 or len(by_id) != 50 or set(by_id) != {c["case_id"] for c in cases}:
        raise ValueError("Expected exactly fifty distinct reviewed cases")
    for case in cases:
        row = by_id[case["case_id"]]
        if row["view_sha256"] != case["view_sha256"] or sha(Path(case["image_path"])) != case["view_sha256"]:
            raise ValueError("Reviewed and delivered bytes differ")
        for task in ("technical_quality", "aesthetics"):
            rating = row[task]
            score, interval = rating["score"], rating["interval"]
            if score is None:
                if interval is not None or not rating.get("unscorable_reason"):
                    raise ValueError("Unscorable reference requires reason")
            elif (
                type(score) not in (float, int)
                or not isinstance(interval, list)
                or len(interval) != 2
                or any(type(x) not in (float, int) for x in interval)
                or not 1 <= interval[0] <= score <= interval[1] <= 5
                or any(Decimal(str(x)) % Decimal("0.1") for x in [score, *interval])
            ):
                raise ValueError("Invalid reference score/interval")
            if not isinstance(rating["evidence"], str) or not rating["evidence"].strip():
                raise ValueError("Evidence required")
    destination = HERE / "references.jsonl"
    destination.write_text("".join(json.dumps(by_id[key], allow_nan=False) + "\n" for key in sorted(by_id)))
    freeze = {
        "created_at": datetime.now(UTC).isoformat(),
        "count": 50,
        "reference_kind": "frozen_assistant_judgments",
        "human_gold": False,
        "dimensions": ["technical_quality", "aesthetics"],
        "references_sha256": sha(destination),
        "manifest_sha256": sha(HERE / "manifest.json"),
        "review_instructions_sha256": sha(HERE / "review-instructions.md"),
        "reviewer_files": files,
        "blinding": {
            "provider_image_requests_before_freeze": 0,
            "reviewers_received_provider_outputs": False,
            "reviewers_received_source_metadata": False,
            "fresh_context_reviewers": 3,
            "each_supplied_image_individually_inspected": True,
        },
    }
    (HERE / "references-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"count": 50, "references_sha256": freeze["references_sha256"]}))


if __name__ == "__main__":
    main()
