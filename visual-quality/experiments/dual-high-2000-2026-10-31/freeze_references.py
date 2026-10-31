"""Validate and freeze the 72 blinded assistant judgments without model outputs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from visual_quality.labels import _validate_rating
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / HERE.name
TASKS = ("technical_quality", "aesthetics")


def main():
    if (HERE / "references-freeze.json").exists():
        raise ValueError("References are already frozen")
    selection = json.loads((HERE / "collage-selection.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    selected = set(selection["case_ids"])
    references, pins = {}, {}
    for number in (1, 2):
        packet_path = DATA / "review-packets" / f"packet-{number}.json"
        review_path = HERE / "reference-reviews" / f"review-{number}.jsonl"
        packet = {case["case_id"]: case for case in json.loads(packet_path.read_text())}
        rows = read_jsonl(review_path)
        if len(rows) != 36 or {row["case_id"] for row in rows} != set(packet):
            raise ValueError("Review must cover exactly its 36 unique assigned images")
        for row in rows:
            case_id = row["case_id"]
            if case_id in references or case_id not in selected:
                raise ValueError("Duplicate or unselected reference")
            case = cases[case_id]
            if (
                row["view_sha256"] != packet[case_id]["view_sha256"]
                or row["view_sha256"] != case["view_sha256"]
            ):
                raise ValueError("Reference image identity mismatch")
            if file_sha256(Path(packet[case_id]["path"])) != row["view_sha256"]:
                raise ValueError("Reviewed image changed")
            if (
                row["eligibility"] not in {"eligible", "excluded", "uncertain"}
                or not row["eligibility_evidence"].strip()
            ):
                raise ValueError("Missing eligibility judgment")
            for task in TASKS:
                rating = row[task]
                if rating["task"] != task or rating["status"] != "ok":
                    raise ValueError("Missing scored reference task")
                _validate_rating({**rating, "task": "technical_quality"})
                if row["eligibility"] == "excluded" and (
                    rating["score"] != 1 or rating["limitations"] != ["unsupported_domain"]
                ):
                    raise ValueError("Excluded reference must obey both policy floors")
            references[case_id] = {**row, "reviewer": f"blind-review-{number}"}
        pins.update({str(path): file_sha256(path) for path in (packet_path, review_path)})
    if set(references) != selected or len(references) != 72:
        raise ValueError("Incomplete reference subset")
    write_jsonl(HERE / "references.jsonl", [references[key] for key in sorted(references)])
    for path in (
        HERE / "review-instructions.md",
        HERE / "photo-scope.txt",
        HERE / "freeze_references.py",
        HERE.parents[1] / "prompts/visual-rating-v1/joint.txt",
    ):
        pins[str(path)] = file_sha256(path)
    freeze = {
        "schema_version": "dual-high-2000-reference-subset-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "count": 72,
        "human_gold": False,
        "student_outputs_seen_by_reviewers": False,
        "selection_fixed_before_inference": True,
        "reference_kind": "fresh_context_assistant_visual_judgments",
        "references_sha256": file_sha256(HERE / "references.jsonl"),
        "manifest_sha256": file_sha256(HERE / "manifest.json"),
        "selection_sha256": file_sha256(HERE / "collage-selection.json"),
        "eligibility_counts": dict(Counter(row["eligibility"] for row in references.values())),
        "inputs": pins,
    }
    atomic_json(HERE / "references-freeze.json", freeze)
    print(json.dumps({key: value for key, value in freeze.items() if key != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
