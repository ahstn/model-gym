"""Attach the separate explanation audit to the frozen quantitative report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/runinfra-string-trial-2026-10-26"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = HERE / "results/report.json"
    report = json.loads(source.read_text())
    selection_path = HERE / "audit-selection.json"
    selected = json.loads(selection_path.read_text())["cases"]
    audits, pins = {}, {}
    for number in range(1, 4):
        path = HERE / f"explanation-audit/audit-{number:02}.jsonl"
        packet_path = DATA / f"audit-packets/audit-{number:02}.json"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        packet = json.loads(packet_path.read_text())
        if [r["case_id"] for r in rows] != [r["case_id"] for r in packet]:
            raise ValueError("Explanation audit coverage or order differs")
        for row, item in zip(rows, packet, strict=True):
            if row["case_id"] in audits or row["view_sha256"] != item["view_sha256"]:
                raise ValueError("Audit image identity differs or repeats")
            if row["auditor_id"] != f"explanation_audit_{number:02}":
                raise ValueError("Unexpected auditor")
            if row["verdict"] not in {"supported", "mixed", "unsupported", "uncertain"}:
                raise ValueError("Unexpected audit verdict")
            if not 1 <= len(row["rationale"].split()) <= 60:
                raise ValueError("Audit rationale must contain 1–60 words")
            if not isinstance(row["unsupported_claims"], list):
                raise TypeError("Unsupported claims must be an array")
            audits[row["case_id"]] = row
        pins[str(path)] = sha(path)
        pins[str(packet_path)] = sha(packet_path)
    if set(audits) != {row["case_id"] for row in selected} or len(audits) != 24:
        raise ValueError("Audit differs from the preselected 24 cases")
    for row in selected:
        if audits[row["case_id"]]["view_sha256"] != row["view_sha256"]:
            raise ValueError("Audit differs from preselected view")
    audit = {
        "audited": True,
        "count": 24,
        "verdict_counts": dict(Counter(row["verdict"] for row in audits.values())),
        "sample_selection": "24 cases selected before image requests; 12 controlled, 6 VizWiz, 6 standalone.",
        "scope": "Explanations only; auditors received no scores or references. These counts do not estimate support across all 200 cases.",
        "invalid_label_case_ids_in_audit": sorted(set(audits) & set(report["invalid_case_ids"])),
        "selection_sha256": sha(selection_path),
        "instructions_sha256": sha(HERE / "explanation-audit-instructions.md"),
        "source_files": pins,
        "records": [audits[row["case_id"]] for row in selected],
    }
    rejections = []
    for case_id in report["invalid_case_ids"]:
        raw = json.loads((DATA / f"images/attempts/{case_id}/result.json").read_text())
        content = json.loads(raw["body"])["choices"][0]["message"]["content"]
        rating = json.loads(content)
        rejections.append({"case_id": case_id, "evidence_words": len(rating["evidence"].split())})
    report.update(
        schema_version="runinfra-string-trial-complete-report-v1",
        frozen_quantitative_report_sha256=sha(source),
        explanation_support=audit,
        rejected_evidence_lengths=rejections,
        assembly_script_sha256=sha(Path(__file__)),
    )
    destination = HERE / "results/complete-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"audit_cases": len(audits), "verdicts": audit["verdict_counts"], "rejections": rejections}
        )
    )


if __name__ == "__main__":
    main()
