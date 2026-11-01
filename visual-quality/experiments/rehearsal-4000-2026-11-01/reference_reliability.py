"""Select a blind repeat panel after primary freeze and summarize its consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import reference_review as review

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference-reliability"
QUOTAS = {"development": 10, "representative": 30, "challenge": 10}


def select(cases: list[dict]) -> list[dict]:
    chosen = []
    for partition, count in QUOTAS.items():
        pool = [case for case in cases if case["benchmark_partition"] == partition]
        pool.sort(
            key=lambda case: hashlib.sha256(
                ("reference-repeat-20261101:" + case["case_id"]).encode()
            ).hexdigest()
        )
        if len(pool) < count:
            raise ValueError("Insufficient cases for repeat panel")
        chosen.extend(pool[:count])
    return sorted(
        chosen, key=lambda case: hashlib.sha256(("repeat-packet:" + case["case_id"]).encode()).hexdigest()
    )


def packets() -> dict:
    frozen_path = HERE / "references-freeze.json"
    frozen = json.loads(frozen_path.read_text())
    primary = Path(frozen["references_path"])
    if review.digest(primary) != frozen["references_sha256"]:
        raise ValueError("Primary references changed after freeze")
    manifest = HERE / "manifest.json"
    if review.digest(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Manifest differs from frozen reference corpus")
    cases = json.loads(manifest.read_text())["cases"]
    selected = select(cases)
    if len(selected) != 50 or len({case["case_id"] for case in selected}) != 50:
        raise ValueError("Repeat selection must contain 50 distinct images")
    items = []
    for start in (0, 25):
        path = ROOT / f"packet-{start // 25 + 1:02d}.json"
        review.write_new(
            path,
            {
                "cases": [
                    {key: case[key] for key in ("case_id", "image_path", "view_sha256")}
                    for case in selected[start : start + 25]
                ]
            },
        )
        items.append({"path": str(path), "sha256": review.digest(path)})
    record = {
        "created_at": review.utc(),
        "selection": "Fixed hash ordering within partition; no score information",
        "quotas": QUOTAS,
        "case_ids": [case["case_id"] for case in selected],
        "packets": items,
        "references_path": str(primary),
        "references_sha256": review.digest(primary),
        "freeze_sha256": review.digest(frozen_path),
        "manifest_sha256": review.digest(manifest),
        "rubric_sha256": review.digest(HERE / "references/rubric.txt"),
        "instructions_sha256": review.digest(HERE / "references/review-instructions.md"),
        "script_sha256": review.digest(Path(__file__)),
    }
    review.write_new(ROOT / "index.json", record)
    return {"images": 50, "packets": len(items), "output": str(ROOT)}


def summarize(primary: list[dict], repeated: list[dict], cases: list[dict]) -> dict:
    first = {row["case_id"]: row for row in primary}
    metadata = {case["case_id"]: case for case in cases}
    panels = {}
    for partition in ("all", *QUOTAS):
        rows = [
            row
            for row in repeated
            if partition == "all" or metadata[row["case_id"]]["benchmark_partition"] == partition
        ]
        result = {
            "n": len(rows),
            "eligibility_agreements": sum(
                row["eligibility"] == first[row["case_id"]]["eligibility"] for row in rows
            ),
            "floor_decision_disagreements": sum(
                (row["eligibility"] == "excluded") != (first[row["case_id"]]["eligibility"] == "excluded")
                for row in rows
            ),
        }
        for task in review.TASKS:
            deltas = [
                row[task]["score"] - first[row["case_id"]][task]["score"]
                for row in rows
                if row[task]["status"] == "ok" and first[row["case_id"]][task]["status"] == "ok"
            ]
            result[task] = {
                "paired_n": len(deltas),
                "mean_absolute_difference": float(np.mean(np.abs(deltas))) if deltas else None,
                "repeat_minus_primary_bias": float(np.mean(deltas)) if deltas else None,
                "differences_at_least_0_75": sum(abs(value) >= 0.75 for value in deltas),
                "absolute_difference_p50_p90": np.quantile(np.abs(deltas), [0.5, 0.9]).tolist()
                if deltas
                else None,
            }
        panels[partition] = result
    return {
        "interpretation": "Assistant repeatability only; primary labels unchanged; not human ground truth",
        "panels": panels,
    }


def analyze() -> dict:
    index = json.loads((ROOT / "index.json").read_text())
    for path, key in (
        (HERE / "manifest.json", "manifest_sha256"),
        (HERE / "references-freeze.json", "freeze_sha256"),
        (Path(index["references_path"]), "references_sha256"),
        (HERE / "references/rubric.txt", "rubric_sha256"),
        (HERE / "references/review-instructions.md", "instructions_sha256"),
        (Path(__file__), "script_sha256"),
    ):
        if review.digest(path) != index[key]:
            raise ValueError("Changed reliability audit input: " + str(path))
    packet_hashes, images = {}, {}
    for packet in index["packets"]:
        path = Path(packet["path"])
        if review.digest(path) != packet["sha256"]:
            raise ValueError("Changed repeat packet")
        for case in json.loads(path.read_text())["cases"]:
            packet_hashes[case["case_id"]] = packet["sha256"]
            images[case["case_id"]] = case
    files = sorted(ROOT.glob("review-*.jsonl"))
    repeated = [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]
    if len(repeated) != 50 or {row["case_id"] for row in repeated} != set(index["case_ids"]):
        raise ValueError("Need exactly 50 unique completed blind repeat assessments")
    for row in repeated:
        review.validate_assessment(row)
        review.validate_timing(row, packet_hashes[row["case_id"]])
        case = images[row["case_id"]]
        if (
            row["view_sha256"] != case["view_sha256"]
            or review.digest(Path(case["image_path"])) != case["view_sha256"]
            or row["price_usd"] != 0
        ):
            raise ValueError("Repeat image identity or zero-price contract changed")
    primary = [json.loads(line) for line in Path(index["references_path"]).read_text().splitlines()]
    result = summarize(primary, repeated, json.loads((HERE / "manifest.json").read_text())["cases"])
    result.update(
        index_sha256=review.digest(ROOT / "index.json"),
        review_files={str(path): review.digest(path) for path in files},
    )
    review.write_new(ROOT / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packets", "analyze"))
    args = parser.parse_args()
    print(json.dumps(packets() if args.command == "packets" else analyze(), indent=2))
