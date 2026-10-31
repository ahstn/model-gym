"""Validate blinded evidence checks and decode model/arm identities afterward."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "explanation-audit"
JUDGMENTS = {"supported", "unsupported_specific_claim", "contradicted", "uncertain"}


def main():
    frozen = json.loads((HERE / "explanation-audit-freeze.json").read_text())
    for name, expected in frozen["inputs"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    mapping = json.loads((AUDIT / "mapping.json").read_text())
    rows = []
    for index in range(1, 4):
        packet = json.loads((AUDIT / f"packet-{index}.json").read_text())
        expected = {r["label_id"]: c["case_id"] for c in packet for r in c["labels"]}
        batch = [
            json.loads(x) for x in (AUDIT / f"review-{index}.jsonl").read_text().splitlines() if x.strip()
        ]
        assert len(batch) == len(expected) and {r["label_id"] for r in batch} == set(expected)
        for case in packet:
            assert hashlib.sha256(Path(case["path"]).read_bytes()).hexdigest() == case["view_sha256"]
        for row in batch:
            assert row["judgment"] in JUDGMENTS and row["rationale"].strip()
            assert row["case_id"] == expected[row["label_id"]] == mapping[row["label_id"]]["case_id"]
            rows.append({**row, **mapping[row["label_id"]], "reviewer": index})
    assert len(rows) == len(mapping)
    summaries = []
    for route in ("deepseek", "luna", "gemini", "haiku"):
        for task in ("technical_quality", "aesthetics"):
            arm_rows = {
                arm: {
                    r["case_id"]: r
                    for r in rows
                    if r["route"] == route and r["task"] == task and r["arm_id"] == arm
                }
                for arm in ("baseline", "combined")
            }
            common = set(arm_rows["baseline"]) & set(arm_rows["combined"])
            bad = lambda row: row["judgment"] in {"unsupported_specific_claim", "contradicted"}
            summaries.append(
                {
                    "route": route,
                    "task": task,
                    "counts": {
                        arm: dict(Counter(r["judgment"] for r in rs.values())) for arm, rs in arm_rows.items()
                    },
                    "paired_n": len(common),
                    "new_material_issues": [
                        cid
                        for cid in sorted(common)
                        if bad(arm_rows["combined"][cid]) and not bad(arm_rows["baseline"][cid])
                    ],
                    "resolved_material_issues": [
                        cid
                        for cid in sorted(common)
                        if bad(arm_rows["baseline"][cid]) and not bad(arm_rows["combined"][cid])
                    ],
                }
            )
    result = {
        "images": 12,
        "labels": len(rows),
        "blinding": {"fresh_context_reviewers": 3, "scores_and_model_names_hidden": True},
        "scope": "Exploratory factual-claim audit on a preselected twelve-image subset. One independent reviewer per image, no inter-reviewer reliability estimate. Only baseline and combined; no direct audit of each individual addition.",
        "summary": summaries,
        "rows": rows,
    }
    (AUDIT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (AUDIT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"labels": len(rows), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
