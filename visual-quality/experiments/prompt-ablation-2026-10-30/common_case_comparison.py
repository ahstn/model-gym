"""Diagnostic model comparisons on identical valid cases, with no fitted weights."""

import json
from pathlib import Path

import report

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/prompt-ablation-2026-10-30"


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    refs = report.read_rows(HERE / "references.jsonl")
    obs = {route: report.read_rows(DATA / route / "images/observations.jsonl") for route in report.ROUTES}
    rows = report.comparison_rows(manifest["cases"], refs, obs)
    result = {
        "scope": "Exploratory comparisons on the same valid cases. Nominal family-bootstrap intervals; no cross-model multiplicity correction and no fitted ensemble weights.",
        "arms": {},
    }
    for arm in ("baseline", "combined"):
        result["arms"][arm] = {}
        for task in report.TASKS:
            grouped = {
                route: {
                    r["case_id"]: r
                    for r in rows
                    if r["route"] == route and r["arm_id"] == arm and r["task"] == task and "error" in r
                }
                for route in report.ROUTES
            }
            common = sorted(set.intersection(*(set(g) for g in grouped.values())))
            values = {route: [g[cid] for cid in common] for route, g in grouped.items()}
            for name, members in {"deepseek_luna": ("deepseek", "luna"), "all_four": report.ROUTES}.items():
                values[name] = []
                for cid in common:
                    row = dict(grouped[members[0]][cid])
                    row["route"] = name
                    report._set_errors(
                        row, sum(grouped[route][cid]["student_score"] for route in members) / len(members)
                    )
                    values[name].append(row)
            result["arms"][arm][task] = {
                "case_ids": common,
                "metrics": {name: report.metrics(rs) for name, rs in values.items()},
                "minus_deepseek_luna": {
                    name: report.paired_bootstrap(rs, values["deepseek_luna"])
                    for name, rs in values.items()
                    if name != "deepseek_luna"
                },
            }
    (HERE / "common-case-comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                arm: {
                    task: {
                        name: round(m["mae"], 3) if m["mae"] is not None else None
                        for name, m in value["metrics"].items()
                    }
                    for task, value in tasks.items()
                }
                for arm, tasks in result["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
