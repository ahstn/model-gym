"""Prepare a blinded, preselected explanation audit and decode its findings."""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/dual-score-trial-2026-10-27"
ROUTES = ("deepseek", "gemini", "luna")
TASKS = ("technical_quality", "aesthetics")


def prepare():
    selected = json.loads((HERE / "audit-selection.json").read_text())["case_ids"]
    cases = {c["case_id"]: c for c in json.loads((HERE / "manifest.json").read_text())["cases"]}
    observed = {}
    for route in ROUTES:
        path = DATA / route / "images/observations.jsonl"
        observed[route] = (
            {r["case_id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
        )
    rng = random.Random(20261027)
    packet, key = [], {}
    for case_id in selected:
        routes = list(ROUTES)
        rng.shuffle(routes)
        case = cases[case_id]
        explanations = []
        for alias, route in zip(("A", "B", "C"), routes):
            key[f"{case_id}:{alias}"] = route
            for task in TASKS:
                rating = observed[route].get(case_id, {}).get(task, {})
                explanations.append(
                    {
                        "alias": alias,
                        "task": task,
                        "evidence": rating.get("evidence"),
                        "available": rating.get("status") in {"ok", "unscorable"},
                    }
                )
        packet.append(
            {
                "case_id": case_id,
                "path": case["image_path"],
                "view_sha256": case["view_sha256"],
                "explanations": explanations,
            }
        )
    folder = DATA / "explanation-audit"
    folder.mkdir(exist_ok=True)
    for name, value in (("packet.json", packet), ("key.json", key)):
        destination = folder / name
        if destination.exists():
            raise ValueError("Audit packet already exists")
        destination.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"cases": len(packet), "packet": str(folder / "packet.json")}))


def summarize():
    folder = DATA / "explanation-audit"
    packet = json.loads((folder / "packet.json").read_text())
    key = json.loads((folder / "key.json").read_text())
    findings = [json.loads(line) for line in (HERE / "explanation-audit.jsonl").read_text().splitlines()]
    expected = {(c["case_id"], e["alias"], e["task"]) for c in packet for e in c["explanations"]}
    actual = {(r["case_id"], r["alias"], r["task"]) for r in findings}
    if len(findings) != len(expected) or actual != expected:
        raise ValueError("Audit coverage mismatch")
    counts = {
        route: {
            task: dict.fromkeys(("supported", "mixed", "unsupported", "unavailable"), 0) for task in TASKS
        }
        for route in ROUTES
    }
    decoded = []
    for row in findings:
        if (
            row["verdict"] not in ("supported", "mixed", "unsupported", "unavailable")
            or not row["reason"].strip()
        ):
            raise ValueError("Invalid audit judgment")
        route = key[f"{row['case_id']}:{row['alias']}"]
        counts[route][row["task"]][row["verdict"]] += 1
        decoded.append({**row, "route": route})
    result = {
        "cases": len(packet),
        "explanations": len(findings),
        "selection": "preselected 12-case sample",
        "scores_and_model_identities_hidden_from_auditor": True,
        "counts": counts,
        "findings": decoded,
    }
    (HERE / "explanation-audit-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else summarize()
