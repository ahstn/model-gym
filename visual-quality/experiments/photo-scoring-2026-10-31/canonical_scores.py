"""Recompute this historical comparison with the current shared ensemble.

Provider calls, prompts and scope decisions remain the original observations.
Only the derived comparison changes. No network access or credentials are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from visual_quality.ensemble import MODEL, configuration, score_ensemble

TASKS = {"technical_quality": "technical", "aesthetics": "aesthetic"}
ARCHIVED = "historical_luna_deepseek_mean"


def load_scores() -> list[dict]:
    """Validate exact image joins and derive scores from native observations."""
    with (ROOT / "scores-72.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 72 or len({r["case_id"] for r in rows}) != 72:
        raise ValueError("Expected the original 72 distinct cases")
    with (ROOT / "scores-2000.csv").open(newline="") as stream:
        teachers = {r["case_id"]: r for r in csv.DictReader(stream)}
    responses = {}
    for line in (ROOT / "selected-responses-72.jsonl").read_text().splitlines():
        response = json.loads(line, parse_float=Decimal)
        key = (response["case_id"], response["model_requested"])
        if key in responses:
            raise ValueError("Duplicate selected response")
        responses[key] = response
    for row in rows:
        cid = row["case_id"]
        teacher = teachers[cid]
        if teacher["view_sha256"] != row["sha256"]:
            raise ValueError("Teacher delivered-view mismatch: " + cid)
        image = ROOT / row["image_path"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Delivered image bytes changed: " + cid)
        components = {model: _teacher_component(teacher, model) for model in ("luna", "deepseek")}
        liquid = responses[cid, "liquid/d1.1"]
        normalized = liquid["normalized"]
        components["liquid"] = {
            "case_id": cid,
            "model": "liquid",
            "arm": "baseline",
            "status": "valid",
            "view_sha256": row["sha256"],
            "model_returned": liquid["model_returned"],
            "excluded_probability": normalized["excluded_probability"],
            "scope_floor_applied": normalized["excluded_probability"] >= Decimal("0.5"),
            "price_usd": liquid["response"].get("usage", {}).get("cost"),
        }
        for task, dimension in TASKS.items():
            # Normalized values retain the study's original floor and native precision.
            components["liquid"][dimension + "_score"] = normalized[task]
            components["liquid"][dimension + "_score_valid"] = normalized[task + "_status"] == "ok"
            row[ARCHIVED + "_" + task] = row.pop("average_" + task, teacher["average_" + task])
            row["liquid_" + task + "_native"] = str(normalized[task + "_native"])
            row["liquid_" + task + "_policy_native"] = str(normalized[task])
        derived = score_ensemble(components, case_id=cid)
        for task, dimension in TASKS.items():
            row[MODEL + "_" + task] = derived[dimension + "_score_display"] or ""
            row[MODEL + "_" + task + "_native"] = derived[dimension + "_score_decimal"] or ""
        row["ensemble_version"] = derived["ensemble_version"]
        row["ensemble_status"] = derived["status"]
        row["ensemble_latency_seconds"] = ""
        row["ensemble_latency_kind"] = derived["latency_kind"]
        row["ensemble_price_usd"] = ""
        row["ensemble_price_basis"] = derived["price_basis"]
    return rows


def _teacher_component(row: dict, model: str) -> dict:
    component = {
        "case_id": row["case_id"],
        "model": model,
        "arm": "baseline",
        "view_sha256": row["view_sha256"],
        "status": "valid",
        "price_usd": row[model + "_selected_response_cost_usd"],
    }
    for task, dimension in TASKS.items():
        component[dimension + "_score"] = row[model + "_" + task]
        component[dimension + "_score_valid"] = row[model + "_" + task + "_status"] == "ok"
    return component


def main() -> None:
    rows = load_scores()
    with (ROOT / "scores-72.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    heldout = [row for row in rows if row["partition"] == "heldout"]
    maes = {
        task: statistics.mean(
            abs(Decimal(row[MODEL + "_" + task + "_native"]) - Decimal(row["reference_" + task]))
            for row in heldout
            if row[MODEL + "_" + task + "_native"]
        )
        for task in TASKS
    }
    audit = {
        **configuration(),
        "images": len(rows),
        "complete_ensembles": sum(row["ensemble_status"] == "valid" for row in rows),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scores-2000.csv",
                "selected-responses-72.jsonl",
                "decision-prompts.json",
                "scores-72.csv",
            )
        },
        "recomputed_from": "Original Luna/DeepSeek schema scores and native normalized Liquid policy scores",
        "observation_status": "Original October prompts and calls; no new model calls or relabeling as the v2 run",
        "historical_columns": ARCHIVED + "_* retain the superseded 50/50 study mean for provenance only",
        "heldout_48_ensemble_mae": {task: str(value) for task, value in maes.items()},
        "heldout_48_ensemble_mean_mae": str(statistics.mean(maes.values())),
    }
    (ROOT / "canonical-ensemble-provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "images": audit["images"],
                "complete_ensembles": audit["complete_ensembles"],
                "mae": audit["heldout_48_ensemble_mae"],
            }
        )
    )


if __name__ == "__main__":
    main()
