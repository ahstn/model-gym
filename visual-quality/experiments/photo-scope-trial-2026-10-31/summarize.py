"""Audit the requested domain floor without repairing any model responses."""

import collections
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / HERE.name
PREVIOUS = HERE.parent / "prompt-ablation-2026-10-30"
TASKS = ("technical_quality", "aesthetics")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    spec = importlib.util.spec_from_file_location("scope_summary_runner", HERE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    refs = {r["case_id"]: r for r in rows(HERE / "references.jsonl")}
    freeze = json.loads((HERE / "references-freeze.json").read_text())
    assert freeze["references_sha256"] == hashlib.sha256((HERE / "references.jsonl").read_bytes()).hexdigest()
    assert len(refs) == 72
    result = {
        "scope": "Photography eligibility audit of raw model outputs; reference judgments are assistant assessments, not human gold.",
        "reference_eligibility": dict(collections.Counter(r["eligibility"] for r in refs.values())),
        "excluded_reference_ids": [key for key, r in refs.items() if r["eligibility"] == "excluded"],
        "uncertain_reference_ids": [key for key, r in refs.items() if r["eligibility"] == "uncertain"],
        "routes": {},
    }
    total = Decimal(0)
    for route in ("deepseek", "luna", "gemini", "haiku"):
        run = runner.read_run(route)
        report = runner.verify(run)
        assert report["status"] == "complete" and report["attempted"] == 72
        observed = rows(DATA / route / "images/observations.jsonl")
        assert len(observed) == 72 and {r["case_id"] for r in observed} == set(refs)
        audit = []
        for row in observed:
            case_id = row["case_id"]
            ref = refs[case_id]
            assert row["view_sha256"] == ref["view_sha256"]
            valid = {t: t not in row["validation_errors"] and row[t]["status"] == "ok" for t in TASKS}
            flagged = [t for t in TASKS if valid[t] and "unsupported_domain" in row[t].get("limitations", [])]
            floor = all(valid[t] and row[t]["score"] == 1.0 and t in flagged for t in TASKS)
            issue = (
                "invalid_rating" if not all(valid.values())
                else "missed_exclusion_or_incomplete_floor" if ref["eligibility"] == "excluded" and not floor
                else "false_exclusion" if ref["eligibility"] == "eligible" and flagged
                else "uncertain_reference_exclusion" if ref["eligibility"] == "uncertain" and flagged
                else None
            )
            audit.append({
                "case_id": case_id,
                "reference_eligibility": ref["eligibility"],
                "reference_evidence": ref["eligibility_evidence"],
                "model_exclusion_tasks": flagged,
                "complete_policy_floor": floor,
                "issue": issue,
                "validation_errors": row["validation_errors"],
                **{t: row[t] for t in TASKS},
            })
        old_data = HERE.parents[1] / "data/prompt-ablation-2026-10-30" / route / "images/observations.jsonl"
        old_rows = {r["case_id"]: r for r in rows(old_data) if r["arm_id"] == "combined"}
        old = old_rows["P048"]
        current = next(r for r in observed if r["case_id"] == "P048")
        quality = {}
        for task in TASKS:
            paired = []
            for row in observed:
                case_id = row["case_id"]
                before = old_rows.get(case_id)
                if before is None or refs[case_id]["eligibility"] != "eligible":
                    continue
                if task in before["validation_errors"] or task in row["validation_errors"]:
                    continue
                scores = (before[task].get("score"), row[task].get("score"), refs[case_id][task].get("score"))
                if any(score is None for score in scores):
                    continue
                prior_score, new_score, reference = scores
                paired.append({
                    "case_id": case_id, "before": prior_score, "after": new_score, "reference": reference,
                    "before_error": abs(prior_score - reference), "after_error": abs(new_score - reference),
                    "error_increase": abs(new_score - reference) - abs(prior_score - reference),
                })
            quality[task] = {
                "n": len(paired),
                "before_mae": sum(r["before_error"] for r in paired) / len(paired) if paired else None,
                "after_mae": sum(r["after_error"] for r in paired) / len(paired) if paired else None,
                "delta_mae": sum(r["error_increase"] for r in paired) / len(paired) if paired else None,
                "cases_worsened_at_least_0_5": [r for r in paired if r["error_increase"] >= 0.5 - 1e-9],
                "pairs": paired,
                "interpretation": "Descriptive comparison on retained eligible images with valid scores in both runs; single draws, not a significance-tested causal effect.",
            }
        route_summary = {
            "report": {k: v for k, v in report.items() if k != "attempts"},
            "exclusion_case_ids": [r["case_id"] for r in audit if r["model_exclusion_tasks"]],
            "complete_policy_floor_case_ids": [r["case_id"] for r in audit if r["complete_policy_floor"]],
            "issues": [r for r in audit if r["issue"]],
            "poster_before": {t: old[t] for t in TASKS},
            "poster_after": {t: current[t] for t in TASKS},
            "retained_photo_quality_comparison": quality,
            "audit": audit,
        }
        total += Decimal(report["known_cost_usd"])
        result["routes"][route] = route_summary
    result["known_cost_usd"] = str(total)
    result["unpriced_attempts"] = sum(v["report"]["unpriced_attempts"] for v in result["routes"].values())
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "reference_eligibility": result["reference_eligibility"],
        "known_cost_usd": result["known_cost_usd"],
        "routes": {k: {"floors": v["complete_policy_floor_case_ids"], "issues": [{"case_id": r["case_id"], "issue": r["issue"]} for r in v["issues"]]} for k, v in result["routes"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
