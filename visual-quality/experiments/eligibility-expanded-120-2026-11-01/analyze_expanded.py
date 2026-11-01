"""Choose on development only, then compute final paired eligibility comparisons."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")
PREV = HERE.parent / "eligibility-optimization-2026-11-01"
spec = importlib.util.spec_from_file_location("prior_eligibility_analyze", PREV / "analyze.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.DATA = DATA
MODELS = ("liquid", "jev")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def data(round_id, model, cases):
    return [a.result(round_id, model, c) for c in cases]


def valid(x):
    return x["status"] == "valid" and x["scope_floor_applied"] is not None


def err(x):
    return (
        (not x["scope_floor_applied"])
        if x["reference_eligibility"] == "excluded"
        else bool(x["scope_floor_applied"])
    )


def select():
    inputs = read(HERE / "trial-freeze.json")["inputs"]
    for p, h in inputs.items():
        assert sha(p) == h
    assert sha(HERE / "selection-rule.json") == read(HERE / "selection-rule-freeze.json")["sha256"]
    cases = [x for x in read(HERE / "reference-manifest.json")["cases"] if x["partition"] == "development"]
    variants = read(HERE / "variants.json")
    all_results = {}
    chosen = {}
    for model in MODELS:
        prev = data("previous", model, cases)
        previous = {x["case_id"]: x for x in prev if valid(x)}
        results = {}
        eligible = []
        for rid in variants:
            rows = data(rid, model, cases)
            metrics = a.evaluate(rows)
            paired = a.comparison(prev, rows)
            p = paired["baseline_paired"]
            c = paired["candidate_paired"]
            rowmap = {x["case_id"]: x for x in rows}
            lost = sum(id not in rowmap or not valid(rowmap[id]) for id in previous)
            penalty = c["missed_exclusions"] + c["false_floors"] + lost
            passes = (
                metrics["valid"] >= 114
                and metrics["eligible_valid"] >= 57
                and metrics["excluded_valid"] >= 57
                and c["false_floors"] <= p["false_floors"]
                and c["missed_exclusions"] + c["false_floors"] < p["missed_exclusions"] + p["false_floors"]
            )
            results[rid] = {
                "metrics": metrics,
                "paired_vs_previous": paired,
                "missing_on_previous_valid": lost,
                "rank_error_with_missing_penalty": penalty,
                "qualifies": passes,
            }
            if passes:
                eligible.append((penalty, c["false_floors"], c["missed_exclusions"], len(variants[rid]), rid))
        rid = min(eligible)[-1] if eligible else "previous"
        chosen[model] = {
            "round": rid,
            "qualified_new_variants": len(eligible),
            "selected_rank": list(min(eligible)) if eligible else None,
        }
        all_results[model] = {
            "original": a.evaluate(data("original", model, cases)),
            "previous": a.evaluate(prev),
            "rounds": results,
        }
    output = {
        "created_at": now(),
        "rule_sha256": sha(HERE / "selection-rule.json"),
        "reference_sha256": sha(HERE / "reference-manifest.json"),
        "models": chosen,
        "development": all_results,
    }
    (HERE / "winner.json").write_text(json.dumps(output, indent=2) + "\n")
    (HERE / "winner-freeze.json").write_text(
        json.dumps(
            {"frozen_at": now(), "sha256": sha(HERE / "winner.json"), "heldout_model_results_used": False},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(chosen))


def final():
    wf = read(HERE / "winner-freeze.json")
    assert wf["sha256"] == sha(HERE / "winner.json")
    cases = read(HERE / "reference-manifest.json")["cases"]
    hold = [x for x in cases if x["partition"] == "heldout"]
    w = read(HERE / "winner.json")
    out = {
        "created_at": now(),
        "reference_sha256": sha(HERE / "reference-manifest.json"),
        "winner_sha256": sha(HERE / "winner.json"),
        "models": {},
        "case_rows": [],
    }
    for model in MODELS:
        by = {arm: data("holdout-" + arm, model, hold) for arm in ("original", "previous", "selected")}
        metrics = {k: a.evaluate(v) for k, v in by.items()}
        comparisons = {k: a.comparison(by[k], by["selected"]) for k in ("original", "previous")}
        out["models"][model] = {
            "chosen_round": w["models"][model]["round"],
            "arms": metrics,
            "selected_vs": comparisons,
        }
        for arm, rows in by.items():
            for row in rows:
                row["display_arm"] = arm
                out["case_rows"].append(row)
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                m: {
                    k: {x: v[x] for x in ("valid", "missed_exclusions", "false_floors")}
                    for k, v in z["arms"].items()
                }
                for m, z in out["models"].items()
            }
        )
    )


if __name__ == "__main__":
    import sys

    {"select": select, "final": final}[sys.argv[1]]()
