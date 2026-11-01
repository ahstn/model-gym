"""Analyze paired eligibility prompts, freeze selection, and export case-level comparisons."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")
MODELS = ("liquid", "jev")
TASKS = ("technical_quality", "aesthetics")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def result(round_id, model, case):
    root = DATA / "attempts" / round_id / model / case["case_id"]
    requests = sorted(root.glob("*/request.json")) if root.exists() else []
    attempts = []
    for req in requests:
        request = read(req)
        assert request["view_sha256"] == case["view_sha256"]
        p = req.with_name("result.json")
        r = read(p) if p.exists() else {"status": "unknown_no_response", "price_usd": None}
        attempts.append(
            {
                "request_path": str(req),
                "request_sha256": sha(req),
                "result_path": str(p) if p.exists() else None,
                "result_sha256": sha(p) if p.exists() else None,
                "result": r,
            }
        )
    final = attempts[-1]["result"] if attempts else {"status": "missing"}
    parsed = final.get("parsed") or {}
    ratings = parsed.get("ratings") or {}
    native = parsed.get("native_scores") or {}
    return {
        "case_id": case["case_id"],
        "partition": case["partition"],
        "group": case["group"],
        "reference_eligibility": case["eligibility"],
        "family_id": case["family_id"],
        "model": model,
        "round": round_id,
        "status": final["status"],
        "exclusion_probability": parsed.get("excluded_probability"),
        "scope_floor_applied": parsed.get("scope_floor_applied") if final["status"] == "valid" else None,
        "technical_score": ratings.get("technical_quality", {}).get("score"),
        "aesthetic_score": ratings.get("aesthetics", {}).get("score"),
        "technical_native": native.get("technical_quality"),
        "aesthetic_native": native.get("aesthetics"),
        "latency_seconds": final.get("latency_seconds"),
        "price_usd": final.get("price_usd"),
        "attempt_count": len(attempts),
        "attempts": [{k: v for k, v in a.items() if k != "result"} for a in attempts],
    }


def evaluate(rows):
    out = {
        "expected": len(rows),
        "valid": 0,
        "excluded_expected": 0,
        "eligible_expected": 0,
        "excluded_valid": 0,
        "eligible_valid": 0,
        "missed_exclusions": 0,
        "false_floors": 0,
        "correct_exclusions": 0,
        "correct_eligible": 0,
        "missing_ids": [],
        "false_floor_ids": [],
        "missed_exclusion_ids": [],
        "by_group": {},
    }
    for r in rows:
        group = out["by_group"].setdefault(r["group"], {"expected": 0, "valid": 0, "errors": 0})
        group["expected"] += 1
        truth = r["reference_eligibility"]
        out[truth + "_expected"] += 1
        if r["status"] != "valid" or r["scope_floor_applied"] is None:
            out["missing_ids"].append(r["case_id"])
            continue
        out["valid"] += 1
        out[truth + "_valid"] += 1
        group["valid"] += 1
        if truth == "excluded" and r["scope_floor_applied"]:
            out["correct_exclusions"] += 1
        elif truth == "excluded":
            out["missed_exclusions"] += 1
            out["missed_exclusion_ids"].append(r["case_id"])
            group["errors"] += 1
        elif r["scope_floor_applied"]:
            out["false_floors"] += 1
            out["false_floor_ids"].append(r["case_id"])
            group["errors"] += 1
        else:
            out["correct_eligible"] += 1
    return out


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - delta), min(1, center + delta)]


def comparison(baseline, candidate):
    b = {r["case_id"]: r for r in baseline}
    pairs = [
        (b[r["case_id"]], r)
        for r in candidate
        if r["case_id"] in b and r["status"] == "valid" and b[r["case_id"]]["status"] == "valid"
    ]
    base = evaluate([a for a, _ in pairs])
    cand = evaluate([z for _, z in pairs])
    changes = {}
    for key in ("technical_native", "aesthetic_native"):
        v = [
            z[key] - a[key]
            for a, z in pairs
            if z["reference_eligibility"] == "eligible" and a[key] is not None and z[key] is not None
        ]
        changes[key] = {
            "paired_photos": len(v),
            "mean_signed_change": statistics.mean(v) if v else None,
            "mean_absolute_change": statistics.mean(map(abs, v)) if v else None,
            "changes_at_least_half_point": sum(abs(x) >= 0.5 - 1e-12 for x in v),
        }
    return {
        "paired_cases": len(pairs),
        "baseline_paired": base,
        "candidate_paired": cand,
        "native_quality_drift": changes,
    }


def manifest():
    return read(HERE / "reference-manifest.json")["cases"]


def dev_summary(rounds):
    cases = [c for c in manifest() if c["partition"] == "development"]
    data = {}
    for model in MODELS:
        base = [result("baseline", model, c) for c in cases]
        bm = evaluate(base)
        entry = {"baseline": bm, "rounds": {}}
        for r in rounds:
            rr = [result(r, model, c) for c in cases]
            entry["rounds"][r] = {"metrics": evaluate(rr), "paired": comparison(base, rr)}
        data[model] = entry
    return data


def choose(dev, prompts):
    out = {}
    for model, entry in dev.items():
        baseline = entry["baseline"]
        candidates = []
        for round_id, x in entry["rounds"].items():
            m = x["metrics"]
            paired = x["paired"]
            b = paired["baseline_paired"]
            c = paired["candidate_paired"]
            eligible = (
                m["valid"] >= 38
                and m["eligible_valid"] >= 19
                and m["excluded_valid"] >= 19
                and b["eligible_valid"] >= 19
                and b["excluded_valid"] >= 19
                and c["false_floors"] <= b["false_floors"] + 1
            )
            if eligible:
                candidates.append(
                    (
                        c["false_floors"] + c["missed_exclusions"],
                        c["false_floors"],
                        len(prompts[round_id]),
                        round_id,
                        b,
                        c,
                    )
                )
        if not candidates:
            raise ValueError("No candidate meets development coverage/safety gate for " + model)
        best = min(candidates)
        _, _, _, round_id, b, c = best
        improves = (
            c["missed_exclusions"] < b["missed_exclusions"]
            and c["false_floors"] + c["missed_exclusions"] <= b["false_floors"] + b["missed_exclusions"]
        )
        out[model] = {
            "candidate_round": round_id,
            "advance_from_development": improves,
            "working_default_after_development": round_id if improves else "baseline",
            "baseline_full": baseline,
            "paired_baseline": b,
            "paired_candidate": c,
            "rule": "coverage >=38/40 and >=19 per class; paired false floors <= baseline+1; fewest paired errors, then false floors, prompt length, round ID; improvement requires fewer missed exclusions and no increase in total errors",
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["first", "select", "final"])
    args = ap.parse_args()
    rounds = [f"r{i:02d}" for i in range(1, 6 if args.stage == "first" else 11)]
    dev = dev_summary(rounds)
    if args.stage == "first":
        p = HERE / "development-first.json"
        p.write_text(
            json.dumps(
                {
                    "generated_at": now(),
                    "reference_sha256": sha(HERE / "reference-manifest.json"),
                    "summary": dev,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "output": str(p),
                    "summary": {
                        m: {
                            r: {k: v["metrics"][k] for k in ("valid", "missed_exclusions", "false_floors")}
                            for r, v in d["rounds"].items()
                        }
                        for m, d in dev.items()
                    },
                }
            )
        )
        return
    prompts = read(HERE / "rounds-first.json")
    prompts.update(read(HERE / "rounds-second.json"))
    if args.stage == "select":
        if (DATA / "attempts" / "heldout-baseline").exists() or (
            DATA / "attempts" / "heldout-selected"
        ).exists():
            raise ValueError("Candidate selection must be frozen before any held-out requests")
        if (HERE / "winners.json").exists() or (HERE / "winners-freeze.json").exists():
            raise ValueError("Refusing to replace an existing frozen candidate selection")
        development = [c for c in manifest() if c["partition"] == "development"]
        for round_id in ["baseline", *rounds]:
            for model in MODELS:
                if any(result(round_id, model, case)["attempt_count"] == 0 for case in development):
                    raise ValueError(f"All ten rounds must attempt every development job: {round_id}/{model}")
        winners = {
            "selected_at": now(),
            "reference_sha256": sha(HERE / "reference-manifest.json"),
            "rounds_first_sha256": sha(HERE / "rounds-first.json"),
            "rounds_second_sha256": sha(HERE / "rounds-second.json"),
            "models": choose(dev, prompts),
            "heldout_model_results_used": False,
        }
        p = HERE / "winners.json"
        p.write_text(json.dumps(winners, indent=2) + "\n")
        (HERE / "winners-freeze.json").write_text(
            json.dumps({"frozen_at": now(), "sha256": sha(p), "heldout_model_results_used": False}, indent=2)
            + "\n"
        )
        print(
            json.dumps(
                {
                    "output": str(p),
                    "winners": {
                        m: {
                            k: v[k]
                            for k in (
                                "candidate_round",
                                "advance_from_development",
                                "working_default_after_development",
                            )
                        }
                        for m, v in winners["models"].items()
                    },
                }
            )
        )
        return
    winners = read(HERE / "winners.json")
    cases = [c for c in manifest() if c["partition"] in ("heldout", "natural_challenge")]
    all_rows = []
    summary = {}
    for model in MODELS:
        for case in (c for c in manifest() if c["partition"] == "development"):
            for display_arm, round_id in (
                ("baseline", "baseline"),
                ("candidate", winners["models"][model]["candidate_round"]),
            ):
                all_rows.append({**result(round_id, model, case), "display_arm": display_arm})
    for partition in ("heldout", "natural_challenge"):
        sub = [c for c in cases if c["partition"] == partition]
        summary[partition] = {}
        for model in MODELS:
            baseline = [result("heldout-baseline", model, c) for c in sub]
            candidate = [result("heldout-selected", model, c) for c in sub]
            all_rows.extend({**row, "display_arm": "baseline"} for row in baseline)
            all_rows.extend({**row, "display_arm": "candidate"} for row in candidate)
            bm = evaluate(baseline)
            cm = evaluate(candidate)
            for metrics in (bm, cm):
                metrics["miss_rate_wilson95"] = wilson(
                    metrics["missed_exclusions"], metrics["excluded_valid"]
                )
                metrics["false_floor_wilson95"] = wilson(metrics["false_floors"], metrics["eligible_valid"])
            summary[partition][model] = {
                "candidate_round": winners["models"][model]["candidate_round"],
                "baseline": bm,
                "candidate": cm,
                "paired": comparison(baseline, candidate),
            }
    p = HERE / "results.json"
    p.write_text(
        json.dumps(
            {
                "generated_at": now(),
                "reference_sha256": sha(HERE / "reference-manifest.json"),
                "winners_sha256": sha(HERE / "winners.json"),
                "rounds_first_sha256": sha(HERE / "rounds-first.json"),
                "rounds_second_sha256": sha(HERE / "rounds-second.json"),
                "development": dev,
                "evaluation": summary,
                "case_rows": all_rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps({"output": str(p), "evaluation": summary}))


if __name__ == "__main__":
    main()
