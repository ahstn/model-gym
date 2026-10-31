"""Paired per-task agreement with frozen assistant references; no combined score."""

import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/dual-score-trial-2026-10-27"
TASKS = ("technical_quality", "aesthetics")
ROUTES = ("deepseek", "gemini", "luna")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mae": float(np.mean([r["absolute_error"] for r in rows])),
        "bias": float(np.mean([r["error"] for r in rows])),
        "mean_interval_distance": float(np.mean([r["interval_distance"] for r in rows])),
        "interval_hits": sum(r["interval_distance"] == 0 for r in rows),
        "interval_hit_rate": sum(r["interval_distance"] == 0 for r in rows) / len(rows),
        "large_interval_errors": sum(r["interval_distance"] >= 1 - 1e-9 for r in rows),
        "critical_contradictions": sum(r["critical_contradiction"] for r in rows),
    }


def paired_bootstrap(rows_a, rows_b, draws=10000, seed=20261027):
    """Resample whole families within each source stratum, using paired ratio means."""
    a = {r["case_id"]: r for r in rows_a}
    b = {r["case_id"]: r for r in rows_b}
    if set(a) != set(b):
        raise ValueError("Paired bootstrap requires identical case IDs")
    if not a:
        return {"n": 0}
    grouped = {}
    for key, row in a.items():
        group = grouped.setdefault(row["stratum"], {}).setdefault(row["family_id"], [])
        group.append(
            (
                row["absolute_error"] - b[key]["absolute_error"],
                row["interval_distance"] - b[key]["interval_distance"],
            )
        )
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws, 2))
    counts = np.zeros(draws)
    for stratum in sorted(grouped):
        families = grouped[stratum]
        ordered = [families[k] for k in sorted(families)]
        sums = np.array([np.sum(f, axis=0) for f in ordered])
        sizes = np.array([len(f) for f in ordered])
        picks = rng.integers(0, len(ordered), size=(draws, len(ordered)))
        totals += sums[picks].sum(axis=1)
        counts += sizes[picks].sum(axis=1)
    samples = totals / counts[:, None]
    result = {"n": len(a), "families": sum(len(g) for g in grouped.values()), "draws": draws, "seed": seed}
    for index, (label, field) in enumerate(
        (("mae_difference", "absolute_error"), ("interval_distance_difference", "interval_distance"))
    ):
        result[label] = {
            "estimate": float(np.mean([r[field] - b[k][field] for k, r in a.items()])),
            "ci95": np.quantile(samples[:, index], [0.025, 0.975]).tolist(),
        }
    return result


def comparison_rows(cases, references, observations):
    reference = {r["case_id"]: r for r in references}
    if len(reference) != len(cases) or set(reference) != {c["case_id"] for c in cases}:
        raise ValueError("Reference coverage does not match cohort")
    observed = {}
    for route in ROUTES:
        rows = observations[route]
        observed[route] = {r["case_id"]: r for r in rows}
        if len(observed[route]) != len(rows) or not set(observed[route]).issubset(reference):
            raise ValueError("Duplicate or unknown student case")
    result = []
    for case in cases:
        ref = reference[case["case_id"]]
        if ref["view_sha256"] != case["view_sha256"]:
            raise ValueError("Reference view mismatch")
        for route, task in itertools.product(ROUTES, TASKS):
            obs = observed[route].get(case["case_id"], {})
            rating = obs.get(task, {})
            if obs and obs.get("view_sha256") != case["view_sha256"]:
                raise ValueError("Student view mismatch")
            r = {
                "case_id": case["case_id"],
                "route": route,
                "task": task,
                "family_id": case["family_id"],
                "stratum": case["stratum"],
                "role": case["role"],
                "operator": case.get("operator"),
                "severity": case.get("severity"),
                "reference_score": ref[task]["score"],
                "reference_interval": ref[task]["interval"],
                "student_score": rating.get("score") if rating.get("status") == "ok" else None,
                "student_status": rating.get("status", "missing"),
                "student_evidence": rating.get("evidence"),
                "reference_evidence": ref[task]["evidence"],
                "validation_error": obs.get("validation_errors", {}).get(task),
                "matched": False,
            }
            if r["student_score"] is not None and r["reference_score"] is not None:
                score, point = r["student_score"], r["reference_score"]
                low, high = r["reference_interval"]
                r.update(
                    error=score - point,
                    absolute_error=abs(score - point),
                    interval_distance=max(low - score, 0, score - high),
                    critical_contradiction=score >= 4 and high <= 2 or score <= 2 and low >= 4,
                )
            result.append(r)
    for task in TASKS:
        eligible = [
            {r["case_id"] for r in result if r["task"] == task and r["route"] == route and "error" in r}
            for route in ROUTES
        ]
        common = set.intersection(*eligible)
        for row in result:
            if row["task"] == task:
                row["matched"] = row["case_id"] in common
    return result


def summarize(rows, draws=10000):
    result = {
        "reference_kind": "frozen_assistant_judgments",
        "human_gold": False,
        "training_ready": False,
        "tasks": {},
        "bootstrap_scope": "Whole families within strata, conditional on these assistant references; not human accuracy.",
    }
    for task in TASKS:
        available, matched = {}, {}
        for route in ROUTES:
            available[route] = [r for r in rows if r["task"] == task and r["route"] == route and "error" in r]
            matched[route] = [r for r in available[route] if r["matched"]]
        task_result = {
            "available": {route: metrics(rs) for route, rs in available.items()},
            "matched": {route: metrics(rs) for route, rs in matched.items()},
            "matched_case_ids": sorted(r["case_id"] for r in matched[ROUTES[0]]),
            "by_stratum": {},
            "paired_differences": {},
        }
        for route in ROUTES:
            task_result["by_stratum"][route] = {
                s: metrics([r for r in available[route] if r["stratum"] == s])
                for s in sorted({r["stratum"] for r in rows})
            }
        for a, b in itertools.combinations(ROUTES, 2):
            task_result["paired_differences"][f"{a}_minus_{b}"] = paired_bootstrap(
                matched[a], matched[b], draws
            )
        task_result["ranking_by_matched_mae"] = (
            sorted(ROUTES, key=lambda route: task_result["matched"][route]["mae"])
            if task_result["matched_case_ids"]
            else []
        )
        task_result["comparison_status"] = (
            "paired" if task_result["matched_case_ids"] else "no_common_scorable_cases"
        )
        result["tasks"][task] = task_result
    result["controlled_diagnostics"] = {}
    for route, task in itertools.product(ROUTES, TASKS):
        families = {}
        for row in rows:
            if (
                row["route"] == route
                and row["task"] == task
                and row["stratum"] == "controlled"
                and "error" in row
            ):
                families.setdefault(row["family_id"], []).append(row)
        complete = [sorted(f, key=lambda r: r["severity"] or 0) for f in families.values() if len(f) == 4]
        result["controlled_diagnostics"][f"{route}:{task}"] = {
            "complete_families": len(complete),
            "nonincreasing_score_families": sum(
                all(a["student_score"] >= b["student_score"] for a, b in itertools.pairwise(f))
                for f in complete
            ),
            "note": "Operator severity ordering is diagnostic, not perceptual or aesthetic ground truth.",
        }
    return result


def main():
    # Runner verification independently reconstructs raw replies before this export.
    import runner

    manifests = json.loads((HERE / "manifest.json").read_text())
    observations, accounting = {}, {}
    for route in ROUTES:
        if not (DATA / route / "images/run.json").exists():
            accounting[route] = {"status": "images_not_started", "planned": 50, "attempted": 0}
            if (DATA / route / "probes/run.json").exists():
                accounting[route]["probes"] = runner.verify(runner.read_run(route, "probes"))
            observations[route] = []
        else:
            run = runner.read_run(route, "images")
            accounting[route] = runner.verify(run)
            observations[route] = read_rows(DATA / route / "images/observations.jsonl")
    rows = comparison_rows(manifests["cases"], read_rows(HERE / "references.jsonl"), observations)
    result = summarize(rows)
    result["accounting"] = accounting
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (HERE / "comparisons.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({task: result["tasks"][task]["matched"] for task in TASKS}, indent=2))


if __name__ == "__main__":
    main()
