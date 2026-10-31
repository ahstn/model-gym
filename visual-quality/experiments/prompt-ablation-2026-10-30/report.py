"""Paired prompt ablations against frozen assistant references, separated by task."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/prompt-ablation-2026-10-30"
TASKS = ("technical_quality", "aesthetics")
ROUTES = ("deepseek", "gemini", "luna", "haiku")
ARMS = ("baseline", "A", "B", "C", "D", "combined")
SEED = 20261030
DRAWS = 10000
TOLERANCE = 1e-12
OPERATORS = {"gaussian_blur": "blur", "gaussian_noise": "noise", "jpeg_compression": "jpeg"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 1 <= number <= 5 else None


def _rank(values):
    values = np.asarray(values)
    return np.asarray([np.sum(values < v) + (np.sum(values == v) - 1) / 2 for v in values])


def metrics(rows):
    """Metrics use only rows with a valid student and scorable reference."""
    if not rows:
        return {
            "n": 0,
            "mae": None,
            "bias": None,
            "mean_interval_distance": None,
            "interval_hits": 0,
            "interval_hit_rate": None,
            "spearman": None,
            "large_interval_errors": 0,
            "critical_contradictions": 0,
        }
    observed = _rank([r["student_score"] for r in rows])
    reference = _rank([r["reference_score"] for r in rows])
    correlation = None
    if len(rows) > 1 and np.ptp(observed) > 0 and np.ptp(reference) > 0:
        correlation = float(np.corrcoef(observed, reference)[0, 1])
    hits = sum(r["interval_distance"] <= TOLERANCE for r in rows)
    return {
        "n": len(rows),
        "mae": float(np.mean([r["absolute_error"] for r in rows])),
        "bias": float(np.mean([r["error"] for r in rows])),
        "mean_interval_distance": float(np.mean([r["interval_distance"] for r in rows])),
        "interval_hits": hits,
        "interval_hit_rate": hits / len(rows),
        "spearman": correlation,
        "large_interval_errors": sum(r["interval_distance"] >= 1 - TOLERANCE for r in rows),
        "critical_contradictions": sum(r["critical_contradiction"] for r in rows),
    }


def _unique(rows, key, message):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(message)
    return result


def paired_bootstrap(rows_arm, rows_baseline, draws=DRAWS, seed=SEED):
    """Arm minus baseline; resample whole paired families within source strata."""
    arm = _unique(rows_arm, lambda r: r["case_id"], "Duplicate paired case")
    baseline = _unique(rows_baseline, lambda r: r["case_id"], "Duplicate paired case")
    if set(arm) != set(baseline):
        raise ValueError("Paired bootstrap requires identical case IDs")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    if not arm:
        return {"n": 0, "families": 0, "case_ids": [], "status": "no_shared_valid_cases"}
    grouped, family_strata = {}, {}
    for key, row in arm.items():
        other = baseline[key]
        if any(row[f] != other[f] for f in ("family_id", "stratum", "task")):
            raise ValueError("Paired case metadata differs")
        family, stratum = row["family_id"], row["stratum"]
        if family in family_strata and family_strata[family] != stratum:
            raise ValueError("A family cannot cross strata")
        family_strata[family] = stratum
        group = grouped.setdefault(stratum, {}).setdefault(family, [])
        group.append(
            (
                row["absolute_error"] - other["absolute_error"],
                row["interval_distance"] - other["interval_distance"],
            )
        )
    rng = np.random.default_rng(seed)
    totals, counts = np.zeros((draws, 2)), np.zeros(draws)
    for stratum in sorted(grouped):
        families = grouped[stratum]
        ordered = [families[k] for k in sorted(families)]
        sums = np.asarray([np.sum(f, axis=0) for f in ordered])
        sizes = np.asarray([len(f) for f in ordered])
        picks = rng.integers(0, len(ordered), size=(draws, len(ordered)))
        totals += sums[picks].sum(axis=1)
        counts += sizes[picks].sum(axis=1)
    samples = totals / counts[:, None]
    resampleable = any(len(families) > 1 for families in grouped.values())
    result = {
        "n": len(arm),
        "families": len(family_strata),
        "case_ids": sorted(arm),
        "draws": draws,
        "seed": seed,
        "strata_family_counts": {s: len(fs) for s, fs in grouped.items()},
        "status": "estimated" if resampleable else "degenerate_family_resampling",
    }
    for index, (label, field) in enumerate(
        (("mae_difference", "absolute_error"), ("interval_distance_difference", "interval_distance"))
    ):
        result[label] = {
            "estimate": float(np.mean([r[field] - baseline[k][field] for k, r in arm.items()])),
            "ci95": np.quantile(samples[:, index], [0.025, 0.975]).tolist(),
            "ci99_375": np.quantile(samples[:, index], [0.003125, 0.996875]).tolist(),
            "one_sided95_upper": float(np.quantile(samples[:, index], 0.95)),
        }
    return result


def _subgroups(case, ref):
    operator = OPERATORS.get(case.get("operator"), case.get("operator"))
    degraded = case.get("role") in {"variant", "degraded"}
    names = {"all", "stratum:" + case["stratum"], "role:" + case["role"]}
    if operator:
        names.add("operator:" + operator)
        if degraded:
            names.add("degraded:" + operator)
    if degraded:
        names.add("degraded")
        if case.get("severity") is not None:
            names.add("severity:" + str(case["severity"]))
    if case["role"] in {"source_parent", "parent"}:
        names.add("parents")
    for tag in case.get("issue_tags", []):
        names.add("issue:" + tag)
    for task in TASKS:
        point = _score(ref[task]["score"])
        if point is not None and point <= 2:
            names.add("reference_low:" + task)
        if point is not None and point >= 4:
            names.add("reference_high:" + task)
    return sorted(names)


def _set_errors(row, value):
    row["student_score"] = value
    low, high = row["reference_interval"]
    error = value - row["reference_score"]
    row.update(
        error=error,
        absolute_error=abs(error),
        interval_distance=max(low - value, 0, value - high),
        critical_contradiction=(value >= 4 and high <= 2) or (value <= 2 and low >= 4),
    )


def comparison_rows(cases, references, observations):
    cases_by_id = _unique(cases, lambda r: r["case_id"], "Duplicate manifest case")
    refs = _unique(references, lambda r: r["case_id"], "Duplicate reference case")
    if set(refs) != set(cases_by_id):
        raise ValueError("Reference coverage does not match cohort")
    if set(observations) - set(ROUTES):
        raise ValueError("Unknown route")
    indexed = {}
    for route in ROUTES:
        indexed[route] = _unique(
            observations.get(route, []), lambda r: (r["case_id"], r["arm_id"]), "Duplicate observation"
        )
        if any(cid not in refs or arm not in ARMS for cid, arm in indexed[route]):
            raise ValueError("Unknown observation case or arm")
    result = []
    for case in cases:
        ref = refs[case["case_id"]]
        if ref["view_sha256"] != case["view_sha256"]:
            raise ValueError("Reference view mismatch")
        groups = _subgroups(case, ref)
        for route, arm, task in itertools.product(ROUTES, ARMS, TASKS):
            obs = indexed[route].get((case["case_id"], arm), {})
            if obs and obs.get("view_sha256") != case["view_sha256"]:
                raise ValueError("Observation view mismatch")
            rating = obs.get(task, {})
            validation = obs.get("validation_errors", {}).get(task)
            value = _score(rating.get("score")) if rating.get("status") == "ok" and not validation else None
            point = _score(ref[task]["score"])
            interval = ref[task].get("interval")
            if point is not None and (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(_score(v) is None for v in interval)
                or not float(interval[0]) <= point <= float(interval[1])
            ):
                raise ValueError("Invalid reference interval")
            status = rating.get("status", "missing")
            if status == "ok" and value is None:
                status = "invalid"
            row = {
                "case_id": case["case_id"],
                "route": route,
                "arm_id": arm,
                "task": task,
                "family_id": case["family_id"],
                "stratum": case["stratum"],
                "role": case["role"],
                "operator": OPERATORS.get(case.get("operator"), case.get("operator")),
                "severity": case.get("severity"),
                "subgroups": groups,
                "reference_score": point,
                "reference_interval": interval,
                "reference_evidence": ref[task].get("evidence"),
                "student_score": value,
                "student_status": status,
                "student_evidence": rating.get("evidence"),
                "validation_error": validation,
            }
            if value is not None and point is not None:
                row["reference_interval"] = [float(x) for x in interval]
                _set_errors(row, value)
            result.append(row)
    return result


def coverage(rows):
    statuses = Counter(r["student_status"] for r in rows)
    return {
        "planned": len(rows),
        "present": len(rows) - statuses["missing"],
        "valid_scores": sum(r["student_score"] is not None for r in rows),
        "reference_scorable": sum(r["reference_score"] is not None for r in rows),
        "agreement_eligible": sum("error" in r for r in rows),
        "invalid": statuses["invalid"],
        "missing": statuses["missing"],
        "unscorable": statuses["unscorable"],
        "statuses": dict(statuses),
    }


def _paired(arm, baseline, draws):
    a = {r["case_id"]: r for r in arm if "error" in r}
    b = {r["case_id"]: r for r in baseline if "error" in r}
    common = sorted(set(a) & set(b))
    result = paired_bootstrap([a[k] for k in common], [b[k] for k in common], draws)
    result["coverage"] = {
        "arm": coverage(arm),
        "baseline": coverage(baseline),
        "excluded_arm_only": sorted(set(a) - set(b)),
        "excluded_baseline_only": sorted(set(b) - set(a)),
        "shared_valid_intersection": common,
    }
    result["paired_metrics"] = {
        "arm": metrics([a[k] for k in common]),
        "baseline": metrics([b[k] for k in common]),
    }
    return result


def _interpret(result, arm):
    if not result["n"]:
        return {"observed_impact": "not_evaluable", "support_status": "no_shared_valid_cases"}
    delta = result["mae_difference"]["estimate"]
    impact = "positive" if delta < -TOLERANCE else "negative" if delta > TOLERANCE else "unchanged"
    interval_key = "ci99_375" if arm == "combined" else "ci95"
    lo, hi = result["mae_difference"][interval_key]
    if result["status"] != "estimated":
        support = "insufficient_resampleable_families"
    elif hi < 0:
        support = "supports_reduced_error"
    elif lo > 0:
        support = "supports_increased_error"
    else:
        support = "uncertain"
    return {
        "observed_impact": impact,
        "support_status": support,
        "support_interval": interval_key,
        "test_role": "primary" if arm == "combined" else "exploratory",
    }


def _targeted(row, arm):
    degraded = row["role"] in {"variant", "degraded"}
    if arm == "A":
        return degraded and row["operator"] in {"noise", "jpeg"}
    if arm == "B":
        return degraded and row["operator"] == "blur"
    if arm == "C":
        return row["stratum"] == "vizwiz"
    return True


def _guardrails(arm, baseline, draws):
    a = {r["case_id"]: r for r in arm if "error" in r}
    b = {r["case_id"]: r for r in baseline if "error" in r}
    shared = set(a) & set(b)
    new = sorted(k for k in shared if a[k]["critical_contradiction"] and not b[k]["critical_contradiction"])
    worsened = sorted(k for k in shared if a[k]["absolute_error"] - b[k]["absolute_error"] >= 0.5 - TOLERANCE)
    ca, cb = coverage(arm), coverage(baseline)
    subgroups = {}
    names = sorted({name for r in arm for name in r["subgroups"]})
    names += ["targeted_heuristic", "untargeted_complement"]
    arm_id = arm[0]["arm_id"] if arm else "combined"
    for name in names:

        def selected(row, name=name):
            if name == "targeted_heuristic":
                return _targeted(row, arm_id)
            if name == "untargeted_complement":
                return not _targeted(row, arm_id)
            return name in row["subgroups"]

        comparison = _paired([r for r in arm if selected(r)], [r for r in baseline if selected(r)], draws)
        if comparison["n"]:
            effect = comparison["mae_difference"]
            comparison["observed_mae_regression_over_0_10"] = effect["estimate"] > 0.10 + TOLERANCE
            comparison["noninferiority_margin"] = 0.10
            comparison["noninferiority_status"] = (
                "insufficient_resampleable_families"
                if comparison["status"] != "estimated"
                else "supported"
                if effect["one_sided95_upper"] <= 0.10
                else "not_established"
            )
        else:
            comparison["observed_mae_regression_over_0_10"] = None
            comparison["noninferiority_status"] = "not_evaluable"
        subgroups[name] = comparison
    flagged = [name for name, value in subgroups.items() if value["observed_mae_regression_over_0_10"]]
    return {
        "new_critical_contradictions": {"count": len(new), "case_ids": new},
        "paired_error_worsened_at_least_0_5": {"count": len(worsened), "case_ids": worsened},
        "invalid_increase": ca["invalid"] - cb["invalid"],
        "nonvalid_score_increase": (ca["planned"] - ca["valid_scores"])
        - (cb["planned"] - cb["valid_scores"]),
        "subgroup_regression_flags": flagged,
        "subgroups": subgroups,
        "flagged": bool(
            new
            or worsened
            or flagged
            or ca["invalid"] > cb["invalid"]
            or ca["valid_scores"] < cb["valid_scores"]
        ),
        "note": "Absence of a flag does not establish no harm. Subgroup noninferiority is exploratory and unadjusted; sparse family groups can be inconclusive.",
        "target_note": "A targets controlled noise/JPEG variants; B controlled blur variants; C uses VizWiz only as an imperfect hard-view proxy; D and combined apply generally. These heuristic groups do not establish the visible issue in every image.",
    }


def controlled_diagnostics(rows):
    families = {}
    for row in rows:
        if row["stratum"] == "controlled" and "error" in row:
            families.setdefault(row["family_id"], []).append(row)
    details = []
    for family, members in sorted(families.items()):
        ordered = sorted(members, key=lambda r: r["severity"] or 0)
        if len(ordered) != 4 or [r["severity"] or 0 for r in ordered] != [0, 1, 2, 3]:
            continue
        scores = [r["student_score"] for r in ordered]
        refs = [r["reference_score"] for r in ordered]
        details.append(
            {
                "family_id": family,
                "operator": ordered[-1]["operator"],
                "case_ids": [r["case_id"] for r in ordered],
                "scores": scores,
                "reference_scores": refs,
                "nonincreasing": all(a >= b for a, b in itertools.pairwise(scores)),
                "strictly_decreasing": all(a > b for a, b in itertools.pairwise(scores)),
                "parent_to_severe_gap": scores[0] - scores[-1],
                "reference_parent_to_severe_gap": refs[0] - refs[-1],
            }
        )
    return {
        "complete_families": len(details),
        "families": details,
        "nonincreasing_families": sum(d["nonincreasing"] for d in details),
        "mean_parent_to_severe_gap": float(np.mean([d["parent_to_severe_gap"] for d in details]))
        if details
        else None,
        "note": "Technical severity ordering is diagnostic; severity is not perceptual ground truth.",
    }


def ensemble_diagnostics(rows, draws):
    result = {}
    for name, members in {"deepseek_luna": ("deepseek", "luna"), "all_four": ROUTES}.items():
        result[name] = {"members": list(members), "diagnostic_only": True, "tasks": {}}
        for task in TASKS:
            arm_rows = {}
            for arm in ARMS:
                indexed = {
                    route: {
                        r["case_id"]: r
                        for r in rows
                        if r["task"] == task and r["arm_id"] == arm and r["route"] == route and "error" in r
                    }
                    for route in members
                }
                common = set.intersection(*(set(rs) for rs in indexed.values()))
                values = []
                for key in sorted(common):
                    row = dict(indexed[members[0]][key])
                    row["route"] = name
                    _set_errors(row, float(np.mean([indexed[r][key]["student_score"] for r in members])))
                    values.append(row)
                arm_rows[arm] = values
            result[name]["tasks"][task] = {
                "arms": {
                    arm: {"metrics": metrics(rs), "case_ids": [r["case_id"] for r in rs]}
                    for arm, rs in arm_rows.items()
                },
                "arm_minus_baseline": {
                    arm: _paired(arm_rows[arm], arm_rows["baseline"], draws) for arm in ARMS[1:]
                },
            }
    return result


def summarize(rows, draws=DRAWS):
    result = {
        "schema_version": "prompt-ablation-results-v1",
        "reference_kind": "frozen_assistant_judgments",
        "human_gold": False,
        "training_ready": False,
        "tasks": {},
        "bootstrap": {
            "unit": "whole families within source strata",
            "draws": draws,
            "seed": SEED,
            "delta_direction": "arm minus baseline; negative error difference is improvement",
            "scope": "Conditional agreement with these frozen assistant references, not human accuracy.",
        },
        "multiplicity": {
            "exploratory_individual_tests": 32,
            "exploratory_ci": 0.95,
            "primary_combined_tests": 8,
            "primary_ci": 0.99375,
            "primary_quantiles": [0.003125, 0.996875],
            "note": "Eight model-by-task combined-arm MAE tests use Bonferroni intervals; interval distance and all subgroup or ensemble analyses are diagnostic.",
        },
    }
    for task in TASKS:
        result["tasks"][task] = {}
        for route in ROUTES:
            by_arm = {
                arm: [r for r in rows if r["task"] == task and r["route"] == route and r["arm_id"] == arm]
                for arm in ARMS
            }
            common = set.intersection(*({r["case_id"] for r in rs if "error" in r} for rs in by_arm.values()))
            route_result = {
                "arms": {},
                "arm_minus_baseline": {},
                "all_arms_shared_valid_case_ids": sorted(common),
            }
            for arm, values in by_arm.items():
                route_result["arms"][arm] = {
                    "coverage": coverage(values),
                    "available": metrics([r for r in values if "error" in r]),
                    "all_arms_shared_valid": metrics([r for r in values if r["case_id"] in common]),
                }
                if task == "technical_quality":
                    route_result["arms"][arm]["controlled_diagnostics"] = controlled_diagnostics(values)
                if arm != "baseline":
                    paired = _paired(values, by_arm["baseline"], draws)
                    paired.update(_interpret(paired, arm))
                    paired["guardrails"] = _guardrails(values, by_arm["baseline"], draws)
                    route_result["arm_minus_baseline"][arm] = paired
            result["tasks"][task][route] = route_result
    result["ensembles"] = ensemble_diagnostics(rows, draws)
    return result


def build_report(manifest, references, observations_by_route, draws=DRAWS):
    rows = comparison_rows(manifest["cases"], references, observations_by_route)
    return summarize(rows, draws), rows


def main():
    import runner

    manifest = json.loads((HERE / "manifest.json").read_text())
    if len(manifest["cases"]) != 50 or len({c["family_id"] for c in manifest["cases"]}) != 32:
        raise ValueError("Expected frozen cohort of 50 images in 32 families")
    observations, accounting = {}, {}
    for route in ROUTES:
        path = DATA / route / "images"
        if (path / "run.json").exists():
            run = runner.read_run(route, "images")
            accounting[route] = runner.verify(run)
            observations[route] = read_rows(path / "observations.jsonl")
        else:
            accounting[route] = {"status": "images_not_started", "planned": 300, "attempted": 0}
            observations[route] = []
    result, rows = build_report(manifest, read_rows(HERE / "references.jsonl"), observations)
    result["accounting"] = accounting
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (HERE / "scores-comparisons.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows
        )
    print(json.dumps({"report": str(HERE / "results.json"), "comparison_rows": len(rows)}))


if __name__ == "__main__":
    main()
