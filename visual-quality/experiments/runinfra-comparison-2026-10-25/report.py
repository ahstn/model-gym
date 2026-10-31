"""Compare RunInfra scores with frozen stage-A references without provider calls."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
ORIGINAL = HERE.parent / "prompt-calibration-2026-10-18"
DATA = PROJECT / "data/runinfra-comparison-2026-10-25"
BASELINE = PROJECT / "data/prompt-calibration-2026-10-18/a/v1"
spec = importlib.util.spec_from_file_location("frozen_prompt_report", ORIGINAL / "report.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
sys.path.insert(0, str(PROJECT / "src"))
from visual_quality.labels import parse_rating

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261018
STRATA = tuple(original.STRATA)
PROMPT_SHA256 = "1eb3ba4333b6651fa5ecaceb2305c9f28cc4516e851683e9de0caa871a9f1858"


def _index(rows: list[dict], key: str, expected: set[str], name: str) -> dict:
    return original.indexed(rows, key, expected, name)


def _score(observation: dict | None) -> float | None:
    if observation is None:
        return None
    rating = parse_rating(json.dumps({"task": "technical_quality", **observation["technical_quality"]}))
    return rating["score"] if rating["status"] == "ok" else None


def _check_identity(case: dict, observation: dict, candidate: bool) -> None:
    expected = {
        "image_id": case["image_id"],
        "view_id": case["view_id"],
        "source_sha256": case["sha256"],
        "view_sha256": case["view_sha256"],
    }
    if any(observation.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Observation identity differs from frozen case {case['case_id']}")
    if not candidate:
        return
    delivery = observation.get("view_comparison")
    delivered_hash = observation.get("delivered_sha256")
    if delivery not in {"identical_bytes", "resized"}:
        raise ValueError("Candidate requires an explicit view comparison")
    if not isinstance(delivered_hash, str) or len(delivered_hash) != 64:
        raise ValueError("Candidate requires a delivered byte hash")
    if (delivered_hash == case["view_sha256"]) != (delivery == "identical_bytes"):
        raise ValueError("View comparison contradicts the delivered byte hash")
    masks = observation.get("task_masks", {})
    if (
        not masks
        or masks.get("technical_quality") is not False
        or any(value is not False for value in masks.values())
    ):
        raise ValueError("All candidate training masks must remain false")
    aesthetics = observation.get("aesthetics", {})
    if aesthetics.get("status") != "not_requested" or aesthetics.get("score") is not None:
        raise ValueError("Aesthetics must remain not requested and null")
    if observation.get("review_status") != "pending_human_review":
        raise ValueError("Candidate must remain pending human review")
    provenance = observation.get("provenance", {})
    if (
        provenance.get("prompt_version") != "visual-rating-v1"
        or provenance.get("prompt_sha256") != PROMPT_SHA256
    ):
        raise ValueError("Candidate differs from the frozen v1 prompt")


def comparison_rows(
    cases: list[dict], references: dict, baseline: list[dict], candidate: list[dict]
) -> list[dict]:
    """Keep every planned case; never assign a numeric value to a missing score."""
    original.validate_cases(cases, "a")
    expected = {case["case_id"] for case in cases}
    if set(references) - expected:
        raise ValueError("Unexpected reference case")
    old = _index(baseline, "view_id", {case["view_id"] for case in cases}, "baseline view")
    new = _index(candidate, "case_id", expected, "candidate case")
    rows = []
    for case in cases:
        reference = references.get(case["case_id"], {})
        original.validate_reference(reference)
        point, interval = reference.get("score"), reference.get("interval")
        if point is None or interval is None:
            point, interval = None, None
        row = {
            key: case.get(key)
            for key in ("case_id", "image_id", "view_id", "family_id", "role", "operator", "severity")
        }
        row.update(
            stratum=original.stratum(case),
            reference_score=point,
            reference_lower=interval[0] if interval else None,
            reference_upper=interval[1] if interval else None,
        )
        for arm, observation in (
            ("baseline", old.get(case["view_id"])),
            ("runinfra", new.get(case["case_id"])),
        ):
            if observation is not None:
                _check_identity(case, observation, candidate=arm == "runinfra")
            score = _score(observation)
            row[f"{arm}_score"] = score
            row[f"{arm}_status"] = observation["technical_quality"]["status"] if observation else "missing"
            row[f"{arm}_distance"] = original.distance(score, interval)
            row[f"{arm}_critical"] = original.critical(score, interval)
        observation = new.get(case["case_id"])
        row["view_comparison"] = observation.get("view_comparison") if observation else "missing"
        row["delivered_sha256"] = observation.get("delivered_sha256") if observation else None
        row["same_view_pair"] = (
            row["view_comparison"] == "identical_bytes"
            and row["baseline_distance"] is not None
            and row["runinfra_distance"] is not None
        )
        if row["view_comparison"] != "identical_bytes":
            # The old reference describes the original view, not a resized delivery.
            row["runinfra_distance"], row["runinfra_critical"] = None, None
        row["score_delta_runinfra_minus_baseline"] = (
            row["runinfra_score"] - row["baseline_score"] if row["same_view_pair"] else None
        )
        row["interval_distance_improvement"] = (
            row["baseline_distance"] - row["runinfra_distance"] if row["same_view_pair"] else None
        )
        rows.append(row)
    return rows


def arm_metrics(rows: list[dict], arm: str) -> dict:
    valid = [row for row in rows if row[f"{arm}_distance"] is not None]
    distances = [row[f"{arm}_distance"] for row in valid]
    return {
        "case_count": len(rows),
        "scorable_outputs": sum(row[f"{arm}_score"] is not None for row in rows),
        "reference_comparisons": len(valid),
        "mean_interval_distance": original.mean(distances),
        "mae_from_assistant_point": original.mean(
            [abs(row[f"{arm}_score"] - row["reference_score"]) for row in valid]
        ),
        "interval_hits": sum(value == 0 for value in distances),
        "interval_hit_rate": original.mean([float(value == 0) for value in distances]),
        "large_interval_error_count": sum(value >= 1 - 1e-12 for value in distances),
        "critical_case_ids": [row["case_id"] for row in valid if row[f"{arm}_critical"]],
        "score_histogram": dict(sorted(Counter(str(row[f"{arm}_score"]) for row in valid).items())),
    }


def bootstrap_interval(rows: list[dict]) -> list[float] | None:
    """Resample retained family clusters; retain observed image weights by stratum.

    Incomplete families keep their available paired members together. Their cluster
    sizes can differ, so each draw uses the ratio of sampled sums to sampled counts.
    This estimates uncertainty for the observed subset, not the missing nine views.
    """
    if not rows:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(BOOTSTRAP_DRAWS)
    for source in STRATA:
        families = defaultdict(list)
        for row in rows:
            if row["stratum"] == source:
                families[row["family_id"]].append(row["interval_distance_improvement"])
        if not families:
            continue
        values = [families[key] for key in sorted(families)]
        sums = np.array([math.fsum(group) for group in values])
        counts = np.array([len(group) for group in values])
        sampled = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        draws += (sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)) * (sum(counts) / len(rows))
    return [0.0 if abs(value) <= 1e-12 else value for value in np.quantile(draws, [0.025, 0.975]).tolist()]


def _paired_metrics(rows: list[dict]) -> dict:
    baseline = arm_metrics(rows, "baseline")
    candidate = arm_metrics(rows, "runinfra")
    return {
        "pairs": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "baseline": baseline,
        "runinfra": candidate,
        "mean_interval_distance_improvement": original.mean(
            [row["interval_distance_improvement"] for row in rows]
        ),
        "mean_score_delta_runinfra_minus_baseline": original.mean(
            [row["score_delta_runinfra_minus_baseline"] for row in rows]
        ),
        "paired_95_percent_family_bootstrap_interval": bootstrap_interval(rows),
        "new_critical_case_ids": [
            row["case_id"] for row in rows if row["runinfra_critical"] and not row["baseline_critical"]
        ],
    }


def build_report(
    cases: list[dict], references: dict, baseline: list[dict], candidate: list[dict]
) -> tuple[dict, list[dict]]:
    """Pure comparison entry point; inputs may represent an unfinished run."""
    rows = comparison_rows(cases, references, baseline, candidate)
    paired = [row for row in rows if row["same_view_pair"]]
    primary = _paired_metrics(paired)
    strata = {
        source: _paired_metrics([row for row in paired if row["stratum"] == source]) for source in STRATA
    }
    planned_families, paired_families = (
        Counter(row["family_id"] for row in rows),
        Counter(row["family_id"] for row in paired),
    )
    complete = {family for family, count in planned_families.items() if paired_families[family] == count}
    complete_rows = [row for row in paired if row["family_id"] in complete]
    effect = primary["mean_interval_distance_improvement"]
    interval = primary["paired_95_percent_family_bootstrap_interval"]
    diagnostic = {
        "mean_improvement_at_least_0_15": effect >= 0.15 - 1e-12 if effect is not None else None,
        "lower_95_percent_bound_above_zero": interval[0] > 0 if interval is not None else None,
        "no_new_critical_contradictions": not primary["new_critical_case_ids"] if paired else None,
        "no_observed_stratum_worsens_more_than_0_10": all(
            value["mean_interval_distance_improvement"] >= -0.10 - 1e-12
            for value in strata.values()
            if value["pairs"]
        )
        if paired
        else None,
    }
    all_50 = len(paired) == 50
    decision = "inconclusive_incomplete_frozen_view_comparison"
    if all_50:
        decision = (
            "improved_against_assistant_references"
            if all(diagnostic.values())
            else "did_not_meet_improvement_rule"
        )
        if interval[0] <= 0 <= interval[1]:
            decision = "inconclusive_bootstrap_interval_includes_zero"
    summary = {
        "schema_version": "runinfra-comparison-report-v1",
        "reference_kind": "frozen_assistant_judgments_not_human_ground_truth",
        "prompt_comparison": "visual-rating-v1 versus visual-rating-v1",
        "planned_cases": len(rows),
        "same_frozen_view_pairs": len(paired),
        "baseline_all_50": arm_metrics(rows, "baseline"),
        "primary_same_view_subset": primary,
        "strata_same_view_subset": strata,
        "complete_original_families_sensitivity": _paired_metrics(complete_rows),
        "incomplete_family_ids": sorted(
            family for family in planned_families if 0 < paired_families[family] < planned_families[family]
        ),
        "entirely_missing_family_ids": sorted(
            family for family in planned_families if paired_families[family] == 0
        ),
        "missing_candidate_case_ids": [row["case_id"] for row in rows if row["runinfra_status"] == "missing"],
        "unscorable_candidate_case_ids": [
            row["case_id"] for row in rows if row["runinfra_status"] == "unscorable"
        ],
        "changed_view_case_ids": [row["case_id"] for row in rows if row["view_comparison"] == "resized"],
        "not_comparable_case_ids": [row["case_id"] for row in rows if not row["same_view_pair"]],
        "all_50_valid_scorable_same_view_pairs": all_50,
        "diagnostic_thresholds_on_observed_subset": diagnostic,
        "decision": decision,
        "explanation_support": {"status": "not_assessed", "unsupported_count": None},
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "method": "Stratified retained-family cluster resampling with fixed observed stratum image weights",
            "partial_family_handling": "Keep all observed paired members together; ratio means allow unequal cluster sizes",
            "uncertainty_scope": "Conditional on selected views and frozen assistant references; missing cases are not imputed",
        },
        "limitations": [
            "A request-size-selected subset does not estimate full-panel or population accuracy.",
            "The historical baseline and RunInfra run also differ in time and deployment.",
            "Changed views are excluded from all reference error metrics.",
            "Explanation support requires a separate image audit.",
            "Assistant intervals do not establish human agreement or measurement precision.",
        ],
        "training_ready": False,
        "review_status": "pending_human_review",
    }
    return summary, rows


def load_saved_report(data: Path = DATA) -> tuple[dict, list[dict]]:
    manifest = original.read_json(ORIGINAL / "manifest.json")
    cases = manifest["stages"]["a"]["cases"]
    references, freeze = original.load_references("a", {case["case_id"] for case in cases})
    baseline = original.read_rows(BASELINE / "observations.jsonl")
    candidate = original.read_rows(data / "observations.jsonl")
    summary, rows = build_report(cases, references, baseline, candidate)
    summary["reference_freeze"] = freeze
    paths = [
        ORIGINAL / "manifest.json",
        ORIGINAL / "references/a.jsonl",
        ORIGINAL / "references/a-freeze.json",
        BASELINE / "observations.jsonl",
        Path(__file__).resolve(),
        data / "run.json",
        data / "report.json",
        data / "observations.jsonl",
    ]
    summary["input_sha256"] = {
        str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path): original.sha(path)
        for path in paths
    }
    if (data / "report.json").exists():
        run_report = original.read_json(data / "report.json")
        summary["run_status"] = {
            key: run_report.get(key)
            for key in (
                "status",
                "attempted",
                "valid_observations",
                "scored",
                "known_cost_usd",
                "unpriced_attempts",
                "blocked_case_ids",
            )
        }
    summary["integrity_scope"] = (
        "Frozen reference hashes, unique case joins, source/view hashes, and disabled training masks; run verification remains separate"
    )
    return summary, rows


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case_id"])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--write", action="store_true", help="Write this experiment's result summary and case CSV"
    )
    args = parser.parse_args()
    summary, rows = load_saved_report(args.data_dir.resolve())
    if args.write:
        output = HERE / "results"
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (output / "cases.csv").write_text(csv_text(rows))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
