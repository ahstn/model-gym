"""Describe a fresh RunInfra sample against frozen assistant judgments.

Only strictly valid observations enter score metrics. Missing and rejected replies
remain null. This report neither repairs provider output nor promotes labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data/runinfra-string-trial-2026-10-26/images"
sys.path.insert(0, str(PROJECT / "src"))
from visual_quality.labels import parse_rating

STRATA = ("controlled", "vizwiz", "standalone_megalith")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261026
WIRE_SCORES = frozenset(f"{number // 10}.{number % 10}0" for number in range(10, 51))
METRICS = {
    "mean_interval_distance": "interval_distance",
    "mae_from_assistant_point": "absolute_error",
    "mean_score_bias": "signed_error",
    "interval_hit_rate": "interval_hit",
    "large_interval_error_rate": "large_interval_error",
    "critical_error_rate": "critical_error",
}


def _indexed(records: list[dict], expected: set[str], name: str) -> dict[str, dict]:
    result = {}
    for record in records:
        case_id = record.get("case_id")
        if case_id not in expected:
            raise ValueError(f"Unexpected {name} case: {case_id}")
        if case_id in result:
            raise ValueError(f"Duplicate {name} case: {case_id}")
        result[case_id] = record
    return result


def _validate_reference(case: dict, reference: dict) -> None:
    if reference.get("view_sha256") != case["view_sha256"]:
        raise ValueError(f"Reference view identity differs: {case['case_id']}")
    if not reference.get("reviewer_id") or not reference.get("evidence"):
        raise ValueError("Reference requires reviewer and evidence")
    score, interval = reference.get("score"), reference.get("interval")
    if score is None:
        if interval is not None or not reference.get("unscorable_reason"):
            raise ValueError("Null reference requires a null interval and unscorable reason")
        return
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("Reference score must be numeric")
    number = Decimal(str(score))
    if not number.is_finite() or not 1 <= number <= 5 or number % Decimal("0.1"):
        raise ValueError("Reference score must be finite, from 1 to 5, in tenth steps")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Reference requires a two-endpoint interval")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in interval):
        raise ValueError("Reference interval must be numeric")
    if (
        not all(math.isfinite(value) for value in interval)
        or not 1 <= interval[0] <= score <= interval[1] <= 5
    ):
        raise ValueError("Reference interval must contain its point and stay within 1 to 5")


def _validated_score(case: dict, observation: dict, run: dict | None) -> float | None:
    identity = {key: case[key] for key in ("case_id", "image_id", "view_id", "source_sha256", "view_sha256")}
    identity["delivered_sha256"] = case["view_sha256"]
    if any(observation.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Observation identity differs: {case['case_id']}")
    if observation.get("adapter_version") != "string-score-to-decimal-v1":
        raise ValueError("Observation must name the frozen string-score adapter")
    if observation.get("task_masks") != {
        "technical_quality": False,
        "aesthetics": False,
        "photo_domain": False,
    }:
        raise ValueError("All three task masks must remain false")
    if any(value is not False for value in observation["task_masks"].values()):
        raise ValueError("Task masks must be boolean false")
    if observation.get("review_status") != "pending_human_review":
        raise ValueError("Observation must remain pending human review")
    if observation.get("aesthetics") != {"status": "not_requested", "score": None}:
        raise ValueError("Aesthetics must remain not requested and null")
    if observation.get("photo_domain_eligible") != case.get("photo_domain_eligible"):
        raise ValueError("Observation eligibility differs from frozen case")
    rating = parse_rating(json.dumps(observation["technical_quality"], allow_nan=False))
    if "wire_score" not in observation:
        raise ValueError("Observation must retain the wire score")
    wire = observation["wire_score"]
    if rating["status"] == "ok":
        if not isinstance(wire, str) or wire not in WIRE_SCORES:
            raise ValueError("Wire score must be an exact permitted two-decimal string")
        if Decimal(wire) != Decimal(str(rating["score"])):
            raise ValueError("Wire score and numeric score differ")
    elif wire is not None:
        raise ValueError("Unscorable wire score must be null")
    if run is not None:
        provenance = observation.get("provenance", {})
        expected = {
            "provider": "runinfra",
            "run_id": run["run_id"],
            "attempt_id": case["case_id"],
            "prompt_version": run["prompt"]["version"],
            "prompt_sha256": run["prompt"]["sha256"],
            "request_sha256": case["request_sha256"],
            "model_requested": run["settings"]["model"],
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Observation provenance differs: {case['case_id']}")
    return rating["score"]


def comparison_rows(
    cases: list[dict],
    references: list[dict],
    observations: list[dict],
    attempts: list[dict],
    run: dict | None = None,
) -> list[dict]:
    """Retain every planned case; only validated, scored observations get errors."""
    expected = {case["case_id"] for case in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate manifest case")
    refs = _indexed(references, expected, "reference")
    if set(refs) != expected:
        raise ValueError("Every planned case needs a frozen assistant reference")
    labels = _indexed(observations, expected, "observation")
    tried = _indexed(attempts, expected, "attempt")
    families = defaultdict(set)
    rows = []
    for case in cases:
        if case.get("stratum") not in STRATA or not case.get("family_id"):
            raise ValueError("Manifest case requires a supported stratum and family")
        families[case["family_id"]].add(case["stratum"])
        reference = refs[case["case_id"]]
        _validate_reference(case, reference)
        observation = labels.get(case["case_id"])
        attempt = tried.get(case["case_id"])
        status = attempt["status"] if attempt else "missing"
        if status not in {"missing", "valid", "invalid", "failed", "unknown"}:
            raise ValueError("Unexpected attempt status")
        if (observation is not None) != (status == "valid"):
            raise ValueError("Valid attempts and observations must match exactly")
        score = _validated_score(case, observation, run) if observation else None
        point, interval = reference["score"], reference["interval"]
        row = {
            key: case.get(key)
            for key in (
                "case_id",
                "image_id",
                "view_id",
                "family_id",
                "stratum",
                "role",
                "operator",
                "severity",
                "view_sha256",
            )
        }
        row.update(
            reference_score=point,
            reference_lower=interval[0] if interval is not None else None,
            reference_upper=interval[1] if interval is not None else None,
            reviewer_id=reference["reviewer_id"],
            attempt_status=status,
            rejection_reason=attempt.get("reason") if attempt else None,
            strict_valid=observation is not None,
            wire_score=observation.get("wire_score") if observation else None,
            wire_score_valid=attempt.get("wire_score_valid") if attempt else None,
            rating_schema_valid=attempt.get("rating_schema_valid") if attempt else None,
            model_status=observation["technical_quality"]["status"] if observation else None,
            model_score=score,
            cost_usd=attempt.get("cost_usd") if attempt else None,
            interval_distance=None,
            absolute_error=None,
            signed_error=None,
            interval_hit=None,
            large_interval_error=None,
            critical_error=None,
        )
        if score is not None and interval is not None:
            distance = max(interval[0] - score, 0.0, score - interval[1])
            row.update(
                interval_distance=distance,
                absolute_error=abs(score - point),
                signed_error=score - point,
                interval_hit=distance <= 1e-12,
                large_interval_error=distance >= 1 - 1e-12,
                critical_error=(score >= 4 and interval[1] <= 2) or (score <= 2 and interval[0] >= 4),
            )
        rows.append(row)
    if any(len(strata) != 1 for strata in families.values()):
        raise ValueError("A family cannot cross strata")
    return rows


def _mean(values: list) -> float | None:
    return math.fsum(values) / len(values) if values else None


def bootstrap_intervals(rows: list[dict]) -> dict[str, list[float] | None]:
    """Resample entire families within strata with fixed observed image weights."""
    usable = [row for row in rows if row["interval_distance"] is not None]
    if not usable:
        return {metric: None for metric in METRICS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros((BOOTSTRAP_DRAWS, len(METRICS)))
    for stratum in STRATA:
        families = defaultdict(list)
        for row in usable:
            if row["stratum"] == stratum:
                families[row["family_id"]].append(row)
        if not families:
            continue
        groups = [families[family] for family in sorted(families)]
        counts = np.array([len(group) for group in groups])
        sums = np.array(
            [[math.fsum(row[key] for row in group) for key in METRICS.values()] for group in groups]
        )
        sampled = rng.integers(0, len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))
        numerator = sums[sampled].sum(axis=1)
        denominator = counts[sampled].sum(axis=1, keepdims=True)
        draws += numerator / denominator * (sum(counts) / len(usable))
    bounds = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        name: [0.0 if abs(value) < 1e-12 else float(value) for value in bounds[:, number]]
        for number, name in enumerate(METRICS)
    }


def agreement_metrics(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["interval_distance"] is not None]
    metrics = {name: _mean([row[key] for row in usable]) for name, key in METRICS.items()}
    return {
        "planned_cases": len(rows),
        "valid_observations": sum(row["strict_valid"] for row in rows),
        "scorable_outputs": sum(row["model_score"] is not None for row in rows),
        "reference_comparisons": len(usable),
        "compared_families": len({row["family_id"] for row in usable}),
        **metrics,
        "interval_hits": sum(row["interval_hit"] for row in usable),
        "large_interval_error_count": sum(row["large_interval_error"] for row in usable),
        "critical_case_ids": [row["case_id"] for row in usable if row["critical_error"]],
        "model_score_histogram": dict(
            sorted(
                Counter(f"{row['model_score']:.2f}" for row in rows if row["model_score"] is not None).items()
            )
        ),
        "reference_score_histogram": dict(
            sorted(
                Counter(
                    f"{row['reference_score']:.2f}" for row in rows if row["reference_score"] is not None
                ).items()
            )
        ),
        "percentile_95_family_bootstrap_intervals": bootstrap_intervals(rows),
    }


def severity_ordering(rows: list[dict]) -> dict:
    """Controlled transformations provide ordering diagnostics, not human labels."""
    groups = defaultdict(list)
    for row in rows:
        if row["stratum"] == "controlled":
            groups[row["family_id"]].append(row)
    pairs = []
    for family in sorted(groups):
        ordered = sorted(
            groups[family], key=lambda row: 0 if row["role"] == "source_parent" else row["severity"]
        )
        for left, right in pairwise(ordered):
            scored = left["model_score"] is not None and right["model_score"] is not None
            delta = right["model_score"] - left["model_score"] if scored else None
            pairs.append(
                {
                    "family_id": family,
                    "operator": right["operator"],
                    "less_degraded_case_id": left["case_id"],
                    "more_degraded_case_id": right["case_id"],
                    "includes_source_parent": left["role"] == "source_parent",
                    "more_minus_less_score": delta,
                    "reversal": delta > 1e-12 if scored else None,
                    "tie": abs(delta) <= 1e-12 if scored else None,
                }
            )
    valid = [pair for pair in pairs if pair["more_minus_less_score"] is not None]
    return {
        "interpretation": "Diagnostic only: transform severity is not human ground truth and does not fix an expected score.",
        "planned_adjacent_pairs": len(pairs),
        "scored_adjacent_pairs": len(valid),
        "nonincreasing_pairs": sum(not pair["reversal"] for pair in valid),
        "ties": sum(pair["tie"] for pair in valid),
        "reversals": [pair for pair in valid if pair["reversal"]],
        "pairs": pairs,
    }


def _format_metrics(rows: list[dict]) -> dict:
    attempted = [row for row in rows if row["attempt_status"] != "missing"]
    valid = sum(row["strict_valid"] for row in rows)
    diagnostics = {}
    for field in ("wire_score_valid", "rating_schema_valid"):
        known = [row[field] for row in attempted if row[field] is not None]
        if any(type(value) is not bool for value in known):
            raise ValueError("Conformance diagnostics must be boolean or null")
        diagnostics[field] = {
            "assessed": len(known),
            "passed": sum(known),
            "unknown": len(attempted) - len(known),
            "rate_among_assessed": _mean(known),
        }
    return {
        "planned_cases": len(rows),
        "attempted_cases": len(attempted),
        "strict_valid_observations": valid,
        "strict_valid_rate_among_attempts": valid / len(attempted) if attempted else None,
        "strict_valid_rate_among_planned_cases": valid / len(rows) if rows else None,
        "all_200_exact_wire_and_full_contract": len(rows) == 200 and valid == 200,
        "statuses": dict(sorted(Counter(row["attempt_status"] for row in rows).items())),
        "rejection_reasons": dict(
            sorted(
                Counter(
                    row["rejection_reason"] or "unspecified"
                    for row in rows
                    if row["attempt_status"] in {"invalid", "failed", "unknown"}
                ).items()
            )
        ),
        "component_conformance": diagnostics,
    }


def build_report(
    cases: list[dict],
    references: list[dict],
    observations: list[dict],
    attempts: list[dict],
    run: dict | None = None,
) -> tuple[dict, list[dict]]:
    rows = comparison_rows(cases, references, observations, attempts, run)
    planned = Counter(row["family_id"] for row in rows)
    compared = Counter(row["family_id"] for row in rows if row["interval_distance"] is not None)
    complete = {family for family, count in planned.items() if compared[family] == count}
    summary = {
        "schema_version": "runinfra-string-trial-report-v1",
        "reference_kind": "frozen_assistant_judgments_not_human_ground_truth",
        "study_design": "Fresh descriptive 200-image sample; no paired historical baseline.",
        "historical_comparison": "Different images and delivered views prevent a paired improvement or causal model comparison.",
        "inference_limit": "Bootstrap intervals describe sample variability conditional on the frozen assistant judgments; they do not measure reference uncertainty or human accuracy.",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "family_within_stratum",
            "weights": "observed_compared_image_count_by_stratum",
            "estimator": "sampled_sums_over_sampled_counts_within_stratum",
        },
        "format_conformance": _format_metrics(rows),
        "agreement": agreement_metrics(rows),
        "by_stratum": {
            stratum: agreement_metrics([row for row in rows if row["stratum"] == stratum])
            for stratum in STRATA
        },
        "controlled_source_parents": agreement_metrics(
            [row for row in rows if row["stratum"] == "controlled" and row["role"] == "source_parent"]
        ),
        "controlled_variants": agreement_metrics(
            [row for row in rows if row["stratum"] == "controlled" and row["role"] == "variant"]
        ),
        "complete_families_sensitivity": agreement_metrics(
            [row for row in rows if row["family_id"] in complete]
        ),
        "incomplete_family_ids": sorted(set(planned) - complete),
        "severity_ordering": severity_ordering(rows),
        "missing_case_ids": [row["case_id"] for row in rows if row["attempt_status"] == "missing"],
        "invalid_case_ids": [row["case_id"] for row in rows if row["attempt_status"] == "invalid"],
        "unscorable_case_ids": [row["case_id"] for row in rows if row["model_status"] == "unscorable"],
        "null_reference_case_ids": [row["case_id"] for row in rows if row["reference_score"] is None],
        "explanation_support": {
            "audited": False,
            "unsupported_count": None,
            "note": "Score agreement does not establish that the model evidence is visually supported.",
        },
        "training_ready": False,
    }
    return summary, rows


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_inputs(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    """Verify preserved replies, request hashes and reference pins before analysis."""
    if run_dir.name != "images":
        raise ValueError("The analysis requires the images run, not conformance probes")
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("string_trial_runner", HERE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.DATA = run_dir.parent
    run = runner.read_run("images")
    operational = runner.verify(run)
    for raw_path, expected in run["inputs"].items():
        path = Path(raw_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {path}")
    references_path = HERE / "references.jsonl"
    freeze_path = HERE / "references-freeze.json"
    manifest_path = HERE / "manifest.json"
    if any(str(path) not in run["inputs"] for path in (references_path, freeze_path, manifest_path)):
        raise ValueError("Run must pin the manifest, references and pre-call reference freeze")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["cases"]) != 200 or len(run["cases"]) != 200:
        raise ValueError("This trial requires exactly 200 frozen cases")
    manifest_cases = {case["case_id"]: case for case in manifest["cases"]}
    for case in run["cases"]:
        original = manifest_cases.get(case["case_id"])
        if original is None or any(case.get(key) != value for key, value in original.items()):
            raise ValueError("Run cases differ from frozen manifest")
    observations_path = run_dir / "observations.jsonl"
    observations = _read_jsonl(observations_path) if observations_path.exists() else []
    return run, _read_jsonl(references_path), observations, operational["attempts"], operational


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DATA)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run, refs, observations, attempts, operational = load_inputs(args.run_dir)
    summary, rows = build_report(run["cases"], refs, observations, attempts, run)
    summary["run_id"] = run["run_id"]
    summary["operational"] = {
        key: operational.get(key)
        for key in ("status", "known_cost_usd", "total_known_cost_usd", "unpriced_attempts")
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        destination = HERE / "results"
        destination.mkdir(exist_ok=True)
        (destination / "report.json").write_text(encoded)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (destination / "per-image.csv").write_text(stream.getvalue())
    print(encoded, end="")


if __name__ == "__main__":
    main()
