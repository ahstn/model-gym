"""Compare frozen prompt runs with assistant references; never call a provider."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
REPO = PROJECT.parent
DATA = PROJECT / "data/prompt-calibration-2026-10-18"
STRATA = {"controlled": (6, 4), "vizwiz": (14, 1), "standalone_megalith": (12, 1)}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261018
MIN_IMPROVEMENT = 0.15
MAX_STRATUM_WORSENING = 0.10


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def indexed(rows: list[dict], key: str, expected: set[str], name: str) -> dict:
    result = {}
    for row in rows:
        value = row.get(key)
        if value not in expected or value in result:
            raise ValueError(f"Duplicate or unexpected {name}: {value}")
        result[value] = row
    return result


def stratum(case: dict) -> str:
    return "controlled" if case["role"] in {"source_parent", "variant"} else case["role"]


def validate_cases(cases: list[dict], stage: str) -> None:
    expected = {f"{stage.upper()}{number:02d}" for number in range(1, 51)}
    if set(indexed(cases, "case_id", expected, "case ID")) != expected:
        raise ValueError(f"Stage {stage} must contain exactly 50 cases")
    for key in ("image_id", "view_id"):
        if len({case[key] for case in cases}) != 50:
            raise ValueError(f"Stage {stage} has duplicate {key}")
    groups = defaultdict(list)
    family_strata = {}
    for case in cases:
        family = case["family_id"]
        if not isinstance(family, str) or not family:
            raise ValueError("A source family requires a nonempty ID")
        if family_strata.setdefault(family, stratum(case)) != stratum(case):
            raise ValueError("A source family cannot occur in different strata")
        groups[(stratum(case), case["family_id"])].append(case)
    if Counter(group[0] for group in groups) != {key: value[0] for key, value in STRATA.items()}:
        raise ValueError("Unexpected source strata or family allocation")
    for (source, _), family in groups.items():
        if len(family) != STRATA[source][1]:
            raise ValueError("Unexpected family size")
        if source == "controlled":
            if Counter(case["role"] for case in family) != {"source_parent": 1, "variant": 3}:
                raise ValueError("A controlled family requires one parent and three variants")
            if {case.get("severity") for case in family if case["role"] == "variant"} != {1, 2, 3}:
                raise ValueError("Controlled variants require severity levels 1, 2, 3")
            if len({case.get("operator") for case in family}) != 1:
                raise ValueError("A controlled family must use one operator")
    if Counter(case["operator"] for case in cases if case["role"] == "source_parent") != {
        "gaussian_blur": 2,
        "gaussian_noise": 2,
        "jpeg_compression": 2,
    }:
        raise ValueError("Expected two controlled families per frozen operator")


def valid_score(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 5


def validate_reference(reference: dict) -> None:
    score, interval = reference.get("score"), reference.get("interval")
    if score is None or interval is None:
        return
    if (
        not valid_score(score)
        or not isinstance(interval, list)
        or len(interval) != 2
        or not all(valid_score(value) for value in interval)
        or not interval[0] <= score <= interval[1]
    ):
        raise ValueError(f"Invalid assistant reference: {reference.get('case_id')}")


def reference_sources(freeze: dict) -> dict[Path, str]:
    sources = freeze.get("source_files_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Reference freeze must pin its source files")
    paths = {}
    for name, digest in sources.items():
        path = (PROJECT / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(PROJECT.resolve()):
            raise ValueError("Reference source path must remain within visual-quality")
        if path in paths:
            raise ValueError("Reference freeze contains duplicate source paths")
        paths[path] = digest
    return paths


def load_references(stage: str, expected: set[str]) -> tuple[dict, dict]:
    merged = HERE / "references" / f"{stage}.jsonl"
    freeze_path = HERE / "references" / f"{stage}-freeze.json"
    rows = read_rows(merged)
    if not freeze_path.exists():
        if rows:
            raise ValueError(f"Stage {stage} references require their frozen hash record")
        return {}, {"status": "missing_reference_freeze"}
    freeze = read_json(freeze_path)
    if (
        freeze.get("stage") != stage
        or freeze.get("human_gold") is not False
        or not merged.is_file()
        or freeze.get("merged_sha256") != sha(merged)
    ):
        raise ValueError(f"Stage {stage} merged references differ from their frozen hash")
    sources = reference_sources(freeze)
    if any(not path.is_file() or sha(path) != digest for path, digest in sources.items()):
        raise ValueError(f"Stage {stage} reference source files differ from their frozen hashes")
    return indexed(rows, "case_id", expected, "reference"), {
        "status": "verified",
        "frozen_at": freeze.get("frozen_at"),
        "source_files": len(sources),
    }


def distance(score: float | None, interval: list[float] | None) -> float | None:
    if score is None or interval is None:
        return None
    return max(interval[0] - score, 0.0, score - interval[1])


def critical(score: float | None, interval: list[float] | None) -> bool | None:
    if score is None or interval is None:
        return None
    return (score >= 4 and interval[1] <= 2) or (score <= 2 and interval[0] >= 4)


def mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def bootstrap_interval(cases: list[dict], deltas: dict[str, float]) -> list[float]:
    """Resample complete paired families within strata, with fixed image weights."""
    if len(cases) != 50 or set(deltas) != {case["case_id"] for case in cases}:
        raise ValueError("The bootstrap requires all 50 paired image differences")
    groups = {source: defaultdict(list) for source in STRATA}
    for case in cases:
        groups[stratum(case)][case["family_id"]].append(deltas[case["case_id"]])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(BOOTSTRAP_DRAWS)
    for source, (count, size) in STRATA.items():
        families = groups[source]
        if len(families) != count or any(len(values) != size for values in families.values()):
            raise ValueError("Bootstrap families differ from the fixed allocation")
        # Each family has equal size inside its stratum. All its images travel together.
        sums = np.array([math.fsum(families[key]) for key in sorted(families)])
        sampled = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
        draws += sums[sampled].sum(axis=1) / 50
    bounds = np.quantile(draws, [0.025, 0.975]).tolist()
    return [0.0 if abs(value) <= 1e-12 else value for value in bounds]


def paired_result(cases: list[dict], rows: list[dict]) -> dict:
    expected = {case["case_id"] for case in cases}
    if set(indexed(rows, "case_id", expected, "paired case")) != expected:
        raise ValueError("Paired analysis must retain every planned case")
    paired = [row for row in rows if row["v1_distance"] is not None and row["v2_distance"] is not None]
    deltas = {row["case_id"]: row["v1_distance"] - row["v2_distance"] for row in paired}
    effect = mean(list(deltas.values()))
    complete = len(paired) == 50
    interval = bootstrap_interval(cases, deltas) if complete else None
    strata = {}
    for source in STRATA:
        selected = [row for row in paired if row["stratum"] == source]
        strata[source] = {
            "valid_pairs": len(selected),
            "v1_mean_distance": mean([row["v1_distance"] for row in selected]),
            "v2_mean_distance": mean([row["v2_distance"] for row in selected]),
            "mean_improvement": mean([deltas[row["case_id"]] for row in selected]),
        }
    new_critical = [row["case_id"] for row in paired if row["v2_critical"] and not row["v1_critical"]]
    gates = {
        "all_50_valid_scorable_pairs": complete,
        "mean_improvement_at_least_0_15": effect >= MIN_IMPROVEMENT - 1e-12 if complete else None,
        "lower_95_percent_bound_above_zero": interval[0] > 0 if interval is not None else None,
        "no_new_critical_contradictions": not new_critical if complete else None,
        "no_stratum_worsens_more_than_0_10": all(
            value["mean_improvement"] >= -MAX_STRATUM_WORSENING - 1e-12 for value in strata.values()
        )
        if complete
        else None,
    }
    baseline = mean([row["v1_distance"] for row in paired])
    if all(value is True for value in gates.values()):
        decision, reason = "improved_against_assistant_references", "All fixed gates passed"
    elif not complete:
        decision, reason = "inconclusive", "Missing or unscorable pairs or assistant references"
    elif baseline < MIN_IMPROVEMENT - 1e-12:
        decision, reason = "inconclusive", "Baseline distance is below the required reduction"
    elif interval[0] <= 0 <= interval[1]:
        decision, reason = "inconclusive", "The bootstrap interval includes zero"
    else:
        decision, reason = "did_not_meet_improvement_rule", "At least one fixed gate failed"
    return {
        "decision": decision,
        "reason": reason,
        "planned_pairs": 50,
        "valid_pairs": len(paired),
        "partial_estimates_are_diagnostic": not complete,
        "missing_pair_case_ids": [row["case_id"] for row in rows if row["case_id"] not in deltas],
        "mean_improvement": effect,
        "paired_95_percent_bootstrap_interval": interval,
        "new_critical_case_ids": new_critical,
        "strata": strata,
        "gates": gates,
    }


def ordering(rows: list[dict], arm: str) -> dict:
    families = defaultdict(list)
    for row in rows:
        if row["stratum"] == "controlled":
            families[row["family_id"]].append(row)
    result = {"severity_adjacent": [], "parent_to_variant": []}
    for family in families.values():
        parent = next(row for row in family if row["role"] == "source_parent")
        variants = sorted(
            (row for row in family if row["role"] == "variant"), key=lambda row: row["severity"]
        )
        for kind, edges in (
            ("severity_adjacent", pairwise(variants)),
            ("parent_to_variant", ((parent, variant) for variant in variants)),
        ):
            for first, second in edges:
                old, new = first[f"{arm}_score"], second[f"{arm}_score"]
                outcome = (
                    "missing"
                    if old is None or new is None
                    else ("tie" if old == new else "nonincreasing" if old > new else "violation")
                )
                result[kind].append(
                    {
                        "from": first["case_id"],
                        "to": second["case_id"],
                        "operator": first["operator"],
                        "outcome": outcome,
                    }
                )
    return {
        kind: {"counts": dict(Counter(edge["outcome"] for edge in edges)), "edges": edges}
        for kind, edges in result.items()
    }


def arm_metrics(rows: list[dict], arm: str) -> dict:
    valid = [row for row in rows if row[f"{arm}_distance"] is not None]
    widths = [
        row["reference_upper"] - row["reference_lower"] for row in rows if row["reference_lower"] is not None
    ]
    judgments = [row for row in rows if row[f"{arm}_evidence_supported"] is not None]
    return {
        "planned": len(rows),
        "scorable_outputs": sum(row[f"{arm}_score"] is not None for row in rows),
        "valid_reference_comparisons": len(valid),
        "mean_interval_distance": mean([row[f"{arm}_distance"] for row in valid]),
        "mae_from_assistant_point": mean(
            [abs(row[f"{arm}_score"] - row["reference_score"]) for row in valid]
        ),
        "interval_hit_rate": mean([float(row[f"{arm}_distance"] == 0) for row in valid]),
        "mean_reference_interval_width": mean(widths),
        "large_interval_error_count": sum(row[f"{arm}_distance"] >= 1 - 1e-12 for row in valid),
        "large_interval_error_rate": mean([float(row[f"{arm}_distance"] >= 1 - 1e-12) for row in valid]),
        "critical_case_ids": [row["case_id"] for row in valid if row[f"{arm}_critical"]],
        "evidence_review": {
            "assessed": len(judgments),
            "supported": sum(row[f"{arm}_evidence_supported"] is True for row in judgments),
            "unsupported": sum(row[f"{arm}_evidence_supported"] is False for row in judgments),
            "issues": dict(Counter(issue for row in rows for issue in row[f"{arm}_evidence_issues"])),
        },
        "ordering_diagnostic": ordering(rows, arm),
    }


def load_run(path: Path, views: Path, cases: list[dict]) -> tuple[dict, dict]:
    sys.path.insert(0, str(PROJECT / "src"))
    from visual_quality.labelling import verify_labels

    required = [path / filename for filename in ("run.json", "report.json", "observations.jsonl")]
    if not all(item.exists() for item in required):
        return {}, {
            "status": "missing_run_files",
            "missing": [str(item.relative_to(REPO)) for item in required if not item.exists()],
            "total_cost_usd": None,
            "known_cost_usd": None,
            "unpriced_attempts": None,
        }
    verification = verify_labels(views, path)
    run, report = read_json(path / "run.json"), read_json(path / "report.json")
    expected = {case["view_id"] for case in cases}
    observations = indexed(read_rows(path / "observations.jsonl"), "view_id", expected, "observation")
    if (
        report["run_id"] != run["run_id"]
        or report["selected"] != 50
        or report["valid_observations"] != len(observations)
        or report["training_ready"] is not False
    ):
        raise ValueError("Run report does not match its verified observations")
    attempts = list(indexed(report["attempts"], "view_id", expected, "attempt").values())
    costs = [Decimal(attempt["cost_usd"]) for attempt in attempts if attempt["cost_usd"] is not None]
    if any(not cost.is_finite() or cost < 0 for cost in costs):
        raise ValueError("Invalid reported cost")
    known = sum(costs, Decimal(0))
    unpriced = len(attempts) - len(costs)
    if (
        known != Decimal(report["known_cost_usd"])
        or unpriced != report["unpriced_attempts"]
        or len(attempts) != report["attempted"]
    ):
        raise ValueError("Run cost or attempt counts are inconsistent")
    for case in cases:
        observation = observations.get(case["view_id"])
        if observation is not None and observation["image_id"] != case["image_id"]:
            raise ValueError("Observation image/view join does not match the manifest")
    return observations, {
        "status": report["status"],
        "verification": verification,
        "selected": report["selected"],
        "attempted": len(attempts),
        "valid_observations": len(observations),
        "scored": report["scored"],
        "known_cost_usd": str(known),
        "unpriced_attempts": unpriced,
        "total_cost_usd": str(known) if unpriced == 0 and len(attempts) == 50 else None,
        "provider_histogram_basis": "verified observations; failed and missing outputs are excluded",
        "provider_histogram": dict(
            Counter(row["provenance"]["provider_returned"] or "unknown" for row in observations.values())
        ),
        "model_histogram": dict(
            Counter(row["provenance"]["model_returned"] for row in observations.values())
        ),
        "prompt_sha256": run["prompt"]["sha256"],
        "prompt_version": run["prompt"]["version"],
        "settings": run["settings"],
    }


def validate_prompts(runs: dict) -> dict:
    from visual_quality.labelling import _prompt_bundle

    original = _prompt_bundle(PROJECT / "prompts/visual-rating-v1")
    for key in ("a/v1", "b/v1"):
        if "prompt_sha256" in runs[key] and (
            runs[key]["prompt_sha256"] != original["sha256"]
            or runs[key]["prompt_version"] != "visual-rating-v1"
        ):
            raise ValueError("Original arm does not match the frozen v1 prompt bundle")
    lock_path = HERE / "prompt-revision.json"
    if not lock_path.exists():
        if "prompt_sha256" in runs["b/v2"]:
            raise ValueError("A revised prompt run requires its frozen revision lock")
        return {"status": "missing_revision_lock"}
    lock = read_json(lock_path)
    directory = PROJECT / "prompts/visual-rating-v2"
    expected = {
        str((directory / name).relative_to(REPO))
        for name in ("system.txt", "technical-quality.txt", "response.schema.json", "prompt-version.json")
    }
    source_inputs = {
        "source_references_sha256": HERE / "references/a.jsonl",
        "source_observations_sha256": DATA / "a/v1/observations.jsonl",
        "source_audits_sha256": HERE / "audits/a.jsonl",
    }
    if (
        lock.get("schema_version") != "prompt-revision-v1"
        or lock.get("source_stage") != "a"
        or lock.get("stage_b_not_inspected") is not True
        or lock.get("prompt_version") != "visual-rating-v2"
        or any(not path.is_file() or lock.get(key) != sha(path) for key, path in source_inputs.items())
        or set(lock.get("files", {})) != expected
        or any(sha(REPO / name) != digest for name, digest in lock.get("files", {}).items())
    ):
        raise ValueError("Prompt revision lock differs from its stage-A inputs or saved files")
    revised = _prompt_bundle(directory)
    if revised["sha256"] != lock.get("prompt_sha256") or revised["version"] != "visual-rating-v2":
        raise ValueError("Revised prompt bundle differs from its frozen lock")
    if "prompt_sha256" in runs["b/v2"] and (
        runs["b/v2"]["prompt_sha256"] != revised["sha256"]
        or runs["b/v2"]["prompt_version"] != revised["version"]
    ):
        raise ValueError("Revised arm does not match the frozen v2 prompt bundle")
    return {"status": "verified", "frozen_at": lock.get("frozen_at"), "prompt_sha256": revised["sha256"]}


def case_rows(stage: str, cases: list[dict], references: dict, audits: dict, runs: dict) -> list[dict]:
    rows = []
    for case in cases:
        reference = references.get(case["case_id"], {})
        validate_reference(reference)
        point, interval = reference.get("score"), reference.get("interval")
        if point is None or interval is None:
            point, interval = None, None
        row = {
            key: case.get(key)
            for key in ("case_id", "image_id", "view_id", "family_id", "role", "operator", "severity")
        }
        row.update(
            stage=stage,
            stratum=stratum(case),
            reference_score=point,
            reference_lower=interval[0] if interval else None,
            reference_upper=interval[1] if interval else None,
            reference_evidence=reference.get("evidence"),
            reference_defects=reference.get("defects", []),
        )
        for arm in ("v1", "v2"):
            observation = runs.get(arm, {}).get(case["view_id"], {})
            rating = observation.get("technical_quality", {})
            score = rating.get("score") if rating.get("status") == "ok" else None
            if score is not None and not valid_score(score):
                raise ValueError("Model score must be finite and within 1..5")
            judgment = audits.get(case["case_id"], {}).get("judgments", {}).get(arm, {})
            supported, issues = judgment.get("evidence_supported"), judgment.get("issues", [])
            if (
                supported is not None
                and type(supported) is not bool
                or not isinstance(issues, list)
                or not all(isinstance(issue, str) for issue in issues)
            ):
                raise ValueError("Invalid evidence audit")
            for key, value in {
                "score": score,
                "status": rating.get("status", "missing"),
                "distance": distance(score, interval),
                "critical": critical(score, interval),
                "evidence": rating.get("evidence"),
                "evidence_supported": supported,
                "evidence_issues": issues,
                "audit_note": judgment.get("note"),
                "provider": observation.get("provenance", {}).get("provider_returned"),
            }.items():
                row[f"{arm}_{key}"] = value
        row["distance_improvement"] = (
            row["v1_distance"] - row["v2_distance"]
            if row["v1_distance"] is not None and row["v2_distance"] is not None
            else None
        )
        rows.append(row)
    return rows


def input_paths(manifest: dict) -> set[Path]:
    paths = {
        HERE / "manifest.json",
        HERE / "protocol.md",
        HERE / "prompt-revision.json",
        Path(__file__).resolve(),
    }
    for version in ("v1", "v2"):
        paths.update(
            PROJECT / f"prompts/visual-rating-{version}" / name
            for name in ("system.txt", "technical-quality.txt", "response.schema.json", "prompt-version.json")
        )
    for stage, arms in (("a", ("v1",)), ("b", ("v1", "v2"))):
        record = manifest["stages"][stage]
        freeze_path = HERE / "references" / f"{stage}-freeze.json"
        paths.update(
            {HERE / "references" / f"{stage}.jsonl", HERE / "audits" / f"{stage}.jsonl", freeze_path}
        )
        if freeze_path.exists():
            paths.update(reference_sources(read_json(freeze_path)))
        for key, filenames in (
            ("views_dir", ("views.jsonl", "preparation.json")),
            ("dataset_dir", ("dataset.json", "train.jsonl")),
        ):
            paths.update(REPO / record[key] / filename for filename in filenames)
        for arm in arms:
            run = DATA / stage / arm
            paths.update(run / filename for filename in ("run.json", "report.json", "observations.jsonl"))
            paths.update(run.glob("attempts/*/*.json"))
    return paths


def build_report() -> tuple[dict, list[dict]]:
    manifest_hash = sha(HERE / "manifest.json")
    manifest = read_json(HERE / "manifest.json")
    paths = input_paths(manifest)
    pins = {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}
    if pins[str((HERE / "manifest.json").relative_to(REPO))] != manifest_hash:
        raise ValueError("Manifest changed during loading")
    all_rows, stages, run_summaries = [], {}, {}
    all_families, all_images, all_views = set(), set(), set()
    for stage, arms in (("a", ("v1",)), ("b", ("v1", "v2"))):
        record = manifest["stages"][stage]
        cases = record["cases"]
        validate_cases(cases, stage)
        for key, seen in (("family_id", all_families), ("image_id", all_images), ("view_id", all_views)):
            current = {case[key] for case in cases}
            if current & seen:
                raise ValueError(f"Stages overlap by {key}")
            seen.update(current)
        expected = {case["case_id"] for case in cases}
        references, reference_freeze = load_references(stage, expected)
        audits = indexed(read_rows(HERE / "audits" / f"{stage}.jsonl"), "case_id", expected, "audit")
        views = REPO / record["views_dir"]
        if sha(views / "views.jsonl") != record["views_sha256"]:
            raise ValueError("View manifest differs from cohort selection")
        view_index = indexed(
            read_rows(views / "views.jsonl"), "view_id", {case["view_id"] for case in cases}, "view"
        )
        if len(view_index) != 50 or any(
            view_index[case["view_id"]]["image_id"] != case["image_id"]
            or view_index[case["view_id"]]["view_sha256"] != case["view_sha256"]
            for case in cases
        ):
            raise ValueError("Frozen case/view coverage differs")
        runs = {}
        for arm in arms:
            runs[arm], run_summaries[f"{stage}/{arm}"] = load_run(DATA / stage / arm, views, cases)
        rows = case_rows(stage, cases, references, audits, runs)
        stages[stage] = {
            "reference_count": len(references),
            "reference_freeze": reference_freeze,
            "audit_case_count": len(audits),
            "arms": {arm: arm_metrics(rows, arm) for arm in arms},
        }
        if stage == "b":
            stages[stage]["paired_comparison"] = paired_result(cases, rows)
        all_rows.extend(rows)
    present = [run for run in run_summaries.values() if "settings" in run]
    if present and any(run["settings"] != present[0]["settings"] for run in present):
        raise ValueError("Request settings differ between prompt runs")
    if present and (
        present[0]["settings"]["model"] != "deepseek/deepseek-v4.1-flash-1017"
        or present[0]["settings"]["max_tokens"] != 1024
    ):
        raise ValueError("Run model or token limit differs from the protocol")
    if all("prompt_sha256" in run_summaries[key] for key in ("a/v1", "b/v1")) and (
        run_summaries["a/v1"]["prompt_sha256"] != run_summaries["b/v1"]["prompt_sha256"]
    ):
        raise ValueError("The original prompt differs between stages")
    prompt_revision = validate_prompts(run_summaries)
    known_costs = [
        Decimal(run["known_cost_usd"]) for run in run_summaries.values() if run["known_cost_usd"] is not None
    ]
    all_costs_known = all(run["total_cost_usd"] is not None for run in run_summaries.values())
    summary = {
        "schema_version": "prompt-calibration-report-v1",
        "reference_kind": "assistant_point_judgments_and_plausible_intervals",
        "human_calibration": False,
        "training_ready": False,
        "review_status": "pending_human_validation",
        "limitations": [
            "Assistant intervals are subjective ranges, not human MOS or confidence intervals.",
            "Bootstrap uncertainty applies to this panel and its assistant references.",
            "Severity ordering is diagnostic; stored severity is not measured perceptual quality.",
            "Parent and variant preprocessing paths differ; parent comparisons are separate.",
        ],
        "method": {
            "distance": "max(L - score, 0, score - U)",
            "effect": "v1_distance - v2_distance",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap": "paired whole-family resampling within source strata; percentile interval",
            "random_generator": "NumPy default_rng PCG64",
            "percentile_method": "NumPy quantile linear at 0.025 and 0.975",
            "inclusive_threshold_float_tolerance": 1e-12,
            "bootstrap_zero_endpoint_tolerance": 1e-12,
            "numpy_version": np.__version__,
            "planned_stratum_image_weights": {
                "controlled": 0.48,
                "vizwiz": 0.28,
                "standalone_megalith": 0.24,
            },
            "minimum_mean_improvement": MIN_IMPROVEMENT,
            "maximum_stratum_worsening": MAX_STRATUM_WORSENING,
        },
        "stages": stages,
        "runs": run_summaries,
        "prompt_revision": prompt_revision,
        "cost": {
            "known_cost_usd": str(sum(known_costs, Decimal(0))),
            "total_cost_usd": str(sum(known_costs, Decimal(0))) if all_costs_known else None,
            "unpriced_attempts": sum(run["unpriced_attempts"] for run in run_summaries.values())
            if all(run["unpriced_attempts"] is not None for run in run_summaries.values())
            else None,
            "basis": "Reported OpenRouter usage cost; unknown charges remain null",
        },
        "input_sha256": pins,
        "report_script_sha256": sha(Path(__file__).resolve()),
    }
    if paths != input_paths(manifest) or pins != {
        str(path.relative_to(REPO)): sha(path) for path in sorted(paths)
    }:
        raise ValueError("Inputs changed during reporting; retry after runs and reviews stop")
    return summary, all_rows


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Write summary.json and cases.csv under results/"
    )
    args = parser.parse_args()
    summary, rows = build_report()
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        output = HERE / "results"
        output.mkdir(exist_ok=True)
        (output / "summary.json").write_text(encoded)
        (output / "cases.csv").write_text(csv_text(rows))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
