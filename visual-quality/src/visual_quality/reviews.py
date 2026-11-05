"""Versioned assistant corrections without altering the original teacher labels.

Review plans pin the exact source artifacts and rubric. Their output remains
pending human review and cannot enable a training task mask.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from visual_quality.labelling import verify_labels
from visual_quality.labels import VIEW_KEYS, LabelValidationError, parse_rating
from visual_quality.storage import atomic_json, file_sha256, write_jsonl
from visual_quality.views import load_views

PLAN_KEYS = {
    "schema_version",
    "review_id",
    "reviewed_at",
    "reviewer",
    "source_observations_sha256",
    "source_views_sha256",
    "rubric_sha256",
    "reviews",
}
REVIEW_KEYS = {
    "image_id",
    "view_id",
    "original_score",
    "technical_quality",
    "reason",
    "uncertainty_interval",
    "independent_review",
}
OUTPUT_FILES = {"reviewed-observations.jsonl", "report.json"}


def _keys(value: object, required: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(required))}")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _exact_float(value: str) -> float:
    """Reject numeric text that loses decimal precision before validation."""
    parsed = float(value)
    canonical = Decimal(str(parsed))
    if not canonical.is_finite() or canonical != Decimal(value):
        raise ValueError("JSON number loses decimal precision; use an exact supported value")
    return parsed


def _read(path: Path) -> dict:
    return json.loads(
        path.read_text(), object_pairs_hook=_object, parse_constant=_constant, parse_float=_exact_float
    )


def _jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records).encode()


def _rating(value: object) -> dict:
    _keys(value, {"status", "score", "evidence", "limitations"}, "Technical quality")
    rating = parse_rating(json.dumps({"task": "technical_quality", **value}, allow_nan=False))
    return {key: item for key, item in rating.items() if key != "task"}


def _interval(value: object, score: float | None) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Uncertainty interval must contain two endpoints")
    for endpoint in value:
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            raise LabelValidationError("Interval endpoints must be finite 1–5 values in 0.1 steps")
        number = Decimal(str(endpoint))
        if not number.is_finite() or not 1 <= number <= 5 or number % Decimal("0.1"):
            raise ValueError("Interval endpoints must be finite 1–5 values in 0.1 steps")
    if value[0] > value[1] or (score is not None and not value[0] <= score <= value[1]):
        raise ValueError("Uncertainty interval must be ordered and contain its score")


def _header(plan: dict, run: dict, source_hash: str, views_hash: str) -> None:
    _keys(plan, PLAN_KEYS, "Review plan")
    if plan["schema_version"] != "label-review-plan-v1":
        raise ValueError("Review plan must use label-review-plan-v1")
    _text(plan["review_id"], "Review ID")
    stamp = plan["reviewed_at"]
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
    if not isinstance(stamp, str) or not re.fullmatch(pattern, stamp):
        raise ValueError("reviewed_at must be an RFC 3339 timestamp with timezone")
    datetime.fromisoformat(stamp)
    reviewer = plan["reviewer"]
    _keys(reviewer, {"kind", "id", "method", "model_id"}, "Reviewer")
    if reviewer["kind"] != "assistant" or reviewer["id"] != "codex" or reviewer["model_id"] is not None:
        raise ValueError("This workflow records assistant review, not human or provider labels")
    _text(reviewer["method"], "Review method")
    expected = {
        "source_observations_sha256": source_hash,
        "source_views_sha256": views_hash,
        "rubric_sha256": run["prompt"]["sha256"],
    }
    for key, value in expected.items():
        if plan[key] != value:
            raise ValueError(f"Review plan {key} does not match the frozen source")
    if not isinstance(plan["reviews"], list) or not plan["reviews"]:
        raise ValueError("Review plan must contain source observations")


def _entry(review: dict, source: dict) -> tuple[dict, str]:
    _keys(review, REVIEW_KEYS, "Review entry")
    if review["image_id"] != source["image_id"] or review["view_id"] != source["view_id"]:
        raise ValueError("Review image and view must match the source observation")
    original = review["original_score"]
    if isinstance(original, bool) or original != source["technical_quality"]["score"]:
        raise ValueError("Review original_score differs from the source observation")
    technical = _rating(review["technical_quality"])
    _text(review["reason"], "Review reason")
    _interval(review["uncertainty_interval"], technical["score"])
    independent = review["independent_review"]
    _keys(independent, {"reviewer", "score", "interval", "evidence", "limitations"}, "Independent review")
    _text(independent["reviewer"], "Independent reviewer")
    _rating({"status": "ok", **{key: independent[key] for key in ("score", "evidence", "limitations")}})
    _interval(independent["interval"], independent["score"])
    old = source["technical_quality"]
    score_changed = any(old[key] != technical[key] for key in ("status", "score"))
    evidence_changed = any(old[key] != technical[key] for key in ("evidence", "limitations"))
    decision = {
        (False, False): "retained",
        (True, False): "score_corrected",
        (False, True): "evidence_corrected",
        (True, True): "score_and_evidence_corrected",
    }[score_changed, evidence_changed]
    return technical, decision


def _paths(views: Path, source: Path, plan: Path, output: Path) -> tuple[Path, Path, Path, Path]:
    views, source, plan, output = (path.resolve() for path in (views, source, plan, output))
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (views, source)):
        raise ValueError("Reviewed output must be separate from source runs and prepared views")
    if plan.is_relative_to(output) or output.is_relative_to(plan):
        raise ValueError("Reviewed output must not contain or overwrite the review plan")
    return views, source, plan, output


def _expected(views: Path, source: Path, plan_path: Path) -> tuple[list[dict], dict, dict[Path, str]]:
    pinned = {
        path: file_sha256(path)
        for path in (
            source / "observations.jsonl",
            source / "run.json",
            views / "views.jsonl",
            views / "preparation.json",
            plan_path,
        )
    }
    view_records = load_views(views)
    verify_labels(views, source)
    plan, run = _read(plan_path), _read(source / "run.json")
    source_hash, views_hash = pinned[source / "observations.jsonl"], pinned[views / "views.jsonl"]
    _header(plan, run, source_hash, views_hash)
    originals = [json.loads(line) for line in (source / "observations.jsonl").read_text().splitlines()]
    for key in ("image_id", "view_id"):
        if len({row[key] for row in originals}) != len(originals):
            raise ValueError("Duplicate source observation IDs")
    source_by_view = {row["view_id"]: row for row in originals}
    view_by_id = {row["view_id"]: row for row in view_records}
    reviewed = {}
    for review in plan["reviews"]:
        _keys(review, REVIEW_KEYS, "Review entry")
        _text(review["view_id"], "Review view ID")
        view_id = review["view_id"]
        if view_id in reviewed or view_id not in source_by_view:
            raise ValueError("Review plan contains a duplicate or unknown view")
        original = source_by_view[view_id]
        if any(original[key] != view_by_id[view_id][key] for key in (*VIEW_KEYS, "photo_domain_eligible")):
            raise ValueError("Source observation does not match its prepared view")
        technical, decision = _entry(review, original)
        reviewed[view_id] = {
            "schema_version": "label-review-v1",
            "label_kind": "assistant_review",
            **{key: original[key] for key in (*VIEW_KEYS, "photo_domain_eligible")},
            "technical_quality": technical,
            "aesthetics": {"status": "not_requested", "score": None},
            "review_status": "pending_human_review",
            "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
            "source_observation": original,
            "review": {
                "id": plan["review_id"],
                "reviewed_at": plan["reviewed_at"],
                "reviewer": plan["reviewer"],
                "plan_sha256": pinned[plan_path],
                "source_observations_sha256": source_hash,
                **{key: review[key] for key in ("reason", "uncertainty_interval", "independent_review")},
                "decision": decision,
            },
        }
    if set(reviewed) != set(source_by_view):
        raise ValueError("Review plan must cover every source observation exactly once")
    records = [reviewed[row["view_id"]] for row in originals]
    report = {
        "schema_version": "label-review-report-v1",
        "review_id": plan["review_id"],
        "source_run_id": run["run_id"],
        "observations": len(records),
        "scored": sum(row["technical_quality"]["status"] == "ok" for row in records),
        "decisions": dict(sorted(Counter(row["review"]["decision"] for row in records).items())),
        "source_observations_sha256": source_hash,
        "source_views_sha256": views_hash,
        "rubric_sha256": plan["rubric_sha256"],
        "review_plan_sha256": pinned[plan_path],
        "reviewed_observations_sha256": hashlib.sha256(_jsonl(records)).hexdigest(),
        "review_status": "pending_human_review",
        "training_ready": False,
    }
    _unchanged(pinned)
    return records, report, pinned


def _unchanged(pinned: dict[Path, str]) -> None:
    if any(file_sha256(path) != digest for path, digest in pinned.items()):
        raise ValueError("An input changed during review export")


def _verify_output(output: Path, records: list[dict], report: dict) -> dict:
    if not output.is_dir() or {path.name for path in output.iterdir()} != OUTPUT_FILES:
        raise ValueError("Reviewed output must contain exactly its observations and report")
    if (output / "reviewed-observations.jsonl").read_bytes() != _jsonl(records):
        raise ValueError("Reviewed observations differ from the pinned review plan")
    if _jsonl([_read(output / "report.json")]) != _jsonl([report]):
        raise ValueError("Reviewed report differs from the pinned review plan")
    return report


def verify_reviewed_labels(
    views_dir: Path,
    source_run_dir: Path,
    review_path: Path,
    output_dir: Path,
) -> dict:
    """Read-only verification against the original replies and pinned review plan."""
    views, source, plan, output = _paths(views_dir, source_run_dir, review_path, output_dir)
    records, report, pinned = _expected(views, source, plan)
    result = _verify_output(output, records, report)
    _unchanged(pinned)
    return result


def build_reviewed_labels(
    views_dir: Path,
    source_run_dir: Path,
    review_path: Path,
    output_dir: Path,
) -> dict:
    """Publish a separate review artifact, or verify and reuse an identical export."""
    views, source, plan, output = _paths(views_dir, source_run_dir, review_path, output_dir)
    records, report, pinned = _expected(views, source, plan)
    if output.exists():
        return _verify_output(output, records, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        write_jsonl(staging / "reviewed-observations.jsonl", records)
        atomic_json(staging / "report.json", report)
        _verify_output(staging, records, report)
        _unchanged(pinned)
        if output.exists():
            return _verify_output(output, records, report)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report
