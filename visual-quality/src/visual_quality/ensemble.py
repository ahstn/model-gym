"""Canonical photographic ensemble and lossless offline observation migration.

Only this module defines the accepted weights. Stored provider observations are
inputs, never overwritten by a derived score. Run with ``python -m
visual_quality.ensemble --help`` to replace retired derived rows in an export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

MODEL = "photo_ensemble"
VERSION = "luna-deepseek-liquid-v1"
WEIGHTS = {"luna": Decimal("0.375"), "deepseek": Decimal("0.375"), "liquid": Decimal("0.25")}
RETIRED_MODELS = frozenset({"luna_deepseek_mean", MODEL})
DIMENSIONS = ("technical", "aesthetic")
CONTEXT_FIELDS = (
    "benchmark_partition",
    "benchmark_order",
    "family_id",
    "image_path",
    "view_sha256",
    "source",
    "stratum",
    "operator",
    "severity",
    "content_cluster",
)
PROVENANCE_FIELDS = (
    "model",
    "arm",
    "status",
    "view_sha256",
    "attempt_id",
    "request_sha256",
    "raw_response_sha256",
    "raw_response_path",
    "model_returned",
    "provider_returned",
    "prompt_sha256",
    "profile_sha256",
    "requested_reasoning",
    "returned_reasoning_effort",
    "request_started_at",
    "response_received_at",
    "latency_seconds",
    "price_usd",
    "price_basis",
    "scope_floor_applied",
    "excluded_probability",
    "reused_from",
    "technical_score",
    "aesthetic_score",
    "technical_score_valid",
    "aesthetic_score_valid",
)


def configuration() -> dict:
    """Return the portable, auditable configuration of the canonical scorer."""
    return {
        "version": VERSION,
        "model": MODEL,
        "weights": {dimension: {m: str(w) for m, w in WEIGHTS.items()} for dimension in DIMENSIONS},
        "missing_component_policy": "null_per_task; no reweighting or two-model fallback",
        "scope_policy": "use component policy scores; no new voting or reference override",
        "precision": "native component precision for arithmetic; two decimals for display only",
        "latency_policy": "no independent request latency; retain component request measurements",
        "price_policy": "no independent charge; component costs recorded separately, never double-counted",
    }


def _decimal(value) -> Decimal | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _true(value) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _score(row: dict | None, dimension: str) -> Decimal | None:
    if row is None or row.get("status") not in {"valid", "partial"}:
        return None
    validity = row.get(f"{dimension}_score_valid")
    if validity is not None and not _true(validity):
        return None
    value = _decimal(row.get(f"{dimension}_score"))
    return value if value is not None and 1 <= value <= 5 else None


def score_ensemble(
    component_rows: dict[str, dict],
    *,
    case_id: str,
    arm: str = "baseline",
    component_arms: dict[str, str] | None = None,
) -> dict:
    """Derive both outcomes; each needs all three valid component outcomes.

    ``component_arms`` must explicitly name a different arm, for example an
    unchanged baseline Liquid observation in a historical stats-arm comparison.
    Prompt versions may differ: their provenance remains attached to the row.
    """
    if not case_id or not arm:
        raise ValueError("case_id and arm are required")
    if set(component_rows) - WEIGHTS.keys():
        raise ValueError("Unexpected ensemble component")
    expected_arms = component_arms or {model: arm for model in WEIGHTS}
    if set(expected_arms) != WEIGHTS.keys():
        raise ValueError("Expected arms must name all three components")
    views = set()
    for model, component in component_rows.items():
        if component.get("model") != model or component.get("case_id") != case_id:
            raise ValueError("Component model/case does not match")
        if component.get("arm") != expected_arms[model]:
            raise ValueError("Component arm does not match")
        if not component.get("view_sha256"):
            raise ValueError("Component delivered-view hash is required")
        views.add(component["view_sha256"])
    if len(views) > 1:
        raise ValueError("Components assessed different delivered views")

    first = next(iter(component_rows.values()), {})
    row = {key: first.get(key) for key in CONTEXT_FIELDS}
    row.update(
        {
            "case_id": case_id,
            "model": MODEL,
            "arm": arm,
            "ensemble_version": VERSION,
            "ensemble_weights": {m: str(w) for m, w in WEIGHTS.items()},
            "component_arms": expected_arms,
            "component_observations": {
                model: {key: component.get(key) for key in PROVENANCE_FIELDS}
                for model, component in component_rows.items()
            },
            "component_attempt_ids": {model: r.get("attempt_id") for model, r in component_rows.items()},
            "latency_seconds": None,
            "latency_kind": "derived_no_http_request",
            "price_usd": None,
            "price_basis": "derived_no_independent_charge; see component observations",
            "review_elapsed_seconds": None,
            "attempt_count": 0,
            "evidence": None,
            "limitations": None,
            "training_ready": False,
        }
    )
    missing = {}
    for dimension in DIMENSIONS:
        values = {model: _score(component_rows.get(model), dimension) for model in WEIGHTS}
        missing[dimension] = [model for model, value in values.items() if value is None]
        value = None if missing[dimension] else sum(WEIGHTS[m] * values[m] for m in WEIGHTS)
        row[f"{dimension}_score"] = float(value) if value is not None else None
        row[f"{dimension}_score_native"] = float(value) if value is not None else None
        row[f"{dimension}_score_decimal"] = str(value) if value is not None else None
        row[f"{dimension}_score_display"] = (
            str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) if value is not None else None
        )
        row[f"{dimension}_score_valid"] = value is not None
    complete = sum(row[f"{d}_score_valid"] for d in DIMENSIONS)
    row["status"] = "valid" if complete == 2 else "partial" if complete else "missing_or_invalid_components"
    row["validation_errors"] = {d: models for d, models in missing.items() if models}
    floors = [_true(component_rows.get(m, {}).get("scope_floor_applied")) for m in WEIGHTS]
    row["scope_floor_applied"] = all(floors) and all(row[f"{d}_score"] == 1 for d in DIMENSIONS)
    row["scope_disagreement"] = any(floors) and not all(floors)
    row["requires_review"] = complete != 2 or row["scope_disagreement"]
    prices = [_decimal(component_rows.get(m, {}).get("price_usd")) for m in WEIGHTS]
    known = sum((p for p in prices if p is not None), Decimal(0))
    row["component_known_price_usd"] = str(known)
    row["component_unpriced_count"] = sum(p is None for p in prices)
    row["component_total_price_usd"] = str(known) if all(p is not None for p in prices) else None
    return row


def replace_ensemble_rows(rows: list[dict]) -> list[dict]:
    """Replace legacy derived means, preserving every non-derived observation.

    Historical own/swapped-stats comparisons use their original Luna/DeepSeek
    arms and explicitly identify unchanged baseline Liquid as the third input.
    A missing third model always leaves the relevant outcome missing.
    """
    output = [row for row in rows if row.get("model") not in RETIRED_MODELS]
    by_case = defaultdict(dict)
    for row in output:
        model = row.get("model")
        if model not in WEIGHTS:
            continue
        key = (model, row["arm"])
        case = by_case[row["case_id"]]
        if key in case:
            raise ValueError(f"Duplicate component observation: {row['case_id']} {key}")
        case[key] = row
    for case_id, observations in sorted(by_case.items()):
        arms = sorted({arm for _, arm in observations})
        for arm in arms:
            expected = {
                model: "baseline" if model == "liquid" and arm in {"own_stats", "swapped_stats"} else arm
                for model in WEIGHTS
            }
            components = {
                model: observations[model, a] for model, a in expected.items() if (model, a) in observations
            }
            output.append(score_ensemble(components, case_id=case_id, arm=arm, component_arms=expected))
    return output


def read_observations(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_observations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in rows
            )
    else:
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def publication_rows(rows: list[dict], identities: list[str]) -> list[dict]:
    """Select a bounded image allowlist, retaining every observation for each image."""
    if (
        not isinstance(identities, list)
        or len(identities) > 1000
        or any(not isinstance(value, str) or not value for value in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("Publication allowlist must contain at most 1,000 distinct case IDs")
    selected = set(identities)
    if selected - {row["case_id"] for row in rows}:
        raise ValueError("Publication allowlist includes unknown case IDs")
    return [row for row in rows if row["case_id"] in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="original CSV or JSONL observations")
    parser.add_argument("--output", type=Path, required=True, help="separate canonical CSV or JSONL export")
    parser.add_argument("--published-case-ids", type=Path, help="JSON array allowlist, at most 1,000 images")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Write a separate export; preserve the original observations")
    rows = replace_ensemble_rows(read_observations(args.input))
    if args.published_case_ids:
        identities = json.loads(args.published_case_ids.read_text())
        try:
            rows = publication_rows(rows, identities)
        except ValueError as error:
            parser.error(str(error))
    write_observations(args.output, rows)
    audit = {
        **configuration(),
        "input": str(args.input),
        "output": str(args.output),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "rows": len(rows),
        "images": len({r["case_id"] for r in rows}),
        "ensemble_rows": sum(r["model"] == MODEL for r in rows),
    }
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps({key: audit[key] for key in ("version", "rows", "images", "ensemble_rows", "output")}))


if __name__ == "__main__":
    main()
