"""Strict teacher responses and pending observations for the first label pilot.

These observations are sidecar records, not accepted training targets. This
version has no promotion operation: every task mask remains false.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

SCHEMA_VERSION = "label-observation-v1"
LIMITATIONS = frozenset({"limited_detail", "partial_view", "ambiguous_style", "unsupported_domain"})
RATING_KEYS = frozenset({"task", "status", "score", "evidence", "limitations"})
VIEW_KEYS = ("image_id", "view_id", "source_sha256", "view_sha256", "split", "split_group")
PROVENANCE_KEYS = frozenset(
    {
        "run_id",
        "attempt_id",
        "provider",
        "model_requested",
        "model_returned",
        "provider_returned",
        "prompt_version",
        "prompt_sha256",
        "request_sha256",
        "created_at",
        "usage",
    }
)
OBSERVATION_KEYS = frozenset(
    {
        "schema_version",
        "label_kind",
        *VIEW_KEYS,
        "technical_quality",
        "aesthetics",
        "photo_domain_eligible",
        "review_status",
        "task_masks",
        "provenance",
    }
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z")


class LabelValidationError(ValueError):
    """A response or observation violates the versioned label contract."""


def _exact_keys(value: Any, keys: frozenset[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise LabelValidationError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def _nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LabelValidationError(f"{name} must be a nonempty string")


def _hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LabelValidationError(f"{name} must be a 64-character SHA-256 hex digest")


def _score(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise LabelValidationError("An ok score must be a finite number from 1 to 5 in 0.1 steps")
    number = Decimal(str(value))
    if not number.is_finite() or number < 1 or number > 5 or number % Decimal("0.1") != 0:
        raise LabelValidationError("An ok score must be a finite number from 1 to 5 in 0.1 steps")


def _validate_rating(rating: Any) -> None:
    _exact_keys(rating, RATING_KEYS, "Rating")
    if rating["task"] != "technical_quality":
        raise LabelValidationError("The pilot accepts only the technical_quality task")
    if rating["status"] == "ok":
        _score(rating["score"])
    elif rating["status"] == "unscorable":
        if rating["score"] is not None:
            raise LabelValidationError("An unscorable rating must have a null score")
    else:
        raise LabelValidationError("Rating status must be ok or unscorable")
    _nonempty_string(rating["evidence"], "Rating evidence")
    if len(rating["evidence"].split()) > 30:
        raise LabelValidationError("Rating evidence must contain at most 30 words")
    limitations = rating["limitations"]
    if (
        not isinstance(limitations, list)
        or any(not isinstance(item, str) or item not in LIMITATIONS for item in limitations)
        or len(set(limitations)) != len(limitations)
    ):
        raise LabelValidationError("Rating limitations must be unique supported limitation names")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LabelValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LabelValidationError(f"Non-finite JSON constant: {value}")


def parse_rating(text: str) -> dict:
    """Parse a bare visual-rating-v1 response without repairing model output.

    Decimal parsing preserves the written score precision before conversion to
    ordinary JSON types. Invalid or failed responses never become score zero.
    """
    if not isinstance(text, str):
        raise LabelValidationError("Rating response must be JSON text")
    try:
        rating = json.loads(
            text, parse_float=Decimal, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (json.JSONDecodeError, RecursionError, InvalidOperation) as exc:
        raise LabelValidationError("Rating response must be a single valid JSON object") from exc
    _validate_rating(rating)
    if isinstance(rating["score"], Decimal):
        rating["score"] = float(rating["score"])
    return rating


def _json_value(value: Any) -> None:
    """Reject non-finite or non-JSON metadata before writing a sidecar."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise LabelValidationError("Observation metadata must contain only finite JSON values")


def _validate_provenance(provenance: Any) -> None:
    _exact_keys(provenance, PROVENANCE_KEYS, "Provenance")
    for key in ("run_id", "attempt_id", "model_requested", "model_returned", "prompt_version"):
        _nonempty_string(provenance[key], f"Provenance {key}")
    if provenance["provider"] != "openrouter":
        raise LabelValidationError("Provenance provider must be openrouter")
    if provenance["provider_returned"] is not None:
        _nonempty_string(provenance["provider_returned"], "Provenance provider_returned")
    for key in ("prompt_sha256", "request_sha256"):
        _hash(provenance[key], f"Provenance {key}")
    timestamp = provenance["created_at"]
    if not isinstance(timestamp, str) or _TIMESTAMP.fullmatch(timestamp) is None:
        raise LabelValidationError("Provenance created_at must be an RFC 3339 timestamp with timezone")
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise LabelValidationError("Provenance created_at must be a valid timestamp") from exc
    if not isinstance(provenance["usage"], dict):
        raise LabelValidationError("Provenance usage must be an object")


def validate_observation(record: dict) -> None:
    """Validate the complete v1 pending teacher sidecar or raise ValueError."""
    _exact_keys(record, OBSERVATION_KEYS, "Observation")
    _json_value(record)
    if record["schema_version"] != SCHEMA_VERSION or record["label_kind"] != "teacher_scalar":
        raise LabelValidationError("Observation must use label-observation-v1 and teacher_scalar")
    for key in ("image_id", "view_id", "split", "split_group"):
        _nonempty_string(record[key], key)
    for key in ("source_sha256", "view_sha256"):
        _hash(record[key], key)
    technical = record["technical_quality"]
    _exact_keys(technical, RATING_KEYS - {"task"}, "Technical quality")
    _validate_rating({"task": "technical_quality", **technical})
    if record["aesthetics"] != {"status": "not_requested", "score": None}:
        raise LabelValidationError("Aesthetics must be not_requested with a null score")
    domain = record["photo_domain_eligible"]
    if domain is not None and type(domain) is not bool:
        raise LabelValidationError("photo_domain_eligible must be true, false, or null")
    if record["review_status"] != "pending":
        raise LabelValidationError("Pilot observations must remain pending review")
    masks = record["task_masks"]
    _exact_keys(masks, frozenset({"technical_quality", "aesthetics", "photo_domain"}), "Task masks")
    if any(value is not False for value in masks.values()):
        raise LabelValidationError("All pilot task masks must be false")
    _validate_provenance(record["provenance"])


def observation(view: dict, rating: dict, provenance: dict) -> dict:
    """Build an independent, pending sidecar from an immutable delivered view."""
    _validate_rating(rating)
    try:
        record = {
            "schema_version": SCHEMA_VERSION,
            "label_kind": "teacher_scalar",
            **{key: view[key] for key in VIEW_KEYS},
            "technical_quality": {key: value for key, value in rating.items() if key != "task"},
            "aesthetics": {"status": "not_requested", "score": None},
            "photo_domain_eligible": view["photo_domain_eligible"],
            "review_status": "pending",
            "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
            "provenance": provenance,
        }
    except KeyError as exc:
        raise LabelValidationError(f"View is missing required field: {exc.args[0]}") from exc
    validate_observation(record)
    return copy.deepcopy(record)
