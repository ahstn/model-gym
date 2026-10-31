"""Versioned string-score transport, with the existing visual rubric and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from visual_quality.labelling import _prompt_bundle
from visual_quality.labels import LabelValidationError, parse_rating

VERSION = "visual-rating-v1-string-score-v1"
ADAPTER_VERSION = "string-score-to-decimal-v1"
SCORES = tuple(f"{n // 10}.{n % 10}0" for n in range(10, 51))


def prompt_bundle(directory: Path) -> dict:
    """Change score transport only; preserve all v1 visual assessment instructions."""
    original = _prompt_bundle(directory)
    result = copy.deepcopy(original)
    old_example = '"score":4.0'
    old_explanation = "The value 4.0 above illustrates the numeric type only; it is not a default score."
    if result["user"].count(old_example) != 1 or result["user"].count(old_explanation) != 1:
        raise ValueError("Original v1 format example changed")
    result["user"] = (
        result["user"]
        .replace(old_example, '"score":"4.00"')
        .replace(
            old_explanation,
            'The value "4.00" above illustrates the string format only; it is not a default score. '
            'For an ok score, use a JSON string with exactly two decimal places, from "1.00" to "5.00" '
            'in steps of "0.10". Keep an unscorable score as JSON null.',
        )
    )
    schema = result["response_format"]["json_schema"]
    schema["name"] = "technical_quality_v1_string_score_v1"
    schema["schema"]["properties"]["score"] = {
        "type": ["string", "null"],
        "enum": [*SCORES, None],
    }
    result["version"] = VERSION
    result["parent_version"] = original["version"]
    result["parent_sha256"] = original["sha256"]
    result.pop("sha256")
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LabelValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LabelValidationError(f"Non-finite JSON constant: {value}")


def _load(text: str) -> dict:
    if not isinstance(text, str):
        raise LabelValidationError("Rating response must be JSON text")
    try:
        rating = json.loads(
            text, parse_float=Decimal, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (json.JSONDecodeError, RecursionError, InvalidOperation) as error:
        raise LabelValidationError("Rating response must be a single valid JSON object") from error
    if not isinstance(rating, dict) or "score" not in rating:
        raise LabelValidationError("Rating must be an object with a score")
    return rating


def score_wire_valid(text: str) -> bool:
    """Diagnose only the score token, without accepting the full response."""
    try:
        score = _load(text)["score"]
        return score is None or (type(score) is str and score in SCORES)
    except LabelValidationError:
        return False


def parse_wire(text: str) -> tuple[dict, str | None]:
    """Validate the exact wire enum, then decode through Decimal without rounding."""
    rating = _load(text)
    score = rating["score"]
    if score is not None and (type(score) is not str or score not in SCORES):
        raise LabelValidationError("Score must be an exact permitted two-decimal string or null")
    # Conversion is a named transport adapter, not repair of invalid provider text.
    adapted = {**rating, "score": float(Decimal(score)) if score is not None else None}
    try:
        validated = parse_rating(json.dumps(adapted, allow_nan=False))
    except (TypeError, ValueError) as error:
        if isinstance(error, LabelValidationError):
            raise
        raise LabelValidationError("Rating contains an unsupported JSON value") from error
    return validated, score
