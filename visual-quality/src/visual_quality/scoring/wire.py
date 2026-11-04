"""Joint v1 rubric with exact score-string transport and independent task validation."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

TASKS = ("technical_quality", "aesthetics")
SCORES = tuple(f"{n // 10}.{n % 10}0" for n in range(10, 51))
LIMITATIONS = ("limited_detail", "partial_view", "ambiguous_style", "unsupported_domain")
VERSION = "visual-rating-v1-joint-string-score-v1"
ADAPTER_VERSION = "joint-string-score-to-decimal-v1"


def prompt_bundle(directory: Path) -> dict:
    user = (directory / "joint.txt").read_text()
    old = "score to a number from 1.0 to 5.0 in steps of 0.1."
    if user.count(old) != 1:
        raise ValueError("Joint prompt contract changed")
    user = user.replace(
        old,
        'score to a JSON string with exactly two decimal places, from "1.00" '
        'to "5.00" in steps of "0.10". Keep an unscorable score as JSON null.',
    )
    user = user.replace(
        "one sentence of at most 30 words",
        "one sentence targeting 12–20 words, with a hard maximum of 30 words",
    )
    task_schemas = {}
    for task in TASKS:
        properties = {
            "task": {"type": "string", "enum": [task]},
            "status": {"type": "string", "enum": ["ok", "unscorable"]},
            "score": {"type": ["string", "null"], "enum": [*SCORES, None]},
            "evidence": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string", "enum": list(LIMITATIONS)}},
        }
        task_schemas[task] = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    result = {
        "version": VERSION,
        "system": (directory / "system.txt").read_text(),
        "user": user,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "joint_visual_rating_string_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": task_schemas,
                    "required": list(TASKS),
                    "additionalProperties": False,
                },
            },
        },
    }
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite JSON constant")


def parse_joint(text: str) -> dict:
    """Reject ambiguous envelopes; preserve a valid task when its sibling is invalid."""
    result = {"ratings": {}, "wire_scores": {}, "errors": {}, "wire_score_valid": {}}
    try:
        value = json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
        if not isinstance(value, dict) or set(value) != set(TASKS):
            raise ValueError("Outer object must contain exactly both tasks")
    except (ValueError, TypeError, RecursionError) as error:
        result["errors"] = {task: f"invalid_json_envelope: {error}" for task in TASKS}
        result["wire_score_valid"] = dict.fromkeys(TASKS, False)
        return result
    for task in TASKS:
        rating = value[task]
        score = rating.get("score") if isinstance(rating, dict) else None
        wire_valid = (
            isinstance(rating, dict)
            and "score" in rating
            and (score is None or type(score) is str and score in SCORES)
        )
        result["wire_score_valid"][task] = wire_valid
        try:
            if not isinstance(rating, dict) or set(rating) != {
                "task",
                "status",
                "score",
                "evidence",
                "limitations",
            }:
                raise ValueError("Task requires exactly five fields")
            if rating["task"] != task or rating["status"] not in {"ok", "unscorable"}:
                raise ValueError("Task/status mismatch")
            if not wire_valid:
                raise ValueError("Score must be an exact permitted two-decimal string or null")
            if (rating["status"] == "unscorable") != (score is None):
                raise ValueError("Score and status disagree")
            evidence = rating["evidence"]
            if not isinstance(evidence, str) or not 1 <= len(evidence.split()) <= 30:
                raise ValueError("Evidence must contain 1–30 words")
            limitations = rating["limitations"]
            if not isinstance(limitations, list) or any(
                type(x) is not str or x not in LIMITATIONS for x in limitations
            ):
                raise ValueError("Invalid limitations")
            if len(set(limitations)) != len(limitations):
                raise ValueError("Duplicate limitations")
            result["ratings"][task] = {**rating, "score": None if score is None else float(Decimal(score))}
            result["wire_scores"][task] = score
        except (ValueError, TypeError) as error:
            result["errors"][task] = str(error)
    return result
