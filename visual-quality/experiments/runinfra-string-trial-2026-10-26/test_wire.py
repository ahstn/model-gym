"""Exact transport constraints and unchanged rating semantics, without provider calls."""

import importlib.util
import json
from pathlib import Path

import pytest

from visual_quality.labels import LabelValidationError

SPEC = importlib.util.spec_from_file_location("trial200_wire_tests", Path(__file__).with_name("wire.py"))
wire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wire)


def rating(score="3.90", **changes):
    return {
        "task": "technical_quality",
        "status": "ok",
        "score": score,
        "evidence": "Fine detail remains clear with slight edge softness.",
        "limitations": [],
        **changes,
    }


@pytest.mark.parametrize("score", wire.SCORES)
def test_exact_enum_is_decoded_without_rounding(score):
    decoded, original = wire.parse_wire(json.dumps(rating(score)))
    assert original == score
    assert decoded["score"] == float(score)


@pytest.mark.parametrize(
    "score", [3.9, 4, True, "3.9", "3.900", "3.91", "0.90", "5.10", " 3.90", "3.90 ", "3e0", "NaN"]
)
def test_wrong_type_precision_step_and_range_are_rejected(score):
    text = json.dumps(rating(score))
    assert wire.score_wire_valid(text) is False
    with pytest.raises(LabelValidationError):
        wire.parse_wire(text)


def test_null_requires_unscorable_and_old_constraints_still_apply():
    decoded, score = wire.parse_wire(json.dumps(rating(None, status="unscorable")))
    assert decoded["score"] is score is None
    bad = [
        rating(None),
        rating("3.90", status="unscorable"),
        rating(evidence="word " * 31),
        rating(limitations=["limited_detail"] * 2),
        rating(extra=True),
        rating(task="aesthetics"),
    ]
    for value in bad:
        with pytest.raises(LabelValidationError):
            wire.parse_wire(json.dumps(value))


@pytest.mark.parametrize(
    "text",
    [
        '{"score":"3.90","score":"4.00"}',
        '{"score":NaN}',
        '{"score":1e99999999999999999999999}',
        "[]",
        "null",
        "```json\n{}\n```",
    ],
)
def test_duplicate_keys_nonfinite_values_and_nonobjects_are_rejected(text):
    with pytest.raises(LabelValidationError):
        wire.parse_wire(text)


def test_prompt_preserves_visual_rubric_and_strict_enum():
    prompt_dir = Path(__file__).resolve().parents[2] / "prompts/visual-rating-v1"
    original = wire._prompt_bundle(prompt_dir)
    changed = wire.prompt_bundle(prompt_dir)
    assert changed["system"] == original["system"]
    prefix = original["user"].split("Return exactly this structure")[0]
    assert changed["user"].startswith(prefix)
    assert changed["parent_sha256"] == original["sha256"]
    assert changed["sha256"] != original["sha256"]
    assert changed["version"] == wire.VERSION
    schema = changed["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["score"] == {
        "type": ["string", "null"],
        "enum": [*wire.SCORES, None],
    }
    assert '"score":"4.00"' in changed["user"]
