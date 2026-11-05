import json
from copy import deepcopy
from pathlib import Path

import pytest

from visual_quality.scoring.wire import SCORES, TASKS, parse_joint, prompt_bundle


def valid():
    return {
        t: {
            "task": t,
            "status": "ok",
            "score": "4.10",
            "evidence": "Clear visible details and balanced framing.",
            "limitations": [],
        }
        for t in TASKS
    }


@pytest.mark.parametrize("score", SCORES)
def test_exact_scores_decode_without_rounding(score):
    value = valid()
    value["aesthetics"]["score"] = score
    parsed = parse_joint(json.dumps(value))
    assert not parsed["errors"]
    assert parsed["ratings"]["aesthetics"]["score"] == float(score)


@pytest.mark.parametrize("score", [4.1, True, "4.1", "4.100", "4.11", "NaN", "5.10"])
def test_invalid_score_does_not_discard_valid_sibling(score):
    value = valid()
    value["aesthetics"]["score"] = score
    parsed = parse_joint(json.dumps(value))
    assert set(parsed["ratings"]) == {"technical_quality"}
    assert set(parsed["errors"]) == {"aesthetics"}


@pytest.mark.parametrize(
    "text",
    [
        '{"technical_quality":{},"technical_quality":{},"aesthetics":{}}',
        '{"technical_quality":NaN,"aesthetics":{}}',
        "{}",
        "[]",
        "```json\n{}\n```",
    ],
)
def test_ambiguous_or_malformed_envelopes_fail_both(text):
    parsed = parse_joint(text)
    assert not parsed["ratings"]
    assert set(parsed["errors"]) == set(TASKS)


def test_nullable_status_and_evidence_contract():
    value = valid()
    value["aesthetics"].update(status="unscorable", score=None)
    assert not parse_joint(json.dumps(value))["errors"]
    bad = deepcopy(value)
    bad["aesthetics"]["status"] = "ok"
    assert "aesthetics" in parse_joint(json.dumps(bad))["errors"]
    value["technical_quality"]["evidence"] = "word " * 31
    parsed = parse_joint(json.dumps(value))
    assert set(parsed["errors"]) == {"technical_quality"}
    assert parsed["wire_score_valid"]["technical_quality"]


def test_prompt_preserves_rubrics_and_uses_enum():
    directory = Path(__file__).resolve().parents[1] / "prompts/visual-rating-v1"
    bundle = prompt_bundle(directory)
    assert (
        bundle["user"].split("OUTPUT CONTRACT")[0]
        == (directory / "joint.txt").read_text().split("OUTPUT CONTRACT")[0]
    )
    schema = bundle["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["aesthetics"]["properties"]["score"]["enum"] == [*SCORES, None]
