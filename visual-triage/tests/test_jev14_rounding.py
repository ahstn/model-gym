"""Regression checks for independently rounded Jev probabilities and scores."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.jev14 import client, rounding


def quality_questions():
    return {
        name: {"type": "score", "criteria": ["Bad", "Poor", "Fair", "Good", "Excellent"]}
        for name in ("technical_quality", "aesthetics")
    }


def captured_response():
    # Exact answer values from primary/quality/quality-012, attempt 31.
    return {
        "answers": {
            "technical_quality": {
                "type": "score",
                "score": 2.42,
                "confidence": 0.5,
                "probabilities": {"0": 0, "1": 0.08, "2": 0.47, "3": 0.38, "4": 0.07},
            },
            "aesthetics": {
                "type": "score",
                "score": 2.58,
                "confidence": 0.57,
                "probabilities": {"0": 0, "1": 0.06, "2": 0.34, "3": 0.54, "4": 0.05},
            },
        }
    }


def test_captured_quantized_response_is_valid_without_rewriting_values():
    body = captured_response()
    original = copy.deepcopy(body)
    with pytest.raises(ValueError, match="sum to one"):
        client.normalize(body, quality_questions())
    parsed = rounding.normalize(body, quality_questions())
    assert parsed["technical_quality"] == 3.42
    assert parsed["aesthetics"] == 3.58
    assert parsed["aesthetics_probabilities"] == original["answers"]["aesthetics"]["probabilities"]
    assert sum(parsed["aesthetics_probabilities"].values()) == pytest.approx(0.99)
    assert parsed["aesthetics_native_confidence"] == 0.57
    assert body == original


def test_regular_valid_response_matches_frozen_parser():
    body = captured_response()
    body["answers"].pop("aesthetics")
    questions = {"technical_quality": quality_questions()["technical_quality"]}
    assert rounding.normalize(body, questions) == client.normalize(body, questions)


@pytest.mark.parametrize("total,valid", [(0.98, True), (1.02, True), (0.97, False), (1.03, False)])
def test_five_level_distribution_uses_only_bounded_quantization_error(total, valid):
    body = {
        "answers": {
            "q": {"type": "score", "score": 0, "probabilities": {"0": total, "1": 0, "2": 0, "3": 0, "4": 0}}
        }
    }
    # Keep every value in range when the rounded total exceeds one.
    if total > 1:
        body["answers"]["q"]["probabilities"].update({"0": 1, "1": round(total - 1, 2)})
    if valid:
        rounding.normalize(body, {"q": quality_questions()["technical_quality"]})
    else:
        with pytest.raises(ValueError):
            rounding.normalize(body, {"q": quality_questions()["technical_quality"]})


@pytest.mark.parametrize("difference,valid", [(0.055, True), (0.056, False)])
def test_weighted_score_bound_includes_rounding_of_all_five_levels(difference, valid):
    body = {
        "answers": {
            "q": {
                "type": "score",
                "score": 2 + difference,
                "probabilities": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0},
            }
        }
    }
    if valid:
        assert (
            rounding.normalize(body, {"q": quality_questions()["technical_quality"]})["q"] == 3 + difference
        )
    else:
        with pytest.raises(ValueError, match="weighted mean"):
            rounding.normalize(body, {"q": quality_questions()["technical_quality"]})


def test_nonquantized_vector_keeps_strict_sum_and_score_limits():
    question = {"q": quality_questions()["technical_quality"]}
    body = {
        "answers": {
            "q": {
                "type": "score",
                "score": 2.4,
                "probabilities": {"0": 0, "1": 0, "2": 0.5901, "3": 0.3999, "4": 0},
            }
        }
    }
    with pytest.raises(ValueError, match="rounding bound"):
        rounding.normalize(body, question)
    body["answers"]["q"].update(
        score=2.4301, probabilities={"0": 0, "1": 0, "2": 0.5999, "3": 0.4001, "4": 0}
    )
    with pytest.raises(ValueError, match="weighted mean"):
        rounding.normalize(body, question)


def test_choice_uses_number_of_options_for_rounding_bound():
    questions = {"medium": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}}
    body = {
        "answers": {
            "medium": {"type": "choice", "choice": "a", "probabilities": {"a": 0.33, "b": 0.33, "c": 0.33}}
        }
    }
    result = rounding.normalize(body, questions)
    assert result["medium_probabilities"] == {"a": 0.33, "b": 0.33, "c": 0.33}
    body["answers"]["medium"]["probabilities"]["a"] = 0.32
    with pytest.raises(ValueError, match="rounding bound"):
        rounding.normalize(body, questions)


def test_optional_probabilities_stay_absent_and_noul_is_not_rescaled():
    questions = {"q": quality_questions()["technical_quality"], "ai": {"type": "noul"}}
    body = {"answers": {"q": {"type": "score", "score": 0}, "ai": {"type": "noul", "noul": 0.17}}}
    result = rounding.normalize(body, questions)
    assert result["q"] == 1 and result["ai"] == 0.17
    assert "q_probabilities" not in result


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(score=5),
        lambda a: a.update(score=True),
        lambda a: a.update(confidence=-0.1),
        lambda a: a.update(type="choice"),
        lambda a: a["probabilities"].pop("0"),
        lambda a: a["probabilities"].update({"0": -0.01}),
        lambda a: a["probabilities"].update({"0": True}),
        lambda a: a["probabilities"].update({"0": float("nan")}),
    ],
)
def test_rounding_does_not_accept_gross_or_malformed_values(mutation):
    body = captured_response()
    mutation(body["answers"]["technical_quality"])
    with pytest.raises((ValueError, TypeError)):
        rounding.normalize(body, quality_questions())
