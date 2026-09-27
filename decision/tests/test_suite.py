import math

import pytest

pytest.importorskip("decision_index")

from decision_index.engines import Unsupported, validate

from decision.schema import NOUL_OPTIONS
from decision.suite import build_answer, question_to_decision


def test_choice_question_maps_options_in_key_order() -> None:
    q = {"type": "choice", "instructions": {"task": "pick"}, "criteria": {"x": "Paris", "y": None, "z": ""}}
    d = question_to_decision("q1", {"city": "?"}, q)
    assert d.kind == "choice"
    assert d.options == ("Paris", "y", "z")
    assert d.state == '{"city":"?"}'
    assert d.question == '{"task":"pick"}'
    assert math.isclose(sum(d.target), 1.0)


def test_noul_question_uses_yes_no_and_empty_state() -> None:
    d = question_to_decision("h", "", {"type": "noul", "instructions": "Hallucinated?"})
    assert d.kind == "noul"
    assert d.options == NOUL_OPTIONS
    assert d.state == ""


@pytest.mark.parametrize(
    "q",
    [
        {"type": "choice", "instructions": "", "criteria": {"only": "one"}},
        {"type": "choice", "instructions": "", "criteria": {str(i): str(i) for i in range(256)}},
        {"type": "rank", "instructions": ""},
    ],
)
def test_unscorable_questions_are_unsupported(q: dict) -> None:
    with pytest.raises(Unsupported):
        question_to_decision("q", "s", q)


def test_answers_pass_kit_validation() -> None:
    questions = {
        "c": {"type": "choice", "instructions": "", "criteria": {"a": "A", "b": "B", "c": "C"}},
        "n": {"type": "noul", "instructions": ""},
    }
    answers = {"c": build_answer(questions["c"], [0.2, 0.5, 0.3]), "n": build_answer(questions["n"], [0.25, 0.75])}
    validate(questions, {"answers": answers})
    assert answers["c"]["choice"] == "b"
    assert answers["c"]["probabilities"] == pytest.approx({"a": 0.2, "b": 0.5, "c": 0.3})
    assert answers["n"]["noul"] == pytest.approx(0.75)
