"""Validate Jev's rounded output without changing reported scores or probabilities.

Jev can round each probability and its weighted score independently to two
decimal places. Their errors add, so the original strict sum check can reject a
valid response. Only vectors on that decimal grid receive the rounding bound.
"""

from bench_protocol import number

EPSILON = 1e-9


def _distribution(answer, expected):
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(expected):
        raise ValueError("Probability labels do not match the rubric")
    values = [number(value, 0, 1) for value in probabilities.values()]
    rounded = all(abs(value - round(value, 2)) <= EPSILON for value in values)
    tolerance = len(values) * 0.005 + EPSILON if rounded else 0.001
    if abs(sum(values) - 1) > tolerance:
        raise ValueError("Probability distribution exceeds its rounding bound")
    return probabilities, rounded


def normalize(response, questions):
    """Use native values; permit only the defined decimal-rounding error bound."""
    answers = response["answers"]
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Decision answer fields do not match the task")
    result = {}
    for key, question in questions.items():
        answer, kind = answers[key], question["type"]
        if kind not in {"noul", "choice", "score"} or not isinstance(answer, dict) or answer["type"] != kind:
            raise ValueError("Unexpected answer type")
        if "confidence" in answer:
            result[key + "_native_confidence"] = number(answer["confidence"], 0, 1)
        if kind == "noul":
            result[key] = number(answer["noul"], 0, 1)
            continue
        expected = question["criteria"] if kind == "choice" else list(map(str, range(5)))
        probabilities, rounded = None, False
        if "probabilities" in answer:
            probabilities, rounded = _distribution(answer, expected)
            result[key + "_probabilities"] = probabilities
        if kind == "choice":
            if answer["choice"] not in expected:
                raise ValueError("Unknown choice")
            result[key] = answer["choice"]
        else:
            score = number(answer["score"], 0, 4)
            if probabilities is not None:
                mean = sum(int(level) * value for level, value in probabilities.items())
                tolerance = 0.005 * sum(map(int, expected)) + 0.005 + EPSILON if rounded else 0.025
                if abs(mean - score) > tolerance:
                    raise ValueError("Score differs from its probability-weighted mean beyond rounding")
            result[key] = 1 + score
    result["evidence"] = {"kind": "typed_decision_only", "visible_explanation": None}
    return result
