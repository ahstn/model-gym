"""Offline checks for the Jev image contract and retained evidence."""

import base64
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.jev14 import client

SECRET = "test-private-openrouter-key-never-retain"


def quality_prompt():
    return {
        "context": "Assess the supplied image without inferring its source.",
        "questions": {
            "technical_quality": {
                "type": "score",
                "instructions": "Assess visible technical quality.",
                "criteria": ["Bad", "Poor", "Fair", "Good", "Excellent"],
            }
        },
    }


def response_body():
    return {
        "id": "gen-test",
        "model": client.REVISION,
        "provider": "TypeSafe",
        "answers": {
            "technical_quality": {
                "type": "score",
                "score": 2.7,
                "confidence": 0.8,
                "probabilities": {"0": 0, "1": 0, "2": 0.3, "3": 0.7, "4": 0},
            }
        },
        "usage": {"input_tokens": 1000, "output_tokens": 10, "cost": 0.000042},
    }


class Transport:
    def __init__(self, body=None, *, status=200, headers=None, raw=None, failure=None):
        self.calls = []
        self.failure = failure
        self.response = SimpleNamespace(
            status_code=status,
            headers=headers or {},
            text=raw if raw is not None else json.dumps(response_body() if body is None else body),
        )

    def __call__(self, endpoint, **arguments):
        self.calls.append((endpoint, arguments))
        if self.failure:
            raise self.failure
        return self.response


@pytest.fixture
def image_case(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    path = tmp_path / "view.jpg"
    Image.new("RGB", (512, 512), "blue").save(path, "JPEG")
    return tmp_path, {
        "image": path.name,
        "delivered_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "image_width": 512,
        "image_height": 512,
        "image_bytes": path.stat().st_size,
    }


def test_payload_sends_exact_frozen_image_once_without_labels(image_case):
    root, case = image_case
    case["gold"] = {"is_ai": True}
    prompt = quality_prompt()
    payload = client.make_payload(root, case, prompt)
    assert payload["model"] == "typesafe/jev-1.14"
    assert payload["questions"] == prompt["questions"]
    assert payload["state"][0] == prompt["context"]
    part = payload["state"][1]
    assert part["type"] == "image_url"
    prefix, data = part["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(data) == (root / case["image"]).read_bytes()
    assert len(payload["state"]) == 2
    assert "gold" not in json.dumps(payload)


@pytest.mark.parametrize(
    "dimensions,eligible",
    [
        ((256, 256), False),
        ((255, 512), False),
        ((512, 255), False),
        ((280, 280), True),
        ((256, 307), True),
        ((512, 512), True),
        ((1568, 1568), True),
        ((1569, 512), False),
        ((512, 1569), False),
    ],
)
def test_image_dimensions_and_area_limits(tmp_path, dimensions, eligible):
    path = tmp_path / "view.png"
    Image.new("RGB", dimensions).save(path, "PNG")
    assert client.image_info(path)["eligible"] is eligible


@pytest.mark.parametrize("format_name,eligible", [("JPEG", True), ("PNG", True), ("WEBP", False)])
def test_image_format_comes_from_bytes(tmp_path, format_name, eligible):
    path = tmp_path / "misleading.extension"
    Image.new("RGB", (512, 512)).save(path, format_name)
    assert client.image_info(path)["eligible"] is eligible


def test_animated_png_is_ineligible(tmp_path):
    path = tmp_path / "animated.png"
    Image.new("RGB", (512, 512), "red").save(
        path,
        "PNG",
        save_all=True,
        append_images=[Image.new("RGB", (512, 512), "blue")],
        duration=100,
        loop=0,
    )
    assert client.image_info(path)["eligible"] is False


def test_oversize_image_file_is_ineligible(image_case):
    root, case = image_case
    path = root / case["image"]
    with path.open("ab") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    assert client.image_info(path)["eligible"] is False


@pytest.mark.parametrize("changed", ["hash", "image_width", "image_height", "image_bytes", "too_small"])
def test_invalid_frozen_input_stops_before_network(image_case, changed):
    root, case = image_case
    if changed == "hash":
        case["delivered_sha256"] = "0" * 64
    elif changed == "too_small":
        Image.new("RGB", (256, 256)).save(root / case["image"], "JPEG")
    else:
        case[changed] += 1
    transport = Transport()
    with pytest.raises(ValueError):
        client.post(client.make_payload(root, case, quality_prompt()), quality_prompt(), transport=transport)
    assert transport.calls == []


def test_image_outside_experiment_is_rejected(image_case, tmp_path):
    _root, case = image_case
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    case["image"] = "../view.jpg"
    with pytest.raises(ValueError, match="within the experiment"):
        client.make_payload(experiment, case, quality_prompt())


@pytest.mark.parametrize("native,normalised", [(0, 1), (2.7, 3.7), (4, 5)])
def test_scores_convert_from_native_zero_based_scale_without_invented_probabilities(native, normalised):
    body = response_body()
    body["answers"]["technical_quality"] = {"type": "score", "score": native}
    result = client.normalize(body, quality_prompt()["questions"])
    assert result["technical_quality"] == normalised
    assert "technical_quality_probabilities" not in result
    assert "technical_quality_native_confidence" not in result
    assert result["evidence"]["visible_explanation"] is None


def test_optional_native_probabilities_and_confidence_are_retained():
    body = response_body()
    result = client.normalize(body, quality_prompt()["questions"])
    assert result["technical_quality"] == 3.7
    assert result["technical_quality_probabilities"] == body["answers"]["technical_quality"]["probabilities"]
    assert result["technical_quality_native_confidence"] == 0.8


@pytest.mark.parametrize(
    "mutation",
    [
        lambda answer: answer.update(score=-0.01),
        lambda answer: answer.update(score=4.01),
        lambda answer: answer.update(score=True),
        lambda answer: answer.update(score=float("nan")),
        lambda answer: answer.update(score=1),
        lambda answer: answer.update(type="noul"),
        lambda answer: answer.update(confidence=1.1),
        lambda answer: answer.update(probabilities=None),
        lambda answer: answer["probabilities"].pop("0"),
        lambda answer: answer["probabilities"].update({"3": 0.6}),
        lambda answer: answer["probabilities"].update({"3": True}),
        lambda answer: answer["probabilities"].update({"3": float("nan")}),
    ],
)
def test_malformed_score_values_and_distributions_are_rejected(mutation):
    body = response_body()
    mutation(body["answers"]["technical_quality"])
    with pytest.raises((TypeError, ValueError, KeyError)):
        client.normalize(body, quality_prompt()["questions"])


def ai_questions():
    return {
        "ai_probability": {"type": "noul"},
        "medium": {"type": "choice", "criteria": {"photo": "Photo", "drawing": "Drawing"}},
    }


def test_noul_and_choice_preserve_values_without_optional_distributions():
    body = {
        "answers": {
            "ai_probability": {"type": "noul", "noul": 0.72},
            "medium": {"type": "choice", "choice": "photo"},
        }
    }
    result = client.normalize(body, ai_questions())
    assert result["ai_probability"] == 0.72
    assert result["medium"] == "photo"
    assert "medium_probabilities" not in result


@pytest.mark.parametrize("noul", [-0.1, 1.1, True, "0.7", float("nan")])
def test_invalid_noul_does_not_become_a_negative_detection(noul):
    with pytest.raises((TypeError, ValueError)):
        client.normalize(
            {"answers": {"ai_probability": {"type": "noul", "noul": noul}}},
            {"ai_probability": {"type": "noul"}},
        )


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "other"},
        {"type": "choice", "choice": "photo", "probabilities": {"photo": 1}},
        {"type": "choice", "choice": "photo", "probabilities": {"photo": 0.6, "drawing": 0.1}},
    ],
)
def test_malformed_choice_is_rejected(answer):
    with pytest.raises(ValueError):
        client.normalize({"answers": {"medium": answer}}, {"medium": ai_questions()["medium"]})


def test_post_uses_explicit_headers_and_retains_cost_revision_and_usage(image_case):
    root, case = image_case
    prompt = quality_prompt()
    transport = Transport(headers={"x-request-id": "req-test"})
    result = client.post(client.make_payload(root, case, prompt), prompt, transport=transport)
    endpoint, arguments = transport.calls[0]
    assert endpoint == "https://openrouter.ai/api/alpha/decisions"
    assert arguments["headers"]["Authorization"] == "Bearer " + SECRET
    assert all(arguments["headers"][key] == "true" for key in client.HEADERS)
    assert arguments["allow_redirects"] is False
    assert arguments["timeout"] == (15, 180)
    assert len(transport.calls) == 1
    assert result["resolved_model"] == client.REVISION
    assert result["usage"]["input_tokens"] == 1000
    assert result["cost_usd"] == 0.000042
    assert result["effective_cost_usd"] == 0.000042
    assert result["normalized"]["technical_quality"] == 3.7
    assert result["request_id"] == "req-test"
    assert result["seconds"] >= 0
    assert SECRET not in json.dumps(result)
    assert "Authorization" not in result["request_headers"]


def test_reflected_credentials_in_response_are_redacted(image_case):
    body = response_body()
    body["echo"] = SECRET
    result = client.post({"model": client.MODEL}, quality_prompt(), transport=Transport(body))
    assert SECRET not in json.dumps(result)
    assert result["response"]["echo"] == "[REDACTED]"


def test_reflected_credential_in_request_id_is_not_retained(image_case):
    result = client.post({}, quality_prompt(), transport=Transport(headers={"x-request-id": SECRET}))
    assert SECRET not in json.dumps(result)


def test_absent_key_stops_before_network(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    transport = Transport()
    with pytest.raises(ValueError, match="unavailable"):
        client.post({}, quality_prompt(), transport=transport)
    assert transport.calls == []


@pytest.mark.parametrize("usage", [{}, {"input_tokens": 1000}, {"cost": None}])
def test_missing_cost_is_unknown_not_zero(image_case, usage):
    body = response_body()
    body["usage"] = usage
    result = client.post({}, quality_prompt(), transport=Transport(body))
    assert result["cost_usd"] is None
    assert result["effective_cost_usd"] is None
    assert result["cost_basis"] == "unknown_openrouter_cost"
    assert result["normalized"] is not None


@pytest.mark.parametrize("cost", [-1, True, "0.01", float("nan")])
def test_invalid_cost_remains_unresolved(image_case, cost):
    body = response_body()
    body["usage"]["cost"] = cost
    result = client.post({}, quality_prompt(), transport=Transport(body))
    assert result["cost_usd"] is None
    assert result["effective_cost_usd"] is None
    assert result["cost_validation_error"] == "Invalid reported cost"


def test_revision_drift_is_not_counted_as_a_valid_prediction(image_case):
    body = response_body()
    body["model"] = "typesafe/jev-1.14-20261016"
    result = client.post({}, quality_prompt(), transport=Transport(body))
    assert result["normalized"] is None
    assert "frozen revision" in result["validation_error"]
    assert result["resolved_model"] == body["model"]
    assert result["cost_usd"] == 0.000042


@pytest.mark.parametrize("tokens", [262145, -1, True, "1000"])
def test_invalid_or_over_context_usage_is_rejected(image_case, tokens):
    body = response_body()
    body["usage"]["input_tokens"] = tokens
    result = client.post({}, quality_prompt(), transport=Transport(body))
    assert result["normalized"] is None
    assert "validation_error" in result


@pytest.mark.parametrize("status", [400, 401, 413, 429, 502])
def test_http_failure_has_no_prediction_and_is_not_retried(image_case, status):
    transport = Transport({"error": {"message": "upstream error"}}, status=status)
    result = client.post({}, quality_prompt(), transport=transport)
    assert result["status"] == status
    assert result["normalized"] is None
    assert "HTTP failure" in result["validation_error"]
    assert len(transport.calls) == 1
    assert result["cost_usd"] is None


def test_transport_failure_retains_type_without_sensitive_exception(image_case):
    transport = Transport(failure=requests.Timeout("request included " + SECRET))
    result = client.post({}, quality_prompt(), transport=transport)
    assert result["transport_error"] == "Timeout"
    assert result["normalized"] is None
    assert result["cost_usd"] is None
    assert SECRET not in json.dumps(result)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("raw", ["upstream unavailable", "[]", "null"])
def test_nonobject_or_invalid_json_has_no_prediction(image_case, raw):
    result = client.post({}, quality_prompt(), transport=Transport(raw=raw))
    assert result["normalized"] is None
    assert result["validation_error"] == "No JSON response object"


def test_answer_keys_must_match_prompt_exactly():
    body = copy.deepcopy(response_body())
    body["answers"]["unexpected"] = {"type": "noul", "noul": 0.2}
    with pytest.raises(ValueError, match="fields do not match"):
        client.normalize(body, quality_prompt()["questions"])
