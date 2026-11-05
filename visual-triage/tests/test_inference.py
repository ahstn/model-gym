import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import QUESTIONS, normalize_chat, normalize_decisions, shadow_route


def decisions():
    answers = {}
    for name, q in QUESTIONS.items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.1 if name == "severe_defect" else 0.9}
        elif q["type"] == "score":
            answers[name] = {
                "type": "score",
                "score": 2,
                "probabilities": {str(i): float(i == 2) for i in range(5)},
            }
        else:
            label = "camera_photo" if name == "origin" else "clear"
            answers[name] = {
                "type": "choice",
                "choice": label,
                "probabilities": {k: float(k == label) for k in q["criteria"]},
            }
    return {"answers": answers}


def test_scores_are_one_based_and_unscorable_is_null():
    data = decisions()
    assert normalize_decisions(data)["technical_quality"] == 3
    data["answers"]["scorable"]["noul"] = 0.2
    out = normalize_decisions(data)
    assert out["technical_quality"] is None and out["aesthetics"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_probability_is_not_an_allow(value):
    data = decisions()
    data["answers"]["scorable"]["noul"] = value
    with pytest.raises((ValueError, TypeError)):
        normalize_decisions(data)
    assert shadow_route(None) == "review_error"


def test_missing_or_incomplete_answers_fail():
    data = decisions()
    del data["answers"]["origin"]["probabilities"]["unknown"]
    with pytest.raises(ValueError):
        normalize_decisions(data)
    data = decisions()
    del data["answers"]["sexual_content"]
    with pytest.raises(KeyError):
        normalize_decisions(data)


def test_separate_medium_does_not_zero_quality():
    out = normalize_decisions(decisions())
    out["origin"] = "diagram"
    assert shadow_route(out) == "separate_medium"
    assert out["technical_quality"] == 3


def test_origin_suspicion_never_authorizes_deletion():
    out = normalize_decisions(decisions())
    out["origin"] = "ai_suspected"
    assert shadow_route(out) == "review_origin"


def test_nudity_uses_review_not_clear():
    out = normalize_decisions(decisions())
    out["sexual_content"] = "nonsexual_nudity"
    assert shadow_route(out) == "review_sexual_content"


def test_vision_null_and_schema_validation():
    out = {
        "description": "Indistinct image.",
        "scorable": False,
        "technical_quality": None,
        "aesthetics": None,
        "severe_defect": True,
        "origin": "unknown",
        "sexual_content": "unknown",
    }
    response = lambda x: {"choices": [{"message": {"content": json.dumps(x)}}]}
    assert normalize_chat(response(out)) == out
    wrong = copy.deepcopy(out)
    wrong["technical_quality"] = 0
    with pytest.raises(ValueError):
        normalize_chat(response(wrong))


def test_image_hash_checked_before_network(tmp_path, monkeypatch):
    import inference

    image = tmp_path / "image.jpg"
    image.write_bytes(b"changed")
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        inference.image_part({"image": "image.jpg", "delivered_sha256": "incorrect"})


def test_text_stage_does_not_receive_pixels_or_source_labels():
    from inference import decision_payload

    payload = decision_payload("typesafe/jev-1.13", description="A blurred scene; origin unclear.")
    assert payload["state"]["observations"] == "A blurred scene; origin unclear."
    assert "image_url" not in json.dumps(payload)
    assert "source_labels" not in json.dumps(payload)


def test_transport_does_not_persist_credentials(monkeypatch):
    import inference

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-a-real-key")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"usage": {"cost": 0.01}}

    calls = []

    def request(url, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(inference.requests, "post", request)
    result = inference.post(inference.DECISIONS, {"model": "test", "state": "text"})
    assert len(calls) == 1
    assert calls[0]["allow_redirects"] is False
    assert "test-secret" not in json.dumps(result)
    assert "headers" not in result


def test_transport_failure_is_not_retried(monkeypatch):
    import inference

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise inference.requests.Timeout()

    monkeypatch.setattr(inference.requests, "post", request)
    result = inference.post(inference.DECISIONS, {"model": "test"})
    assert len(calls) == 1
    assert result["error"] == "Timeout"


def test_initial_runner_signature_ignores_new_benchmark_prompt_directory(tmp_path, monkeypatch):
    import run

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/context.txt").write_text("Original prompt")
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/samples.json").write_text("{}")
    (tmp_path / "inference.py").write_text("Original inference")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    original = run.signature()
    (tmp_path / "prompts/trials").mkdir()
    (tmp_path / "prompts/trials/quality-r1.json").write_text("New benchmark prompt")
    assert run.signature() == original


def test_initial_runner_requires_archive_before_loading_history(tmp_path, monkeypatch):
    import run

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    checks = []

    def require(root, scope=None):
        checks.append((root, scope))
        raise ValueError("Archived evidence is missing; restore it before continuing")

    monkeypatch.setattr(run, "require_restored", require)
    monkeypatch.setattr(run, "post", lambda *args: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="Archived evidence"):
        run.main()
    assert checks == [(tmp_path, None)]
