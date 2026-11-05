import copy
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_protocol as protocol
import bench_runner as runner


@pytest.fixture
def quality_prompt():
    return {
        "id": "quality-test-v1",
        "task": "quality",
        "context": "Assess visible fidelity and visual appeal independently.",
        "questions": {
            field: {
                "type": "score",
                "instructions": f"Rate {field}.",
                "criteria": ["bad", "poor", "fair", "good", "excellent"],
            }
            for field in ("technical_quality", "aesthetics")
        },
    }


@pytest.fixture
def ai_prompt():
    return {
        "id": "ai-test-v1",
        "task": "ai",
        "context": "Pixels alone cannot verify provenance.",
        "questions": {
            "ai_probability": {"type": "noul", "instructions": "Estimate AI origin probability."},
            "medium": {
                "type": "choice",
                "instructions": "Choose visible medium.",
                "criteria": {
                    "photo": "Camera-like image",
                    "render": "Conventional rendered image",
                    "other": "Other medium",
                },
            },
            "uncertainty": {"type": "noul", "instructions": "Estimate uncertainty from visible cues."},
        },
    }


def decision_response(prompt, score=2):
    return {
        "model": "resolved-model",
        "provider": "test-provider",
        "usage": {"cost": 0.001, "input_tokens": 42},
        "answers": {
            field: {
                "type": "score",
                "score": score,
                "probabilities": {str(i): float(i == score) for i in range(5)},
            }
            for field in prompt["questions"]
        },
    }


def chat_response(data, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(data)}}]}


def test_quality_distribution_one_based_without_blank_abstention(quality_prompt):
    out = protocol.normalize_decisions(
        decision_response(quality_prompt, score=0), quality_prompt["questions"]
    )
    assert out["technical_quality"] == out["aesthetics"] == 1
    assert out["evidence"] == {
        "kind": "selected_rubric_only",
        "criteria": {"technical_quality": "bad", "aesthetics": "bad"},
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_probability_rejected(bad, quality_prompt):
    response = decision_response(quality_prompt)
    response["answers"]["aesthetics"]["probabilities"]["0"] = bad
    with pytest.raises((ValueError, TypeError)):
        protocol.normalize_decisions(response, quality_prompt["questions"])


def test_distribution_must_cover_rubric_sum_to_one_and_match_score(quality_prompt):
    response = decision_response(quality_prompt)
    response["answers"]["aesthetics"]["score"] = 4
    with pytest.raises(ValueError, match="weighted mean"):
        protocol.normalize_decisions(response, quality_prompt["questions"])
    response = decision_response(quality_prompt)
    response["answers"]["aesthetics"]["probabilities"]["0"] = 0.1
    with pytest.raises(ValueError, match="sum to one"):
        protocol.normalize_decisions(response, quality_prompt["questions"])
    del response["answers"]["aesthetics"]["probabilities"]["0"]
    with pytest.raises(ValueError, match="labels"):
        protocol.normalize_decisions(response, quality_prompt["questions"])


def test_ai_probability_does_not_conflate_conventional_render(ai_prompt):
    response = {
        "answers": {
            "ai_probability": {"type": "noul", "noul": 0.05},
            "uncertainty": {"type": "noul", "noul": 0.6},
            "medium": {
                "type": "choice",
                "choice": "render",
                "probabilities": {"photo": 0.01, "render": 0.98, "other": 0.01},
            },
        }
    }
    result = protocol.normalize_decisions(response, ai_prompt["questions"])
    assert result["medium"] == "render"
    assert result["ai_probability"] == 0.05
    assert result["uncertainty"] == 0.6


def test_chat_validates_scores_fields_and_incomplete_output(quality_prompt):
    output = {"technical_quality": 1, "aesthetics": 2.5, "evidence": "Little detail remains visible."}
    assert protocol.normalize_chat(chat_response(output), quality_prompt["questions"]) == output
    with pytest.raises(ValueError, match="incomplete"):
        protocol.normalize_chat(chat_response(output, "length"), quality_prompt["questions"])
    for invalid in (None, True, 0, 5.1, float("nan")):
        output["technical_quality"] = invalid
        with pytest.raises((TypeError, ValueError)):
            protocol.normalize_chat(chat_response(output), quality_prompt["questions"])


def test_chat_refusal_is_invalid(quality_prompt):
    response = {"choices": [{"message": {"content": "{}", "refusal": "Declined"}}]}
    with pytest.raises(ValueError, match="refused"):
        protocol.normalize_chat(response, quality_prompt["questions"])


def test_image_hash_and_format_checked_without_source_information(tmp_path, quality_prompt):
    path = tmp_path / "blank.png"
    Image.new("RGB", (12, 12)).save(path)
    sample = {
        "id": "secret-id",
        "image": "blank.png",
        "source": "hidden-source",
        "labels": {"gold": 1},
        "delivered_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    image = protocol.image_part(sample, tmp_path)
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    for arm in protocol.MODELS:
        _, payload = protocol.make_request(arm, image, quality_prompt)
        serialized = json.dumps(payload)
        assert all(
            secret not in serialized for secret in ("secret-id", "hidden-source", "gold", "delivered_sha256")
        )
    sample["delivered_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.image_part(sample, tmp_path)
    sample["image"] = "../outside.png"
    with pytest.raises(ValueError, match="within"):
        protocol.image_part(sample, tmp_path)


def test_reasoning_is_explicit_and_sampling_parameters_absent(quality_prompt):
    for arm in ("gemini", "luna"):
        _, payload = protocol.make_request(arm, {"type": "image_url"}, quality_prompt, "low")
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["max_tokens"] == 4096
        assert "temperature" not in payload and "top_p" not in payload
        assert payload["provider"]["require_parameters"] is True


def frozen_context(tmp_path):
    from tracks.image_context import extract_image_context, model_context

    path = tmp_path / "metadata.png"
    Image.new("RGB", (20, 10), (80, 100, 120)).save(path)
    return model_context(extract_image_context(path))


def test_optional_context_changes_request_without_mutating_prompt(tmp_path, quality_prompt):
    context = frozen_context(tmp_path)
    original = copy.deepcopy(quality_prompt)
    image = {"type": "image_url", "image_url": {"url": "test"}}
    for arm in protocol.MODELS:
        _, plain = protocol.make_request(arm, image, quality_prompt)
        _, supplied = protocol.make_request(arm, image, quality_prompt, extra_context=context)
        assert protocol.digest(plain) != protocol.digest(supplied)
        assert "encoded_rgb_luma_quantiles" not in json.dumps(plain)
        assert "encoded_rgb_luma_quantiles" in json.dumps(supplied)
        assert "native_source" not in json.dumps(supplied)
    assert quality_prompt == original
    context["source_label"] = "generated"
    with pytest.raises((ValueError, TypeError)):
        protocol.make_request("decider", image, quality_prompt, extra_context=context)


def test_metadata_preflight_checks_frozen_hash_before_paid_call(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    context = frozen_context(tmp_path)
    manifest = json.loads(args.manifest.read_text())
    manifest["samples"][0].update(model_context=context, model_context_sha256="wrong")
    args.manifest.write_text(json.dumps(manifest))
    args.sample_context_mode = "pixel_stats"
    calls = []
    monkeypatch.setattr(runner, "post", lambda *arguments: calls.append(arguments))
    with pytest.raises(ValueError, match="context hash"):
        runner.run(args)
    assert calls == []
    assert not args.output.exists()


def test_pixel_statistics_rejects_even_hash_valid_camera_metadata(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    context = frozen_context(tmp_path)
    context["camera"] = {"iso": 100, "exposure_seconds": None, "f_number": None, "focal_length_mm": None}
    manifest = json.loads(args.manifest.read_text())
    manifest["samples"][0].update(model_context=context, model_context_sha256=protocol.digest(context))
    args.manifest.write_text(json.dumps(manifest))
    args.sample_context_mode = "pixel_stats"
    calls = []
    monkeypatch.setattr(runner, "post", lambda *arguments: calls.append(arguments))
    with pytest.raises(ValueError, match="camera"):
        runner.run(args)
    with pytest.raises(ValueError, match="camera"):
        protocol.make_request("decider", {"type": "image_url"}, quality_prompt, extra_context=context)
    assert calls == []


def test_metadata_request_records_identity_and_resumes(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    context = frozen_context(tmp_path)
    manifest = json.loads(args.manifest.read_text())
    manifest["samples"][0].update(model_context=context, model_context_sha256=protocol.digest(context))
    args.manifest.write_text(json.dumps(manifest))
    args.sample_context_mode = "pixel_stats"
    calls = []

    def post(endpoint, payload):
        calls.append(payload)
        return {"status": 200, "seconds": 0.01, "response": decision_response(quality_prompt)}

    monkeypatch.setattr(runner, "post", post)
    runner.run(args)
    runner.run(args)
    assert len(calls) == 1
    record = json.loads(args.output.read_text())
    assert record["model_context_sha256"] == protocol.digest(context)
    assert record["sample_context_mode"] == "pixel_stats"
    args.sample_context_mode = "image_only"
    with pytest.raises(ValueError, match="Request changed"):
        runner.run(args)


def test_prompt_validation_disallows_unscorable_branch(quality_prompt, ai_prompt):
    protocol.validate_prompt(quality_prompt, "quality")
    protocol.validate_prompt(ai_prompt, "ai")
    bad = copy.deepcopy(quality_prompt)
    bad["questions"]["scorable"] = {"type": "noul", "instructions": "Is image scorable?"}
    with pytest.raises(ValueError, match="task fields"):
        protocol.validate_prompt(bad, "quality")


def test_transport_retains_error_without_retry_or_secret(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    attempts = []

    def request(*args, **kwargs):
        attempts.append(kwargs)
        raise requests_timeout("Sensitive request text: test-secret")

    requests_timeout = runner.requests.Timeout
    monkeypatch.setattr(runner.requests, "post", request)
    result = runner.post("https://example.invalid", {"model": "test"})
    assert len(attempts) == 1
    assert attempts[0]["allow_redirects"] is False
    assert result["transport_error"] == "Timeout"
    assert "test-secret" not in json.dumps(result)


def setup_run(tmp_path, monkeypatch, prompt):
    manifest_path = tmp_path / "manifest.json"
    prompt_path = tmp_path / "prompt.json"
    manifest_path.write_text(
        json.dumps(
            {"benchmark": "test", "samples": [{"id": "one", "image": "test.jpg", "delivered_sha256": "sha"}]}
        )
    )
    prompt_path.write_text(json.dumps(prompt))
    monkeypatch.setattr(
        runner, "image_part", lambda sample: {"type": "image_url", "image_url": {"url": "test"}}
    )
    return Namespace(
        manifest=manifest_path,
        prompt=prompt_path,
        task="quality",
        ids=None,
        limit=None,
        models=["decider"],
        output=tmp_path / "ledger.jsonl",
        effort="medium",
        max_tokens=4096,
        max_calls=120,
        reserve_usd=0.1,
        budget_usd=10,
    )


def test_resume_skips_completed_and_rejects_changed_request(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    calls = []

    def post(endpoint, payload):
        calls.append(payload)
        return {"status": 200, "seconds": 0.01, "response": decision_response(quality_prompt)}

    monkeypatch.setattr(runner, "post", post)
    runner.run(args)
    runner.run(args)
    assert len(calls) == 1
    quality_prompt["context"] += " Changed rubric."
    args.prompt.write_text(json.dumps(quality_prompt))
    with pytest.raises(ValueError, match="Request changed"):
        runner.run(args)
    assert len(calls) == 1


def test_unknown_cost_is_written_then_stops_and_cannot_resume(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    args.models = ["decider", "gemini"]
    attempts = []

    def post(endpoint, payload):
        attempts.append(payload)
        return {"status": 503, "seconds": 0.01, "response": {"error": {"message": "Unavailable"}}}

    monkeypatch.setattr(runner, "post", post)
    with pytest.raises(ValueError, match="Unknown request cost"):
        runner.run(args)
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(attempts) == len(records) == 1
    assert records[0]["normalized"] is None and records[0]["status"] == 503
    with pytest.raises(ValueError, match="unknown cost"):
        runner.run(args)
    assert len(attempts) == 1


def test_spend_guard_prevents_request(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    args.budget_usd = 0.05
    monkeypatch.setattr(runner, "post", lambda *args: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="spend guard"):
        runner.run(args)
    assert not args.output.exists()


def test_reasoning_usage_preserved_without_claiming_observed_effort(quality_prompt):
    data = {"technical_quality": 4, "aesthetics": 3, "evidence": "Clear detail, ordinary framing."}
    response = chat_response(data)
    response["usage"] = {"cost": 0.012, "completion_tokens_details": {"reasoning_tokens": 512}}
    record = runner.response_record({"status": 200, "response": response}, "luna", quality_prompt, "medium")
    assert record["requested_reasoning"] == {"effort": "medium"}
    assert record["observed_reasoning"]["effort"] is None
    assert record["observed_reasoning"]["completion_tokens_details"]["reasoning_tokens"] == 512
    assert record["cost_usd"] == 0.012


def test_invalid_result_with_known_cost_remains_a_completed_attempt(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    attempts = []

    def post(endpoint, payload):
        attempts.append(payload)
        return {"status": 200, "seconds": 0.01, "response": {"usage": {"cost": 0.001}}}

    monkeypatch.setattr(runner, "post", post)
    runner.run(args)
    runner.run(args)
    assert len(attempts) == 1
    assert json.loads(args.output.read_text())["normalized"] is None


def test_byok_preserves_router_charge_and_counts_upstream(quality_prompt):
    response = decision_response(quality_prompt)
    response["usage"] = {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.0001036}}
    record = runner.response_record(
        {"status": 200, "response": response}, "decider", quality_prompt, "medium"
    )
    assert record["cost_usd"] == 0
    assert record["effective_cost_usd"] == 0.0001036
    assert record["cost_basis"] == "byok_upstream_plus_openrouter_charge"
    cost, _ = runner.effective_cost(
        {"cost": 0.001, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.01}}
    )
    assert cost == pytest.approx(0.011)


def test_old_byok_ledger_uses_raw_usage_without_rewriting(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    record = {
        "benchmark": "old",
        "task": "quality",
        "sample": "one",
        "arm": "luna",
        "cost_usd": 0,
        "response": {
            "usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.04}}
        },
    }
    ledger.write_text(json.dumps(record) + "\n")
    original = ledger.read_bytes()
    records, _, spent = runner.load_ledger(ledger)
    assert spent == 0.04
    assert records == [record]
    assert ledger.read_bytes() == original


def test_unknown_byok_upstream_stops_after_persisting_response(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    args.models = ["decider", "gemini"]
    calls = []

    def post(endpoint, payload):
        calls.append(payload)
        response = decision_response(quality_prompt)
        response["usage"] = {"cost": 0, "is_byok": True}
        return {"status": 200, "seconds": 0.01, "response": response}

    monkeypatch.setattr(runner, "post", post)
    with pytest.raises(ValueError, match="Unknown request cost"):
        runner.run(args)
    record = json.loads(args.output.read_text())
    assert len(calls) == 1
    assert record["cost_usd"] == 0 and record["effective_cost_usd"] is None
    assert record["cost_basis"] == "unknown_byok_upstream_cost"
    with pytest.raises(ValueError, match="unknown cost"):
        runner.run(args)
    assert len(calls) == 1


def test_resume_spend_guard_counts_old_byok_cost(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    args.output.write_text(
        json.dumps(
            {
                "benchmark": "other",
                "task": "quality",
                "sample": "old",
                "arm": "luna",
                "cost_usd": 0,
                "usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.95}},
            }
        )
        + "\n"
    )
    args.budget_usd = 1
    monkeypatch.setattr(runner, "post", lambda *args: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="spend guard"):
        runner.run(args)


@pytest.mark.parametrize("upstream", [float("nan"), float("inf"), -1, True])
def test_invalid_byok_upstream_is_not_free(upstream, quality_prompt):
    response = decision_response(quality_prompt)
    response["usage"] = {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": upstream}}
    record = runner.response_record(
        {"status": 200, "response": response}, "decider", quality_prompt, "medium"
    )
    assert record["cost_usd"] == 0 and record["effective_cost_usd"] is None
    assert record["cost_basis"] == "invalid_reported_cost"


@pytest.mark.parametrize("choice", [None, "invalid", {"message": None}, {"message": []}])
def test_malformed_chat_container_is_recorded_without_retry(tmp_path, monkeypatch, quality_prompt, choice):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    args.models = ["gemini"]
    calls = []

    def post(endpoint, payload):
        calls.append(payload)
        return {"status": 200, "seconds": 0.01, "response": {"usage": {"cost": 0.001}, "choices": [choice]}}

    monkeypatch.setattr(runner, "post", post)
    runner.run(args)
    runner.run(args)
    assert len(calls) == 1
    record = json.loads(args.output.read_text())
    assert record["normalized"] is None
    assert record["validation_error"].startswith("TypeError: Chat")
    assert record["response"]["choices"] == [choice]


def test_historical_manifest_requires_archive_even_for_fresh_output(tmp_path, monkeypatch, quality_prompt):
    args = setup_run(tmp_path, monkeypatch, quality_prompt)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    args.output = tmp_path / "new-run.jsonl"
    checks = []

    def require(root, scope=None):
        checks.append((root, scope))
        raise ValueError("Archived evidence is missing; restore it before continuing")

    monkeypatch.setattr(runner, "require_restored", require)
    monkeypatch.setattr(runner, "post", lambda *args: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="Archived evidence"):
        runner.run(args)
    assert checks == [(tmp_path, None)]
    assert not args.output.exists()


def test_archived_ledger_cannot_be_loaded_as_an_empty_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    ledger = tmp_path / "evidence/final/quality.jsonl"
    checks = []

    def require(root, scope=None):
        checks.append((root, scope))
        raise ValueError("Archived evidence is missing; restore it before continuing")

    monkeypatch.setattr(runner, "require_restored", require)
    with pytest.raises(ValueError, match="Archived evidence"):
        runner.load_ledger(ledger)
    assert checks == [(tmp_path, "evidence/final/quality.jsonl")]
