"""End-to-end local pilot tests using real views and a simulated provider."""

import json
from pathlib import Path

import pytest
from PIL import Image

from visual_quality import labelling
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl
from visual_quality.views import prepare_views

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "visual-rating-v1"


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    data = tmp_path / "data"
    dataset = data / "dataset"
    dataset.mkdir(parents=True)
    rows = []
    for index, component in enumerate(["megalith", "vizwiz", "distortions"]):
        image = data / f"{index}.png"
        Image.new("RGB", (16, 12), (index * 60, 20, 30)).save(image)
        rows.append(
            {
                "id": str(index),
                "sha256": file_sha256(image),
                "image_path": image.name,
                "split": "train",
                "split_group": str(index),
                "component": component,
                "photo_domain_eligible": None,
            }
        )
    atomic_json(
        dataset / "dataset.json",
        {
            "schema_version": 1,
            "image_root": "..",
            "manifests": {"train": "train.jsonl"},
        },
    )
    write_jsonl(dataset / "train.jsonl", rows)
    views = tmp_path / "views"
    prepare_views(dataset, views, limit=3)
    info = {
        "id": labelling.DEFAULT_MODEL,
        "context_length": 1_048_576,
        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
    }
    monkeypatch.setattr(labelling.openrouter, "model_info", lambda model: info)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    return views, tmp_path / "run"


def response(*, cost=0.001, rating=None, model=labelling.DEFAULT_MODEL, finish="stop"):
    body = {
        "model": model,
        "provider": "Test",
        "usage": {"cost": cost},
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": json.dumps(
                        rating
                        or {
                            "task": "technical_quality",
                            "status": "ok",
                            "score": 3.7,
                            "evidence": "Some detail is blurred.",
                            "limitations": [],
                        }
                    )
                },
            }
        ],
    }
    return {"http_status": 200, "body": json.dumps(body), "retry_after": None, "transport_error": None}


def test_pilot_sends_frozen_image_and_resumes_without_paid_duplicates(pilot, monkeypatch):
    views, output = pilot
    calls = []

    def complete(key, payload):
        assert key == "test-secret"
        calls.append(payload)
        assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["response_format"]["json_schema"]["strict"] is True
        return response()

    monkeypatch.setattr(labelling.openrouter, "complete", complete)
    report = labelling.label_pilot(views, output, PROMPTS)
    assert report["status"] == "complete"
    assert report["scored"] == 3
    assert report["known_cost_usd"] == "0.003"
    assert len(calls) == 3
    assert labelling.label_pilot(views, output, PROMPTS) == report
    assert len(calls) == 3
    labels = read_jsonl(output / "observations.jsonl")
    assert all(x["technical_quality"]["score"] == 3.7 for x in labels)
    assert all(x["review_status"] == "pending" and not any(x["task_masks"].values()) for x in labels)
    assert all(x["photo_domain_eligible"] is None for x in labels)
    assert labelling.verify_labels(views, output)["observations"] == 3
    assert "test-secret" not in "".join(p.read_text() for p in output.rglob("*.json"))


def test_dry_run_has_no_network_or_credentials(pilot, monkeypatch):
    views, output = pilot
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(labelling.openrouter, "model_info", lambda _: pytest.fail("Network call"))
    assert labelling.label_pilot(views, output, PROMPTS, dry_run=True)["api_calls"] == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "result,reason",
    [
        (
            {
                "http_status": 429,
                "body": '{"error":{"code":429}}',
                "retry_after": "600",
                "transport_error": None,
            },
            "http_or_provider_error",
        ),
        (
            {"http_status": None, "body": None, "retry_after": None, "transport_error": "ReadTimeout"},
            "transport_outcome_unknown",
        ),
        (response(cost=None), "cost_unavailable"),
        (response(finish="length"), "invalid_rating"),
        (response(model="wrong/model"), "invalid_rating"),
        (
            response(
                rating={
                    "task": "technical_quality",
                    "status": "ok",
                    "score": 0,
                    "evidence": "Poor quality.",
                    "limitations": [],
                }
            ),
            "invalid_rating",
        ),
    ],
)
def test_failure_stops_and_is_not_retried_on_resume(pilot, monkeypatch, result, reason):
    views, output = pilot
    calls = []
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *args: calls.append(args) or result)
    report = labelling.label_pilot(views, output, PROMPTS)
    assert report["status"] == "blocked"
    assert report["attempted"] == 1
    assert report["attempts"][0]["reason"].startswith(reason)
    assert labelling.label_pilot(views, output, PROMPTS) == report
    assert len(calls) == 1
    if result["http_status"] == 429:
        assert report["attempts"][0]["retry_after"] == "600"


def test_interrupted_request_blocks_replay(pilot, monkeypatch):
    views, output = pilot

    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(labelling.openrouter, "complete", interrupted)
    with pytest.raises(KeyboardInterrupt):
        labelling.label_pilot(views, output, PROMPTS)
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: pytest.fail("Duplicate attempt"))
    report = labelling.label_pilot(views, output, PROMPTS)
    assert report["status"] == "blocked"
    assert report["attempts"][0]["reason"] == "interrupted_request"


def test_saved_response_is_recovered_without_resending(pilot, monkeypatch):
    views, output = pilot
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: response())
    original_report = labelling._report

    def interrupted_report(*args):
        if list(output.rglob("result.json")):
            raise KeyboardInterrupt
        return original_report(*args)

    monkeypatch.setattr(labelling, "_report", interrupted_report)
    with pytest.raises(KeyboardInterrupt):
        labelling.label_pilot(views, output, PROMPTS)
    monkeypatch.setattr(labelling, "_report", original_report)
    calls = []
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *args: calls.append(args) or response())
    assert labelling.label_pilot(views, output, PROMPTS)["scored"] == 3
    assert len(calls) == 2


def test_budget_reserves_before_a_request(pilot, monkeypatch):
    views, output = pilot
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: pytest.fail("Budget bypass"))
    report = labelling.label_pilot(views, output, PROMPTS, max_cost_usd="0.01")
    assert report["status"] == "budget_stopped"
    assert report["attempted"] == 0


def test_settings_and_artifact_changes_are_rejected(pilot, monkeypatch):
    views, output = pilot
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: response())
    labelling.label_pilot(views, output, PROMPTS)
    with pytest.raises(ValueError, match="settings"):
        labelling.label_pilot(views, output, PROMPTS, max_tokens=2048)
    observations = read_jsonl(output / "observations.jsonl")
    observations[0]["technical_quality"]["score"] = 5
    write_jsonl(output / "observations.jsonl", observations)
    with pytest.raises(ValueError, match="preserved responses"):
        labelling.verify_labels(views, output)
    run = json.loads((output / "run.json").read_text())
    run["settings"]["model"] = "other"
    atomic_json(output / "run.json", run)
    with pytest.raises(ValueError, match="integrity"):
        labelling.label_pilot(views, output, PROMPTS)


def test_price_bound_includes_overrides_and_rejects_unknown_fees():
    info = {
        "context_length": 1000,
        "pricing": {
            "prompt": "0.000001",
            "completion": "0.000002",
            "overrides": [{"utc_days": ["monday"], "prompt": "0.000003", "completion": "0.000004"}],
        },
    }
    assert labelling._price_bound(info, 100)["reserve_usd"] == "0.003400"
    info["pricing"]["image"] = "1"
    with pytest.raises(ValueError, match="fee type"):
        labelling._price_bound(info, 100)


def test_orphan_attempt_directory_is_not_silently_skipped(pilot, monkeypatch):
    views, output = pilot
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: pytest.fail("Network call"))
    labelling.label_pilot(views, output, PROMPTS, max_cost_usd="0.01")
    row = read_jsonl(views / "views.jsonl")[0]
    (output / "attempts" / labelling._digest(row["view_id"])).mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete attempt"):
        labelling.label_pilot(views, output, PROMPTS, max_cost_usd="0.01")


@pytest.mark.parametrize("choices", [None, [None], [3], [{"finish_reason": "stop", "message": None}]])
def test_malformed_envelopes_are_failed_attempts(pilot, monkeypatch, choices):
    views, output = pilot
    result = response()
    body = json.loads(result["body"])
    body["choices"] = choices
    result["body"] = json.dumps(body)
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: result)
    report = labelling.label_pilot(views, output, PROMPTS)
    assert report["status"] == "blocked"
    assert report["valid_observations"] == 0
    assert report["attempts"][0]["reason"].startswith("invalid_rating:")
    assert labelling.label_pilot(views, output, PROMPTS) == report
