"""Label contract checks independent of network access and model behaviour."""

import json
from copy import deepcopy
from importlib.resources import files

import pytest

from visual_quality.labels import observation, parse_rating, validate_observation


def _rating(**changes) -> dict:
    return {
        "task": "technical_quality",
        "status": "ok",
        "score": 3.7,
        "evidence": "Fine detail is soft, but the main subject remains clear.",
        "limitations": [],
        **changes,
    }


def _view(**changes) -> dict:
    return {
        "image_id": "source/image-01",
        "view_id": "full-view-01",
        "source_sha256": "a" * 64,
        "view_sha256": "b" * 64,
        "split": "train",
        "split_group": "scene-01",
        "photo_domain_eligible": None,
        **changes,
    }


def _provenance(**changes) -> dict:
    return {
        "run_id": "pilot-01",
        "attempt_id": "attempt-01",
        "provider": "openrouter",
        "model_requested": "vendor/vision-model",
        "model_returned": "vendor/vision-model-20261018",
        "provider_returned": None,
        "prompt_version": "visual-rating-v1",
        "prompt_sha256": "c" * 64,
        "request_sha256": "d" * 64,
        "created_at": "2026-10-18T10:30:01.123456+00:00",
        "usage": {"prompt_tokens": 600, "completion_tokens": 100, "cost": 0.001},
        **changes,
    }


@pytest.mark.parametrize("score", [1, 1.1, 2.3, 3.7, 4.9, 5])
def test_valid_scores_preserve_tenths(score: float) -> None:
    parsed = parse_rating(json.dumps(_rating(score=score)))
    assert parsed["score"] == score
    json.dumps(parsed, allow_nan=False)


def test_unscorable_has_null_score_and_reason() -> None:
    rating = _rating(status="unscorable", score=None, evidence="The input cannot be interpreted.")
    parsed = parse_rating(json.dumps(rating))
    assert parsed == rating
    record = observation(_view(), parsed, _provenance())
    assert record["technical_quality"]["score"] is None
    assert record["task_masks"]["technical_quality"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"score": 0},
        {"score": 5.1},
        {"score": 2.35},
        {"score": True},
        {"score": "3.7"},
        {"score": None},
        {"score": float("inf")},
        {"score": float("nan")},
        {"task": "aesthetics"},
        {"status": "not_requested"},
        {"status": "unscorable", "score": 1},
        {"evidence": " "},
        {"evidence": None},
        {"evidence": " ".join(["word"] * 31)},
        {"limitations": ["limited_detail", "limited_detail"]},
        {"limitations": ["unsure"]},
        {"limitations": {}},
        {"limitations": [[]]},
        {"confidence": 0.8},
    ],
)
def test_invalid_ratings_fail_without_imputation(changes: dict) -> None:
    with pytest.raises(ValueError):
        parse_rating(json.dumps(_rating(**changes)))


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        "true",
        "3",
        "{}",
        "not JSON",
        "```json\n{}\n```",
        "{} {}",
        '{"task":"technical_quality","task":"technical_quality"}',
        '{"score":1e99999999999999999999999999}',
    ],
)
def test_bare_single_object_and_unique_keys_required(text: str) -> None:
    with pytest.raises(ValueError):
        parse_rating(text)


def test_precision_is_checked_before_float_conversion() -> None:
    text = json.dumps(_rating()).replace("3.7", "3.7000000000000000000000000000001")
    with pytest.raises(ValueError):
        parse_rating(text)
    for written in ("3.70", "37e-1"):
        assert parse_rating(json.dumps(_rating()).replace("3.7", written))["score"] == 3.7


@pytest.mark.parametrize("domain", [True, False, None])
def test_observation_keeps_domain_unknown_and_masks_disabled(domain: bool | None) -> None:
    record = observation(_view(photo_domain_eligible=domain), _rating(), _provenance())
    validate_observation(record)
    assert record["schema_version"] == "label-observation-v1"
    assert record["label_kind"] == "teacher_scalar"
    assert record["photo_domain_eligible"] is domain
    assert record["aesthetics"] == {"status": "not_requested", "score": None}
    assert record["review_status"] == "pending"
    assert record["task_masks"] == {"technical_quality": False, "aesthetics": False, "photo_domain": False}


def test_observation_is_independent_of_source_objects() -> None:
    view, rating, provenance = _view(), _rating(), _provenance()
    before = deepcopy((view, rating, provenance))
    record = observation(view, rating, provenance)
    assert (view, rating, provenance) == before
    rating["limitations"].append("limited_detail")
    provenance["usage"]["cost"] = 999
    assert record["technical_quality"]["limitations"] == []
    assert record["provenance"]["usage"]["cost"] == 0.001


@pytest.mark.parametrize(
    "changes",
    [
        {"image_id": ""},
        {"view_id": " "},
        {"source_sha256": "a" * 63},
        {"view_sha256": "z" * 64},
        {"split": None},
        {"split_group": []},
        {"photo_domain_eligible": 0},
        {"photo_domain_eligible": "unknown"},
    ],
)
def test_invalid_view_identity_is_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        observation(_view(**changes), _rating(), _provenance())


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": ""},
        {"model_returned": None},
        {"provider_returned": ""},
        {"provider": "other"},
        {"request_sha256": "no hash"},
        {"created_at": "2026-10-18"},
        {"created_at": "2026-10-18T10:30:01"},
        {"created_at": "2026-02-31T10:30:01Z"},
        {"created_at": "2026-10-18T10:30:01+00:60"},
        {"usage": None},
        {"usage": {"cost": float("nan")}},
        {"usage": {"nested": [float("inf")]}},
        {"usage": {1: "wrong key"}},
        {"secret": "extra fields are prohibited"},
    ],
)
def test_invalid_provenance_is_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        observation(_view(), _rating(), _provenance(**changes))


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), "label-observation-v2"),
        (("label_kind",), "human_mos"),
        (("review_status",), "accepted"),
        (("task_masks", "technical_quality"), True),
        (("task_masks", "aesthetics"), 0),
        (("task_masks", "photo_domain"), None),
        (("aesthetics", "score"), 3),
        (("technical_quality", "score"), 0),
    ],
)
def test_training_promotion_or_contract_changes_require_new_validation(path: tuple, value) -> None:
    record = observation(_view(), _rating(), _provenance())
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_observation(record)


def test_missing_fields_fail_before_writing() -> None:
    view = _view()
    del view["photo_domain_eligible"]
    with pytest.raises(ValueError, match="missing required field"):
        observation(view, _rating(), _provenance())
    record = observation(_view(), _rating(), _provenance())
    del record["provenance"]["usage"]
    with pytest.raises(ValueError, match="Provenance"):
        validate_observation(record)


def test_packaged_schema_declares_pending_only_contract() -> None:
    path = files("visual_quality").joinpath("schemas/label-observation-v1.schema.json")
    schema = json.loads(path.read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["review_status"] == {"const": "pending"}
    assert schema["$defs"]["technical_quality"]["properties"]["score"]["multipleOf"] == 0.1
