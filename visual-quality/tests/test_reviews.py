"""Assistant corrections preserve source evidence and remain unaccepted labels."""

import copy
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from visual_quality import labelling, reviews
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl
from visual_quality.views import prepare_views

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "visual-rating-v1"


@pytest.fixture
def review_case(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for index in range(4):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (16, 12), (index * 40, 20, 30)).save(image)
        rows.append(
            {
                "id": str(index),
                "sha256": file_sha256(image),
                "image_path": image.name,
                "split": "train",
                "split_group": str(index),
                "component": "megalith",
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
    views, source = tmp_path / "views", tmp_path / "run"
    prepare_views(dataset, views, limit=4)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr(
        labelling.openrouter,
        "model_info",
        lambda _: {
            "id": labelling.DEFAULT_MODEL,
            "context_length": 1024,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
    )
    rating = {
        "task": "technical_quality",
        "status": "ok",
        "score": 3.7,
        "evidence": "Some image detail is blurred.",
        "limitations": [],
    }
    body = {
        "model": labelling.DEFAULT_MODEL,
        "usage": {"cost": 0.001},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(rating)}}],
    }
    monkeypatch.setattr(
        labelling.openrouter,
        "complete",
        lambda *_: {
            "http_status": 200,
            "body": json.dumps(body),
            "retry_after": None,
            "transport_error": None,
        },
    )
    labelling.label_pilot(views, source, PROMPTS)
    monkeypatch.setattr(labelling.openrouter, "complete", lambda *_: pytest.fail("Unexpected provider call"))
    originals = read_jsonl(source / "observations.jsonl")
    plan = {
        "schema_version": "label-review-plan-v1",
        "review_id": "assistant-review-1",
        "reviewed_at": "2026-10-18T12:00:00Z",
        "reviewer": {
            "kind": "assistant",
            "id": "codex",
            "method": "Direct fixed-view inspection",
            "model_id": None,
        },
        "source_observations_sha256": file_sha256(source / "observations.jsonl"),
        "source_views_sha256": file_sha256(views / "views.jsonl"),
        "rubric_sha256": json.loads((source / "run.json").read_text())["prompt"]["sha256"],
        "reviews": [
            {
                "image_id": row["image_id"],
                "view_id": row["view_id"],
                "original_score": 3.7,
                "technical_quality": copy.deepcopy(row["technical_quality"]),
                "reason": "Reviewed the complete frozen view.",
                "uncertainty_interval": [3.0, 4.5],
                "independent_review": {
                    "reviewer": "independent-agent",
                    "score": 3.7,
                    "interval": [3.2, 4.2],
                    "evidence": "Some image detail is blurred.",
                    "limitations": [],
                },
            }
            for row in originals
        ],
    }
    plan["reviews"][0]["technical_quality"]["score"] = 4.3
    plan["reviews"][1]["technical_quality"]["evidence"] = "Fine detail remains clear."
    plan["reviews"][2]["technical_quality"].update(score=4.0, evidence="Subject detail is clear.")
    path = tmp_path / "review-plan.json"
    atomic_json(path, plan)
    return views, source, path, tmp_path / "reviewed"


def _edit_plan(case, edit):
    plan = json.loads(case[2].read_text())
    edit(plan)
    atomic_json(case[2], plan)


def test_export_corrections_preserves_sources_and_refuses_training_masks(review_case):
    views, source, plan, output = review_case
    inputs = {p: file_sha256(p) for root in (views, source) for p in root.rglob("*") if p.is_file()}
    _edit_plan(review_case, lambda p: p["reviews"].reverse())
    report = reviews.build_reviewed_labels(*review_case)
    records = read_jsonl(output / "reviewed-observations.jsonl")
    originals = read_jsonl(source / "observations.jsonl")
    assert [r["source_observation"] for r in records] == originals
    assert [r["technical_quality"]["score"] for r in records] == [4.3, 3.7, 4.0, 3.7]
    assert report["decisions"] == {
        "retained": 1,
        "score_corrected": 1,
        "evidence_corrected": 1,
        "score_and_evidence_corrected": 1,
    }
    assert report["review_plan_sha256"] == file_sha256(plan)
    assert report["reviewed_observations_sha256"] == file_sha256(output / "reviewed-observations.jsonl")
    assert report["training_ready"] is False
    for record in records:
        assert record["label_kind"] == "assistant_review"
        assert record["review_status"] == "pending_human_review"
        assert record["task_masks"] == {
            "technical_quality": False,
            "aesthetics": False,
            "photo_domain": False,
        }
        assert record["aesthetics"] == {"status": "not_requested", "score": None}
        assert record["photo_domain_eligible"] is None
        assert "provenance" not in record
    assert {p: file_sha256(p) for p in inputs} == inputs
    mtimes = {p: p.stat().st_mtime_ns for p in output.iterdir()}
    assert reviews.verify_reviewed_labels(*review_case) == report
    assert reviews.build_reviewed_labels(*review_case) == report
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes


@pytest.mark.parametrize("score", [True, 0, 6, 3.71, None, float("nan"), float("inf")])
def test_invalid_review_scores_never_publish(review_case, score):
    _edit_plan(review_case, lambda p: p["reviews"][0]["technical_quality"].update(score=score))
    with pytest.raises(ValueError):
        reviews.build_reviewed_labels(*review_case)
    assert not review_case[3].exists()


@pytest.mark.parametrize(
    "field,literal",
    [
        (("technical_quality", "score"), "4.3000000000000001"),
        (("independent_review", "score"), "3.7000000000000001"),
        (("uncertainty_interval", 0), "3.0000000000000001"),
        (("independent_review", "interval", 0), "3.2000000000000001"),
        (("original_score",), "3.7000000000000001"),
    ],
)
def test_lossy_numeric_text_is_rejected_before_rating_validation(review_case, field, literal):
    plan = json.loads(review_case[2].read_text())
    target = plan["reviews"][0]
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = "__LOSSY_NUMBER__"
    review_case[2].write_text(json.dumps(plan).replace('"__LOSSY_NUMBER__"', literal))
    with pytest.raises(ValueError, match="loses decimal precision"):
        reviews.build_reviewed_labels(*review_case)
    assert not review_case[3].exists()


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p["reviews"].pop(),
        lambda p: p["reviews"].append(p["reviews"][0]),
        lambda p: p["reviews"][0].update(image_id="wrong-image"),
        lambda p: p["reviews"][0].update(view_id="wrong-view"),
        lambda p: p["reviews"][0].update(original_score=4.0),
        lambda p: p["reviews"][0].update(uncertainty_interval=[3.0, 4.0]),
        lambda p: p["reviews"][0].update(uncertainty_interval=[4.5, 3.0]),
        lambda p: p["reviews"][0].update(uncertainty_interval=[1.01, 5.0]),
        lambda p: p["reviews"][0]["independent_review"].update(interval=[4.0, 4.5]),
        lambda p: p["reviews"][0]["independent_review"].update(score=False),
        lambda p: p["reviews"][0]["technical_quality"].update(evidence="word " * 31),
        lambda p: p["reviews"][0]["technical_quality"].update(limitations=["invented"]),
        lambda p: p["reviews"][0].update(task_masks={"technical_quality": True}),
        lambda p: p["reviewer"].update(kind="human"),
        lambda p: p["reviewer"].update(model_id="invented/model"),
        lambda p: p.update(reviewed_at="2026-10-18T12:00:00"),
        lambda p: p.update(reviewed_at="2026-13-18T12:00:00Z"),
        lambda p: p.update(reviewed_at="2026-10-18T12:00:00+01:99"),
        lambda p: p.update(reviewed_at="2026-10-18T12:00:00+24:00"),
        lambda p: p.update(source_observations_sha256="0" * 64),
        lambda p: p.update(source_views_sha256="0" * 64),
        lambda p: p.update(rubric_sha256="0" * 64),
    ],
)
def test_invalid_join_or_contract_never_publishes(review_case, edit):
    _edit_plan(review_case, edit)
    with pytest.raises(ValueError):
        reviews.build_reviewed_labels(*review_case)
    assert not review_case[3].exists()


def test_unscorable_review_keeps_null_score(review_case):
    _edit_plan(
        review_case, lambda p: p["reviews"][0]["technical_quality"].update(status="unscorable", score=None)
    )
    report = reviews.build_reviewed_labels(*review_case)
    assert report["scored"] == 3
    assert read_jsonl(review_case[3] / "reviewed-observations.jsonl")[0]["technical_quality"]["score"] is None


@pytest.mark.parametrize("change", ["score", "mask", "source", "report", "extra"])
def test_review_tamper_is_rejected_on_verify_and_replay(review_case, change):
    reviews.build_reviewed_labels(*review_case)
    output = review_case[3]
    path = output / "reviewed-observations.jsonl"
    records = read_jsonl(path)
    if change == "report":
        report = json.loads((output / "report.json").read_text())
        report["training_ready"] = True
        atomic_json(output / "report.json", report)
    elif change == "extra":
        (output / "unrelated.txt").write_text("Do not overwrite me.")
    else:
        if change == "score":
            records[0]["technical_quality"]["score"] = 5.0
        elif change == "mask":
            records[0]["task_masks"]["technical_quality"] = True
        else:
            records[0]["source_observation"]["provenance"]["model_requested"] = "other-model"
        write_jsonl(path, records)
    before = {p: file_sha256(p) for p in output.iterdir()}
    for operation in (reviews.verify_reviewed_labels, reviews.build_reviewed_labels):
        with pytest.raises(ValueError):
            operation(*review_case)
    assert {p: file_sha256(p) for p in before} == before


@pytest.mark.parametrize("field,replacement", [("training_ready", 0), ("observations", 4.0)])
def test_report_comparison_preserves_json_types(review_case, field, replacement):
    reviews.build_reviewed_labels(*review_case)
    path = review_case[3] / "report.json"
    report = json.loads(path.read_text())
    report[field] = replacement
    atomic_json(path, report)
    before = file_sha256(path)
    for operation in (reviews.verify_reviewed_labels, reviews.build_reviewed_labels):
        with pytest.raises(ValueError, match="Reviewed report differs"):
            operation(*review_case)
    assert file_sha256(path) == before


def test_changed_plan_requires_new_output_directory(review_case):
    reviews.build_reviewed_labels(*review_case)
    _edit_plan(review_case, lambda p: p["reviews"][0].update(reason="Revised review explanation."))
    with pytest.raises(ValueError, match="pinned review plan"):
        reviews.build_reviewed_labels(*review_case)
    assert (
        reviews.build_reviewed_labels(*review_case[:3], review_case[3].with_name("reviewed-v2"))[
            "observations"
        ]
        == 4
    )


def test_modified_original_cannot_be_laundered_through_review(review_case):
    path = review_case[1] / "observations.jsonl"
    records = read_jsonl(path)
    records[0]["technical_quality"]["score"] = 5.0
    write_jsonl(path, records)
    _edit_plan(review_case, lambda p: p.update(source_observations_sha256=file_sha256(path)))
    with pytest.raises(ValueError, match="preserved responses"):
        reviews.build_reviewed_labels(*review_case)


def test_partial_run_requires_only_its_existing_observations(review_case):
    source = review_case[1]
    path = source / "observations.jsonl"
    records = read_jsonl(path)
    for row in records[2:]:
        shutil.rmtree(source / "attempts" / row["provenance"]["attempt_id"])
    write_jsonl(path, records[:2])
    _edit_plan(
        review_case,
        lambda p: p.update(
            source_observations_sha256=file_sha256(path),
            reviews=p["reviews"][:2],
        ),
    )
    assert reviews.build_reviewed_labels(*review_case)["observations"] == 2
    assert reviews.verify_reviewed_labels(*review_case)["observations"] == 2


def test_duplicate_source_records_are_rejected(review_case):
    path = review_case[1] / "observations.jsonl"
    records = read_jsonl(path)
    write_jsonl(path, [*records, records[0]])
    _edit_plan(review_case, lambda p: p.update(source_observations_sha256=file_sha256(path)))
    with pytest.raises(ValueError):
        reviews.build_reviewed_labels(*review_case)


@pytest.mark.parametrize(
    "destination", ["views", "views-child", "source", "source-child", "plan", "ancestor"]
)
def test_output_cannot_overlap_inputs(review_case, destination):
    views, source, plan, _ = review_case
    output = {
        "views": views,
        "views-child": views / "output",
        "source": source,
        "source-child": source / "output",
        "plan": plan,
        "ancestor": plan.parent,
    }[destination]
    with pytest.raises(ValueError, match="separate|contain or overwrite"):
        reviews.build_reviewed_labels(views, source, plan, output)


def test_incomplete_existing_directory_is_not_overwritten(review_case):
    review_case[3].mkdir()
    with pytest.raises(ValueError, match="exactly"):
        reviews.build_reviewed_labels(*review_case)
    assert list(review_case[3].iterdir()) == []
