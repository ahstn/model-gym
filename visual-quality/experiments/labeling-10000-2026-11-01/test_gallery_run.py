"""Guard reference-only selection and separation of gallery review metadata."""

import copy
import json

import gallery_run as run
import pytest


def example_inputs():
    cases = [
        {"case_id": f"X{1000 - i:04}", "publication_order": i + 1, "benchmark_order": i + 1}
        for i in range(1000)
    ]
    references = {
        case["case_id"]: {"technical_quality": {"score": 1 + i / 250}} for i, case in enumerate(cases)
    }
    return cases, references


def test_selection_uses_explicit_publication_order_and_is_frozen(tmp_path):
    cases, references = example_inputs()
    cases.append({"case_id": "PRIVATE", "publication_order": 1001, "benchmark_order": 1001})
    references["PRIVATE"] = {"technical_quality": {"score": 5}}
    path = tmp_path / "selection.json"
    selection = run.freeze_selection(cases, references, "reference-hash", "manifest-hash", path)
    assert len(selection["case_ids"]) == len(set(selection["case_ids"])) == 72
    assert "PRIVATE" not in selection["case_ids"]
    assert selection["case_ids"][0] == "X1000"
    assert selection["case_ids"][-1] == "X0001"
    assert (
        run.freeze_selection(list(reversed(cases)), references, "reference-hash", "manifest-hash", path)
        == selection
    )
    with pytest.raises(ValueError, match="Frozen collage"):
        run.freeze_selection(cases, references, "different-reference-hash", "manifest-hash", path)


def test_selection_does_not_read_model_scores(tmp_path):
    cases, references = example_inputs()
    original = copy.deepcopy(references)
    a = run.freeze_selection(cases, references, "refs", "manifest", tmp_path / "a.json")
    for case in cases:
        case["technical_score"] = 5
        case["assessments"] = {"luna": {"technical_score": 1}, "photo_ensemble": {"technical_score": 5}}
    b = run.freeze_selection(cases, references, "refs", "manifest", tmp_path / "b.json")
    assert a["case_ids"] == b["case_ids"]
    assert references == original


def test_metadata_remains_outside_assessments_and_rejects_changed_views(tmp_path):
    path = tmp_path / "images.jsonl"
    row = {
        "case_id": "X",
        "view_sha256": "view",
        "assessments": {"luna": {"score": 5}},
        "image_stats": {"focus_area": 1, "used_in_inference": False, "training_features_enabled": False},
    }
    path.write_text(json.dumps(row) + "\n")
    metadata = run.load_metadata(path, {"X": {"view_sha256": "view"}}, {})
    assert set(metadata["X"]) == {"image_stats"}
    with pytest.raises(ValueError, match="identity/view mismatch"):
        run.load_metadata(path, {"X": {"view_sha256": "other-view"}}, {})


def test_unknown_price_and_reference_latency_remain_missing(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "case_id,model,arm,status,technical_score,aesthetic_score,latency_seconds,price_usd\nX,photo_ensemble,baseline,valid,2.3,3.4,,\nX,reference,reference,valid,2.0,3.0,,0.00\n"
    )
    rows = run.gallery.read_rows(path, recompute_ensemble=False)
    assert rows["X"]["photo_ensemble:baseline"]["price_usd"] is None
    assert rows["X"]["reference:reference"]["latency_seconds"] is None
    assert run.gallery.money(None) == "—"
    assert run.gallery.money(0) == "$0.00"


def test_exported_score_display_preserves_shared_rounding():
    row = {"technical_score": 3.125, "technical_score_display": "3.13"}
    assert run.gallery.score_text(row, "technical_score") == "3.13"
    assert row["technical_score"] == 3.125
