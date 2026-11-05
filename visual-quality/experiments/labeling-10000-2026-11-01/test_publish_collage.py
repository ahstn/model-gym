"""Exercise the final-state, publication-boundary and artifact-copy guards."""

import csv
import json
from copy import deepcopy

import publish_collage as publish
import pytest


def summary():
    return {
        "images": 10000,
        "benchmark_images": 1500,
        "statistics_images": 10000,
        "reference_records": 1500,
        "coverage": {
            model: {
                "expected": 1500 if model == "reference" else 10000,
                "status_counts": {"valid": 1500 if model == "reference" else 10000},
                "numeric_pairs": 1500 if model == "reference" else 10000,
            }
            for model in publish.MODELS
        },
    }


@pytest.mark.parametrize(
    "state", ["missing", "in_flight", "observation_not_finalized", "pending", "interrupted"]
)
def test_unfinished_lite_cannot_publish_even_with_operator_reason(state):
    value = summary()
    value["coverage"]["qrealign_lite"].update(status_counts={"valid": 9999, state: 1}, numeric_pairs=9999)
    with pytest.raises(ValueError, match="Unfinished observation statuses: qrealign_lite"):
        publish.check_final_summary(
            value, {"full_run_summary": value, "incomplete_models_reason": "Reviewed"}
        )


@pytest.mark.parametrize("state", ["failed", "http_error", "transport_or_parse_error", "valid"])
def test_terminal_numeric_gaps_require_documented_decision(state):
    value = summary()
    value["coverage"]["qrealign_lite"].update(
        status_counts={"valid": 10000} if state == "valid" else {"valid": 9999, state: 1}, numeric_pairs=9999
    )
    with pytest.raises(ValueError, match="Numeric gaps require"):
        publish.check_final_summary(value, {"full_run_summary": value})
    publish.check_final_summary(
        value, {"full_run_summary": value, "incomplete_models_reason": "Reviewed failure"}
    )


def test_gallery_and_records_must_describe_same_final_snapshot():
    value, previous = summary(), summary()
    previous["coverage"]["jev"]["numeric_pairs"] -= 1
    with pytest.raises(ValueError, match="different full-run summaries"):
        publish.check_final_summary(value, {"full_run_summary": previous})


def test_ensemble_can_retain_one_valid_outcome_with_documented_gap():
    value = summary()
    value["coverage"]["photo_ensemble"].update(
        status_counts={"valid": 9999, "partial": 1}, numeric_pairs=9999
    )
    publish.check_final_summary(
        value, {"full_run_summary": value, "incomplete_models_reason": "One component outcome unscorable"}
    )


def test_1001_image_publication_is_rejected_before_reading_gallery(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    for name in publish.ALLOWED_FILES - {"provenance.json"}:
        (records / name).write_text("[]")
    (records / "publication-case-ids.json").write_text(json.dumps([f"R{i}" for i in range(1001)]))
    (records / "provenance.json").write_text(
        json.dumps(
            {
                "records": {
                    name: publish.sha(records / name) for name in publish.ALLOWED_FILES - {"provenance.json"}
                }
            }
        )
    )
    with pytest.raises(ValueError, match="1–1,000 distinct image IDs"):
        publish.publish_collage(tmp_path / "absent-gallery", records, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.fixture
def selected_scores(tmp_path):
    rows = [
        {
            "case_id": f"R{i:04}",
            "model": model,
            "arm": "reference" if model == "reference" else "baseline",
            "status": "valid",
            "view_sha256": f"view-{i}",
            "technical_score": 3.0,
            "aesthetic_score": 4.0,
            "latency_seconds": None if model in {"reference", "photo_ensemble"} else 1.25,
            "price_usd": "0.00" if model in {"reference", "qrealign_lite"} else "0.001",
        }
        for i in range(72)
        for model in sorted(publish.MODELS)
    ]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases = {row["case_id"]: {"view_sha256": row["view_sha256"]} for row in rows}
    return rows, cases


@pytest.mark.parametrize(
    "field,value",
    [
        ("technical_score", 3.1),
        ("latency_seconds", 1.5),
        ("price_usd", "0.002"),
        ("view_sha256", "other-view"),
    ],
)
def test_selected_source_values_and_views_must_match_records(tmp_path, selected_scores, field, value):
    rows, cases = selected_scores
    source = tmp_path / "source.csv"

    def write(values):
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(values)

    write(rows)
    publish.check_selected_scores(source, tmp_path, set(cases), cases)
    changed = deepcopy(rows)
    changed[0][field] = value
    write(changed)
    with pytest.raises(ValueError, match="differs|view mismatch"):
        publish.check_selected_scores(source, tmp_path, set(cases), cases)


def stub_validated_gallery(tmp_path, monkeypatch):
    directory, records = tmp_path / "gallery", tmp_path / "records"
    directory.mkdir()
    records.mkdir()
    for name in ("generation-proof.json", "png-validation.json", "data.js", "index.html"):
        (directory / name).write_text("private full-gallery content")
    (records / "provenance.json").write_text("bounded provenance")
    pages = []
    for i in range(1, 5):
        path = directory / f"comparison-page-{i}.png"
        path.write_bytes(b"already validated png fixture" + bytes([i]))
        pages.append({"png_sha256": publish.sha(path)})
    selection = dict.fromkeys(
        (
            "method",
            "references_sha256",
            "manifest_sha256",
            "publication_ids_sha256",
            "frozen_selection_sha256",
        ),
        "pin",
    )
    selection["case_ids"] = [f"R{i:04}" for i in range(72)]
    pins = {
        path: publish.sha(path)
        for path in (
            directory / "generation-proof.json",
            directory / "png-validation.json",
            records / "provenance.json",
        )
    }
    monkeypatch.setattr(publish, "check_gallery", lambda *_: (selection, pages, pins, {"records": {}}))
    return directory, records


def test_publication_copies_only_bounded_artifacts(tmp_path, monkeypatch):
    directory, records = stub_validated_gallery(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    publish.publish_collage(directory, records, destination)
    assert {p.name for p in destination.iterdir()} == {
        *(f"comparison-page-{i}.png" for i in range(1, 5)),
        "selection.json",
        "provenance.json",
        "README.md",
    }
    with pytest.raises(ValueError, match="destination must be empty"):
        publish.publish_collage(directory, records, destination)


def test_png_changed_after_validation_is_not_published(tmp_path, monkeypatch):
    directory, records = stub_validated_gallery(tmp_path, monkeypatch)
    (directory / "comparison-page-2.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Copied PNG hash mismatch"):
        publish.publish_collage(directory, records, tmp_path / "published")
    assert not (tmp_path / "published").exists()
    assert not list(tmp_path.glob(".collage-*"))
