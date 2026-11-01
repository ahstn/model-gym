"""Checks preventing reference leakage, wrong-image reviews and fabricated timing."""

import json
from pathlib import Path

import pytest
import reference_review as review


def assessment():
    return {
        "eligibility": "eligible",
        "eligibility_evidence": "A physical photographed scene is visible.",
        "technical_quality": {
            "task": "technical_quality",
            "status": "ok",
            "score": 3.2,
            "evidence": "Important details remain visible despite minor softness.",
            "limitations": [],
        },
        "aesthetics": {
            "task": "aesthetics",
            "status": "ok",
            "score": 2.8,
            "evidence": "Cluttered framing weakens the central subject's visual emphasis.",
            "limitations": [],
        },
    }


def packet(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image bytes")
    path = tmp_path / "packet.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "R0001",
                        "image_path": str(image),
                        "view_sha256": review.digest(image),
                    }
                ]
            }
        )
    )
    return path


def test_review_duration_is_measured_and_distinct_from_api_latency(tmp_path, monkeypatch):
    p = packet(tmp_path)
    ticks = iter([100.0, 112.5])
    monkeypatch.setattr(review.time, "monotonic", lambda: next(ticks))
    review.begin(p, "R0001", tmp_path / "timing.json")
    row = review.complete(p, "R0001", tmp_path / "timing.json", tmp_path / "review.jsonl", assessment())
    assert row["review_elapsed_seconds"] == 12.5
    assert row["latency_seconds"] is None
    assert row["price_usd"] == 0


def test_changed_pixels_cannot_receive_existing_timing(tmp_path):
    p = packet(tmp_path)
    review.begin(p, "R0001", tmp_path / "timing.json")
    (tmp_path / "image.png").write_bytes(b"different image")
    with pytest.raises(ValueError, match="image changed"):
        review.complete(p, "R0001", tmp_path / "timing.json", tmp_path / "review.jsonl", assessment())
    assert not (tmp_path / "review.jsonl").exists()


def test_existing_reference_is_not_replaced(tmp_path):
    p = packet(tmp_path)
    review.begin(p, "R0001", tmp_path / "timing.json")
    out = tmp_path / "review.jsonl"
    review.complete(p, "R0001", tmp_path / "timing.json", out, assessment())
    before = out.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        review.complete(p, "R0001", tmp_path / "timing.json", out, assessment())
    assert out.read_bytes() == before


@pytest.mark.parametrize("score", [True, 0.9, 5.1, 3.25, float("nan"), None])
def test_invalid_reference_scores_rejected(score):
    row = assessment()
    row["technical_quality"]["score"] = score
    with pytest.raises(ValueError):
        review.validate_assessment(row)


def test_policy_floor_requires_both_dimensions():
    row = assessment()
    row["eligibility"] = "excluded"
    with pytest.raises(ValueError, match="both policy floors"):
        review.validate_assessment(row)
    for task in review.TASKS:
        row[task]["score"] = 1.0
        row[task]["limitations"] = ["unsupported_domain"]
    review.validate_assessment(row)


def test_nonexcluded_does_not_accept_scope_floor_flag():
    row = assessment()
    row["technical_quality"]["limitations"] = ["unsupported_domain"]
    with pytest.raises(ValueError, match="Nonexcluded"):
        review.validate_assessment(row)


def completed_row(tmp_path, monkeypatch):
    p = packet(tmp_path)
    with monkeypatch.context() as patch:
        ticks = iter([100.0, 112.5])
        timestamps = iter(["2026-01-01T12:00:00+00:00", "2026-01-01T12:00:12.500000+00:00"])
        patch.setattr(review.time, "monotonic", lambda: next(ticks))
        patch.setattr(review, "utc", lambda: next(timestamps))
        timing_path = tmp_path / "timing.json"
        review.begin(p, "R0001", timing_path)
        row = review.complete(p, "R0001", timing_path, tmp_path / "review.jsonl", assessment())
    return row


def test_completed_timing_validates_against_its_saved_record(tmp_path, monkeypatch):
    review.validate_timing(completed_row(tmp_path, monkeypatch))


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), True, None, "12.5"])
def test_invalid_elapsed_review_time_rejected(tmp_path, monkeypatch, elapsed):
    row = completed_row(tmp_path, monkeypatch)
    row["review_elapsed_seconds"] = elapsed
    with pytest.raises(ValueError, match="measured and nonnegative"):
        review.validate_timing(row)


def test_fabricated_positive_elapsed_time_rejected(tmp_path, monkeypatch):
    row = completed_row(tmp_path, monkeypatch)
    row["review_elapsed_seconds"] = 8.0
    with pytest.raises(ValueError, match="disagrees with monotonic"):
        review.validate_timing(row)


def test_replaced_start_timestamp_rejected(tmp_path, monkeypatch):
    row = completed_row(tmp_path, monkeypatch)
    row["review_started_at"] = "2026-01-01T11:59:00+00:00"
    with pytest.raises(ValueError, match="different image"):
        review.validate_timing(row)


@pytest.mark.parametrize("timestamp", ["2026-01-01T11:59:00+00:00", "2026-01-01T12:00:12", "invalid"])
def test_invalid_completion_timestamp_rejected(tmp_path, monkeypatch, timestamp):
    row = completed_row(tmp_path, monkeypatch)
    row["review_completed_at"] = timestamp
    with pytest.raises(ValueError):
        review.validate_timing(row)


@pytest.mark.parametrize("field,value", [("latency_kind", "api_request"), ("latency_seconds", 12.5)])
def test_review_duration_cannot_be_presented_as_api_latency(tmp_path, monkeypatch, field, value):
    row = completed_row(tmp_path, monkeypatch)
    row[field] = value
    with pytest.raises(ValueError, match="no HTTP latency"):
        review.validate_timing(row)


def test_changed_timing_file_rejected(tmp_path, monkeypatch):
    row = completed_row(tmp_path, monkeypatch)
    timing_path = Path(row["review_timing_path"])
    timing = json.loads(timing_path.read_text())
    timing["monotonic_started"] = 101.0
    timing_path.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match="timing record changed"):
        review.validate_timing(row)


def test_changed_packet_cannot_complete_a_started_review(tmp_path):
    p = packet(tmp_path)
    review.begin(p, "R0001", tmp_path / "timing.json")
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="different input"):
        review.complete(p, "R0001", tmp_path / "timing.json", tmp_path / "review.jsonl", assessment())


def full_review_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "HERE", tmp_path)
    directory = tmp_path / "references"
    directory.mkdir()
    for name in (
        "references/rubric.txt",
        "references/review-instructions.md",
        "protocol.md",
        "api-teacher-prompt.json",
    ):
        (tmp_path / name).write_text("fixed content")
    cases = []
    for i in range(500):
        image = tmp_path / f"B{i:04}.png"
        image.write_bytes(f"fixture-image-{i}".encode())
        cases.append(
            {
                "case_id": f"B{i:04}",
                "family_id": f"family-{i}",
                "image_path": str(image),
                "view_sha256": review.digest(image),
                "benchmark_partition": "development"
                if i < 100
                else "representative"
                if i < 400
                else "challenge",
            }
        )
    early = tmp_path / "early-benchmark.json"
    early.write_text(json.dumps({"cases": cases}))
    index = review.make_packets(early, 25)
    for entry in index["packets"]:
        p = Path(entry["path"])
        for case in json.loads(p.read_text())["cases"]:
            timing = directory / (case["case_id"] + ".timing.json")
            review.begin(p, case["case_id"], timing)
            review.complete(p, case["case_id"], timing, directory / "review-fixture.jsonl", assessment())
    full = tmp_path / "manifest.json"
    full.write_text(
        json.dumps(
            {
                "cases": cases
                + [
                    {"case_id": f"R{i:04}", "family_id": f"rehearsal-{i}", "benchmark_partition": "rehearsal"}
                    for i in range(3500)
                ]
            }
        )
    )
    return early, full


def test_freeze_binds_early_500_reviews_to_full_4000_manifest(tmp_path, monkeypatch):
    early, full = full_review_fixture(tmp_path, monkeypatch)
    frozen = review.freeze(full)
    assert len(frozen["case_ids"]) == 500
    assert frozen["manifest_sha256"] == review.digest(full)
    assert frozen["review_input_manifest_sha256"] == review.digest(early)
    assert frozen["manifest_sha256"] != frozen["review_input_manifest_sha256"]
    assert len((tmp_path / "references.jsonl").read_text().splitlines()) == 500


def test_freeze_rejects_reference_duration_tampering(tmp_path, monkeypatch):
    _, full = full_review_fixture(tmp_path, monkeypatch)
    path = tmp_path / "references/review-fixture.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["review_elapsed_seconds"] = -1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="measured and nonnegative"):
        review.freeze(full)
    assert not (tmp_path / "references.jsonl").exists()


def test_freeze_rejects_changed_benchmark_family_in_full_manifest(tmp_path, monkeypatch):
    _, full = full_review_fixture(tmp_path, monkeypatch)
    manifest = json.loads(full.read_text())
    manifest["cases"][0]["family_id"] = "another-family"
    full.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed a frozen benchmark case"):
        review.freeze(full)
    assert not (tmp_path / "references.jsonl").exists()
