"""Checks preventing reference leakage, wrong-image reviews and fabricated timing."""

import json
from pathlib import Path

import pytest

from visual_quality.scoring import reference_review as review


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
                "candidate_id": f"C{i:04}",
                "image_id": f"image-{i}",
                "family_id": f"family-{i}",
                "benchmark_stratum": "fixture",
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


def revision_fixture(tmp_path, monkeypatch, *, complete=True):
    early, full = full_review_fixture(tmp_path, monkeypatch)
    revised = json.loads(early.read_text())
    previous = {key: revised["cases"][0][key] for key in review.REVISION_FIELDS}
    image = tmp_path / "replacement.png"
    image.write_bytes(b"fresh replacement fixture")
    revised["cases"][0].update(
        candidate_id="C-new",
        image_id="image-new",
        family_id="family-new",
        image_path=str(image),
        view_sha256=review.digest(image),
    )
    replacement = {key: revised["cases"][0][key] for key in review.REVISION_FIELDS}
    active = tmp_path / "benchmark-v2.json"
    active.write_text(json.dumps(revised))
    combined = json.loads(full.read_text())
    combined["cases"][0] = revised["cases"][0]
    full.write_text(json.dumps(combined))
    visual = tmp_path / "historical-review.json"
    visual.write_text(json.dumps({"decision": "confirmed near duplicate"}))
    path = tmp_path / "benchmark-revisions.json"
    declaration = {
        "schema_version": 1,
        "original_manifest": {"path": str(early), "sha256": review.digest(early)},
        "active_manifest": {"path": str(active), "sha256": review.digest(active)},
        "revisions": [
            {
                "case_id": "B0000",
                "previous": previous,
                "replacement": replacement,
                "reason_code": "confirmed_historical_near_duplicate",
                "historical_peer": {
                    "candidate_id": "C-historical",
                    "family_id": "old-family",
                    "view_sha256": "historical-hash",
                },
                "selection": {"seed": 20261101, "tag": "objective-fixture-replacement"},
                "visual_review": {
                    "path": str(visual),
                    "artifact_sha256": review.digest(visual),
                    "decision": "confirmed near duplicate",
                },
                "excluded_from_final_candidates": ["C0000", "C-historical"],
            }
        ],
    }
    path.write_text(json.dumps(declaration))
    packet_index = review.make_revision_packets(path)
    packet = packet_index["packets"][0]
    if complete:
        timing = Path(packet["timing_path"])
        review.begin(Path(packet["path"]), "B0000", timing)
        revised_assessment = assessment()
        revised_assessment["technical_quality"]["score"] = 4.1
        review.complete(
            Path(packet["path"]), "B0000", timing, Path(packet["review_path"]), revised_assessment
        )
    return full, path, packet_index


def test_revision_freeze_replaces_only_declared_case_and_preserves_original_reviews(tmp_path, monkeypatch):
    full, revisions, packet_index = revision_fixture(tmp_path, monkeypatch)
    original = tmp_path / "references/review-fixture.jsonl"
    original_bytes = original.read_bytes()
    original_rows = {row["case_id"]: row for row in map(json.loads, original_bytes.splitlines())}
    frozen = review.freeze(full, revisions=revisions)
    final = {
        row["case_id"]: row
        for row in map(json.loads, (tmp_path / "references.jsonl").read_text().splitlines())
    }
    assert len(final) == 500 and original.read_bytes() == original_bytes
    assert all(final[identity] == row for identity, row in original_rows.items() if identity != "B0000")
    assert final["B0000"]["technical_quality"]["score"] == 4.1
    assert original_rows["B0000"]["technical_quality"]["score"] == 3.2
    assert final["B0000"]["view_sha256"] != original_rows["B0000"]["view_sha256"]
    assert frozen["superseded_reviews"]["B0000"]["review_path"] == str(original)
    assert frozen["review_sources"]["B0000"]["review_path"] == packet_index["packets"][0]["review_path"]
    assert frozen["review_files"][str(original)] == review.digest(original)
    assert frozen["reference_revision"]["declaration_sha256"] == review.digest(revisions)
    assert len(frozen["review_sources"]) == 500 and len(frozen["superseded_reviews"]) == 1
    with pytest.raises(ValueError, match="already frozen"):
        review.freeze(full, revisions=revisions)


def test_revision_requires_explicit_opt_in(tmp_path, monkeypatch):
    full, _, _ = revision_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="changed a frozen benchmark"):
        review.freeze(full)
    assert not (tmp_path / "references.jsonl").exists()


def test_revision_requires_fresh_separate_review(tmp_path, monkeypatch):
    full, revisions, _ = revision_fixture(tmp_path, monkeypatch, complete=False)
    with pytest.raises(ValueError, match="Missing or undeclared replacement review"):
        review.freeze(full, revisions=revisions)
    assert not (tmp_path / "references.jsonl").exists()


@pytest.mark.parametrize(
    "mutation",
    ["undeclared_case", "wrong_previous", "missing_replacement", "nonobjective_reason", "changed_stratum"],
)
def test_revision_rejects_unauthorized_contract_changes(tmp_path, monkeypatch, mutation):
    full, revisions, _ = revision_fixture(tmp_path, monkeypatch)
    declaration = json.loads(revisions.read_text())
    active_path = Path(declaration["active_manifest"]["path"])
    active = json.loads(active_path.read_text())
    if mutation == "undeclared_case":
        active["cases"][1]["image_id"] = "undeclared-image"
    elif mutation == "wrong_previous":
        declaration["revisions"][0]["previous"]["view_sha256"] = "wrong"
    elif mutation == "missing_replacement":
        declaration["revisions"] = []
    elif mutation == "nonobjective_reason":
        declaration["revisions"][0]["reason_code"] = "unfavorable_score"
    else:
        active["cases"][0]["benchmark_stratum"] = "changed-stratum"
        declaration["revisions"][0]["replacement"]["benchmark_stratum"] = "changed-stratum"
    active_path.write_text(json.dumps(active))
    declaration["active_manifest"]["sha256"] = review.digest(active_path)
    revisions.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="declared revisions|replacement contract|unique and preserve"):
        review.freeze(full, revisions=revisions)
    assert not (tmp_path / "references.jsonl").exists()


@pytest.mark.parametrize(
    "artifact",
    ["old_manifest", "old_index", "old_review", "rubric", "revision_packet", "timing", "extra_review"],
)
def test_revision_rejects_changed_or_undeclared_review_provenance(tmp_path, monkeypatch, artifact):
    full, revisions, packet_index = revision_fixture(tmp_path, monkeypatch)
    paths = {
        "old_manifest": tmp_path / "early-benchmark.json",
        "old_index": tmp_path / "references/packet-index.json",
        "old_review": tmp_path / "references/review-fixture.jsonl",
        "rubric": tmp_path / "references/rubric.txt",
        "revision_packet": Path(packet_index["packets"][0]["path"]),
        "timing": Path(packet_index["packets"][0]["timing_path"]),
    }
    if artifact == "extra_review":
        (tmp_path / "references/revisions/review-extra.jsonl").write_text("{}\n")
    else:
        changed = paths[artifact]
        changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError):
        review.freeze(full, revisions=revisions)
    assert not (tmp_path / "references.jsonl").exists()


def test_revision_rejects_timing_from_original_review(tmp_path, monkeypatch):
    full, revisions, packet_index = revision_fixture(tmp_path, monkeypatch)
    original = next(
        row
        for row in map(json.loads, (tmp_path / "references/review-fixture.jsonl").read_text().splitlines())
        if row["case_id"] == "B0000"
    )
    replacement_path = Path(packet_index["packets"][0]["review_path"])
    replacement = json.loads(replacement_path.read_text())
    for key, value in original.items():
        if key.startswith("review_"):
            replacement[key] = value
    replacement_path.write_text(json.dumps(replacement) + "\n")
    with pytest.raises(ValueError, match="different frozen packet"):
        review.freeze(full, revisions=revisions)


def test_revision_packet_contains_only_anonymous_image_contract(tmp_path, monkeypatch):
    _, revisions, packet_index = revision_fixture(tmp_path, monkeypatch, complete=False)
    packet = json.loads(Path(packet_index["packets"][0]["path"]).read_text())
    assert set(packet) == {"cases"}
    assert set(packet["cases"][0]) == {"case_id", "image_path", "view_sha256"}
    assert (
        Path(packet_index["instructions_path"]).read_bytes()
        == (tmp_path / "references/review-instructions.md").read_bytes()
    )
    with pytest.raises(ValueError, match="already exists"):
        review.make_revision_packets(revisions)


def test_revision_rejects_review_started_before_revision_packet_freeze(tmp_path, monkeypatch):
    full, revisions, packet_index = revision_fixture(tmp_path, monkeypatch)
    packet = packet_index["packets"][0]
    row_path, timing_path = Path(packet["review_path"]), Path(packet["timing_path"])
    row, timing = json.loads(row_path.read_text()), json.loads(timing_path.read_text())
    row["review_started_at"] = timing["review_started_at"] = "2026-01-01T00:00:00+00:00"
    timing_path.write_text(json.dumps(timing))
    row["review_timing_sha256"] = review.digest(timing_path)
    row_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="fresh review after"):
        review.freeze(full, revisions=revisions)


def test_revision_rejects_final_manifest_drift_outside_declared_revision(tmp_path, monkeypatch):
    full, revisions, _ = revision_fixture(tmp_path, monkeypatch)
    manifest = json.loads(full.read_text())
    manifest["cases"][1]["image_id"] = "undeclared-final-change"
    full.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicitly revised benchmark contract"):
        review.freeze(full, revisions=revisions)
