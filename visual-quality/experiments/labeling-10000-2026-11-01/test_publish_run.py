"""Exercise publication boundaries and joins with small synthetic records."""

import json

import publish_run as publish
import pytest

from visual_quality.publication import sha, verify_publication


def inputs():
    cases = [
        {
            "case_id": name,
            "family_id": f"family-{name}",
            "view_sha256": f"view-{name}",
            "image_path": f"/images/{name}.png",
            "benchmark_order": order,
            "benchmark_partition": "representative" if order else "rehearsal",
        }
        for name, order in (("R9000", 1), ("R0001", 2), ("R0002", None))
    ]
    refs = []
    for case in cases[:2]:
        refs.append(
            {
                "case_id": case["case_id"],
                "view_sha256": case["view_sha256"],
                "eligibility": "eligible",
                "price_usd": 0,
                "latency_seconds": None,
                "review_elapsed_seconds": 12.5,
                **{
                    task: {"status": "ok", "score": 3.0, "evidence": "Visible detail.", "limitations": []}
                    for task in ("technical_quality", "aesthetics")
                },
            }
        )
    observations = {
        model: [
            {
                **case,
                "model": model,
                "arm": "baseline",
                "status": "valid",
                "technical_score": 3.0,
                "aesthetic_score": 4.0,
                "technical_score_valid": True,
                "aesthetic_score_valid": True,
                "latency_seconds": 1.25,
                "latency_kind": "http_request_through_complete_response",
                "price_usd": "0.00" if model == "qrealign_lite" else "0.001",
                "attempt_count": 1,
            }
            for case in cases
        ]
        for model in publish.MODELS
    }
    stats = [
        {
            "case_id": case["case_id"],
            "view_sha256": case["view_sha256"],
            "image_stats": {
                **dict.fromkeys(publish.STATS, 0.5),
                "used_in_inference": False,
                "training_features_enabled": False,
            },
        }
        for case in cases
    ]
    return {"cases": cases}, refs, observations, stats


def build(values=None):
    return publish.build_exports(*(values or inputs()), expected_images=3, expected_references=2)


def test_full_export_uses_canonical_ensemble_and_separate_statistics():
    rows, labelled, summary = build()
    assert len(rows) == 20
    assert summary["images"] == 3 and summary["reference_records"] == 2
    assert sum(row["model"] == "photo_ensemble" for row in rows) == 3
    assert not any(row["model"] == "luna_deepseek_mean" for row in rows)
    assert all("image_stats" not in row for row in rows)
    assert all(record["image_stats"]["histogram_entropy"] == 0.5 for record in labelled)
    for row in rows:
        if row["model"] in ("reference", "qrealign_lite"):
            assert row["price_usd"] == "0.00"
        if row["model"] == "reference":
            assert row["latency_seconds"] is None and row["review_elapsed_seconds"] == 12.5


def test_missing_model_has_explicit_slot_and_missing_ensemble_without_reweighting():
    values = inputs()
    values[2]["liquid"].pop()
    rows, _, summary = build(values)
    liquid = next(r for r in rows if r["case_id"] == "R0002" and r["model"] == "liquid")
    ensemble = next(r for r in rows if r["case_id"] == "R0002" and r["model"] == "photo_ensemble")
    assert liquid["status"] == "missing" and liquid["price_usd"] is None
    assert ensemble["technical_score"] is None
    assert summary["coverage"]["liquid"]["numeric_pairs"] == 2


@pytest.mark.parametrize("source", ["reference", "stats", "api"])
def test_mismatched_view_cannot_join(source):
    values = inputs()
    target = (
        values[1][0] if source == "reference" else values[3][0] if source == "stats" else values[2]["luna"][0]
    )
    target["view_sha256"] = "other-view"
    with pytest.raises(ValueError, match="Changed delivered view"):
        build(values)


def test_family_leakage_and_bad_order_are_rejected():
    values = inputs()
    values[0]["cases"][2]["family_id"] = values[0]["cases"][0]["family_id"]
    with pytest.raises(ValueError, match="family leaks"):
        build(values)
    values = inputs()
    values[0]["cases"][1]["benchmark_order"] = 1
    with pytest.raises(ValueError, match="unique and contiguous"):
        build(values)


def test_publication_selects_explicit_order_not_case_id_and_verifies_every_file(tmp_path):
    manifest, refs, _, _ = inputs()
    rows, labelled, summary = build()
    result = publish.write_publication(tmp_path / "staged", manifest, refs, rows, labelled, summary, limit=1)
    assert result["views"] == 1 and result["rows"] == 7
    ids = json.loads((tmp_path / "staged/publication-case-ids.json").read_text())
    assert ids == ["R9000"]
    assert verify_publication(tmp_path / "staged")["views"] == 1


def stage(tmp_path):
    manifest, refs, _, _ = inputs()
    rows, labelled, summary = build()
    destination = tmp_path / "staged"
    publish.write_publication(destination, manifest, refs, rows, labelled, summary, limit=1)
    return destination


@pytest.mark.parametrize("name", ["all-scores.csv", "collage.png", "hidden/labels.jsonl", "results.tar.gz"])
def test_extra_export_paths_cannot_hide_nonpublic_labels(tmp_path, name):
    destination = stage(tmp_path)
    extra = destination / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("not permitted")
    with pytest.raises(ValueError, match="Unexpected or missing"):
        verify_publication(destination)


def test_nested_nonpublic_score_is_rejected_even_with_updated_file_hash(tmp_path):
    destination = stage(tmp_path)
    path = destination / "labelled-images.jsonl"
    row = json.loads(path.read_text())
    row["hidden"] = {"case_id": "R0002", "technical_score": 4.5}
    path.write_text(json.dumps(row) + "\n")
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["records"][path.name] = sha(path)
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Non-public image"):
        verify_publication(destination)


def test_incomplete_inputs_do_not_publish_silently(tmp_path):
    values = inputs()
    values[2]["liquid"].pop()
    rows, labelled, summary = build(values)
    with pytest.raises(ValueError, match="Incomplete model coverage"):
        publish.write_publication(
            tmp_path / "blocked", values[0], values[1], rows, labelled, summary, limit=1
        )
    result = publish.write_publication(
        tmp_path / "reviewed",
        values[0],
        values[1],
        rows,
        labelled,
        summary,
        limit=1,
        incomplete_reason="Provider failed after bounded retries; gap retained.",
    )
    assert result["views"] == 1
    values[3].pop()
    rows, labelled, summary = build(values)
    with pytest.raises(ValueError, match="complete statistics"):
        publish.write_publication(
            tmp_path / "nostats",
            values[0],
            values[1],
            rows,
            labelled,
            summary,
            limit=1,
            incomplete_reason="Model gap reviewed",
        )


def test_stats_cannot_enter_model_observations_or_change_entropy_alias():
    values = inputs()
    values[2]["luna"][0]["image_stats"] = {"sharpness_peak": 1}
    with pytest.raises(ValueError, match="outside inference"):
        build(values)
    values = inputs()
    values[3][0]["image_stats"]["histogram_entropy"] = 0.1
    with pytest.raises(ValueError, match="alias differs"):
        build(values)


def test_interrupted_request_is_counted_with_unknown_cost(tmp_path):
    path = tmp_path / "luna/attempts/R9000/01/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"case_id": "R9000", "model": "luna", "view_sha256": "view-R9000", "plan_id": "first"})
    )
    observations, audit = publish.load_api(tmp_path)
    assert observations["luna"][0]["status"] == "observation_not_finalized"
    assert observations["luna"][0]["technical_score"] is None
    assert observations["luna"][0]["source_request_plan_id"] == "first"
    assert audit["luna"] == {
        "attempts": 1,
        "known_price_usd": "0",
        "unpriced_attempts": 1,
        "total_price_usd": None,
    }


def test_lite_transport_continuation_keeps_original_attempt_and_measured_latency():
    common = {
        "case_id": "R1",
        "family_id": "F1",
        "view_sha256": "view",
        "checkpoint": "lite",
        "revision": "revision",
        "request_sha256": "request",
        "price_usd": "0.00",
    }
    failed = dict(
        common,
        attempt_id="a",
        status="http_error",
        http_status=520,
        request_started_at="2026-11-01T00:00:00+00:00",
        latency_seconds=1.5,
    )
    valid = dict(
        common,
        attempt_id="b",
        status="valid",
        http_status=200,
        request_started_at="2026-11-01T00:01:00+00:00",
        latency_seconds=4.0,
        technical_score=3.2,
        aesthetic_score=2.9,
    )
    result = publish.merge_lite_attempts([failed, valid])[0]
    assert result["technical_score"] == 3.2 and result["latency_seconds"] == 4.0
    assert result["attempt_count"] == 2 and result["total_attempt_latency_seconds"] == 5.5
    assert result["attempt_provenance"][0]["http_status"] == 520
    with pytest.raises(ValueError, match="Multiple Lite assessments"):
        publish.merge_lite_attempts([dict(failed, status="valid"), valid])
    with pytest.raises(ValueError, match="changed view_sha256"):
        publish.merge_lite_attempts([failed, dict(valid, view_sha256="other")])


def test_benchmark_model_call_cannot_precede_its_reference():
    values = inputs()
    values[1][0]["review_completed_at"] = "2026-11-01T12:00:00+00:00"
    values[2]["luna"][0]["request_started_at"] = "2026-11-01T11:59:00+00:00"
    with pytest.raises(ValueError, match="preceded independent reference"):
        build(values)
    values[2]["luna"][0]["request_started_at"] = "2026-11-01T12:01:00+00:00"
    build(values)


def test_lite_transport_failure_remains_explicit_without_a_raw_response(tmp_path):
    values = inputs()
    failed = values[2]["qrealign_lite"][0]
    failed.update(
        status="transport_or_parse_error",
        technical_score=None,
        aesthetic_score=None,
        technical_score_valid=False,
        aesthetic_score_valid=False,
        raw_response_path=str(tmp_path / "absent.json"),
    )
    rows, _, _ = build(values)
    row = next(r for r in rows if r["case_id"] == failed["case_id"] and r["model"] == "qrealign_lite")
    assert row["technical_score"] is None and row["raw_response_missing"] is True
    assert row["price_usd"] == "0.00"
