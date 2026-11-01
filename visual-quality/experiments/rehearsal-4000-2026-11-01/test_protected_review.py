"""Synthetic protected-slice boundaries, incomplete labels and artifact bindings."""

from __future__ import annotations

import copy
import json

import analyze
import api_contract
import protected_review as p
import pytest
from test_analysis import FROZEN, cases, reference


def scored(technical=3.0, aesthetic=4.0, *, floor=False):
    return {
        "technical_score": technical,
        "aesthetic_score": aesthetic,
        "technical_score_valid": technical is not None,
        "aesthetic_score_valid": aesthetic is not None,
        "scope_floor_applied": floor,
    }


def ref(eligibility="eligible", technical=3.0, aesthetic=4.0):
    row = reference(cases(1)[0], technical, aesthetic)
    row["eligibility"] = eligibility
    return row


def test_exact_half_point_increase_is_flagged_without_output_rounding():
    result = p.style_comparison(ref(technical=1.0), scored(1.8), scored(2.3), "technical_score")
    assert result["delta_absolute_error"] == 0.4999999999999998
    assert result["large_error_increase"]
    nearby = p.style_comparison(ref(technical=1.0), scored(1.8), scored(2.29999999), "technical_score")
    assert not nearby["large_error_increase"]
    improvement = p.style_comparison(ref(technical=1.0), scored(2.3), scored(1.8), "technical_score")
    assert improvement["improvement"] and improvement["delta_absolute_error"] < 0
    assert not improvement["large_error_increase"]


def test_new_missing_task_is_flagged_and_sibling_stays_comparable():
    base, treatment = scored(), scored(None, 4.2)
    technical = p.style_comparison(ref(), base, treatment, "technical_score")
    aesthetic = p.style_comparison(ref(), base, treatment, "aesthetic_score")
    assert technical["new_missing_task"] and technical["delta_absolute_error"] is None
    assert not technical["paired_comparison_complete"]
    assert aesthetic["paired_comparison_complete"] and not aesthetic["new_missing_task"]
    recovered = p.style_comparison(ref(), treatment, base, "technical_score")
    assert recovered["recovered_missing_task"] and not recovered["new_missing_task"]


def test_eligible_new_floor_requires_observed_baseline_scope_state():
    result = p.scope_comparison(ref(), scored(floor=False), scored(1, 1, floor=True))
    assert result["new_scope_error"] and result["new_scope_error_type"] == "new_declared_floor_on_eligible"
    missing_base = p.scope_comparison(ref(), scored(None, floor=False), scored(1, 1, floor=True))
    assert not missing_base["new_scope_error"] and missing_base["incomplete_comparison"]
    absent_flag = p.scope_comparison(ref(), scored(floor=None), scored(1, 1, floor=True))
    assert absent_flag["incomplete_comparison"] and not absent_flag["new_scope_error"]


def test_excluded_new_numeric_violation_can_coexist_with_missing_sibling():
    excluded = ref("excluded", 1, 1)
    result = p.scope_comparison(excluded, scored(1, 1), scored(1.2, None))
    assert result["new_scope_error"] and result["incomplete_comparison"]
    assert result["new_scope_error_type"] == "new_score_above_one_on_excluded"
    unknown = p.scope_comparison(excluded, scored(1, 1), scored(1, None))
    assert unknown["treatment"]["scope_error"] is None
    assert unknown["incomplete_comparison"] and not unknown["new_scope_error"]
    unpaired = p.scope_comparison(excluded, scored(None, 1), scored(2, 2))
    assert not unpaired["new_scope_error"] and unpaired["treatment"]["scope_error"] is True


def test_uncertain_floor_flip_is_descriptive_never_new_scope_error():
    result = p.scope_comparison(ref("uncertain"), scored(), scored(1, 1, floor=True))
    assert result["uncertain_floor_flip_descriptive_only"]
    assert not result["new_scope_error"]
    assert result["baseline"]["scope_error"] is result["treatment"]["scope_error"] is None
    assert result["adjudication_status"] == "pending_visual_review"


def fixture_data():
    corpus = cases(4000)
    for case in corpus[:500]:
        case["benchmark_partition"] = "representative"
    references = [reference(case) for case in corpus[:500]]
    for item in references:
        item["eligibility_evidence"] = "Synthetic photograph reference."
    scores = analyze.build_long(corpus, [], references, frozen_at=FROZEN)
    protected = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "style_cases": [
            {"case_id": case["case_id"], "view_sha256": case["view_sha256"], "tags": ["synthetic_style"]}
            for case in corpus[:12]
        ],
        "scope_cases": [],
    }
    return corpus, references, scores, protected


def test_all_style_tasks_and_all_500_scope_cases_remain_pending_without_scores():
    _, references, scores, protected = fixture_data()
    original = copy.deepcopy(scores)
    result = p.evaluate(protected, references, scores)
    assert len(result["style_rows"]) == 12 * 3 * 2 * 2
    assert len(result["scope_rows"]) == 500 * 3 * 2
    assert len(result["queue"]) == 3000
    assert all(row["incomplete_comparison"] for row in result["scope_rows"])
    assert result["summary"]["adjudication_status"] == "pending_visual_review"
    assert not result["summary"]["advancement_ready"] and not result["summary"]["labels_changed"]
    assert scores == original


def test_binding_validation_rejects_changed_source_reviews_or_missing_export_rows(tmp_path):
    corpus, references, scores, protected = fixture_data()
    paths = {
        name: tmp_path / filename
        for name, filename in (
            ("protected", "protected.json"),
            ("benchmark", "benchmark.json"),
            ("manifest", "manifest.json"),
            ("references", "references.jsonl"),
            ("freeze", "freeze.json"),
            ("scores", "scores.jsonl"),
        )
    }
    benchmark = {"cases": corpus[:500]}
    paths["benchmark"].write_text(json.dumps(benchmark))
    manifest = {
        "cases": corpus,
        "inputs": {"benchmark_manifest_sha256": api_contract.file_sha(paths["benchmark"])},
    }
    paths["manifest"].write_text(json.dumps(manifest))
    for name, rows in (("references", references), ("scores", scores)):
        paths[name].write_text("\n".join(json.dumps(row) for row in rows))
    original_reviews = tmp_path / "original-reviews.jsonl"
    original_reviews.write_bytes(paths["references"].read_bytes())
    protected.update(
        benchmark_manifest_sha256=api_contract.file_sha(paths["benchmark"]),
        reference_files_sha256={str(original_reviews): api_contract.file_sha(original_reviews)},
    )
    paths["protected"].write_text(json.dumps(protected))
    freeze = {
        "references_path": str(paths["references"]),
        "references_sha256": api_contract.file_sha(paths["references"]),
        "manifest_sha256": api_contract.file_sha(paths["manifest"]),
        "case_ids": [row["case_id"] for row in references],
        "frozen_at": FROZEN,
        "review_files": protected["reference_files_sha256"],
    }
    paths["freeze"].write_text(json.dumps(freeze))
    bindings = p.validate_bindings(protected, benchmark, manifest, references, freeze, scores, paths=paths)
    assert bindings["dispatch_timing_unverified"] == []
    with pytest.raises(ValueError, match="identity slots"):
        p.validate_bindings(protected, benchmark, manifest, references, freeze, scores[:-1], paths=paths)
    original_reviews.write_text(original_reviews.read_text() + "\n")
    with pytest.raises(ValueError, match="Original reviewed references changed"):
        p.validate_bindings(protected, benchmark, manifest, references, freeze, scores, paths=paths)


def revision_fixture(tmp_path, eligibility="eligible", replaced=45):
    corpus, references, _, protected = fixture_data()
    old_cases = corpus[:500]
    for i, case in enumerate(old_cases):
        case["benchmark_partition"] = "development" if i < 100 else "representative" if i < 400 else "challenge"
        case.update(candidate_id=case["case_id"], image_id=case["case_id"],
                    image_path="/synthetic/" + case["case_id"], benchmark_stratum="representative:test")
    active_cases = copy.deepcopy(old_cases)
    active_cases[replaced].update(candidate_id="replacement", image_id="replacement",
                                 image_path="/synthetic/replacement", family_id="replacement-family",
                                 view_sha256="replacement-hash")
    old_path, active_path = tmp_path / "old.json", tmp_path / "active.json"
    old_path.write_text(json.dumps({"cases": old_cases}))
    active_path.write_text(json.dumps({"cases": active_cases}))
    old_reviews = tmp_path / "original-reviews.jsonl"
    old_reviews.write_text("\n".join(json.dumps(row) for row in references))
    references[replaced]["view_sha256"] = "replacement-hash"
    references[replaced]["eligibility"] = eligibility
    if eligibility == "excluded":
        for task in p.reference_review.TASKS:
            references[replaced][task].update(score=1.0, limitations=["unsupported_domain"])
    refs_path = tmp_path / "references.jsonl"
    refs_path.write_text("\n".join(json.dumps(row) for row in references))
    fresh_path = tmp_path / "replacement.jsonl"
    fresh_path.write_text(json.dumps(references[replaced]))
    protected.update(benchmark_manifest_sha256=api_contract.file_sha(old_path),
                     reference_files_sha256={str(old_reviews): api_contract.file_sha(old_reviews)})
    original_path = tmp_path / "protected.json"
    original_path.write_text(json.dumps(protected))
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"decision":"replace"}')
    declaration = {
        "schema_version": 1,
        "original_manifest": {"path": str(old_path), "sha256": api_contract.file_sha(old_path)},
        "active_manifest": {"path": str(active_path), "sha256": api_contract.file_sha(active_path)},
        "revisions": [{
            "case_id": old_cases[replaced]["case_id"],
            "previous": {k: old_cases[replaced][k] for k in p.reference_review.REVISION_FIELDS},
            "replacement": {k: active_cases[replaced][k] for k in p.reference_review.REVISION_FIELDS},
            "reason_code": "confirmed_historical_near_duplicate",
            "selection": {"seed": 1},
            "visual_review": {"path": str(evidence), "artifact_sha256": api_contract.file_sha(evidence),
                              "decision": "replace"},
            "excluded_from_final_candidates": [old_cases[replaced]["candidate_id"], "peer"],
            "historical_peer": {"candidate_id": "peer"},
        }],
    }
    revisions_path = tmp_path / "revisions.json"
    revisions_path.write_text(json.dumps(declaration))
    freeze = {
        "references_path": str(refs_path), "references_sha256": api_contract.file_sha(refs_path),
        "frozen_at": FROZEN,
        "review_files": {**protected["reference_files_sha256"], str(fresh_path): api_contract.file_sha(fresh_path)},
        "reference_revision": {"declaration": declaration, "declaration_path": str(revisions_path),
                               "declaration_sha256": api_contract.file_sha(revisions_path)},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    return original_path, revisions_path, freeze_path


@pytest.mark.parametrize("eligibility,scope_count", [("eligible", 0), ("excluded", 1), ("uncertain", 1)])
def test_explicit_revision_preserves_original_and_only_rebinds_declared_scope(tmp_path, eligibility, scope_count):
    original_path, revisions_path, freeze_path = revision_fixture(tmp_path, eligibility)
    original_bytes = original_path.read_bytes()
    revised = p.rebind_protected(original_path, revisions_path, freeze_path)
    assert original_path.read_bytes() == original_bytes
    assert revised["style_cases"] == json.loads(original_bytes)["style_cases"]
    assert len(revised["scope_cases"]) == scope_count
    assert revised["revision_binding"]["revised_case_ids"] == ["R0045"]
    assert revised["revision_binding"]["style_membership_changed"] is False
    p.validate_revision_binding(revised, json.loads(freeze_path.read_text()), freeze_path)
    revised["style_cases"][0]["tags"] = ["silently changed"]
    with pytest.raises(ValueError, match="differs from declared revision"):
        p.validate_revision_binding(revised, json.loads(freeze_path.read_text()), freeze_path)


def test_revision_requires_explicit_binding_and_rejects_original_mutation(tmp_path):
    original_path, revisions_path, freeze_path = revision_fixture(tmp_path)
    original, freeze = json.loads(original_path.read_text()), json.loads(freeze_path.read_text())
    with pytest.raises(ValueError, match="requires an explicit"):
        p.validate_revision_binding(original, freeze, freeze_path)
    revised = p.rebind_protected(original_path, revisions_path, freeze_path)
    original_path.write_text(original_path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance changed"):
        p.validate_revision_binding(revised, freeze, freeze_path)


def test_replaced_style_case_requires_fresh_attribute_review(tmp_path):
    paths = revision_fixture(tmp_path, replaced=0)
    with pytest.raises(ValueError, match="new explicit visual attribute review"):
        p.rebind_protected(*paths)
