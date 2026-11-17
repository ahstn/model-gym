"""Git aggregates must not carry nested per-image evidence or arbitrary text."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("publish_40k_aggregate", HERE / "publish_aggregate.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def fixture():
    ledger = {
        model: {
            "known_price_usd": "1.20",
            "unpriced_attempts": 1,
            "total_price_usd": None,
            "attempts": [{"case_id": "R19999", "price_usd": "0.01"}],
        }
        for model in publication.MODELS
    }
    summary = {key: 40000 if key in ("images", "total_images") else 0 for key in publication.COUNTS}
    summary.update(
        training_ready=False,
        stats_in_inference=False,
        exports={"labelled-images.jsonl": "a" * 64},
        current_plans={
            model: {"phase": "expansion", "plan_id": model + "-plan", "cases": 20000}
            for model in publication.MODELS
        },
        coverage={
            "luna": {
                "expected": 40000,
                "numeric_pairs": 39999,
                "status_counts": {"valid": 39999, "failed": 1},
                "missing_case_ids": ["R19999"],
            }
        },
        incremental_coverage={},
        historical_cost_per_model=ledger,
        combined_cost_per_model=ledger,
        accounting=ledger,
        compatibility_probe_accounting=ledger,
        ensemble={
            "weights": {
                dimension: {"luna": "0.375", "deepseek": "0.375", "liquid": "0.25"}
                for dimension in ("technical", "aesthetic")
            },
            "components": [{"case_id": "R19999", "technical_score": 1.8}],
        },
        new_model_metrics=[
            {
                "model": "luna",
                "numeric_pairs": 20000,
                "expected": 20000,
                "mean_technical": 3.2,
                "mean_aesthetic": 3.4,
                "median_final_attempt_seconds": 8.1,
                "requested_high_reasoning": 20000,
                "example": {"case_id": "R19999"},
            }
        ],
        corpus_coverage={
            "categories": {"megalith": 15000},
            "neighbors": ["R19999"],
            "clusters": [
                {
                    "cluster": 0,
                    "candidate_count": 400,
                    "initial_count": 200,
                    "new_count": 300,
                    "theme": "example R19999",
                    "review_ids": ["R19999"],
                }
            ],
        },
        possible_duplicate_pairs={"pairs": [{"left": "R19999", "right": "R38888", "cosine": 0.99}]},
        embedding_cleanup={"status": "deleted", "review_case": "R19999"},
        image_path="/private/R19999.jpg",
        assessments=[{"case_id": "R19999", "score": 1.8}],
    )
    audit = {
        key: 0
        for key in (
            "new_images",
            "raw_observations_reparsed",
            "image_statistics_joins",
            "initial_records_unchanged",
            "new_qrealign_calls",
            "reference_images",
            "publication_images",
        )
    }
    audit.update(
        passed=True,
        total_images=40000,
        review_ids=["R19999"],
        exports_sha256="a" * 64,
        current_plans=json.loads(json.dumps(summary["current_plans"])),
    )
    return summary, audit


def test_nested_image_evidence_is_omitted_from_json_and_report():
    summary, audit = fixture()
    result = publication.project(summary, audit)
    encoded = json.dumps(result)
    output = encoded + publication.report(result)
    for identity in ("R19999", "R38888"):
        assert identity not in output
    for field in ("image_path", "technical_score", "neighbors", "review_ids", "assessments"):
        assert f'"{field}"' not in encoded
    assert result["corpus_coverage"]["possible_duplicate_pairs"] == 1
    assert result["combined_cost_per_model"]["luna"]["total_price_usd"] is None
    assert result["new_cost_per_model"]["luna"] == {
        "known_price_usd": "2.40",
        "unpriced_attempts": 2,
        "total_price_usd": None,
    }


def test_unrecognized_model_or_status_cannot_hide_an_image_id():
    summary, audit = fixture()
    summary["coverage"]["luna"]["status_counts"]["R19999"] = 1
    with pytest.raises(ValueError, match="status"):
        publication.project(summary, audit)
    summary, audit = fixture()
    summary["new_model_metrics"][0]["model"] = "R19999"
    with pytest.raises(ValueError, match="grader"):
        publication.project(summary, audit)


def test_numeric_fields_cannot_carry_arbitrary_text():
    summary, audit = fixture()
    summary["corpus_coverage"]["clusters"][0]["new_count"] = "R19999"
    with pytest.raises(ValueError, match="counts"):
        publication.project(summary, audit)
    summary, audit = fixture()
    summary["accounting"]["luna"]["known_price_usd"] = "R19999"
    with pytest.raises(ValueError, match="monetary"):
        publication.project(summary, audit)


def test_final_audit_and_canonical_ensemble_are_required():
    summary, audit = fixture()
    with pytest.raises(ValueError, match="audit"):
        publication.project(summary, {**audit, "passed": False})
    summary["ensemble"]["weights"]["technical"]["luna"] = "0.5"
    with pytest.raises(ValueError, match="weights"):
        publication.project(summary, audit)


@pytest.mark.parametrize("mismatch", ["hash", "plan"])
def test_matching_40k_count_does_not_admit_a_stale_audit(mismatch):
    summary, audit = fixture()
    if mismatch == "hash":
        audit["exports_sha256"] = "b" * 64
    else:
        audit["current_plans"]["luna"]["plan_id"] = "previous-plan"
    assert audit["total_images"] == summary["total_images"] == 40000
    with pytest.raises(ValueError, match="hash or current plans"):
        publication.project(summary, audit)
