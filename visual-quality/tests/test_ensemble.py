"""Canonical weighting, missingness, provenance, and legacy migration contracts."""

import copy
import json
from pathlib import Path

import pytest

from visual_quality.ensemble import (
    MODEL,
    configuration,
    publication_rows,
    read_observations,
    replace_ensemble_rows,
    score_ensemble,
    write_observations,
)


def components():
    return {
        model: {
            "case_id": "R0481",
            "model": model,
            "arm": "baseline",
            "view_sha256": "same-view",
            "status": "valid",
            "technical_score": tech,
            "aesthetic_score": aest,
            "technical_score_valid": True,
            "aesthetic_score_valid": True,
            "scope_floor_applied": False,
            "attempt_id": model + "-attempt",
            "latency_seconds": 2.0,
            "price_usd": "0.0001",
            "prompt_sha256": model + "-prompt",
        }
        for model, tech, aest in (
            ("luna", 2.6, 3.2),
            ("deepseek", 3.0, 3.3),
            ("liquid", 1.6992983898753753, 2.008303607609424),
        )
    }


def test_accepted_weights_use_full_precision_and_retain_independent_dimensions():
    rows = components()
    original = copy.deepcopy(rows)
    result = score_ensemble(rows, case_id="R0481")
    assert result["technical_score"] == pytest.approx(2.524824597468844)
    assert result["aesthetic_score"] == pytest.approx(2.939575901902356)
    assert result["technical_score_display"] == "2.52"
    assert result["aesthetic_score_display"] == "2.94"
    assert result["technical_score_decimal"] == "2.524824597468843825"
    assert result["model"] == MODEL and result["status"] == "valid"
    assert result["latency_seconds"] is None and result["price_usd"] is None
    assert result["component_total_price_usd"] == "0.0003"
    assert result["component_observations"]["liquid"]["prompt_sha256"] == "liquid-prompt"
    assert rows == original


def test_missing_task_does_not_invent_a_score_or_discard_other_task():
    rows = components()
    rows["liquid"]["technical_score"] = None
    result = score_ensemble(rows, case_id="R0481")
    assert result["technical_score"] is None
    assert result["technical_score_valid"] is False
    assert result["aesthetic_score"] is not None
    assert result["status"] == "partial" and result["requires_review"] is True
    assert result["validation_errors"] == {"technical": ["liquid"]}


@pytest.mark.parametrize("invalid", [True, False, float("nan"), float("inf"), "NaN", 0.99, 5.01, "n/a"])
def test_invalid_score_does_not_enter_ensemble(invalid):
    rows = components()
    rows["liquid"]["technical_score"] = invalid
    assert score_ensemble(rows, case_id="R0481")["technical_score"] is None


def test_failed_or_absent_component_never_renormalizes():
    rows = components()
    rows["deepseek"]["status"] = "failed"
    result = score_ensemble(rows, case_id="R0481")
    assert result["technical_score"] is None and result["aesthetic_score"] is None
    del rows["deepseek"]
    result = score_ensemble(rows, case_id="R0481")
    assert result["status"] == "missing_or_invalid_components"
    assert result["component_total_price_usd"] is None
    assert result["component_unpriced_count"] == 1


def test_validity_flag_wins_over_plausible_numeric_value():
    rows = components()
    rows["liquid"]["aesthetic_score_valid"] = "False"
    assert score_ensemble(rows, case_id="R0481")["aesthetic_score"] is None


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("case_id", "another", "model/case"),
        ("model", "jev", "model/case"),
        ("view_sha256", "different", "different delivered"),
        ("view_sha256", None, "hash is required"),
        ("arm", "own_stats", "arm does not match"),
    ],
)
def test_mismatched_component_identity_rejected(field, value, error):
    rows = components()
    rows["liquid"][field] = value
    with pytest.raises(ValueError, match=error):
        score_ensemble(rows, case_id="R0481")


def test_disputed_floor_is_weighted_and_flagged_without_inventing_vote_policy():
    rows = components()
    rows["liquid"].update(technical_score=1, aesthetic_score=1, scope_floor_applied=True)
    result = score_ensemble(rows, case_id="R0481")
    assert result["technical_score"] == pytest.approx(2.35)
    assert result["scope_floor_applied"] is False
    assert result["scope_disagreement"] is True and result["requires_review"] is True
    for row in rows.values():
        row.update(technical_score=1, aesthetic_score=1, scope_floor_applied=True)
    result = score_ensemble(rows, case_id="R0481")
    assert result["technical_score"] == result["aesthetic_score"] == 1
    assert result["scope_floor_applied"] is True


def test_migration_preserves_raw_rows_and_explicitly_maps_baseline_liquid_for_stats():
    rows = list(components().values())
    rows += [dict(row, arm="own_stats") for row in rows if row["model"] != "liquid"]
    old = {"case_id": "R0481", "model": "luna_deepseek_mean", "arm": "baseline"}
    original = copy.deepcopy(rows)
    result = replace_ensemble_rows(rows + [old])
    assert result[: len(rows)] == original
    assert not any(r["model"] == "luna_deepseek_mean" for r in result)
    derived = [r for r in result if r["model"] == MODEL]
    assert len(derived) == 2
    stats = next(r for r in derived if r["arm"] == "own_stats")
    assert stats["component_arms"] == {"luna": "own_stats", "deepseek": "own_stats", "liquid": "baseline"}
    assert stats["component_observations"]["liquid"]["arm"] == "baseline"
    assert replace_ensemble_rows(result) == result


def test_duplicate_component_observation_rejected():
    rows = list(components().values())
    with pytest.raises(ValueError, match="Duplicate component"):
        replace_ensemble_rows(rows + [rows[0]])


def test_csv_and_jsonl_migration_preserve_values_and_metadata(tmp_path):
    rows = list(components().values())
    csv_path = tmp_path / "input.csv"
    write_observations(csv_path, rows)
    derived = replace_ensemble_rows(read_observations(csv_path))[-1]
    assert derived["technical_score"] == pytest.approx(2.524824597468844)
    jsonl_path = tmp_path / "canonical.jsonl"
    write_observations(jsonl_path, [derived])
    assert read_observations(jsonl_path) == [derived]


def test_committed_config_matches_canonical_code():
    path = Path(__file__).resolve().parents[1] / "labeling" / "ensemble.json"
    assert json.loads(path.read_text()) == configuration()


def test_publication_limit_is_images_not_observation_rows():
    rows = [{"case_id": f"R{i:04}", "model": model} for i in range(1001) for model in ("luna", "liquid")]
    selected = publication_rows(rows, [f"R{i:04}" for i in range(1000)])
    assert len(selected) == 2000
    assert all(row["case_id"] != "R1000" for row in selected)
    with pytest.raises(ValueError, match="at most 1,000"):
        publication_rows(rows, [f"R{i:04}" for i in range(1001)])


@pytest.mark.parametrize("identities", [["unknown"], ["R0481", "R0481"], [None], [""], "R0481"])
def test_publication_rejects_invalid_or_unknown_identities(identities):
    with pytest.raises(ValueError, match="Publication allowlist"):
        publication_rows(list(components().values()), identities)
