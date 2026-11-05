"""Check missingness, native precision, matched cohorts and accounting semantics."""

import json
from collections import Counter
from decimal import Decimal

import pytest
import report_run as report


def observation(identity, model, technical, aesthetic=3, *, valid=True, status="valid"):
    return {
        "case_id": identity,
        "model": model,
        "view_sha256": f"view-{identity}",
        "status": status,
        "technical_score": technical,
        "aesthetic_score": aesthetic,
        "technical_score_valid": valid,
        "aesthetic_score_valid": valid,
        "reference_eligibility": "eligible",
    }


def test_unscorable_or_invalid_is_missing_not_zero_even_with_valid_flag():
    for value in (None, True, float("nan"), float("inf"), 0, 6):
        assert report.score(observation("a", "deepseek", value), "technical") is None
    assert report.score(observation("a", "deepseek", 3.125), "technical") == 3.125
    assert report.score(observation("a", "deepseek", 3, valid=False), "technical") is None
    assert report.score(observation("a", "deepseek", 3, status="failed"), "technical") is None


def test_tied_ranks_and_constant_rank_correlation():
    assert report.ranks([3, 1, 1, 2]) == [4, 1.5, 1.5, 3]
    assert report.spearman([1, 2, 2, 4], [4, 2, 2, 1]) == pytest.approx(-1)
    assert report.spearman([1, 1], [2, 3]) is None
    assert report.spearman([], []) is None
    assert report.quantile([1, 3, 5, 7], 0.25) == 2.5
    assert report.quantile([2], 0.95) == 2


def test_missing_coverage_does_not_change_paired_comparison_denominator():
    rows = []
    for identity, reference in [("a", 1), ("b", 5)]:
        rows.append(observation(identity, "reference", reference))
        for model in report.MODELS:
            value = 2 if identity == "a" else 4
            if model == "qrealign_lite" and identity == "b":
                value = None
            rows.append(observation(identity, model, value))
    indexed = report.index_rows(rows)
    result = report.group_metrics(indexed, ["a", "b"], {})
    assert result["per_model"]["luna"]["technical"]["n"] == 2
    assert result["per_model"]["qrealign_lite"]["technical"]["n"] == 1
    assert result["common_coverage"]["all_six"]["technical"]["n"] == 1
    assert result["common_coverage"]["four_api_and_ensemble"]["technical"]["n"] == 2
    assert result["paired_ensemble"]["qrealign_lite"]["technical"]["n"] == 1
    assert result["paired_ensemble"]["qrealign_lite"]["technical"]["ensemble_minus_model_mae"] == 0


def test_tail_thresholds_use_native_precision_and_exact_boundaries():
    tails = report.technical_tails([(2, 2.9999), (2, 3), (2.001, 4), (4, 2.9999), (4, 3), (3.99, 1)])
    assert tails == {
        "poor_reference_n": 2,
        "poor_predicted_at_least_3": 1,
        "strong_reference_n": 2,
        "strong_predicted_below_3": 1,
    }
    assert report.agreement([(1, 1.11111)])["mae"] == pytest.approx(0.11111)


def test_duplicate_or_changed_views_are_rejected():
    first = observation("a", "luna", 2)
    with pytest.raises(ValueError, match="Duplicate"):
        report.index_rows([first, first])
    changed = observation("a", "reference", 2)
    changed["view_sha256"] = "other"
    with pytest.raises(ValueError, match="Different delivered"):
        report.index_rows([first, changed])


def test_unknown_costs_are_preserved_and_transport_latency_excluded():
    rows, summary = [], {"cost_per_model": {}}
    for model in (*report.MODELS[:-1], "reference"):
        rows.extend(
            [
                {
                    **observation("a", model, 2),
                    "price_usd": ".001",
                    "latency_seconds": 4,
                    "latency_kind": "http_request_through_complete_response",
                },
                {
                    **observation("b", model, 2),
                    "price_usd": None,
                    "latency_seconds": 200,
                    "latency_kind": "http_attempt_until_transport_failure",
                },
            ]
        )
        summary["cost_per_model"][model] = {
            "known_price_usd": ".003",
            "total_price_usd": None,
            "unpriced_attempts": 2,
        }
    costs, latency = report.cost_and_latency(rows, summary)
    assert Decimal(costs["jev"]["selected_known_price_usd"]) == Decimal(".001")
    assert costs["jev"]["total_price_usd"] is None
    assert costs["jev"]["unpriced_attempts"] == 2
    assert costs["jev"]["selected_unpriced_observations"] == 1
    assert latency["jev"]["median_seconds"] == 4
    assert latency["jev"]["complete_response_n"] == 1


def full_fixture():
    rows = []
    for order in range(1, 1501):
        for model in (*report.MODELS, "reference"):
            row = observation(f"private-{order}", model, 3)
            row.update(
                benchmark_order=order,
                price_usd="0.00",
                latency_seconds=None,
                latency_kind="no_request_recorded",
            )
            rows.append(row)
    coverage = {
        model: {"expected": 1500, "numeric_pairs": 1500, "status_counts": {"valid": 1500}}
        for model in (*report.MODELS, "reference")
    }
    costs = {
        model: {
            "known_price_usd": "0.00",
            "total_price_usd": "0.00",
            "unpriced_attempts": 0,
            "attempts": 1500,
        }
        for model in (*report.MODELS[:-1], "reference")
    }
    return rows, {"images": 1500, "benchmark_images": 1500, "coverage": coverage, "cost_per_model": costs}


def test_finished_rejects_incomplete_request_and_summary_mismatch():
    rows, summary = full_fixture()
    rows[0].update(status="in_flight", technical_score=None, aesthetic_score=None)
    with pytest.raises(ValueError, match="coverage disagree"):
        report.build_report(rows, summary, {}, run_state="in-progress")
    summary["coverage"]["luna"].update(
        numeric_pairs=1499,
        status_counts=dict(Counter(row["status"] for row in rows if row["model"] == "luna")),
    )
    with pytest.raises(ValueError, match="unfinished"):
        report.build_report(rows, summary, {}, run_state="finished")
    result = report.build_report(rows, summary, {}, run_state="in-progress")
    assert result["groups"]["original_500"]["per_model"]["luna"]["technical"]["n"] == 499


def test_aggregate_report_has_no_individual_labels_and_cost_table_is_last():
    rows, summary = full_fixture()
    result = report.build_report(rows, summary, {}, run_state="finished")
    serialized = json.dumps(result)
    assert "case_id" not in serialized and "private-" not in serialized
    markdown = report.render(result)
    assert "private-" not in markdown
    assert markdown.splitlines()[-1].startswith("| Reference |")
    assert "37.5% Luna + 37.5% DeepSeek + 25% Liquid" in markdown
    assert result["groups"]["confirmation_500"]["expected"] == 500


def test_softness_excludes_scope_excluded_references():
    rows = []
    for identity in ["a", "b"]:
        rows.append(observation(identity, "reference", 2))
        rows.extend(observation(identity, model, 3) for model in report.MODELS)
    rows[0]["reference_eligibility"] = "excluded"
    stats = {"a": {"sharpness_peak": 1000}, "b": {"sharpness_peak": 1}}
    result = report.group_metrics(report.index_rows(rows), ["a", "b"], stats)["softness_diagnostic"]
    assert result["eligible_n"] == 1 and result["threshold"] == 1
    assert result["models"]["luna"]["soft"]["bias"] == 1


def test_infrastructure_is_posted_receipt_not_inference_price(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "billing": {
                    "metadata": {"totals": {"totalAmount": 0.3}},
                    "records": [
                        {"totalAmount": 0.1, "endTime": "2026-11-01T22:00:00Z"},
                        {"totalAmount": 0.2, "endTime": "2026-11-01T23:00:00Z"},
                    ],
                },
                "receipt_saved_at": "2026-11-02T00:00:00Z",
                "completeness": "Partial",
            }
        )
    )
    result = report.infrastructure_receipt(path)
    assert result["posted_price_usd"] == "0.3"
    assert result["posted_through"] == "2026-11-01T23:00:00Z"
    assert result["completeness"] == "Partial"


def test_scope_confusion_keeps_uncertain_and_missing_separate_from_errors():
    references = ["eligible", "eligible", "excluded", "excluded", "uncertain", "eligible"]
    rows = []
    for index, eligibility in enumerate(references):
        identity = str(index)
        rows.append({**observation(identity, "reference", 1), "reference_eligibility": eligibility})
        if index == 1:  # Missing response must not become a correct non-floor decision.
            continue
        predicted = {**observation(identity, "luna", 1), "scope_floor_applied": index in (0, 4)}
        if index == 3:
            predicted["status"] = "failed"  # A placeholder false flag is not an assessment.
        rows.append(predicted)
    result = report.scope_confusion(report.index_rows(rows), [str(i) for i in range(6)], "luna")
    assert result["reference_eligible_n"] == 3
    assert result["eligible_classified_n"] == 2
    assert result["eligible_false_floor_n"] == 1
    assert result["eligible_unknown_n"] == 1
    assert result["reference_excluded_n"] == 2
    assert result["excluded_classified_n"] == 1
    assert result["excluded_miss_n"] == 1
    assert result["excluded_unknown_n"] == 1
    assert result["reference_uncertain_or_missing_n"] == 1
    assert result["classification_unknown_n"] == 2
    # Numeric 1 alone says nothing about a scope floor; conflicting flags are unknown.
    assert report.scope_decision(observation("a", "liquid", 1)) is None
    assert (
        report.scope_decision(
            {**observation("a", "liquid", 1), "scope_excluded": True, "scope_floor_applied": False}
        )
        is None
    )
    assert (
        report.scope_decision(
            {**observation("a", "liquid", 1), "scope_excluded": False, "scope_validation_error": "missing"}
        )
        is None
    )
