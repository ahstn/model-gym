"""Check strict null handling, identity joins and clustered uncertainty."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("string_trial_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def panel():
    cases, references, observations, attempts = [], [], [], []
    for number in range(16):
        controlled = number < 8
        case = {
            "case_id": f"T{number:03d}",
            "image_id": f"image-{number}",
            "view_id": f"view-{number}",
            "source_sha256": f"{number:064x}",
            "view_sha256": f"{number + 100:064x}",
            "family_id": f"family-{number // 4 if controlled else number}",
            "stratum": "controlled" if controlled else "vizwiz" if number < 12 else "standalone_megalith",
            "role": ("source_parent" if number % 4 == 0 else "variant")
            if controlled
            else "vizwiz"
            if number < 12
            else "standalone_megalith",
            "operator": "gaussian_blur" if controlled else None,
            "severity": number % 4 or None if controlled else None,
            "photo_domain_eligible": True,
        }
        cases.append(case)
        references.append(
            {
                "case_id": case["case_id"],
                "view_sha256": case["view_sha256"],
                "score": 3.0,
                "interval": [2.8, 3.2],
                "evidence": "Visible detail has mild softness.",
                "defects": ["blur"],
                "reviewer_id": "assistant-test",
            }
        )
        observations.append(
            {
                **{
                    key: case[key]
                    for key in (
                        "case_id",
                        "image_id",
                        "view_id",
                        "source_sha256",
                        "view_sha256",
                        "photo_domain_eligible",
                    )
                },
                "delivered_sha256": case["view_sha256"],
                "wire_score": "3.00",
                "adapter_version": "string-score-to-decimal-v1",
                "technical_quality": {
                    "task": "technical_quality",
                    "status": "ok",
                    "score": 3.0,
                    "evidence": "Visible detail has mild softness.",
                    "limitations": [],
                },
                "aesthetics": {"status": "not_requested", "score": None},
                "review_status": "pending_human_review",
                "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
            }
        )
        attempts.append(
            {
                "case_id": case["case_id"],
                "status": "valid",
                "reason": None,
                "cost_usd": 0.001,
                "wire_score_valid": True,
            }
        )
    return cases, references, observations, attempts


def set_score(observation, score):
    observation["technical_quality"]["score"] = score
    observation["wire_score"] = f"{score:.2f}"


def test_missing_invalid_and_unscorable_are_null_not_zero():
    cases, references, observations, attempts = panel()
    observations[0]["technical_quality"].update(status="unscorable", score=None)
    observations[0]["wire_score"] = None
    attempts[1].update(status="invalid", reason="invalid string score", wire_score_valid=False)
    observations.pop(1)
    observations.pop()
    attempts.pop()
    summary, rows = report.build_report(cases, references, observations, attempts)
    assert len(rows) == 16
    assert rows[0]["model_score"] is None
    assert rows[1]["interval_distance"] is None
    assert rows[-1]["attempt_status"] == "missing"
    assert summary["agreement"]["reference_comparisons"] == 13
    assert summary["agreement"]["mean_interval_distance"] == 0
    assert summary["format_conformance"]["strict_valid_rate_among_attempts"] == 14 / 15
    assert summary["format_conformance"]["strict_valid_rate_among_planned_cases"] == 14 / 16
    assert summary["format_conformance"]["all_200_exact_wire_and_full_contract"] is False
    assert summary["format_conformance"]["component_conformance"]["wire_score_valid"] == {
        "assessed": 15,
        "passed": 14,
        "unknown": 0,
        "rate_among_assessed": 14 / 15,
    }
    assert summary["unscorable_case_ids"] == ["T000"]
    assert summary["invalid_case_ids"] == ["T001"]
    assert summary["missing_case_ids"] == ["T015"]
    assert summary["training_ready"] is False
    assert summary["explanation_support"]["unsupported_count"] is None


def test_metrics_boundaries_bias_and_null_references():
    cases, references, observations, attempts = panel()
    set_score(observations[0], 4.0)
    references[0].update(score=1.5, interval=[1, 2])
    set_score(observations[1], 2.0)
    references[1].update(score=4.5, interval=[4, 5])
    set_score(observations[2], 4.0)
    references[2].update(score=3.0, interval=[3, 3])
    references[3].update(score=None, interval=None, unscorable_reason="Image could not be inspected")
    summary, rows = report.build_report(cases, references, observations, attempts)
    metrics = summary["agreement"]
    assert metrics["reference_comparisons"] == 15
    assert metrics["mean_interval_distance"] == pytest.approx(5 / 15)
    assert metrics["mae_from_assistant_point"] == pytest.approx(6 / 15)
    assert metrics["mean_score_bias"] == pytest.approx(1 / 15)
    assert metrics["large_interval_error_count"] == 3
    assert metrics["critical_case_ids"] == ["T000", "T001"]
    assert metrics["interval_hits"] == 12
    assert rows[3]["interval_distance"] is None
    assert summary["null_reference_case_ids"] == ["T003"]


def test_family_bootstrap_respects_dependence_strata_and_order():
    cases, references, observations, attempts = panel()
    for observation in observations[:4]:
        set_score(observation, 5)
    _, rows = report.build_report(cases, references, observations, attempts)
    intervals = report.bootstrap_intervals(rows)
    assert intervals == report.bootstrap_intervals(list(reversed(rows)))
    assert intervals["mean_interval_distance"] == pytest.approx([0, 0.9])
    naive = (
        np.random.default_rng(report.BOOTSTRAP_SEED)
        .choice([row["interval_distance"] for row in rows], size=(10_000, len(rows)))
        .mean(axis=1)
    )
    assert np.quantile(naive, 0.025) > intervals["mean_interval_distance"][0]
    # Incomplete families have unequal sizes; ratio sums/counts keeps constant errors exact.
    for row in rows:
        for key in report.METRICS.values():
            row[key] = 0.5
    subset = rows[2:]
    assert all(value == pytest.approx([0.5, 0.5]) for value in report.bootstrap_intervals(subset).values())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c, r, o, a: o[0].update(delivered_sha256="f" * 64), "identity"),
        (lambda c, r, o, a: r[0].update(view_sha256="f" * 64), "Reference view"),
        (lambda c, r, o, a: o[0].update(wire_score="3.0"), "two-decimal"),
        (lambda c, r, o, a: o[0].update(wire_score="3.10"), "numeric score differ"),
        (lambda c, r, o, a: o[0]["task_masks"].update(technical_quality=True), "masks"),
        (lambda c, r, o, a: o[0]["task_masks"].update(technical_quality=0), "boolean false"),
        (lambda c, r, o, a: o[0].update(review_status="approved"), "pending"),
        (lambda c, r, o, a: o[0]["aesthetics"].update(score=3), "Aesthetics"),
        (lambda c, r, o, a: a[0].update(status="invalid"), "must match"),
        (lambda c, r, o, a: o.append(copy.deepcopy(o[0])), "Duplicate"),
        (lambda c, r, o, a: r.pop(), "Every planned"),
        (lambda c, r, o, a: c[-1].update(family_id=c[0]["family_id"]), "cross strata"),
    ],
)
def test_identity_contract_and_join_fail_closed(mutation, match):
    args = panel()
    mutation(*args)
    with pytest.raises(ValueError, match=match):
        report.build_report(*args)


def test_unattempted_sample_has_null_metrics_and_no_invented_agreement():
    cases, references, _, _ = panel()
    summary, rows = report.build_report(cases, references, [], [])
    assert len(rows) == len(cases)
    assert summary["agreement"]["mean_interval_distance"] is None
    assert summary["agreement"]["percentile_95_family_bootstrap_intervals"]["mean_interval_distance"] is None
    assert summary["format_conformance"]["strict_valid_rate_among_attempts"] is None
    assert summary["severity_ordering"]["scored_adjacent_pairs"] == 0


def test_controlled_severity_is_diagnostic_with_ties_and_reversals():
    cases, references, observations, attempts = panel()
    for observation, score in zip(observations[:4], [4.0, 3.5, 3.5, 3.8]):
        set_score(observation, score)
    summary, _ = report.build_report(cases, references, observations, attempts)
    ordering = summary["severity_ordering"]
    assert ordering["planned_adjacent_pairs"] == 6
    assert ordering["scored_adjacent_pairs"] == 6
    assert ordering["ties"] == 4
    assert ordering["nonincreasing_pairs"] == 5
    assert ordering["reversals"][0]["less_degraded_case_id"] == "T002"
    assert ordering["reversals"][0]["more_degraded_case_id"] == "T003"
    assert summary["controlled_source_parents"]["planned_cases"] == 2
    assert summary["controlled_variants"]["planned_cases"] == 6


def test_frozen_provenance_must_match_numeric_records():
    cases, references, observations, attempts = panel()
    run = {
        "run_id": "run-id",
        "prompt": {"version": "visual-rating-v1-string-score-v1", "sha256": "e" * 64},
        "settings": {"model": "deepseek-v4-1-flash-1017"},
    }
    for case, observation in zip(cases, observations):
        case["request_sha256"] = "d" * 64
        observation["provenance"] = {
            "provider": "runinfra",
            "run_id": run["run_id"],
            "attempt_id": case["case_id"],
            "prompt_version": run["prompt"]["version"],
            "prompt_sha256": run["prompt"]["sha256"],
            "request_sha256": case["request_sha256"],
            "model_requested": run["settings"]["model"],
        }
    summary, _ = report.build_report(cases, references, observations, attempts, run)
    assert summary["agreement"]["reference_comparisons"] == 16
    observations[0]["provenance"]["prompt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        report.build_report(cases, references, observations, attempts, run)
