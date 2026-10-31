"""Focused checks for partial paired evidence and family-level uncertainty."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("runinfra_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def panel(old=3.5, new=3.0):
    cases = []
    for family in range(6):
        for severity in range(4):
            cases.append(
                {
                    "role": "source_parent" if severity == 0 else "variant",
                    "family_id": f"controlled-{family}",
                    "severity": severity or None,
                    "operator": ("gaussian_blur", "gaussian_noise", "jpeg_compression")[family // 2],
                }
            )
    for role, count in (("vizwiz", 14), ("standalone_megalith", 12)):
        cases.extend(
            {"role": role, "family_id": f"{role}-{number}", "severity": None, "operator": None}
            for number in range(count)
        )
    references, baseline, candidate = {}, [], []
    for number, case in enumerate(cases, 1):
        case.update(
            case_id=f"A{number:02d}",
            image_id=f"image-{number}",
            view_id=f"view-{number}",
            sha256=f"{number:064x}",
            view_sha256=f"{number + 50:064x}",
        )
        references[case["case_id"]] = {"score": 3.0, "interval": [3.0, 3.0]}
        base = {
            "image_id": case["image_id"],
            "view_id": case["view_id"],
            "source_sha256": case["sha256"],
            "view_sha256": case["view_sha256"],
            "technical_quality": {
                "task": "technical_quality",
                "status": "ok",
                "score": old,
                "evidence": "Visible detail is clear.",
                "limitations": [],
            },
        }
        baseline.append(base)
        fresh = copy.deepcopy(base)
        fresh["technical_quality"]["score"] = new
        fresh.update(
            case_id=case["case_id"],
            delivered_sha256=case["view_sha256"],
            view_comparison="identical_bytes",
            review_status="pending_human_review",
            aesthetics={"status": "not_requested", "score": None},
            provenance={"prompt_version": "visual-rating-v1", "prompt_sha256": report.PROMPT_SHA256},
            task_masks={"technical_quality": False, "aesthetics": False, "photo_domain": False},
        )
        candidate.append(fresh)
    return cases, references, baseline, candidate


def test_partial_panel_keeps_pairs_without_claiming_full_success():
    cases, references, baseline, candidate = panel()
    summary, rows = report.build_report(cases, references, baseline, candidate[:41])
    assert summary["same_frozen_view_pairs"] == 41
    assert summary["all_50_valid_scorable_same_view_pairs"] is False
    assert summary["decision"] == "inconclusive_incomplete_frozen_view_comparison"
    assert all(summary["diagnostic_thresholds_on_observed_subset"].values())
    assert summary["primary_same_view_subset"]["mean_interval_distance_improvement"] == 0.5
    assert summary["primary_same_view_subset"][
        "paired_95_percent_family_bootstrap_interval"
    ] == pytest.approx([0.5, 0.5])
    assert summary["baseline_all_50"]["reference_comparisons"] == 50
    assert summary["missing_candidate_case_ids"] == [f"A{x:02d}" for x in range(42, 51)]
    assert rows[41]["runinfra_score"] is None
    assert rows[41]["interval_distance_improvement"] is None
    assert summary["explanation_support"]["unsupported_count"] is None


def test_changed_views_and_unscorable_results_remain_separate():
    cases, references, baseline, candidate = panel()
    candidate[0].update(view_comparison="resized", delivered_sha256="f" * 64)
    candidate[1]["technical_quality"].update(status="unscorable", score=None)
    summary, rows = report.build_report(cases, references, baseline, candidate)
    assert summary["same_frozen_view_pairs"] == 48
    assert summary["changed_view_case_ids"] == ["A01"]
    assert summary["unscorable_candidate_case_ids"] == ["A02"]
    assert rows[0]["runinfra_score"] == 3
    assert rows[0]["runinfra_distance"] is None
    assert rows[0]["interval_distance_improvement"] is None
    assert summary["incomplete_family_ids"] == ["controlled-0"]
    assert summary["complete_original_families_sensitivity"]["pairs"] == 46
    assert summary["primary_same_view_subset"]["pairs"] == 48


def test_family_bootstrap_preserves_cluster_dependence_and_partial_cluster_weights():
    cases, references, baseline, candidate = panel(old=3, new=3)
    for number, item in enumerate(baseline[:24]):
        item["technical_quality"]["score"] = 4 if number < 12 else 3
    for number, item in enumerate(candidate[:24]):
        item["technical_quality"]["score"] = 3 if number < 12 else 4
    _, rows = report.build_report(cases, references, baseline, candidate)
    interval = report.bootstrap_interval(rows)
    assert interval == pytest.approx([-0.32, 0.32])
    assert report.bootstrap_interval(list(reversed(rows))) == interval
    naive = (
        np.random.default_rng(report.BOOTSTRAP_SEED)
        .choice([row["interval_distance_improvement"] for row in rows], size=(10_000, 50))
        .mean(axis=1)
    )
    assert np.quantile(naive, 0.975) < interval[1]
    # Unequal retained family sizes must retain a constant per-image effect.
    _, constant = report.build_report(*panel())
    subset = [row for row in constant if row["case_id"] not in {"A01", "A02", "A07"}]
    assert report.bootstrap_interval(subset) == pytest.approx([0.5, 0.5])


def test_scores_and_integrity_reject_bad_joins_false_delivery_and_training_masks():
    cases, references, baseline, candidate = panel()
    candidate[0]["delivered_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contradicts"):
        report.build_report(cases, references, baseline, candidate)
    candidate[0]["delivered_sha256"] = cases[0]["view_sha256"]
    candidate[0]["task_masks"]["technical_quality"] = True
    with pytest.raises(ValueError, match="masks"):
        report.build_report(cases, references, baseline, candidate)
    candidate[0]["task_masks"]["technical_quality"] = False
    candidate[0]["image_id"] = "wrong-image"
    with pytest.raises(ValueError, match="identity"):
        report.build_report(cases, references, baseline, candidate)
    candidate[0]["image_id"] = cases[0]["image_id"]
    candidate[0]["technical_quality"]["score"] = 3.15
    with pytest.raises(ValueError, match="0.1"):
        report.build_report(cases, references, baseline, candidate)


def test_full_panel_gate_and_critical_boundaries():
    cases, references, baseline, candidate = panel()
    summary, _ = report.build_report(cases, references, baseline, candidate)
    assert summary["decision"] == "improved_against_assistant_references"
    references["A01"] = {"score": 1.5, "interval": [1, 2]}
    candidate[0]["technical_quality"]["score"] = 4
    summary, _ = report.build_report(cases, references, baseline, candidate)
    assert summary["primary_same_view_subset"]["new_critical_case_ids"] == ["A01"]
    assert summary["diagnostic_thresholds_on_observed_subset"]["no_new_critical_contradictions"] is False
    assert summary["decision"] != "improved_against_assistant_references"
    summary, _ = report.build_report(*panel(old=3, new=3))
    assert summary["decision"] == "inconclusive_bootstrap_interval_includes_zero"


def test_empty_or_duplicate_candidate_results():
    cases, references, baseline, candidate = panel()
    summary, rows = report.build_report(cases, references, baseline, [])
    assert summary["primary_same_view_subset"]["pairs"] == 0
    assert summary["primary_same_view_subset"]["paired_95_percent_family_bootstrap_interval"] is None
    assert len(rows) == 50
    with pytest.raises(ValueError, match="Duplicate"):
        report.build_report(cases, references, baseline, candidate + [candidate[0]])
