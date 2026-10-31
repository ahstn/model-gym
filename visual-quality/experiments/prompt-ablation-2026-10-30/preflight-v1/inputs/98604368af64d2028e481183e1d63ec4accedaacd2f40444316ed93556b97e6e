from copy import deepcopy

import pytest
from report import (
    ARMS,
    ROUTES,
    TASKS,
    build_report,
    comparison_rows,
    controlled_diagnostics,
    coverage,
    ensemble_diagnostics,
    metrics,
    paired_bootstrap,
    summarize,
)


def fixture(count=2):
    cases = [
        {
            "case_id": f"P{i:03}",
            "view_sha256": f"hash{i}",
            "family_id": f"f{i}",
            "stratum": "standalone_megalith",
            "role": "standalone_megalith",
            "issue_tags": ["framing"],
        }
        for i in range(count)
    ]
    refs = [
        {
            "case_id": c["case_id"],
            "view_sha256": c["view_sha256"],
            **{t: {"score": 3, "interval": [2.8, 3.2], "evidence": "visible"} for t in TASKS},
        }
        for c in cases
    ]
    obs = {
        route: [
            {
                "case_id": c["case_id"],
                "arm_id": arm,
                "view_sha256": c["view_sha256"],
                **{t: {"status": "ok", "score": 4 if arm == "baseline" else 3.2} for t in TASKS},
            }
            for c in cases
            for arm in ARMS
        ]
        for route in ROUTES
    }
    return cases, refs, obs


def pair(rows, task="technical_quality", arm="A", route="deepseek"):
    return [r for r in rows if r["task"] == task and r["arm_id"] == arm and r["route"] == route]


def test_direction_intervals_and_primary_interpretation():
    rows = comparison_rows(*fixture())
    result = paired_bootstrap(pair(rows), pair(rows, arm="baseline"), draws=100)
    assert result["mae_difference"]["estimate"] == pytest.approx(-0.8)
    assert result["interval_distance_difference"]["estimate"] == pytest.approx(-0.8)
    assert result["mae_difference"]["ci95"] == pytest.approx([-0.8, -0.8])
    report = summarize(rows, draws=30)
    changes = report["tasks"]["technical_quality"]["deepseek"]["arm_minus_baseline"]
    assert changes["A"]["observed_impact"] == "positive"
    assert changes["A"]["support_interval"] == "ci95"
    assert changes["combined"]["support_interval"] == "ci99_375"
    assert report["multiplicity"]["primary_combined_tests"] == 8
    assert report["multiplicity"]["exploratory_individual_tests"] == 32
    assert report["multiplicity"]["primary_quantiles"] == [0.003125, 0.996875]


def test_family_pairing_keeps_all_members_together():
    rows = comparison_rows(*fixture())
    a, b = pair(rows), pair(rows, arm="baseline")
    for members in (a, b):
        for row in members:
            row["family_id"] = "one-family"
    a[0]["absolute_error"], b[0]["absolute_error"] = 2, 0
    a[1]["absolute_error"], b[1]["absolute_error"] = 0, 2
    result = paired_bootstrap(a, list(reversed(b)), draws=100)
    assert result["families"] == 1
    assert result["mae_difference"]["estimate"] == 0
    assert result["mae_difference"]["ci95"] == [0, 0]
    assert result["status"] == "degenerate_family_resampling"
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(a, b[:1])
    b[0]["stratum"] = "other"
    with pytest.raises(ValueError, match="metadata"):
        paired_bootstrap(a, b)


def test_strata_are_resampled_independently_and_quantiles_are_nested():
    rows = comparison_rows(*fixture(3))
    a, b = pair(rows), pair(rows, arm="baseline")
    for group in (a, b):
        group[2]["stratum"] = "vizwiz"
    a[0]["absolute_error"], a[1]["absolute_error"], a[2]["absolute_error"] = 0, 3, 1
    result = paired_bootstrap(a, b, draws=1000)
    assert result["strata_family_counts"] == {"standalone_megalith": 2, "vizwiz": 1}
    wide, narrow = result["mae_difference"]["ci99_375"], result["mae_difference"]["ci95"]
    assert wide[0] <= narrow[0] <= narrow[1] <= wide[1]
    assert result == paired_bootstrap(list(reversed(a)), b, draws=1000)


def test_missingness_preserves_task_and_pair_specific_intersection():
    cases, refs, obs = fixture()
    bad = next(r for r in obs["deepseek"] if r["arm_id"] == "A")
    bad["technical_quality"] = {"status": "invalid", "score": None}
    report, rows = build_report({"cases": cases}, refs, obs, draws=30)
    technical = report["tasks"]["technical_quality"]["deepseek"]
    aesthetics = report["tasks"]["aesthetics"]["deepseek"]
    assert technical["arm_minus_baseline"]["A"]["n"] == 1
    assert aesthetics["arm_minus_baseline"]["A"]["n"] == 2
    assert technical["arm_minus_baseline"]["B"]["n"] == 2
    assert technical["arms"]["A"]["coverage"]["invalid"] == 1
    assert technical["arm_minus_baseline"]["A"]["coverage"]["excluded_baseline_only"] == ["P000"]
    assert technical["arm_minus_baseline"]["A"]["guardrails"]["invalid_increase"] == 1
    assert metrics(pair(rows, task="aesthetics"))["mae"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "change,message",
    [
        ("duplicate_case", "Duplicate manifest"),
        ("duplicate_ref", "Duplicate reference"),
        ("missing_ref", "coverage"),
        ("duplicate_obs", "Duplicate observation"),
        ("unknown_case", "Unknown observation"),
        ("unknown_arm", "Unknown observation"),
        ("changed_view", "view mismatch"),
        ("bad_interval", "reference interval"),
    ],
)
def test_bad_coverage_and_identity_rejected(change, message):
    cases, refs, obs = fixture()
    if change == "duplicate_case":
        cases.append(deepcopy(cases[0]))
    elif change == "duplicate_ref":
        refs.append(deepcopy(refs[0]))
    elif change == "missing_ref":
        refs.pop()
    elif change == "duplicate_obs":
        obs["haiku"].append(deepcopy(obs["haiku"][0]))
    elif change == "unknown_case":
        obs["haiku"][0]["case_id"] = "unknown"
    elif change == "unknown_arm":
        obs["haiku"][0]["arm_id"] = "unknown"
    elif change == "changed_view":
        obs["haiku"][0]["view_sha256"] = "changed"
    else:
        refs[0]["technical_quality"]["interval"] = [4, 5]
    with pytest.raises(ValueError, match=message):
        comparison_rows(cases, refs, obs)


def test_absent_route_and_degenerate_groups_do_not_claim_support():
    cases, refs, obs = fixture(1)
    obs["haiku"] = []
    result = summarize(comparison_rows(cases, refs, obs), draws=30)
    for task in TASKS:
        absent = result["tasks"][task]["haiku"]["arm_minus_baseline"]["combined"]
        assert absent["n"] == 0
        assert absent["support_status"] == "no_shared_valid_cases"
        present = result["tasks"][task]["luna"]["arm_minus_baseline"]["combined"]
        assert present["observed_impact"] == "positive"
        assert present["support_status"] == "insufficient_resampleable_families"
        assert (
            present["guardrails"]["subgroups"]["all"]["noninferiority_status"]
            == "insufficient_resampleable_families"
        )
    assert result["ensembles"]["all_four"]["tasks"]["aesthetics"]["arms"]["A"]["metrics"]["n"] == 0


def test_new_critical_contradictions_and_large_regressions():
    cases, refs, obs = fixture()
    for ref in refs:
        ref["technical_quality"] = {"score": 1.5, "interval": [1, 2]}
    for route in ROUTES:
        for row in obs[route]:
            row["technical_quality"]["score"] = 4 if row["arm_id"] == "A" else 1.5
    result = summarize(comparison_rows(cases, refs, obs), draws=30)
    change = result["tasks"]["technical_quality"]["deepseek"]["arm_minus_baseline"]["A"]
    assert change["observed_impact"] == "negative"
    assert change["guardrails"]["new_critical_contradictions"]["case_ids"] == ["P000", "P001"]
    assert change["guardrails"]["paired_error_worsened_at_least_0_5"]["count"] == 2
    assert "all" in change["guardrails"]["subgroup_regression_flags"]
    assert change["guardrails"]["subgroups"]["all"]["noninferiority_status"] == "not_established"
    assert change["guardrails"]["subgroups"]["untargeted_complement"]["n"] == 2
    assert "reference_low:technical_quality" in change["guardrails"]["subgroups"]


def test_ensemble_averages_axes_separately():
    cases, refs, obs = fixture()
    for route, value in zip(ROUTES, [1, 3, 5, 3]):
        for row in obs[route]:
            row["technical_quality"]["score"] = value
            row["aesthetics"]["score"] = 5
    result = ensemble_diagnostics(comparison_rows(cases, refs, obs), draws=30)
    ensemble = result["deepseek_luna"]["tasks"]
    assert ensemble["technical_quality"]["arms"]["baseline"]["metrics"]["mae"] == 0
    assert ensemble["aesthetics"]["arms"]["baseline"]["metrics"]["mae"] == 2


def test_controlled_ordering_and_parent_gap():
    cases, refs, obs = fixture(4)
    for i, case in enumerate(cases):
        case.update(
            family_id="family",
            stratum="controlled",
            role="source_parent" if i == 0 else "variant",
            operator="gaussian_blur",
            severity=i or None,
        )
    for route in ROUTES:
        for row in obs[route]:
            row["technical_quality"]["score"] = 5 - int(row["case_id"][1:])
    rows = comparison_rows(cases, refs, obs)
    result = controlled_diagnostics(pair(rows))
    assert result["complete_families"] == 1
    assert result["nonincreasing_families"] == 1
    assert result["mean_parent_to_severe_gap"] == 3
    assert result["families"][0]["operator"] == "blur"
    assert "degraded:blur" in pair(rows)[1]["subgroups"]


def test_invalid_numeric_scores_and_unscorable_references():
    cases, refs, obs = fixture()
    refs[0]["aesthetics"].update(score=None, interval=None)
    for row in obs["deepseek"]:
        row["technical_quality"]["score"] = float("nan")
    rows = comparison_rows(cases, refs, obs)
    assert coverage(pair(rows))["invalid"] == 2
    assert coverage(pair(rows, task="aesthetics"))["agreement_eligible"] == 1
    assert metrics([])["mae"] is None
