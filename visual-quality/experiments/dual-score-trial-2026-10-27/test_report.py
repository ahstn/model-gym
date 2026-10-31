from copy import deepcopy

import pytest
from report import ROUTES, TASKS, comparison_rows, metrics, paired_bootstrap, summarize


def fixture():
    cases = [
        {
            "case_id": "D001",
            "view_sha256": "hash",
            "family_id": "f",
            "stratum": "standalone",
            "role": "standalone",
        }
    ]
    refs = [
        {
            "case_id": "D001",
            "view_sha256": "hash",
            **{t: {"score": 3, "interval": [2.8, 3.2], "evidence": "visible"} for t in TASKS},
        }
    ]
    observations = {
        route: [
            {"case_id": "D001", "view_sha256": "hash", **{t: {"status": "ok", "score": score} for t in TASKS}}
        ]
        for route, score in zip(ROUTES, [4.2, 3.1, 2.8])
    }
    return cases, refs, observations


def test_known_distances_and_bias():
    rows = comparison_rows(*fixture())
    stats = metrics([r for r in rows if r["route"] == "deepseek"])
    assert stats["mae"] == pytest.approx(1.2)
    assert stats["bias"] == pytest.approx(1.2)
    assert stats["mean_interval_distance"] == pytest.approx(1.0)
    assert stats["large_interval_errors"] == 2
    assert all(r["matched"] for r in rows)


def test_task_specific_missingness_and_shared_intersection():
    cases, refs, obs = fixture()
    obs["luna"][0]["technical_quality"] = {"status": "invalid", "score": None}
    rows = comparison_rows(cases, refs, obs)
    assert not any(r["matched"] for r in rows if r["task"] == "technical_quality")
    assert all(r["matched"] for r in rows if r["task"] == "aesthetics")
    assert len([r for r in rows if "error" in r]) == 5


def test_bootstrap_preserves_family_and_pairing():
    rows = comparison_rows(*fixture())
    a = [r for r in rows if r["route"] == "deepseek" and r["task"] == "technical_quality"]
    b = [r for r in rows if r["route"] == "gemini" and r["task"] == "technical_quality"]
    a2, b2 = deepcopy(a[0]), deepcopy(b[0])
    a2["case_id"] = b2["case_id"] = "D002"
    a2["absolute_error"], b2["absolute_error"] = 0, 0
    result = paired_bootstrap([*a, a2], [*b, b2], draws=100)
    assert result["families"] == 1
    assert result["mae_difference"]["estimate"] == pytest.approx(0.55)
    assert result["mae_difference"]["ci95"] == pytest.approx([0.55, 0.55])
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(a, [*b, b2])


def test_changed_view_and_duplicate_case_rejected():
    cases, refs, obs = fixture()
    obs["gemini"][0]["view_sha256"] = "different"
    with pytest.raises(ValueError, match="view mismatch"):
        comparison_rows(cases, refs, obs)
    cases, refs, obs = fixture()
    obs["luna"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        comparison_rows(cases, refs, obs)


def test_absent_route_does_not_create_arbitrary_ranking():
    cases, refs, obs = fixture()
    obs["luna"] = []
    results = summarize(comparison_rows(cases, refs, obs), draws=100)
    for task in TASKS:
        assert results["tasks"][task]["ranking_by_matched_mae"] == []
        assert results["tasks"][task]["comparison_status"] == "no_common_scorable_cases"
        assert results["tasks"][task]["available"]["deepseek"]["n"] == 1
