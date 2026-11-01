"""Guard repeat-panel selection and agreement denominators with synthetic labels."""

from collections import Counter

import pytest
import reference_reliability as reliability


def test_repeat_selection_is_score_blind_order_independent_and_stratified():
    cases = [
        {"case_id": f"{partition}-{index}", "benchmark_partition": partition}
        for partition in (*reliability.QUOTAS, "rehearsal")
        for index in range(100)
    ]
    selected = reliability.select(cases)
    assert selected == reliability.select(list(reversed(cases)))
    assert Counter(case["benchmark_partition"] for case in selected) == reliability.QUOTAS
    assert len({case["case_id"] for case in selected}) == 50


def test_repeat_selection_refuses_an_incomplete_partition():
    with pytest.raises(ValueError, match="Insufficient"):
        reliability.select([])


def test_task_specific_pairs_and_scope_disagreements_remain_separate():
    def row(case_id, eligibility, technical, aesthetic):
        return {
            "case_id": case_id,
            "eligibility": eligibility,
            **{
                task: {"score": score, "status": "ok" if score is not None else "unscorable"}
                for task, score in zip(reliability.review.TASKS, (technical, aesthetic))
            },
        }

    primary = [row("one", "eligible", 3.0, 4.0), row("two", "excluded", 1.0, 1.0)]
    repeated = [row("one", "eligible", 2.0, None), row("two", "uncertain", 2.0, 1.5)]
    cases = [
        {"case_id": "one", "benchmark_partition": "representative"},
        {"case_id": "two", "benchmark_partition": "challenge"},
    ]
    panels = reliability.summarize(primary, repeated, cases)["panels"]
    all_rows = panels["all"]
    assert all_rows["n"] == 2
    assert all_rows["eligibility_agreements"] == 1
    assert all_rows["floor_decision_disagreements"] == 1
    assert all_rows["technical_quality"]["paired_n"] == 2
    assert all_rows["technical_quality"]["mean_absolute_difference"] == 1.0
    assert all_rows["technical_quality"]["repeat_minus_primary_bias"] == 0.0
    assert all_rows["technical_quality"]["differences_at_least_0_75"] == 2
    assert all_rows["aesthetics"]["paired_n"] == 1
    assert all_rows["aesthetics"]["mean_absolute_difference"] == 0.5
    assert panels["development"]["technical_quality"]["mean_absolute_difference"] is None
