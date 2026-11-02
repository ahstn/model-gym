"""Operational sampling must retain slow requests rather than favor fast successes."""

import api_jev_concurrency_audit as comparison


def test_first_started_cohort_waits_for_its_slow_case(tmp_path, monkeypatch):
    runner = comparison.runner
    monkeypatch.setattr(runner, "DATA", tmp_path)
    runner.atomic(
        tmp_path / "api-operational-amendment-04.json",
        {
            "initial_case_ids_with_requests": [],
            "comparison_case_count": 2,
            "baseline": {"http_503_share": 0.0},
        },
    )
    root = tmp_path / "results/api/jev"
    for number in range(1, 4):
        case_id = f"R{number:04d}"
        runner.atomic(
            root / "attempts" / case_id / "01/request.json",
            {"case_id": case_id, "created_at": f"2026-11-01T00:00:0{number}+00:00"},
        )
        if number != 2:
            runner.atomic(
                root / "observations" / (case_id + ".json"),
                {
                    "case_id": case_id,
                    "status": "valid",
                    "response_received_at": f"2026-11-01T00:00:1{number}+00:00",
                },
            )
            runner.atomic(root / "attempts" / case_id / "01/result.json", {"http_status": 200})
    waiting = comparison.audit()
    assert not waiting["complete"]
    assert waiting["case_ids"] == ["R0001"]
    runner.atomic(
        root / "observations/R0002.json",
        {
            "case_id": "R0002",
            "status": "retryable_failure",
            "response_received_at": "2026-11-01T00:00:30+00:00",
        },
    )
    runner.atomic(root / "attempts/R0002/01/result.json", {"http_status": 503})
    complete = comparison.audit()
    assert complete["complete"]
    assert complete["case_ids"] == ["R0001", "R0002"]
    assert complete["terminal_failure_rate"] == 0.5
    assert complete["rollback_required"]
