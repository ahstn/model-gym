"""Receipt-bound half-open circuit checks; synthetic data only."""
import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import api_recovery
import api_runner as runner
import jev_circuit_reset as reset
import pytest


def control():
    return {"control_id": "synthetic", "stopped_at": "2026-01-01T00:00:00+00:00",
        "prefix": {"plans": {"benchmark": "frozen"}, "attempt_receipts": [{"attempt_id": "old", "hash": "old-hash"}],
                   "observed_prefix": [{"attempt_id": "old", "known_image_503": True}],
                   "probe_jobs": [{"phase": "benchmark", "job_id": f"job{i}", "request_sha256": f"sha{i}"} for i in range(10)]},
        "helper_sha256": {str(Path(reset.__file__).resolve()): "source"}, "probe_approval": {"sha256": "approval"}}


@pytest.fixture(autouse=True)
def no_calls(monkeypatch):
    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: pytest.fail("Live calls forbidden"))


@pytest.mark.parametrize("seconds,allowed", [(119.9, False), (120, True), (121, True)])
def test_cooldown_is_at_least_120_seconds(seconds, allowed):
    now = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    if allowed:
        reset.cooldown(control(), now)
    else:
        with pytest.raises(ValueError, match="120 seconds"):
            reset.cooldown(control(), now)


def test_prefix_receipt_order_hash_and_unowned_calls_are_rejected():
    approved = control()
    current = copy.deepcopy(approved["prefix"])
    reset.validate_history(approved, current)
    for field in ("attempt_receipts", "observed_prefix", "plans"):
        changed = copy.deepcopy(current)
        changed[field] = []
        with pytest.raises(ValueError, match="prefix changed|receipt order"):
            reset.validate_history(approved, changed)
    current["attempt_receipts"].append({"attempt_id": "unowned", "hash": "x"})
    with pytest.raises(ValueError, match="unowned"):
        reset.validate_history(approved, current)


def test_probe_stops_on_second_exact503_and_validates_prefix_inside_lock():
    validations = []
    breaker = reset.ResetBreaker(control(), "probe", api_recovery.circuit_breaker, lambda: validations.append(True))
    assert breaker([True], reset.POLICY) is None
    assert breaker([True, False, True, False], reset.POLICY) is None
    assert breaker([True, False, True, False, True], reset.POLICY) == "half_open_second_known_503"
    assert validations == [True]
    with pytest.raises(ValueError, match="prefix changed"):
        breaker([False], reset.POLICY)
    with pytest.raises(ValueError, match="prefix changed"):
        breaker([], reset.POLICY)
    with pytest.raises(ValueError, match="ten-request"):
        breaker([True] + [False] * 11, reset.POLICY)


def test_continuation_keeps_probe_events_and_original_five_twenty_breakers():
    breaker = reset.ResetBreaker(control(), "continue", api_recovery.circuit_breaker, lambda: None)
    assert breaker([True] + [False] * 10, reset.POLICY) is None
    assert breaker([True] + [False] * 10 + [True] * 5, reset.POLICY) == "consecutive_known_provider_failures"
    assert breaker([True] + [True, False, False, False, False] * 20, reset.POLICY) == "known_provider_failure_rate"
    with pytest.raises(ValueError, match="policy"):
        breaker([True], {**reset.POLICY, "maximum_failures_in_window": 21})


def observations(count=10, failures=0):
    return [{"attempt_id": f"new{i}", "status": "failed" if i < failures else "valid", "known": i < failures,
             "price_usd": None if i < failures else ".001", "http_status": 503 if i < failures else 200,
             "response_received_at": "2026-01-01T00:10:00+00:00"} for i in range(count)]


@pytest.mark.parametrize("count,failures,success", [(10, 0, True), (10, 1, True), (10, 2, False), (9, 0, False), (11, 0, False)])
def test_exact_ten_observed_with_at_most_one_known503_passes(monkeypatch, count, failures, success):
    monkeypatch.setattr(api_recovery, "known_image_failure", lambda row: row["known"])
    assert reset.probe_success(observations(count, failures)) is success


@pytest.mark.parametrize("change", ["unknown_bill", "missing_response", "other_failure", "overreserve", "duplicate"])
def test_probe_unknown_or_new_error_is_not_success(monkeypatch, change):
    monkeypatch.setattr(api_recovery, "known_image_failure", lambda row: row["known"])
    rows = observations()
    if change == "unknown_bill":
        rows[0]["price_usd"] = None
    elif change == "missing_response":
        rows[0]["response_received_at"] = None
    elif change == "other_failure":
        rows[0]["status"] = "failed"
    elif change == "overreserve":
        rows[0]["price_usd"] = "2"
    else:
        rows[0]["attempt_id"] = rows[1]["attempt_id"]
    assert not reset.probe_success(rows)


@pytest.mark.parametrize("status,count", [("probe_failed", 10), ("running", 10), ("probe_succeeded", 9)])
def test_continuation_blocks_failed_or_partial_probe(tmp_path, monkeypatch, status, count):
    monkeypatch.setattr(runner, "DATA", tmp_path)
    approved = control()
    path = reset.control_directory(approved) / "probe.json"
    runner.write(path, {"status": status, "control_id": "synthetic", "attempt_ids": list(range(count)),
                        "new_receipts": list(range(count)), "effective_concurrency": 1, "known_image_503_count": 0})
    with pytest.raises(ValueError, match="successful complete"):
        reset.continuation_receipts(approved)


@pytest.mark.parametrize("stage,expected_limit", [("probe", ["--limit", "10"]), ("continue", [])])
def test_runtime_adaptation_restores_phase_policy_main_send_and_argv_on_error(tmp_path, monkeypatch, stage, expected_limit):
    import sys
    previous = (runner.PHASES, runner.send, runner.fatal, api_recovery.circuit_breaker)
    observed = []
    def original_main():
        observed.append(list(sys.argv))
        assert runner.PHASES == ("benchmark", "rehearsal")
        raise RuntimeError("synthetic stop")
    monkeypatch.setattr(api_recovery, "main", original_main)
    monkeypatch.setattr(sys, "argv", ["api_recovery.py", "--concurrency", "1"])
    original_argv = sys.argv
    with (
        pytest.raises(RuntimeError, match="synthetic stop"),
        reset.adapted_execution(control(), stage, {"attempt_ids": []}, tmp_path / "state.json", []),
    ):
        api_recovery.main()
    assert observed == [["api_recovery.py", "--concurrency", "1", *expected_limit]]
    assert sys.argv is original_argv
    assert (runner.PHASES, runner.send, runner.fatal, api_recovery.circuit_breaker) == previous
    assert api_recovery.main is original_main


def test_probe_never_attempted_order_is_enforced_before_send(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(runner, "send", lambda *args: sent.append(args))
    state = {"attempt_ids": []}
    with reset.adapted_execution(control(), "probe", state, tmp_path / "state.json", []):
        runner.send({"phase": "benchmark"}, {"job_id": "job0", "request_sha256": "sha0"}, 1, "secret", 0)
        with pytest.raises(ValueError, match="replay"):
            runner.send({"phase": "benchmark"}, {"job_id": "job0", "request_sha256": "sha0"}, 1, "secret", 0)
        with pytest.raises(ValueError, match="frozen unattempted job order"):
            runner.send({"phase": "benchmark"}, {"job_id": "job2", "request_sha256": "sha2"}, 1, "secret", 0)
    assert len(sent) == 1


def test_maximum_three_failed_halfopens_stops_before_any_reads(tmp_path):
    for number in range(3):
        runner.write(tmp_path / f"failure{number}.json", {"status": "probe_failed"})
    with pytest.raises(ValueError, match="Three consecutive"):
        reset.freeze_control(None, None, None, None, tmp_path / "control.json",
                             [tmp_path / f"failure{i}.json" for i in range(3)])
