"""Availability thresholds do not alter scoring or retry eligibility."""
import copy

import jev_availability as availability
import pytest


@pytest.mark.parametrize(("failures", "expected"), [
    ([False] * 80 + [True, False] * 20, None),
    ([False] * 195 + [True] * 5, "consecutive_known_provider_failures"),
    ([False] * 400 + [True, False] * 30 + [False] * 40, "rolling_known_provider_failures_30_of_100"),
    ([True, False, False, False, False] * 20, "cumulative_known_provider_failures_at_least_20_percent"),
    ([True, False, False, False, False] * 19 + [False] * 5, None),
    ([True, False] * 49, None),
])
def test_bounded_full_history_thresholds(failures, expected):
    original = copy.deepcopy(failures)
    assert availability.breaker(failures, availability.COMPATIBILITY_POLICY) == expected
    assert failures == original


def test_cumulative_guard_uses_full_history_not_only_recent_window():
    # No five consecutive failures, recent100 completely healthy, lifetime20%.
    history = [True, False] * 100 + [False] * 300
    assert availability.breaker(history, availability.COMPATIBILITY_POLICY) == "cumulative_known_provider_failures_at_least_20_percent"


def test_compatibility_policy_drift_is_rejected():
    with pytest.raises(ValueError, match="compatibility"):
        availability.breaker([False] * 200, {**availability.COMPATIBILITY_POLICY, "maximum_failures_in_window": 30})


def fixture_approval(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta
    runner, contract = availability.runner, availability.contract
    files = {}
    for name in ['previous_stopped_segment', 'interruption_audit', 'evidence']:
        path = tmp_path / (name + '.json')
        value = ({'status': 'stopped_requires_audit', 'route': 'jev',
                  'finished_at': (datetime.now(UTC) - timedelta(minutes=3)).isoformat()}
                 if name == 'previous_stopped_segment' else {'audit': True})
        runner.write(path, value)
        files[name] = {'path': str(path), 'sha256': contract.file_sha(path)}
    prefix = {'receipts': [], 'plans': {}}
    monkeypatch.setattr(availability, 'snapshot', lambda: copy.deepcopy(prefix))
    monkeypatch.setattr(runner, 'directory', lambda *args: tmp_path)
    doc = {**files, 'availability_schema': 'jev-first-pass-availability-v1',
           'scope': 'remaining_initial_jobs_only', 'effective_availability_policy': availability.POLICY,
           'helper_sha256': availability.helper_pins(), 'effective_concurrency': 4,
           'replay_completed_attempts': False, 'postpass_policy_changed': False, 'initial_prefix': prefix}
    doc['approval_id'] = contract.digest(doc)
    path = tmp_path / 'approval.json'
    runner.write(path, doc)
    return path, contract.file_sha(path), doc


@pytest.mark.parametrize('mutation', ['hash', 'scope', 'policy', 'helper', 'evidence', 'prefix', 'cooldown'])
def test_approval_and_receipt_binding(tmp_path, monkeypatch, mutation):
    from datetime import UTC, datetime
    path, sha, doc = fixture_approval(tmp_path, monkeypatch)
    if mutation == 'hash':
        sha = '0' * 64
    elif mutation == 'scope':
        doc['scope'] = 'postpass'
    elif mutation == 'policy':
        doc['effective_availability_policy'] = {**availability.POLICY, 'maximum_consecutive_failures': 6}
    elif mutation == 'helper':
        doc['helper_sha256'] = {}
    elif mutation == 'evidence':
        evidence = tmp_path / 'evidence.json'
        evidence.write_text('{}')
    elif mutation == 'prefix':
        monkeypatch.setattr(availability, 'snapshot', lambda: {'changed': True})
    else:
        previous = tmp_path / 'previous_stopped_segment.json'
        availability.runner.write(previous, {'route': 'jev', 'status': 'stopped_requires_audit', 'finished_at': datetime.now(UTC).isoformat()})
        doc['previous_stopped_segment']['sha256'] = availability.contract.file_sha(previous)
    if mutation not in ['hash', 'evidence', 'prefix']:
        doc['approval_id'] = availability.contract.digest({k: v for k, v in doc.items() if k != 'approval_id'})
        availability.runner.write(path, doc)
        sha = availability.contract.file_sha(path)
    with pytest.raises(ValueError):
        loaded = availability.read_approval(path, sha)
        availability.validate_start(loaded)


@pytest.mark.parametrize('fails', [False, True])
def test_execution_restores_original_helpers_without_replay(tmp_path, monkeypatch, fails):
    path, sha, _ = fixture_approval(tmp_path, monkeypatch)
    original = availability.api_recovery.circuit_breaker
    original_phases = availability.runner.PHASES
    def fake_execution():
        assert availability.runner.PHASES == ('benchmark', 'rehearsal')
        assert availability.api_recovery.circuit_breaker([False] * 100, availability.COMPATIBILITY_POLICY) is None
        if fails:
            raise RuntimeError('synthetic stop')
    monkeypatch.setattr(availability.interruption_resume, 'main', fake_execution)
    if fails:
        with pytest.raises(RuntimeError, match='synthetic stop'):
            availability.execute(path, sha)
    else:
        result = availability.execute(path, sha)
        assert result['status'] == 'all_initial_jobs_attempted' and result['new_attempts'] == 0
    assert availability.api_recovery.circuit_breaker is original
    assert availability.runner.PHASES == original_phases
    with pytest.raises(FileExistsError):
        availability.execute(path, sha)
