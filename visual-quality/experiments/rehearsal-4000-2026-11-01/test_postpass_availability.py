import copy
from decimal import Decimal

import postpass_availability as availability
import pytest


@pytest.mark.parametrize(('observed', 'expected'), [
    ({'observed_postpass_responses': 236, 'known503': 40, 'last100_known503': 20, 'consecutive_known503': 1}, None),
    ({'observed_postpass_responses': 100, 'known503': 20, 'last100_known503': 20, 'consecutive_known503': 1}, 'cumulative_postpass_known503_at_least_20_percent'),
    ({'observed_postpass_responses': 1000, 'known503': 150, 'last100_known503': 30, 'consecutive_known503': 1}, 'rolling_postpass_known503_30_of_100'),
    ({'observed_postpass_responses': 1000, 'known503': 100, 'last100_known503': 5, 'consecutive_known503': 5}, 'consecutive_known_provider_failures'),
    ({'observed_postpass_responses': 99, 'known503': 20, 'last100_known503': 20, 'consecutive_known503': 1}, None),
])
def test_bounded_thresholds(observed, expected):
    original = copy.deepcopy(observed)
    assert availability.availability_reason(observed) == expected
    assert observed == original


@pytest.mark.parametrize(('route', 'old_reason', 'expected'), [
    ('jev', 'known_provider_failure_rate', None),
    ('jev', 'new_unreviewed_error_or_unknown_bill', 'new_unreviewed_error_or_unknown_bill'),
    ('luna', 'known_provider_failure_rate', 'known_provider_failure_rate'),
])
def test_only_known_jev_availability_is_overridden(monkeypatch, route, old_reason, expected):
    monkeypatch.setattr(availability, 'evidence', lambda *args: {'observed_postpass_responses': 236, 'known503': 40, 'last100_known503': 20, 'consecutive_known503': 1})
    assert availability.stop_reason(lambda *args: old_reason, [], {}, route, Decimal('0.01')) == expected


def test_evidence_excludes_initial_and_unobserved_attempts(monkeypatch):
    rows = [{'attempt_id': '1', 'response_received_at': '2026-11-01T12:00:00+00:00', 'http_status': 503},
            {'attempt_id': '2', 'response_received_at': '2026-11-01T12:00:01+00:00', 'http_status': 200},
            {'attempt_id': '3', 'response_received_at': None, 'http_status': None}]
    monkeypatch.setattr(availability.recovery, 'recovery_rows', lambda *args: rows)
    monkeypatch.setattr(availability.recovery, 'known_failure', lambda row: row['http_status'] == 503)
    assert availability.evidence([], {}) == {'observed_postpass_responses': 2, 'known503': 1, 'last100_known503': 1, 'consecutive_known503': 0}


def fixture_amendment(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from datetime import UTC, datetime, timedelta
    r, c, recovery = availability.runner, availability.contract, availability.recovery
    monkeypatch.setattr(recovery, 'root', lambda: tmp_path / 'postpass')
    monkeypatch.setattr(recovery, 'exclusive_dispatch', nullcontext)
    monkeypatch.setattr(recovery, 'service_snapshot', lambda: None)
    approved = {'_verified_file_sha256': 'allow-sha', 'allowlist_id': 'original', 'policy_sha256': 'policy-sha'}
    monkeypatch.setattr(recovery, 'read_allowlist', lambda *args: approved)
    monkeypatch.setattr(recovery, 'load_plans', lambda: {'jev': {'rehearsal': {'reserve_per_attempt_usd': '.011010048'}}})
    records = {'jev': []}
    monkeypatch.setattr(recovery, 'all_records', lambda *args: records)
    monkeypatch.setattr(availability, 'receipt_prefix', lambda rows: copy.deepcopy(rows))
    files = {}
    for name in ['stopped_segment', 'assessment']:
        path = tmp_path / (name + '.json')
        r.write(path, {'status': 'stopped_requires_audit', 'finished_at': (datetime.now(UTC) - timedelta(minutes=3)).isoformat()})
        files[name] = {'path': str(path), 'sha256': c.file_sha(path)}
    doc = {**files, 'schema': 'jev-serial-postpass-availability-v1', 'effective_policy': availability.POLICY,
           'allowlist_sha256': 'allow-sha', 'allowlist_id': 'original', 'original_policy_sha256': 'policy-sha',
           'helper_sha256': availability.helper_pins(), 'allowances_changed': False, 'receipt_prefix': []}
    doc['amendment_id'] = c.digest(doc)
    path = tmp_path / 'amendment.json';r.write(path, doc)
    return path, c.file_sha(path), doc, approved, records


@pytest.mark.parametrize('mutation', ['hash', 'source', 'policy', 'allowance', 'allowlist', 'receipt_evidence'])
def test_amendment_provenance_is_bound(tmp_path, monkeypatch, mutation):
    path, sha, doc, approved, _ = fixture_amendment(tmp_path, monkeypatch)
    if mutation == 'hash':
        sha = 'bad'
    elif mutation == 'source':
        doc['helper_sha256'] = {}
    elif mutation == 'policy':
        doc['effective_policy'] = {**availability.POLICY, 'concurrency': 4}
    elif mutation == 'allowance':
        doc['allowances_changed'] = True
    elif mutation == 'allowlist':
        doc['allowlist_sha256'] = 'different'
    else:
        (tmp_path / 'assessment.json').write_text('{}')
    if mutation not in ['hash', 'receipt_evidence']:
        doc['amendment_id'] = availability.contract.digest({k: v for k, v in doc.items() if k != 'amendment_id'})
        availability.runner.write(path, doc);sha = availability.contract.file_sha(path)
    with pytest.raises(ValueError):
        availability.read_amendment(path, sha, approved)


@pytest.mark.parametrize('failure', [None, 'dispatch', 'prefix', 'budget'])
def test_original_dispatch_and_loader_preserved_with_per_admission_guard(tmp_path, monkeypatch, failure):
    path, sha, _, _, records = fixture_amendment(tmp_path, monkeypatch)
    if failure == 'prefix':
        records['jev'].append({'attempt_id': 'unexpected'})
    evidence = {'observed_postpass_responses': 236, 'known503': 40, 'last100_known503': 20, 'consecutive_known503': 1}
    monkeypatch.setattr(availability, 'evidence', lambda *args: evidence)
    monkeypatch.setattr(availability.runner, 'liability', lambda *args: Decimal(15) if failure == 'budget' else Decimal(8))
    old_stop = lambda *args: 'known_provider_failure_rate'
    sent = []
    def old_send(plan, job, sequence, secret, admitted_at):
        admissions = list((tmp_path / 'postpass/availability-amendments').glob('*/admissions/**/availability.json'))
        assert len(admissions) == 1
        meta = availability.runner.read(admissions[0])
        assert meta['guard_evidence'] == evidence and meta['effective_policy']['concurrency'] == 1
        sent.append((job['job_id'], sequence))
        return {'attempt_id': 'rehearsal/baseline--R9999/02'}
    monkeypatch.setattr(availability.recovery, 'stop_reason', old_stop)
    monkeypatch.setattr(availability.runner, 'send', old_send)
    def dispatch(route, plans, current, approval, allowlist, digest, limit):
        assert route == 'jev' and limit is None
        row = availability.runner.send({'route': 'jev', 'phase': 'rehearsal'}, {'job_id': 'baseline--R9999', 'request_sha256': 'payload'}, 2, 'redacted', 0)
        current['jev'].append(row)
        if failure == 'dispatch':
            raise RuntimeError('synthetic stop')
        return {'status': 'allowed_attempts_complete'}
    monkeypatch.setattr(availability.recovery, 'dispatch', dispatch)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            availability.execute(tmp_path / 'allowlist.json', 'allow-sha', path, sha)
    else:
        result = availability.execute(tmp_path / 'allowlist.json', 'allow-sha', path, sha)
        assert result['new_attempts'] == 1 and result['status'] == 'allowed_attempts_complete'
    assert availability.runner.send is old_send and availability.recovery.stop_reason is old_stop
    assert len(sent) == (0 if failure in ['prefix', 'budget'] else 1)
