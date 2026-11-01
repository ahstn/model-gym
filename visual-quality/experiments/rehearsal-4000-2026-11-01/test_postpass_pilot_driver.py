from contextlib import nullcontext

import postpass_pilot_driver as pilot
import pytest


@pytest.mark.parametrize('failure_route', [None, 'luna', 'liquid'])
def test_order_single_validation_and_stop_on_failure(tmp_path, monkeypatch, failure_route):
    calls = []
    monkeypatch.setattr(pilot.recovery, 'root', lambda: tmp_path)
    monkeypatch.setattr(pilot.recovery, 'exclusive_dispatch', nullcontext)
    monkeypatch.setattr(pilot.recovery, 'service_snapshot', lambda: calls.append('services'))
    monkeypatch.setattr(pilot.recovery, 'read_allowlist', lambda *args: calls.append('allowlist') or {})
    monkeypatch.setattr(pilot.recovery, 'load_plans', lambda: calls.append('plans') or {})
    monkeypatch.setattr(pilot.recovery, 'all_records', lambda *args: calls.append('records') or {})
    def dispatch(route, plans, records, approval, path, sha, limit):
        calls.append((route, limit))
        if route == failure_route:
            raise RuntimeError('review required')
        if route == 'jev' and limit == 50:
            pilot.runner.write(tmp_path / 'jev/report.json', {'synthetic_report': True})
        return {'status': 'complete'}
    monkeypatch.setattr(pilot.recovery, 'dispatch', dispatch)
    if failure_route:
        with pytest.raises(RuntimeError, match='review required'):
            pilot.execute(tmp_path / 'allowlist.json', 'fixed-sha')
    else:
        assert pilot.execute(tmp_path / 'allowlist.json', 'fixed-sha')['status'] == 'serial_recovery_complete'
    expected = ['services', 'allowlist', 'plans', 'records', ('luna', None)]
    if failure_route != 'luna':
        expected.append(('liquid', None))
    if failure_route is None:
        expected.extend([('jev', 50), ('jev', None)])
    assert calls == expected
    with pytest.raises(ValueError, match='already exists'):
        pilot.execute(tmp_path / 'allowlist.json', 'fixed-sha')
