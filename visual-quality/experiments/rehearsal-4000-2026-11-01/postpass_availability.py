"""Explicit serial postpass availability override; original dispatch/loader stay intact."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_runner as runner
import postpass_recovery as recovery

POLICY = {'maximum_consecutive_failures': 5, 'rolling_window_requests': 100,
          'maximum_failures_in_window': 30, 'cumulative_minimum_responses': 100,
          'maximum_cumulative_failure_fraction': '0.20', 'history_scope': 'all_observed_jev_postpass_responses',
          'concurrency': 1, 'route_cap_usd': '15', 'cooldown_seconds': 120}


def helper_pins():
    return {**recovery.helper_pins(), str(Path(__file__).resolve()): contract.file_sha(__file__)}


def evidence(records, approval):
    rows = sorted((row for row in recovery.recovery_rows(records, approval, 'jev')
                   if row.get('response_received_at') and row.get('http_status') is not None),
                  key=lambda row: (row['response_received_at'], row['attempt_id']))
    failures = [recovery.known_failure(row) for row in rows]
    consecutive = 0
    for failure in reversed(failures):
        if not failure:
            break
        consecutive += 1
    return {'observed_postpass_responses': len(rows), 'known503': sum(failures),
            'last100_known503': sum(failures[-100:]), 'consecutive_known503': consecutive}


def availability_reason(observed):
    count = observed['observed_postpass_responses']
    if observed['consecutive_known503'] >= 5:
        return 'consecutive_known_provider_failures'
    if count >= 100 and observed['last100_known503'] >= 30:
        return 'rolling_postpass_known503_30_of_100'
    if count >= 100 and observed['known503'] * 5 >= count:
        return 'cumulative_postpass_known503_at_least_20_percent'
    return None


def stop_reason(original, records, approval, route, reserve):
    reason = original(records, approval, route, reserve)
    if route != 'jev' or reason not in {None, 'known_provider_failure_rate', 'consecutive_known_provider_failures'}:
        return reason
    return availability_reason(evidence(records, approval))


def receipt_prefix(records):
    return sorted((recovery.receipt(row) for row in records), key=lambda row: row['attempt_id'])


def read_amendment(path: Path, expected_sha: str, approval: dict):
    if contract.file_sha(path) != expected_sha:
        raise ValueError('Postpass availability amendment hash changed')
    doc = runner.read(path)
    if (doc['schema'] != 'jev-serial-postpass-availability-v1' or doc['effective_policy'] != POLICY
            or doc['allowlist_sha256'] != approval['_verified_file_sha256']
            or doc['allowlist_id'] != approval['allowlist_id'] or doc['original_policy_sha256'] != approval['policy_sha256']
            or doc['helper_sha256'] != helper_pins() or doc['allowances_changed'] is not False
            or doc['amendment_id'] != contract.digest({k: v for k, v in doc.items() if k != 'amendment_id'})):
        raise ValueError('Postpass availability scope, policy, source or original allowance changed')
    for key in ['stopped_segment', 'assessment']:
        if contract.file_sha(doc[key]['path']) != doc[key]['sha256']:
            raise ValueError('Postpass availability evidence changed')
    return doc


def execute(allowlist, allowlist_sha, amendment_path, amendment_sha):
    with recovery.exclusive_dispatch():
        recovery.service_snapshot()
        approval = recovery.read_allowlist(allowlist, allowlist_sha)
        doc = read_amendment(amendment_path, amendment_sha, approval)
        plans = recovery.load_plans()
        records = recovery.all_records(plans, approval)
        if receipt_prefix(records['jev']) != doc['receipt_prefix']:
            raise ValueError('Pre-amendment receipt prefix changed or new calls appeared')
        stopped = runner.read(Path(doc['stopped_segment']['path']))
        finished = datetime.fromisoformat(stopped['finished_at'])
        if finished.tzinfo is None or (datetime.now(UTC) - finished).total_seconds() < 120 or stopped['status'] != 'stopped_requires_audit':
            raise ValueError('Stopped serial segment and120-second cooldown required')
        folder = recovery.root() / 'availability-amendments' / doc['amendment_id']
        state_path = folder / 'segment.json'
        state = {'route': 'jev', 'status': 'running', 'started_at': runner.now(),
                 'amendment_path': str(amendment_path.resolve()), 'amendment_sha256': amendment_sha,
                 'amendment_id': doc['amendment_id'], 'driver_sha256': contract.file_sha(__file__),
                 'allowlist_sha256': allowlist_sha, 'effective_policy': POLICY,
                 'attempt_ids': [], 'new_receipts': [], 'effective_concurrency': 1}
        runner.write(state_path, state, exclusive=True)
        original_stop, original_send = recovery.stop_reason, runner.send
        reserve = Decimal(plans['jev']['rehearsal']['reserve_per_attempt_usd'])
        def overridden_stop(rows, allowed, route, bound):
            return stop_reason(original_stop, rows, allowed, route, bound)
        def guarded_send(plan, job, sequence, secret, admitted_at):
            if plan['route'] != 'jev':
                raise ValueError('Availability override is scoped only to Jev')
            reason = overridden_stop(records['jev'], approval, 'jev', reserve)
            if reason:
                raise RuntimeError(reason)
            if runner.liability(records['jev'], reserve) + reserve > Decimal(15):
                raise RuntimeError('Original full-reserve cap exhausted')
            identity = plan['phase'] + '/' + job['job_id'] + f'/{sequence:02d}'
            # This additional audit binds each admission to the later guards. The
            # original dispatcher has already written its unchanged admission.
            runner.write(folder / 'admissions' / identity / 'availability.json', {
                'amendment_id': doc['amendment_id'], 'amendment_sha256': amendment_sha,
                'adapter_sha256': contract.file_sha(__file__), 'allowlist_sha256': allowlist_sha,
                'admitted_at': runner.now(), 'request_sha256': job['request_sha256'],
                'attempt_id': identity, 'effective_policy': POLICY,
                'guard_evidence': evidence(records['jev'], approval),
                'liability_before_usd': str(runner.liability(records['jev'], reserve)),
                'reserved_usd': str(reserve)}, exclusive=True)
            state['attempt_ids'].append(identity)
            runner.write(state_path, state)
            return original_send(plan, job, sequence, secret, admitted_at)
        try:
            recovery.stop_reason, runner.send = overridden_stop, guarded_send
            result = recovery.dispatch('jev', plans, records, approval, allowlist, allowlist_sha, None)
            state.update(status='allowed_attempts_complete', original_dispatch_segment=result)
        except BaseException as error:
            state.update(status='stopped_requires_audit', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            recovery.stop_reason, runner.send = original_stop, original_send
            current = recovery.all_records(plans, approval)['jev']
            chosen = set(state['attempt_ids'])
            state['new_receipts'] = receipt_prefix([row for row in current if row['attempt_id'] in chosen])
            state['finished_at'] = runner.now()
            runner.write(state_path, state)
    return {'state_path': str(state_path), 'status': state['status'], 'new_attempts': len(state['attempt_ids'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--allowlist-sha256', required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--amendment-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.allowlist, args.allowlist_sha256, args.amendment, args.amendment_sha256)), flush=True)


if __name__ == '__main__':
    main()
