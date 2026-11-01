"""Run authorized serial recovery, preserving a fifty-attempt Jev pilot checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import api_contract as contract
import api_runner as runner
import postpass_recovery as recovery


def execute(allowlist: Path, expected_sha: str):
    status_path = recovery.root() / 'pilot-driver-status.json'
    status = {'started_at': runner.now(), 'status': 'running', 'driver_sha256': contract.file_sha(__file__),
              'allowlist_path': str(allowlist.resolve()), 'allowlist_sha256': expected_sha,
              'route_sequence': ['luna', 'liquid', 'jev_pilot', 'jev_remaining'], 'jev_new_attempt_limit': 50, 'phases': {}}
    if status_path.exists():
        raise ValueError('Pilot orchestrator already exists; inspect receipts and resume through explicit route commands')
    runner.write(status_path, status, exclusive=True)
    try:
        with recovery.exclusive_dispatch():
            recovery.service_snapshot()
            approval = recovery.read_allowlist(allowlist, expected_sha)
            plans = recovery.load_plans()
            records = recovery.all_records(plans, approval)
            for stage, route, limit in [('luna', 'luna', None), ('liquid', 'liquid', None), ('jev_pilot', 'jev', 50), ('jev_remaining', 'jev', None)]:
                status['active_route'] = route
                status['active_stage'] = stage
                runner.write(status_path, status)
                result = recovery.dispatch(route, plans, records, approval, allowlist, expected_sha, limit)
                status['phases'][stage] = result
                runner.write(status_path, status)
                if stage == 'jev_pilot':
                    runner.write(recovery.root() / 'jev-serial-pilot.json', {'created_at': runner.now(), 'segment': result,
                                 'allowlist_sha256': expected_sha, 'report': runner.read(recovery.root() / 'jev/report.json')}, exclusive=True)
                    print(json.dumps({'pilot_checkpoint': str(recovery.root() / 'jev-serial-pilot.json')}), flush=True)
        status['status'] = 'serial_recovery_complete'
    except BaseException as error:
        status.update(status='stopped_requires_audit', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status['finished_at'] = runner.now()
        runner.write(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--allowlist-sha256', required=True)
    args = parser.parse_args()
    result = execute(args.allowlist, args.allowlist_sha256)
    print(json.dumps({'status': result['status'], 'phases': result['phases']}), flush=True)


if __name__ == '__main__':
    main()
