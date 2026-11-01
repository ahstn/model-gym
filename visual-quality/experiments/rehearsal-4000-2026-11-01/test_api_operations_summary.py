"""Operational reporting preserves failed latencies, unknown bills and segment identity."""

import copy
import json
from pathlib import Path

import api_operations_summary as ops
import pytest


@pytest.fixture(autouse=True)
def forbid_external_calls(monkeypatch):
    monkeypatch.setattr(ops.runner.requests, "post", lambda *a, **k: pytest.fail("Operational audit tests must not make HTTP calls"))


def rows():
    return [{"attempt_id": f"rehearsal/baseline--R{i:04}/01", "model": "jev", "phase": "rehearsal",
             "status": "valid" if i < 2 else "failed", "http_status": 200 if i < 2 else 503,
             "price_usd": "0.001" if i < 2 else None, "latency_seconds": float(i + 1),
             "request_started_at": f"2026-11-01T14:00:0{i}+00:00", "raw_response_sha256": f"raw-{i}"}
            for i in range(3)]


def segment():
    return (Path("serial.json"), {"status": "all_jobs_attempted_with_reviewed_missing_results",
            "started_at": "2026-11-01T14:00:01+00:00", "finished_at": "2026-11-01T14:00:04+00:00",
            "effective_concurrency": 1, "attempt_ids": [rows()[2]["attempt_id"]],
            "quarantined_attempt_ids": [rows()[2]["attempt_id"]]})


def test_smoke_initial_and_serial_segments_preserve_unknown_costs(monkeypatch):
    original = rows()
    untouched = copy.deepcopy(original)
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: row["status"] == "failed")
    bound = ops.bind_segments(original, [segment()], {"raw_response_sha256": "raw-0"}, 4)
    assert original == untouched
    assert [row["effective_concurrency_limit"] for row in bound] == [1, 4, 1]
    assert [row["operational_segment_id"] for row in bound] == ["initial_single_request_smoke", "original_frozen_plan", "serial"]
    summary = ops.summarize(bound)
    failed = next(row for row in summary if row["operational_segment_id"] == "serial")
    assert failed["total_price_usd"] is None and failed["known_price_usd"] == "0"
    assert failed["unpriced_attempts"] == 1 and failed["known_image_processing_503"] == 1
    assert failed["all_attempt_latency"]["p50_seconds"] == 3
    assert failed["completed_response_latency"]["p50_seconds"] is None


@pytest.mark.parametrize("kind", ["overlap", "ongoing", "outside_time", "wrong_quarantine"])
def test_bad_segment_membership_is_rejected(monkeypatch, kind):
    s = segment()
    segments = [s]
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: kind != "wrong_quarantine")
    if kind == "overlap":
        segments.append(copy.deepcopy(s))
    elif kind == "ongoing":
        s[1]["status"] = "running"
    elif kind == "outside_time":
        s[1]["started_at"] = s[1]["finished_at"]
    with pytest.raises(ValueError):
        ops.bind_segments(rows(), segments, {"raw_response_sha256": "raw-0"}, 4)


def test_interrupted_segment_reconstruction_keeps_raw_state_and_unknown_timing(tmp_path, monkeypatch):
    path, snapshot = tmp_path / "old.json", tmp_path / "snapshot.json"
    state = {**segment()[1], "status": "running", "finished_at": None,
             "attempt_ids": [], "quarantined_attempt_ids": [], "prospective_known_failure_policy": {"bounded": True}}
    path.write_text(json.dumps(state));snapshot.write_bytes(path.read_bytes())
    audit = {"created_at": "2026-11-01T14:01:00+00:00", "stale_state_snapshots": [{
        "path": str(path), "sha256": ops.contract.file_sha(path), "snapshot_path": str(snapshot),
        "snapshot_sha256": ops.contract.file_sha(snapshot)}]}
    original = rows()
    missing = {**original[2], "attempt_id": "rehearsal/baseline--R9999/01", "status": "unknown",
               "request_started_at": None, "latency_seconds": None, "_admission_journal_created_at": "2026-11-01T14:00:03+00:00"}
    original.append(missing)
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: row["status"] == "failed")
    rebound = ops.apply_interruption_audit(path, state, audit, original)
    assert json.loads(path.read_text()) == state
    assert rebound["status"] == "interrupted_no_live_process" and rebound["finished_at"] is None
    assert len(rebound["attempt_ids"]) == 3
    bound = ops.bind_segments(original, [(path, rebound)], {"raw_response_sha256": "raw-0"}, 4)
    unknown = next(row for row in bound if row["status"] == "unknown")
    assert unknown["effective_concurrency_limit"] == 1
    assert unknown["latency_seconds"] is unknown["request_started_at"] is None
    snapshot.write_text(snapshot.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance changed"):
        ops.apply_interruption_audit(path, state, audit, original)


def test_postpass_approval_survives_multiple_route_segment_approvals(tmp_path, monkeypatch):
    """Each segment's policy path must not overwrite the recovery allowlist object."""
    import sys
    from types import SimpleNamespace

    base = tmp_path / "api"
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    approval = {"allowlist_id": "audit-1", "helper_sha256": {}, "policy_path": "policy.json", "policy_sha256": "pin"}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(experiment / "api-launch.json", {"driver_sha256": "pin", "smoke_review": {
        route: {"raw_response_sha256": "not-smoke"} for route in ("luna", "liquid")}})
    write(experiment / "api-client-interruption-audit.json", {"interrupted_attempts": []})
    by_route = {}
    for route in ("luna", "liquid"):
        row = {**rows()[0], "model": route}
        by_route[route] = [row]
        write(base / route / "driver-status.json", {"status": "stopped"})
        write(base / route / "rehearsal" / "plan.json", {"route": route, "route_config": {"concurrency": 4}})
        state = {**segment()[1], "started_at": "2026-11-01T13:59:00+00:00", "route": route,
                 "attempt_ids": [row["attempt_id"]], "quarantined_attempt_ids": [],
                 "approval_path": str(tmp_path / (route + "-policy.json")),
                 "approval_sha256": "pin", "driver_sha256": "pin"}
        write(base / route / "recovery-segments" / "segment.json", state)
    report = {"allowlist_id": "audit-1", "routes": {route: {"eligible_jobs": 0} for route in by_route}}
    write(base / "postpass" / "report.json", report)
    def reconstruct(plan, passed):
        assert passed is approval
        return by_route[plan["route"]]
    fake = SimpleNamespace(read_allowlist=lambda *args: approval, reconstruct=reconstruct,
                           root=lambda: base / "postpass")
    monkeypatch.setitem(sys.modules, "postpass_recovery", fake)
    monkeypatch.setattr(ops, "HERE", experiment)
    monkeypatch.setattr(ops.contract, "file_sha", lambda path: "pin")
    monkeypatch.setattr(ops.runner, "directory", lambda route, phase=None: base / route / phase if phase else base / route)
    monkeypatch.setattr(ops.runner, "ROUTES", ("luna", "liquid"))
    monkeypatch.setattr(ops.runner, "PHASES", ("rehearsal",))
    monkeypatch.setattr(ops.runner, "validate_plan", lambda plan: None)
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: False)
    monkeypatch.setattr(ops.deepseek_recovery, "no_answer_exhaustion", lambda row: False)
    monkeypatch.setattr(sys, "argv", ["ops", "--postpass-allowlist", "allowlist.json", "--postpass-allowlist-sha256", "pin"])
    ops.main()
    result = json.loads((experiment / "operations-summary.json").read_text())
    assert result["attempts"] == 2 and result["postpass_recovery"] == report


def circuit_fixture(tmp_path, monkeypatch):
    import jev_circuit_reset as reset

    old = {**rows()[0], "request_journal_sha256": "old-request", "response_received_at": "2026-11-01T14:00:00+00:00"}
    new = [{**rows()[0], "attempt_id": f"benchmark/baseline--P{i:04}/01", "phase": "benchmark", "job_id": f"baseline--P{i:04}",
            "request_journal_sha256": f"request-{i}", "request_sha256": f"payload-{i}", "raw_response_sha256": f"raw-new-{i}",
            "request_started_at": f"2026-11-01T14:03:{i:02}+00:00",
            "response_received_at": f"2026-11-01T14:04:{i:02}+00:00"} for i in range(10)]
    approval_path = tmp_path / "approval-06.json"
    ops.runner.write(approval_path, {"route": "jev", "effective_concurrency": 1})
    control = {"control_id": "control-1", "helper_sha256": reset.helper_pins(),
               "prior_failed_halfopens": [], "phase_order": ["benchmark", "rehearsal"],
               "probe_approval": {"path": str(approval_path), "sha256": ops.contract.file_sha(approval_path)},
               "prefix": {"attempt_receipts": [reset.receipt(old)],
                          "observed_prefix": [{**reset.receipt(old), "known_image_503": False}],
                          "probe_jobs": [{"phase": row["phase"], "job_id": row["job_id"],
                                          "request_sha256": row["request_sha256"]} for row in new]}}
    control_path = tmp_path / "control.json"
    ops.runner.write(control_path, control)
    state = {"control_id": control["control_id"], "control_path": str(control_path),
             "control_sha256": ops.contract.file_sha(control_path), "stage": "probe", "status": "probe_succeeded",
             "driver_sha256": ops.contract.file_sha(reset.__file__), "helper_sha256": control["helper_sha256"],
             "approval_path": str(approval_path), "approval_sha256": ops.contract.file_sha(approval_path),
             "effective_concurrency": 1, "phase_order": ["benchmark", "rehearsal"],
             "started_at": "2026-11-01T14:02:30+00:00", "finished_at": "2026-11-01T14:05:00+00:00",
             "attempt_ids": [row["attempt_id"] for row in new], "new_receipts": [reset.receipt(row) for row in new],
             "known_image_503_count": 0, "reset_observed_prefix_count": 1, "consecutive_failed_halfopens": 0}
    path = tmp_path / "circuit-resets" / control["control_id"] / "probe.json"
    ops.runner.write(path, state)
    monkeypatch.setattr(reset, "read_control", lambda *args: control)
    monkeypatch.setattr(ops.api_recovery, "known_image_failure", lambda row: row.get("status") == "failed")
    return path, state, control, [old, *new]


def test_circuit_audit_binds_receipts_without_mutating_rows(tmp_path, monkeypatch):
    path, state, _control, history = circuit_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(history)
    audits, pins = ops.audit_circuit_resets(tmp_path, history)
    assert history == before
    assert audits[0]["status"] == "probe_succeeded" and len(audits[0]["attempt_ids"]) == 10
    assert pins[str(path)] == ops.contract.file_sha(path)
    assert pins[state["control_path"]] == state["control_sha256"]


@pytest.mark.parametrize("mutation", ["ongoing", "wrong_driver", "wrong_concurrency", "wrong_phase_order",
                                      "changed_owned_receipt", "changed_historical_receipt", "outside_lifetime", "duplicate_owned"])
def test_circuit_audit_rejects_existing_receipt_and_provenance_failures(tmp_path, monkeypatch, mutation):
    path, state, _control, history = circuit_fixture(tmp_path, monkeypatch)
    if mutation == "ongoing":
        state["status"] = "running"
    elif mutation == "wrong_driver":
        state["driver_sha256"] = "wrong"
    elif mutation == "wrong_concurrency":
        state["effective_concurrency"] = 4
    elif mutation == "wrong_phase_order":
        state["phase_order"] = ["rehearsal", "benchmark"]
    elif mutation == "changed_owned_receipt":
        history[-1]["raw_response_sha256"] = "changed"
    elif mutation == "changed_historical_receipt":
        history[0]["request_journal_sha256"] = "changed"
    elif mutation == "outside_lifetime":
        state["started_at"] = state["finished_at"]
    else:
        state["attempt_ids"].append(state["attempt_ids"][0])
    ops.runner.write(path, state)
    with pytest.raises(ValueError):
        ops.audit_circuit_resets(tmp_path, history)


@pytest.mark.parametrize("mutation", ["approval_path", "approval_sha256", "helper_sha256", "known_image_503_count",
                                      "reset_observed_prefix_count", "consecutive_failed_halfopens", "incomplete_success"])
def test_circuit_audit_rejects_unbound_stage_claims(tmp_path, monkeypatch, mutation):
    path, state, _control, history = circuit_fixture(tmp_path, monkeypatch)
    if mutation == "approval_path":
        state["approval_path"] = str(tmp_path / "unapproved.json")
    elif mutation in {"approval_sha256", "helper_sha256"}:
        state[mutation] = "changed"
    elif mutation in {"known_image_503_count", "reset_observed_prefix_count", "consecutive_failed_halfopens"}:
        state[mutation] += 1
    else:
        state["attempt_ids"] = state["attempt_ids"][:-1]
        state["new_receipts"] = state["new_receipts"][:-1]
    ops.runner.write(path, state)
    with pytest.raises(ValueError):
        ops.audit_circuit_resets(tmp_path, history)


@pytest.mark.parametrize("mutation", ["historical_response_relabelled", "probe_order_changed"])
def test_circuit_audit_rejects_old_or_reordered_probe_ownership(tmp_path, monkeypatch, mutation):
    import jev_circuit_reset as reset

    path, state, _control, history = circuit_fixture(tmp_path, monkeypatch)
    if mutation == "historical_response_relabelled":
        state["attempt_ids"][0] = history[0]["attempt_id"]
        selected = [row for row in history if row["attempt_id"] in state["attempt_ids"]]
        state["new_receipts"] = sorted((reset.receipt(row) for row in selected), key=lambda row: row["attempt_id"])
        state["started_at"] = "2026-11-01T13:59:00+00:00"
    else:
        state["attempt_ids"] = list(reversed(state["attempt_ids"]))
    ops.runner.write(path, state)
    with pytest.raises(ValueError):
        ops.audit_circuit_resets(tmp_path, history)


@pytest.mark.parametrize('mutation', [None, 'policy', 'historical', 'new', 'sequence', 'incomplete_coverage', 'old_ownership'])
def test_first_pass_availability_audit_keeps_receipts_and_scope(tmp_path, monkeypatch, mutation):
    import jev_availability as availability

    history = rows()
    for row in history:
        row.update(sequence=1, job_id=row['attempt_id'].split('/')[1])
    doc = {'approval_id': 'bounded-first-pass', 'initial_prefix': {'receipts': [availability.prior_reset.receipt(history[0])]},
           'helper_sha256': {}}
    state = {'route': 'jev', 'status': 'stopped_requires_audit', 'started_at': '2026-11-01T14:00:01+00:00',
             'finished_at': '2026-11-01T14:00:04+00:00', 'approval_path': str(tmp_path / 'approval.json'),
             'approval_sha256': 'pinned', 'approval_id': doc['approval_id'],
             'driver_sha256': ops.contract.file_sha(availability.__file__), 'effective_concurrency': 4,
             'effective_availability_policy': availability.POLICY, 'history_scope': 'all_observed_initial_responses',
             'attempt_ids': [row['attempt_id'] for row in history[1:]],
             'new_receipts': [availability.prior_reset.receipt(row) for row in history[1:]]}
    path = tmp_path / 'availability-amendments' / 'state.json'
    path.parent.mkdir()
    if mutation == 'policy':
        state['effective_availability_policy'] = {**availability.POLICY, 'maximum_failures_in_window': 40}
    elif mutation == 'historical':
        history[0]['raw_response_sha256'] = 'changed'
    elif mutation == 'new':
        history[1]['raw_response_sha256'] = 'changed'
    elif mutation == 'sequence':
        history[1]['sequence'] = 2
        state['new_receipts'][0] = availability.prior_reset.receipt(history[1])
    elif mutation == 'incomplete_coverage':
        state['status'] = 'all_initial_jobs_attempted'
    elif mutation == 'old_ownership':
        state['attempt_ids'][0] = history[0]['attempt_id']
        state['new_receipts'][0] = availability.prior_reset.receipt(history[0])
    path.write_text(json.dumps(state))
    monkeypatch.setattr(availability, 'read_approval', lambda *args: doc)
    monkeypatch.setattr(ops.api_recovery, 'known_image_failure', lambda row: row['status'] == 'failed')
    if mutation:
        with pytest.raises(ValueError):
            ops.audit_availability_amendments(tmp_path, history)
    else:
        audited, pins = ops.audit_availability_amendments(tmp_path, history)
        assert audited[0]['known_image_processing_503'] == 1
        assert audited[0]['history_scope'] == 'all_observed_initial_responses'
        assert str(path) in pins


@pytest.mark.parametrize('mutation', [None, 'guard', 'budget', 'late_admission', 'policy', 'missing_metadata', 'omitted_tail', 'unowned_receipt', 'dispatch_membership'])
def test_postpass_availability_audits_each_chronological_admission(tmp_path, monkeypatch, mutation):
    import postpass_availability as availability
    import postpass_recovery as postpass
    r = ops.runner
    reserve = '0.011010048'
    plan_path = tmp_path / 'plan.json';r.write(plan_path, {'reserve_per_attempt_usd': reserve})
    identity = 'rehearsal/baseline--R9999/02'
    row = {'attempt_id': identity, 'model': 'jev', 'phase': 'rehearsal', 'job_id': 'baseline--R9999',
           'sequence': 2, 'status': 'valid', 'price_usd': '.001', 'request_sha256': 'payload',
           'http_status': 200, 'request_started_at': '2026-11-01T16:00:09+00:00', 'response_received_at': '2026-11-01T16:00:10+00:00'}
    dispatch_path = tmp_path / 'dispatch.json';r.write(dispatch_path, {'attempt_ids': [identity], 'effective_concurrency': 1, 'approval_sha256': 'allow-sha'})
    original_path = tmp_path / 'original.json';r.write(original_path, {'admitted_at': '2026-11-01T16:00:06+00:00', 'liability_before_usd': '0', 'segment_path': str(dispatch_path)})
    request_path = tmp_path / 'request.json';r.write(request_path, {'created_at': '2026-11-01T16:00:08+00:00'})
    stopped_path = tmp_path / 'stopped.json';r.write(stopped_path, {'finished_at': '2026-11-01T15:57:00+00:00'})
    receipt = {'attempt_id': identity, 'request_journal_path': str(request_path)}
    approval = {'_verified_file_sha256': 'allow-sha', 'plans': {'jev': {'rehearsal': {'path': str(plan_path)}}},
                'initial_receipts': {'jev': {'rehearsal': []}},
                'allowlist': [{'route': 'jev', 'phase': 'rehearsal', 'job_id': 'baseline--R9999', 'case_id': 'R9999'}]}
    doc = {'amendment_id': 'amendment', 'created_at': '2026-11-01T16:00:00+00:00', 'receipt_prefix': [],
           'stopped_segment': {'path': str(stopped_path)}, 'helper_sha256': {}}
    folder = tmp_path / 'availability-amendments/amendment'
    state = {'status': 'allowed_attempts_complete', 'started_at': '2026-11-01T16:00:04+00:00', 'finished_at': '2026-11-01T16:00:11+00:00',
             'amendment_path': str(tmp_path / 'amendment.json'), 'amendment_sha256': 'amendment-sha', 'amendment_id': 'amendment',
             'driver_sha256': ops.contract.file_sha(availability.__file__), 'effective_policy': availability.POLICY,
             'effective_concurrency': 1, 'allowlist_sha256': 'allow-sha', 'attempt_ids': [identity], 'new_receipts': [receipt]}
    meta = {'amendment_id': 'amendment', 'amendment_sha256': 'amendment-sha', 'adapter_sha256': state['driver_sha256'],
            'allowlist_sha256': 'allow-sha', 'attempt_id': identity, 'request_sha256': 'payload', 'effective_policy': availability.POLICY,
            'admitted_at': '2026-11-01T16:00:07+00:00', 'guard_evidence': {'observed_postpass_responses': 0, 'known503': 0,
            'last100_known503': 0, 'consecutive_known503': 0}, 'liability_before_usd': '0', 'reserved_usd': reserve}
    if mutation == 'guard':
        meta['guard_evidence']['known503'] = 1
    elif mutation == 'budget':
        meta['liability_before_usd'] = '14.999'
    elif mutation == 'late_admission':
        meta['admitted_at'] = '2026-11-01T16:00:10+00:00'
    elif mutation == 'policy':
        state['effective_concurrency'] = 4
    r.write(folder / 'segment.json', state);r.write(folder / 'admissions' / identity / 'availability.json', meta)
    monkeypatch.setattr(postpass, 'root', lambda: tmp_path)
    monkeypatch.setattr(postpass, 'metadata_path', lambda *args: original_path)
    monkeypatch.setattr(postpass, 'receipt', lambda row: receipt)
    monkeypatch.setattr(availability, 'read_amendment', lambda *args: doc)
    records = [row]
    if mutation == 'missing_metadata':
        import shutil
        shutil.rmtree(folder)
    elif mutation == 'omitted_tail':
        state.update(status='stopped_requires_audit', attempt_ids=[], new_receipts=[])
        r.write(folder / 'segment.json', state)
    elif mutation == 'unowned_receipt':
        records.append({**row, 'attempt_id': 'rehearsal/baseline--R8888/02'})
    elif mutation == 'dispatch_membership':
        r.write(dispatch_path, {'attempt_ids': [], 'effective_concurrency': 1, 'approval_sha256': 'allow-sha'})
    expected_amendments = {state['amendment_path']: state['amendment_sha256']}
    if mutation:
        with pytest.raises(ValueError):
            ops.audit_postpass_availability(records, approval, expected_amendments)
    else:
        audited, pins = ops.audit_postpass_availability(records, approval, expected_amendments)
        assert audited[0]['verified_chronological_admissions'] == 1
        assert str(folder / 'admissions' / identity / 'availability.json') in pins
