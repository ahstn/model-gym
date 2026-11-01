"""Synthetic postpass authorization, accounting and crash/resume checks. No API calls."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import analyze
import api_contract as contract
import api_recovery
import api_runner as runner
import postpass_recovery as recovery
import pytest
from test_analysis import FROZEN, attempt, cases, reference
from test_api import envelope, result, small_plan


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No live HTTP calls are permitted in postpass tests")
    monkeypatch.setattr(runner.requests, "post", forbidden)


def journal(plan, job, sequence, body=None):
    folder = runner.directory(plan["route"], plan["phase"]) / "attempts" / job["job_id"] / f"{sequence:02d}"
    runner.write(folder / "request.json", {
        "plan_id": plan["plan_id"], "job_id": job["job_id"], "sequence": sequence,
        "request_sha256": job["request_sha256"], "request_bytes": job["request_bytes"],
        "created_at": runner.now()}, exclusive=True)
    if body is not None:
        runner.write(folder / "result.json", {**body,
            "request_started_at": runner.now(),
            "response_received_at": runner.now(),
            "latency_seconds": 60.0}, exclusive=True)
    return runner.attempt_record(plan, job, folder)


def fixture_approval(tmp_path, monkeypatch, route="jev", interrupted=False):
    monkeypatch.setattr(runner, "DATA", tmp_path)
    plan, job = small_plan(tmp_path, route)
    path = runner.directory(route, "rehearsal") / "plan.json"
    runner.write(path, plan)
    row = journal(plan, job, 1, None if interrupted else result(api_recovery.ERROR_BODY, 503))
    kind = "audited_client_interruption" if interrupted else "exact_jev_image_processing_503"
    item = {"route": route, "phase": "rehearsal", "job_id": job["job_id"], "case_id": job["case"]["case_id"],
            "source_class": kind, "source_attempt_id": row["attempt_id"], "source_receipt": recovery.receipt(row),
            "original_history": [recovery.receipt(row)], "initial_sequence": 1,
            "maximum_additional_attempts": recovery.LIMITS[kind], "maximum_sequence": 1 + recovery.LIMITS[kind]}
    approval = {"allowlist_id": "synthetic-approval", "policy_sha256": "synthetic-policy",
                "created_at": "2026-01-01T00:00:00+00:00", "_verified_file_sha256": "hash",
                "helper_sha256": {str(Path(recovery.__file__).resolve()): contract.file_sha(recovery.__file__)},
                "plans": {route: {"rehearsal": {"path": str(path), "sha256": contract.file_sha(path),
                    "plan_id": plan["plan_id"], "job_ids": [job["job_id"]]}}},
                "initial_receipts": {route: {"rehearsal": [recovery.receipt(row)]}}, "allowlist": [item]}
    return plan, job, row, approval


def admission(row, approval):
    item = approval["allowlist"][0]
    runner.write(recovery.metadata_path(row["model"], row["attempt_id"]), {
        "allowlist_id": approval["allowlist_id"], "driver_sha256": contract.file_sha(recovery.__file__),
        "approval_sha256": approval["_verified_file_sha256"], "policy_sha256": approval["policy_sha256"],
        "admitted_at": "2026-01-02T00:00:00+00:00", "source_class": item["source_class"],
        "source_receipt": item["source_receipt"],
        "effective_concurrency": 1, "sequence": row["sequence"], "request_sha256": row["request_sha256"],
        "source_attempt_id": item["source_attempt_id"]}, exclusive=True)


@pytest.mark.parametrize("status", ["valid", "partial", "invalid"])
def test_no_scored_or_completed_history_is_selected(status, monkeypatch):
    monkeypatch.setattr(recovery, "known_failure", lambda row: True)
    rows = [{"status": status}, {"status": "failed"}]
    assert recovery.classify_initial(rows, {}) is None


def test_native_or_partial_score_protects_even_unexpected_status():
    assert recovery.protected({"status": "failed", "technical_score_native": 2.0})
    assert recovery.protected({"status": "unknown", "aesthetic_score": 2.0})


@pytest.mark.parametrize("change", [None, "body", "http", "bill", "hash"])
def test_only_exact_original_jev_error_is_eligible(tmp_path, monkeypatch, change):
    _plan, _job, row, _approval = fixture_approval(tmp_path, monkeypatch)
    if change == "body":
        path = Path(row["raw_response_path"])
        raw = runner.read(path)
        raw["body"] = json.dumps({"error": {"message": "Other HTTP 503", "code": 503}})
        runner.write(path, raw)
        row["raw_response_sha256"] = contract.file_sha(path)
    elif change == "http":
        row["http_status"] = 502
    elif change == "bill":
        row["price_usd"] = "0.01"
    elif change == "hash":
        row["raw_response_sha256"] = "changed"
    assert recovery.classify_initial([row], {}) == ("exact_jev_image_processing_503" if change is None else None)


def test_interruption_is_receipt_bound(tmp_path, monkeypatch):
    _plan, _job, row, _approval = fixture_approval(tmp_path, monkeypatch, "luna", True)
    audited = {row["attempt_id"]: recovery.receipt(row)}
    assert recovery.classify_initial([row], audited) == "audited_client_interruption"
    audited[row["attempt_id"]]["request_sha256"] = "changed"
    assert recovery.classify_initial([row], audited) is None


def test_jev_503_allows_two_extra_attempts_and_no_more(tmp_path, monkeypatch):
    plan, job, original, approval = fixture_approval(tmp_path, monkeypatch)
    rows = [original]
    for sequence in (2, 3):
        assert recovery.next_job(approval["allowlist"], rows)[1] == sequence
        row = journal(plan, job, sequence, result(api_recovery.ERROR_BODY, 503))
        admission(row, approval)
        rows.append(row)
        assert recovery.reconstruct(plan, approval) == rows
    assert recovery.next_job(approval["allowlist"], rows) is None
    extra = journal(plan, job, 4, result(api_recovery.ERROR_BODY, 503))
    admission(extra, approval)
    with pytest.raises(ValueError, match="exceeded cap"):
        recovery.reconstruct(plan, approval)


def test_interrupted_jev_followed_by_503_does_not_gain_another_retry(tmp_path, monkeypatch):
    plan, job, row, approval = fixture_approval(tmp_path, monkeypatch, interrupted=True)
    extra = journal(plan, job, 2, result(api_recovery.ERROR_BODY, 503))
    admission(extra, approval)
    assert len(recovery.reconstruct(plan, approval)) == 2
    assert recovery.next_job(approval["allowlist"], [row, extra]) is None
    third = journal(plan, job, 3, result(api_recovery.ERROR_BODY, 503))
    admission(third, approval)
    with pytest.raises(ValueError, match="Unauthorized retry"):
        recovery.reconstruct(plan, approval)


@pytest.mark.parametrize("invalid", [False, True])
def test_completed_recovery_is_never_replayed_even_when_invalid(tmp_path, monkeypatch, invalid):
    plan, job, original, approval = fixture_approval(tmp_path, monkeypatch, "luna", True)
    body = envelope()
    if invalid:
        content = json.loads(body["choices"][0]["message"]["content"])
        content["aesthetics"]["score"] = "9.00"
        body["choices"][0]["message"]["content"] = json.dumps(content)
    extra = journal(plan, job, 2, result(body))
    admission(extra, approval)
    assert extra["status"] == ("invalid" if invalid else "valid")
    assert recovery.next_job(approval["allowlist"], [original, extra]) is None
    assert recovery.stop_reason([original, extra], approval, "luna", Decimal(".23")) is None
    assert len(recovery.reconstruct(plan, approval)) == 2
    third = journal(plan, job, 3, result(envelope()))
    admission(third, approval)
    with pytest.raises(ValueError, match="replay"):
        recovery.reconstruct(plan, approval)


@pytest.mark.parametrize("target", ["request", "result", "plan"])
def test_original_receipt_and_plan_changes_are_rejected(tmp_path, monkeypatch, target):
    plan, _job, row, approval = fixture_approval(tmp_path, monkeypatch)
    path = (Path(approval["plans"]["jev"]["rehearsal"]["path"]) if target == "plan" else
            Path(recovery.receipt(row)["request_journal_path"] if target == "request" else row["raw_response_path"]))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        recovery.reconstruct(plan, approval)


def test_original_request_only_receipt_cannot_later_be_replaced(tmp_path, monkeypatch):
    plan, _job, row, approval = fixture_approval(tmp_path, monkeypatch, "luna", True)
    path = Path(recovery.receipt(row)["request_journal_path"]).with_name("result.json")
    runner.write(path, result(envelope()))
    with pytest.raises(ValueError, match="Original request/response"):
        recovery.reconstruct(plan, approval)


def test_new_request_only_attempt_consumes_slot_and_blocks_resume(tmp_path, monkeypatch):
    plan, job, _original, approval = fixture_approval(tmp_path, monkeypatch)
    extra = journal(plan, job, 2)
    admission(extra, approval)
    records = recovery.reconstruct(plan, approval)
    assert recovery.stop_reason(records, approval, "jev", Decimal(".2")) == "new_unreviewed_error_or_unknown_bill"
    assert recovery.accounting(records, Decimal(".2"))["admission_liability_usd"] == "0.4"
    with pytest.raises(ValueError, match="cannot acquire"):
        recovery.next_job(approval["allowlist"], records)


def test_missing_admission_provenance_rejects_extra_journal(tmp_path, monkeypatch):
    plan, job, _original, approval = fixture_approval(tmp_path, monkeypatch)
    journal(plan, job, 2, result(api_recovery.ERROR_BODY, 503))
    with pytest.raises(FileNotFoundError):
        recovery.reconstruct(plan, approval)


@pytest.mark.parametrize("field", ["approval_sha256", "policy_sha256", "source_class", "source_receipt",
                                   "admitted_after_request", "admitted_before_allowlist", "naive_time"])
def test_admission_hashes_source_and_pre_request_timing_are_bound(tmp_path, monkeypatch, field):
    plan, job, _original, approval = fixture_approval(tmp_path, monkeypatch)
    extra = journal(plan, job, 2, result(api_recovery.ERROR_BODY, 503))
    admission(extra, approval)
    path = recovery.metadata_path("jev", extra["attempt_id"])
    metadata = runner.read(path)
    if field == "admitted_after_request":
        metadata["admitted_at"] = "2100-01-01T00:00:00+00:00"
    elif field == "admitted_before_allowlist":
        metadata["admitted_at"] = "2025-01-01T00:00:00+00:00"
    elif field == "naive_time":
        metadata["admitted_at"] = "2026-01-02T00:00:00"
    else:
        metadata[field] = "changed"
    runner.write(path, metadata)
    with pytest.raises(ValueError, match="metadata changed|admission must precede"):
        recovery.reconstruct(plan, approval)


def test_appended_outcome_receipt_is_bound_on_resume(tmp_path, monkeypatch):
    plan, job, _original, approval = fixture_approval(tmp_path, monkeypatch)
    extra = journal(plan, job, 2, result(api_recovery.ERROR_BODY, 503))
    admission(extra, approval)
    outcome = recovery.metadata_path("jev", extra["attempt_id"]).with_name("outcome.json")
    runner.write(outcome, {"receipt": recovery.receipt(extra)})
    Path(extra["raw_response_path"]).write_text(Path(extra["raw_response_path"]).read_text() + "\n")
    with pytest.raises(ValueError, match="outcome receipt changed"):
        recovery.reconstruct(plan, approval)


@pytest.mark.parametrize("partial", [False, True])
def test_reports_distinguish_completed_from_fully_scored_recovery(tmp_path, monkeypatch, partial):
    plan, job, original, approval = fixture_approval(tmp_path, monkeypatch, "luna", True)
    recovered = journal(plan, job, 2, result(envelope()))
    if partial:
        recovered.update(status="invalid", aesthetic_score=None)
        recovered["task_valid"]["aesthetics"] = False
    report = recovery.route_report("luna", {"rehearsal": plan}, [original, recovered], approval)
    assert report["recovered"] == 1 and report["persistent"] == 0
    assert report["fully_scored_recovered"] == int(not partial)
    assert report["partial_or_invalid_recovered"] == int(partial)
    assert report["missing_jobs"] == int(partial)


def test_case_order_does_not_depend_on_input_order_or_labels():
    entries, rows = [], []
    for case in ("R0200", "R0001", "R0002"):
        entries.append({"case_id": case, "phase": "rehearsal", "job_id": case,
                        "initial_sequence": 1, "maximum_sequence": 2})
        rows.append({"phase": "rehearsal", "job_id": case, "sequence": 1, "status": "unknown"})
    assert recovery.next_job(entries, rows)[0]["case_id"] == "R0001"


@pytest.mark.parametrize("pattern, expected", [([True] * 5, "consecutive_known_provider_failures"),
    ([True, False, False, False, False] * 20, "known_provider_failure_rate"),
    ([True, False] * 4, None)])
def test_circuit_breakers_count_observed_responses_across_segments(monkeypatch, pattern, expected):
    rows = [{"attempt_id": str(i), "known": flag, "http_status": 503 if flag else 200,
             "response_received_at": f"2026-11-01T12:{i:03d}", "status": "failed" if flag else "valid",
             "price_usd": None if flag else ".01"} for i, flag in enumerate(pattern)]
    rows.append({"attempt_id": "orphan", "status": "unknown", "http_status": None, "response_received_at": None})
    approval = {"initial_receipts": {"jev": {"rehearsal": rows}}}
    monkeypatch.setattr(recovery, "known_failure", lambda row: row.get("known", False))
    assert recovery.stop_reason(rows, approval, "jev", Decimal(".2")) == expected


def test_service_gate_reads_no_environment_and_rejects_active(monkeypatch):
    calls = []
    def systemctl(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="ActiveState=active\nMainPID=123\nControlGroup=/synthetic\n")
    monkeypatch.setattr(recovery.subprocess, "run", systemctl)
    with pytest.raises(ValueError, match="still live"):
        recovery.service_snapshot()
    assert calls[0][-1] == "--property=ActiveState,MainPID,ControlGroup"


def test_service_gate_accepts_stopped_units_without_old_pid_pins(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="ActiveState=inactive\nMainPID=0\nControlGroup=\n"))
    assert set(recovery.service_snapshot()["units"]) == set(runner.ROUTES)


def test_prepare_refuses_live_service_before_reading_initial_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DATA", tmp_path)
    for route in runner.ROUTES:
        runner.directory(route).mkdir(parents=True)
    def live():
        raise ValueError("Initial service still live")
    monkeypatch.setattr(recovery, "service_snapshot", live)
    monkeypatch.setattr(recovery, "load_plans", lambda **kwargs: pytest.fail("Must not read initial response data"))
    with pytest.raises(ValueError, match="still live"):
        recovery.prepare(tmp_path / "never-created.json")
    assert not (tmp_path / "never-created.json").exists()


def test_unknown_bills_keep_full_reserve_and_known_bills_are_per_attempt():
    rows = [{"price_usd": None}, {"price_usd": ".02"}, {"price_usd": None}]
    totals = recovery.accounting(rows, Decimal(".25"))
    assert totals["admission_liability_usd"] == "0.52"
    assert totals["known_price_usd"] == "0.02" and totals["total_price_usd"] is None
    assert totals["unpriced_attempts"] == 2


@pytest.mark.parametrize("first_status", ["failed", "unknown"])
def test_analysis_keeps_failed_history_and_one_completed_response(first_status):
    corpus = cases()
    first = attempt(corpus[0], price=None)
    first.update(status=first_status, technical_score=None, aesthetic_score=None,
                 technical_score_native=None, aesthetic_score_native=None, task_valid={})
    second = attempt(corpus[0], sequence=2, price=".004")
    rows = analyze.build_long(corpus, [first, second], [reference(corpus[0])], frozen_at=FROZEN)
    row = next(row for row in rows if row["model"] == "luna" and row["case_id"] == corpus[0]["case_id"] and row["arm"] == "baseline")
    assert row["attempt_count"] == 2 and row["technical_score"] == second["technical_score"]
    assert row["unpriced_attempts"] == 1 and row["total_attempt_price_usd"] is None
    assert Decimal(row["total_attempt_known_price_usd"]) == Decimal(".004")
    assert first["price_usd"] is None


def test_two_completed_analysis_rows_are_rejected_even_if_one_invalid():
    corpus = cases()
    first = attempt(corpus[0])
    first["status"] = "invalid"
    second = attempt(corpus[0], sequence=2)
    with pytest.raises(ValueError, match="Multiple completed"):
        analyze.build_long(corpus, [first, second], [], frozen_at=FROZEN)


def test_dispatch_budget_includes_other_phase_and_unknown_originals(tmp_path, monkeypatch):
    plan, _job, original, approval = fixture_approval(tmp_path, monkeypatch)
    plan["reserve_per_attempt_usd"] = "8"
    plans = {"jev": {"rehearsal": plan, "benchmark": plan}}
    records = {"jev": [original, {**original, "attempt_id": "benchmark/other/01", "phase": "benchmark", "job_id": "other"}]}
    monkeypatch.setattr(runner, "ROUTES", ("jev",))
    monkeypatch.setattr(recovery, "all_records", lambda *a: records)
    monkeypatch.setattr(recovery, "write_reports", lambda *a: None)
    monkeypatch.setattr(runner, "send", lambda *a: pytest.fail("Budget must reject before dispatch"))
    with pytest.raises(RuntimeError, match="budget admission exhausted"):
        recovery.dispatch("jev", plans, records, approval, tmp_path / "approval.json", "hash", None)


def test_policy_is_post_hoc_bounded_and_interruption_audit_pinned():
    policy = recovery.policy_document()
    assert "not a preregistered" in policy["amendment"]
    assert policy["maximum_additional_attempts"] == recovery.LIMITS
    assert sum(sum(phases.values()) for phases in policy["expected_initial_jobs"].values()) == 18000


def complete_initial_fixture(tmp_path, monkeypatch):
    """Small corpus retaining the actual audited interruption count per route."""
    monkeypatch.setattr(runner, "DATA", tmp_path)
    plans, records, audited = {}, {}, []
    for route in runner.ROUTES:
        plans[route], records[route] = {}, {}
        for phase in runner.PHASES:
            base, template = small_plan(tmp_path, route)
            count = {"luna": 16, "liquid": 4, "jev": 1}.get(route, 0) if phase == "rehearsal" else 0
            base.update(phase=phase, jobs=[])
            for index in range(max(count, 1)):
                job = copy.deepcopy(template)
                job["case"]["case_id"] = f"R{index:04}"
                job["job_id"] = "baseline--" + job["case"]["case_id"]
                base["jobs"].append(job)
            base.pop("plan_id")
            base["plan_id"] = contract.digest(base)
            runner.write(runner.directory(route, phase) / "plan.json", base)
            rows = []
            for job in base["jobs"]:
                row = journal(base, job, 1)
                if count:
                    audited.append({"route": route, **recovery.receipt(row)})
                else:
                    # Frozen receipt metadata is enough to test completion gating;
                    # exact parser output and journal replay are covered separately.
                    row.update(status="valid", reason=None, price_usd=".01")
                rows.append(row)
            plans[route][phase], records[route][phase] = base, rows
    original_read = runner.read
    monkeypatch.setattr(runner, "read", lambda path: {"interrupted_attempts": audited}
                        if Path(path).name == "api-client-interruption-audit.json" else original_read(path))
    policy = {"interruption_counts": {"luna": 16, "liquid": 4, "jev": 1}}
    return policy, plans, records


def test_freeze_binds_all_initial_receipts_and_all_21_exact_interruptions(tmp_path, monkeypatch):
    policy, plans, records = complete_initial_fixture(tmp_path, monkeypatch)
    approval = recovery.freeze_document(policy, plans, records, {"units": "synthetic-stopped"})
    assert len(approval["allowlist"]) == 21
    assert {item["route"] for item in approval["allowlist"]} == {"luna", "liquid", "jev"}
    assert all(item["maximum_sequence"] == 2 for item in approval["allowlist"])
    assert len(approval["initial_receipts"]["luna"]["rehearsal"]) == 16
    assert approval["initial_status_snapshot"]["luna"]["unpriced_attempts"] == 16
    assert approval["helper_sha256"][str(Path(recovery.__file__).resolve())] == contract.file_sha(recovery.__file__)


@pytest.mark.parametrize("change", ["incomplete", "capacity_pending", "new_orphan", "missing_audited"])
def test_freeze_rejects_incomplete_or_unreviewed_initial_history(tmp_path, monkeypatch, change):
    policy, plans, records = complete_initial_fixture(tmp_path, monkeypatch)
    if change == "incomplete":
        records["deepseek"]["benchmark"] = []
    elif change == "capacity_pending":
        records["deepseek"]["benchmark"][0]["status"] = "rejected_retryable"
    elif change == "new_orphan":
        records["deepseek"]["benchmark"][0].update(status="unknown", reason="interrupted_request")
    else:
        records["luna"]["rehearsal"][0].update(status="valid", reason=None)
    with pytest.raises(ValueError):
        recovery.freeze_document(policy, plans, records, {})


def test_allowlist_hash_and_helper_drift_are_rejected(tmp_path, monkeypatch):
    policy, plans, records = complete_initial_fixture(tmp_path, monkeypatch)
    approval = recovery.freeze_document(policy, plans, records, {})
    path = tmp_path / "allowlist.json"
    runner.write(path, approval)
    monkeypatch.setattr(recovery, "policy_document", lambda: policy)
    digest = contract.file_sha(path)
    assert recovery.read_allowlist(path, digest) == {**approval, "_verified_file_sha256": digest}
    with pytest.raises(ValueError, match="file hash"):
        recovery.read_allowlist(path, "wrong")
    monkeypatch.setattr(recovery, "helper_pins", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="helper source"):
        recovery.read_allowlist(path, digest)


def test_dispatch_serially_saves_one_attempt_and_resumes_only_remaining(tmp_path, monkeypatch):
    plan, job, original, approval = fixture_approval(tmp_path, monkeypatch)
    plans, records = {"jev": {"rehearsal": plan}}, {"jev": [original]}
    monkeypatch.setattr(runner, "ROUTES", ("jev",))
    monkeypatch.setattr(recovery, "read_allowlist", lambda *args: approval)
    monkeypatch.setattr(runner, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery, "write_reports", lambda *args: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-never-sent")
    sent = []
    def send(frozen, selected, sequence, secret, admitted_at):
        sent.append((sequence, selected["request_sha256"]))
        return journal(frozen, selected, sequence, result(api_recovery.ERROR_BODY, 503))
    monkeypatch.setattr(runner, "send", send)
    first = recovery.dispatch("jev", plans, records, approval, tmp_path / "approval.json", "hash", 1)
    assert first["new_attempts"] == 1 and first["status"] == "limited_dispatch_complete"
    records = recovery.all_records(plans, approval)
    second = recovery.dispatch("jev", plans, records, approval, tmp_path / "approval.json", "hash", None)
    assert second["new_attempts"] == 1 and second["status"] == "allowed_attempts_complete"
    assert sent == [(2, job["request_sha256"]), (3, job["request_sha256"])]
    assert contract.file_sha(Path(recovery.receipt(original)["request_journal_path"])) == original["request_journal_sha256"]
    for path in (recovery.root() / "jev" / "segments").glob("*.json"):
        segment = runner.read(path)
        assert segment["effective_concurrency"] == 1 and segment["finished_at"]
