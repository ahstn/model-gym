"""Audit API operational segments without changing scoring or receipt journals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner
import deepseek_recovery
import interruption_resume
import numpy as np

HERE = Path(__file__).resolve().parent


def latency(rows: list[dict]) -> dict:
    values = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
    return {
        "measured": len(values), "unmeasured": len(rows) - len(values),
        "p50_seconds": float(np.quantile(values, 0.5)) if values else None,
        "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["phase"], row["operational_segment_id"], row["effective_concurrency_limit"])].append(row)
    result = []
    for (model, phase, segment, concurrency), group in sorted(grouped.items()):
        completed = [r for r in group if r["status"] in {"valid", "invalid"}]
        known = sum((Decimal(r["price_usd"]) for r in group if r.get("price_usd") is not None), Decimal(0))
        unknown = sum(r.get("price_usd") is None for r in group)
        result.append({
            "model": model, "phase": phase, "operational_segment_id": segment,
            "effective_concurrency_limit": concurrency, "attempts": len(group),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "http_status_counts": dict(Counter(str(r.get("http_status")) for r in group)),
            "known_image_processing_503": sum(r["known_image_processing_503"] for r in group),
            "billed_no_answer_exhaustions": sum(r.get("billed_no_answer_exhaustion", False) for r in group),
            "quarantined_under_prospective_policy": sum(r["prospectively_quarantined"] for r in group),
            "all_attempt_latency": latency(group), "completed_response_latency": latency(completed),
            "known_price_usd": str(known), "unpriced_attempts": unknown,
            "total_price_usd": None if unknown else str(known),
        })
    return result


def bind_segments(records: list[dict], segments: list[tuple[Path, dict]], smoke: dict, concurrency: int) -> list[dict]:
    by_id = {row["attempt_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate operational attempt identity")
    mapping = {}
    for path, segment in segments:
        end = segment.get("finished_at") or segment.get("termination_observed_at")
        if segment["status"] == "running" or not end:
            raise ValueError("Operational segment still running or incomplete")
        ids = segment["attempt_ids"]
        quarantined = set(segment.get("quarantined_attempt_ids", []))
        if len(set(ids)) != len(ids) or not quarantined <= set(ids):
            raise ValueError("Duplicate or extraneous segment/quarantine membership")
        for identity in ids:
            if identity not in by_id or identity in mapping:
                raise ValueError("Missing attempt or overlapping operational segments")
            row = by_id[identity]
            event_time = row.get("request_started_at") or row.get("_admission_journal_created_at")
            if not event_time or not segment["started_at"] <= event_time <= end:
                raise ValueError("Attempt falls outside its operational segment")
            mapping[identity] = (path.stem, segment["effective_concurrency"], identity in quarantined)
    result = []
    for row in records:
        identity = row["attempt_id"]
        if identity in mapping:
            segment, effective, quarantined = mapping[identity]
        elif row["raw_response_sha256"] == smoke["raw_response_sha256"]:
            segment, effective, quarantined = "initial_single_request_smoke", 1, False
        else:
            segment, effective, quarantined = "original_frozen_plan", concurrency, False
        known = api_recovery.known_image_failure(row)
        if quarantined and not known:
            raise ValueError("Quarantine membership does not match the exact reviewed error class")
        result.append({
            **row, "operational_segment_id": segment, "effective_concurrency_limit": effective,
            "frozen_plan_concurrency_limit": concurrency, "known_image_processing_503": known,
            "prospectively_quarantined": quarantined,
            "billed_no_answer_exhaustion": deepseek_recovery.no_answer_exhaustion(row),
        })
    return result


def apply_interruption_audit(path: Path, state: dict, audit: dict, records: list[dict] | None = None) -> dict:
    """Overlay observed termination without rewriting or fabricating old timestamps."""
    matching = [row for row in audit.get("stale_state_snapshots", []) if Path(row["path"]).resolve() == path.resolve()]
    if not matching:
        return state
    if len(matching) != 1:
        raise ValueError("Duplicate interruption-state declaration")
    evidence = matching[0]
    if (contract.file_sha(path) != evidence["sha256"]
            or contract.file_sha(evidence["snapshot_path"]) != evidence["snapshot_sha256"]):
        raise ValueError("Audited interrupted process-state provenance changed")
    if state["status"] != "running":
        return state
    result = {**state, "status": "interrupted_no_live_process", "termination_observed_at": audit["created_at"],
              "original_status": state["status"], "interruption_evidence_sha256": evidence["snapshot_sha256"]}
    if records is not None:
        included = [row for row in records if state["started_at"] <= (
            row.get("request_started_at") or row.get("_admission_journal_created_at") or ""
        ) <= audit["created_at"]]
        result["attempt_ids"] = sorted(set(state["attempt_ids"]) | {row["attempt_id"] for row in included})
        result["membership_reconstruction"] = "Original checkpoint plus immutable attempt journals within the audited process lifetime; interruption observation is not an actual response timestamp"
        if state.get("prospective_known_failure_policy"):
            result["quarantined_attempt_ids"] = sorted(set(state.get("quarantined_attempt_ids", [])) | {
                row["attempt_id"] for row in included if api_recovery.known_image_failure(row)
            })
    return result


def audit_circuit_resets(root: Path, records: list[dict]) -> tuple[list[dict], dict]:
    """Bind later circuit-reset metadata to original receipt-backed segments."""
    paths = sorted((root / "circuit-resets").glob("*/*.json"))
    if not paths:
        return [], {}
    import jev_circuit_reset as reset

    by_id = {row["attempt_id"]: row for row in records}
    audits, pins, assigned = [], {}, set()
    for path in paths:
        state = runner.read(path)
        if state["status"] == "running" or not state.get("finished_at"):
            raise ValueError("Circuit-reset stage is still running")
        control_path = Path(state["control_path"])
        control = reset.read_control(control_path, state["control_sha256"])
        if (state["control_id"] != control["control_id"] or state["driver_sha256"] != contract.file_sha(reset.__file__)
                or state["effective_concurrency"] != {"probe": 1, "continue": 4}[state["stage"]]
                or state["phase_order"] != ["benchmark", "rehearsal"]):
            raise ValueError("Circuit-reset stage provenance changed")
        stage_approval = control[state["stage"] + "_approval"]
        if (state["approval_path"] != stage_approval["path"] or state["approval_sha256"] != stage_approval["sha256"]
                or state["helper_sha256"] != control["helper_sha256"]
                or state["reset_observed_prefix_count"] != len(control["prefix"]["observed_prefix"])
                or state["consecutive_failed_halfopens"] != len(control["prior_failed_halfopens"]) + (state["status"] == "probe_failed")):
            raise ValueError("Circuit-reset stage approval or derived metadata changed")
        historical_ids = {row["attempt_id"] for row in control["prefix"]["attempt_receipts"]}
        if historical_ids.intersection(state["attempt_ids"]):
            raise ValueError("Circuit-reset stage relabels an historical attempt")
        if state["stage"] == "probe":
            expected_ids = [row["phase"] + "/" + row["job_id"] + "/01" for row in control["prefix"]["probe_jobs"]]
            if state["attempt_ids"] != expected_ids[:len(state["attempt_ids"])]:
                raise ValueError("Circuit-reset probe job order changed")
        selected = [by_id[identity] for identity in state["attempt_ids"]]
        if state["known_image_503_count"] != sum(api_recovery.known_image_failure(row) for row in selected):
            raise ValueError("Circuit-reset failure count differs from receipts")
        if state["stage"] == "probe" and (state["status"] == "probe_succeeded") != reset.probe_success(selected):
            raise ValueError("Circuit-reset probe status differs from receipts")
        if assigned.intersection(state["attempt_ids"]) or len(selected) != len(set(state["attempt_ids"])):
            raise ValueError("Circuit-reset stages overlap")
        assigned.update(state["attempt_ids"])
        if sorted((reset.receipt(row) for row in selected), key=lambda row: row["attempt_id"]) != state["new_receipts"]:
            raise ValueError("Circuit-reset owned receipt changed")
        for frozen in control["prefix"]["attempt_receipts"]:
            if reset.receipt(by_id[frozen["attempt_id"]]) != frozen:
                raise ValueError("Circuit-reset historical receipt changed")
        for row in selected:
            if not state["started_at"] <= row["request_started_at"] <= state["finished_at"]:
                raise ValueError("Circuit-reset request outside recorded lifetime")
        pins[str(path)] = contract.file_sha(path)
        pins[str(control_path)] = state["control_sha256"]
        pins.update(control["helper_sha256"])
        audits.append({key: state[key] for key in ("control_id", "stage", "status", "started_at", "finished_at",
                      "effective_concurrency", "phase_order", "attempt_ids", "known_image_503_count",
                      "reset_observed_prefix_count", "consecutive_failed_halfopens")})
    return sorted(audits, key=lambda row: row["started_at"]), pins


def audit_availability_amendments(root: Path, records: list[dict]) -> tuple[list[dict], dict]:
    paths = sorted((root / "availability-amendments").glob("*.json"))
    if not paths:
        return [], {}
    import jev_availability as availability

    by_id = {row["attempt_id"]: row for row in records}
    audits, pins = [], {}
    assigned = set()
    for path in paths:
        state = runner.read(path)
        if state["status"] == "running" or not state.get("finished_at"):
            raise ValueError("First-pass availability amendment still running")
        approval_path = Path(state["approval_path"])
        doc = availability.read_approval(approval_path, state["approval_sha256"])
        if (state["approval_id"] != doc["approval_id"] or state["driver_sha256"] != contract.file_sha(availability.__file__)
                or state["effective_concurrency"] != 4 or state["effective_availability_policy"] != availability.POLICY
                or state["history_scope"] != "all_observed_initial_responses"):
            raise ValueError("First-pass availability stage provenance changed")
        old = {row["attempt_id"]: row for row in doc["initial_prefix"]["receipts"]}
        identities = state["attempt_ids"]
        if len(identities) != len(set(identities)) or assigned.intersection(identities) or set(old).intersection(identities):
            raise ValueError("First-pass availability attempt ownership overlaps")
        assigned.update(identities)
        if any(availability.prior_reset.receipt(by_id[identity]) != receipt for identity, receipt in old.items()):
            raise ValueError("First-pass availability historical receipt changed")
        selected = [by_id[identity] for identity in identities]
        if sorted((availability.prior_reset.receipt(row) for row in selected), key=lambda row: row["attempt_id"]) != state["new_receipts"]:
            raise ValueError("First-pass availability new receipt changed")
        if any(row["sequence"] != 1 or not state["started_at"] <= row["request_started_at"] <= state["finished_at"] for row in selected):
            raise ValueError("First-pass availability sequence/lifetime changed")
        if state["status"] == "all_initial_jobs_attempted" and len({(row["phase"], row["job_id"]) for row in records if row["sequence"] == 1}) != 4000:
            raise ValueError("First-pass availability claims incomplete initial coverage")
        pins[str(path)], pins[str(approval_path)] = contract.file_sha(path), state["approval_sha256"]
        pins.update(doc["helper_sha256"])
        audits.append({key: state[key] for key in ("approval_id", "status", "started_at", "finished_at",
                       "effective_concurrency", "effective_availability_policy", "history_scope", "attempt_ids")}
                      | {"known_image_processing_503": sum(api_recovery.known_image_failure(row) for row in selected)})
    return sorted(audits, key=lambda row: row["started_at"]), pins


def audit_postpass_availability(records: list[dict], approval: dict, expected_amendments: dict) -> tuple[list[dict], dict]:
    import postpass_availability as availability
    import postpass_recovery as postpass

    by_id = {row["attempt_id"]: row for row in records}
    audits, pins, owned, required_ownership = [], {}, set(), set()
    paths = sorted((postpass.root() / "availability-amendments").glob("*/segment.json"))
    if expected_amendments is not None:
        expected_ids = {availability.read_amendment(Path(path), digest, approval)["amendment_id"]
                        for path, digest in expected_amendments.items()}
        if {path.parent.name for path in paths} != expected_ids:
            raise ValueError("Missing or unexpected postpass availability amendment metadata")
    reserve = Decimal(runner.read(Path(approval["plans"]["jev"]["rehearsal"]["path"]))["reserve_per_attempt_usd"])
    for path in paths:
        state = runner.read(path)
        if state["status"] == "running" or not state.get("finished_at"):
            raise ValueError("Postpass availability amendment still running")
        amendment_path = Path(state["amendment_path"])
        if expected_amendments is not None and expected_amendments.get(str(amendment_path)) != state["amendment_sha256"]:
            raise ValueError("Postpass availability amendment differs from recorded launch")
        doc = availability.read_amendment(amendment_path, state["amendment_sha256"], approval)
        if (state["amendment_id"] != doc["amendment_id"] or state["driver_sha256"] != contract.file_sha(availability.__file__)
                or state["effective_policy"] != availability.POLICY or state["effective_concurrency"] != 1
                or state["allowlist_sha256"] != approval["_verified_file_sha256"]):
            raise ValueError("Postpass availability stage provenance changed")
        history = []
        for frozen in doc["receipt_prefix"]:
            row = by_id[frozen["attempt_id"]]
            if postpass.receipt(row) != frozen:
                raise ValueError("Postpass availability historical receipt changed")
            history.append(row)
        historical = {row["attempt_id"] for row in history}
        required_ownership.update(set(by_id) - historical)
        identities = state["attempt_ids"]
        admission_files = list((path.parent / "admissions").glob("**/availability.json"))
        admitted_ids = {"/".join(file.relative_to(path.parent / "admissions").parts[:-1]) for file in admission_files}
        if admitted_ids != set(identities) or len(admission_files) != len(identities):
            raise ValueError("Orphan or omitted postpass availability admission")
        dispatch_paths = {runner.read(postpass.metadata_path("jev", identity))["segment_path"] for identity in identities}
        declared_dispatch = state.get("original_dispatch_segment", {}).get("segment_path")
        if declared_dispatch:
            dispatch_paths.add(declared_dispatch)
        if len(dispatch_paths) != 1:
            raise ValueError("Postpass availability lacks one bound original dispatch segment")
        dispatch_path = Path(next(iter(dispatch_paths)))
        dispatch = runner.read(dispatch_path)
        if (dispatch["attempt_ids"] != identities or dispatch["effective_concurrency"] != 1
                or dispatch["approval_sha256"] != approval["_verified_file_sha256"]):
            raise ValueError("Original dispatch segment and availability ownership differ")
        pins[str(dispatch_path)] = contract.file_sha(dispatch_path)
        if len(identities) != len(set(identities)) or historical.intersection(identities) or owned.intersection(identities):
            raise ValueError("Postpass availability ownership overlaps")
        owned.update(identities)
        last_response = max((row["response_received_at"] for row in history if row.get("response_received_at")), default=doc["created_at"])
        stopped = runner.read(Path(doc["stopped_segment"]["path"]))
        for identity in identities:
            row = by_id[identity]
            metadata_path = path.parent / "admissions" / identity / "availability.json"
            meta = runner.read(metadata_path)
            original = runner.read(postpass.metadata_path("jev", identity))
            request = runner.read(Path(postpass.receipt(row)["request_journal_path"]))
            from datetime import datetime

            timeline = [datetime.fromisoformat(value) for value in (
                doc["created_at"], original["admitted_at"], meta["admitted_at"], request["created_at"],
                *([row["request_started_at"]] if row.get("request_started_at") else []))]
            if (any(value.tzinfo is None for value in timeline) or timeline != sorted(timeline)
                    or meta["admitted_at"] < last_response
                    or (timeline[2] - datetime.fromisoformat(stopped["finished_at"])).total_seconds() < 120
                    or not state["started_at"] <= meta["admitted_at"] <= state["finished_at"]):
                raise ValueError("Postpass availability chronological/serial admission changed")
            if (meta["amendment_id"] != doc["amendment_id"] or meta["amendment_sha256"] != state["amendment_sha256"]
                    or meta["adapter_sha256"] != state["driver_sha256"] or meta["allowlist_sha256"] != state["allowlist_sha256"]
                    or meta["attempt_id"] != identity or meta["request_sha256"] != row["request_sha256"]
                    or meta["effective_policy"] != availability.POLICY or meta["guard_evidence"] != availability.evidence(history, approval)):
                raise ValueError("Postpass availability per-admission guard evidence changed")
            liability = runner.liability(history, reserve)
            if (Decimal(meta["liability_before_usd"]) != liability or Decimal(original["liability_before_usd"]) != liability
                    or Decimal(meta["reserved_usd"]) != reserve or liability + reserve > Decimal(15)
                    or availability.stop_reason(postpass.stop_reason, history, approval, "jev", reserve) is not None):
                raise ValueError("Postpass availability guard/budget did not permit admission")
            history.append(row)
            last_response = row.get("response_received_at") or meta["admitted_at"]
            pins[str(metadata_path)] = contract.file_sha(metadata_path)
        selected = [by_id[identity] for identity in identities]
        if availability.receipt_prefix(selected) != state["new_receipts"]:
            raise ValueError("Postpass availability owned receipt changed")
        entries = [entry for entry in approval["allowlist"] if entry["route"] == "jev"]
        if state["status"] == "allowed_attempts_complete" and postpass.next_job(entries, history):
            raise ValueError("Postpass availability incorrectly claims complete allowance consumption")
        pins[str(path)] = contract.file_sha(path)
        pins[str(amendment_path)] = state["amendment_sha256"]
        pins.update(doc["helper_sha256"])
        audits.append({key: state[key] for key in ("status", "started_at", "finished_at", "amendment_id",
                       "effective_policy", "effective_concurrency", "attempt_ids")}
                      | {"verified_chronological_admissions": len(identities)})
    if required_ownership != owned:
        raise ValueError("Post-prefix Jev receipts lack exact availability ownership")
    return audits, pins


def report_fragment(groups: list[dict], postpass_report: dict | None = None, circuit_resets: list[dict] | None = None, availability_amendments: list[dict] | None = None, postpass_availability: list[dict] | None = None) -> str:
    lines = [
        "## Operational segments and provider failures", "",
        "Jev required an explicit provider-failure recovery. Its concurrency limit changed4→1→4 in separate documented segments: serial operation did not remove the observed503 failures and was substantially slower in the measured windows. DeepSeek retained billed no-answer token-limit failures and continued with unchanged concurrency8 and token/reasoning settings. A client-process interruption left21 request-only journals (Luna16,Liquid4,Jev1); they retain unknown original latency and cost. Subsequent execution uses durable user services. Single-request smokes are separate. These timings include upload, queueing and the complete response and are descriptive, not a controlled model-speed comparison. Failed labels remain null; known charges and unknown bills remain distinct. The original verified pre-serve capacity429 retry policy remains separate.", "",
        "| Model | Phase | Segment | Concurrency limit | Attempts | Valid | Invalid | Failures | Known503 | Exhaustions | p50 all(s) | p95 all(s) | p50 scored(s) | p95 scored(s) | Unpriced |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def display(value):
        return "—" if value is None else f"{value:.2f}"
    for row in groups:
        counts = row["status_counts"]
        failed = row["attempts"] - counts.get("valid", 0) - counts.get("invalid", 0)
        all_times, scored_times = row["all_attempt_latency"], row["completed_response_latency"]
        values = [row["model"], row["phase"], row["operational_segment_id"], row["effective_concurrency_limit"],
                  row["attempts"], counts.get("valid", 0), counts.get("invalid", 0), failed,
                  row["known_image_processing_503"], row["billed_no_answer_exhaustions"], display(all_times["p50_seconds"]), display(all_times["p95_seconds"]),
                  display(scored_times["p50_seconds"]), display(scored_times["p95_seconds"]), row["unpriced_attempts"]]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "The exact known503 class was quarantined under a prospective bounded policy: five consecutive failures or20 in the last100 requests stops further dispatch. Other error classes stop immediately. All unknown costs keep their conservative budget reservations; known subtotals are not complete billed totals. [Detailed operational audit](operations-summary.json).", ""])
    if circuit_resets:
        lines.extend(["Later Jev circuit resets are explicit operational amendments. Each requires at least120seconds of cooldown and a separate ten-request serial half-open batch using never-attempted jobs; a second exact503 stops that batch. Continuation requires all ten responses and at most one exact503. The original five-consecutive/20-of100 thresholds then apply to the documented new window, including the probes. Historical failures, labels and unknown-cost reservations remain intact. The frozen benchmark phase was prioritized for coverage during provider instability, before completing the rehearsal; score magnitudes did not determine this order.", "",
                      "| Stage | Status | Concurrency | New attempts | Exact503 | Prior observed responses retained |",
                      "|---|---|---:|---:|---:|---:|"])
        for state in circuit_resets:
            values = [state["stage"], state["status"], state["effective_concurrency"], len(state["attempt_ids"]),
                      state["known_image_503_count"], state["reset_observed_prefix_count"]]
            lines.append("| " + " | ".join(map(str, values)) + " |")
        lines.append("")
    if availability_amendments:
        lines.extend(["The remaining Jev first pass used a separately approved availability amendment after1,945 observed responses showed296 exact503s(15.22%), a longest failure run of4 and repeated20-of100 rolling trips despite prompt successful responses. It retained the five-consecutive stop, raised the rolling stop to30of100 and added a cumulative≥20% stop after100 observed responses using the complete initial history. Only availability tolerance changed: original request payloads, grading acceptance, budgets, unknown-cost reservations and the ban on replaying first-pass jobs remained intact. This later amendment does not alter the separately bounded postpass policy.", ""])
    if postpass_availability:
        lines.extend(["Serial recovery later reached the original20-of100 availability stop after236 additional Jev attempts produced196 fully scored recoveries and40 exact503s(16.95%). A separately pinned operational override retained serial dispatch, the original immutable allowance/job caps, protected-output rules and$15/full-reserve limits; it kept the five-consecutive stop, used30of100, and applied cumulative≥20% after100 across the entire observed postpass history. Five already exhausted cases remained missing. Each subsequent admission has separate amendment/source hashes, chronological guard evidence and a verified budget reservation; original admission receipts and the loader remain unchanged.", ""])
    if postpass_report:
        lines.extend([
            "After every initial API job had been attempted, a separate procedural amendment allowed bounded recovery of the exact Jev503 class (up to two additional attempts) and the21 interrupted requests (one additional attempt). Existing valid, partial or invalid scored responses were never replayed. Original unknown bills remain unknown, including possible charges for interrupted requests. DeepSeek no-answer token-limit failures remain missing. This recovery was an explicit later amendment, not part of the preregistered initial protocol.", "",
            "| Model | Eligible jobs | Completed responses obtained | Fully scored recoveries | Partial/invalid recoveries | Persistent without response | All jobs missing either score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for route, row in postpass_report["routes"].items():
            values = [route, row["eligible_jobs"], row.get("recovered", 0), row.get("fully_scored_recovered", 0),
                      row.get("partial_or_invalid_recovered", 0), row.get("persistent", 0), row.get("missing_jobs", 0)]
            lines.append("| " + " | ".join(map(str, values)) + " |")
        lines.extend(["", "Source, partition, category, operator and content-cluster missingness remain in the detailed recovery report. Recovered-response counts do not imply complete two-task labels; final accuracy uses task-level validity and reports coverage.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postpass-allowlist", type=Path)
    parser.add_argument("--postpass-allowlist-sha256")
    args = parser.parse_args()
    if bool(args.postpass_allowlist) != bool(args.postpass_allowlist_sha256):
        raise ValueError("Both postpass allowlist path and externally pinned SHA256 are required")
    postpass, approval = None, None
    if args.postpass_allowlist:
        import postpass_recovery as postpass

        approval = postpass.read_allowlist(args.postpass_allowlist, args.postpass_allowlist_sha256)
    launch = runner.read(HERE / "api-launch.json")
    if contract.file_sha(HERE / "api_route_driver.py") != launch["driver_sha256"]:
        raise ValueError("Initial operational driver source changed")
    source_hashes = {
        "jev": {contract.file_sha(path) for path in (HERE / "api_recovery.py", HERE / "pipeline-snapshots/api_recovery-v1.py", HERE / "pipeline-snapshots/api_recovery-v2.py", HERE / "pipeline-snapshots/api_recovery-v3.py")},
        "deepseek": {contract.file_sha(HERE / "deepseek_recovery.py")},
        "luna": {contract.file_sha(path) for path in (HERE / "interruption_resume.py", HERE / "pipeline-snapshots/interruption_resume-v1.py")},
        "liquid": {contract.file_sha(path) for path in (HERE / "interruption_resume.py", HERE / "pipeline-snapshots/interruption_resume-v1.py")},
    }
    all_rows, states, circuit_resets, availability_amendments, postpass_availability = [], {}, [], [], []
    pins = {str(HERE / "api-launch.json"): contract.file_sha(HERE / "api-launch.json")}
    if postpass:
        for route in ("luna", "liquid", "jev"):
            source_hashes[route].add(contract.file_sha(HERE / "postpass_recovery.py"))
        pins[str(args.postpass_allowlist.resolve())] = args.postpass_allowlist_sha256
        pins.update(approval["helper_sha256"])
        pins[approval["policy_path"]] = approval["policy_sha256"]
    audit_path = HERE / "api-client-interruption-audit.json"
    audit = runner.read(audit_path)
    pins[str(audit_path)] = contract.file_sha(audit_path)
    orphaned = {(row["route"], row["attempt_id"]): row for row in audit["interrupted_attempts"]}
    for route in runner.ROUTES:
        root = runner.directory(route)
        state_path = root / "driver-status.json"
        state = apply_interruption_audit(state_path, runner.read(state_path), audit)
        segment_paths = sorted((root / "recovery-segments").glob("*.json"))
        if postpass:
            segment_paths += sorted((postpass.root() / route / "segments").glob("*.json"))
        segments = [(path, runner.read(path)) for path in segment_paths]
        if state["status"] == "running":
            raise ValueError("Run operational audit after every API driver has stopped")
        records = []
        plans = []
        for phase in runner.PHASES:
            path = runner.directory(route, phase) / "plan.json"
            plan = runner.read(path)
            runner.validate_plan(plan)
            plans.append(plan)
            records.extend(postpass.reconstruct(plan, approval) if postpass else runner.reconstruct(plan))
            pins[str(path)] = contract.file_sha(path)
        if len({plan["route_config"]["concurrency"] for plan in plans}) != 1:
            raise ValueError("Frozen phase concurrency differs")
        for row in records:
            interrupted = orphaned.get((route, row["attempt_id"]))
            if interrupted:
                if not interruption_resume.approved_interruption(row, {row["attempt_id"]: interrupted}):
                    raise ValueError("Audited interrupted request no longer matches")
                row["_admission_journal_created_at"] = interrupted["request_journal_created_at"]
        segments = [(path, apply_interruption_audit(path, value, audit, records)) for path, value in segments]
        if any(s["status"] == "running" for _, s in segments):
            raise ValueError("Run operational audit after every recovery segment has stopped")
        states[route] = {"initial_driver_status": state["status"], "recovery_statuses": [s["status"] for _, s in segments]}
        for path in (root / "interruption-admissions").glob("*.json"):
            value = runner.read(path)
            if value["status"] == "running":
                raise ValueError("Interruption recovery is still running")
            admission_audit_path = Path(value["approval_path"])
            admission_audit = runner.read(admission_audit_path)
            if (value["driver_sha256"] not in source_hashes["luna"]
                    or value["approval_sha256"] != contract.file_sha(admission_audit_path)
                    or admission_audit["interrupted_attempts"] != audit["interrupted_attempts"]
                    or admission_audit["created_at"] != audit["created_at"]):
                raise ValueError("Interruption recovery source/audit changed")
            pins[str(path)] = contract.file_sha(path)
            pins[str(admission_audit_path)] = contract.file_sha(admission_audit_path)
        for path, segment in segments:
            segment_approval = Path(segment["approval_path"])
            if (contract.file_sha(segment_approval) != segment["approval_sha256"]
                    or segment["driver_sha256"] not in source_hashes.get(route, set())
                    or segment["route"] != route):
                raise ValueError("Operational segment approval/source provenance changed")
            pins[str(path)] = contract.file_sha(path)
            pins[str(segment_approval)] = contract.file_sha(segment_approval)
        if route == "jev":
            circuit_resets, circuit_pins = audit_circuit_resets(root, records)
            pins.update(circuit_pins)
            availability_amendments, availability_pins = audit_availability_amendments(root, records)
            pins.update(availability_pins)
            if postpass:
                launch_path = HERE / "postpass-availability-launch.json"
                expected_amendments = {}
                if launch_path.exists():
                    availability_launch = runner.read(launch_path)
                    expected_amendments = availability_launch["expected_amendments"]
                    pins[str(launch_path)] = contract.file_sha(launch_path)
                elif list(HERE.glob("postpass-availability-amendment-*.json")):
                    raise ValueError("Postpass availability launch evidence is missing")
                postpass_availability, postpass_availability_pins = audit_postpass_availability(records, approval, expected_amendments)
                pins.update(postpass_availability_pins)
        all_rows.extend(bind_segments(records, segments, launch["smoke_review"][route], plans[0]["route_config"]["concurrency"]))
    groups = summarize(all_rows)
    postpass_report = None
    if postpass:
        report_path = postpass.root() / "report.json"
        postpass_report = runner.read(report_path)
        if postpass_report["allowlist_id"] != approval["allowlist_id"]:
            raise ValueError("Postpass report belongs to a different allowlist")
        pins[str(report_path)] = contract.file_sha(report_path)
        for path in postpass.root().glob("*/attempt-metadata/**/admission.json"):
            pins[str(path)] = contract.file_sha(path)
    summary = {
        "created_at": runner.now(), "inputs": pins, "driver_states": states, "segments": groups,
        "attempts": len(all_rows), "known_image_processing_503": sum(r["known_image_processing_503"] for r in all_rows),
        "quarantined_under_prospective_policy": sum(r["prospectively_quarantined"] for r in all_rows),
        "billed_no_answer_exhaustions": sum(r["billed_no_answer_exhaustion"] for r in all_rows),
        "audited_interrupted_requests": len(orphaned),
        "postpass_recovery": postpass_report, "circuit_resets": circuit_resets, "availability_amendments": availability_amendments, "postpass_availability_amendments": postpass_availability,
        "helper_sha256": contract.file_sha(__file__),
        "latency_definition": "HTTP invocation through complete nonstreaming response; excludes local preparation and scheduling",
        "interpretation": "Descriptive operational segments; no causal speed comparison or imputation of missing labels/costs",
    }
    sidecar = [{k: row.get(k) for k in ("model", "phase", "attempt_id", "case_id", "arm", "status", "latency_seconds",
               "price_usd", "operational_segment_id", "effective_concurrency_limit", "frozen_plan_concurrency_limit",
               "known_image_processing_503", "prospectively_quarantined", "billed_no_answer_exhaustion", "raw_response_sha256")} for row in all_rows]
    runner.write(HERE / "operations-summary.json", summary)
    runner.jsonl(HERE / "operations-attempt-segments.jsonl", sidecar)
    (HERE / "operations-report.md").write_text(report_fragment(groups, postpass_report, circuit_resets, availability_amendments, postpass_availability))
    print(json.dumps({"attempts": len(all_rows), "segments": len(groups), "known503": summary["known_image_processing_503"]}))


if __name__ == "__main__":
    main()
