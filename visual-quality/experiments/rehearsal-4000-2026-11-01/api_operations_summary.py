"""Audit API operational segments without changing scoring or receipt journals."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import api_contract as contract
import api_recovery
import api_runner as runner
import deepseek_recovery
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
        if segment["status"] == "running" or not segment.get("finished_at"):
            raise ValueError("Operational segment still running or incomplete")
        ids = segment["attempt_ids"]
        quarantined = set(segment.get("quarantined_attempt_ids", []))
        if len(set(ids)) != len(ids) or not quarantined <= set(ids):
            raise ValueError("Duplicate or extraneous segment/quarantine membership")
        for identity in ids:
            if identity not in by_id or identity in mapping:
                raise ValueError("Missing attempt or overlapping operational segments")
            row = by_id[identity]
            if not segment["started_at"] <= row["request_started_at"] <= segment["finished_at"]:
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


def report_fragment(groups: list[dict]) -> str:
    lines = [
        "## Operational segments and provider failures", "",
        "Jev required an explicit provider-failure recovery. Its original concurrency limit was4; later segments used1. DeepSeek retained billed no-answer token-limit failures and continued with unchanged concurrency8 and token/reasoning settings. Single-request smokes are separate. These timings include upload, queueing and the complete response and are descriptive, not a controlled model-speed comparison. Failed labels remain null; known charges and unknown bills remain distinct. No completed failure was replayed; the original verified pre-serve capacity429 retry policy remains separate.", "",
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
    return "\n".join(lines)


def main() -> None:
    launch = runner.read(HERE / "api-launch.json")
    if contract.file_sha(HERE / "api_route_driver.py") != launch["driver_sha256"]:
        raise ValueError("Initial operational driver source changed")
    source_hashes = {
        "jev": {contract.file_sha(path) for path in (HERE / "api_recovery.py", HERE / "pipeline-snapshots/api_recovery-v1.py")},
        "deepseek": {contract.file_sha(HERE / "deepseek_recovery.py")},
    }
    all_rows, states = [], {}
    pins = {str(HERE / "api-launch.json"): contract.file_sha(HERE / "api-launch.json")}
    for route in runner.ROUTES:
        root = runner.directory(route)
        state = runner.read(root / "driver-status.json")
        segment_paths = sorted((root / "recovery-segments").glob("*.json"))
        segments = [(path, runner.read(path)) for path in segment_paths]
        if state["status"] == "running" or any(s["status"] == "running" for _, s in segments):
            raise ValueError("Run operational audit after every API driver has stopped")
        states[route] = {"initial_driver_status": state["status"], "recovery_statuses": [s["status"] for _, s in segments]}
        records = []
        plans = []
        for phase in runner.PHASES:
            path = runner.directory(route, phase) / "plan.json"
            plan = runner.read(path)
            runner.validate_plan(plan)
            plans.append(plan)
            records.extend(runner.reconstruct(plan))
            pins[str(path)] = contract.file_sha(path)
        if len({plan["route_config"]["concurrency"] for plan in plans}) != 1:
            raise ValueError("Frozen phase concurrency differs")
        for path, segment in segments:
            approval = Path(segment["approval_path"])
            if (contract.file_sha(approval) != segment["approval_sha256"]
                    or segment["driver_sha256"] not in source_hashes.get(route, set())
                    or segment["route"] != route):
                raise ValueError("Operational segment approval/source provenance changed")
            pins[str(path)] = contract.file_sha(path)
            pins[str(approval)] = contract.file_sha(approval)
        all_rows.extend(bind_segments(records, segments, launch["smoke_review"][route], plans[0]["route_config"]["concurrency"]))
    groups = summarize(all_rows)
    summary = {
        "created_at": runner.now(), "inputs": pins, "driver_states": states, "segments": groups,
        "attempts": len(all_rows), "known_image_processing_503": sum(r["known_image_processing_503"] for r in all_rows),
        "quarantined_under_prospective_policy": sum(r["prospectively_quarantined"] for r in all_rows),
        "billed_no_answer_exhaustions": sum(r["billed_no_answer_exhaustion"] for r in all_rows),
        "helper_sha256": contract.file_sha(__file__),
        "latency_definition": "HTTP invocation through complete nonstreaming response; excludes local preparation and scheduling",
        "interpretation": "Descriptive operational segments; no causal speed comparison or imputation of missing labels/costs",
    }
    sidecar = [{k: row.get(k) for k in ("model", "phase", "attempt_id", "case_id", "arm", "status", "latency_seconds",
               "price_usd", "operational_segment_id", "effective_concurrency_limit", "frozen_plan_concurrency_limit",
               "known_image_processing_503", "prospectively_quarantined", "billed_no_answer_exhaustion", "raw_response_sha256")} for row in all_rows]
    runner.write(HERE / "operations-summary.json", summary)
    runner.jsonl(HERE / "operations-attempt-segments.jsonl", sidecar)
    (HERE / "operations-report.md").write_text(report_fragment(groups))
    print(json.dumps({"attempts": len(all_rows), "segments": len(groups), "known503": summary["known_image_processing_503"]}))


if __name__ == "__main__":
    main()
