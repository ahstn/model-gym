"""Supplemental read-only reconciliation; does not alter or execute frozen runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/prompt-ablation-2026-10-30"
ROUTES = ("deepseek", "gemini", "luna", "haiku")
ARMS = ("baseline", "A", "B", "C", "D", "combined")
TASKS = ("technical_quality", "aesthetics")


def snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def amount(value):
    if value is None or isinstance(value, bool):
        raise ValueError("Expected a recorded nonnegative cost")
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid recorded cost")
    return value


def costs(attempts):
    def total(key):
        return str(sum((amount(a[key]) for a in attempts if a.get(key) is not None), Decimal(0)))

    return {
        "known_cost_usd": total("cost_usd"),
        "gateway_cost_usd": total("gateway_cost_usd"),
        "upstream_inference_cost_usd": total("upstream_inference_cost_usd"),
        "unpriced_attempts": sum(a.get("cost_usd") is None for a in attempts),
        "unpriced_attempt_ids": [a["attempt_id"] for a in attempts if a.get("cost_usd") is None],
        "cost_source_counts": dict(Counter(a.get("cost_source", "unavailable") for a in attempts)),
        "confirmed_byok_attempts": sum(a.get("is_byok") is True for a in attempts),
        "conservative_unconfirmed_cost_usd": str(
            sum(
                (
                    amount(a["cost_usd"])
                    for a in attempts
                    if a.get("cost_usd") is not None
                    and str(a.get("cost_source", "")).startswith("conservative")
                ),
                Decimal(0),
            )
        ),
        "note": "Use known_cost_usd for the runner's recorded cost basis. Gateway and upstream amounts overlap for non-BYOK calls and must not be blindly added. Conservative unconfirmed amounts are identified separately.",
    }


def coverage(cases, attempts, journaled):
    planned = {c["attempt_id"] for c in cases}
    by_id = {a["attempt_id"]: a for a in attempts}
    if len(planned) != len(cases) or len(by_id) != len(attempts):
        raise ValueError("Duplicate planned or recorded attempt")
    if not set(by_id) <= journaled <= planned:
        raise ValueError("Attempt journal, report and plan disagree")
    statuses = Counter(a["status"] for a in attempts)
    pending = sorted(journaled - set(by_id))
    return {
        "planned": len(cases),
        "attempted_in_report": len(attempts),
        "journaled_requests": len(journaled),
        "in_flight_or_unreconciled": len(pending),
        "in_flight_or_unreconciled_ids": pending,
        "unsent": len(planned - journaled),
        "unsent_attempt_ids": sorted(planned - journaled),
        "valid_joint": statuses["valid"],
        "invalid_rating": statuses["invalid"],
        "failed_request": statuses["failed"],
        "unknown_request": statuses["unknown"],
        "task_valid": {t: sum(a["task_valid"][t] for a in attempts) for t in TASKS},
        "task_invalid_rating": {
            t: sum(a["status"] == "invalid" and not a["task_valid"][t] for a in attempts) for t in TASKS
        },
        "task_without_valid_result": {t: sum(not a["task_valid"][t] for a in attempts) for t in TASKS},
        "statuses": dict(statuses),
        "note": "Journaled requests are durably admitted; an interrupted process may not have sent them. Unreconciled entries can be in flight or completed after the saved report snapshot. They are never called unsent.",
    }


def phase(folder):
    if not (folder / "run.json").exists():
        return {"status": "not_started", "costs": costs([]), "coverage": None, "by_arm": {}, "attempts": []}
    run, run_source = snapshot(folder / "run.json")
    digest = hashlib.sha256(
        json.dumps({k: v for k, v in run.items() if k != "run_id"}, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != run["run_id"]:
        raise ValueError("Run identity mismatch: " + str(folder))
    report_path = folder / "report.json"
    if report_path.exists():
        report, report_source = snapshot(report_path)
        if report["run_id"] != run["run_id"]:
            raise ValueError("Report/run identity mismatch")
    else:
        report, report_source = {"status": "awaiting_report", "attempts": []}, None
    attempts = report["attempts"]
    summary_costs = costs(attempts)
    if "known_cost_usd" in report and amount(report["known_cost_usd"]) != amount(
        summary_costs["known_cost_usd"]
    ):
        raise ValueError("Attempt costs do not reconcile with saved report")
    if "attempted" in report and report["attempted"] != len(attempts):
        raise ValueError("Attempt count differs from saved report")
    cases = run["cases"]
    expected = {c["attempt_id"]: c for c in cases}
    journaled, provenance = set(), []
    root = folder / "attempts"
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.name not in expected:
            raise ValueError("Unexpected attempt directory: " + str(path))
        journaled.add(path.name)
        entry = {"attempt_id": path.name}
        request_path = path / "request.json"
        if request_path.exists():
            request, entry["request"] = snapshot(request_path)
            if request["run_id"] != run["run_id"] or any(
                request.get(k) != expected[path.name].get(k)
                for k in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")
            ):
                raise ValueError("Request metadata differs from plan")
        else:
            entry["request"] = {"status": "journal_directory_not_yet_published"}
        result_path = path / "result.json"
        if result_path.exists():
            _, entry["result"] = snapshot(result_path)
        provenance.append(entry)
    observed_coverage = coverage(cases, attempts, journaled)
    status = report["status"]
    if observed_coverage["in_flight_or_unreconciled"] and status != "blocked":
        status = "partial_with_unreconciled_journals"
    return {
        "status": status,
        "reported_status": report["status"],
        "run_id": run["run_id"],
        "costs": summary_costs,
        "coverage": observed_coverage,
        "by_arm": {
            arm: coverage(
                [c for c in cases if c["arm_id"] == arm],
                [a for a in attempts if a["arm_id"] == arm],
                {k for k in journaled if expected[k]["arm_id"] == arm},
            )
            for arm in ARMS
            if any(c["arm_id"] == arm for c in cases)
        },
        "attempts": attempts,
        "attempt_artifacts": provenance,
        "run_source": run_source,
        "report_source": report_source,
        "ignored_cumulative_total_known_cost_usd": report.get("total_known_cost_usd"),
        "note": "Only per-phase known_cost_usd is summed. Image report cumulative totals already include current probes and are intentionally not summed again.",
    }


def archive_hashes(archive):
    evidence = []
    for original, item in sorted(archive["inputs"].items()):
        path = Path(item["archived_copy"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence.append(
            {
                "original_path": original,
                "archived_path": str(path),
                "expected_sha256": item["sha256"],
                "observed_sha256": actual,
                "matches": actual == item["sha256"],
            }
        )
    if not all(row["matches"] for row in evidence):
        raise ValueError("Archived method input hash mismatch")
    return {"status": "verified", "files": len(evidence), "evidence": evidence}


def reconcile():
    started = datetime.now(UTC).isoformat()
    archive, archive_source = snapshot(HERE / "preflight-v1/archive.json")
    verification = archive_hashes(archive)
    routes = {}
    all_phases = []
    for route in ROUTES:
        archived = phase(HERE / "preflight-v1/runs" / route / "probes")
        saved = archive["routes"][route]
        if archived["run_id"] != saved["run_id"] or amount(archived["costs"]["known_cost_usd"]) != amount(
            saved["known_cost_usd"]
        ):
            raise ValueError("Archive index differs from archived run")
        probes, images = phase(DATA / route / "probes"), phase(DATA / route / "images")
        stages = {"archived_preflight": archived, "current_probes": probes, "images": images}
        all_phases.extend(stages.values())
        known = sum((amount(p["costs"]["known_cost_usd"]) for p in stages.values()), Decimal(0))
        reserve = amount(saved["admission_debit_usd"]) - amount(saved["known_cost_usd"])
        if reserve < 0:
            raise ValueError("Archive reserve cannot be negative")
        routes[route] = {
            "status": images["status"],
            "stages": stages,
            "known_cost_usd": str(known),
            "archived_unpriced_reserve_usd": str(reserve),
            "known_plus_archived_reserve_usd": str(known + reserve),
            "unpriced_attempts": sum(p["costs"]["unpriced_attempts"] for p in stages.values()),
            "route_cap_usd": "5.00",
            "balance_before_unreconciled_request_reserves_usd": str(Decimal("5.00") - known - reserve),
        }
    totals = costs([a for p in all_phases for a in p["attempts"]])
    # Attempt IDs are scoped to a run; retain full run identity for every unknown charge.
    totals["unpriced_attempts_provenance"] = [
        {
            "route": route,
            "stage": name,
            "run_id": p["run_id"],
            "attempt_id": a["attempt_id"],
            "reason": a.get("reason"),
        }
        for route, r in routes.items()
        for name, p in r["stages"].items()
        for a in p["attempts"]
        if a.get("cost_usd") is None
    ]
    total_reserve = sum((amount(r["archived_unpriced_reserve_usd"]) for r in routes.values()), Decimal(0))
    totals.update(
        archived_unpriced_reserve_usd=str(total_reserve),
        known_plus_archived_reserve_usd=str(amount(totals["known_cost_usd"]) + total_reserve),
        overall_cap_usd="20.00",
    )
    return {
        "schema_version": "prompt-ablation-supplemental-accounting-v1",
        "snapshot_started_at": started,
        "snapshot_finished_at": datetime.now(UTC).isoformat(),
        "status": "image_runs_complete"
        if all(r["status"] == "complete" for r in routes.values())
        else "partial_or_blocked",
        "totals": totals,
        "routes": routes,
        "archive_source": archive_source,
        "archive_hash_verification": verification,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Supplemental artifact; frozen method and run files remain unchanged. A live snapshot is not atomic across routes. Re-run after every route completes or stops for final accounting.",
        "reserve_note": "The archived unpriced request reserve is a conservative admission debit, not an observed charge. Actual total spend remains unknown while any request is unpriced. Pending journals may contain charges not yet included in saved reports.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "accounting-summary.json")
    args = parser.parse_args()
    result = reconcile()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"], "totals": result["totals"]}))


if __name__ == "__main__":
    main()
