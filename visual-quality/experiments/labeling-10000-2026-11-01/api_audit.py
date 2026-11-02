"""Audit API coverage/provenance and costs without displaying benchmark score values."""

import json
from collections import Counter
from decimal import Decimal

import api_runner as runner


def model_audit(model, manifest):
    root = runner.DATA / "results/api" / model
    states = {path.stem: runner.read(path) for path in root.glob("*-status.json")}
    rows = [runner.read(path) for path in (root / "observations").glob("*.json")]
    statuses = Counter(row["status"] for row in rows)
    numeric = sum(
        all(isinstance(row.get(key), (int, float)) for key in ("technical_score", "aesthetic_score"))
        for row in rows
    )
    failures, problems = [], []
    for row in rows:
        case = manifest.get(row["case_id"])
        if case is None or row.get("view_sha256") != case["view_sha256"]:
            problems.append({"case_id": row["case_id"], "problem": "manifest image mismatch"})
        if row.get("status") != "valid":
            failures.append(
                {
                    key: row.get(key)
                    for key in ("case_id", "status", "reason", "validation_errors", "attempt_count")
                }
            )
    known = Decimal(0)
    unknown = pending = attempts = 0
    transports, http_statuses, result_statuses = Counter(), Counter(), Counter()
    for request in root.glob("attempts/*/*/request.json"):
        attempts += 1
        result = request.with_name("result.json")
        if not result.exists():
            pending += 1
            continue
        raw = runner.read(result)
        if raw.get("price_usd") is None:
            unknown += 1
        else:
            known += Decimal(raw["price_usd"])
        http_statuses[str(raw.get("http_status"))] += 1
        if raw.get("transport_error"):
            transports[raw["transport_error"]] += 1
        result_statuses[raw["parsed"]["status"]] += 1
    return {
        "observations": len(rows),
        "both_numeric": numeric,
        "status_counts": dict(statuses),
        "missing_observations": len(set(manifest) - {r["case_id"] for r in rows}),
        "known_price_usd": str(known),
        "unpriced_completed_attempts": unknown,
        "pending_or_interrupted_requests": pending,
        "http_attempts": attempts,
        "http_statuses": dict(http_statuses),
        "transport_errors": dict(transports),
        "attempt_statuses": dict(result_statuses),
        "failures": failures,
        "join_problems": problems,
        "phases": {
            name: {k: state.get(k) for k in ("status", "outcomes", "updated_at", "stop_reason", "plan_id")}
            for name, state in states.items()
        },
    }


def main():
    manifest = {row["case_id"]: row for row in runner.read(runner.DATA / "manifest.json")["cases"]}
    result = {
        "checked_at": runner.now(),
        "manifest_sha256": runner.sha(runner.DATA / "manifest.json"),
        "models": {model: model_audit(model, manifest) for model in runner.MODELS},
    }
    runner.atomic(runner.DATA / "api-audit.json", result)
    summary = {
        model: {
            key: row[key]
            for key in (
                "observations",
                "both_numeric",
                "status_counts",
                "known_price_usd",
                "unpriced_completed_attempts",
                "pending_or_interrupted_requests",
                "join_problems",
            )
        }
        for model, row in result["models"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
