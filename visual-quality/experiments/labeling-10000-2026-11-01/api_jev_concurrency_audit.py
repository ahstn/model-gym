"""Compare the first 200 new Jev cases after the approved worker-count adjustment."""

import json
from datetime import datetime

import api_runner as runner


def audit():
    data = runner.DATA
    frozen = data / "api-jev-concurrency-comparison.json"
    if frozen.exists():
        return runner.read(frozen)
    amendment_path = data / "api-operational-amendment-04.json"
    amendment = runner.read(amendment_path)
    previous = set(amendment["initial_case_ids_with_requests"])
    root = data / "results/api/jev"
    starts = [
        runner.read(path)
        for path in root.glob("attempts/*/01/request.json")
        if path.parent.parent.name not in previous
    ]
    starts.sort(key=lambda row: row["created_at"])
    starts = starts[: amendment["comparison_case_count"]]
    rows = [
        runner.read(root / "observations" / (start["case_id"] + ".json"))
        for start in starts
        if (root / "observations" / (start["case_id"] + ".json")).exists()
    ]
    results = [
        path for row in rows for path in sorted((root / "attempts" / row["case_id"]).glob("*/result.json"))
    ]
    attempts = [runner.read(path) for path in results]
    count = len(rows)
    share = sum(row.get("http_status") == 503 for row in attempts) / len(attempts) if attempts else 0
    failed = sum(row["status"] != "valid" for row in rows) / count if count else 0
    times = [datetime.fromisoformat(row["response_received_at"]) for row in rows]
    elapsed = (max(times) - min(times)).total_seconds() if count > 1 else 0
    complete = count == amendment["comparison_case_count"]
    rollback = complete and (share - amendment["baseline"]["http_503_share"] > 0.10 or failed > 0.02)
    result = {
        "checked_at": runner.now(),
        "sampling": "First 200 newly started cases, waiting for all their terminal outcomes; slow failures are retained",
        "amendment_sha256": runner.sha(amendment_path),
        "complete": complete,
        "case_count": count,
        "http_attempt_count": len(attempts),
        "http_503_share": share,
        "http_503_share_change": share - amendment["baseline"]["http_503_share"],
        "terminal_failure_rate": failed,
        "completion_window_seconds": elapsed,
        "completions_per_minute": (count - 1) * 60 / elapsed if elapsed else None,
        "baseline": {k: v for k, v in amendment["baseline"].items() if k != "case_ids"},
        "rollback_required": rollback,
        "case_ids": [row["case_id"] for row in rows],
        "attempt_receipts": {str(path): runner.sha(path) for path in results},
        "interpretation": "Operational before/after observation; time and different images may affect the comparison",
    }
    runner.atomic(frozen if complete else data / "api-jev-concurrency-progress.json", result)
    return result


if __name__ == "__main__":
    print(
        json.dumps({k: v for k, v in audit().items() if k not in {"case_ids", "attempt_receipts"}}, indent=2)
    )
