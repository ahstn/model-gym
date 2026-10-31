"""Read-only OpenRouter generation metadata audit; never performs inference."""

import concurrent.futures
import json
import os
from decimal import Decimal
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/dual-score-trial-2026-10-27"


def fetch(item):
    path, route, phase = item
    key = os.environ["OPENROUTER_API_KEY"]
    body = json.loads(json.loads(path.read_text())["body"])
    generation_id = body.get("id")
    if not generation_id:
        return {"route": route, "phase": phase, "case_id": path.parent.name, "status": "no_generation_id"}
    target = DATA / route / phase / "generation-audit" / (path.parent.name + ".json")
    if target.exists():
        result = json.loads(target.read_text())
    else:
        response = requests.get(
            "https://openrouter.ai/api/v1/generation",
            params={"id": generation_id},
            headers={"Authorization": "Bearer " + key},
            timeout=(15, 60),
            allow_redirects=False,
        )
        result = {"http_status": response.status_code, "body": response.text.replace(key, "[REDACTED]")}
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
    details = json.loads(result["body"]).get("data", {})
    reported = body.get("usage", {}).get("cost")
    settled = details.get("total_cost")
    matches = (
        None
        if reported is None or settled is None
        else abs(Decimal(str(reported)) - Decimal(str(settled))) <= Decimal("0.000000001")
    )
    return {
        "route": route,
        "phase": phase,
        "case_id": path.parent.name,
        "generation_id": generation_id,
        "http_status": result["http_status"],
        "model": details.get("model"),
        "provider": details.get("provider_name"),
        "reported_cost": reported,
        "settled_cost": settled,
        "cost_matches": matches,
        "reasoning_tokens": details.get("native_tokens_reasoning"),
        "finish_reason": details.get("finish_reason"),
    }


def main():
    items = [
        (p, route, phase)
        for route in ("gemini", "luna")
        for phase in ("probes", "images")
        for p in sorted((DATA / route / phase / "attempts").glob("*/result.json"))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, items))
    (HERE / "usage-audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            {
                "lookups": len(rows),
                "cost_matches": sum(r.get("cost_matches") is True for r in rows),
                "providers": sorted({str(r.get("provider")) for r in rows}),
            }
        )
    )


if __name__ == "__main__":
    main()
