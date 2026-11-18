# ruff: noqa: I001
"""Score the new 20K images with the unchanged evaluated four-model profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

# This scorer owns a fixed run context. Importing corpus would also configure
# historical acquisition globals, although no acquisition code is needed here.
PACKAGE = Path(__file__).resolve().parents[3]
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-40000-2026-11-17")
sys.path[:0] = [str(PACKAGE / "src"), str(PACKAGE / "experiments/labeling-10000-2026-11-01")]
import api_runner as adapter
from visual_quality.labeling_profile import MODELS
from visual_quality.scoring import batch

PHASE = "expansion"
POLICY = {
    model: {
        **adapter.POLICY[model],
        "workers": 16,
        "budget_usd": budget,
        "content_denials": "terminal missing; no retry or payload modification",
    }
    for model, budget in (("luna", "80.00"), ("deepseek", "80.00"), ("liquid", "5.00"), ("jev", "20.00"))
}


def preflight():
    """Read provider catalogs without inference and save the current snapshot."""
    import requests

    from visual_quality.labeling_profile import load_profile

    profile = load_profile(PACKAGE / "labeling")
    needed = {route["key_env"] for route in profile["routes"]["routes"].values()}
    missing = sorted(name for name in needed if not os.environ.get(name))
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(missing))
    target = DATA / "api-provider-snapshot.json"
    if target.exists():
        raise ValueError("Provider snapshot exists; preserve it for frozen plans")
    records = {}
    for model in ("luna", "liquid", "jev"):
        route = profile["routes"]["routes"][model]
        url = f"https://openrouter.ai/api/v1/models/{route['requested_model']}/endpoints"
        response = requests.get(url, timeout=30, allow_redirects=False)
        response.raise_for_status()
        body = response.json()
        providers = set(route.get("accepted_providers", []))
        if not any(e.get("provider_name") in providers for e in body["data"]["endpoints"]):
            raise ValueError("Required provider unavailable: " + model)
        records[url] = body
    url = "https://api.runinfra.ai/v1/models"
    response = requests.get(
        url,
        headers={"Authorization": "Bearer " + os.environ["RUN_INFRA_API_KEY"]},
        timeout=30,
        allow_redirects=False,
    )
    response.raise_for_status()
    rows = [row for row in response.json()["data"] if row["id"] == "deepseek-v4-1-flash-1017"]
    if not rows:
        raise ValueError("Requested DeepSeek model unavailable")
    records[url] = {"data": rows}
    adapter.atomic(target, {"checked_at": adapter.now(), "sources": records})
    return {"provider_snapshot": str(target), "sha256": adapter.sha(target), "models": list(MODELS)}


def freeze_policy(args):
    """Recovery is a new phase, never a mutation of a frozen execution plan."""
    policy = dict(POLICY[args.model])
    if args.max_attempts is not None or args.budget_usd is not None:
        if args.phase == PHASE:
            raise ValueError("Policy overrides require a separately named phase")
        if args.max_attempts is not None:
            if not 1 <= args.max_attempts <= 10:
                raise ValueError("Maximum attempts must be between 1 and 10")
            policy["max_attempts"] = args.max_attempts
        if args.budget_usd is not None:
            try:
                budget = Decimal(args.budget_usd)
            except InvalidOperation as error:
                raise ValueError("Budget must be finite and positive") from error
            if not budget.is_finite() or budget <= 0:
                raise ValueError("Budget must be finite and positive")
            policy["budget_usd"] = str(budget)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--phase", default=PHASE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, help="Lower concurrency without changing the frozen plan")
    parser.add_argument(
        "--max-attempts", type=int, help="Freeze only: total attempts, including earlier phases"
    )
    parser.add_argument(
        "--budget-usd", help="Freeze only: conservative route budget, including earlier phases"
    )
    args = parser.parse_args()
    if args.command != "freeze" and (args.max_attempts is not None or args.budget_usd is not None):
        parser.error("Policy overrides are only accepted when freezing a new phase")
    if args.command == "preflight":
        result = preflight()
    else:
        if not args.model:
            parser.error("--model is required for freeze/run")
        options = SimpleNamespace(
            data=DATA,
            model=args.model,
            phase=args.phase,
            manifest=DATA / "manifest.json",
            profile_directory=PACKAGE / "labeling",
            limit=args.limit,
            worker_limit=args.workers,
        )
        result = (
            batch.freeze(options, adapter, freeze_policy(args), [Path(__file__)])
            if args.command == "freeze"
            else batch.run(options, adapter)
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
