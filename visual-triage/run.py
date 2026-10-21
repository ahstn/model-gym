"""Run a bounded, resumable comparison. Every response and failure is retained."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from evidence_archive import require_restored
from inference import (
    CHAT,
    DECISIONS,
    ROOT,
    chat_payload,
    decision_payload,
    normalize_chat,
    normalize_decisions,
    post,
    shadow_route,
)

MODELS = {
    "clef": "cloudflare/clef-flash",
    "decider": "perplexity/pplx-decider-v1.1-27b",
    "vision": "google/gemini-2.5-flash-lite",
    "jev": "typesafe/jev-1.13",
}


def signature():
    paths = sorted(path for path in (ROOT / "prompts").glob("*") if path.is_file()) + [
        ROOT / "evidence/samples.json",
        ROOT / "inference.py",
    ]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths) + json.dumps(MODELS).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--arms", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/run.jsonl")
    parser.add_argument("--max-calls", type=int, default=125)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--clef-provider", choices=["Cloudflare", "PrimeIntellect"])
    args = parser.parse_args()
    require_restored(ROOT)
    if args.limit < 1 or args.max_calls < 1 or not 0 < args.budget_usd <= 10:
        parser.error("Use positive limits and a budget no larger than $10 for this diagnostic runner")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is absent. Load your normal environment; never pass it as an argument."
        )
    samples = json.loads((ROOT / "evidence/samples.json").read_text())["samples"][: args.limit]
    past = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    run_sig = signature()
    if args.clef_provider:
        run_sig = hashlib.sha256((run_sig + args.clef_provider).encode()).hexdigest()
    if any(r["signature"] != run_sig for r in past):
        raise SystemExit("Inputs changed; use a new output path to retain the earlier run.")
    done = {(r["sample"], r["arm"]): r for r in past}
    spent = sum(r.get("cost_usd") or 0 for r in past)
    if any(r.get("cost_usd") is None for r in past):
        raise SystemExit("An earlier request has unknown cost; reconcile it before resuming.")
    calls = len(past)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        for arm in args.arms:
            key = (sample["id"], arm)
            if key in done:
                continue
            # Reserve 5 cents per next request; this is a local guard, not an account spending limit.
            if calls >= args.max_calls or spent + 0.05 > args.budget_usd:
                raise SystemExit("Local request or spend guard reached")
            if arm == "jev":
                upstream = done.get((sample["id"], "vision"), {}).get("normalized")
                if upstream is None:
                    record = {
                        "sample": sample["id"],
                        "arm": arm,
                        "signature": run_sig,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "requested_model": MODELS[arm],
                        "status": None,
                        "seconds": 0,
                        "cost_usd": 0,
                        "normalized": None,
                        "response": {},
                        "validation_error": "Upstream vision observation failed validation; no request sent",
                        "shadow_route": "review_error",
                        "skipped": True,
                    }
                    with args.output.open("a") as out:
                        out.write(json.dumps(record) + "\n")
                    done[key] = record
                    print(sample["id"], arm, "skipped: invalid upstream observation", flush=True)
                    continue
                payload = decision_payload(MODELS[arm], description=upstream["description"])
            elif arm == "vision":
                payload = chat_payload(MODELS[arm], sample)
            else:
                payload = decision_payload(MODELS[arm], sample)
            if arm == "clef" and args.clef_provider:
                payload["provider"] = {"only": [args.clef_provider], "allow_fallbacks": False}
            record = post(CHAT if arm == "vision" else DECISIONS, payload)
            record.update(
                sample=sample["id"], arm=arm, signature=run_sig, timestamp=datetime.now(UTC).isoformat()
            )
            response = record.get("response", {})
            record["cost_usd"] = response.get("usage", {}).get("cost")
            try:
                if record.get("status") != 200:
                    raise ValueError("HTTP failure")
                normalizer = normalize_chat if arm == "vision" else normalize_decisions
                record["normalized"] = normalizer(response)
            except (ValueError, KeyError, TypeError) as exc:
                record["validation_error"] = type(exc).__name__ + ": " + str(exc)
                record["normalized"] = None
            record["shadow_route"] = shadow_route(record["normalized"])
            with args.output.open("a") as out:
                out.write(json.dumps(record) + "\n")
            done[key] = record
            calls += 1
            print(
                sample["id"],
                arm,
                record.get("status"),
                record["shadow_route"],
                f"{record['seconds']:.3f}s",
                record["cost_usd"],
                flush=True,
            )
            if record["cost_usd"] is None:
                raise SystemExit("Unknown request cost; stop before further spending")
            spent += record["cost_usd"]
    print(f"Total recorded requests: {calls}; reported cost: ${spent:.6f}")


if __name__ == "__main__":
    main()
