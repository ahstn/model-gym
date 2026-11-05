"""Run bounded, sequential image benchmarks with an append-only request ledger."""

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from bench_protocol import (
    MODELS,
    ROOT,
    digest,
    image_part,
    make_request,
    normalize_chat,
    normalize_decisions,
    number,
    validate_prompt,
)
from evidence_archive import require_restored


def post(endpoint, payload):
    """One HTTP attempt; retain no request image, authorization header or secret."""
    started = time.perf_counter()
    record = {"endpoint": endpoint, "requested_model": payload["model"]}
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                "Content-Type": "application/json",
                "X-Title": "model-gym visual-triage benchmark",
            },
            timeout=(15, 180),
            allow_redirects=False,
        )
        record["status"] = response.status_code
        # Keep malformed non-finite constants as strings so the ledger stays valid JSON.
        record["response"] = response.json(parse_constant=lambda value: value)
    except (requests.RequestException, ValueError) as exc:
        # Exceptions may include request details; only retain their class name.
        record["transport_error"] = type(exc).__name__
    record["seconds"] = time.perf_counter() - started
    return record


def effective_cost(usage, fallback_charge=None):
    """Count BYOK upstream cost for the guard, assuming any router charge is separate."""
    charge = usage.get("cost", fallback_charge)
    if charge is None:
        return None, "unknown_openrouter_cost"
    charge = number(charge, 0, float("inf"))
    byok = usage.get("is_byok", False)
    if type(byok) is not bool:
        raise ValueError("Invalid BYOK cost flag")
    if not byok:
        return charge, "openrouter_usage_cost"
    details = usage.get("cost_details")
    upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
    if upstream is None:
        return None, "unknown_byok_upstream_cost"
    upstream = number(upstream, 0, float("inf"))
    return charge + upstream, "byok_upstream_plus_openrouter_charge"


def response_record(record, arm, prompt, effort):
    response = record.get("response", {})
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    record.update(
        resolved_model=response.get("model"),
        provider=response.get("provider"),
        generation_id=response.get("id"),
        usage=usage,
        cost_usd=None,
        effective_cost_usd=None,
        cost_basis="unknown_openrouter_cost",
        requested_reasoning=None if arm == "decider" else {"effort": effort},
        # Absence means unreported, not proof that reasoning was disabled.
        observed_reasoning={
            "effort": response.get("reasoning", {}).get("effort")
            if isinstance(response.get("reasoning"), dict)
            else None,
            "completion_tokens_details": usage.get("completion_tokens_details"),
            "output_tokens_details": usage.get("output_tokens_details"),
        },
    )
    try:
        if usage.get("cost") is not None:
            record["cost_usd"] = number(usage["cost"], 0, float("inf"))
        record["effective_cost_usd"], record["cost_basis"] = effective_cost(usage)
    except (TypeError, ValueError):
        record["cost_validation_error"] = "Invalid provider-reported cost"
        record["cost_basis"] = "invalid_reported_cost"
    try:
        if record.get("status") != 200:
            raise ValueError("HTTP failure or transport error")
        normalizer = normalize_decisions if arm == "decider" else normalize_chat
        record["normalized"] = normalizer(response, prompt["questions"])
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        record["normalized"] = None
        record["validation_error"] = type(exc).__name__ + ": " + str(exc)
    return record


def load_ledger(path):
    try:
        scope = str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        scope = None
    if scope is not None:
        require_restored(ROOT, scope=scope)
    records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    spent = 0
    done = {}
    for record in records:
        # Derive from usage even for older rows, without rewriting their evidence.
        usage = record.get("usage") or record.get("response", {}).get("usage") or {}
        try:
            cost, _ = effective_cost(usage, record.get("cost_usd"))
        except (TypeError, ValueError):
            cost = None
        if cost is None:
            from cost_reservations import guard_cost

            try:
                cost = guard_cost(record, path, cost)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("Earlier request has unknown cost; reconcile it before resuming") from exc
        spent += cost
        key = (record["benchmark"], record["task"], record["sample"], record["arm"])
        if key in done:
            raise ValueError("Ledger contains duplicate attempts; use a fresh output path")
        done[key] = record
    return records, done, spent


def select_samples(manifest, ids=None, limit=None):
    samples = manifest["samples"]
    names = [sample["id"] for sample in samples]
    if len(set(names)) != len(names):
        raise ValueError("Manifest contains duplicate sample IDs")
    if ids:
        if not set(ids) <= set(names):
            raise ValueError("Requested sample IDs are absent from the manifest")
        samples = [sample for sample in samples if sample["id"] in set(ids)]
    return samples[:limit] if limit else samples


def sample_context(sample, mode):
    if mode == "image_only":
        return None
    if mode != "pixel_stats":
        raise ValueError("Unknown sample context mode")
    from tracks.image_context import validate_model_context

    context = sample.get("model_context")
    validate_model_context(context)
    if "camera" in context:
        raise ValueError("Pixel statistics mode excludes camera metadata")
    if sample.get("model_context_sha256") != digest(context):
        raise ValueError("Frozen model context hash mismatch")
    return context


def run(args):
    # A new output path must not bypass missing evidence for historical inputs.
    if args.manifest.resolve().is_relative_to(ROOT.resolve()):
        require_restored(ROOT)
    prompt = json.loads(args.prompt.read_text())
    validate_prompt(prompt, args.task)
    manifest = json.loads(args.manifest.read_text())
    benchmark = manifest.get("benchmark", args.manifest.stem)
    samples = select_samples(manifest, args.ids, args.limit)
    # Validate every selected delivered view before the first paid call.
    images = {sample["id"]: image_part(sample) for sample in samples}
    context_mode = getattr(args, "sample_context_mode", "image_only")
    contexts = {sample["id"]: sample_context(sample, context_mode) for sample in samples}
    records, done, spent = load_ledger(args.output)
    calls = len(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        for arm in args.models:
            endpoint, payload = make_request(
                arm,
                images[sample["id"]],
                prompt,
                args.effort,
                args.max_tokens,
                extra_context=contexts[sample["id"]],
            )
            request_hash = digest({"endpoint": endpoint, "payload": payload})
            key = (benchmark, args.task, sample["id"], arm)
            if key in done:
                if done[key]["request_sha256"] != request_hash:
                    raise ValueError("Request changed; use a new output path to retain previous evidence")
                continue
            # This reserves headroom, not a provider-side spending cap.
            if calls >= args.max_calls or spent + args.reserve_usd > args.budget_usd:
                raise ValueError("Local request or spend guard reached")
            record = post(endpoint, payload)
            record.update(
                benchmark=benchmark,
                task=args.task,
                sample=sample["id"],
                arm=arm,
                timestamp=datetime.now(UTC).isoformat(),
                request_sha256=request_hash,
                prompt_sha256=digest(prompt),
                manifest_sha256=digest(manifest),
                protocol_sha256=hashlib.sha256((ROOT / "bench_protocol.py").read_bytes()).hexdigest(),
                delivered_sha256=sample["delivered_sha256"],
                prompt_id=prompt.get("id"),
            )
            if context_mode != "image_only":
                record.update(
                    sample_context_mode=context_mode,
                    model_context_sha256=sample["model_context_sha256"],
                )
            response_record(record, arm, prompt, args.effort)
            with args.output.open("a") as ledger:
                ledger.write(json.dumps(record, allow_nan=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            done[key] = record
            calls += 1
            print(
                sample["id"],
                arm,
                record.get("status"),
                f"{record['seconds']:.3f}s",
                record["effective_cost_usd"],
                flush=True,
            )
            if record["effective_cost_usd"] is None:
                raise ValueError("Unknown request cost; stopped before further spending")
            spent += record["effective_cost_usd"]
    from cost_reservations import eligible, reservation

    reserved = sum(reservation(record, args.output) or 0 for record in records if eligible(record))
    if reserved:
        print(
            f"Total ledger requests: {calls}; budget-accounted cost: ${spent:.6f}; "
            f"known reported cost including BYOK: ${spent - reserved:.6f}; "
            f"reserved unverified cost: ${reserved:.6f}; actual total cost: unknown"
        )
    else:
        print(f"Total ledger requests: {calls}; effective reported cost including BYOK: ${spent:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--task", choices=["quality", "ai"], required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-context-mode", choices=["image_only", "pixel_stats"], default="image_only")
    parser.add_argument(
        "--effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default="medium"
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--budget-usd", type=float, default=10)
    parser.add_argument("--reserve-usd", type=float, default=0.10)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Use a positive image limit")
    if not 1 <= args.max_calls <= 360 or not 0 < args.reserve_usd <= args.budget_usd <= 10:
        parser.error("Use 1..360 requests and a positive local budget no larger than $10")
    if not 128 <= args.max_tokens <= 16384:
        parser.error("Use an output-token cap between 128 and 16384")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error(
            "OPENROUTER_API_KEY is absent; load the existing environment without passing a key argument"
        )
    try:
        run(args)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
