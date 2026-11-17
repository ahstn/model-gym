"""Freeze and execute new batches through an explicitly supplied request adapter.

The adapter owns wire validation, HTTP timing, attempt receipts and accounting.
This module owns batch selection, immutable provenance and scheduler policy.
No historical module globals or plans are changed to start a new batch.
"""

from __future__ import annotations

import concurrent.futures
import copy
import fcntl
import json
import os
import re
import threading
from collections import Counter, deque
from pathlib import Path

from visual_quality.labeling_profile import load_profile
from visual_quality.scoring import api_contract as contract


def freeze(args, adapter, policy, source_paths):
    """Create one immutable plan; subsequent executions load its saved policy."""
    if not re.fullmatch(r"[a-z0-9_-]+", args.phase):
        raise ValueError("Unsafe phase")
    path = args.data / "results/api" / args.model / "plans" / (args.phase + ".json")
    if path.exists():
        raise ValueError("Phase already frozen")
    manifest = adapter.read(args.manifest)
    cases = manifest["cases"]
    if not cases or len(cases) != len({c["case_id"] for c in cases}):
        raise ValueError("Unique nonempty selected cases required")
    if any(c.get("benchmark_partition") != "rehearsal" for c in cases):
        raise ValueError("Expansion scoring requires rehearsal cases only")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", c["case_id"]) for c in cases):
        raise ValueError("Unsafe case ID")
    profile = load_profile(args.profile_directory)
    cases = [{key: case.get(key) for key in adapter.CASE_FIELDS} for case in cases]
    cases.sort(key=lambda c: contract.digest(["expansion-order-v1", args.model, c["case_id"]]))
    sources = [Path(__file__), *source_paths]
    snapshot = args.data / "api-provider-snapshot.json"
    plan = {
        "schema_version": "labeling-expansion-api-v2",
        "created_at": adapter.now(),
        "model": args.model,
        "phase": args.phase,
        "selection": "rehearsal",
        "profile": profile,
        "policy": copy.deepcopy(policy),
        "cases": cases,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": adapter.sha(args.manifest),
        "reference_pins": {},
        "code_pins": {
            **adapter.code_pins(),
            **{str(path.resolve()): adapter.sha(path) for path in sources},
        },
        "provider_snapshot_path": str(snapshot.resolve()),
        "provider_snapshot_sha256": adapter.sha(snapshot),
        "retry_policy": "Bounded transport/429/502/503/504 retries; never retry scores or content denials",
        "latency_policy": "HTTP invocation through complete response, excluding preparation and scheduler wait",
        "stats_sent_to_models": False,
        "reference_scores_sent_to_models": False,
        "training_ready": False,
    }
    plan["plan_id"] = contract.digest(plan)
    adapter.atomic(path, plan)
    return {"plan_path": str(path), "plan_id": plan["plan_id"], "cases": len(cases)}


def validate_execution(args, adapter):
    """Verify immutable inputs and validate runtime-only limits before any call."""
    root = args.data / "results/api" / args.model
    plan = adapter.read(root / "plans" / (args.phase + ".json"))
    adapter.validate_plan(plan)
    if plan["model"] != args.model or plan["phase"] != args.phase:
        raise ValueError("Plan identity differs from requested route/phase")
    for path_key, sha_key in (
        ("manifest_path", "manifest_sha256"),
        ("provider_snapshot_path", "provider_snapshot_sha256"),
    ):
        if adapter.sha(Path(plan[path_key])) != plan[sha_key]:
            raise ValueError("Frozen input changed: " + plan[path_key])
    worker_limit = getattr(args, "worker_limit", None)
    if worker_limit is not None and (type(worker_limit) is not int or worker_limit < 1):
        raise ValueError("Worker limit must be a positive integer")
    if args.limit is not None and (type(args.limit) is not int or args.limit < 1):
        raise ValueError("Case limit must be a positive integer")
    workers = min(plan["policy"]["workers"], worker_limit or plan["policy"]["workers"])
    cases = plan["cases"][: args.limit] if args.limit else plan["cases"]
    return root, plan, workers, cases


class Circuit:
    """Per-image denials do not indicate an outage; credentials always stop."""

    def __init__(self):
        self.recent = deque(maxlen=100)
        self.denials = 0

    def observe(self, row, raw, adapter):
        denied = adapter.image_content_rejection(raw)
        self.denials += int(denied)
        if raw.get("http_status") in {401, 402, 403} and not denied:
            return "Credential, billing or unrecognized authorization failure"
        if not denied:
            self.recent.append(row["status"] not in {"valid", "invalid"})
        if len(self.recent) >= 20 and sum(self.recent) / len(self.recent) > 0.25:
            return "More than 25% exhausted service failures, excluding terminal image denials"
        return None


def _execute(plan, cases, root, secret, ledger, workers, adapter, save, state):
    """Drain active calls after a stop, retaining their attempt receipts."""
    stop = threading.Event()
    circuit, counts = Circuit(), Counter()
    pending, active = iter(cases), {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while True:
                while not stop.is_set() and len(active) < workers:
                    case = next(pending, None)
                    if case is None:
                        break
                    active[pool.submit(adapter.job, plan, case, root, secret, ledger, stop)] = case
                if not active:
                    break
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    row = future.result()
                    if row is None:
                        continue
                    counts[row["status"]] += 1
                    raw = adapter.read(row["raw_response_path"])
                    reason = circuit.observe(row, raw, adapter)
                    if reason:
                        state.setdefault("stop_reason", reason)
                        stop.set()
                    state.update(
                        outcomes=dict(counts),
                        completed_cases=sum(counts.values()),
                        content_denials=circuit.denials,
                    )
                    if sum(counts.values()) % 50 == 0:
                        save("stopping" if stop.is_set() else "running")
                        print(json.dumps(state), flush=True)
        except BaseException as error:
            stop.set()
            state["stop_reason"] = type(error).__name__ + ": " + str(error).replace(secret, "[REDACTED]")
            raise
    if stop.is_set():
        return "stopped"
    if sum(counts.values()) != len(cases):
        state["stop_reason"] = "Not every selected case produced an observation"
        return "stopped"
    return "complete" if len(cases) == len(plan["cases"]) else "partial"


def run(args, adapter):
    """Resume a frozen plan without spending again on terminal observations."""
    root, plan, workers, cases = validate_execution(args, adapter)
    route = plan["profile"]["routes"]["routes"][args.model]
    secret = os.environ[route["key_env"]]
    if not secret:
        raise ValueError("Missing credential: " + route["key_env"])
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = adapter.Ledger(root, plan["policy"])
        state = {
            "model": args.model,
            "phase": args.phase,
            "plan_id": plan["plan_id"],
            "started_at": adapter.now(),
            "planned_cases": len(plan["cases"]),
            "selected_cases": len(cases),
            "completed_cases": 0,
            "active_worker_limit": workers,
            "content_denials": 0,
            "outcomes": {},
            "content_denial_policy": "Terminal missing assessment; no retry; unrelated allowed images continue",
        }

        def save(status):
            state.update(status=status, updated_at=adapter.now(), **ledger.summary())
            adapter.atomic(root / (args.phase + "-status.json"), state)

        save("running")
        try:
            status = _execute(plan, cases, root, secret, ledger, workers, adapter, save, state)
        except BaseException:
            # _execute has drained its executor, so this includes late receipts.
            save("stopped")
            raise
        save(status)
        return state
