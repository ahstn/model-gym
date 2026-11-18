"""Freeze and perform one identical retry for verified empty token-limit completions.

This is a separate, explicitly selected recovery phase. It never changes the
baseline retry policy, profile, observations with scores, or existing receipts.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import signal
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import score

adapter = score.adapter
PHASE = "empty-length-once"
KIND = "empty_length_once_v1"


@contextmanager
def model_lock(data, model):
    root = Path(data) / "results/api" / model
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield root


def completed_baseline(data, model):
    args = SimpleNamespace(data=Path(data), model=model, phase=score.PHASE, limit=None, worker_limit=1)
    root, plan, _, cases = score.batch.validate_execution(args, adapter)
    status_path = root / f"{score.PHASE}-status.json"
    state = adapter.read(status_path)
    if (
        state.get("status") != "complete"
        or state.get("plan_id") != plan["plan_id"]
        or any(state.get(key) != len(cases) for key in ("planned_cases", "selected_cases", "completed_cases"))
        or sum(state.get("outcomes", {}).values()) != len(cases)
    ):
        raise ValueError("Baseline must be fully complete before recovery")
    manifest = adapter.read(Path(data) / "manifest.json")["cases"]
    expected = {row["case_id"]: row["view_sha256"] for row in manifest}
    if {row["case_id"]: row["view_sha256"] for row in cases} != expected:
        raise ValueError("Baseline does not cover the full frozen manifest")
    observed = {path.stem: adapter.read(path) for path in (root / "observations").glob("*.json")}
    if set(observed) != set(expected) or any(
        row["case_id"] != identity or row["view_sha256"] != expected[identity]
        for identity, row in observed.items()
    ):
        raise ValueError("Baseline observations do not cover the delivered manifest")
    route = plan["profile"]["routes"]["routes"][model]
    settings = route["settings"]
    effort = settings.get("reasoning_effort", settings.get("reasoning", {}).get("effort"))
    if model not in ("luna", "deepseek") or route["kind"] != "chat" or effort != "high":
        raise ValueError("Recovery requires the unchanged high-reasoning teacher route")
    if settings.get("max_tokens") != 16384:
        raise ValueError("Recovery must preserve the evaluated token budget")
    return root, plan, observed


def empty_length(result, route):
    """Fail closed unless the complete raw envelope proves an empty reasoning cutoff."""
    try:
        if result.get("http_status") != 200 or result.get("transport_error"):
            return False
        body = adapter.contract.load_json(result["body"])
        if body.get("error") or body.get("model") not in route["accepted_models"]:
            return False
        if route.get("accepted_providers") and body.get("provider") not in route["accepted_providers"]:
            return False
        choices = body["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "length":
            return False
        message = choices[0]["message"]
        if (
            message.get("role") != "assistant"
            or "content" not in message
            or message["content"] not in (None, "")
            or message.get("refusal")
            or message.get("tool_calls")
            or message.get("function_call")
        ):
            return False
        usage = body["usage"]
        tokens = usage["completion_tokens"]
        if type(tokens) is not int or tokens != route["settings"]["max_tokens"]:
            return False
        reasoning = usage["completion_tokens_details"]["reasoning_tokens"]
        if type(reasoning) is not int or reasoning != tokens:
            return False
        parsed = adapter.parsed_result(result, route)
        return (
            parsed == result["parsed"]
            and parsed["status"] == "failed"
            and parsed["reason"] == "Incomplete completion"
            and parsed["technical_score"] is None
            and parsed["aesthetic_score"] is None
            and not any(parsed["task_valid"].values())
        )
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return False


def previous_transient(result):
    """Earlier transport failures are allowed; earlier assessment envelopes are not."""
    if not result.get("parsed", {}).get("retryable"):
        return False
    if result.get("http_status") not in (None, 429, 502, 503, 504):
        return False
    try:
        body = adapter.contract.load_json(result["body"]) if result.get("body") else {}
        return isinstance(body, dict) and not body.get("choices")
    except (ValueError, TypeError):
        return False


def eligibility(root, baseline, case, row):
    if (
        row.get("status") != "failed"
        or row.get("reason") != "Incomplete completion"
        or any(row.get(key) is not None for key in ("technical_score", "aesthetic_score"))
        or any(row.get("task_valid", {}).values())
    ):
        return None
    route = baseline["profile"]["routes"]["routes"][baseline["model"]]
    _, request_sha = adapter.prepare(baseline, case)
    paths = adapter.attempts_for(root, case["case_id"])
    if not paths or row.get("attempt_count") != len(paths):
        return None
    pins, results = {}, []
    for number, path in enumerate(paths, 1):
        request = adapter.read(path)
        if (
            path.parent.name != f"{number:02d}"
            or request.get("plan_id") != baseline["plan_id"]
            or request.get("attempt") != number
            or request.get("model") != baseline["model"]
            or request.get("case_id") != case["case_id"]
            or request.get("view_sha256") != case["view_sha256"]
            or request.get("profile_sha256") != baseline["profile"]["sha256"]
            or request.get("request_sha256") != request_sha
        ):
            raise ValueError("Source request differs from the frozen baseline")
        result_path = path.with_name("result.json")
        if not result_path.exists():
            return None  # An ambiguous prior call is not proven to lack an assessment.
        result = adapter.read(result_path)
        parsed = result.get("parsed", {})
        if (
            parsed.get("status") in ("valid", "invalid", "partial")
            or any(parsed.get(key) is not None for key in ("technical_score", "aesthetic_score"))
            or any(parsed.get("task_valid", {}).values())
            or adapter.image_content_rejection(result)
        ):
            return None
        results.append(result)
        pins.update({str(path): adapter.sha(path), str(result_path): adapter.sha(result_path)})
    if not empty_length(results[-1], route) or not all(previous_transient(r) for r in results[:-1]):
        return None
    if (
        row.get("raw_response_sha256") != adapter.sha(paths[-1].with_name("result.json"))
        or row.get("raw_response_path") != str(paths[-1].with_name("result.json"))
        or row.get("request_sha256") != request_sha
        or row.get("plan_id") != baseline["plan_id"]
    ):
        raise ValueError("Baseline observation points to changed source evidence")
    return {"previous_attempt_count": len(paths), "request_sha256": request_sha, "source_pins": pins}


def freeze(data, model):
    with model_lock(data, model) as root:
        target = root / "plans" / f"{PHASE}.json"
        if target.exists():
            raise ValueError("Recovery phase already frozen; run its existing plan")
        root, baseline, observed = completed_baseline(data, model)
        eligible = {}
        for case in baseline["cases"]:
            item = eligibility(root, baseline, case, observed[case["case_id"]])
            if item:
                eligible[case["case_id"]] = item
        if not eligible:
            return {"status": "no_eligible_empty_completions", "cases": 0}
        references = {
            str(root / "plans" / f"{score.PHASE}.json"): adapter.sha(root / "plans" / f"{score.PHASE}.json"),
            str(root / f"{score.PHASE}-status.json"): adapter.sha(root / f"{score.PHASE}-status.json"),
        }
        for identity in observed:
            path = root / "observations" / f"{identity}.json"
            if identity in eligible:
                archive = root / "recovery-evidence" / PHASE / f"{identity}.json"
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists() and archive.read_bytes() != path.read_bytes():
                    raise ValueError("Archived baseline observation changed")
                archive.write_bytes(path.read_bytes())
                references.update(eligible[identity]["source_pins"])
                eligible[identity]["baseline_observation_path"] = str(archive)
                path = archive
            references[str(path)] = adapter.sha(path)
        limits = {identity: row["previous_attempt_count"] + 1 for identity, row in eligible.items()}
        plan = copy.deepcopy(baseline)
        plan.pop("plan_id")
        plan.update(
            created_at=adapter.now(),
            phase=PHASE,
            cases=[case for case in baseline["cases"] if case["case_id"] in eligible],
            reference_pins={**baseline["reference_pins"], **references},
            code_pins={**baseline["code_pins"], str(Path(__file__).resolve()): adapter.sha(__file__)},
            retry_policy="Exactly one identical request per verified empty length cutoff; no retry of its result",
            recovery={
                "kind": KIND,
                "baseline_phase": score.PHASE,
                "baseline_plan_id": baseline["plan_id"],
                "eligibility": eligible,
                "attempt_limits": limits,
            },
        )
        plan["policy"]["max_attempts"] = max(limits.values())
        plan["plan_id"] = adapter.contract.digest(plan)
        adapter.atomic(target, plan)
        return {
            "status": "frozen",
            "cases": len(eligible),
            "plan_id": plan["plan_id"],
            "plan_path": str(target),
        }


def validate_recovery(data, model):
    root, baseline, observed = completed_baseline(data, model)
    plan = adapter.read(root / "plans" / f"{PHASE}.json")
    adapter.validate_plan(plan)
    recovery = plan["recovery"]
    if (
        plan["phase"] != PHASE
        or plan["model"] != model
        or recovery["kind"] != KIND
        or recovery["baseline_phase"] != score.PHASE
        or recovery["baseline_plan_id"] != baseline["plan_id"]
        or plan["profile"] != baseline["profile"]
        or plan["manifest_sha256"] != baseline["manifest_sha256"]
        or plan["manifest_path"] != baseline["manifest_path"]
        or plan["provider_snapshot_sha256"] != baseline["provider_snapshot_sha256"]
    ):
        raise ValueError("Recovery changed its baseline/profile/manifest contract")
    expected_code = {**baseline["code_pins"], str(Path(__file__).resolve()): adapter.sha(__file__)}
    if plan["code_pins"] != expected_code:
        raise ValueError("Recovery execution source pins are incomplete or changed")
    identities = {case["case_id"] for case in plan["cases"]}
    if not identities or len(identities) != len(plan["cases"]) or identities != set(recovery["eligibility"]):
        raise ValueError("Recovery allowlist differs from its cases")
    if identities != set(recovery["attempt_limits"]):
        raise ValueError("Recovery needs exactly one limit for each allowlisted case")
    for identity in set(observed) - identities:
        path = root / "observations" / f"{identity}.json"
        if plan["reference_pins"].get(str(path)) != adapter.sha(path):
            raise ValueError("Recovery must pin every noneligible observation")
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    for case in plan["cases"]:
        item = recovery["eligibility"][case["case_id"]]
        limit = recovery["attempt_limits"][case["case_id"]]
        previous = item["previous_attempt_count"]
        if (
            case != baseline_cases.get(case["case_id"])
            or type(previous) is not int
            or previous < 1
            or type(limit) is not int
            or limit != previous + 1
        ):
            raise ValueError("Recovery case or exact one-attempt limit changed")
        source_row = adapter.read(item["baseline_observation_path"])
        archive = Path(item["baseline_observation_path"])
        if plan["reference_pins"].get(str(archive)) != adapter.sha(archive):
            raise ValueError("Original observation archive is not pinned")
        # Original receipt pins are immutable; appended recovery attempts are checked below.
        source_paths = adapter.attempts_for(root, case["case_id"])[: item["previous_attempt_count"]]
        if len(source_paths) != item["previous_attempt_count"]:
            raise ValueError("Original attempts disappeared")
        expected_pins = {
            str(path): adapter.sha(path)
            for request in source_paths
            for path in (request, request.with_name("result.json"))
        }
        if item["source_pins"] != expected_pins or any(
            plan["reference_pins"].get(path) != sha for path, sha in expected_pins.items()
        ):
            raise ValueError("Original request/response pins are incomplete")
        route = baseline["profile"]["routes"]["routes"][model]
        if not empty_length(adapter.read(source_paths[-1].with_name("result.json")), route):
            raise ValueError("Frozen source no longer proves an empty length completion")
        if source_row.get("technical_score") is not None or source_row.get("aesthetic_score") is not None:
            raise ValueError("Recovery cannot replace an existing numeric assessment")
    return root, plan


def recover_case(root, plan, case, secret, ledger):
    identity = case["case_id"]
    item = plan["recovery"]["eligibility"][identity]
    limit = plan["recovery"]["attempt_limits"][identity]
    body, request_sha = adapter.prepare(plan, case)
    if request_sha != item["request_sha256"]:
        raise ValueError("Recovery request bytes differ from the original request")
    paths = adapter.attempts_for(root, identity)
    if len(paths) == limit:
        request = adapter.read(paths[-1])
        if (
            request.get("plan_id") != plan["plan_id"]
            or request.get("model") != plan["model"]
            or request.get("case_id") != identity
            or request.get("request_sha256") != request_sha
            or request.get("view_sha256") != case["view_sha256"]
            or request.get("profile_sha256") != plan["profile"]["sha256"]
            or request.get("attempt") != limit
            or paths[-1].parent.name != f"{limit:02d}"
        ):
            raise ValueError("Existing recovery request differs from its frozen plan")
        if not paths[-1].with_name("result.json").exists():
            raise ValueError(
                "Recovery request lacks a response; manual reconciliation required, never resend"
            )
    elif len(paths) == item["previous_attempt_count"]:
        adapter.request_once(plan, case, root, limit, body, request_sha, secret, ledger)
    else:
        raise ValueError("Recovery attempt count exceeds or differs from the one-retry allowance")
    result = adapter.read(root / "attempts" / identity / f"{limit:02d}" / "result.json")
    route = plan["profile"]["routes"]["routes"][plan["model"]]
    if result.get("parsed") != adapter.parsed_result(result, route):
        raise ValueError("Recovery response no longer parses identically")
    return adapter.observation(plan, case, root, request_sha)


def run(data, model, stop=None):
    stop = stop or threading.Event()
    with model_lock(data, model):
        root, plan = validate_recovery(data, model)
        secret = os.environ[plan["profile"]["routes"]["routes"][model]["key_env"]]
        if not secret:
            raise ValueError("Missing recovery route credential")
        ledger = adapter.Ledger(root, plan["policy"])
        circuit, counts = score.batch.Circuit(), Counter()
        state = {
            "model": model,
            "phase": PHASE,
            "plan_id": plan["plan_id"],
            "started_at": adapter.now(),
            "planned_cases": len(plan["cases"]),
            "selected_cases": len(plan["cases"]),
            "completed_cases": 0,
            "active_worker_limit": 1,
            "outcomes": {},
            "content_denials": 0,
        }

        def save(status):
            state.update(status=status, updated_at=adapter.now(), **ledger.summary())
            adapter.atomic(root / f"{PHASE}-status.json", state)

        save("running")
        try:
            for case in plan["cases"]:
                if stop.is_set():
                    raise InterruptedError("Recovery stopped before admitting another request")
                row = recover_case(root, plan, case, secret, ledger)
                counts[row["status"]] += 1
                reason = circuit.observe(row, adapter.read(row["raw_response_path"]), adapter)
                state.update(
                    outcomes=dict(counts),
                    completed_cases=sum(counts.values()),
                    content_denials=circuit.denials,
                )
                if reason:
                    raise RuntimeError(reason)
                save("running")
            adapter.validate_plan(plan)
            save("complete")
        except BaseException as error:
            state["stop_reason"] = (type(error).__name__ + ": " + str(error)).replace(secret, "[REDACTED]")
            save("stopped")
            raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "run"))
    parser.add_argument("--model", required=True, choices=("luna", "deepseek"))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(score.DATA, args.model)
    elif args.command == "check":
        with model_lock(score.DATA, args.model):
            _, plan = validate_recovery(score.DATA, args.model)
        result = {"status": "validated", "cases": len(plan["cases"]), "network_requests": 0}
    else:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        result = run(score.DATA, args.model, stop)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
