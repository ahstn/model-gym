"""Resumable baseline scoring with frozen requests and complete attempt accounting."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")
sys.path.insert(0, str(PACKAGE / "src"))
from visual_quality.labeling_profile import MODELS, load_profile, request_payload
from visual_quality.scoring import api_contract as contract

POLICY = {
    "luna": {
        "workers": 16,
        "max_attempts": 3,
        "budget_usd": "25.00",
        "reserve_usd": "0.222288",
        "timeout": 740,
    },
    "deepseek": {
        "workers": 16,
        "max_attempts": 5,
        "budget_usd": "15.00",
        "reserve_usd": "0.167117",
        "timeout": 740,
    },
    "liquid": {
        "workers": 4,
        "max_attempts": 5,
        "budget_usd": "2.00",
        "reserve_usd": "0.00131072",
        "timeout": 180,
    },
    "jev": {
        "workers": 4,
        "max_attempts": 5,
        "budget_usd": "8.00",
        "reserve_usd": "0.011010048",
        "timeout": 180,
    },
}
CASE_FIELDS = (
    "case_id",
    "family_id",
    "image_path",
    "view_sha256",
    "benchmark_partition",
    "benchmark_order",
    "source",
    "stratum",
    "content_cluster",
    "rehearsal_content_cluster",
    "operator",
    "severity",
)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return contract.load_json(Path(path).read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, allow_nan=False, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    return contract.file_sha(path)


def code_pins():
    files = [
        Path(__file__),
        PACKAGE / "src/visual_quality/labeling_profile.py",
        Path(contract.__file__),
        contract.WIRE_PATH,
    ]
    return {str(path): sha(path) for path in files}


def reference_pins(cases, freeze_path):
    if not freeze_path:
        raise ValueError("Benchmark requests require a reference freeze")
    freeze = read(freeze_path)
    path = Path(freeze["references_path"])
    if sha(path) != freeze["references_sha256"]:
        raise ValueError("Reference file changed")
    refs = {row["case_id"]: row for row in map(contract.load_json, path.read_text().splitlines())}
    for case in cases:
        ref = refs.get(case["case_id"])
        if not ref or ref.get("view_sha256") != case["view_sha256"]:
            raise ValueError("Reference missing or image mismatched: " + case["case_id"])
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    if frozen_at.tzinfo is None or frozen_at > datetime.now(UTC):
        raise ValueError("Reference freeze must have a past timezone-aware timestamp")
    return {str(freeze_path.resolve()): sha(freeze_path), str(path.resolve()): sha(path)}


def freeze(args):
    if not re.fullmatch(r"[a-z0-9_-]+", args.phase):
        raise ValueError("Unsafe phase")
    path = args.data / "results/api" / args.model / "plans" / (args.phase + ".json")
    if path.exists():
        raise ValueError("Phase already frozen")
    manifest = read(args.manifest)
    cases = [
        c
        for c in manifest["cases"]
        if (c["benchmark_partition"] == "rehearsal") == (args.selection == "rehearsal")
    ]
    if args.case_ids:
        wanted = set(read(args.case_ids))
        cases = [c for c in cases if c["case_id"] in wanted]
        if {c["case_id"] for c in cases} != wanted:
            raise ValueError("Requested IDs absent from selected partition")
    if not cases or len(cases) != len({c["case_id"] for c in cases}):
        raise ValueError("Unique nonempty selected cases required")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", c["case_id"]) for c in cases):
        raise ValueError("Unsafe case ID")
    pins = reference_pins(cases, args.references) if args.selection == "benchmark" else {}
    profile = load_profile(PACKAGE / "labeling")
    cases = [{key: case.get(key) for key in CASE_FIELDS} for case in cases]
    cases.sort(key=lambda c: contract.digest(["labeling-10000-order-v1", args.model, c["case_id"]]))
    plan = {
        "schema_version": "labeling-10000-api-v1",
        "created_at": now(),
        "model": args.model,
        "phase": args.phase,
        "selection": args.selection,
        "profile": profile,
        "policy": POLICY[args.model],
        "cases": cases,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "reference_pins": pins,
        "code_pins": code_pins(),
        "provider_snapshot_sha256": sha(args.data / "api-provider-snapshot.json"),
        "retry_policy": "Retry bounded transport/429/502/503/504 failures, never a valid or invalid score response",
        "latency_policy": "HTTP invocation through complete response, excluding preparation and scheduler wait",
        "stats_sent_to_models": False,
        "reference_scores_sent_to_models": False,
        "training_ready": False,
    }
    plan["plan_id"] = contract.digest(plan)
    atomic(path, plan)
    return {"plan_path": str(path), "plan_id": plan["plan_id"], "cases": len(cases)}


def validate_plan(plan):
    if contract.digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Frozen plan changed")
    for path, expected in {**plan["reference_pins"], **plan["code_pins"]}.items():
        if sha(path) != expected:
            raise ValueError("Pinned input changed: " + path)


def parsed_result(result, route):
    row = contract.interpret(result, route)
    if route["kind"] == "chat" and row.get("scope_floor_applied"):
        ratings = row.get("ratings", {})
        if not all(
            ratings.get(t, {}).get("score") == 1.0
            and "unsupported_domain" in ratings.get(t, {}).get("limitations", [])
            for t in contract.TASKS
        ):
            row.update(status="invalid", technical_score=None, aesthetic_score=None)
            row["task_valid"] = dict.fromkeys(contract.TASKS, False)
            row["validation_errors"] = dict.fromkeys(
                contract.TASKS, "Photo scope floor requires both scores 1.00 with unsupported_domain"
            )
    retryable = result.get("transport_error") is not None or result.get("http_status") in {429, 502, 503, 504}
    row["retryable"] = retryable
    if retryable:
        row["status"] = "retryable_failure"
    return row


class Ledger:
    def __init__(self, root, policy):
        self.lock = threading.Lock()
        self.reserve = Decimal(policy["reserve_usd"])
        self.cap = Decimal(policy["budget_usd"])
        self.known = Decimal(0)
        self.unknown = self.attempts = self.active = 0
        for request in root.glob("attempts/*/*/request.json"):
            path = request.with_name("result.json")
            row = read(path) if path.exists() else {}
            self.attempts += 1
            price = row.get("price_usd")
            if price is None:
                self.unknown += 1
            else:
                self.known += Decimal(price)

    def acquire(self):
        with self.lock:
            if self.known + self.reserve * (self.unknown + self.active + 1) > self.cap:
                raise RuntimeError("Conservative route budget guard reached")
            self.active += 1

    def settle(self, price):
        with self.lock:
            self.active -= 1
            self.attempts += 1
            if price is None:
                self.unknown += 1
            else:
                self.known += Decimal(price)

    def summary(self):
        with self.lock:
            return {
                "http_attempts": self.attempts,
                "known_price_usd": str(self.known),
                "unpriced_attempts": self.unknown,
                "admission_liability_usd": str(self.known + self.reserve * (self.unknown + self.active)),
                "budget_usd": str(self.cap),
            }


def attempts_for(root, case_id):
    return sorted((root / "attempts" / case_id).glob("*/request.json"))


def prepare(plan, case):
    image = Path(case["image_path"]).read_bytes()
    if hashlib.sha256(image).hexdigest() != case["view_sha256"]:
        raise ValueError("Delivered image changed")
    body = contract.encoded(request_payload(plan["profile"], plan["model"], image))
    if len(body) > (3670016 if plan["model"] in {"luna", "deepseek"} else 33554432):
        raise ValueError("Request exceeds provider size ceiling")
    return body, hashlib.sha256(body).hexdigest()


def request_once(plan, case, root, number, body, request_sha, secret, ledger):
    route = plan["profile"]["routes"]["routes"][plan["model"]]
    folder = root / "attempts" / case["case_id"] / f"{number:02d}"
    ledger.acquire()
    folder.mkdir(parents=True, exist_ok=False)
    request = {
        "plan_id": plan["plan_id"],
        "model": plan["model"],
        "case_id": case["case_id"],
        "attempt": number,
        "created_at": now(),
        "request_sha256": request_sha,
        "request_bytes": len(body),
        "view_sha256": case["view_sha256"],
        "profile_sha256": plan["profile"]["sha256"],
    }
    atomic(folder / "request.json", request)
    result = {
        "http_status": None,
        "body": None,
        "headers": {},
        "transport_error": None,
        "request_started_at": now(),
    }
    start = time.perf_counter()
    try:
        response = requests.post(
            route["endpoint"],
            data=body,
            headers={
                **route["headers"],
                "Authorization": "Bearer " + secret,
                "Content-Type": "application/json",
            },
            timeout=(30, plan["policy"]["timeout"]),
            allow_redirects=False,
        )
        result.update(
            latency_seconds=time.perf_counter() - start,
            response_received_at=now(),
            http_status=response.status_code,
            body=response.text.replace(secret, "[REDACTED]"),
            headers={
                key.lower(): value.replace(secret, "[REDACTED]")
                for key, value in response.headers.items()
                if key.lower().startswith(("retry-after", "x-runinfra-", "x-request-id"))
            },
        )
    except requests.RequestException as error:
        result.update(
            latency_seconds=time.perf_counter() - start,
            response_received_at=now(),
            transport_error=type(error).__name__,
        )
    parsed = parsed_result(result, route)
    result["parsed"] = parsed
    result["price_usd"] = parsed["price_usd"]
    atomic(folder / "result.json", result)
    ledger.settle(parsed["price_usd"])
    return result


def observation(plan, case, root, request_sha):
    paths = attempts_for(root, case["case_id"])
    results = [read(p.with_name("result.json")) if p.with_name("result.json").exists() else {} for p in paths]
    last = results[-1]
    parsed = last.get("parsed", {"status": "unknown", "technical_score": None, "aesthetic_score": None})
    row = {
        **case,
        **parsed,
        "model": plan["model"],
        "arm": "baseline",
        "plan_id": plan["plan_id"],
        "attempt_id": str(paths[-1].parent.relative_to(root)),
        "attempt_count": len(paths),
        "request_sha256": request_sha,
        "request_started_at": last.get("request_started_at"),
        "response_received_at": last.get("response_received_at"),
        "latency_seconds": last.get("latency_seconds"),
        "latency_kind": "http_request_through_complete_response",
        "price_usd": last.get("price_usd"),
        "profile_version": plan["profile"]["teacher"]["version"],
        "prompt_sha256": plan["profile"]["sha256"][
            "teacher" if plan["model"] in {"luna", "deepseek"} else "decision"
        ],
        "profile_sha256": plan["profile"]["sha256"],
        "requested_reasoning": "high" if plan["model"] in {"luna", "deepseek"} else None,
        "returned_reasoning_effort": last.get("headers", {}).get("x-runinfra-reasoning-effort"),
        "raw_response_path": str(paths[-1].with_name("result.json")),
        "raw_response_sha256": sha(paths[-1].with_name("result.json")) if last else None,
        "total_attempt_known_price_usd": str(
            sum((Decimal(r["price_usd"]) for r in results if r.get("price_usd") is not None), Decimal(0))
        ),
        "unpriced_attempts": sum(r.get("price_usd") is None for r in results),
        "total_attempt_latency_seconds": sum(r.get("latency_seconds", 0) for r in results),
        "training_ready": False,
    }
    row["total_attempt_price_usd"] = (
        row["total_attempt_known_price_usd"] if row["unpriced_attempts"] == 0 else None
    )
    for task, prefix in zip(contract.TASKS, ("technical", "aesthetic"), strict=True):
        value = row.get(prefix + "_score")
        row[prefix + "_score_valid"] = parsed.get("task_valid", {}).get(task, False)
        row[prefix + "_score_display"] = None if value is None else f"{value:.2f}"
    row["evidence"] = {t: parsed.get("ratings", {}).get(t, {}).get("evidence") for t in contract.TASKS}
    row["limitations"] = {
        t: parsed.get("ratings", {}).get(t, {}).get("limitations", []) for t in contract.TASKS
    }
    atomic(root / "observations" / (case["case_id"] + ".json"), row)
    return row


def job(plan, case, root, secret, ledger, stop):
    body, request_sha = prepare(plan, case)
    previous = attempts_for(root, case["case_id"])
    for path in previous:
        request = read(path)
        if request["request_sha256"] != request_sha or request["view_sha256"] != case["view_sha256"]:
            raise ValueError("Existing request differs from frozen profile/image")
        result_path = path.with_name("result.json")
        if result_path.exists() and not read(result_path).get("parsed", {}).get("retryable", False):
            return observation(plan, case, root, request_sha)
    for number in range(len(previous) + 1, plan["policy"]["max_attempts"] + 1):
        if stop.is_set():
            return None
        result = request_once(plan, case, root, number, body, request_sha, secret, ledger)
        parsed = result["parsed"]
        if result["http_status"] in {401, 403, 402}:
            stop.set()
        if not parsed["retryable"]:
            break
        if number < plan["policy"]["max_attempts"]:
            try:
                server_delay = float(result["headers"].get("retry-after", 0))
            except ValueError:
                server_delay = 0
            stop.wait(min(60, max(server_delay, 2**number)))
    return observation(plan, case, root, request_sha)


def run(args):
    root = args.data / "results/api" / args.model
    plan = read(root / "plans" / (args.phase + ".json"))
    validate_plan(plan)
    route = plan["profile"]["routes"]["routes"][args.model]
    secret = os.environ[route["key_env"]]
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Ledger(root, plan["policy"])
        stop = threading.Event()
        cases = plan["cases"][: args.limit] if args.limit else plan["cases"]
        counts, recent = Counter(), deque(maxlen=100)
        state = {
            "model": args.model,
            "phase": args.phase,
            "plan_id": plan["plan_id"],
            "started_at": now(),
            "planned_cases": len(cases),
        }
        path = root / (args.phase + "-status.json")

        def save(status):
            state.update(status=status, updated_at=now(), outcomes=dict(counts), **ledger.summary())
            atomic(path, state)

        save("running")
        pending, active = iter(cases), {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan["policy"]["workers"]) as pool:
            try:
                while True:
                    while not stop.is_set() and len(active) < plan["policy"]["workers"]:
                        case = next(pending, None)
                        if case is None:
                            break
                        active[pool.submit(job, plan, case, root, secret, ledger, stop)] = case
                    if not active:
                        break
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        row = future.result()
                        if row is None:
                            continue
                        counts[row["status"]] += 1
                        recent.append(row["status"] not in {"valid", "invalid"})
                        if len(recent) >= 20 and sum(recent) / len(recent) > 0.25:
                            stop.set()
                            state["stop_reason"] = (
                                "More than 25% exhausted failures among the latest 20–100 cases"
                            )
                        if sum(counts.values()) % 50 == 0:
                            save("stopping" if stop.is_set() else "running")
                            print(json.dumps(state), flush=True)
            except BaseException as error:
                stop.set()
                state["stop_reason"] = type(error).__name__ + ": " + str(error).replace(secret, "[REDACTED]")
                save("stopped")
                raise
        save("stopped" if stop.is_set() else "complete")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--selection", choices=("rehearsal", "benchmark"), default="rehearsal")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--case-ids", type=Path)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(freeze(args) if args.command == "freeze" else run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
