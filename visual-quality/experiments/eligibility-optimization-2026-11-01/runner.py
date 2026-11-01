"""Run bounded, resumable Decisions trials without storing credentials or image payloads."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "rehearsal-4000-2026-11-01"
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")
sys.path.insert(0, str(OLD))
import api_contract as contract

ROUTES = {"liquid": "liquid", "jev": "jev"}
RESERVE = {"liquid": Decimal("0.00131072"), "jev": Decimal("0.011010048")}
CAP = Decimal("10.00")
LOCK = threading.Lock()
STOP = threading.Event()
LEDGER = {}
OUTSTANDING = {}


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def freeze_check():
    freeze = read(HERE / "trial-freeze.json")
    for path, sha in freeze["inputs"].items():
        if file_sha(path) != sha:
            raise ValueError("Frozen trial input changed: " + path)
    return freeze


def init_ledger():
    for model in ROUTES:
        known = Decimal(0)
        unknown = 0
        for req in (DATA / "attempts").glob(f"*/{model}/*/*/request.json"):
            result = req.with_name("result.json")
            if not result.exists():
                unknown += 1
                continue
            value = read(result).get("price_usd")
            if value is None:
                unknown += 1
            else:
                known += Decimal(str(value))
        LEDGER[model] = {"known": known, "unknown": unknown}
        OUTSTANDING[model] = 0


def acquire(model):
    with LOCK:
        info = LEDGER[model]
        liability = info["known"] + RESERVE[model] * (info["unknown"] + OUTSTANDING[model] + 1)
        if liability > CAP:
            STOP.set()
            raise RuntimeError("Conservative cost cap reached for " + model)
        OUTSTANDING[model] += 1


def settle(model, price):
    with LOCK:
        OUTSTANDING[model] -= 1
        if price is None:
            LEDGER[model]["unknown"] += 1
        else:
            LEDGER[model]["known"] += Decimal(str(price))


def job(round_id, model, case, question, routes, prompts, key):
    root = DATA / "attempts" / round_id / model / case["case_id"]
    route = routes[model]
    existing = sorted(root.glob("*/request.json")) if root.exists() else []
    for req in existing:
        result = req.with_name("result.json")
        if not result.exists():
            return "unknown_no_replay"
        prior = read(result)
        if prior["status"] != "retryable_503":
            return "existing_" + prior["status"]
    if len(existing) >= 3:
        return "exhausted_503"
    raw = Path(case["image_path"]).read_bytes()
    if digest(raw) != case["view_sha256"]:
        STOP.set()
        raise ValueError("Image changed: " + case["case_id"])
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode(), "detail": "high"},
    }
    questions = copy.deepcopy(prompts["prompts"][route["requested_model"]]["questions"])
    if question is not None:
        questions["excluded"]["instructions"] = question
    payload = {**route["settings"], "state": [prompts["common"], image], "questions": questions}
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    request_sha = digest(payload_bytes)
    for number in range(len(existing) + 1, 4):
        if STOP.is_set():
            return "stopped"
        acquire(model)
        attempt = root / f"{number:02d}"
        started = now()
        request = {
            "round": round_id,
            "model": model,
            "case_id": case["case_id"],
            "attempt": number,
            "started_at": started,
            "view_sha256": case["view_sha256"],
            "request_sha256": request_sha,
            "question_sha256": digest((question or questions["excluded"]["instructions"]).encode()),
            "route_model": route["requested_model"],
            "conservative_reservation_usd": str(RESERVE[model]),
        }
        atomic(attempt / "request.json", request)
        clock = time.monotonic()
        response_body = None
        status = None
        headers = {}
        transport_error = None
        try:
            response = requests.post(
                route["endpoint"],
                data=payload_bytes,
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    **route["headers"],
                },
                timeout=(20, 180),
            )
            status = response.status_code
            response_body = response.text
            headers = {
                k: response.headers.get(k)
                for k in ("date", "retry-after", "x-request-id")
                if response.headers.get(k)
            }
        except requests.RequestException as error:
            transport_error = type(error).__name__
        elapsed = time.monotonic() - clock
        parsed = {}
        price = None
        kind = "error"
        body = None
        if response_body is not None:
            try:
                body = contract.load_json(response_body)
            except (ValueError, TypeError):
                body = None
        if isinstance(body, dict):
            price = contract.reported_cost(body, route)["price_usd"]
        if status == 200 and isinstance(body, dict):
            try:
                parsed = contract.parse_scores(body, route)
                ratings = parsed["ratings"]
                kind = (
                    "valid"
                    if all(ratings.get(t, {}).get("score") is not None for t in contract.TASKS)
                    and parsed.get("excluded_probability") is not None
                    else "invalid"
                )
            except (ValueError, TypeError, KeyError) as error:
                kind = "invalid"
                parsed = {"validation_error": str(error)}
        elif (
            status == 503
            and isinstance(body, dict)
            and "Image processing is unavailable." in str(body.get("error", {}))
        ):
            kind = "retryable_503"
        elif status in (401, 403):
            STOP.set()
            kind = "authorization_error"
        result = {
            "status": kind,
            "http_status": status,
            "received_at": now() if response_body is not None else None,
            "latency_seconds": elapsed,
            "transport_error": transport_error,
            "headers": headers,
            "price_usd": price,
            "body_sha256": digest(response_body.encode()) if response_body is not None else None,
            "body": body if isinstance(body, dict) else None,
            "parsed": parsed,
        }
        atomic(attempt / "result.json", result)
        settle(model, price)
        if kind != "retryable_503":
            return kind
        if number < 3:
            time.sleep(0.5 * number)
    return "exhausted_503"


def run_round(round_id, jobs, questions, routes, prompts, key):
    print(json.dumps({"time": now(), "round": round_id, "jobs": len(jobs), "event": "start"}), flush=True)
    counts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(job, round_id, m, c, questions.get(m), routes, prompts, key) for m, c in jobs
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                status = f.result()
                counts[status] = counts.get(status, 0) + 1
            except Exception as error:  # noqa: BLE001 -- stop and drain all workers on unexpected failures.
                STOP.set()
                counts[type(error).__name__] = counts.get(type(error).__name__, 0) + 1
                print(
                    json.dumps({"event": "job_error", "type": type(error).__name__, "message": str(error)}),
                    flush=True,
                )
    state = {
        "time": now(),
        "round": round_id,
        "counts": counts,
        "ledger": {
            m: {
                "known_usd": str(LEDGER[m]["known"]),
                "unpriced_attempts": LEDGER[m]["unknown"],
                "liability_usd": str(LEDGER[m]["known"] + RESERVE[m] * LEDGER[m]["unknown"]),
            }
            for m in ROUTES
        },
        "stopped": STOP.is_set(),
    }
    atomic(DATA / "progress.json", state)
    print(json.dumps(state), flush=True)
    if STOP.is_set():
        raise SystemExit("Stopped after budget, authorization or validation error")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["smoke", "first", "second", "holdout"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    freeze_check()
    manifest = read(HERE / "reference-manifest.json")
    routes = read(OLD / "api-routes.json")
    prompts = read(OLD / "api-decision-prompts.json")
    variants = read(HERE / "rounds-first.json")
    if args.stage in ("second", "holdout"):
        second_freeze = read(HERE / "rounds-second-freeze.json")
        if file_sha(HERE / "rounds-second.json") != second_freeze["sha256"]:
            raise ValueError("Second-round prompts changed after their freeze")
        variants.update(read(HERE / "rounds-second.json"))
    if args.stage == "smoke":
        rounds = ["baseline"]
        cases = [x for x in manifest["cases"] if x["partition"] == "development"][:1]
    elif args.stage == "first":
        rounds = ["baseline"] + [f"r{i:02d}" for i in range(1, 6)]
        cases = [x for x in manifest["cases"] if x["partition"] == "development"]
    elif args.stage == "second":
        rounds = [f"r{i:02d}" for i in range(6, 11)]
        cases = [x for x in manifest["cases"] if x["partition"] == "development"]
    else:
        rounds = ["heldout-baseline", "heldout-selected"]
        cases = [x for x in manifest["cases"] if x["partition"] in ("heldout", "natural_challenge")]
        winner_freeze = read(HERE / "winners-freeze.json")
        if file_sha(HERE / "winners.json") != winner_freeze["sha256"]:
            raise ValueError("Candidate selection changed after its freeze")
        winners = read(HERE / "winners.json")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "rounds": rounds,
                    "cases": len(cases),
                    "models": list(ROUTES),
                    "expected_jobs": len(rounds) * len(cases) * 2,
                }
            )
        )
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be set")
    DATA.mkdir(parents=True, exist_ok=True)
    init_ledger()
    for r in rounds:
        if args.stage == "holdout":
            questions = {
                m: None if r == "heldout-baseline" else variants[winners["models"][m]["candidate_round"]]
                for m in ROUTES
            }
        else:
            questions = {m: variants[r] for m in ROUTES}
        run_round(r, [(m, c) for m in ROUTES for c in cases], questions, routes, prompts, key)


if __name__ == "__main__":
    main()
