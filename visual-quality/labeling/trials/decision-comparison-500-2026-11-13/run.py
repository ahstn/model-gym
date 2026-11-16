"""Bounded, journaled Decisions calls for the frozen 500-image comparison."""

import argparse
import base64
import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests
from prepare import DATA, sha, write

from visual_quality.scoring.api_contract import encoded, interpret


def now():
    return datetime.now(UTC).isoformat()


class Runner:
    def __init__(self, model):
        self.model = model
        self.plan = json.loads((DATA / "plan.json").read_text())
        self.plan_sha = sha(DATA / "plan.json")
        assert sha(DATA / "subset.json") == self.plan["subset_sha256"]
        for path, expected in self.plan["source_pins"].items():
            assert sha(path) == expected, f"Changed frozen code: {path}"
        self.route = self.plan["routes"][model]
        self.root = DATA / "results" / model
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.accounted = 0.0
        self.done = 0
        for path in self.root.glob("attempts/*/*/request.json"):
            result = path.with_name("result.json")
            cost = json.loads(result.read_text()).get("price_usd") if result.exists() else None
            self.accounted += float(cost) if cost is not None else self.plan["request_reservation_usd"]

    def once(self, case, body, request_sha, number):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Admission stopped")
            reserve = self.plan["request_reservation_usd"]
            if self.accounted + reserve > self.plan["budget_per_model_usd"]:
                self.stop.set()
                raise RuntimeError("Budget guard reached")
            self.accounted += reserve
        folder = self.root / "attempts" / case["case_id"] / f"{number:02}"
        folder.mkdir(parents=True, exist_ok=False)
        write(
            folder / "request.json",
            {
                "case_id": case["case_id"],
                "model": self.route["requested_model"],
                "plan_sha256": self.plan_sha,
                "request_sha256": request_sha,
                "view_sha256": case["view_sha256"],
                "attempt": number,
                "created_at": now(),
            },
        )
        result = {
            "request_started_at": now(),
            "http_status": None,
            "body": None,
            "headers": {},
            "transport_error": None,
        }
        secret = os.environ["OPENROUTER_API_KEY"]
        start = time.perf_counter()
        try:
            response = requests.post(
                self.route["endpoint"],
                data=body,
                headers={
                    "Authorization": "Bearer " + secret,
                    "Content-Type": "application/json",
                },
                timeout=(30, self.plan["timeout_seconds"]),
                allow_redirects=False,
            )
            result.update(
                latency_seconds=time.perf_counter() - start,
                response_received_at=now(),
                http_status=response.status_code,
                body=response.text.replace(secret, "[REDACTED]"),
                headers={
                    k.lower(): v
                    for k, v in response.headers.items()
                    if k.lower().startswith(("retry-after", "x-ratelimit"))
                },
            )
        except requests.RequestException as error:
            result.update(
                latency_seconds=time.perf_counter() - start,
                response_received_at=now(),
                transport_error=type(error).__name__,
            )
        parsed = interpret(result, self.route)
        result.update(parsed=parsed, price_usd=parsed["price_usd"])
        write(folder / "result.json", result)
        with self.lock:
            if parsed["price_usd"] is not None:
                self.accounted += float(parsed["price_usd"]) - reserve
        if result["http_status"] in (401, 402, 403, 404):
            self.stop.set()
        return result

    def case(self, case):
        import hashlib

        raw = Path(case["trial_image_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == case["view_sha256"]
        body = encoded(
            {
                "model": self.route["requested_model"],
                "state": [
                    self.plan["common"],
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(raw).decode(),
                            "detail": "high",
                        },
                    },
                ],
                "questions": self.plan["questions"],
            }
        )
        request_sha = hashlib.sha256(body).hexdigest()
        output = self.root / "observations" / (case["case_id"] + ".json")
        if output.exists():
            existing = json.loads(output.read_text())
            assert existing["request_sha256"] == request_sha and existing["plan_sha256"] == self.plan_sha
            return
        attempts = sorted((self.root / "attempts" / case["case_id"]).glob("*/request.json"))
        results = []
        for path in attempts:
            previous = json.loads(path.read_text())
            assert previous["request_sha256"] == request_sha and previous["plan_sha256"] == self.plan_sha
            result_path = path.with_name("result.json")
            results.append(
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"transport_error": "interrupted_unknown", "price_usd": None}
            )
        while len(results) < self.plan["max_attempts"]:
            if results:
                last = results[-1]
                if not last.get("transport_error") and last.get("http_status") not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    break
                delay = max(5 * len(results), float(last.get("headers", {}).get("retry-after", "5")))
                if delay > 300:
                    self.stop.set()
                    raise RuntimeError("Long provider quota wait; preserve progress")
                time.sleep(delay)
            results.append(self.once(case, body, request_sha, len(results) + 1))
        final = results[-1]
        parsed = final.get("parsed", {"status": "unknown", "technical_score": None, "aesthetic_score": None})
        row = {
            **parsed,
            "case_id": case["case_id"],
            "model": self.model,
            "arm": "baseline",
            "view_sha256": case["view_sha256"],
            "request_sha256": request_sha,
            "plan_sha256": self.plan_sha,
            "latency_seconds": final.get("latency_seconds"),
            "latency_kind": "http_request_through_complete_response"
            if not final.get("transport_error")
            else "http_attempt_until_transport_failure",
            "request_started_at": final.get("request_started_at"),
            "response_received_at": final.get("response_received_at"),
            "attempt_count": len(results),
            "training_ready": False,
            "raw_response_path": str(
                self.root / "attempts" / case["case_id"] / f"{len(results):02}" / "result.json"
            ),
        }
        write(output, row)
        with self.lock:
            self.done += 1
            if self.done % 25 == 0 or self.done == 1:
                print(
                    json.dumps(
                        {
                            "model": self.model,
                            "new_completed": self.done,
                            "guard_accounted_usd": self.accounted,
                        }
                    ),
                    flush=True,
                )
        if parsed["status"] != "valid":
            print(
                json.dumps(
                    {
                        "model": self.model,
                        "case_id": case["case_id"],
                        "status": parsed["status"],
                        "http": final.get("http_status"),
                        "errors": parsed.get("validation_errors"),
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["perplexity", "gemini"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    runner = Runner(args.model)
    with (runner.root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = json.loads((DATA / "subset.json").read_text())[: args.limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(runner.case, cases):
                pass
    print(json.dumps({"model": args.model, "finished": len(cases), "new_completed": runner.done}), flush=True)


if __name__ == "__main__":
    main()
