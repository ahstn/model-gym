"""Bounded RunInfra comparison; preserve every attempt and never retry it."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import requests

from visual_quality.labelling import _prompt_bundle, verify_labels
from visual_quality.labels import parse_rating
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl
from visual_quality.views import load_views

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / "experiments/prompt-calibration-2026-10-18"
DATA = PROJECT / "data/runinfra-comparison-2026-10-25"
VIEWS = PROJECT / "data/prompt-calibration-2026-10-18/a/views"
HISTORICAL = Path("/Users/ahstn/git/model-gym/visual-quality/data/prompt-calibration-2026-10-18/a")
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SLUG = "deepseek-v4-1-flash-1017"
ENDPOINT = "https://api.runinfra.ai/v1/chat/completions"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def payload(run: dict, case: dict) -> dict:
    view = case["view"]
    raw = (VIEWS / view["image_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != view["view_sha256"]:
        raise ValueError("Frozen image bytes changed")
    return {
        **run["settings"],
        "messages": [
            {"role": "system", "content": run["prompt"]["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": run["prompt"]["user"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(raw).decode(),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "response_format": run["prompt"]["response_format"],
    }


def encoded(body: dict) -> bytes:
    return json.dumps(body, allow_nan=False).encode()


def freeze(catalog_path: Path) -> dict:
    """Run with the documented path wrapper to verify historical records."""
    if (DATA / "run.json").exists():
        raise ValueError("Run already frozen; use verify or run")
    catalog = json.loads(catalog_path.read_text())
    if len(catalog) != 1:
        raise ValueError("Expected one exact model catalog entry")
    model = catalog[0]
    if (
        model["id"] != SLUG
        or model["availability"] != "available"
        or not {"text", "image"}.issubset(model["input_modalities"])
        or not model["json_mode"]
        or "none" not in model["reasoning_efforts"]
    ):
        raise ValueError("Required model capabilities are unavailable")
    if model["pricing"] != {"input": 0.15, "output": 0.6}:
        raise ValueError("Published prices changed; review the budget")
    baseline = verify_labels(HISTORICAL / "views", HISTORICAL / "v1")
    if baseline["observations"] != 50:
        raise ValueError("Expected all 50 historical observations")
    verified = load_views(HISTORICAL / "views")
    views = read_jsonl(VIEWS / "views.jsonl")
    if views != verified:
        raise ValueError("Active views differ from the preserved historical views")
    by_id = {v["view_id"]: v for v in views}
    cases = json.loads((OLD / "manifest.json").read_text())["stages"]["a"]["cases"]
    prompt = _prompt_bundle(PROJECT / "prompts/visual-rating-v1")
    old_run = json.loads((HISTORICAL / "v1/run.json").read_text())
    if prompt != old_run["prompt"]:
        raise ValueError("Prompt differs from the baseline")
    files = [
        HERE / "runner.py",
        HERE / "report.py",
        HERE / "protocol.md",
        OLD / "report.py",
        OLD / "manifest.json",
        OLD / "references/a.jsonl",
        OLD / "references/a-freeze.json",
        VIEWS / "views.jsonl",
        VIEWS / "preparation.json",
        PROJECT / "data/prompt-calibration-2026-10-18/a/v1/observations.jsonl",
        PROJECT / "src/visual_quality/labels.py",
        PROJECT / "src/visual_quality/labelling.py",
    ]
    files += list((PROJECT / "prompts/visual-rating-v1").glob("*"))
    run = {
        "schema_version": "runinfra-comparison-run-v1",
        "created_at": now(),
        "endpoint": ENDPOINT,
        "requested_model": MODEL,
        "model_snapshot": model,
        "prompt": prompt,
        "settings": {
            "model": SLUG,
            "max_tokens": 1024,
            "temperature": 0,
            "reasoning_effort": "none",
            "stream": False,
        },
        "policy": {
            "automatic_retries": False,
            "fallback": False,
            "view_policy": "original PNG bytes only; oversized views remain missing",
            "reference_kind": "frozen_assistant_judgments",
            "human_gold": False,
        },
        "budget": {
            "max_cost_usd": "1.00",
            "reserve_per_request_usd": str(
                Decimal(str(model["pricing"]["input"])) * model["context_window"] / 1_000_000
                + Decimal(str(model["pricing"]["output"])) * 1024 / 1_000_000
            ),
        },
        "inputs": {str(p.relative_to(PROJECT)): file_sha256(p) for p in files},
        "cases": [],
        "training_ready": False,
    }
    for source in cases:
        view = by_id[source["view_id"]]
        if source["image_id"] != view["image_id"] or source["view_sha256"] != view["view_sha256"]:
            raise ValueError("Case/view identity mismatch")
        case = {"case_id": source["case_id"], "family_id": source["family_id"], "view": view}
        body = payload(run, case)
        size = len(encoded(body))
        case.update(
            request_sha256=digest(body),
            request_bytes=size,
            skip_reason="request_exceeds_provider_limit" if size > model["max_request_bytes"] else None,
        )
        run["cases"].append(case)
    run["run_id"] = digest(run)
    DATA.mkdir(parents=True, exist_ok=True)
    atomic_json(DATA / "run.json", run)
    return run


def read_run() -> dict:
    run = json.loads((DATA / "run.json").read_text())
    identity = {k: v for k, v in run.items() if k != "run_id"}
    if digest(identity) != run["run_id"]:
        raise ValueError("Run metadata changed")
    for name, expected in run["inputs"].items():
        if file_sha256(PROJECT / name) != expected:
            raise ValueError(f"Pinned input changed: {name}")
    for case in run["cases"]:
        if digest(payload(run, case)) != case["request_sha256"]:
            raise ValueError("Request differs from frozen inputs")
    return run


def cost(body: dict) -> Decimal | None:
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return None
    try:
        value = usage.get("cost")
        if isinstance(value, bool) or value is None:
            return None
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0:
            return None
        micro = usage.get("runinfra", {}).get("cost_microcents")
        if type(micro) is not int or micro < 0 or Decimal(micro) / 100_000_000 != amount:
            return None
        return amount
    except (ValueError, TypeError, ArithmeticError, AttributeError):
        return None


def interpret(run: dict, case: dict, request: dict, result: dict) -> tuple[dict, dict | None]:
    details = {"case_id": case["case_id"], "status": "failed", "cost_usd": None, "reason": None}
    if result["transport_error"]:
        details["reason"] = "transport_outcome_unknown"
        return details, None
    try:
        body = json.loads(result["body"])
        if not isinstance(body, dict):
            raise TypeError("Non-object envelope")
        amount = cost(body)
        details["cost_usd"] = str(amount) if amount is not None else None
        if result["http_status"] != 200 or body.get("error"):
            details["reason"] = "http_or_provider_error"
            return details, None
        if body.get("model") not in {MODEL, SLUG}:
            raise ValueError("Unexpected returned model")
        choices = body["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("Incomplete response")
        message = choices[0]["message"]
        if message.get("refusal"):
            raise ValueError("Refusal")
        rating = parse_rating(message["content"])
        view = case["view"]
        record = {
            "schema_version": "runinfra-label-observation-v1",
            "label_kind": "teacher_scalar",
            "case_id": case["case_id"],
            "image_id": view["image_id"],
            "view_id": view["view_id"],
            "source_sha256": view["source_sha256"],
            "view_sha256": view["view_sha256"],
            "delivered_sha256": view["view_sha256"],
            "view_comparison": "identical_bytes",
            "technical_quality": rating,
            "aesthetics": {
                "status": "not_requested",
                "score": None,
                "reason": "Separate validation required",
            },
            "photo_domain_eligible": view["photo_domain_eligible"],
            "review_status": "pending_human_review",
            "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
            "provenance": {
                "provider": "runinfra",
                "model_requested": run["settings"]["model"],
                "model_returned": body["model"],
                "run_id": run["run_id"],
                "attempt_id": case["case_id"],
                "prompt_version": run["prompt"]["version"],
                "prompt_sha256": run["prompt"]["sha256"],
                "request_sha256": request["request_sha256"],
                "created_at": request["created_at"],
                "usage": body.get("usage", {}),
            },
        }
        details.update(status="valid", reason=None if amount is not None else "cost_unavailable")
        return details, record
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        details["reason"] = f"invalid_rating: {type(error).__name__}"
        return details, None


def reconstruct(run: dict) -> tuple[dict, list[dict]]:
    observations, attempts = [], []
    expected_dirs = {c["case_id"] for c in run["cases"] if not c["skip_reason"]}
    attempt_root = DATA / "attempts"
    if attempt_root.exists() and not {p.name for p in attempt_root.iterdir()}.issubset(expected_dirs):
        raise ValueError("Unexpected attempt directory")
    for case in run["cases"]:
        folder = attempt_root / case["case_id"]
        if not folder.exists():
            continue
        if not (folder / "request.json").exists():
            raise ValueError("Incomplete attempt directory; inspect manually")
        request = json.loads((folder / "request.json").read_text())
        expected = {
            "case_id": case["case_id"],
            "run_id": run["run_id"],
            "request_sha256": case["request_sha256"],
            "request_bytes": case["request_bytes"],
        }
        if any(request.get(k) != v for k, v in expected.items()):
            raise ValueError("Attempt request metadata differs from the run")
        if not (folder / "result.json").exists():
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "status": "unknown",
                    "cost_usd": None,
                    "reason": "interrupted_request",
                }
            )
            continue
        details, record = interpret(run, case, request, json.loads((folder / "result.json").read_text()))
        attempts.append(details)
        if record is not None:
            observations.append(record)
    blocked = [c["case_id"] for c in run["cases"] if c["skip_reason"]]
    stopped = any(a["status"] != "valid" or a["cost_usd"] is None for a in attempts)
    status = "blocked" if stopped else "complete" if len(attempts) + len(blocked) == 50 else "partial"
    if status == "complete" and blocked:
        status = "complete_with_size_exclusions"
    return {
        "schema_version": "runinfra-run-report-v1",
        "run_id": run["run_id"],
        "status": status,
        "planned": 50,
        "attempted": len(attempts),
        "valid_observations": len(observations),
        "scored": sum(r["technical_quality"]["score"] is not None for r in observations),
        "known_cost_usd": str(
            sum((Decimal(a["cost_usd"]) for a in attempts if a["cost_usd"] is not None), Decimal(0))
        ),
        "unpriced_attempts": sum(a["cost_usd"] is None for a in attempts),
        "blocked_case_ids": blocked,
        "attempts": attempts,
        "training_ready": False,
    }, observations


def save_report(run: dict) -> dict:
    report, observations = reconstruct(run)
    write_jsonl(DATA / "observations.jsonl", observations)
    atomic_json(DATA / "report.json", report)
    return report


def execute(run: dict) -> dict:
    key = os.environ.get("RUN_INFRA_API_KEY")
    if not key:
        raise ValueError("RUN_INFRA_API_KEY is unavailable")
    with (DATA / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = save_report(run)
        if report["status"] == "blocked":
            raise ValueError("An existing failed or uncertain attempt prevents further calls")
        for case in run["cases"]:
            folder = DATA / "attempts" / case["case_id"]
            if case["skip_reason"] or folder.exists():
                continue
            reserve = Decimal(run["budget"]["reserve_per_request_usd"])
            if Decimal(report["known_cost_usd"]) + reserve > Decimal(run["budget"]["max_cost_usd"]):
                raise ValueError("Next request exceeds the admission budget")
            body = payload(run, case)
            if digest(body) != case["request_sha256"]:
                raise ValueError("Request changed")
            request = {k: case[k] for k in ("case_id", "request_sha256", "request_bytes")}
            request.update(run_id=run["run_id"], created_at=now())
            folder.mkdir(parents=True)
            atomic_json(folder / "request.json", request)
            try:
                response = requests.post(
                    ENDPOINT,
                    data=encoded(body),
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    timeout=(15, 180),
                    allow_redirects=False,
                )
                headers = {
                    k.lower(): v.replace(key, "[REDACTED]")
                    for k, v in response.headers.items()
                    if k.lower().startswith(("x-runinfra-", "x-request-id", "retry-after"))
                }
                result = {
                    "http_status": response.status_code,
                    "body": response.text.replace(key, "[REDACTED]"),
                    "headers": headers,
                    "transport_error": None,
                    "received_at": now(),
                }
            except requests.RequestException as error:
                result = {
                    "http_status": None,
                    "body": None,
                    "headers": {},
                    "transport_error": type(error).__name__,
                    "received_at": now(),
                }
            atomic_json(folder / "result.json", result)
            report = save_report(run)
            print(
                json.dumps(
                    {
                        "attempted": report["attempted"],
                        "scored": report["scored"],
                        "known_cost_usd": report["known_cost_usd"],
                        "status": report["status"],
                    }
                ),
                flush=True,
            )
            if report["status"] == "blocked":
                break
        return report


def verify(run: dict) -> dict:
    report, observations = reconstruct(run)
    if (
        json.loads((DATA / "report.json").read_text()) != report
        or read_jsonl(DATA / "observations.jsonl") != observations
    ):
        raise ValueError("Saved exports differ from preserved replies")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        run = freeze(args.catalog)
        report = save_report(run)
    else:
        run = read_run()
        report = execute(run) if args.command == "run" else verify(run)
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
