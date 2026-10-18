"""Frozen, resumable technical-quality pilot runs with separate attempt records."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from visual_quality import openrouter
from visual_quality.labels import observation, parse_rating, validate_observation
from visual_quality.storage import atomic_json, file_sha256, write_jsonl
from visual_quality.views import load_views

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
RUN_VERSION = "technical-quality-pilot-v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def _run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process owns this labelling run") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _prompt_bundle(directory: Path) -> dict:
    names = ["system.txt", "technical-quality.txt", "response.schema.json"]
    files = {name: directory.joinpath(name).read_text() for name in names}
    schema = json.loads(files["response.schema.json"])["$defs"]["technical_quality"]
    # Use the common provider schema subset. The ingestion validator enforces the
    # status/score relation, evidence length, and unique limitations locally.
    schema.pop("oneOf", None)
    schema["properties"]["task"] = {"type": "string", "enum": ["technical_quality"]}
    schema["properties"]["status"]["type"] = "string"
    schema["properties"]["score"] = {
        "type": ["number", "null"],
        "enum": [n / 10 for n in range(10, 51)] + [None],
    }
    schema["properties"]["limitations"].pop("uniqueItems", None)
    schema["properties"]["limitations"].pop("maxItems", None)
    schema["properties"]["limitations"]["items"]["type"] = "string"
    return {
        "version": "visual-rating-v1",
        "sha256": _digest(files),
        "system": files["system.txt"],
        "user": files["technical-quality.txt"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "technical_quality_v1",
                "strict": True,
                "schema": schema,
            },
        },
    }


def _price_bound(info: dict, max_tokens: int) -> dict:
    """Reserve a full context window at maximum listed rates before each call.

    This pilot supports token-billed models only. Fail closed on additional fee
    types rather than silently treating unknown image/request prices as zero.
    """
    prices = info["pricing"]
    allowed = {"prompt", "completion", "input_cache_read", "input_cache_write", "overrides"}
    tiers = [prices, *prices.get("overrides", [])]
    input_rate, output_rate = Decimal(0), Decimal(0)
    for tier in tiers:
        for key, value in tier.items():
            if key in {"min_prompt_tokens", "utc_days", "utc_start", "utc_end", "overrides"}:
                continue
            amount = Decimal(str(value))
            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid catalogue price")
            if key not in allowed and amount:
                raise ValueError(f"Pilot budget cannot bound this fee type: {key}")
        input_rate = max(
            input_rate,
            *(Decimal(str(tier.get(k, 0))) for k in ("prompt", "input_cache_read", "input_cache_write")),
        )
        output_rate = max(output_rate, Decimal(str(tier.get("completion", 0))))
    if "prompt" not in prices or "completion" not in prices:
        raise ValueError("Model has no token price")
    context = info.get("context_length")
    if type(context) is not int or context <= 0:
        raise ValueError("Model has no context bound")
    reserve = input_rate * context + output_rate * max_tokens
    return {
        "reserve_usd": str(reserve),
        "max_price": {
            "prompt": float(input_rate * 1_000_000),
            "completion": float(output_rate * 1_000_000),
            "request": 0,
            "image": 0,
        },
    }


def _payload(view: dict, views_dir: Path, run: dict) -> dict:
    data = (views_dir / view["image_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != view["view_sha256"]:
        raise ValueError("Delivered view changed before request")
    prompt = run["prompt"]
    return {
        **run["settings"],
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt["user"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(data).decode(),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "response_format": prompt["response_format"],
    }


def _cost(body: dict) -> Decimal | None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = Decimal(str(value))
    except ArithmeticError:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _interpret(result: dict, request: dict, view: dict, run: dict) -> tuple[dict, dict | None]:
    details = {
        "attempt_id": request["attempt_id"],
        "view_id": view["view_id"],
        "cost_usd": None,
        "status": "failed",
        "reason": None,
    }
    if result["transport_error"]:
        details["reason"] = "transport_outcome_unknown"
        return details, None
    try:
        body = json.loads(result["body"] or "null")
        if not isinstance(body, dict):
            raise TypeError("Non-object response")
    except (ValueError, TypeError):
        details["reason"] = "invalid_response_envelope"
        return details, None
    cost = _cost(body)
    details["cost_usd"] = str(cost) if cost is not None else None
    if result["http_status"] != 200 or body.get("error"):
        details["reason"] = "http_or_provider_error"
        details["retry_after"] = result["retry_after"]
        return details, None
    try:
        if body.get("model") != run["settings"]["model"]:
            raise ValueError("Returned model differs from the frozen model")
        choices = body["choices"]
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
        ):
            raise ValueError("Response did not finish with one complete choice")
        message = choices[0]["message"]
        if not isinstance(message, dict):
            raise TypeError("Response message must be an object")
        if message.get("refusal"):
            raise ValueError("Provider refused the request")
        rating = parse_rating(message["content"])
        label = observation(
            view,
            rating,
            {
                "run_id": run["run_id"],
                "attempt_id": request["attempt_id"],
                "provider": "openrouter",
                "model_requested": run["settings"]["model"],
                "model_returned": body["model"],
                "provider_returned": body.get("provider"),
                "prompt_version": run["prompt"]["version"],
                "prompt_sha256": run["prompt"]["sha256"],
                "request_sha256": request["request_sha256"],
                "created_at": request["created_at"],
                "usage": body.get("usage", {}),
            },
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        details["reason"] = f"invalid_rating: {error}"
        return details, None
    details["status"] = "valid"
    details["reason"] = "cost_unavailable" if cost is None else None
    return details, label


def _report(output: Path, run: dict, views: list[dict]) -> dict:
    labels, attempts = [], []
    for view in views:
        directory = output / "attempts" / _digest(view["view_id"])
        request_path = directory / "request.json"
        if not request_path.exists():
            if directory.exists():
                raise ValueError("Incomplete attempt directory; inspect it before resuming")
            continue
        request = json.loads(request_path.read_text())
        _validate_request(request, directory, view, run)
        result_path = directory / "result.json"
        if not result_path.exists():
            attempts.append(
                {
                    "attempt_id": request["attempt_id"],
                    "view_id": view["view_id"],
                    "status": "unknown",
                    "cost_usd": None,
                    "reason": "interrupted_request",
                }
            )
            continue
        result = json.loads(result_path.read_text())
        details, label = _interpret(result, request, view, run)
        attempts.append(details)
        if label:
            validate_observation(label)
            labels.append(label)
    spent = sum((Decimal(a["cost_usd"]) for a in attempts if a["cost_usd"] is not None), Decimal(0))
    blocked = any(a["status"] != "valid" or a["cost_usd"] is None for a in attempts)
    report = {
        "schema_version": RUN_VERSION,
        "run_id": run["run_id"],
        "model": run["settings"]["model"],
        "selected": len(views),
        "attempted": len(attempts),
        "valid_observations": len(labels),
        "scored": sum(x["technical_quality"]["status"] == "ok" for x in labels),
        "known_cost_usd": str(spent),
        "unpriced_attempts": sum(a["cost_usd"] is None for a in attempts),
        "status": "blocked" if blocked else ("complete" if len(attempts) == len(views) else "ready"),
        "training_ready": False,
        "review_status": "pending",
        "attempts": attempts,
        "components": dict(Counter(v["component"] for v in views)),
    }
    write_jsonl(output / "observations.jsonl", labels)
    atomic_json(output / "report.json", report)
    return report


def _validate_run(run: dict) -> None:
    copy = dict(run)
    expected = copy.pop("contract_sha256", None)
    if expected != _digest(copy) or run["run_id"] != _digest(run["identity"]):
        raise ValueError("Frozen run metadata failed its integrity check")


def _validate_request(request: dict, directory: Path, view: dict, run: dict) -> None:
    payload = _payload(view, Path(run["identity"]["views_dir"]), run)
    if (
        request["attempt_id"] != directory.name
        or request["view_id"] != view["view_id"]
        or request["view_sha256"] != view["view_sha256"]
        or request["request_sha256"] != _digest(payload)
    ):
        raise ValueError("Stored attempt does not match the frozen view and request")


def verify_labels(views_dir: Path, output_dir: Path) -> dict:
    """Read-only validation of observations against frozen inputs and raw replies."""
    views_dir, output_dir = views_dir.resolve(), output_dir.resolve()
    views = load_views(views_dir)
    run = json.loads((output_dir / "run.json").read_text())
    _validate_run(run)
    if (
        run["identity"]["views_dir"] != str(views_dir)
        or run["identity"]["views_sha256"] != file_sha256(views_dir / "views.jsonl")
        or run["identity"]["preparation_sha256"] != file_sha256(views_dir / "preparation.json")
    ):
        raise ValueError("Prepared views differ from this run")
    expected = []
    for view in views:
        directory = output_dir / "attempts" / _digest(view["view_id"])
        if not (directory / "request.json").exists():
            continue
        request = json.loads((directory / "request.json").read_text())
        _validate_request(request, directory, view, run)
        if (directory / "result.json").exists():
            _, label = _interpret(json.loads((directory / "result.json").read_text()), request, view, run)
            if label:
                expected.append(label)
    actual = [json.loads(line) for line in (output_dir / "observations.jsonl").read_text().splitlines()]
    for label in actual:
        validate_observation(label)
    if actual != expected:
        raise ValueError("Observations differ from the preserved responses")
    return {
        "status": "verified",
        "observations": len(actual),
        "training_ready": False,
        "note": "Artifact integrity only; human agreement has not been measured",
    }


def _initialize(output: Path, identity: dict, prompt: dict) -> dict:
    path = output / "run.json"
    if path.exists():
        run = json.loads(path.read_text())
        _validate_run(run)
        if run["identity"] != identity or run["prompt"] != prompt:
            raise ValueError("Run settings, prompt, or prepared views changed; use a new run directory")
        return run
    if any(p.name != ".lock" for p in output.iterdir()):
        raise ValueError("Run directory is not empty and has no run.json")
    info = openrouter.model_info(identity["model"])
    bound = _price_bound(info, identity["max_tokens"])
    settings = {
        "model": identity["model"],
        "stream": False,
        "max_tokens": identity["max_tokens"],
        "reasoning": {"enabled": False},
        "provider": {"require_parameters": True, "allow_fallbacks": False, "max_price": bound["max_price"]},
    }
    if "temperature" in info.get("supported_parameters", []):
        settings["temperature"] = 0
    run = {
        "schema_version": RUN_VERSION,
        "run_id": _digest(identity),
        "created_at": _now(),
        "identity": identity,
        "prompt": prompt,
        "settings": settings,
        "model_snapshot": info,
        "budget_bound": bound,
        "purpose": "pipeline_smoke_not_teacher_validation",
    }
    run["contract_sha256"] = _digest(run)
    atomic_json(path, run)
    return run


def label_pilot(
    views_dir: Path,
    output_dir: Path,
    prompt_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    max_cost_usd: str = "1.00",
    dry_run: bool = False,
) -> dict:
    """One attempt per frozen view; a failed or uncertain attempt blocks resumption."""
    views_dir, output_dir = views_dir.resolve(), output_dir.resolve()
    if views_dir.is_relative_to(output_dir) or output_dir.is_relative_to(views_dir):
        raise ValueError("Use a separate run directory outside the prepared views")
    if not 128 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 128 and 4096")
    budget = Decimal(str(max_cost_usd))
    if not budget.is_finite() or not 0 < budget <= 10:
        raise ValueError("Small-pilot budget must be greater than zero and at most 10 USD")
    views = load_views(views_dir)
    if not 1 <= len(views) <= 100 or any(v["split"] != "train" for v in views):
        raise ValueError("This pilot requires 1..100 prepared training views")
    prompt = _prompt_bundle(prompt_dir)
    identity = {
        "schema_version": RUN_VERSION,
        "views_dir": str(views_dir),
        "views_sha256": file_sha256(views_dir / "views.jsonl"),
        "preparation_sha256": file_sha256(views_dir / "preparation.json"),
        "model": model,
        "max_tokens": max_tokens,
        "max_cost_usd": str(budget),
        "prompt_sha256": prompt["sha256"],
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ["labelling.py", "labels.py", "openrouter.py"]
        },
    }
    if dry_run:
        return {
            "status": "dry_run",
            "selected": len(views),
            "identity": identity,
            "api_calls": 0,
            "training_ready": False,
        }
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not present; load it through your environment")
    with _run_lock(output_dir):
        run = _initialize(output_dir, identity, prompt)
        report = _report(output_dir, run, views)
        if report["status"] in {"blocked", "complete"}:
            return report
        for view in views:
            directory = output_dir / "attempts" / _digest(view["view_id"])
            if directory.exists():
                continue
            spent = Decimal(report["known_cost_usd"])
            if spent + Decimal(run["budget_bound"]["reserve_usd"]) > budget:
                report["status"] = "budget_stopped"
                atomic_json(output_dir / "report.json", report)
                break
            payload = _payload(view, views_dir, run)
            request = {
                "attempt_id": directory.name,
                "created_at": _now(),
                "view_id": view["view_id"],
                "view_sha256": view["view_sha256"],
                "request_sha256": _digest(payload),
            }
            atomic_json(directory / "request.json", request)
            result = openrouter.complete(api_key, payload)
            atomic_json(directory / "result.json", result)
            report = _report(output_dir, run, views)
            print(
                f"Pilot: {report['attempted']}/{len(views)} attempted; "
                f"{report['valid_observations']} valid; known cost ${report['known_cost_usd']}",
                flush=True,
            )
            if report["status"] == "blocked":
                break
        return report
