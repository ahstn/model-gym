"""Frozen request construction and independent score/billing validation for the rehearsal."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("technical_quality", "aesthetics")
ARMS = ("baseline", "own_stats", "swapped_stats")
WIRE_PATH = HERE.parent / "dual-score-trial-2026-10-27/wire.py"
_spec = importlib.util.spec_from_file_location("rehearsal_joint_wire", WIRE_PATH)
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)
STATS_INSTRUCTION = (
    "\n\nAUTOMATIC PIXEL MEASUREMENTS — TECHNICAL QUALITY CHECK ONLY\n"
    "Use these measurements to check visible technical detail, noise and exposure in the supplied image. "
    "Measurements can be misleading because of subject matter, artistic effects, encoding or estimator failure. "
    "Resolve conflicts by inspecting the image; do not infer a defect, photographic eligibility or a score from "
    "a percentile alone. Assess aesthetics independently from the full image under the unchanged aesthetic "
    "rubric. Do not use these measurements as aesthetic anchors. Return the same two-task JSON object.\n"
)


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=wire._unique, parse_constant=wire._constant)


def number(value: object, low: float, high: float) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid finite numeric value")
    return float(value)


def amount(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("Invalid cost")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid cost")
    return result


def reported_cost(body: dict, route: dict) -> dict:
    """No token-price estimates substitute for a missing provider receipt."""
    result = {
        "price_usd": None,
        "price_basis": "missing_receipt",
        "gateway_price_usd": None,
        "upstream_price_usd": None,
    }
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return result
    try:
        price = amount(usage.get("cost"))
        if route["provider"] == "runinfra":
            micro = usage.get("runinfra", {}).get("cost_microcents")
            if type(micro) is not int or micro < 0 or Decimal(micro) / 100_000_000 != price:
                raise ValueError("RunInfra cost fields disagree")
            result.update(price_usd=str(price), price_basis="runinfra_usage_and_microcents")
            return result
        byok = usage.get("is_byok", False)
        if type(byok) is not bool:
            raise ValueError("Invalid BYOK flag")
        result["gateway_price_usd"] = str(price)
        if byok:
            upstream = amount(usage.get("cost_details", {}).get("upstream_inference_cost"))
            result.update(
                price_usd=str(price + upstream),
                upstream_price_usd=str(upstream),
                price_basis="byok_upstream_plus_gateway_charge",
            )
        else:
            result.update(price_usd=str(price), price_basis="openrouter_usage_cost")
    except (ValueError, TypeError, ArithmeticError, AttributeError):
        result.update(price_usd=None, price_basis="invalid_or_incomplete_receipt")
    return result


def payload(plan: dict, job: dict) -> dict:
    case, route = job["case"], plan["route_config"]
    raw = Path(case["image_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["view_sha256"]:
        raise ValueError("Delivered view changed: " + case["case_id"])
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode(), "detail": "high"},
    }
    if route["kind"] == "decisions":
        prompt = plan["decision_prompt"]
        # The selected prior experiment used this three-field Decisions contract.
        return {
            **route["settings"],
            "state": [prompt["common"], image],
            "questions": prompt["prompts"][route["requested_model"]]["questions"],
        }
    prompt = plan["teacher_prompt"]
    user = prompt["user"]
    if job["arm"] != "baseline":
        user += STATS_INSTRUCTION + job["stats_text"]
    return {
        **route["settings"],
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": [{"type": "text", "text": user}, image]},
        ],
        "response_format": prompt["response_format"],
    }


def distribution(answer: dict, *, allow_rounded: bool) -> dict | None:
    probabilities = answer.get("probabilities")
    if probabilities is None:
        return None
    if not isinstance(probabilities, dict) or set(probabilities) != set("01234"):
        raise ValueError("Five probability buckets required")
    values = [number(probabilities[str(i)], 0, 1) for i in range(5)]
    rounded = allow_rounded and all(abs(v - round(v, 2)) <= 1e-9 for v in values)
    if abs(sum(values) - 1) > (0.025000001 if rounded else 0.001):
        raise ValueError("Probability total exceeds its rounding bound")
    expected = sum(i * v for i, v in enumerate(values))
    if abs(expected - answer["score"]) > (0.055000001 if rounded else 0.025):
        raise ValueError("Score disagrees with probability-weighted mean")
    return probabilities


def parse_decisions(body: dict, route: dict) -> dict:
    parsed = {
        "ratings": {},
        "errors": {},
        "native_scores": {},
        "excluded_probability": None,
        "scope_floor_applied": False,
        "scope_validation_error": None,
    }
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != {*TASKS, "excluded"}:
        parsed["errors"] = dict.fromkeys(TASKS, "Invalid Decisions answer envelope")
        return parsed
    try:
        excluded = answers["excluded"]
        if not isinstance(excluded, dict) or excluded.get("type") != "noul":
            raise ValueError("Invalid excluded answer")
        parsed["excluded_probability"] = number(excluded.get("noul"), 0, 1)
        parsed["scope_floor_applied"] = parsed["excluded_probability"] >= 0.5
    except (ValueError, TypeError) as error:
        parsed["scope_validation_error"] = str(error)
    for task in TASKS:
        try:
            answer = answers[task]
            if not isinstance(answer, dict) or answer.get("type") != "score":
                raise ValueError("Invalid score answer type")
            native = 1 + number(answer.get("score"), 0, 4)
            distribution(answer, allow_rounded=route["requested_model"].startswith("typesafe/"))
            if "confidence" in answer:
                number(answer["confidence"], 0, 1)
            parsed["native_scores"][task] = native
            if parsed["scope_validation_error"]:
                raise ValueError("Scope probability invalid; native score retained separately")
            parsed["ratings"][task] = {
                "status": "ok",
                "score": 1.0 if parsed["scope_floor_applied"] else native,
                "evidence": None,
                "limitations": ["unsupported_domain"] if parsed["scope_floor_applied"] else [],
            }
        except (ValueError, TypeError, KeyError) as error:
            parsed["errors"][task] = str(error)
    return parsed


def parse_scores(body: dict, route: dict) -> dict:
    if body.get("model") not in route["accepted_models"]:
        raise ValueError("Returned model is not the frozen model revision")
    if route.get("accepted_providers") and body.get("provider") not in route["accepted_providers"]:
        raise ValueError("Returned provider is not allowed")
    if route["kind"] == "decisions":
        return parse_decisions(body, route)
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete completion")
    message = choices[0].get("message", {})
    if message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("Refusal or missing content")
    parsed = wire.parse_joint(message["content"])
    parsed["native_scores"] = {task: rating["score"] for task, rating in parsed["ratings"].items()}
    parsed["scope_floor_applied"] = any(
        "unsupported_domain" in rating.get("limitations", []) for rating in parsed["ratings"].values()
    )
    parsed["excluded_probability"] = None
    return parsed


def capacity_rejection(result: dict, body: dict, route: dict) -> dict | None:
    """Only the previously verified RunInfra pre-serve refusal is auto-retryable."""
    if route["provider"] != "runinfra" or result.get("transport_error") or result.get("http_status") != 429:
        return None
    try:
        error, identity = body["error"], body["request_id"]
        delay = amount(result["headers"]["retry-after"])
        if (
            error.get("type") != "rate_limit_error"
            or error.get("code") != "hosted_shared_model_concurrency_limit"
            or "choices" in body
            or body.get("usage")
            or not isinstance(identity, str)
            or not identity.startswith("req_")
            or result["headers"].get("x-request-id") != identity
        ):
            return None
        return {"retry_after_seconds": float(delay), "reason": error["code"]}
    except (ValueError, KeyError, TypeError, ArithmeticError, AttributeError):
        return None


def interpret(result: dict, route: dict) -> dict:
    row = {
        "status": "unknown" if result.get("transport_error") else "failed",
        "reason": None,
        "technical_score": None,
        "aesthetic_score": None,
        "technical_score_native": None,
        "aesthetic_score_native": None,
        "task_valid": dict.fromkeys(TASKS, False),
        "validation_errors": {},
        "price_usd": None,
        "price_basis": "missing_receipt",
        "usage": {},
    }
    try:
        body = load_json(result["body"]) if result.get("body") is not None else None
        if not isinstance(body, dict):
            raise TypeError("No JSON response object")
        row.update(reported_cost(body, route))
        row.update(
            model_returned=body.get("model"),
            provider_returned=body.get("provider"),
            provider_response_id=body.get("id"),
            usage=body.get("usage", {}),
        )
        details = row["usage"].get("completion_tokens_details") if isinstance(row["usage"], dict) else None
        row["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
        retry = capacity_rejection(result, body, route)
        if retry:
            row.update(status="rejected_retryable", **retry)
            return row
        if result.get("transport_error") or result.get("http_status") != 200 or body.get("error"):
            raise ValueError("Transport, HTTP or provider error")
        effort = result.get("headers", {}).get("x-runinfra-reasoning-effort")
        if route["provider"] == "runinfra" and effort is not None and effort != "high":
            raise ValueError("Returned reasoning effort differs from requested high")
        parsed = parse_scores(body, route)
        row.update(
            status="invalid" if parsed["errors"] else "valid",
            validation_errors=parsed["errors"],
            task_valid={t: t in parsed["ratings"] for t in TASKS},
            scope_excluded=parsed["scope_floor_applied"],
            scope_floor_applied=parsed["scope_floor_applied"],
            excluded_probability=parsed["excluded_probability"],
            ratings=parsed["ratings"],
            wire_scores=parsed.get("wire_scores"),
            scope_validation_error=parsed.get("scope_validation_error"),
        )
        for task, name in zip(TASKS, ("technical", "aesthetic"), strict=True):
            row[name + "_score"] = parsed["ratings"].get(task, {}).get("score")
            row[name + "_score_native"] = parsed["native_scores"].get(task)
        if row["price_usd"] is None:
            row["reason"] = "cost_unavailable"
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
        row["reason"] = str(error)
    return row
