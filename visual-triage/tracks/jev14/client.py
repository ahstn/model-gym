"""Jev 1.14 native-image Decisions adapter; one call, no automatic retries."""

import hashlib
import json
import os
import time
from datetime import UTC, datetime

import requests
from PIL import Image

from bench_protocol import DECISIONS, digest, image_part, number, probability_distribution
from bench_runner import effective_cost

MODEL = "typesafe/jev-1.14"
REVISION = "typesafe/jev-1.14-20261014"
HEADERS = {"TypeSafe-Danger-Zone-Long-Context": "true", "TypeSafe-Danger-Zone-Images": "true"}
CONTEXT_TOKENS = 262144


def image_info(path):
    with Image.open(path) as image:
        width, height = image.size
        mime = Image.MIME.get(image.format)
        animated = getattr(image, "is_animated", False)
        image.verify()
    size = path.stat().st_size
    eligible = (
        not animated
        and mime in {"image/jpeg", "image/png"}
        and 256 <= width <= 1568
        and 256 <= height <= 1568
        and 78400 <= width * height <= 2458624
        and size <= 16 * 1024 * 1024
    )
    return {
        "image_width": width,
        "image_height": height,
        "image_bytes": size,
        "image_mime": mime,
        "eligible": eligible,
    }


def make_payload(root, case, prompt):
    info = image_info(root / case["image"])
    if not info["eligible"]:
        raise ValueError("Image is outside Jev's published input limits")
    for key in ("image_width", "image_height", "image_bytes"):
        if info[key] != case[key]:
            raise ValueError("Frozen image properties changed")
    payload = {
        "model": MODEL,
        "state": [prompt["context"], image_part(case, root)],
        "questions": prompt["questions"],
    }
    if len(json.dumps(payload).encode()) > 32 * 1024 * 1024:
        raise ValueError("Request exceeds Jev's body limit")
    return payload


def normalize(response, questions):
    """Validate typed values; optional probability fields remain optional, never fabricated."""
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError("Decision answer fields do not match the task")
    result = {}
    for key, question in questions.items():
        answer, kind = answers[key], question["type"]
        if answer["type"] != kind:
            raise ValueError("Unexpected answer type")
        if "confidence" in answer:
            result[key + "_native_confidence"] = number(answer["confidence"], 0, 1)
        if kind == "noul":
            result[key] = number(answer["noul"], 0, 1)
            continue
        expected = question["criteria"] if kind == "choice" else list(map(str, range(5)))
        probabilities = probability_distribution(answer, expected) if "probabilities" in answer else None
        if probabilities is not None:
            result[key + "_probabilities"] = probabilities
        if kind == "choice":
            if answer["choice"] not in expected:
                raise ValueError("Unknown choice")
            result[key] = answer["choice"]
        else:
            score = number(answer["score"], 0, 4)
            if probabilities is not None:
                mean = sum(int(level) * value for level, value in probabilities.items())
                if abs(mean - score) > 0.025:
                    raise ValueError("Score differs from its probability-weighted mean")
            result[key] = 1 + score
    result["evidence"] = {"kind": "typed_decision_only", "visible_explanation": None}
    return result


def post(payload, prompt, *, transport=requests.post):
    secret = os.environ.get("OPENROUTER_API_KEY")
    if not secret:
        raise ValueError("OPENROUTER_API_KEY is unavailable")
    record = {
        "endpoint": DECISIONS,
        "requested_model": MODEL,
        "request_headers": HEADERS.copy(),
        "timestamp": datetime.now(UTC).isoformat(),
        "request_bytes": len(json.dumps(payload).encode()),
        "request_sha256": digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload}),
        "cost_usd": None,
        "effective_cost_usd": None,
        "cost_basis": "unknown_openrouter_cost",
        "requested_reasoning": None,
        "normalized": None,
        "usage": {},
    }
    started = time.perf_counter()
    try:
        response = transport(
            DECISIONS,
            json=payload,
            headers={
                **HEADERS,
                "Authorization": "Bearer " + secret,
                "Content-Type": "application/json",
                "X-Title": "model-gym Jev14",
            },
            timeout=(15, 180),
            allow_redirects=False,
        )
        record["status"] = response.status_code
        request_id = response.headers.get("x-request-id")
        record["request_id"] = request_id.replace(secret, "[REDACTED]") if request_id else None
        # Redact a reflected credential before retaining a response. Never store request headers wholesale.
        raw = response.text.replace(secret, "[REDACTED]")
        record["response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Response exceeds the evidence size limit")
        record["response"] = json.loads(raw, parse_constant=lambda value: value)
    except (requests.RequestException, ValueError) as exc:
        record["transport_error"] = type(exc).__name__
    record["seconds"] = time.perf_counter() - started
    response = record.get("response")
    if not isinstance(response, dict):
        record["validation_error"] = "No JSON response object"
        return record
    record.update(
        resolved_model=response.get("model"),
        provider=response.get("provider"),
        generation_id=response.get("id"),
    )
    usage = response.get("usage")
    record["usage"] = usage if isinstance(usage, dict) else {}
    try:
        if record["usage"].get("cost") is not None:
            record["cost_usd"] = number(record["usage"]["cost"], 0, float("inf"))
        record["effective_cost_usd"], record["cost_basis"] = effective_cost(record["usage"])
    except (TypeError, ValueError):
        record["cost_validation_error"] = "Invalid reported cost"
    try:
        if record.get("status") != 200:
            raise ValueError("HTTP failure")
        if record["resolved_model"] != REVISION:
            raise ValueError("Resolved model differs from the frozen revision")
        tokens = record["usage"].get("input_tokens")
        if tokens is not None:
            number(tokens, 0, CONTEXT_TOKENS)
        record["normalized"] = normalize(response, prompt["questions"])
    except (TypeError, ValueError, KeyError) as exc:
        record["validation_error"] = type(exc).__name__ + ": " + str(exc)
    return record
