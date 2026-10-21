"""Send native image bytes to OpenAI's provenance endpoint without retries.

A missing signal is not evidence of human origin. This client retains each
supported signal independently and never assigns a real/AI class label.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://api.openai.com/v1/content_provenance_checks"
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
TIMEOUT = (15, 120)
FORMATS = {
    "JPEG": ("image.jpg", "image/jpeg"),
    "PNG": ("image.png", "image/png"),
    "WEBP": ("image.webp", "image/webp"),
}
RESPONSE_HEADERS = {
    "x-request-id",
    "retry-after",
    "retry-after-ms",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "openai-processing-ms",
}


def _native_image(path: str | Path, expected_sha256: str) -> tuple[bytes, str, str, tuple[int, int]]:
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must contain 64 lowercase hexadecimal characters")
    try:
        with Path(path).open("rb") as stream:
            content = stream.read(MAX_IMAGE_BYTES + 1)
    except OSError:
        raise ValueError("Image could not be read") from None
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image must contain between 1 byte and 50 MiB")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Native image SHA-256 does not match the frozen input")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in FORMATS:
                raise ValueError("Unsupported image format")
            filename, mime = FORMATS[image.format]
            dimensions = image.size
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("File must be a valid PNG, JPEG, or WebP image") from None
    return content, filename, mime, dimensions


def _parse_signals(payload: object) -> dict:
    """Validate the documented image response; reject ambiguous partial results."""
    if not isinstance(payload, dict) or payload.get("object") != "content_provenance_check":
        raise ValueError("Unexpected response object")
    created = payload.get("created_at")
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        raise TypeError("Missing or invalid created_at")
    if isinstance(created, float) and not math.isfinite(created):
        raise ValueError("Missing or invalid created_at")
    if created < 0 or "outcome" in payload:
        raise ValueError("Unexpected top-level outcome or invalid created_at")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("An image response must contain C2PA and SynthID results")
    signals = {}
    for result in results:
        if not isinstance(result, dict) or result.get("type") not in ("c2pa", "synthid"):
            raise ValueError("Unexpected signal type")
        kind = result["type"]
        if kind in signals or result.get("outcome") not in ("detected", "not_detected"):
            raise ValueError("Duplicate signal or invalid outcome")
        nullable = ("model", "generated_at", "issuer") if kind == "c2pa" else ("model", "generated_at")
        if any(
            field not in result or (result[field] is not None and not isinstance(result[field], str))
            for field in nullable
        ):
            raise ValueError("Missing or invalid signal fields")
        if kind == "c2pa":
            state = result.get("validation_state")
            if state not in ("trusted", "valid", "invalid", "not_present"):
                raise ValueError("Invalid C2PA validation state")
            if result["outcome"] == "detected" and state not in {"trusted", "valid"}:
                raise ValueError("C2PA detection requires a valid or trusted manifest")
            if result["outcome"] == "detected" and not re.search(
                r"\bopenai\b", result["issuer"] or "", flags=re.IGNORECASE
            ):
                raise ValueError("C2PA detection requires an OpenAI issuer")
        fields = ("type", "outcome", *nullable, *(("validation_state",) if kind == "c2pa" else ()))
        signals[kind] = {field: result[field] for field in fields}
    if set(signals) != {"c2pa", "synthid"}:
        raise ValueError("Missing image signal")
    return signals


def _redact(value: object, api_key: str) -> object:
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    if isinstance(value, dict):
        return {str(_redact(key, api_key)): _redact(item, api_key) for key, item in value.items()}
    return value


def _read_response(response, record: dict, api_key: str) -> None:
    content = response.content
    record["response_sha256"] = hashlib.sha256(content).hexdigest()
    record["response_bytes"] = len(content)
    if len(content) > MAX_RESPONSE_BYTES:
        record["error"] = {"kind": "response_too_large"}
        return
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        record["error"] = {"kind": "non_json_response"}
        return
    payload = _redact(payload, api_key)
    if not 200 <= response.status_code < 300:
        # Error bodies can contain account details. Keep only a bounded error object.
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        record["error"] = {
            "kind": "http_error",
            "code": str(code)[:128] if code is not None else None,
            "detail": json.dumps(error, ensure_ascii=True)[:2048],
        }
        return
    record["raw_response"] = payload
    try:
        record["signals"] = _parse_signals(payload)
    except (TypeError, ValueError) as exc:
        record["error"] = {"kind": "invalid_schema", "detail": str(exc)}
        return
    record["schema_valid"] = True
    record["success"] = True


def check_image(path: str | Path, expected_sha256: str, session=None) -> dict:
    """Check one immutable original image and return an audit record.

    Preflight failures raise before submission. HTTP and transport failures are
    returned as failed attempts. No failed response becomes a negative detection.
    The optional requests-compatible session must not install automatic retries.
    """
    content, filename, mime, dimensions = _native_image(path, expected_sha256)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available")
    record = {
        "endpoint": ENDPOINT,
        "started_at": datetime.now(UTC).isoformat(),
        "input_sha256": expected_sha256,
        "input_bytes": len(content),
        "input_mime": mime,
        "input_width": dimensions[0],
        "input_height": dimensions[1],
        "elapsed_seconds": None,
        "http_status": None,
        "request_id": None,
        "response_headers": {},
        "response_sha256": None,
        "response_bytes": None,
        "raw_response": None,
        "signals": {},
        "success": False,
        "schema_valid": False,
        "transport_error_class": None,
        "error": None,
        "cost_usd": None,
    }
    started = time.perf_counter()
    try:
        response = (session or requests).post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, content, mime)},
            timeout=TIMEOUT,
            allow_redirects=False,
        )
        record["http_status"] = response.status_code
        record["response_headers"] = {
            key.lower(): str(_redact(value, api_key))[:512]
            for key, value in response.headers.items()
            if key.lower() in RESPONSE_HEADERS
        }
        record["request_id"] = record["response_headers"].get("x-request-id")
        _read_response(response, record, api_key)
    except requests.RequestException as exc:
        record["transport_error_class"] = type(exc).__name__
        record["error"] = {"kind": "transport_error"}
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
    return record
