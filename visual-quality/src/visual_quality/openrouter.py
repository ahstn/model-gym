"""Bounded OpenRouter calls with no redirects, retries, or model substitution."""

from __future__ import annotations

from typing import Any

import requests

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT = (15, 120)


def validate_model(info: dict) -> None:
    """Require explicit image, text, and JSON-schema support from the catalogue."""
    architecture = info.get("architecture")
    parameters = info.get("supported_parameters")
    if not isinstance(architecture, dict) or not isinstance(parameters, list):
        raise TypeError("Model metadata lacks explicit modality or parameter support")
    inputs = architecture.get("input_modalities")
    outputs = architecture.get("output_modalities")
    if not isinstance(inputs, list) or not {"image", "text"}.issubset(inputs):
        raise ValueError("The requested model must support image and text inputs")
    if not isinstance(outputs, list) or "text" not in outputs:
        raise ValueError("The requested model must support text output")
    if not {"response_format", "structured_outputs"}.issubset(parameters):
        raise ValueError("The requested model must support JSON-schema structured outputs")


def model_info(model: str) -> dict:
    """Fetch the exact model entry and fail if its required capabilities are absent."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An exact OpenRouter model ID is required")
    with requests.get(_MODELS_URL, timeout=_TIMEOUT, allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError(f"OpenRouter model catalogue returned HTTP {response.status_code}")
        catalogue = response.json()
    if not isinstance(catalogue, dict) or not isinstance(catalogue.get("data"), list):
        raise TypeError("OpenRouter model catalogue has an invalid data structure")
    matches = [entry for entry in catalogue["data"] if isinstance(entry, dict) and entry.get("id") == model]
    if len(matches) != 1:
        raise ValueError("The exact model ID was not uniquely found in the OpenRouter catalogue")
    validate_model(matches[0])
    return matches[0]


def complete(api_key: str, payload: dict) -> dict[str, Any]:
    """Make one billable attempt; preserve the raw result without exposing the key.

    HTTP errors are returned to the caller, including Retry-After. A transport
    failure has an unknown billing outcome and must not trigger an automatic retry.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("An OpenRouter API key is required")
    try:
        with requests.post(
            _COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=_TIMEOUT,
            allow_redirects=False,
        ) as response:
            retry_after = response.headers.get("Retry-After")
            return {
                "http_status": response.status_code,
                "body": response.text.replace(api_key, "[REDACTED]"),
                "retry_after": retry_after.replace(api_key, "[REDACTED]") if retry_after else None,
                "transport_error": None,
            }
    except requests.RequestException as error:
        return {
            "http_status": None,
            "body": None,
            "retry_after": None,
            "transport_error": type(error).__name__,
        }
