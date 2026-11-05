"""Provider boundaries: exact capabilities, one attempt, and safe failure records."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
import requests

from visual_quality import openrouter

MODEL = {
    "id": "example/vision-v1",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
    "supported_parameters": ["response_format", "structured_outputs"],
    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
}


def response(status=200, body="{}", headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = body.encode()
    result._content_consumed = True
    result.headers.update(headers or {})
    return result


def test_model_info_preserves_exact_catalogue_entry(monkeypatch):
    get = Mock(return_value=response(body=json.dumps({"data": [MODEL]})))
    monkeypatch.setattr(openrouter.requests, "get", get)
    assert openrouter.model_info(MODEL["id"]) == MODEL
    get.assert_called_once_with(
        "https://openrouter.ai/api/v1/models", timeout=(15, 120), allow_redirects=False
    )


@pytest.mark.parametrize("model_id", ["example/vision", "unknown/model", "example/vision-v1 "])
def test_model_info_does_not_substitute_models(monkeypatch, model_id):
    get = Mock(return_value=response(body=json.dumps({"data": [MODEL]})))
    monkeypatch.setattr(openrouter.requests, "get", get)
    with pytest.raises(ValueError, match="exact model ID"):
        openrouter.model_info(model_id)
    assert get.call_count == 1


@pytest.mark.parametrize("status", [302, 429, 503])
def test_model_info_does_not_follow_redirect_or_retry(monkeypatch, status):
    get = Mock(return_value=response(status=status))
    monkeypatch.setattr(openrouter.requests, "get", get)
    with pytest.raises(ValueError, match=f"HTTP {status}"):
        openrouter.model_info(MODEL["id"])
    assert get.call_count == 1
    assert get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "patch",
    [
        {"architecture": None},
        {"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
        {"architecture": {"input_modalities": "text+image", "output_modalities": ["text"]}},
        {"architecture": {"input_modalities": ["image", "text"], "output_modalities": ["image"]}},
        {"supported_parameters": ["response_format"]},
        {"supported_parameters": ["structured_outputs"]},
        {"supported_parameters": None},
    ],
)
def test_model_capabilities_fail_closed(patch):
    info = deepcopy(MODEL)
    info.update(patch)
    with pytest.raises((ValueError, TypeError)):
        openrouter.validate_model(info)


def test_complete_keeps_body_and_makes_one_fixed_endpoint_call(monkeypatch):
    body = '{"usage": {"cost": 0.001}, "choices": []}\n'
    post = Mock(return_value=response(body=body))
    monkeypatch.setattr(openrouter.requests, "post", post)
    payload = {"model": MODEL["id"], "messages": []}
    assert openrouter.complete("test-secret", payload) == {
        "http_status": 200,
        "body": body,
        "retry_after": None,
        "transport_error": None,
    }
    post.assert_called_once_with(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer test-secret", "Content-Type": "application/json"},
        json=payload,
        timeout=(15, 120),
        allow_redirects=False,
    )


@pytest.mark.parametrize("status", [302, 400, 429, 503])
def test_complete_returns_http_failure_without_retry(monkeypatch, status):
    post = Mock(return_value=response(status, '{"error":"wait"}', {"Retry-After": "120"}))
    monkeypatch.setattr(openrouter.requests, "post", post)
    result = openrouter.complete("test-secret", {})
    assert result == {
        "http_status": status,
        "body": '{"error":"wait"}',
        "retry_after": "120",
        "transport_error": None,
    }
    assert post.call_count == 1
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError])
def test_transport_error_omits_exception_text_and_does_not_retry(monkeypatch, error_type):
    post = Mock(side_effect=error_type("Authorization: Bearer test-secret"))
    monkeypatch.setattr(openrouter.requests, "post", post)
    assert openrouter.complete("test-secret", {}) == {
        "http_status": None,
        "body": None,
        "retry_after": None,
        "transport_error": error_type.__name__,
    }
    assert post.call_count == 1


def test_complete_redacts_key_echoed_in_response(monkeypatch):
    post = Mock(return_value=response(401, "invalid test-secret", {"Retry-After": "test-secret"}))
    monkeypatch.setattr(openrouter.requests, "post", post)
    result = openrouter.complete("test-secret", {})
    assert result["body"] == "invalid [REDACTED]"
    assert result["retry_after"] == "[REDACTED]"
    assert "test-secret" not in json.dumps(result)


@pytest.mark.parametrize("key", ["", "  ", None])
def test_complete_rejects_missing_key_before_http(monkeypatch, key):
    post = Mock()
    monkeypatch.setattr(openrouter.requests, "post", post)
    with pytest.raises(ValueError, match="API key"):
        openrouter.complete(key, {})
    post.assert_not_called()
