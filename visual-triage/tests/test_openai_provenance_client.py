import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.openai_provenance import client


def payload():
    return {
        "object": "content_provenance_check",
        "created_at": 1778000000,
        "results": [
            {
                "type": "c2pa",
                "outcome": "detected",
                "validation_state": "trusted",
                "issuer": "OpenAI OpCo, LLC",
                "model": "gpt-image",
                "generated_at": "2026-07-27T18:34:12Z",
            },
            {"type": "synthid", "outcome": "not_detected", "model": None, "generated_at": None},
        ],
    }


class Session:
    def __init__(self, body=None, status=200, headers=None, failure=None):
        self.calls = []
        self.failure = failure
        self.response = SimpleNamespace(
            content=json.dumps(payload() if body is None else body).encode(),
            status_code=status,
            headers=headers or {},
        )

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        if self.failure:
            raise self.failure
        return self.response


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-private-key-do-not-retain")
    path = tmp_path / "private-source-name.png"
    # Deliberately mismatch the extension and preserve a JPEG EXIF segment.
    exif = Image.Exif()
    exif[271] = "Native camera maker"
    Image.new("RGB", (24, 16), "blue").save(path, format="JPEG", exif=exif)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_upload_preserves_native_bytes_and_does_not_send_source_name(native):
    session = Session(
        headers={"X-Request-ID": "req_123", "OpenAI-Processing-Ms": "15", "Set-Cookie": "private"}
    )
    result = client.check_image(*native, session=session)
    endpoint, arguments = session.calls[0]
    assert endpoint == client.ENDPOINT
    assert arguments["files"] == {"file": ("image.jpg", native[0].read_bytes(), "image/jpeg")}
    assert "Exif" in arguments["files"]["file"][1].decode("latin1")
    assert arguments["headers"] == {"Authorization": "Bearer test-private-key-do-not-retain"}
    assert arguments["allow_redirects"] is False
    assert arguments["timeout"] == (15, 120)
    assert len(session.calls) == 1
    assert result["success"] and result["schema_valid"]
    assert result["input_sha256"] == native[1]
    assert result["input_bytes"] == len(native[0].read_bytes())
    assert result["input_mime"] == "image/jpeg"
    assert result["input_width"] == 24 and result["input_height"] == 16
    assert result["request_id"] == "req_123"
    assert result["response_headers"] == {"x-request-id": "req_123", "openai-processing-ms": "15"}
    assert result["raw_response"] == payload()
    assert result["signals"]["c2pa"]["outcome"] == "detected"
    assert result["signals"]["synthid"]["outcome"] == "not_detected"
    assert result["elapsed_seconds"] >= 0
    assert result["cost_usd"] is None
    assert "test-private-key-do-not-retain" not in json.dumps(result)
    assert "private-source-name" not in json.dumps(result)
    assert "outcome" not in result and "is_ai" not in result


@pytest.mark.parametrize("format_name,mime", [("PNG", "image/png"), ("WEBP", "image/webp")])
def test_other_supported_formats_are_detected_from_original_bytes(native, format_name, mime):
    path, _ = native
    Image.new("RGB", (20, 10)).save(path, format=format_name)
    session = Session()
    result = client.check_image(path, hashlib.sha256(path.read_bytes()).hexdigest(), session=session)
    assert result["success"] and result["input_mime"] == mime
    assert session.calls[0][1]["files"]["file"][1] == path.read_bytes()


def test_valid_adobe_manifest_is_preserved_as_supported_openai_signal_not_detected(native):
    body = payload()
    body["results"][0].update(
        issuer="Adobe Inc.", validation_state="valid", outcome="not_detected", model=None
    )
    result = client.check_image(*native, session=Session(body))
    assert result["success"]
    assert result["signals"]["c2pa"]["validation_state"] == "valid"
    assert result["signals"]["c2pa"]["issuer"] == "Adobe Inc."
    assert all(signal["outcome"] == "not_detected" for signal in result["signals"].values())
    assert "is_real" not in result


def test_synthid_is_read_independently(native):
    body = payload()
    body["results"][0].update(
        validation_state="not_present", issuer=None, model=None, generated_at=None, outcome="not_detected"
    )
    body["results"][1]["outcome"] = "detected"
    result = client.check_image(*native, session=Session(body))
    assert result["success"]
    assert result["signals"]["c2pa"]["outcome"] == "not_detected"
    assert result["signals"]["synthid"]["outcome"] == "detected"


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body.update(outcome="not_detected"),
        lambda body: body.update(created_at=True),
        lambda body: body.update(created_at=float("nan")),
        lambda body: body.update(results=body["results"][:1]),
        lambda body: body.update(results=[body["results"][0], copy.deepcopy(body["results"][0])]),
        lambda body: body["results"][0].update(validation_state="invalid"),
        lambda body: body["results"][0].update(validation_state="not_present"),
        lambda body: body["results"][0].update(issuer="Adobe Inc."),
        lambda body: body["results"][0].update(validation_state=[]),
        lambda body: body["results"][0].update(type=[]),
        lambda body: body["results"][1].update(outcome=[]),
        lambda body: body["results"][1].pop("model"),
        lambda body: body["results"][1].update(generated_at=42),
    ],
)
def test_malformed_or_inconsistent_results_cannot_become_negative_detections(native, change):
    body = payload()
    change(body)
    result = client.check_image(*native, session=Session(body))
    assert not result["success"] and not result["schema_valid"]
    assert result["signals"] == {}
    assert result["error"]["kind"] == "invalid_schema"
    assert result["raw_response"] is not None


@pytest.mark.parametrize("state", ["invalid", "not_present"])
def test_invalid_or_missing_c2pa_is_not_supported_signal_detection(native, state):
    body = payload()
    body["results"][0].update(validation_state=state, outcome="not_detected", issuer=None)
    result = client.check_image(*native, session=Session(body))
    assert result["success"]
    assert result["signals"]["c2pa"]["validation_state"] == state


def test_file_hash_and_format_validation_precede_request(native, monkeypatch):
    path, digest = native
    session = Session()
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        client.check_image(path, "0" * 64, session=session)
    with pytest.raises(ValueError, match="64 lowercase"):
        client.check_image(path, "missing", session=session)
    monkeypatch.setattr(client, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="50 MiB"):
        client.check_image(path, digest, session=session)
    monkeypatch.setattr(client, "MAX_IMAGE_BYTES", 50 * 1024 * 1024)
    Image.new("RGB", (5, 5)).save(path, format="GIF")
    with pytest.raises(ValueError, match="valid PNG, JPEG, or WebP"):
        client.check_image(path, hashlib.sha256(path.read_bytes()).hexdigest(), session=session)
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="valid PNG, JPEG, or WebP"):
        client.check_image(path, hashlib.sha256(path.read_bytes()).hexdigest(), session=session)
    assert session.calls == []


def test_missing_key_rejects_without_request(native, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    session = Session()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not available"):
        client.check_image(*native, session=session)
    assert session.calls == []


def test_http_error_is_bounded_redacted_and_not_retried(native):
    body = {
        "error": {"message": "test-private-key-do-not-retain " + "x" * 5000, "code": "rate_limit_exceeded"}
    }
    session = Session(body, status=429, headers={"Retry-After": "30", "x-request-id": "req_limited"})
    result = client.check_image(*native, session=session)
    assert len(session.calls) == 1
    assert not result["success"] and result["signals"] == {}
    assert result["http_status"] == 429
    assert result["request_id"] == "req_limited"
    assert result["response_headers"]["retry-after"] == "30"
    assert result["error"]["kind"] == "http_error"
    assert result["error"]["code"] == "rate_limit_exceeded"
    assert len(result["error"]["detail"]) <= 2048
    assert "test-private-key-do-not-retain" not in json.dumps(result)
    assert result["raw_response"] is None and result["cost_usd"] is None


@pytest.mark.parametrize("failure", [requests.ReadTimeout, requests.ConnectionError])
def test_transport_failure_keeps_class_without_exception_secrets_or_retries(native, failure):
    session = Session(failure=failure("test-private-key-do-not-retain in private URL"))
    result = client.check_image(*native, session=session)
    assert len(session.calls) == 1
    assert result["transport_error_class"] == failure.__name__
    assert not result["success"] and result["http_status"] is None
    assert result["signals"] == {} and result["cost_usd"] is None
    assert "test-private-key-do-not-retain" not in json.dumps(result)
    assert "private URL" not in json.dumps(result)


def test_non_json_response_is_a_failure(native):
    session = Session()
    session.response.content = b"<html>service unavailable</html>"
    result = client.check_image(*native, session=session)
    assert not result["success"] and result["signals"] == {}
    assert result["error"]["kind"] == "non_json_response"


def test_oversized_response_is_a_failure(native, monkeypatch):
    monkeypatch.setattr(client, "MAX_RESPONSE_BYTES", 10)
    result = client.check_image(*native, session=Session())
    assert not result["success"] and result["signals"] == {}
    assert result["error"]["kind"] == "response_too_large"
