import json
import threading

import gateway_recovery
import guardrail_recovery as recovery


def test_only_the_confirmed_secret_scanner_is_eligible():
    def response(message):
        return {"http_status": 403, "body": json.dumps({"error": {"message": message}})}

    assert recovery.secret_scanner_denial(response("Request blocked by content filter: [SECRET:bitcoin-wif-compressed]"))
    assert recovery.secret_scanner_denial(response("Request blocked by content filter: [SECRET:bitcoin-wif-uncompressed]"))
    assert recovery.secret_scanner_denial(response("Request blocked by content filter: [SECRET:google-api-key]"))
    assert not recovery.secret_scanner_denial(response("Request blocked by content filter: other restriction"))
    assert not recovery.secret_scanner_denial(response("Invalid API key"))


def test_gateway_recovery_does_not_repeat_scores_or_content_denials(tmp_path):
    path = tmp_path / "result.json"
    row = {"status": "failed", "raw_response_path": str(path)}
    path.write_text(json.dumps({"http_status": 520}))
    assert gateway_recovery.eligible(row)
    assert not gateway_recovery.eligible({**row, "technical_score": 3.0})
    assert not gateway_recovery.eligible({**row, "status": "valid"})
    for status in (400, 401, 403):
        path.write_text(json.dumps({"http_status": status}))
        assert not gateway_recovery.eligible(row)


def test_recovery_preserves_old_denial_and_never_repeats_new_terminal_result(tmp_path, monkeypatch):
    runner = recovery.runner
    paths = []
    calls = []
    case = {"case_id": "R1", "view_sha256": "view"}
    request = {"request_sha256": "request", "view_sha256": "view"}
    old = {
        "http_status": 403,
        "body": json.dumps({"error": {"message": "Request blocked by content filter: [SECRET:bitcoin-wif-compressed]"}}),
        "parsed": {"retryable": False},
    }

    def append(number, result):
        folder = tmp_path / str(number)
        folder.mkdir()
        path = folder / "request.json"
        path.write_text(json.dumps(request))
        path.with_name("result.json").write_text(json.dumps(result))
        paths.append(path)

    append(1, old)
    original_hash = recovery.digest(paths[0].with_name("result.json"))
    plan = {"policy": {"additional_attempts": 3, "authorized_denials": {"R1": {"attempt_count": 1, "result_sha256": original_hash}}}}
    monkeypatch.setattr(runner, "prepare", lambda *a: ({}, "request"))
    monkeypatch.setattr(runner, "attempts_for", lambda *a: paths)
    monkeypatch.setattr(runner, "observation", lambda *a: "latest preserved observation")

    def send(plan, case, root, number, *args):
        calls.append(number)
        result = {"http_status": 200, "parsed": {"retryable": False, "status": "valid"}}
        append(number, result)
        return result

    monkeypatch.setattr(runner, "request_once", send)
    for _ in range(2):
        assert recovery.recovery_job(plan, case, tmp_path, "unused", None, threading.Event()) == "latest preserved observation"
    assert calls == [2]
    assert recovery.digest(paths[0].with_name("result.json")) == original_hash
