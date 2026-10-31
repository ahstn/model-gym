"""Check narrow post-hoc inclusion without repairing invalid labels."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "raw_diagnostic", Path(__file__).with_name("raw_score_diagnostic.py")
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def content(score="3.8999999999999999"):
    return (
        '{"task":"technical_quality","status":"ok","score":'
        + score
        + ',"evidence":"Fine detail remains visible.","limitations":[]}'
    )


def envelope(text=None):
    run = {
        "settings": {"model": "deepseek-v4-1-flash-1017"},
        "requested_model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    }
    body = {
        "model": run["settings"]["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": text or content(), "refusal": None}}],
        "usage": {"cost": 0.00030555, "runinfra": {"cost_microcents": 30555}},
    }
    result = {"http_status": 200, "transport_error": None, "body": json.dumps(body)}
    return run, result


def test_precision_only_failure_preserves_exact_value_and_content():
    raw = content()
    original = raw[:]
    parsed = diagnostic.classify_content(raw)
    assert parsed["included_in_numeric_diagnostic"] is True
    assert parsed["strict_label_valid"] is False
    assert parsed["precision_only_tenths_failure"] is True
    assert parsed["raw_score_decimal"] == "3.8999999999999999"
    assert parsed["response_content"] == original == raw
    valid = diagnostic.classify_content(content("3.9"))
    assert valid["strict_label_valid"] is True
    assert valid["precision_only_tenths_failure"] is False
    with pytest.raises(ValueError):
        diagnostic.parse_rating(raw)


def test_long_evidence_stays_excluded_but_retains_raw_score_and_evidence():
    evidence = " ".join(["word"] * 31)
    raw = content("3.2000000000000002").replace("Fine detail remains visible.", evidence)
    parsed = diagnostic.classify_content(raw)
    assert parsed["raw_score_decimal"] == "3.2000000000000002"
    assert parsed["evidence"] == evidence
    assert parsed["included_in_numeric_diagnostic"] is False
    assert parsed["strict_label_valid"] is False
    assert parsed["precision_only_tenths_failure"] is False
    assert parsed["exclusion_reason"] == "invalid_rating_contract_or_json"


@pytest.mark.parametrize(
    "raw",
    [
        content("3.87"),
        content("3.900000000002"),
        content("5.0000000000000001"),
        content("0.9999999999999999"),
        content("NaN"),
        content('"3.9"'),
        content("true"),
        content().replace('"score":', '"score":3.9,"score":'),
        content().replace('"limitations":[]', '"limitations":[],"extra":true'),
        content().replace('"limitations":[]', '"limitations":["invented"]'),
        content().replace("Fine detail remains visible.", " ".join(["word"] * 31)),
    ],
)
def test_real_off_grid_or_other_contract_errors_are_excluded(raw):
    parsed = diagnostic.classify_content(raw)
    assert parsed["included_in_numeric_diagnostic"] is False
    assert parsed["strict_label_valid"] is False
    assert parsed["precision_only_tenths_failure"] is False


@pytest.mark.parametrize(
    "change", ["http", "transport", "model", "finish", "refusal", "cost", "duplicate", "malformed"]
)
def test_invalid_envelopes_are_excluded(change):
    run, result = envelope()
    body = json.loads(result["body"])
    if change == "http":
        result["http_status"] = 500
    elif change == "transport":
        result["transport_error"] = "Timeout"
    elif change == "model":
        body["model"] = "different-model"
    elif change == "finish":
        body["choices"][0]["finish_reason"] = "length"
    elif change == "refusal":
        body["choices"][0]["message"]["refusal"] = "Refused"
    elif change == "cost":
        body["usage"]["runinfra"]["cost_microcents"] = 0
    result["body"] = json.dumps(body)
    if change == "duplicate":
        result["body"] = result["body"].replace('"model":', '"model":"wrong","model":')
    if change == "malformed":
        result["body"] = "not json"
    original = copy.deepcopy(result)
    parsed = diagnostic.diagnose_result(run, result)
    assert parsed["included_in_numeric_diagnostic"] is False
    assert parsed["exclusion_reason"] == "invalid_response_envelope_or_unknown_cost"
    assert result == original


def test_successful_envelope_retains_known_cost_and_no_repaired_label():
    run, result = envelope()
    parsed = diagnostic.diagnose_result(run, result)
    assert parsed["included_in_numeric_diagnostic"] is True
    assert parsed["cost_usd"] == "0.00030555"
    assert parsed["strict_label_valid"] is False
    assert "technical_quality" not in parsed


def test_exact_response_content_stays_in_private_output(tmp_path):
    run, result = envelope()
    attempt = diagnostic.diagnose_result(run, result)
    summary = {"post_hoc": True, "attempts": [attempt]}
    rows = [
        {
            "case_id": "A02",
            "raw_score_decimal": attempt["raw_score_decimal"],
            "runinfra_evidence": attempt["evidence"],
        }
    ]
    diagnostic.write_results(
        summary, rows, {"training_ready": False}, output=tmp_path / "public", data=tmp_path / "private"
    )
    public = json.loads((tmp_path / "public/raw-score-diagnostic.json").read_text())
    private = json.loads((tmp_path / "private/results/raw-score-diagnostic-full.json").read_text())
    assert "response_content" not in public["attempts"][0]
    assert private["attempts"][0]["response_content"] == content()
    assert public["attempts"][0]["raw_score_decimal"] == "3.8999999999999999"
    assert "response_content" not in (tmp_path / "public/raw-score-diagnostic.csv").read_text()


def test_saved_attempt_bytes_unchanged_and_request_pins_required(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "PROJECT", tmp_path)
    journal = tmp_path / "attempts"
    folder = journal / "A02"
    folder.mkdir(parents=True)
    run, result = envelope()
    payload = {"fixed": True}
    case = {
        "case_id": "A02",
        "skip_reason": None,
        "request_sha256": diagnostic.runner.digest(payload),
        "request_bytes": len(diagnostic.runner.encoded(payload)),
        "view": {"view_sha256": "a" * 64},
    }
    run.update(run_id="frozen-run", cases=[case])
    request = {
        "case_id": "A02",
        "run_id": run["run_id"],
        "request_sha256": case["request_sha256"],
        "request_bytes": case["request_bytes"],
    }
    (folder / "request.json").write_text(json.dumps(request))
    (folder / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(diagnostic.runner, "payload", lambda run, case: payload)
    before = {path: path.read_bytes() for path in folder.iterdir()}
    records, pins = diagnostic.load_journals(run, [journal])
    assert records[0]["raw_score_decimal"] == "3.8999999999999999"
    assert set(pins) == set(before)
    assert {path: path.read_bytes() for path in folder.iterdir()} == before
    request["request_sha256"] = "changed"
    (folder / "request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="metadata"):
        diagnostic.load_journals(run, [journal])


def test_duplicate_attempt_across_journals_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "PROJECT", tmp_path)
    run, result = envelope()
    payload = {"fixed": True}
    case = {
        "case_id": "A02",
        "skip_reason": None,
        "request_sha256": diagnostic.runner.digest(payload),
        "request_bytes": len(diagnostic.runner.encoded(payload)),
        "view": {"view_sha256": "a" * 64},
    }
    run.update(run_id="frozen-run", cases=[case])
    monkeypatch.setattr(diagnostic.runner, "payload", lambda run, case: payload)
    journals = [tmp_path / "original", tmp_path / "continuation"]
    for journal in journals:
        folder = journal / "A02"
        folder.mkdir(parents=True)
        (folder / "request.json").write_text(
            json.dumps(
                {
                    "case_id": "A02",
                    "run_id": "frozen-run",
                    "request_sha256": case["request_sha256"],
                    "request_bytes": case["request_bytes"],
                }
            )
        )
        (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Duplicate case"):
        diagnostic.load_journals(run, journals)
