"""Keep provenance abstentions, classifier predictions and audit evidence separate."""

import copy
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.openai_provenance import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def cohort(tmp_path):
    track = tmp_path / "tracks/openai_provenance"
    data = tmp_path / "data/openai-provenance-v1"
    source = tmp_path / "tracks/metadata_v3"
    for path in (track / "client.py", track / "study.py", data / "research/guide.md",
                 source / "dataset.json", source / "protocol.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    cases = [{"case": name, "source": name, "gold_ai": name != "real",
              "openai_origin": name in {"dalle2", "dalle3"}, "native_sha256": str(index) * 64,
              "delivered_sha256": str(index + 5) * 64}
             for index, name in enumerate(report.SOURCES)]
    write(data / "manifest.json", {"cases": cases})
    protocol = {"cases": 5, "source_counts": dict.fromkeys(report.SOURCES, 1),
                "manifest_sha256": report.sha(data / "manifest.json"),
                "guide_sha256": report.sha(data / "research/guide.md"),
                "source_dataset_sha256": report.sha(source / "dataset.json"),
                "source_protocol_sha256": report.sha(source / "protocol.json"),
                "execution_sha256": {str(path.relative_to(tmp_path)): report.sha(path)
                                     for path in (track / "client.py", track / "study.py")},
                "endpoint": "https://api.openai.com/v1/content_provenance_checks",
                "limitations": ["No current GPT-image positive controls"]}
    write(track / "protocol.json", protocol)
    rows = []
    for arm in report.ARMS:
        for case, probability in zip(cases, (0.1, 0.7, 0.8, 0.3, 0.6), strict=True):
            rows.append({"case": case["case"], "condition": "native", "prompt_variant": "selected",
                         "arm": arm, "native_sha256": case["native_sha256"],
                         "delivered_sha256": case["delivered_sha256"], "gold": int(case["gold_ai"]),
                         "prediction": probability, "valid": "True", "effective_cost_usd": 0.01,
                         "seconds": 0.5})
    write_csv(source / "results/cases.csv", rows)
    return {"root": tmp_path, "track": track, "data": data, "source": source,
            "protocol": protocol, "cases": cases, "decider": rows}


def attempt(cohort, index, *, detected=False, adobe=False, success=True):
    case = cohort["cases"][index]
    c2pa = {"type": "c2pa", "outcome": "detected" if detected else "not_detected",
            "model": None, "generated_at": None, "issuer": "OpenAI" if detected else "Adobe Inc." if adobe else None,
            "validation_state": "valid" if detected or adobe else "not_present"}
    synthid = {"type": "synthid", "outcome": "not_detected", "model": None, "generated_at": None}
    return {"case": case["case"], "source": case["source"], "input_sha256": case["native_sha256"],
            "protocol_sha256": report.sha(cohort["track"] / "protocol.json"),
            "endpoint": cohort["protocol"]["endpoint"], "attempt": index + 1,
            "http_status": 200 if success else 502,
            "success": success, "schema_valid": success, "elapsed_seconds": float(index + 1),
            "cost_usd": None, "response_headers": {"openai-processing-ms": "500"},
            "signals": {"c2pa": c2pa, "synthid": synthid} if success else {},
            "raw_response": {"object": "content_provenance_check", "created_at": 123,
                             "results": [c2pa, synthid]} if success else None,
            "error": None if success else {"kind": "http_error"}}


def ledger(cohort, rows):
    (cohort["data"] / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_partial_report_separates_unresolved_failed_and_unattempted(cohort):
    rows = [attempt(cohort, 0), attempt(cohort, 1, detected=True), attempt(cohort, 2, success=False),
            attempt(cohort, 3, adobe=True)]
    ledger(cohort, rows)
    result = report.build_report(cohort["root"])
    summary = result["summary"]
    assert (summary["attempted"], summary["success"], summary["failed"], summary["unattempted"]) == (4, 3, 1, 1)
    assert summary["status"] == "partial"
    assert summary["openai_origin_detections"] == 1
    assert summary["by_source"]["firefly"]["c2pa_states"] == {"valid": 1}
    assert summary["by_source"]["firefly"]["openai_signals"] == 0
    assert summary["by_source"]["firefly"]["unresolved"] == 1
    assert [row["disposition"] for row in result["cases"]] == [
        "unresolved", "openai_signal", "failed", "unresolved", "unattempted"]
    assert result["cases"][2]["openai_signal_detected"] is None
    assert result["cases"][4]["openai_signal_detected"] is None
    assert summary["total_cost_usd"] is None
    assert summary["known_cost_usd"] is None
    assert summary["unknown_cost_attempts"] == 4
    assert summary["successful_latency"] == {"count": 3, "p50_seconds": 2.0, "p95_seconds": 3.8}
    assert summary["failed_latency"] == {"count": 1, "p50_seconds": 3.0, "p95_seconds": 3.0}
    assert summary["successful_server_processing"]["p50_seconds"] == 0.5
    assert "accuracy" not in summary and "brier" not in summary and "auroc" not in summary


def test_complete_results_keep_false_attribution_classes_separate(cohort):
    ledger(cohort, [attempt(cohort, index, detected=index in {0, 3}) for index in range(5)])
    result = report.build_report(cohort["root"])
    assert result["summary"]["status"] == "complete"
    assert result["summary"]["real_false_attributions"] == 1
    assert result["summary"]["other_ai_false_attributions"] == 1
    assert result["summary"]["openai_origin_detections"] == 0


def test_decider_metrics_recomputed_and_no_raw_responses_exported(cohort):
    ledger(cohort, [attempt(cohort, 0)])
    result = report.build_report(cohort["root"])
    for metrics in result["summary"]["decider"].values():
        assert metrics["correct"] == 4
        assert metrics["ai_detected"] == 3
        assert metrics["real_false_flags"] == 0
        assert metrics["brier"] == pytest.approx(0.158)
        assert metrics["cost_usd"] == pytest.approx(0.05)
        assert metrics["latency"]["p50_seconds"] == 0.5
    text = json.dumps(result)
    assert "raw_response" not in text
    assert "response_headers" not in text
    assert result["cases"][1]["decider_image_only_ai_probability"] == 0.7
    prose = report.render_markdown(result)
    assert "Missing signals remain unresolved" in prose
    assert "Unknown does not mean free" in prose
    assert "no current GPT-image" in prose


@pytest.mark.parametrize("change", ["duplicate", "case", "native", "protocol", "endpoint", "order", "signal"])
def test_corrupt_ledger_is_rejected(cohort, change):
    rows = [attempt(cohort, 0)]
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "case":
        rows[0]["case"] = "unexpected"
    elif change == "native":
        rows[0]["input_sha256"] = "f" * 64
    elif change == "protocol":
        rows[0]["protocol_sha256"] = "f" * 64
    elif change == "endpoint":
        rows[0]["endpoint"] = "https://wrong.example"
    elif change == "order":
        rows[0]["attempt"] = 2
    else:
        rows[0]["signals"]["c2pa"]["outcome"] = "detected"
    ledger(cohort, rows)
    with pytest.raises(ValueError):
        report.build_report(cohort["root"])


@pytest.mark.parametrize("name", ["dataset", "manifest", "guide", "client"])
def test_changed_frozen_input_is_rejected(cohort, name):
    paths = {"dataset": cohort["source"] / "dataset.json", "manifest": cohort["data"] / "manifest.json",
             "guide": cohort["data"] / "research/guide.md", "client": cohort["track"] / "client.py"}
    path = paths[name]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen study input"):
        report.build_report(cohort["root"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "native_hash", "delivered_hash", "gold"])
def test_decider_join_requires_exact_case_and_hashes(cohort, change):
    rows = copy.deepcopy(cohort["decider"])
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    elif change == "native_hash":
        rows[0]["native_sha256"] = "f" * 64
    elif change == "delivered_hash":
        rows[0]["delivered_sha256"] = "f" * 64
    else:
        rows[0]["gold"] = 1
    write_csv(cohort["source"] / "results/cases.csv", rows)
    with pytest.raises(ValueError, match="Decider"):
        report.build_report(cohort["root"])


def test_snapshot_is_immutable_and_unknown_costs_stay_blank(cohort, tmp_path):
    result = report.build_report(cohort["root"])
    destination = tmp_path / "snapshot"
    report.write_report(result, destination)
    with (destination / "results/cases.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["cost_usd"] == "" for row in rows)
    before = (destination / "report.md").read_bytes()
    with pytest.raises(ValueError, match="snapshot exists"):
        report.write_report(result, destination)
    assert (destination / "report.md").read_bytes() == before


def test_historical_overlap_is_rejected(cohort):
    write_csv(cohort["root"] / "summaries/initial_benchmark.csv", [{
        "panel": "ai", "phase": "primary", "delivered_sha256": cohort["cases"][0]["native_sha256"],
    }])
    with pytest.raises(ValueError, match="overlaps"):
        report.build_report(cohort["root"])


def amended_cohort(cohort, *, retry_success=True):
    rows = [attempt(cohort, index, success=index != 2) for index in range(5)]
    resume = cohort["track"] / "resume.py"
    resume.write_text("# frozen resume\n")
    amendment = {"protocol_sha256": report.sha(cohort["track"] / "protocol.json"),
                 "resume_code_sha256": report.sha(resume), "max_retries_per_case": 1,
                 "eligible_status": 502, "max_total_attempts": 6,
                 "observed_failures": [{"case": rows[2]["case"], "attempt": 3,
                                        "record_sha256": report.digest(rows[2])}]}
    path = cohort["track"] / "resume-amendment.json"
    write(path, amendment)
    for row in rows[3:]:
        row["amendment_sha256"] = report.sha(path)
    ledger(cohort, rows)
    retry = attempt(cohort, 2, detected=retry_success, success=retry_success)
    retry.update(attempt=6, amendment_sha256=report.sha(path), retry_of_attempt=3,
                 retry_of_record_sha256=report.digest(rows[2]))
    (cohort["data"] / "retries.jsonl").write_text(json.dumps(retry) + "\n")
    return rows, retry


def test_successful_retry_recovers_case_but_keeps_failed_attempt_and_cost(cohort):
    amended_cohort(cohort)
    result = report.build_report(cohort["root"])
    summary = result["summary"]
    assert summary["status"] == "complete"
    assert (summary["attempted"], summary["attempted_cases"], summary["success"]) == (6, 5, 5)
    assert (summary["failed"], summary["failed_cases"], summary["retry_attempts"], summary["recovered_cases"]) == (1, 0, 1, 1)
    assert summary["unknown_cost_attempts"] == 6
    assert summary["successful_latency"]["count"] == 5
    assert summary["failed_latency"]["count"] == 1
    case = result["cases"][2]
    assert case["disposition"] == "openai_signal"
    assert case["effective_attempt"] == 6
    assert case["effective_retry_of_attempt"] == 3
    assert case["primary_success"] is False
    assert case["attempt_count"] == 2
    assert summary["decider_matched_valid_api"]["image_only"]["images"] == 5


def test_failed_retry_remains_unresolved_and_is_counted_twice(cohort):
    amended_cohort(cohort, retry_success=False)
    result = report.build_report(cohort["root"])
    summary = result["summary"]
    assert summary["status"] == "partial"
    assert summary["failed"] == 2
    assert summary["failed_cases"] == 1
    assert summary["unknown_cost_attempts"] == 6
    assert result["cases"][2]["openai_signal_detected"] is None
    assert summary["decider_matched_valid_api"]["image_only"]["images"] == 4
    assert "valid API cases only" in report.render_markdown(result)


@pytest.mark.parametrize("change", ["parent", "parent_attempt", "amendment", "native", "duplicate", "resume_code",
                                    "reviewed_failure", "primary_amendment", "success_parent"])
def test_retry_links_and_amendment_are_verified(cohort, change):
    rows, retry = amended_cohort(cohort)
    if change == "parent":
        retry["retry_of_record_sha256"] = "f" * 64
    elif change == "parent_attempt":
        retry["retry_of_attempt"] = 2
    elif change == "amendment":
        retry["amendment_sha256"] = "f" * 64
    elif change == "native":
        retry["input_sha256"] = "f" * 64
    elif change == "resume_code":
        (cohort["track"] / "resume.py").write_text("# changed\n")
    elif change == "reviewed_failure":
        rows[2]["elapsed_seconds"] = 100
    elif change == "primary_amendment":
        rows[3]["amendment_sha256"] = "f" * 64
    elif change == "success_parent":
        retry["case"] = rows[0]["case"]
    ledger(cohort, rows)
    content = json.dumps(retry) + "\n"
    (cohort["data"] / "retries.jsonl").write_text(content * (2 if change == "duplicate" else 1))
    with pytest.raises(ValueError):
        report.build_report(cohort["root"])


def test_rate_limit_reports_status_and_cooldown_without_error_body(cohort):
    rows = [attempt(cohort, index, success=index < 2) for index in range(4)]
    rows[2]["error"] = {"kind": "non_json_response"}
    rows[3]["http_status"] = 429
    rows[3]["error"] = {
        "kind": "http_error", "code": "rate_limit_exceeded",
        "detail": json.dumps({"message": "Account private-account-marker: slow down and try again after 2932.782 seconds."}),
    }
    rows[3]["response_headers"] = {}
    ledger(cohort, rows)
    result = report.build_report(cohort["root"])
    summary = result["summary"]
    assert summary["failed_http_status_counts"] == {"502": 1, "429": 1}
    failure = summary["failure_records"][1]
    assert failure["error_code"] == "rate_limit_exceeded"
    assert failure["server_cooldown_seconds"] == 2932.782
    assert failure["cooldown_source"] == "error_message"
    assert failure["retry_after_header_present"] is False
    assert summary["openai_origin_successes"] == 1
    assert summary["real_successes"] == 1
    assert summary["other_ai_successes"] == 0
    prose = report.render_markdown(result)
    assert "0/1 successfully checked older DALL-E images (2 planned)" in prose
    assert "0/2 older DALL-E" not in prose
    assert "HTTP 429: 1, HTTP 502: 1" in prose
    assert "2932.782-second wait (48.9 minutes)" in prose
    assert "not a live countdown" in prose
    assert "small coverage sample cannot establish a ranking" in prose
    assert "Valid / tried / planned" in prose
    assert "private-account-marker" not in json.dumps(result)
    assert "private-account-marker" not in prose


@pytest.mark.parametrize("message", ["try again after 99999999 seconds", "try again after -1 seconds",
                                    "try again after NaN seconds", "account info without cooldown"])
def test_unrecognised_or_unbounded_cooldown_is_not_published(cohort, message):
    row = attempt(cohort, 0, success=False)
    row.update(http_status=429, error={"kind": "http_error", "detail": message})
    assert report.failure_details([row])[0]["server_cooldown_seconds"] is None
