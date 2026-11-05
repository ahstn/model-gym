import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.openai_provenance import resume


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, UTC).isoformat()


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.delays = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay


@pytest.fixture
def study(tmp_path, monkeypatch):
    track, data = tmp_path / "tracks/openai_provenance", tmp_path / "data/openai-provenance-v1"
    track.mkdir(parents=True)
    data.mkdir(parents=True)
    protocol = {"maximum_attempts": 3, "minimum_interval_seconds": 6.2}
    (track / "protocol.json").write_text(json.dumps(protocol))
    for name, value in (
        ("TRACK", track),
        ("DATA", data),
        ("ROOT", tmp_path),
        ("AMENDMENT", track / "resume-amendment.json"),
    ):
        monkeypatch.setattr(resume, name, value)
    cases = [
        {"case": name, "source": "real", "native_image": f"images/{name}.jpg", "native_sha256": name * 64}
        for name in "abc"
    ]
    rows = [
        {
            "case": case["case"],
            "source": "real",
            "input_sha256": case["native_sha256"],
            "success": index == 0,
            "http_status": 200 if index == 0 else 502,
            "protocol_sha256": resume.sha(track / "protocol.json"),
            "attempt": index + 1,
            "started_at": timestamp(900 + index * 10),
            "elapsed_seconds": 5.0,
        }
        for index, case in enumerate(cases[:2])
    ]
    ledger = data / "attempts.jsonl"

    def save_rows(value):
        ledger.write_text("".join(json.dumps(row) + "\n" for row in value))

    def load():
        return protocol, cases, [json.loads(line) for line in ledger.read_text().splitlines()]

    save_rows(rows)
    monkeypatch.setattr(resume, "load_study", load)
    clock = Clock()
    monkeypatch.setattr(resume.time, "time", clock.time)
    monkeypatch.setattr(resume.time, "sleep", clock.sleep)
    calls, replies = [], []

    def check(path, expected_sha256):
        calls.append((path, expected_sha256))
        status = replies.pop(0) if replies else 200
        return {
            "input_sha256": expected_sha256,
            "success": status == 200,
            "http_status": status,
            "started_at": timestamp(clock.now),
            "elapsed_seconds": 2.0,
        }

    monkeypatch.setattr(resume, "check_image", check)
    return {
        "track": track,
        "data": data,
        "cases": cases,
        "protocol": protocol,
        "load": load,
        "save_rows": save_rows,
        "calls": calls,
        "replies": replies,
        "clock": clock,
    }


def amendment(study):
    resume.freeze()
    return json.loads(resume.AMENDMENT.read_text())


def test_freeze_keeps_original_protocol_and_pins_observed_failure(study):
    before = (study["track"] / "protocol.json").read_bytes()
    value = amendment(study)
    rows = study["load"]()[2]
    assert (study["track"] / "protocol.json").read_bytes() == before
    assert value["observed_failures"] == [{"case": "b", "attempt": 2, "record_sha256": digest(rows[1])}]
    assert value["resume_code_sha256"] == resume.sha(Path(resume.__file__))
    assert value["max_retries_per_case"] == 1
    assert value["max_total_attempts"] == study["protocol"]["maximum_attempts"] + 16
    assert value["minimum_retry_delay_seconds"] == 30
    with pytest.raises(FileExistsError):
        resume.freeze()
    assert study["calls"] == []


@pytest.mark.parametrize("status", [200, 400, 404, 429, 503, None])
def test_amendment_cannot_authorise_unreviewed_failure_status(study, status):
    rows = study["load"]()[2]
    rows[1].update(success=status == 200, http_status=status)
    study["save_rows"](rows)
    with pytest.raises(ValueError, match="HTTP 502"):
        resume.freeze()
    assert study["calls"] == []


def test_remaining_primary_cases_precede_one_linked_retry_and_originals_are_kept(study):
    value = amendment(study)
    original = (study["data"] / "attempts.jsonl").read_bytes()
    first = resume.run(1)
    assert first["status"] == "partial"
    assert [path.stem for path, _ in study["calls"]] == ["c"]
    assert not (study["data"] / "retries.jsonl").exists()
    second = resume.run(2)
    assert second == {"status": "complete", "valid_cases": 3, "attempts": 4}
    assert [path.stem for path, _ in study["calls"]] == ["c", "b"]
    assert (study["data"] / "attempts.jsonl").read_bytes().startswith(original)
    original_failure = study["load"]()[2][1]
    retry = resume.retry_rows()[0]
    assert retry["retry_of_record_sha256"] == digest(original_failure)
    assert retry["retry_of_attempt"] == original_failure["attempt"]
    assert retry["input_sha256"] == original_failure["input_sha256"]
    assert retry["protocol_sha256"] == value["protocol_sha256"]
    assert retry["amendment_sha256"] == resume.sha(resume.AMENDMENT)
    assert not original_failure["success"] and original_failure["http_status"] == 502
    assert resume.run(3)["status"] == "complete"
    assert len(study["calls"]) == 2


@pytest.mark.parametrize("field", ["resume_code_sha256", "protocol_sha256"])
def test_changed_execution_or_protocol_pin_stops_before_submission(study, field):
    value = amendment(study)
    value[field] = "0" * 64
    resume.AMENDMENT.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        resume.run(1)
    assert study["calls"] == []


def test_reviewed_failure_cannot_be_replaced_by_a_success(study):
    amendment(study)
    rows = study["load"]()[2]
    rows[1].update(success=True, http_status=200)
    study["save_rows"](rows)
    with pytest.raises(ValueError, match="Reviewed failure was changed"):
        resume.run(1)
    assert study["calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("retry_of_record_sha256", "0" * 64),
        ("retry_of_attempt", 987),
        ("input_sha256", "0" * 64),
        ("protocol_sha256", "0" * 64),
        ("amendment_sha256", "0" * 64),
    ],
)
def test_retry_links_are_validated_before_another_submission(study, field, value):
    amendment(study)
    resume.run(3)
    retry = resume.retry_rows()[0]
    retry[field] = value
    (study["data"] / "retries.jsonl").write_text(json.dumps(retry) + "\n")
    calls_before = len(study["calls"])
    with pytest.raises(ValueError, match="Retry does not match"):
        resume.run(1)
    assert len(study["calls"]) == calls_before


def test_a_case_cannot_be_retried_twice(study):
    amendment(study)
    resume.run(3)
    retry = resume.retry_rows()[0]
    with (study["data"] / "retries.jsonl").open("a") as stream:
        stream.write(json.dumps(retry) + "\n")
    with pytest.raises(ValueError, match="Unexpected repeated attempt"):
        resume.run(1)
    assert len(study["calls"]) == 2


def test_failed_502_retry_is_retained_and_never_repeated(study):
    amendment(study)
    study["replies"].extend([200, 502])
    result = resume.run(3)
    assert result == {"status": "stopped", "attempts": 4}
    assert len(resume.retry_rows()) == 1 and not resume.retry_rows()[0]["success"]
    assert resume.run(3)["status"] == "partial"
    assert len(study["calls"]) == 2


@pytest.mark.parametrize("status", [400, 404, 429, 503, None])
def test_non_502_retry_failure_stops_later_calls_on_resume(study, status):
    rows = study["load"]()[2]
    rows[0].update(success=False, http_status=502)
    study["save_rows"](rows)
    amendment(study)
    study["replies"].extend([200, status])
    assert resume.run(3)["status"] == "stopped"
    assert [path.stem for path, _ in study["calls"]] == ["c", "a"]
    with pytest.raises(ValueError, match="failure needs review"):
        resume.run(1)
    assert len(study["calls"]) == 2


def test_non_502_primary_failure_stops_later_calls_on_resume(study):
    amendment(study)
    study["replies"].append(404)
    assert resume.run(3)["status"] == "stopped"
    with pytest.raises(ValueError, match="failure needs review"):
        resume.run(1)
    assert len(study["calls"]) == 1


def test_retry_waits_30_seconds_after_original_attempt_completed(study):
    rows = study["load"]()[2]
    rows[1].update(started_at=timestamp(900), elapsed_seconds=99.0)
    # Complete the primary cohort so the next request is the reviewed retry.
    last = copy.deepcopy(rows[0])
    last.update(case="c", input_sha256="c" * 64, attempt=3)
    rows.append(last)
    study["save_rows"](rows)
    amendment(study)
    resume.run(1)
    assert study["clock"].delays == [29.0]


def test_total_attempt_cap_stops_without_submission(study):
    value = amendment(study)
    value["max_total_attempts"] = 2
    resume.AMENDMENT.write_text(json.dumps(value))
    assert resume.run(1) == {"status": "attempt_limit", "attempts": 2}
    assert study["calls"] == []


@pytest.mark.parametrize("max_new", [0, -1, 177])
def test_public_runner_rejects_invalid_attempt_request_before_submission(study, max_new):
    amendment(study)
    with pytest.raises(ValueError, match="max-new"):
        resume.run(max_new)
    assert study["calls"] == []
