"""Bounded parser continuation preserves the original request ledger."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_jev14_run import fake_post, save_json, valid_row, write_rows
from test_jev14_run import frozen as frozen  # noqa: PLC0414 -- re-export the pytest fixture.

from tracks.jev14 import resume
from tracks.jev14 import run as study


@pytest.fixture
def reviewed(frozen, monkeypatch):
    for key in ("ROOT", "TRACK", "PROTOCOL", "MANIFEST", "LEDGER"):
        monkeypatch.setattr(resume, key, getattr(study, key))
    monkeypatch.setattr(resume, "AMENDMENT", study.TRACK / "parser-amendment.json")
    parser = study.TRACK / "rounding.py"
    parser.write_text("# frozen parser fixture\n")
    write_rows([valid_row(frozen)])
    pin_amendment()
    return frozen


def pin_amendment():
    save_json(
        resume.AMENDMENT,
        {
            "protocol_sha256": study.sha(study.PROTOCOL),
            "prefix_rows": 1,
            "prefix_sha256": hashlib.sha256(study.LEDGER.read_bytes()).hexdigest(),
            "parser_sha256": study.sha(study.TRACK / "rounding.py"),
            "resume_sha256": study.sha(Path(resume.__file__)),
        },
    )


def test_preserved_prefix_cannot_change_even_when_json_values_match(reviewed):
    row = json.loads(study.LEDGER.read_text())
    study.LEDGER.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="preserved ledger prefix"):
        resume.load_reviewed()


def test_parser_amendment_pin_cannot_change(reviewed):
    (study.TRACK / "rounding.py").write_text("# changed parser\n")
    with pytest.raises(ValueError, match="Parser amendment"):
        resume.load_reviewed()


def test_only_rounding_legacy_error_is_eligible_and_raw_values_are_kept(reviewed):
    row = valid_row(reviewed)
    row["validation_error"] = "ValueError: Probability distribution does not sum to one"
    row["normalized"] = None
    row["response"]["answers"]["aesthetics"].update(
        score=2.58, probabilities={"0": 0, "1": 0.06, "2": 0.34, "3": 0.54, "4": 0.05}
    )
    write_rows([row])
    pin_amendment()
    original = study.LEDGER.read_bytes()
    _, _, rows = resume.load_reviewed()
    assert rows[0]["normalized"] is None
    assert rows[0]["response"]["answers"]["aesthetics"]["score"] == 2.58
    assert study.LEDGER.read_bytes() == original
    row["validation_error"] = "ValueError: Resolved model differs from the frozen revision"
    write_rows([row])
    pin_amendment()
    with pytest.raises(ValueError, match="outside probability rounding"):
        resume.load_reviewed()


@pytest.mark.parametrize("change", ["status", "model", "usage", "cost"])
def test_bad_response_or_cost_blocks_continuation(reviewed, change):
    row = valid_row(reviewed)
    if change == "status":
        row["status"] = 502
    elif change == "model":
        row["resolved_model"] = "typesafe/other"
    elif change == "usage":
        row["usage"] = {**row["usage"], "input_tokens": 2}
    else:
        row["effective_cost_usd"] = 999
    write_rows([row])
    pin_amendment()
    with pytest.raises(ValueError):
        resume.load_reviewed()


def test_resume_sends_only_unattempted_case_and_preserves_all_original_bytes(reviewed, monkeypatch):
    original = study.LEDGER.read_bytes()
    calls = []

    def post(payload, task_prompt):
        calls.append(payload)
        return fake_post(payload, task_prompt)

    monkeypatch.setattr(resume, "post", post)
    result = resume.run(2)
    assert result["status"] == "complete" and result["attempts"] == 2
    assert len(calls) == 1
    assert study.LEDGER.read_bytes().startswith(original)
    rows = [json.loads(line) for line in study.LEDGER.read_text().splitlines()]
    assert rows[1]["case_id"] == reviewed["cases"][1]["case_id"]
    assert rows[1]["reviewed_normalized"] == rows[1]["normalized"]
    assert rows[1]["amendment_sha256"] == study.sha(resume.AMENDMENT)
    resume.load_reviewed()
    assert resume.run(1)["attempts"] == 2
    assert len(calls) == 1


def test_budget_guard_and_call_cap_precede_network(reviewed, monkeypatch):
    calls = []
    monkeypatch.setattr(resume, "post", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="request count"):
        resume.run(3)
    protocol = copy.deepcopy(reviewed["protocol"])
    protocol["budget_usd"] = 0.001
    save_json(study.PROTOCOL, protocol)
    write_rows([valid_row(reviewed)])
    pin_amendment()
    with pytest.raises(ValueError, match="Budget guard"):
        resume.run(1)
    assert calls == []
