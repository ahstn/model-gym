"""The threshold choice must be sealed before held-out inference starts."""

import json
import sys
from types import SimpleNamespace

import pytest
import study


@pytest.fixture
def selection_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "HERE", tmp_path)
    monkeypatch.setattr(study, "DATA", tmp_path)
    (tmp_path / "protocol.json").write_text('{"frozen": true}\n')
    cases, rows = [], []
    for index in range(600):
        expected = index >= 300
        cases.append({"row_id": str(index), "split": "calibration", "expected_ai": expected})
        rows.append(
            {
                "row_id": str(index),
                "normalized": {"ai_probability": 0.9 if expected else 0.1, "uncertainty": "low"},
            }
        )
    (tmp_path / "calibration.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fake_runner = SimpleNamespace(
        load_study=lambda: ({}, cases, {}, {"calibration": rows, "validation": []}),
        failed=lambda row: row.get("failed", False),
    )
    monkeypatch.setitem(sys.modules, "runner", fake_runner)
    return tmp_path, cases, rows


def test_selection_records_inputs_and_locks_choice(selection_inputs):
    path, _, _ = selection_inputs
    selected = study.select()
    assert selected["rule"] == {"threshold": 0.9, "gate": "none"}
    assert selected["calibration_cases"] == 600
    assert selected["validation_attempts_at_selection"] == 0
    assert selected["protocol_sha256"] == study.sha(path / "protocol.json")
    assert selected["calibration_ledger_sha256"] == study.sha(path / "calibration.jsonl")
    assert study.read(path / "selection-lock.json")["selection_sha256"] == study.sha(path / "selection.json")
    with pytest.raises(ValueError, match="once before validation"):
        study.select()


def test_selection_refuses_any_validation_ledger(selection_inputs):
    path, _, _ = selection_inputs
    (path / "validation.jsonl").write_text("")
    with pytest.raises(ValueError, match="once before validation"):
        study.select()
    assert not (path / "selection.json").exists()


def test_selection_refuses_incomplete_calibration(selection_inputs):
    path, _, rows = selection_inputs
    rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        study.select()
    assert not (path / "selection.json").exists()


def test_selection_refuses_failed_calibration(selection_inputs):
    path, _, rows = selection_inputs
    rows[0]["failed"] = True
    with pytest.raises(ValueError, match="unresolved"):
        study.select()
    assert not (path / "selection.json").exists()
