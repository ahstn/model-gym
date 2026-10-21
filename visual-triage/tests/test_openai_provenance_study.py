"""Offline checks for native cohort freezing, safe resume and failure handling."""

import copy
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.openai_provenance import study
from tracks.openai_provenance.client import _native_image


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    track = tmp_path / "tracks/openai_provenance"
    source = tmp_path / "tracks/metadata_v3"
    data = tmp_path / "data/openai-provenance-v1"
    track.mkdir(parents=True)
    (track / "client.py").write_text("# frozen test client\n")
    (track / "study.py").write_text("# frozen test runner\n")
    guide = data / "research/guide.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("not_detected does not establish camera origin\n")
    for name, value in (("ROOT", tmp_path), ("TRACK", track), ("SOURCE", source), ("DATA", data)):
        monkeypatch.setattr(study, name, value)
    monkeypatch.setattr(study.time, "sleep", lambda seconds: None)
    samples = []
    for generator, count in (("real", 80), ("dalle2", 20), ("dalle3", 20),
                             ("firefly", 20), ("midjourney-v5", 20)):
        for index in range(count):
            case = f"{generator}-{index:03}"
            native = tmp_path / "native" / f"{case}.png"
            native.parent.mkdir(exist_ok=True)
            Image.new("RGB", (9, 7), (len(samples), index, 13)).save(native)
            samples.append({
                "id": case, "split": "test", "group_id": f"group-{generator}-{index // 8}",
                "gold": generator != "real", "source": {"generator": None if generator == "real" else generator},
                "native_image": str(native.relative_to(tmp_path)), "native_sha256": study.sha(native),
                "image": f"delivered/{case}.jpg", "delivered_sha256": "0" * 64,
                "features": {"provenance": {"status": "no_manifest", "manifest_present": False}},
            })
    # A non-test record with no usable image must never enter the frozen cohort.
    samples.append({"id": "training-only", "split": "train"})
    write_json(source / "dataset.json", {"samples": samples})
    write_json(source / "protocol.json", {"dataset_sha256": study.sha(source / "dataset.json")})
    return {"root": tmp_path, "track": track, "source": source, "data": data, "samples": samples}


def freeze_inputs(inputs):
    study.freeze()
    return study.load_study()


def result(case, *, success=True):
    return {
        "input_sha256": case["native_sha256"], "started_at": datetime.now(UTC).isoformat(),
        "success": success, "http_status": 200 if success else 429, "elapsed_seconds": 0.1,
        "signals": {"c2pa": {"outcome": "not_detected"}, "synthid": {"outcome": "not_detected"}},
        "error": None if success else {"kind": "http_error"},
    }


def ledger_row(case, **changes):
    row = result(case)
    row.update(case=case["case"], protocol_sha256=study.sha(study.TRACK / "protocol.json"), attempt=1)
    row.update(changes)
    return row


def write_ledger(inputs, rows):
    (inputs["data"] / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_freeze_pins_complete_native_cohort_and_interleaves_sources(inputs):
    receipt = study.freeze()
    protocol, cases, rows = study.load_study()
    assert receipt["frozen_cases"] == 160
    assert len(cases) == len({case["case"] for case in cases}) == 160
    assert Counter(case["gold_ai"] for case in cases) == {True: 80, False: 80}
    assert Counter(case["source"] for case in cases) == {
        "real": 80, "dalle2": 20, "dalle3": 20, "firefly": 20, "midjourney-v5": 20,
    }
    assert sum(case["openai_origin"] for case in cases) == 40
    expected_order = ["dalle3", "real", "dalle2", "real", "firefly", "real", "midjourney-v5", "real"]
    assert [case["source"] for case in cases] == expected_order * 20
    assert rows == []
    assert protocol["manifest_sha256"] == study.sha(inputs["data"] / "manifest.json")
    assert protocol["source_dataset_sha256"] == study.sha(inputs["source"] / "dataset.json")
    assert protocol["guide_sha256"] == study.sha(inputs["data"] / "research/guide.md")
    assert protocol["maximum_attempts"] == 160
    for case in cases:
        assert study.sha(inputs["root"] / case["native_image"]) == case["native_sha256"]
        assert case["native_sha256"] != case["delivered_sha256"]


@pytest.mark.parametrize("change", ["dataset", "native", "cohort"])
def test_bad_inputs_fail_before_frozen_output_or_submission(inputs, monkeypatch, change):
    monkeypatch.setattr(study, "check_image", lambda *_: pytest.fail("Preflight must not call the API"))
    if change == "dataset":
        (inputs["source"] / "dataset.json").write_text("{}")
    elif change == "native":
        (inputs["root"] / inputs["samples"][0]["native_image"]).write_bytes(b"changed")
    else:
        write_json(inputs["source"] / "dataset.json", {"samples": inputs["samples"][1:]})
        write_json(inputs["source"] / "protocol.json", {
            "dataset_sha256": study.sha(inputs["source"] / "dataset.json"),
        })
    with pytest.raises(ValueError):
        study.freeze()
    assert not (inputs["track"] / "protocol.json").exists()
    assert not (inputs["data"] / "manifest.json").exists()
    assert not (inputs["data"] / "attempts.jsonl").exists()


def test_freeze_refuses_to_replace_existing_inputs(inputs):
    freeze_inputs(inputs)
    manifest = (inputs["data"] / "manifest.json").read_bytes()
    protocol = (inputs["track"] / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        study.freeze()
    assert (inputs["data"] / "manifest.json").read_bytes() == manifest
    assert (inputs["track"] / "protocol.json").read_bytes() == protocol


@pytest.mark.parametrize("target", ["manifest", "client", "study"])
def test_load_rejects_changed_manifest_or_execution(inputs, target):
    freeze_inputs(inputs)
    path = inputs["data"] / "manifest.json" if target == "manifest" else inputs["track"] / f"{target}.py"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen .* changed"):
        study.load_study()


@pytest.mark.parametrize("change", ["duplicate", "unexpected_case", "input_hash", "protocol_hash"])
def test_load_rejects_ledger_identity_errors(inputs, change):
    _, cases, _ = freeze_inputs(inputs)
    row = ledger_row(cases[0])
    rows = [row]
    if change == "duplicate":
        rows.append(copy.deepcopy(row))
    elif change == "unexpected_case":
        row["case"] = "not-in-the-study"
    elif change == "input_hash":
        row["input_sha256"] = "0" * 64
    else:
        row["protocol_sha256"] = "0" * 64
    write_ledger(inputs, rows)
    with pytest.raises(ValueError):
        study.load_study()


def test_run_skips_completed_case_and_submits_original_bytes(inputs, monkeypatch):
    _, cases, _ = freeze_inputs(inputs)
    original = ledger_row(cases[0])
    write_ledger(inputs, [original])
    seen = []

    def check(path, expected):
        seen.append((path, expected, path.read_bytes()))
        case = next(case for case in cases if case["native_sha256"] == expected)
        return result(case)

    monkeypatch.setattr(study, "check_image", check)
    assert study.run(2) == {"status": "partial", "attempts": 3}
    assert [(path, expected) for path, expected, _ in seen] == [
        (inputs["root"] / case["native_image"], case["native_sha256"]) for case in cases[1:3]
    ]
    _, _, rows = study.load_study()
    assert rows[0] == original
    assert [row["case"] for row in rows] == [case["case"] for case in cases[:3]]
    assert [row["attempt"] for row in rows] == [1, 2, 3]
    assert all(raw == path.read_bytes() for path, _, raw in seen)


def test_native_change_after_freeze_fails_before_submission_or_ledger(inputs, monkeypatch):
    _, cases, _ = freeze_inputs(inputs)
    (inputs["root"] / cases[0]["native_image"]).write_bytes(b"changed after freeze")

    def local_preflight(path, expected):
        _native_image(path, expected)
        pytest.fail("A changed native image passed local preflight")

    monkeypatch.setattr(study, "check_image", local_preflight)
    with pytest.raises(ValueError, match="SHA-256"):
        study.run(1)
    assert not (inputs["data"] / "attempts.jsonl").exists()


def test_first_failed_request_is_saved_and_blocks_all_retries(inputs, monkeypatch):
    _, cases, _ = freeze_inputs(inputs)
    calls = []

    def check(path, expected):
        calls.append(expected)
        return result(cases[len(calls) - 1], success=len(calls) == 1)

    monkeypatch.setattr(study, "check_image", check)
    assert study.run(5) == {"status": "stopped", "attempts": 2, "unattempted": 158}
    before = (inputs["data"] / "attempts.jsonl").read_bytes()
    _, _, rows = study.load_study()
    assert [row["success"] for row in rows] == [True, False]
    with pytest.raises(ValueError, match="previous failure"):
        study.run(5)
    assert len(calls) == 2
    assert (inputs["data"] / "attempts.jsonl").read_bytes() == before


@pytest.mark.parametrize("maximum", [0, -1, 161])
def test_invalid_run_limit_is_rejected_without_submission(inputs, monkeypatch, maximum):
    freeze_inputs(inputs)
    monkeypatch.setattr(study, "check_image", lambda *_: pytest.fail("Invalid limit must not submit"))
    with pytest.raises(ValueError, match="max-new"):
        study.run(maximum)
    assert not (inputs["data"] / "attempts.jsonl").exists()


def test_complete_ledger_does_not_submit_any_images(inputs, monkeypatch):
    _, cases, _ = freeze_inputs(inputs)
    write_ledger(inputs, [ledger_row(case, attempt=index) for index, case in enumerate(cases, 1)])
    monkeypatch.setattr(study, "check_image", lambda *_: pytest.fail("Completed study must not submit"))
    assert study.run(160) == {"status": "complete", "attempts": 160}
