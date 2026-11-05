"""Frozen inputs, append-only execution, and image compatibility checks."""

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.jev14 import client
from tracks.jev14 import run as study


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def prompt(task):
    if task == "quality":
        questions = {
            name: {
                "type": "score",
                "instructions": name,
                "criteria": ["Bad", "Poor", "Fair", "Good", "Excellent"],
            }
            for name in ("technical_quality", "aesthetics")
        }
    else:
        questions = {
            "ai_probability": {"type": "noul", "instructions": "Is this AI generated?"},
            "medium": {
                "type": "choice",
                "instructions": "Which medium?",
                "criteria": {"photo": "Photo", "art": "Art"},
            },
            "uncertainty": {"type": "noul", "instructions": "Is the origin uncertain?"},
        }
    return {
        "id": f"test-{task}",
        "task": task,
        "context": "Assess only the supplied pixels.",
        "questions": questions,
    }


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path
    track = root / "tracks/jev14"
    data = root / "data/jev14-v1"
    track.mkdir(parents=True)
    data.mkdir(parents=True)
    for key, value in {
        "ROOT": root,
        "TRACK": track,
        "DATA": data,
        "PROTOCOL": track / "protocol.json",
        "MANIFEST": data / "manifest.json",
        "LEDGER": data / "attempts.jsonl",
    }.items():
        monkeypatch.setattr(study, key, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    source = root / "frozen-source.txt"
    source.write_text("unchanged source")
    execution = root / "frozen-code.py"
    execution.write_text("# pinned execution\n")
    cases = []
    for index in range(2):
        path = root / f"view-{index}.jpg"
        Image.new("RGB", (512, 512), (index * 60, 40, 80)).save(path, "JPEG")
        sample = {"id": f"quality-{index:03d}", "image": path.name, "delivered_sha256": study.sha(path)}
        cases.append(study.case_record(sample, "quality", "primary"))
    save_json(study.MANIFEST, {"cases": cases, "primary_exclusions": []})
    sources = {source.name: {"sha256": study.sha(source)}}
    for task in ("quality", "ai"):
        path = root / f"prompts/final/{task}.json"
        save_json(path, prompt(task))
        sources[str(path.relative_to(root))] = {"sha256": study.sha(path)}
    protocol = {
        "manifest_sha256": study.sha(study.MANIFEST),
        "source_inputs": sources,
        "execution_sha256": {execution.name: study.sha(execution)},
        "maximum_attempts": 2,
        "budget_usd": 2,
        "reserve_per_request_usd": 0.012,
        "minimum_interval_seconds": 0,
        "expected_revision": client.REVISION,
    }
    save_json(study.PROTOCOL, protocol)
    return {"root": root, "cases": cases, "protocol": protocol}


def fake_post(payload, task_prompt):
    answers = {}
    for key, question in task_prompt["questions"].items():
        if question["type"] == "score":
            answers[key] = {"type": "score", "score": 2}
        elif question["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.5}
        else:
            answers[key] = {"type": "choice", "choice": next(iter(question["criteria"]))}
    body = {
        "id": "gen-offline",
        "model": client.REVISION,
        "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 1000, "output_tokens": 0, "cost": 0.000042},
    }
    return client.post(
        payload,
        task_prompt,
        transport=lambda *args, **kwargs: SimpleNamespace(status_code=200, headers={}, text=json.dumps(body)),
    )


def valid_row(frozen, index=0):
    case = frozen["cases"][index]
    task_prompt = prompt(case["task"])
    row = fake_post(client.make_payload(study.ROOT, case, task_prompt), task_prompt)
    row.update({key: value for key, value in case.items() if key not in {"image", "eligible"}})
    row.update(
        arm="jev14",
        attempt=index + 1,
        prompt_id=task_prompt["id"],
        prompt_sha256=digest(task_prompt),
        manifest_sha256=study.sha(study.MANIFEST),
        protocol_sha256=study.sha(study.PROTOCOL),
    )
    return row


def write_rows(rows):
    study.LEDGER.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_accepts_frozen_study_and_complete_prefix(frozen):
    write_rows([valid_row(frozen)])
    _, cases, rows = study.load_study()
    assert len(cases) == 2 and len(rows) == 1


@pytest.mark.parametrize("target", ["manifest", "source", "execution", "prompt"])
def test_changed_frozen_inputs_block_loading(frozen, target):
    paths = {
        "manifest": study.MANIFEST,
        "source": study.ROOT / "frozen-source.txt",
        "execution": study.ROOT / "frozen-code.py",
        "prompt": study.ROOT / "prompts/final/quality.json",
    }
    with paths[target].open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen"):
        study.load_study()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(case_id="primary/quality/skipped"),
        lambda row: row.update(delivered_sha256="0" * 64),
        lambda row: row.update(protocol_sha256="0" * 64),
        lambda row: row.update(manifest_sha256="0" * 64),
        lambda row: row.update(prompt_sha256="0" * 64),
        lambda row: row.update(attempt=2),
        lambda row: row.update(status=502),
        lambda row: row.update(resolved_model="typesafe/jev-1.14-other"),
        lambda row: row.update(normalized=None),
        lambda row: row.update(validation_error="invalid response"),
        lambda row: row.update(transport_error="Timeout"),
        lambda row: row.update(cost_validation_error="invalid cost"),
        lambda row: row.update(effective_cost_usd=None),
        lambda row: row.update(effective_cost_usd=-1),
        lambda row: row.update(effective_cost_usd=float("nan")),
        lambda row: row.update(effective_cost_usd=True),
    ],
)
def test_incomplete_or_corrupt_ledger_cannot_skip_case(frozen, mutation):
    row = valid_row(frozen)
    mutation(row)
    write_rows([row])
    with pytest.raises((ValueError, TypeError)):
        study.load_study()


def test_out_of_order_or_excess_ledger_is_rejected(frozen):
    write_rows([valid_row(frozen, 1)])
    with pytest.raises(ValueError):
        study.load_study()
    write_rows([valid_row(frozen), valid_row(frozen, 1), valid_row(frozen, 1)])
    with pytest.raises(ValueError, match="Too many attempts"):
        study.load_study()


def test_run_checks_all_image_hashes_before_any_call(frozen, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "post", lambda *args: calls.append(args))
    path = study.ROOT / frozen["cases"][1]["image"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen image changed"):
        study.run(1)
    assert calls == []
    assert not study.LEDGER.exists()


@pytest.mark.parametrize("count", [0, -1, 3])
def test_invalid_call_limit_does_not_call_provider(frozen, monkeypatch, count):
    calls = []
    monkeypatch.setattr(study, "post", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="request count"):
        study.run(count)
    assert calls == []


def test_budget_guard_stops_before_call(frozen, monkeypatch):
    protocol = copy.deepcopy(frozen["protocol"])
    protocol["budget_usd"] = 0.001
    save_json(study.PROTOCOL, protocol)
    calls = []
    monkeypatch.setattr(study, "post", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="budget guard"):
        study.run(1)
    assert calls == []


def test_run_appends_only_next_cases_and_preserves_previous_line(frozen, monkeypatch):
    monkeypatch.setattr(study, "post", fake_post)
    assert study.run(1)["status"] == "partial"
    first = study.LEDGER.read_bytes()
    result = study.run(1)
    assert result["status"] == "complete"
    assert result["attempts"] == 2
    assert study.LEDGER.read_bytes().startswith(first)
    rows = [json.loads(line) for line in study.LEDGER.read_text().splitlines()]
    assert [row["case_id"] for row in rows] == [case["case_id"] for case in frozen["cases"]]
    assert [row["attempt"] for row in rows] == [1, 2]
    assert result["reported_cost_usd"] == pytest.approx(0.000084)


@pytest.mark.parametrize("failure", ["validation_error", "unknown_cost"])
def test_failed_or_unpriced_call_is_preserved_and_blocks_resume(frozen, monkeypatch, failure):
    calls = []

    def fail(payload, task_prompt):
        calls.append(payload)
        row = fake_post(payload, task_prompt)
        if failure == "validation_error":
            row.update(validation_error="invalid result", normalized=None)
        else:
            row.update(cost_usd=None, effective_cost_usd=None)
        return row

    monkeypatch.setattr(study, "post", fail)
    result = study.run(2)
    assert result == {"status": "stopped", "attempts": 1}
    assert len(calls) == 1
    assert len(study.LEDGER.read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        study.run(2)
    assert len(calls) == 1


def test_prepare_preserves_primary_bytes_and_separates_padding_and_size_probes(frozen):
    root = study.ROOT
    photo = root / "original.jpg"
    panorama = root / "panorama.jpg"
    Image.new("RGB", (512, 512), "green").save(photo, "JPEG")
    Image.new("RGB", (512, 128), "orange").save(panorama, "JPEG")
    originals = {path: path.read_bytes() for path in (photo, panorama)}
    for task in ("quality", "ai"):
        samples = []
        for index in range(60):
            path = panorama if task == "quality" and index in (0, 19) else photo
            samples.append(
                {"id": f"{task}-{index:03d}", "image": path.name, "delivered_sha256": study.sha(path)}
            )
        save_json(root / f"evidence/benchmarks/{task}.json", {"samples": samples})
    save_json(
        root / "evidence/samples.json",
        {
            "samples": [
                {"id": f"sample-{index:03d}", "image": photo.name, "delivered_sha256": study.sha(photo)}
                for index in (24, 25)
            ]
        },
    )
    manifest = study.prepare_cases()
    cases = manifest["cases"]
    assert Counter(case["phase"] for case in cases) == {"primary": 118, "compatibility": 2, "size_probe": 6}
    assert Counter(case["task"] for case in cases if case["phase"] == "primary") == {"quality": 58, "ai": 60}
    assert {case["sample"] for case in manifest["primary_exclusions"]} == {"quality-000", "quality-019"}
    for path, original in originals.items():
        assert path.read_bytes() == original
    for case in cases:
        assert case["eligible"]
        if case["phase"] == "primary":
            assert case["delivered_sha256"] == case["parent_delivered_sha256"]
        elif case["phase"] == "compatibility":
            with Image.open(root / case["image"]) as padded, Image.open(panorama) as original:
                assert padded.size == (512, 256)
                assert padded.crop((0, 64, 512, 192)).tobytes() == original.convert("RGB").tobytes()
                assert padded.getpixel((0, 0)) == (128, 128, 128)
        else:
            assert case["image_width"] == case["image_height"]
            assert case["image_width"] in (320, 512, 1024)


def test_write_new_refuses_to_replace_existing_evidence(tmp_path):
    path = tmp_path / "frozen.json"
    study.write_new(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.write_new(path, {"replacement": True})
    assert path.read_bytes() == original
