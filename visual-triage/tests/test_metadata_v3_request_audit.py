"""Offline reconstruction proves which frozen metadata was placed in task requests."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest, image_part, make_request
from tracks.metadata_v3 import request_audit as audit
from tracks.metadata_v3.features import ARMS, extract_features
from tracks.metadata_v3.runtime import context_for, feature_text


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_complete", lambda root: None)
    track = tmp_path / audit.HERE
    runs = tmp_path / "data/metadata-v3/runs"
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())
    prompt_path = track / "baseline.json"
    write_json(prompt_path, prompt)
    samples = []
    for identifier, gold, generator in (("real", False, None), ("firefly", True, "firefly")):
        path = tmp_path / (identifier + ".jpg")
        Image.new("RGB", (16, 12), (100, 120, 140)).save(path)
        features = extract_features(path, path)
        samples.append(
            {
                "id": identifier,
                "gold": gold,
                "split": "test",
                "group_id": identifier,
                "source": {"generator": generator, "id": "HIDDEN_SOURCE_ID"},
                "image": path.name,
                "delivered_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "features": features,
                "features_sha256": digest(features),
            }
        )
    train = {**samples[0], "id": "train", "split": "train"}
    samples.append(train)
    control = copy.deepcopy(samples[0])
    control.update(id="control", condition="swap", arms=["all_context"])
    control["features"]["tone"]["channel_moments"]["r"]["mean"] = 0.9
    control["features_sha256"] = digest(control["features"])
    write_json(track / "dataset.json", {"samples": samples})
    write_json(track / "controls.json", {"cases": [control]})
    protocol = {
        "dataset_sha256": audit._sha(track / "dataset.json"),
        "controls_sha256": audit._sha(track / "controls.json"),
        "code_sha256": {},
    }
    write_json(track / "protocol.json", protocol)
    protocol_hash = audit._sha(track / "protocol.json")

    def record(sample, arm):
        request_prompt = copy.deepcopy(prompt)
        text = feature_text(sample, arm)
        if text:
            request_prompt["context"] += "\n\n" + text
        endpoint, payload = make_request(
            "decider", image_part(sample, root=tmp_path), request_prompt, "low", 4096
        )
        return {
            "record_type": "task",
            "sample": sample["id"],
            "metadata_arm": arm,
            "task": "ai",
            "arm": "decider",
            "split": sample["split"],
            "endpoint": endpoint,
            "requested_model": payload["model"],
            "protocol_sha256": protocol_hash,
            "prompt_sha256": digest(prompt),
            "features_sha256": sample["features_sha256"],
            "context_sha256": digest(context_for(sample, arm)),
            "delivered_sha256": sample["delivered_sha256"],
            "request_sha256": digest({"endpoint": endpoint, "payload": payload}),
        }

    def ledger(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    for arm in ARMS:
        write_json(
            runs / arm / "selection.json",
            {"baseline_prompt_path": str(prompt_path), "selected_prompt_path": str(prompt_path)},
        )
        ledger(runs / arm / "final/baseline.jsonl", [record(sample, arm) for sample in samples[:2]])
    candidate = runs / "image_only/candidates/seed"
    write_json(candidate / "prompt.json", prompt)
    ledger(candidate / "train.jsonl", [record(train, "image_only")])
    ledger(runs / "all_context/controls/swap.jsonl", [record(control, "all_context")])
    ledger(runs / "image_only/reflection.jsonl", [{"record_type": "reflection"}])
    return tmp_path, runs


def test_all_arms_phases_and_examples_match_without_image_export(completed, monkeypatch):
    root, _ = completed
    calls = []
    original = audit.image_part

    def image(sample, root):
        calls.append(sample["image"])
        return original(sample, root=root)

    monkeypatch.setattr(audit, "image_part", image)
    result = audit.audit(root)
    assert result["task_attempts"] == 16
    assert result["reflection_records_not_audited"] == 1
    assert len(calls) == result["unique_delivered_images_verified"] == 2
    assert len(result["examples"]) == 14
    assert {row["phase"] for row in result["counts"]} == {"train", "native_final", "controls:swap"}
    assert result["gold_and_source_fields_used_to_construct_context"] is False
    rendered = json.dumps(result)
    assert "base64" in result["limits"] and "data:image/" not in rendered
    assert "HIDDEN_SOURCE_ID" not in rendered
    for example in result["examples"]:
        assert bool(example["feature_text"]) is (example["metadata_arm"] != "image_only")
        assert len(example["raw_record_sha256"]) == 64


@pytest.mark.parametrize(
    "field", ["request_sha256", "context_sha256", "features_sha256", "delivered_sha256", "prompt_sha256"]
)
def test_mismatched_raw_request_hashes_are_rejected(completed, field):
    root, runs = completed
    ledger = runs / "camera/final/baseline.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0][field] = "changed"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=field):
        audit.audit(root)


def test_delivered_bytes_are_verified_and_not_replaced(completed):
    root, _ = completed
    image = root / "real.jpg"
    image.write_bytes(b"changed delivered image")
    with pytest.raises(ValueError, match="Delivered image hash"):
        audit.audit(root)
    assert image.read_bytes() == b"changed delivered image"


def test_missing_held_out_arm_example_is_rejected(completed):
    root, runs = completed
    (runs / "camera/final/baseline.jsonl").unlink()
    with pytest.raises(ValueError, match="all seven arms"):
        audit.audit(root)
