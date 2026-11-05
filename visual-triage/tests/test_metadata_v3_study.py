"""Offline study freeze, source isolation and forty-image final resume checks."""

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import MODELS, digest, image_part
from tracks.metadata_v3 import build_controls, runtime, study
from tracks.metadata_v3.features import extract_features


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    here = tmp_path / "tracks/metadata_v3"
    here.mkdir(parents=True)
    runs = tmp_path / "data/metadata-v3/runs"
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "HERE", here)
    monkeypatch.setattr(study, "RUNS", runs)
    monkeypatch.setattr(study, "CODE", ("execution.py",))
    (tmp_path / "execution.py").write_text("# frozen implementation\n")
    monkeypatch.setattr(runtime, "image_part", lambda sample: image_part(sample, root=tmp_path))
    samples = []
    for split, count in (("train", 40), ("dev", 40), ("test", 80)):
        for gold in (0, 1):
            for index in range(count):
                serial = len(samples)
                path = tmp_path / "views" / f"{serial}.png"
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (12, 8), (serial % 256, serial // 256, 80)).save(path)
                features = extract_features(path, path)
                image = str(path.relative_to(tmp_path))
                view_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                samples.append(
                    {
                        "id": f"{split}-{gold}-{index}",
                        "group_id": f"{split}-{gold}-{index // 8}",
                        "split": split,
                        "gold": gold,
                        "image": image,
                        "native_image": image,
                        "delivered_sha256": view_hash,
                        "native_sha256": view_hash,
                        "features": features,
                        "features_sha256": digest(features),
                    }
                )
    dataset = {"samples": samples}
    test = next(sample for sample in samples if sample["split"] == "test")
    controls = {
        "cases": [
            {
                **copy.deepcopy(test),
                "id": "control-one",
                "base_sample_id": test["id"],
                "condition": "strip_metadata",
                "arms": ["all_context"],
            }
        ]
    }
    baseline = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())
    write(here / "dataset.json", dataset)
    write(here / "controls.json", controls)
    write(here / "baseline.json", baseline)
    return here, dataset, controls, baseline, runs


def selections(fixture, protocol_hash):
    _here, _dataset, _controls, prompt, runs = fixture
    for arm in runtime.ARMS:
        path = runs / arm / "candidates/seed/prompt.json"
        runtime.write_frozen(path, prompt)
        runtime.write_frozen(
            runs / arm / "selection.json",
            {
                "metadata_arm": arm,
                "protocol_sha256": protocol_hash,
                "selected_prompt_path": str(path),
                "selected_prompt_sha256": digest(prompt),
                "selection_split": "dev",
                "test_used": False,
            },
        )


def test_freeze_and_code_hash_reject_mutation(frozen_inputs):
    here, _dataset, _controls, _baseline, _runs = frozen_inputs
    protocol = study.freeze()
    assert len(protocol["arms"]) == 7 and protocol["max_proposals"] == 20
    assert study.load_study()[0] == protocol
    with pytest.raises(FileExistsError):
        study.freeze()
    (study.ROOT / "execution.py").write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="execution code changed"):
        study.load_study()
    assert (here / "protocol.json").exists()


def test_control_builder_output_is_consumed_by_study_freeze(frozen_inputs):
    here, dataset, _controls, _baseline, _runs = frozen_inputs
    original = copy.deepcopy(dataset)
    controls = build_controls.build_controls(dataset, root=study.ROOT)
    assert dataset == original
    assert "samples" not in controls
    assert len(controls["cases"]) == 640
    assert Counter(case["condition"] for case in controls["cases"]) == {
        "camera_stripped": 160,
        "camera_swapped": 160,
        "resolution_stripped": 160,
        "all_stripped": 160,
    }
    bases = {sample["id"]: sample for sample in dataset["samples"]}
    for case in controls["cases"]:
        base = bases[case["base_sample_id"]]
        assert case["split"] == base["split"] == "test"
        assert case["gold"] == base["gold"]
        assert case["group_id"] == base["group_id"]
        assert case["image"] == base["image"]
        assert case["delivered_sha256"] == base["delivered_sha256"]
    write(here / "controls.json", controls)
    protocol = study.freeze()
    assert protocol["controls_sha256"] == study.sha(here / "controls.json")
    assert study.load_study()[0] == protocol


@pytest.mark.parametrize("field", ["dataset.json", "controls.json", "baseline.json"])
def test_frozen_inputs_cannot_change(frozen_inputs, field):
    here, _dataset, _controls, _baseline, _runs = frozen_inputs
    study.freeze()
    value = study.read(here / field)
    value["changed"] = True
    write(here / field, value)
    with pytest.raises(ValueError, match="changed"):
        study.load_study()


@pytest.mark.parametrize("change", ["float_gold", "group_split", "delivered_duplicate", "native_duplicate"])
def test_dataset_rejects_label_group_and_exact_duplicate_errors(frozen_inputs, change):
    _here, dataset, _controls, _baseline, _runs = frozen_inputs
    value = copy.deepcopy(dataset)
    if change == "float_gold":
        value["samples"][0]["gold"] = 0.0
    elif change == "group_split":
        value["samples"][80]["group_id"] = value["samples"][0]["group_id"]
    else:
        field = "delivered_sha256" if change == "delivered_duplicate" else "native_sha256"
        value["samples"][1][field] = value["samples"][0][field]
    with pytest.raises(ValueError):
        study.check_dataset(value)


@pytest.mark.parametrize(
    "change", ["duplicate_id", "wrong_split", "wrong_group", "wrong_image", "wrong_native_hash"]
)
def test_controls_are_frozen_to_held_out_base_pixels(frozen_inputs, change):
    here, _dataset, controls, _baseline, _runs = frozen_inputs
    if change == "duplicate_id":
        controls["cases"].append(copy.deepcopy(controls["cases"][0]))
    elif change == "wrong_split":
        controls["cases"][0]["split"] = "train"
    elif change == "wrong_group":
        controls["cases"][0]["group_id"] = "unrelated"
    elif change == "wrong_image":
        controls["cases"][0]["image"] = "views/0.png"
    else:
        controls["cases"][0]["native_sha256"] = "changed"
    write(here / "controls.json", controls)
    with pytest.raises(ValueError):
        study.freeze()
    assert not (here / "protocol.json").exists()


def test_final_forty_image_slices_resume_without_repeated_requests(frozen_inputs, monkeypatch):
    here, _dataset, _controls, baseline, runs = frozen_inputs
    study.freeze()
    protocol_hash = study.sha(here / "protocol.json")
    selections(frozen_inputs, protocol_hash)
    changed = {**baseline, "context": baseline["context"] + " Use the frozen camera evidence cautiously."}
    selected_path = runs / "camera/candidates/changed/prompt.json"
    runtime.write_frozen(selected_path, changed)
    receipt = study.read(runs / "camera/selection.json")
    write(
        runs / "camera/selection.json",
        {**receipt, "selected_prompt_path": str(selected_path), "selected_prompt_sha256": digest(changed)},
    )
    requests, slices = [], []
    interrupted = [False]

    def send(endpoint, payload):
        requests.append(payload)
        return {
            "status": 200,
            "seconds": 0.01,
            "response": {
                "model": MODELS["decider"],
                "usage": {"cost": 0.000001},
                "answers": {
                    "ai_probability": {"type": "noul", "noul": 0.5},
                    "medium": {
                        "type": "choice",
                        "choice": "photo",
                        "probabilities": {"photo": 1, "render": 0, "graphic": 0, "unknown": 0},
                    },
                    "uncertainty": {
                        "type": "choice",
                        "choice": "high",
                        "probabilities": {"low": 0, "medium": 0, "high": 1},
                    },
                },
            },
        }

    def evaluator(*args, **kwargs):
        if len(slices) == 3 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("Offline interruption before batch transport")
        result = runtime.evaluate(*args, **kwargs, post_fn=send)
        slices.append(result)
        assert result["summary"]["expected"] == len(args[0])
        assert len(result["per_image"]) == len(args[0])
        return result

    monkeypatch.setattr(study, "evaluate", evaluator)
    with pytest.raises(RuntimeError, match="Offline interruption"):
        study.run("final")
    assert len(requests) == 120
    study.run("final")
    assert len(requests) == 8 * 160
    assert all(len(runtime.read_ledger(runs / arm / "final/baseline.jsonl")) == 160 for arm in runtime.ARMS)
    assert len(runtime.read_ledger(runs / "camera/final/selected.jsonl")) == 160
    assert not any((runs / arm / "final/selected.jsonl").exists() for arm in runtime.ARMS if arm != "camera")
    study.run("final")
    assert len(requests) == 8 * 160
    study.run("controls")
    assert len(requests) == 8 * 160 + 1


def test_controls_require_all_native_attempts(frozen_inputs):
    here, _dataset, _controls, _baseline, runs = frozen_inputs
    study.freeze()
    protocol_hash = study.sha(here / "protocol.json")
    selections(frozen_inputs, protocol_hash)
    with pytest.raises(ValueError, match="Complete all native"):
        study.run("controls")
    assert not list(runs.glob("*/controls/*.jsonl"))


def test_native_completion_counts_invalid_attempts_but_rejects_missing_selected(frozen_inputs):
    here, dataset, _controls, baseline, runs = frozen_inputs
    study.freeze()
    protocol_hash = study.sha(here / "protocol.json")
    selections(frozen_inputs, protocol_hash)
    budget = study.load_study()[3]
    expected = [sample for sample in dataset["samples"] if sample["split"] == "test"]
    for arm in runtime.ARMS:
        for sample in expected:
            runtime.append_record(
                runs / arm / "final/baseline.jsonl",
                {
                    "metadata_arm": arm,
                    "split": "test",
                    "sample": sample["id"],
                    "protocol_sha256": protocol_hash,
                    "prompt_sha256": digest(baseline),
                    "delivered_sha256": sample["delivered_sha256"],
                    "normalized": None,
                    "status": 500,
                    "effective_cost_usd": 0.000001,
                },
            )
    study.require_native_completion(dataset, baseline, budget, protocol_hash)
    arm = runtime.ARMS[0]
    changed = {**baseline, "context": baseline["context"] + " Changed selected instructions."}
    path = runs / arm / "candidates/changed/prompt.json"
    runtime.write_frozen(path, changed)
    receipt = study.read(runs / arm / "selection.json")
    write(
        runs / arm / "selection.json",
        {**receipt, "selected_prompt_path": str(path), "selected_prompt_sha256": digest(changed)},
    )
    with pytest.raises(ValueError, match="Complete all native"):
        study.require_native_completion(dataset, baseline, budget, protocol_hash)
