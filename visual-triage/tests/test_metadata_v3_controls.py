import copy
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.metadata_v3 import build_controls as controls
from tracks.metadata_v3.features import extract_features
from tracks.metadata_v3.provenance import extract_c2pa


@pytest.fixture
def base_dataset(tmp_path):
    samples = []
    for index in range(8):
        native = tmp_path / f"native-{index}.jpg"
        delivered = tmp_path / f"delivered-{index}.jpg"
        exif = Image.Exif()
        exif[34855] = 100 * (index + 1)
        image = Image.new("RGB", (12, 8), (index * 25, 10, 80))
        image.save(native, exif=exif)
        image.resize((6, 4)).save(delivered)
        features = extract_features(
            native,
            delivered,
            {
                "status": "no_manifest",
                "manifest_present": False,
                "signature_valid": None,
                "hard_binding_valid": None,
                "trust_status": "unknown",
                "ai_generation_claim": None,
            },
        )
        samples.append(
            {
                "id": f"sample-{index}",
                "group_id": f"source-{index}",
                "split": "test",
                "gold": bool(index % 2),
                "image": delivered.name,
                "delivered_sha256": hashlib.sha256(delivered.read_bytes()).hexdigest(),
                "native_image": native.name,
                "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                "features": features,
                "features_sha256": digest(features),
            }
        )
    return {"version": 3, "track": "metadata_v3", "samples": samples}


def test_context_controls_preserve_native_data_and_delivered_measurements(base_dataset, tmp_path):
    original = copy.deepcopy(base_dataset)
    first = controls.build_controls(base_dataset, root=tmp_path)
    second = controls.build_controls(base_dataset, root=tmp_path)
    assert first == second
    assert base_dataset == original
    assert len(first["cases"]) == 32
    assert len({row["id"] for row in first["cases"]}) == 32
    bases = {row["id"]: row for row in base_dataset["samples"]}
    for row in first["cases"]:
        base = bases[row["base_sample_id"]]
        assert row["group_id"] == base["group_id"] and row["gold"] == base["gold"]
        for key in ("image", "delivered_sha256", "native_image", "native_sha256"):
            assert row[key] == base[key]
        for group in ("tone", "texture"):
            assert row["features"][group] == base["features"][group]
        assert row["features_sha256"] == digest(row["features"])
        assert row["control_kind"] == "context_only" and row["c2pa_validation_pass"] is None
        if row["condition"] == "all_stripped":
            assert row["arms"] == ["all_context"]
            assert row["features"]["resolution"]["native"]["status"] == "not_supplied"
            assert row["features"]["resolution"]["delivered"] == base["features"]["resolution"]["delivered"]
            assert row["features"]["provenance"]["status"] == "no_manifest"
        elif row["condition"] == "camera_stripped":
            assert row["features"]["camera"]["iso"] is None
            assert row["features"]["provenance"] == base["features"]["provenance"]
        elif row["condition"] == "resolution_stripped":
            assert row["arms"] == ["resolution", "all_context"]
            assert row["features"]["resolution"]["native"] == {
                "status": "not_supplied",
                "width": None,
                "height": None,
                "aspect_ratio": None,
            }
            assert row["features"]["resolution"]["delivered"] == base["features"]["resolution"]["delivered"]
            assert row["features"]["camera"] == base["features"]["camera"]
            assert row["features"]["provenance"] == base["features"]["provenance"]


def test_camera_donors_are_balanced_within_recipient_class(base_dataset, tmp_path):
    result = controls.build_controls(base_dataset, root=tmp_path)
    bases = {s["id"]: s for s in base_dataset["samples"]}
    counts = {False: Counter(), True: Counter()}
    donor_ids = []
    for row in result["cases"]:
        if row["condition"] != "camera_swapped":
            continue
        donor = bases[row["donor_id"]]
        donor_ids.append(donor["id"])
        counts[row["gold"]][donor["gold"]] += 1
        assert donor["group_id"] != row["group_id"]
        assert row["features"]["camera"] == donor["features"]["camera"]
        assert donor["split"] == row["split"] == "test"
    assert counts == {False: Counter({False: 2, True: 2}), True: Counter({False: 2, True: 2})}
    assert len(set(donor_ids)) == len(donor_ids) == len(bases)
    assert set(donor_ids) == set(bases)


def test_permutation_handles_grouped_samples_without_reusing_donors():
    samples = [
        {
            "id": f"sample-{i:03}",
            "group_id": f"group-{i // 8}",
            "gold": int(i >= 80),
            "source": {
                "camera_group": f"camera-{i // 8}" if i < 80 else None,
                "generator": f"generator-{(i - 80) // 20}" if i >= 80 else None,
            },
        }
        for i in range(160)
    ]
    mapping = controls._camera_donors(samples)
    assert len({donor["id"] for donor in mapping.values()}) == 160
    assert {donor["id"] for donor in mapping.values()} == {sample["id"] for sample in samples}
    by_id = {sample["id"]: sample for sample in samples}
    counts = {0: Counter(), 1: Counter()}
    for recipient_id, donor in mapping.items():
        recipient = by_id[recipient_id]
        assert donor["group_id"] != recipient["group_id"]
        counts[recipient["gold"]][donor["gold"]] += 1
    assert counts == {0: Counter({0: 40, 1: 40}), 1: Counter({0: 40, 1: 40})}
    for recipient_gold in (0, 1):
        recipients = [(key, donor) for key, donor in mapping.items() if by_id[key]["gold"] == recipient_gold]
        cameras = Counter(donor["source"]["camera_group"] for _, donor in recipients if donor["gold"] == 0)
        generators = Counter(donor["source"]["generator"] for _, donor in recipients if donor["gold"] == 1)
        assert len(cameras) == 10 and set(cameras.values()) == {4}
        assert len(generators) == 4 and set(generators.values()) == {10}
    assert controls._camera_donors(list(reversed(samples))) == mapping


def test_minimum_four_group_permutation():
    samples = [{"id": str(i), "group_id": str(i), "gold": i // 2} for i in range(4)]
    mapping = controls._camera_donors(samples)
    assert len({donor["id"] for donor in mapping.values()}) == 4
    by_id = {sample["id"]: sample for sample in samples}
    for gold in (0, 1):
        assert Counter(donor["gold"] for key, donor in mapping.items() if by_id[key]["gold"] == gold) == {
            0: 1,
            1: 1,
        }
    assert all(key != donor["group_id"] for key, donor in mapping.items())


def test_integer_gold_labels_are_supported(base_dataset, tmp_path):
    for sample in base_dataset["samples"]:
        sample["gold"] = int(sample["gold"])
    assert len(controls.build_controls(base_dataset, root=tmp_path)["cases"]) == 32
    base_dataset["samples"][0]["gold"] = 0.0
    with pytest.raises(ValueError, match="gold"):
        controls.build_controls(base_dataset, root=tmp_path)


def test_train_dev_and_external_cases_are_not_controlled(base_dataset, tmp_path):
    for split in ("train", "dev", "external"):
        extra = copy.deepcopy(base_dataset["samples"][0])
        extra.update(id="excluded-" + split, split=split)
        base_dataset["samples"].append(extra)
    result = controls.build_controls(base_dataset, root=tmp_path)
    assert len(result["cases"]) == 32
    assert all(row["split"] == "test" for row in result["cases"])


def test_frozen_hash_and_path_boundaries_are_checked(base_dataset, tmp_path):
    base_dataset["samples"][0]["features"]["camera"]["iso"] = 999
    with pytest.raises(ValueError, match="feature hash"):
        controls.build_controls(base_dataset, root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        controls._file(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError, match="inside"):
        controls.build_controls(base_dataset, root=tmp_path, output_dir=tmp_path.parent)


def test_changed_source_bytes_reject_control_generation(base_dataset, tmp_path):
    (tmp_path / base_dataset["samples"][0]["native_image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash"):
        controls.build_controls(base_dataset, root=tmp_path)


def test_reusable_native_byte_interventions_with_local_signed_fixture(tmp_path):
    sdk = pytest.importorskip("c2pa")
    if sdk.__version__ != "0.38.0" or sdk.sdk_version() != "0.91.0":
        pytest.skip("frozen SDK required")
    directory = Path(__file__).resolve().parents[1] / "data/metadata-v3/candidates/ai"
    sources = sorted(directory.glob("firefly-*"))
    if len(sources) < 2:
        pytest.skip("local signed fixtures are not distributed")
    source, donor = sources[0].read_bytes(), sources[1].read_bytes()
    transforms = {
        "stripped": (controls.strip_credentials(source), "no_manifest", None, None),
        "swapped": (controls.copy_credentials(source, donor), "invalid", True, False),
        "signature": (controls.tamper_signature(source), "invalid", False, True),
    }
    with Image.open(sources[0]) as image:
        original = (image.size, image.convert("RGB").tobytes())
    for name, (data, status, signature, binding) in transforms.items():
        path = tmp_path / (name + ".jpg")
        path.write_bytes(data)
        summary = extract_c2pa(path)
        assert (summary["status"], summary["signature_valid"], summary["hard_binding_valid"]) == (
            status,
            signature,
            binding,
        )
        assert summary["ai_generation_claim"] is None
        with Image.open(path) as image:
            assert (image.size, image.convert("RGB").tobytes()) == original


@pytest.mark.parametrize("data", [b"", b"not a JPEG", b"\xff\xd8\xff\xeb\x00\x01", b"\xff\xd8\xff\xd9"])
def test_malformed_byte_controls_fail_explicitly(data):
    with pytest.raises(ValueError):
        controls.strip_credentials(data)
