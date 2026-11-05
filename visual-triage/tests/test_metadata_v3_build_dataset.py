"""Native integrity, group balance, collision and fixed preprocessing checks."""

import copy
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.metadata_v3 import build_dataset as builder


def cohort():
    real, ai = {"samples": []}, {"samples": []}
    device, family = 0, 0
    for split, count in (("train", 40), ("dev", 40), ("test", 80)):
        for index in range(count):
            real["samples"].append(
                {
                    "id": f"real-{split}-{index}",
                    "split": split,
                    "group_id": f"device-{device + index // 8}",
                    "image": f"real/{split}/{index}.jpg",
                    "original_sha256": "a" * 64,
                    "license": "CC-BY-SA-4.0",
                    "source_url": "https://example.com/real",
                }
            )
        for index in range(count // 4):
            for generator in ("a", "b", "c", "d"):
                ai["samples"].append(
                    {
                        "image_id": str(family + index),
                        "group_id": f"family-{family + index}",
                        "split": split,
                        "generator": generator,
                        "file": f"ai/{generator}-{family + index}.png",
                        "original_sha256": "b" * 64,
                        "decoded_rgb_sha256": "c" * 64,
                        "prompt_sha256": f"prompt-{family + index}",
                        "license": {"id": "CC-BY-NC-SA-4.0"},
                        "archive_url": "https://example.com/archive",
                        "archive_member": f"{generator}/{family + index}.png",
                    }
                )
        device += count // 8
        family += count // 4
    return real, ai


def test_balanced_eight_image_groups_and_deterministic_ai_bundles():
    real, ai = cohort()
    rows = builder._normalise(real, ai)
    assert len(rows) == 320
    builder._validate_cohort(rows)
    shuffled = copy.deepcopy(ai)
    random.Random(123).shuffle(shuffled["samples"])
    assert builder._normalise(real, shuffled) == rows
    generated = [row for row in rows if row["gold"]]
    groups = {}
    for row in generated:
        groups.setdefault(row["group_id"], set()).add(row["source"]["native_family_id"])
    assert len(groups) == 20
    assert all(len(families) == 2 for families in groups.values())


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "prompt_leak", "family_leak", "generator_duplicate"]
)
def test_source_cohort_rejects_invalid_group_or_split(mutation):
    real, ai = cohort()
    if mutation == "missing":
        real["samples"].pop()
    elif mutation == "duplicate":
        real["samples"][-1]["id"] = real["samples"][0]["id"]
    elif mutation == "prompt_leak":
        ai["samples"][-1]["prompt_sha256"] = ai["samples"][0]["prompt_sha256"]
    elif mutation == "family_leak":
        ai["samples"][-1]["group_id"] = ai["samples"][0]["group_id"]
    else:
        ai["samples"][0]["generator"] = ai["samples"][1]["generator"]
    with pytest.raises(ValueError):
        builder._normalise(real, ai)


def test_fixed_delivery_preserves_native_and_removes_metadata(tmp_path):
    native, delivered = tmp_path / "native.jpg", tmp_path / "delivered.jpg"
    exif = Image.Exif()
    exif[34855] = 200
    exif[270] = "private caption"
    Image.new("RGB", (80, 40), "blue").save(native, exif=exif)
    before = builder.file_sha(native)
    builder._deliver(native, delivered)
    builder._deliver(native, delivered)
    assert builder.file_sha(native) == before
    with Image.open(delivered) as image:
        assert image.size == (512, 512)
        assert not image.getexif()
    delivered.write_bytes(b"conflicting bytes")
    with pytest.raises(ValueError, match="cache differs"):
        builder._deliver(native, delivered)


def fingerprint(identifier, phash):
    return {
        "original_sha256": identifier * 64,
        "decoded_rgb_sha256": (identifier + "rgb") * 20,
        "sized_rgb_sha256": (identifier + "size") * 20,
        "phash": phash,
    }


@pytest.mark.parametrize("kind", ["prior", "cross_split", "exact"])
def test_collision_is_reported_and_never_silently_dropped(kind):
    rows = [{"id": "left", "split": "train"}, {"id": "right", "split": "test"}]
    fps = {"left": fingerprint("a", "0000000000000000"), "right": fingerprint("b", "ffffffffffffffff")}
    priors = []
    if kind == "prior":
        priors = [{"id": "prior", **fingerprint("c", "0000000000000001")}]
    elif kind == "cross_split":
        fps["right"]["phash"] = "000000000000003f"  # Distance6 must be rejected.
    else:
        fps["right"]["decoded_rgb_sha256"] = fps["left"]["decoded_rgb_sha256"]
    with pytest.raises(ValueError, match="contact the experiment owner"):
        builder._check_collisions(rows, fps, priors)


def test_related_ai_variants_can_remain_within_one_split():
    rows = [{"id": "left", "split": "train"}, {"id": "right", "split": "train"}]
    fps = {"left": fingerprint("a", "0000000000000000"), "right": fingerprint("b", "0000000000000001")}
    assert builder._check_collisions(rows, fps, [])["cohort_images"] == 2


def test_measure_cache_is_ignored_data_and_hash_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "DATA", tmp_path / "data")
    path = tmp_path / "native.png"
    Image.new("RGB", (16, 16), "grey").save(path)
    row = {
        "id": "example",
        "native_image": "native.png",
        "native_sha256": builder.file_sha(path),
        "input_fingerprints": {},
        "gold": False,
        "split": "train",
    }
    summary = {
        "status": "no_manifest",
        "manifest_present": False,
        "signature_valid": None,
        "hard_binding_valid": None,
        "trust_status": "unknown",
        "ai_generation_claim": None,
    }
    monkeypatch.setattr(builder.provenance, "extract_c2pa", lambda path: summary)
    first = builder._measure(row, {"feature_helper_sha256": "code1"})
    assert first["features_sha256"] == digest(first["features"])
    assert first["preprocess_seconds"] >= 0
    monkeypatch.setattr(
        builder.provenance, "extract_c2pa", lambda path: pytest.fail("cache should avoid extraction")
    )
    assert builder._measure(row, {"feature_helper_sha256": "code1"}) == first
    cache = next((builder.DATA / "cache/features").glob("*.json"))
    record = json.loads(cache.read_bytes())
    record["features_sha256"] = "wrong"
    cache.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cache hash"):
        builder._measure(row, {"feature_helper_sha256": "code1"})


def test_manual_eligibility_required_before_final_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    real, ai = cohort()
    real_path, ai_path = tmp_path / "real.json", tmp_path / "ai.json"
    real_path.write_text(json.dumps(real))
    ai_path.write_text(json.dumps(ai))
    with pytest.raises(ValueError, match="manual AI eligibility"):
        builder.assemble(real_path, ai_path, tmp_path / "dataset.json")
    assert not (tmp_path / "dataset.json").exists()


def test_source_id_cannot_escape_delivered_directory():
    real, ai = cohort()
    real["samples"][0]["id"] = "../escape"
    with pytest.raises(ValueError, match="filename"):
        builder._normalise(real, ai)


def test_missing_prior_assets_do_not_count_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "tracks/old").mkdir(parents=True)
    (tmp_path / "tracks/old/dataset.json").write_text(
        json.dumps({"samples": [{"id": "old", "image": "data/old.png"}]})
    )
    with pytest.raises(ValueError, match="Prior image is absent"):
        builder._prior_fingerprints()


def test_fingerprint_cache_requires_current_source_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "DATA", tmp_path / "data")
    path = tmp_path / "native.png"
    Image.new("RGB", (16, 16), "grey").save(path)
    first = builder._fingerprints(path)
    assert first["original_sha256"] == builder.file_sha(path)
    monkeypatch.setattr(
        builder.acquire_real, "fingerprints", lambda path: pytest.fail("cache should avoid RGB decode")
    )
    assert builder._fingerprints(path) == first
    cache = next((builder.DATA / "cache/fingerprints").glob("*.json"))
    value = json.loads(cache.read_bytes())
    value["fingerprints"]["phash"] = "0000000000000000"
    cache.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fingerprint hash changed"):
        builder._fingerprints(path)


def test_source_hash_failure_occurs_before_any_delivered_write(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "DATA", tmp_path / "data")
    native = tmp_path / "native.png"
    Image.new("RGB", (8, 8), "grey").save(native)
    real_path, ai_path = tmp_path / "real.json", tmp_path / "ai.json"
    real_path.write_text('{"samples":[]}')
    ai_path.write_text('{"samples":[]}')
    monkeypatch.setattr(
        builder,
        "_normalise",
        lambda real, ai: [
            {
                "id": "example",
                "native_image": "native.png",
                "native_sha256": "0" * 64,
                "input_fingerprints": {},
            }
        ],
    )
    with pytest.raises(ValueError, match="Native source SHA256"):
        builder.assemble(real_path, ai_path, tmp_path / "assembled.json", staging=True)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "assembled.json").exists()


def test_staging_pipeline_produces_compact_hash_pinned_features_without_freezing_track(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "DATA", tmp_path / "data")
    native = tmp_path / "native.png"
    Image.new("RGB", (8, 8), "grey").save(native)
    real_path, ai_path = tmp_path / "real.json", tmp_path / "ai.json"
    real_path.write_text('{"samples":[]}')
    ai_path.write_text('{"samples":[]}')
    row = {
        "id": "example",
        "native_image": "native.png",
        "native_sha256": builder.file_sha(native),
        "input_fingerprints": {},
        "gold": False,
        "split": "train",
        "group_id": "device",
        "source": {},
    }
    monkeypatch.setattr(builder, "_normalise", lambda real, ai: [row])
    monkeypatch.setattr(builder, "_prior_fingerprints", list)
    monkeypatch.setattr(
        builder.provenance,
        "extract_c2pa",
        lambda path: {
            "status": "no_manifest",
            "manifest_present": False,
            "signature_valid": None,
            "hard_binding_valid": None,
            "trust_status": "unknown",
            "ai_generation_claim": None,
        },
    )
    target = tmp_path / "assembled.json"
    receipt = builder.assemble(real_path, ai_path, target, staging=True)
    assert receipt["sha256"] == builder.file_sha(target)
    data = json.loads(target.read_bytes())
    assert data["manual_eligibility_complete"] is False
    assert data["samples"][0]["features_sha256"] == digest(data["samples"][0]["features"])
    assert data["collision_audit"]["delivered_views"]["cohort_images"] == 1
    assert len(target.read_text().splitlines()) == 1
    assert not (tmp_path / "tracks/metadata_v3/dataset.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        builder.assemble(real_path, ai_path, target, staging=True)
