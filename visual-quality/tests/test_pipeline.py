import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from visual_quality.assemble import SOURCES, assemble, assign_splits, load_config, source_counts
from visual_quality.storage import atomic_json, download, file_sha256, read_jsonl, write_jsonl
from visual_quality.validation import (
    deduplicate,
    family_groups,
    inspect_records,
    verify_dataset,
    verify_manifest,
)


def record(root, identity, color, *, group=None, source="test", component="megalith"):
    path = root / f"{identity}.png"
    Image.new("RGB", (20, 30), color).save(path)
    return {
        "id": identity,
        "source": source,
        "source_id": identity,
        "source_group": group or identity,
        "image_path": str(path),
        "license": {"id": "CC0-1.0"},
        "content_kind": "photo",
        "source_split": "train",
        "component": component,
    }


def test_inspection_resume_and_modified_bytes(tmp_path):
    first = record(tmp_path, "first", "red")
    second = record(tmp_path, "second", "blue")
    rows, failures = inspect_records([first], tmp_path)
    assert not failures
    assert rows[0]["image_path"] == "first.png"
    original = rows[0]["sha256"]
    inspect_records([second], tmp_path)
    assert len(read_jsonl(tmp_path / "acquisition/validated.jsonl")) == 2
    with patch("visual_quality.validation.inspect_image", side_effect=AssertionError("cache missed")):
        cached, _ = inspect_records([first, second], tmp_path)
        assert cached[0]["sha256"] == original
    Image.new("RGB", (30, 30), "green").save(tmp_path / "first.png")
    changed, _ = inspect_records([first], tmp_path)
    assert changed[0]["sha256"] != original
    assert changed[0]["width"] == 30


def test_broken_and_outside_images_are_rejected(tmp_path):
    row = record(tmp_path, "bad", "red")
    (tmp_path / "bad.png").write_bytes(b"not an image")
    valid, failures = inspect_records([row], tmp_path)
    assert not valid and len(failures) == 1
    row["image_path"] = str(tmp_path.parent / "missing.png")
    assert inspect_records([row], tmp_path)[1]


def test_duplicate_pixels_and_source_scoped_exclusions(tmp_path):
    rows = [
        record(tmp_path, "first", "red", source="source_a"),
        record(tmp_path, "copy", "red", source="source_a"),
        record(tmp_path, "third", "blue", source="source_b"),
    ]
    rows, _ = inspect_records(rows, tmp_path)
    rows[0]["source_id"] = rows[2]["source_id"] = "same-external-id"
    kept, removed = deduplicate(rows, [{"source": "source_b", "source_id": "same-external-id"}])
    assert [row["id"] for row in kept] == ["first"]
    assert {row["reason"] for row in removed} == {"duplicate_pixels_or_bytes", "evaluation_exclusion"}


def test_near_photo_families_are_grouped_but_charts_are_not():
    records = [
        {"id": "a", "source_group": "a", "content_kind": "photo", "dhash": "0000000000000000"},
        {"id": "b", "source_group": "b", "content_kind": "photo", "dhash": "0000000000000001"},
        {"id": "c", "source_group": "c", "content_kind": "chart", "dhash": "0000000000000000"},
        {"id": "d", "source_group": "a", "content_kind": "photo", "dhash": "ffffffffffffffff"},
    ]
    groups, matches = family_groups(records, 2)
    assert groups["a"] == groups["b"] == groups["d"]
    assert groups["c"] != groups["a"]
    assert matches == [{"left": "a", "right": "b", "method": "dhash64<=2"}]


def test_exact_quota_allocation_never_splits_families():
    records, groups = [], {}
    for source in SOURCES:
        for group in range(6):
            for variant in range(2):
                identity = f"{source}-{group}-{variant}"
                records.append({"id": identity, "component": source, "content_kind": "photo"})
                groups[identity] = f"{source}-{group}"
    config = {
        "seed": 10,
        "splits": {
            split: {source: count for source in SOURCES}
            for split, count in [("train", 8), ("development", 2), ("test", 2)]
        },
    }
    assigned = assign_splits(records, groups, config)
    assert len(assigned) == len(records)
    assert assign_splits(list(reversed(records)), groups, config) != []
    for group in set(groups.values()):
        assert len({row["split"] for row in assigned if row["split_group"] == group}) == 1
    assert {r["id"]: r["split"] for r in assigned} == {
        r["id"]: r["split"] for r in assign_splits(list(reversed(records)), groups, config)
    }
    config["splits"]["test"]["megalith"] = 1
    with pytest.raises(ValueError, match="without breaking source families"):
        assign_splits(records, groups, config)


def test_verification_reports_leaks_and_tampering(tmp_path):
    records = [record(tmp_path, "one", "red"), record(tmp_path, "two", "blue")]
    records, _ = inspect_records(records, tmp_path)
    for row, split in zip(records, ["train", "test"]):
        row.update(split=split, split_group="same-scene", technical_quality=None, aesthetics=None)
    (tmp_path / "two.png").write_bytes(b"modified")
    errors = verify_manifest(records, tmp_path)["errors"]
    assert "Family leaks: same-scene" in errors
    assert "Missing or changed image: two" in errors


def test_atomic_records_and_config_quota_totals(tmp_path):
    path = tmp_path / "nested/records.jsonl"
    write_jsonl(path, [{"id": "first"}])
    write_jsonl(path, [{"id": "second"}])
    assert read_jsonl(path) == [{"id": "second"}]
    config = load_config(Path(__file__).parents[1] / "configs/pilot-10k.json")
    assert sum(config["splits"]["train"].values()) + config["distortion_parents"] * 3 == 10000
    assert sum(config["splits"]["development"].values()) == 400
    assert sum(config["splits"]["test"].values()) == 400
    assert source_counts(config)["megalith"] == 5500


def test_download_reuses_cache_and_rejects_oversize(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"cached")
    with patch("visual_quality.storage.requests.get", side_effect=AssertionError("network called")):
        assert download("https://example.org/image", path) == path
    path.unlink()

    class Response:
        def __init__(self):
            self.headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

    with (
        patch("visual_quality.storage.requests.get", return_value=Response()),
        pytest.raises(ValueError, match="exceeds"),
    ):
        download("https://example.org/image", path, max_bytes=10)
    assert not path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_cache_cannot_hide_path_or_license_change(tmp_path):
    row = record(tmp_path, "same-id", "red")
    a, b = tmp_path / "a.bmp", tmp_path / "b.bmp"
    Image.new("RGB", (20, 30), "red").save(a)
    Image.new("RGB", (20, 30), "blue").save(b)
    os.utime(b, ns=(a.stat().st_atime_ns, a.stat().st_mtime_ns))
    assert a.stat().st_size == b.stat().st_size
    row["image_path"] = str(a)
    first, _ = inspect_records([row], tmp_path)
    row["image_path"] = str(b)
    changed, _ = inspect_records([row], tmp_path)
    assert changed[0]["image_path"] == "b.bmp"
    assert changed[0]["sha256"] != first[0]["sha256"]
    row["license"] = {}
    assert inspect_records([row], tmp_path)[1]


def test_verifier_requires_complete_manifests_and_quotas(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/pilot-10k.json")
    destination = tmp_path / config["name"]
    atomic_json(destination / "config.json", config)
    atomic_json(destination / "dataset.json", {})
    atomic_json(destination / "software.json", {})
    atomic_json(destination / "report.json", {"complete": True})
    write_jsonl(destination / "train.jsonl", [])
    assert "Missing build file: development.jsonl" in verify_dataset(tmp_path, config)["errors"]
    for split in ["development", "test"]:
        write_jsonl(destination / f"{split}.jsonl", [])
    hashes = {split: file_sha256(destination / f"{split}.jsonl") for split in config["splits"]}
    atomic_json(destination / "report.json", {"complete": True, "manifest_sha256": hashes})
    errors = verify_dataset(tmp_path, config)["errors"]
    assert any("Wrong component quotas in train" in error for error in errors)
    (destination / "train.jsonl").write_text("\n")
    assert "Manifest hash differs: train" in verify_dataset(tmp_path, config)["errors"]


def test_failed_assembly_clears_old_success(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/pilot-10k.json")
    report_path = tmp_path / config["name"] / "report.json"
    atomic_json(report_path, {"complete": True})
    with pytest.raises(ValueError, match="Source not acquired"):
        assemble(tmp_path, config, [])
    assert json.loads(report_path.read_text())["complete"] is False


def test_offline_assembly_round_trip_is_complete_and_reproducible(tmp_path):
    config = {
        "name": "fixture-pilot",
        "seed": 42,
        "distortion_parents": 2,
        "variants_per_parent": 3,
        "near_duplicate_hamming_distance": 0,
        "splits": {
            split: {source: count for source in SOURCES}
            for split, count in [("train", 2), ("development", 1), ("test", 1)]
        },
        "spares": {source: 0 for source in SOURCES},
    }
    for source_index, source in enumerate(SOURCES):
        rows = []
        for index in range(4):
            identity = f"{source}-{index}"
            row = record(tmp_path, identity, "red", source=source, component=source)
            pixels = np.random.default_rng(source_index * 10 + index).integers(
                0, 256, (32, 32, 3), dtype="uint8"
            )
            Image.fromarray(pixels).save(row["image_path"])
            row.update(
                source_revision="fixture-v1",
                source_url="https://example.org/fixture",
                download_url=None,
                photo_domain_eligible=None,
                source_origin="camera",
                source_labels={"human_mos": 3.0},
            )
            rows.append(row)
        write_jsonl(tmp_path / "acquisition" / f"{source}.jsonl", rows)
    first = assemble(tmp_path, config, [])
    assert first["complete"]
    assert first["counts_by_split"] == {"development": 5, "test": 5, "train": 16}
    assert verify_dataset(tmp_path, config)["errors"] == []
    train = read_jsonl(tmp_path / config["name"] / "train.jsonl")
    variants = [row for row in train if row.get("parent_id")]
    assert len(variants) == 6
    assert all(row["source_labels"] == {} for row in variants)
    second = assemble(tmp_path, config, [])
    assert second["manifest_sha256"] == first["manifest_sha256"]


def test_owned_templates_are_separate_and_balanced_in_all_splits():
    records, groups = [], {}
    kinds = ("chart", "flowchart", "circuit_diagram", "mock_interface", "isometric_render")
    for kind in kinds:
        for template in range(12):
            for variant in range(4):
                identity = f"{kind}-{template}-{variant}"
                groups[identity] = f"{kind}-{template}"
                records.append({"id": identity, "component": "owned_controls", "content_kind": kind})
    config = {
        "seed": 42,
        "splits": {
            split: {source: count if source == "owned_controls" else 0 for source in SOURCES}
            for split, count in [("train", 200), ("development", 20), ("test", 20)]
        },
    }
    assigned = assign_splits(records, groups, config)
    for split, per_kind in [("train", 40), ("development", 4), ("test", 4)]:
        for kind in kinds:
            assert sum(row["split"] == split and row["content_kind"] == kind for row in assigned) == per_kind
