"""Offline checks for image provenance and reproducible generated pixels."""

import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_quality.generate import MAX_VARIANT_EDGE, make_owned_controls, make_variants
from visual_quality.storage import file_sha256


def _parent(root: Path, identifier: str, group: str) -> dict:
    path = root / f"{identifier}.png"
    pixels = np.random.default_rng(42).integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return {
        "id": identifier,
        "source": "test_photos",
        "source_id": identifier,
        "source_url": "https://example.org/photo",
        "download_url": "https://example.org/photo.png",
        "source_revision": "pinned-revision",
        "source_split": "train",
        "source_group": group,
        "split": "development",
        "image_path": str(path),
        "license": {"name": "CC0"},
        "source_labels": {"mos": 4.7, "human_votes": 30},
        "labels": {"quality": 4.7},
        "teacher_labels": {"technical_quality": 4.6},
        "technical_quality": 4.7,
        "aesthetics": 4.2,
        "score": 4.7,
        "mos": 4.7,
        "content_kind": "photograph",
        "source_origin": "camera",
        "photo_domain_eligible": True,
    }


def test_variants_preserve_sources_and_groups_without_inheriting_scores(tmp_path: Path) -> None:
    parents = [_parent(tmp_path, "one", "scene-a"), _parent(tmp_path, "two", "scene-b")]
    before = deepcopy(parents)
    source_hashes = [file_sha256(Path(parent["image_path"])) for parent in parents]
    rows = make_variants(tmp_path / "output", parents, seed=7)
    assert len(rows) == 6
    assert len({row["id"] for row in rows}) == 6
    assert Counter(row["source_group"] for row in rows) == {"scene-a": 3, "scene-b": 3}
    for row in rows:
        assert Image.open(row["image_path"]).size == (96, 64)
        assert row["source_labels"] == {}
        assert row["technical_quality"] is None and row["aesthetics"] is None
        assert not {"mos", "score", "labels", "teacher_labels"} & row.keys()
        assert row["source_url"] == parents[0]["source_url"]
        assert row["parent_source"]["source_revision"] == "pinned-revision"
        assert row["license"] == {"name": "CC0"}
        assert row["photo_domain_eligible"] is True
        assert row["source"] == "generated_distortion"
        assert row["split"] == "development"
        assert row["preprocessing"]["resized"] is False
        assert row["preprocessing"]["resampling"] == "none"
        assert Path(row["image_path"]).is_absolute()
        assert file_sha256(Path(row["image_path"])) == row["sha256"]
    assert parents == before
    assert [file_sha256(Path(parent["image_path"])) for parent in parents] == source_hashes


def test_variants_are_identical_across_roots_input_order_and_reruns(tmp_path: Path) -> None:
    parents = [_parent(tmp_path, "one", "scene-a"), _parent(tmp_path, "two", "scene-b")]
    first = make_variants(tmp_path / "first", parents, seed=13)
    second = make_variants(tmp_path / "second", list(reversed(parents)), seed=13)
    assert [(row["id"], row["sha256"], row["transform"]) for row in first] == [
        (row["id"], row["sha256"], row["transform"]) for row in second
    ]
    assert make_variants(tmp_path / "first", parents, seed=13) == first


def test_jpeg_variants_are_real_lossy_encodings(tmp_path: Path) -> None:
    parents = [_parent(tmp_path, f"parent-{index}", f"scene-{index}") for index in range(12)]
    rows = make_variants(tmp_path / "output", parents, seed=0)
    jpeg = [row for row in rows if row["transform"]["operator"] == "jpeg_compression"]
    assert jpeg
    lookup = {parent["id"]: parent for parent in parents}
    for row in jpeg:
        with (
            Image.open(row["image_path"]) as image,
            Image.open(lookup[row["parent_id"]]["image_path"]) as original,
        ):
            assert image.format == "JPEG"
            assert image.size == original.size
            assert np.any(np.asarray(image) != np.asarray(original))
    assert {row["transform"]["parameters"]["quality"] for row in jpeg} == {70, 30, 8}


def test_repeated_parent_ids_are_rejected(tmp_path: Path) -> None:
    parent = _parent(tmp_path, "one", "scene-a")
    with pytest.raises(ValueError, match="unique IDs"):
        make_variants(tmp_path, [parent, parent], seed=0)


def test_variants_preserve_display_orientation_without_editing_exif_source(tmp_path: Path) -> None:
    parent = _parent(tmp_path, "rotated", "scene-a")
    with Image.open(parent["image_path"]) as image:
        exif = Image.Exif()
        exif[274] = 6
        image.save(tmp_path / "rotated.jpg", exif=exif)
    parent["image_path"] = str(tmp_path / "rotated.jpg")
    original_hash = file_sha256(Path(parent["image_path"]))
    rows = make_variants(tmp_path / "output", [parent], seed=0)
    assert {(row["width"], row["height"]) for row in rows} == {(64, 96)}
    for row in rows:
        with Image.open(row["image_path"]) as image:
            assert image.getexif().get(274, 1) == 1
    assert file_sha256(Path(parent["image_path"])) == original_hash


def test_large_variant_views_are_bounded_and_record_resampling(tmp_path: Path) -> None:
    parent = _parent(tmp_path, "large", "scene-a")
    original = Image.new("RGB", (2048, 1024), "#bc9a75")
    original.save(parent["image_path"])
    source_hash = file_sha256(Path(parent["image_path"]))
    rows = make_variants(tmp_path / "output", [parent], seed=0)
    for row in rows:
        assert (row["width"], row["height"]) == (MAX_VARIANT_EDGE, MAX_VARIANT_EDGE // 2)
        assert row["preprocessing"] == {
            "orientation": "exif_transpose",
            "source_dimensions": {"width": 2048, "height": 1024},
            "delivered_dimensions": {"width": 1536, "height": 768},
            "max_edge": 1536,
            "resized": True,
            "resampling": "lanczos",
            "aspect_ratio_policy": "preserve_with_integer_rounding",
            "source_aspect_ratio": 2.0,
            "delivered_aspect_ratio": 2.0,
        }
        assert row["technical_quality"] is None and row["aesthetics"] is None
    assert file_sha256(Path(parent["image_path"])) == source_hash
    with Image.open(parent["image_path"]) as unchanged:
        assert unchanged.size == (2048, 1024)


def test_owned_controls_are_reproducible_and_grouped_without_quality_claims(tmp_path: Path) -> None:
    rows = make_owned_controls(tmp_path / "first", count=240, seed=23)
    repeated = make_owned_controls(tmp_path / "second", count=240, seed=23)
    assert len({row["id"] for row in rows}) == 240
    assert len({row["sha256"] for row in rows}) == 240
    assert len({row["source_group"] for row in rows}) == 60
    assert set(Counter(row["source_group"] for row in rows).values()) == {4}
    assert [row["sha256"] for row in repeated] == [row["sha256"] for row in rows]
    assert sum(row["generation"]["layout_disrupted"] for row in rows) == 60
    for row in rows:
        assert row["photo_domain_eligible"] is False
        assert row["source_labels"] == {}
        assert row["technical_quality"] is None and row["aesthetics"] is None
        assert row["license"]["id"] == "LicenseRef-Project-Owned"
        assert row["license"]["font"]["terms"] == "No Rights Reserved"
        assert row["width"] == 640 and row["height"] == 480
        assert row["download_url"] is None
    short = make_owned_controls(tmp_path / "short", count=10, seed=23)
    assert [row["sha256"] for row in short] == [row["sha256"] for row in rows[:10]]


def test_control_count_bounds(tmp_path: Path) -> None:
    assert make_owned_controls(tmp_path, count=0, seed=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        make_owned_controls(tmp_path, count=-1, seed=0)


def test_resume_reuses_image_and_sidecar_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _parent(tmp_path, "one", "scene-a")
    rows = make_variants(tmp_path / "output", [parent], seed=7)
    files = [Path(row["image_path"]) for row in rows]
    files += [path.with_suffix(path.suffix + ".json") for path in files]
    old_mtime = 1_700_000_000_000_000_000
    for path in files:
        os.utime(path, ns=(old_mtime, old_mtime))

    def fail_decode(*args, **kwargs):
        raise AssertionError("A complete cache must not decode or transform the parent")

    monkeypatch.setattr("visual_quality.generate._parent_view", fail_decode)
    assert make_variants(tmp_path / "output", [parent], seed=7) == rows
    assert all(path.stat().st_mtime_ns == old_mtime for path in files)


@pytest.mark.parametrize("damage", ["corrupt", "missing", "invalid_sidecar"])
def test_resume_repairs_only_invalid_variant(tmp_path: Path, damage: str) -> None:
    parent = _parent(tmp_path, "one", "scene-a")
    rows = make_variants(tmp_path / "output", [parent], seed=7)
    first, second, third = [Path(row["image_path"]) for row in rows]
    stable_times = {path: path.stat().st_mtime_ns for path in (second, third)}
    if damage == "corrupt":
        first.write_bytes(b"broken image")
    elif damage == "missing":
        first.unlink()
    else:
        first.with_suffix(first.suffix + ".json").write_text("{incomplete")
    repaired = make_variants(tmp_path / "output", [parent], seed=7)
    assert repaired == rows
    assert file_sha256(first) == rows[0]["sha256"]
    assert {path: path.stat().st_mtime_ns for path in (second, third)} == stable_times


def test_changed_parent_invalidates_cache_even_when_supplied_hash_is_stale(tmp_path: Path) -> None:
    parent = _parent(tmp_path, "one", "scene-a")
    parent["sha256"] = file_sha256(Path(parent["image_path"]))
    previous = make_variants(tmp_path / "output", [parent], seed=7)
    Image.new("RGB", (96, 64), "red").save(parent["image_path"])
    current = make_variants(tmp_path / "output", [parent], seed=7)
    assert [row["id"] for row in current] == [row["id"] for row in previous]
    assert all(a["sha256"] != b["sha256"] for a, b in zip(previous, current, strict=True))
    actual_parent_hash = file_sha256(Path(parent["image_path"]))
    assert actual_parent_hash != parent["sha256"]
    assert all(row["parent_source"]["sha256"] == actual_parent_hash for row in current)


def test_changed_split_does_not_return_stale_cached_provenance(tmp_path: Path) -> None:
    parent = _parent(tmp_path, "one", "scene-a")
    make_variants(tmp_path / "output", [parent], seed=7)
    parent["split"] = "test"
    rows = make_variants(tmp_path / "output", [parent], seed=7)
    assert all(row["split"] == "test" for row in rows)
