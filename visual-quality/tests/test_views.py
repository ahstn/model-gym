import json

import pytest
from PIL import Image, ImageCms

from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl
from visual_quality.views import load_views, prepare_views


def dataset(tmp_path, *, size=(120, 60)):
    root = tmp_path / "data"
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    images = root / "sources"
    images.mkdir()
    rows = []
    for component in ("megalith", "vizwiz", "distortions", "clevr"):
        for index in range(3):
            identity = f"{component}-{index}"
            path = images / f"{identity}.png"
            Image.new("RGB", size, (index * 50, 50, 80)).save(path)
            rows.append(
                {
                    "id": identity,
                    "sha256": file_sha256(path),
                    "image_path": f"sources/{path.name}",
                    "split": "train",
                    "split_group": identity,
                    "component": component,
                    "photo_domain_eligible": False if component == "clevr" else None,
                }
            )
    write_jsonl(dataset_dir / "train.jsonl", rows)
    atomic_json(
        dataset_dir / "dataset.json",
        {
            "schema_version": 1,
            "image_root": "..",
            "manifests": {"train": "train.jsonl"},
        },
    )
    return dataset_dir


def replace_first(dataset_dir, image, **save_kwargs):
    rows = read_jsonl(dataset_dir / "train.jsonl")
    row = rows[0]
    path = dataset_dir.parent / row["image_path"]
    image.save(path, format="PNG", **save_kwargs)
    row["sha256"] = file_sha256(path)
    write_jsonl(dataset_dir / "train.jsonl", rows)
    return row["id"]


def test_round_robin_deterministic_reuse_and_source_manifests_unchanged(tmp_path):
    source = dataset(tmp_path)
    before = {name: file_sha256(source / name) for name in ("dataset.json", "train.jsonl")}
    output = tmp_path / "views"
    first = prepare_views(source, output, limit=6)
    rows = load_views(output)
    assert first["counts_by_component"] == {"megalith": 2, "vizwiz": 2, "distortions": 2}
    assert [r["component"] for r in rows] == ["megalith", "vizwiz", "distortions"] * 2
    assert all(row["photo_domain_eligible"] is None for row in rows)
    assert prepare_views(source, output, limit=6) == first
    assert first == prepare_views(source, tmp_path / "other", limit=6)
    assert before == {name: file_sha256(source / name) for name in before}


def test_full_image_resize_preserves_aspect_and_orientation(tmp_path):
    source = dataset(tmp_path)
    image = Image.new("RGB", (120, 60), "red")
    image.paste(Image.new("RGB", (60, 60), "blue"), (60, 0))
    exif = Image.Exif()
    exif[274] = 6
    identity = replace_first(source, image, exif=exif)
    output = tmp_path / "views"
    prepare_views(source, output, limit=9, max_edge=64)
    row = next(row for row in load_views(output) if row["image_id"] == identity)
    assert (row["width"], row["height"]) == (32, 64)
    assert row["preprocessing"]["native_dimensions"] == [120, 60]
    assert row["preprocessing"]["decoded_dimensions"] == [60, 120]
    assert row["preprocessing"]["orientation_applied"] is True
    assert row["preprocessing"]["crop_bounds"] is None
    with Image.open(output / row["image_path"]) as result:
        assert result.getpixel((16, 0)) == (255, 0, 0)
        assert result.getpixel((16, 63)) == (0, 0, 255)
        assert not result.getexif()
        assert result.info["srgb"] == 0


def test_transparency_white_and_no_upscale(tmp_path):
    source = dataset(tmp_path, size=(12, 8))
    identity = replace_first(source, Image.new("RGBA", (12, 8), (255, 0, 0, 0)))
    output = tmp_path / "views"
    prepare_views(source, output, limit=9)
    row = next(row for row in load_views(output) if row["image_id"] == identity)
    assert (row["width"], row["height"]) == (12, 8)
    assert row["preprocessing"]["alpha_composited_on_white"] is True
    assert row["preprocessing"]["resized"] is False
    with Image.open(output / row["image_path"]) as result:
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)


def test_valid_icc_converted_and_invalid_icc_fails_atomically(tmp_path):
    source = dataset(tmp_path)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    identity = replace_first(source, Image.new("RGB", (120, 60), "red"), icc_profile=profile)
    prepare_views(source, tmp_path / "views", limit=9)
    row = next(row for row in load_views(tmp_path / "views") if row["image_id"] == identity)
    assert row["preprocessing"]["source_icc_sha256"]
    assert row["preprocessing"]["colour_action"] == "embedded ICC converted to sRGB"
    replace_first(source, Image.new("RGB", (120, 60), "red"), icc_profile=b"not a profile")
    with pytest.raises(ValueError, match="ICC"):
        prepare_views(source, tmp_path / "bad-views", limit=9)
    assert not (tmp_path / "bad-views").exists()
    assert not list(tmp_path.glob(".bad-views-*"))


@pytest.mark.parametrize("change", ["view", "views_manifest", "source", "dataset", "train_manifest"])
def test_tampering_rejected(tmp_path, change):
    source = dataset(tmp_path)
    output = tmp_path / "views"
    prepare_views(source, output, limit=6)
    row = load_views(output)[0]
    paths = {
        "view": output / row["image_path"],
        "views_manifest": output / "views.jsonl",
        "source": source.parent / "sources" / f"{row['image_id']}.png",
        "dataset": source / "dataset.json",
        "train_manifest": source / "train.jsonl",
    }
    with paths[change].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash changed|manifest changed"):
        load_views(output)


def test_changed_request_or_incomplete_output_rejected(tmp_path):
    source = dataset(tmp_path)
    output = tmp_path / "views"
    prepare_views(source, output, limit=6)
    for kwargs in ({"limit": 5}, {"limit": 6, "seed": 1}, {"limit": 6, "max_edge": 512}):
        with pytest.raises(ValueError, match="different frozen"):
            prepare_views(source, output, **kwargs)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        prepare_views(source, incomplete, limit=6)


def test_bad_source_hash_path_escape_and_output_overlap_rejected(tmp_path):
    source = dataset(tmp_path)
    for output in (source, source / "views", source.parent):
        with pytest.raises(ValueError, match="separate"):
            prepare_views(source, output, limit=6)
    rows = read_jsonl(source / "train.jsonl")
    rows[0]["sha256"] = "0" * 64
    write_jsonl(source / "train.jsonl", rows)
    with pytest.raises(ValueError, match="Source image hash changed"):
        prepare_views(source, tmp_path / "bad-hash", limit=9)
    rows[0]["image_path"] = "../outside.png"
    write_jsonl(source / "train.jsonl", rows)
    with pytest.raises(ValueError, match="escapes"):
        prepare_views(source, tmp_path / "escape", limit=9)


def test_wrong_split_and_limit_rejected(tmp_path):
    source = dataset(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        prepare_views(source, tmp_path / "views", limit=101)
    with pytest.raises(ValueError, match="requested"):
        prepare_views(source, tmp_path / "views", limit=12)
    rows = read_jsonl(source / "train.jsonl")
    rows[0]["split"] = "test"
    write_jsonl(source / "train.jsonl", rows)
    with pytest.raises(ValueError, match="training-only"):
        prepare_views(source, tmp_path / "views", limit=6)


def test_preparation_metadata_integrity_rejected(tmp_path):
    source = dataset(tmp_path)
    output = tmp_path / "views"
    prepare_views(source, output, limit=6)
    path = output / "preparation.json"
    metadata = json.loads(path.read_text())
    metadata["policy"]["max_edge"] = 100
    atomic_json(path, metadata)
    with pytest.raises(ValueError, match="policy hash"):
        load_views(output)
