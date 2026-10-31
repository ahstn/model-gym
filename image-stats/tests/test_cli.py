import json

import numpy as np
import pytest

from PIL import Image

from image_stats import EVIDENCE_NAMES, evidence_maps
from image_stats.cli import main
from tests.images import blur, encode, texture


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def corpus(directory, count=20):
    directory.mkdir()
    for seed in range(count):
        (directory / f"{seed:02}.png").write_bytes(encode(texture(64, 96, seed=seed)))
    return directory


def test_analyse_calibrate_describe(tmp_path, capsys):
    images = corpus(tmp_path / "images")
    stats = tmp_path / "stats.jsonl"
    assert main(["analyse", str(images), "--output", str(stats), "--workers", "2"]) == 0
    records = read_jsonl(stats)
    assert [record["id"] for record in records] == [f"{seed:02}.png" for seed in range(20)]
    assert json.loads(stats.with_suffix(".run.json").read_text())["images"] == 20

    reference = tmp_path / "reference.json"
    assert main(["calibrate", str(stats), "--output", str(reference), "--name", "tiny"]) == 0
    prompts = tmp_path / "prompts.jsonl"
    assert main(["describe", str(stats), "--reference", str(reference), "--output", str(prompts)]) == 0
    rows = read_jsonl(prompts)
    assert len(rows) == 20
    assert "ranked against 20 images in 'tiny'" in rows[0]["prompt_block"]
    capsys.readouterr()


def test_manifest_mode_uses_ids_and_records_errors(tmp_path, capsys):
    views = tmp_path / "views"
    (views / "images").mkdir(parents=True)
    (views / "images" / "a.png").write_bytes(encode(texture(64, 64)))
    (views / "images" / "broken.png").write_bytes(b"not an image")
    manifest = views / "views.jsonl"
    manifest.write_text(
        json.dumps({"view_id": "v1", "image_path": "images/a.png"})
        + "\n"
        + json.dumps({"view_id": "v2", "image_path": "images/broken.png"})
        + "\n"
    )
    stats = tmp_path / "stats.jsonl"
    assert main(["analyse", "--manifest", str(manifest), "--output", str(stats)]) == 1
    first, second = read_jsonl(stats)
    assert first["id"] == "v1" and "analysis" in first
    assert second["id"] == "v2" and "UnidentifiedImageError" in second["error"]
    capsys.readouterr()


def test_several_directories_get_prefixed_ids(tmp_path, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        (directory / "x.png").write_bytes(encode(texture(64, 64)))
    stats = tmp_path / "stats.jsonl"
    assert main(["analyse", str(first), str(second), "--output", str(stats)]) == 0
    assert [record["id"] for record in read_jsonl(stats)] == ["first/x.png", "second/x.png"]
    capsys.readouterr()


def test_duplicate_ids_are_refused(tmp_path, capsys):
    for name in ("one", "two"):
        directory = tmp_path / name / "photos"
        directory.mkdir(parents=True)
        (directory / "x.png").write_bytes(encode(texture(64, 64)))
    with pytest.raises(SystemExit) as stopped:
        main(["analyse", str(tmp_path / "one" / "photos"), str(tmp_path / "two" / "photos"), "--output", "x"])
    assert stopped.value.code == 2
    assert "Duplicate image IDs: photos/x.png" in capsys.readouterr().err


def test_unexpected_per_image_errors_become_records(tmp_path, monkeypatch, capsys):
    path = tmp_path / "a.png"
    path.write_bytes(encode(texture(64, 64)))

    def corrupt_exif(*args, **kwargs):
        raise KeyError("broken EXIF IFD")

    monkeypatch.setattr("image_stats.cli.analyse_file", corrupt_exif)
    stats = tmp_path / "stats.jsonl"
    assert main(["analyse", str(path), "--output", str(stats)]) == 1
    assert read_jsonl(stats)[0]["error"] == "KeyError: 'broken EXIF IFD'"
    capsys.readouterr()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["analyse", "x.png", "--output", "o.jsonl", "--max-edge", "10"], "--max-edge must be from 64 to 8192"),
        (["analyse", "x.png", "--output", "o.jsonl", "--workers", "0"], "--workers must be at least 1"),
    ],
)
def test_invalid_options_fail_before_work(argv, message, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_reference_mismatch_is_a_clean_error(tmp_path, capsys):
    images = corpus(tmp_path / "images")
    stats, small = tmp_path / "stats.jsonl", tmp_path / "small.jsonl"
    main(["analyse", str(images), "--output", str(stats)])
    main(["analyse", str(images), "--output", str(small), "--max-edge", "64"])
    main(["calibrate", str(stats), "--output", str(tmp_path / "ref.json"), "--name", "tiny"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        main(["describe", str(small), "--reference", str(tmp_path / "ref.json")])
    assert stopped.value.code == 2
    assert "error: Reference built with" in capsys.readouterr().err


def test_evidence_maps(tmp_path, capsys):
    path = tmp_path / "photo.png"
    path.write_bytes(encode(texture(200, 300)))
    maps = evidence_maps(path)
    assert tuple(maps) == EVIDENCE_NAMES
    assert maps["clipping"].size == (300, 200)
    assert maps["spectrum"].size == (200, 200)
    assert main(["evidence", str(path), "--output-dir", str(tmp_path / "out")]) == 0
    written = sorted(item.name for item in (tmp_path / "out").iterdir())
    assert written == sorted(f"photo.{name}.png" for name in EVIDENCE_NAMES)
    assert Image.open(tmp_path / "out" / "photo.sharpness.png").mode == "RGB"
    # A fixed scale keeps a blurred frame visibly darker instead of normalising it to its own peak.
    blurred = evidence_maps(encode(blur(texture(200, 300), 6)))["sharpness"]
    assert np.asarray(blurred).mean() < np.asarray(maps["sharpness"]).mean() - 20
    capsys.readouterr()
