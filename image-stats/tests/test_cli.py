import json

from PIL import Image

from image_stats import EVIDENCE_NAMES, evidence_maps
from image_stats.cli import main
from tests.images import encode, texture


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
    capsys.readouterr()
