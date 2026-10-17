from visual_quality.megalith import CC0_URL, MAX_IMAGE_PIXELS, candidate_indices, eligible


def test_selection_cache_does_not_modify_source_shards(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from visual_quality import megalith

    shard = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": str(index)} for index in range(10)]), shard)
    original = shard.read_bytes()
    monkeypatch.setattr(megalith, "_metadata", lambda root: [shard])
    selected = megalith._selection(tmp_path, 2, 123)
    assert (tmp_path / "selection-123.jsonl").is_file()
    assert shard.read_bytes() == original
    assert selected == megalith._selection(tmp_path, 2, 123)
    assert {row["metadata_row_index"] for row in selected} == set(range(10))


def test_download_keeps_original_bytes_orientation_and_source_label_provenance(tmp_path, monkeypatch):
    import shutil

    from PIL import Image

    from visual_quality import megalith

    original = tmp_path / "original.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 8), "red").save(original, exif=exif)
    downloads = []

    def fake_download(url, path, **kwargs):
        downloads.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
        return path

    monkeypatch.setattr(megalith, "download", fake_download)
    row = {
        "id": "abc123",
        "url": "https://example.org/image.jpg",
        "license": CC0_URL,
        "metadata_shard": "source.parquet",
        "metadata_row_index": 10,
        "selection_rank": 2,
        "aesthetic_score": 7,
        "hash": "0" * 32,
        "width": 12,
        "height": 8,
    }
    record = megalith._fetch_image(tmp_path, row)
    assert (tmp_path / "images/abc123.image").read_bytes() == original.read_bytes()
    assert (record["original_width"], record["original_height"]) == (12, 8)
    assert record["original_exif_orientation"] == 6
    assert record["photo_domain_eligible"] is None
    assert record["source_metadata_hash"] == row["hash"]
    assert record["source_hash_matches_downloaded_md5"] is False
    assert record["source_metadata_width"] == 12
    assert record["source_labels"]["aesthetic_score"]["human_ground_truth"] is False
    assert record == megalith._fetch_image(tmp_path, row)
    assert downloads == [row["url"]]
    assert megalith._fetch_image(tmp_path, {**row, "selection_rank": 40})["selection_rank"] == 40

    # A same-length local edit must not retain the original checksum/provenance.
    import pytest

    cached_image = tmp_path / "images/abc123.image"
    damaged = bytearray(cached_image.read_bytes())
    damaged[-2] ^= 1
    cached_image.write_bytes(damaged)
    with pytest.raises(ValueError, match="acquisition checksum"):
        megalith._fetch_image(tmp_path, row)


def test_truncated_pixel_data_is_not_counted_as_an_acquired_image(tmp_path, monkeypatch):
    import pytest
    from PIL import Image

    from visual_quality import megalith

    original = tmp_path / "original.jpg"
    Image.effect_noise((512, 512), 70).convert("RGB").save(original, quality=95)
    incomplete = original.read_bytes()[:-150]

    def fake_download(url, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(incomplete)
        return path

    monkeypatch.setattr(megalith, "download", fake_download)
    row = {
        "id": "badjpeg",
        "url": "https://example.org/image.jpg",
        "license": CC0_URL,
        "metadata_shard": "source.parquet",
        "metadata_row_index": 11,
        "selection_rank": 3,
        "aesthetic_score": 7,
    }
    with pytest.raises(OSError):
        megalith._fetch_image(tmp_path, row)
    assert not (tmp_path / "records/badjpeg.json").exists()


def test_failures_and_duplicate_bytes_are_replaced_in_candidate_order(tmp_path, monkeypatch):
    import requests

    from visual_quality import megalith

    candidates = [
        {
            "id": identity,
            "selection_rank": rank,
            "license": CC0_URL,
            "url": f"https://example.org/{identity}.jpg",
            "mime_type": "image/jpeg",
            "width": 100,
            "height": 100,
        }
        for rank, identity in enumerate("abcde")
    ]
    attempts = []

    def fake_fetch(root, row):
        attempts.append(row["id"])
        if row["id"] == "b":
            raise requests.HTTPError("404 Client Error")
        return {"source_id": row["id"], "sha256": "a" if row["id"] == "c" else row["id"]}

    monkeypatch.setattr(megalith, "_selection", lambda *args: candidates)
    monkeypatch.setattr(megalith, "_fetch_image", fake_fetch)
    first = megalith._acquire_source(tmp_path, 3, 1)
    assert [row["source_id"] for row in first] == ["a", "d", "e"]
    failures = megalith.read_jsonl(tmp_path / "failures-1.jsonl")
    assert [row["source_id"] for row in failures] == ["b", "c"]
    assert megalith._acquire_source(tmp_path, 3, 1) == first
    assert attempts.count("b") == 1


def test_public_adapter_resolves_source_directory_below_shared_data_root(tmp_path, monkeypatch):
    from visual_quality import megalith

    monkeypatch.setattr(megalith, "_acquire_source", lambda root, count, seed: (root, count, seed))
    assert megalith.acquire(tmp_path, 10, 12) == (tmp_path / "sources/megalith", 10, 12)


def test_seeded_selection_uses_entire_pool_and_keeps_prefix():
    selection = candidate_indices(10_000, 100, 42)
    assert selection == candidate_indices(10_000, 100, 42)
    assert selection == candidate_indices(10_000, 150, 42)[:100]
    assert len(set(selection)) == 100
    assert min(selection) < 1000 and max(selection) > 9000
    assert selection != candidate_indices(10_000, 100, 43)


def test_source_license_and_resource_limits_are_enforced_without_aesthetic_filter():
    row = {
        "license": CC0_URL,
        "url": "https://example.org/image.jpg",
        "mime_type": "image/jpeg",
        "width": 1000,
        "height": 1000,
        "aesthetic_score": 0,
    }
    assert eligible(row)
    assert eligible({**row, "aesthetic_score": 10})
    assert not eligible({**row, "license": "https://creativecommons.org/licenses/by/4.0/"})
    assert not eligible({**row, "license": "https://creativecommons.org/publicdomain/mark/1.0/"})
    assert not eligible({**row, "width": MAX_IMAGE_PIXELS})
    assert not eligible({**row, "url": "file:///tmp/image.jpg"})
