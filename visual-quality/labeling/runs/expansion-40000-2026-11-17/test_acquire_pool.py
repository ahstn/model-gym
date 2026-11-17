"""Candidate widening must remain deterministic, license-safe and score-blind."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expansion_40000_acquire_pool", HERE / "acquire_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def source(tmp_path, *, count=12):
    metadata = tmp_path / "source/metadata"
    metadata.mkdir(parents=True)
    rows = [
        {
            "id": f"source{i}",
            "url": f"https://example.test/{i}.jpg",
            "width": 100,
            "height": 100,
            "mime_type": "image/jpeg",
            "hash": str(i),
            "license": pool.megalith.CC0_URL,
            "aesthetic_score": i,
            "caption": f"subject{i}",
        }
        for i in range(count)
    ]
    path = metadata / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    readme = b"Pinned CC0 source description\n"
    (metadata / "README.md").write_bytes(readme)
    tree = [
        {"path": "part.parquet", "lfs": {"oid": pool.file_sha256(path)}},
        {
            "path": "README.md",
            "oid": hashlib.sha1(
                b"blob " + str(len(readme)).encode() + b"\0" + readme, usedforsecurity=False
            ).hexdigest(),
        },
    ]
    (metadata / "hub-tree.json").write_text(json.dumps(tree))
    (metadata / "hub-repo.json").write_text(json.dumps({"sha": pool.megalith.REVISION}))
    return metadata.parent, path, rows


def test_all_metadata_is_verified_against_pinned_revision(tmp_path):
    root, shard, _ = source(tmp_path)
    shards, pins = pool.verified_metadata(root)
    assert shards == [(shard, 12)]
    assert pins[str(shard)] == pool.file_sha256(shard)
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        pool.verified_metadata(root)


def test_selection_ignores_scores_and_captions_and_excludes_old_ids(tmp_path):
    _, shard, rows = source(tmp_path)
    first, _ = pool.select_rows([(shard, len(rows))], 5, 123, {"source1", "source2"})
    for row in rows:
        row["aesthetic_score"] = 1000 - row["aesthetic_score"]
        row["caption"] = "completely different"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    second, _ = pool.select_rows([(shard, len(rows))], 5, 123, {"source1", "source2"})
    assert first == second
    assert not ({row["id"] for row in first} & {"source1", "source2"})
    assert all("aesthetic_score" not in row and "caption" not in row for row in first)


def test_ineligible_source_terms_and_duplicate_urls_are_excluded(tmp_path):
    _, shard, rows = source(tmp_path)
    rows[0]["license"] = "https://example.test/not-cc0"
    rows[1]["width"] = 0
    rows[2]["url"] = "http://example.test/image.jpg"
    rows[3]["url"] = rows[4]["url"]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    selected, _ = pool.select_rows([(shard, len(rows))], 8, 123, set())
    assert len(selected) == len({row["url"] for row in selected}) == 8
    assert not ({row["id"] for row in selected} & {"source0", "source1", "source2"})


def test_new_pool_does_not_mutate_source_and_is_immutable_on_resume(tmp_path):
    root, shard, _ = source(tmp_path)
    original = shard.read_bytes()
    target = tmp_path / "new/acquisition/candidates.jsonl"
    first = pool.build_pool(root, target, 5, 123, set())
    assert pool.build_pool(root, target, 5, 123, set()) == first
    assert shard.read_bytes() == original
    assert Path(first["image_cache_root"]).is_relative_to(target.parent)
    assert first["source_aesthetic_scores_used"] is False
    with pytest.raises(ValueError, match="selection policy"):
        pool.build_pool(root, target, 6, 123, set())
    target.write_text(target.read_text() + "{}\n")
    with pytest.raises(ValueError, match="changed"):
        pool.build_pool(root, target, 5, 123, set())


def test_insufficient_eligible_pool_is_explicit(tmp_path):
    _, shard, rows = source(tmp_path)
    with pytest.raises(ValueError, match="Insufficient"):
        pool.select_rows([(shard, len(rows))], 12, 123, {"source1"})
