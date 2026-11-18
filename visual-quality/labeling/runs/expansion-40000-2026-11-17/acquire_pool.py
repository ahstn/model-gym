"""Widen Megalith candidates from locally cached, revision-pinned CC0 metadata.

The original 22K selection is immutable. This creates a new deterministic pool
under the expansion's ignored acquisition directory; it never downloads images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE / "src"))

from visual_quality import megalith
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl

# Neither captions nor supplied aesthetic ratings enter selection or acquisition.
COLUMNS = ("id", "url", "width", "height", "mime_type", "hash", "license")


def verified_metadata(source):
    import pyarrow.parquet as pq

    metadata = source / "metadata"
    repo = json.loads((metadata / "hub-repo.json").read_text())
    if repo["sha"] != megalith.REVISION:
        raise ValueError("Cached metadata has a different source revision")
    tree = json.loads((metadata / "hub-tree.json").read_text())
    entries = sorted(
        (entry for entry in tree if entry["path"].endswith(".parquet")), key=lambda row: row["path"]
    )
    if not entries:
        raise ValueError("Pinned parquet metadata unavailable locally")
    shards, pins = [], {}
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe metadata shard path")
        path = metadata / relative
        actual = file_sha256(path)
        if actual != entry.get("lfs", {}).get("oid"):
            raise ValueError(f"Pinned metadata checksum mismatch: {path}")
        shards.append((path, pq.ParquetFile(path).metadata.num_rows))
        pins[str(path)] = actual
    readme = metadata / "README.md"
    body = readme.read_bytes()
    git_hash = hashlib.sha1(
        b"blob " + str(len(body)).encode() + b"\0" + body, usedforsecurity=False
    ).hexdigest()
    if git_hash != next(entry["oid"] for entry in tree if entry["path"] == "README.md"):
        raise ValueError("Pinned source README differs from the cached tree")
    for path in (metadata / "hub-repo.json", metadata / "hub-tree.json", readme):
        pins[str(path)] = file_sha256(path)
    return shards, pins


def select_rows(shards, count, seed, excluded):
    """Sample from all shards; enlarge a deterministic prefix if eligibility needs it."""
    import pyarrow.parquet as pq

    if count < 1:
        raise ValueError("Candidate count must be positive")
    total = sum(size for _, size in shards)
    requested = min(total, max(count * 2, count + 10000))
    order = megalith.candidate_indices(total, total, seed)
    while True:
        indices = order[:requested]
        ranks = {index: rank for rank, index in enumerate(indices)}
        rows, rejected, offset = [], Counter(), 0
        for path, size in shards:
            local = [index - offset for index in indices if offset <= index < offset + size]
            if local:
                table = pq.read_table(path, columns=list(COLUMNS)).take(local)
                for index, row in zip(local, table.to_pylist(), strict=True):
                    identity = row["id"]
                    if identity in excluded:
                        rejected["previous_or_protected_identity"] += 1
                    elif not isinstance(identity, str) or not identity.isalnum():
                        rejected["unsafe_source_id"] += 1
                    elif not megalith.eligible(row):
                        rejected["source_terms_or_resource_limit"] += 1
                    else:
                        rows.append(
                            {
                                **row,
                                "metadata_row_index": offset + index,
                                "metadata_shard": path.name,
                                "selection_rank": ranks[offset + index],
                            }
                        )
            offset += size
        rows.sort(key=lambda row: row["selection_rank"])
        selected, identities, urls = [], set(), set()
        for row in rows:
            if row["id"] in identities or row["url"] in urls:
                rejected["duplicate_source_identity_or_url"] += 1
                continue
            selected.append(row)
            identities.add(row["id"])
            urls.add(row["url"])
            if len(selected) == count:
                return selected, {
                    "source_rows": total,
                    "sampled_candidates": requested,
                    "rejections": dict(rejected),
                }
        if requested == total:
            raise ValueError(f"Insufficient eligible fresh metadata: {len(selected)}/{count}")
        requested = min(total, requested * 2)


def build_pool(source, target, count, seed, excluded):
    provenance = target.with_suffix(".provenance.json")
    exclusions_sha = hashlib.sha256(json.dumps(sorted(excluded)).encode()).hexdigest()
    if target.exists() or provenance.exists():
        if not target.exists() or not provenance.exists():
            raise ValueError("Incomplete existing pool; inspect it before regenerating")
        policy = json.loads(provenance.read_text())
        if (policy["count"], policy["seed"], policy["excluded_ids_sha256"]) != (count, seed, exclusions_sha):
            raise ValueError("Existing candidate pool has a different frozen selection policy")
        if file_sha256(target) != policy["records_sha256"]:
            raise ValueError("Existing candidate pool changed")
        return policy
    shards, pins = verified_metadata(source)
    rows, sampling = select_rows(shards, count, seed, excluded)
    cache = target.parent / "megalith-source"
    readme = cache / "metadata/README.md"
    original = source / "metadata/README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    if readme.exists():
        if file_sha256(readme) != file_sha256(original):
            raise ValueError("New source-cache license evidence changed")
    else:
        shutil.copyfile(original, readme)
    write_jsonl(target, rows)
    policy = {
        "repository": megalith.REPOSITORY,
        "revision": megalith.REVISION,
        "count": len(rows),
        "seed": seed,
        "excluded_ids": len(excluded),
        "excluded_ids_sha256": exclusions_sha,
        "records_sha256": file_sha256(target),
        "metadata_pins": pins,
        "columns_read": list(COLUMNS),
        "sampling": "PCG64 permutation across all pinned metadata rows; eligible fresh identities only",
        "source_aesthetic_scores_used": False,
        "source_captions_used": False,
        "max_image_pixels": megalith.MAX_IMAGE_PIXELS,
        "max_image_bytes": megalith.MAX_IMAGE_BYTES,
        "image_cache_root": str(cache),
        "license_evidence_sha256": file_sha256(readme),
        "historical_metadata_unchanged": True,
        "script_sha256": file_sha256(Path(__file__)),
        **sampling,
    }
    atomic_json(provenance, policy)
    return policy


def candidates(path):
    """Return the verified pool and existing bounded acquisition adapter contract."""
    from corpus_acquire import bounded_download

    path = Path(path)
    policy = json.loads(path.with_suffix(".provenance.json").read_text())
    if policy["repository"] != megalith.REPOSITORY or policy["revision"] != megalith.REVISION:
        raise ValueError("Unexpected candidate source revision")
    if file_sha256(path) != policy["records_sha256"]:
        raise ValueError("Candidate metadata changed after freeze")
    cache = Path(policy["image_cache_root"])
    if cache.resolve() != (path.parent / "megalith-source").resolve():
        raise ValueError("Fresh acquisition must use its own source-cache directory")
    if file_sha256(cache / "metadata/README.md") != policy["license_evidence_sha256"]:
        raise ValueError("Source license evidence changed")
    rows = read_jsonl(path)
    if len(rows) != policy["count"] or any(not megalith.eligible(row) for row in rows):
        raise ValueError("Candidate count or eligibility changed")
    megalith.download = bounded_download
    return rows, lambda row: megalith._fetch_image(cache, row), path


def main():
    from corpus import DATA, DATA_ROOT, SEED, old_cases, pin_initial_evidence, read_lines

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=DATA / "acquisition/megalith-candidates.jsonl")
    args = parser.parse_args()
    if (DATA / "manifest.json").exists():
        raise ValueError("Selection is already frozen")
    if not args.output.resolve().is_relative_to((DATA / "acquisition").resolve()):
        raise ValueError("New pool must stay under this expansion's acquisition directory")
    pin_initial_evidence()
    protected = read_lines(DATA_ROOT / "pilot-10k-v1/development.jsonl") + read_lines(
        DATA_ROOT / "pilot-10k-v1/test.jsonl"
    )
    excluded = {row["source_id"] for row in old_cases() + protected}
    policy = build_pool(DATA_ROOT / "sources/megalith", args.output, args.count, SEED, excluded)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "count": policy["count"],
                "sha256": policy["records_sha256"],
                "source_rows": policy["source_rows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
