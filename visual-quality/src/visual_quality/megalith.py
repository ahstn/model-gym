"""Acquire a seeded, source-pinned sample of Megalith's CC0 image pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image

from .storage import atomic_json, download, file_sha256, read_jsonl, write_jsonl

REPOSITORY = "Spawning/megalith-cc0"
REVISION = "3aa3a5760d944af63939db9eaf94fdd6c68af888"
SOURCE_URL = f"https://huggingface.co/datasets/{REPOSITORY}"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
MAX_IMAGE_BYTES = 48 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000


def _image_hashes(path: Path) -> dict[str, str]:
    sha256, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            md5.update(block)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def _source_observation(row: dict, hashes: dict[str, str]) -> dict:
    source_hash = row.get("hash")
    return {
        "source_metadata_hash": source_hash,
        "source_hash_semantics": "Undocumented; MD5 comparison is diagnostic, not required",
        "downloaded_md5": hashes["md5"],
        "source_hash_matches_downloaded_md5": source_hash == hashes["md5"] if source_hash else None,
        "source_metadata_width": row.get("width"),
        "source_metadata_height": row.get("height"),
    }


def candidate_indices(total: int, count: int, seed: int) -> list[int]:
    """Return a stable random prefix across the complete metadata pool."""
    if total < 0 or count < 0:
        raise ValueError("Pool and sample sizes must be nonnegative")
    return np.random.default_rng(seed).permutation(total)[:count].tolist()


def eligible(row: dict) -> bool:
    """Enforce source terms and a declared resource limit, without quality filtering."""
    width, height = row.get("width", 0), row.get("height", 0)
    return (
        str(row.get("license", "")).rstrip("/") == CC0_URL.rstrip("/")
        and str(row.get("url", "")).startswith("https://")
        and str(row.get("mime_type", "")).startswith("image/")
        and isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
        and width * height <= MAX_IMAGE_PIXELS
    )


def _metadata(root: Path) -> list[Path]:
    metadata = root / "metadata"
    tree_path = metadata / "hub-tree.json"
    if not tree_path.exists():
        response = requests.get(
            f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}",
            params={"recursive": "true", "expand": "false"},
            timeout=60,
        )
        response.raise_for_status()
        atomic_json(tree_path, response.json())
    tree = json.loads(tree_path.read_text())
    shards = sorted(
        (entry for entry in tree if entry["path"].endswith(".parquet")), key=lambda entry: entry["path"]
    )
    if not shards:
        raise RuntimeError("Pinned repository contains no parquet metadata")

    def fetch(entry: dict) -> Path:
        path = metadata / entry["path"]
        download(f"{SOURCE_URL}/resolve/{REVISION}/{entry['path']}", path, max_bytes=110_000_000)
        expected = entry.get("lfs", {}).get("oid")
        if expected and file_sha256(path) != expected:
            raise ValueError(f"Metadata checksum does not match the pinned source: {path}")
        return path

    with ThreadPoolExecutor(max_workers=3) as executor:
        paths = list(executor.map(fetch, shards))
    download(f"{SOURCE_URL}/resolve/{REVISION}/README.md", metadata / "README.md")
    return paths


def _selection(root: Path, count: int, seed: int) -> list[dict]:
    selection_path = root / f"selection-{seed}.jsonl"
    cached = read_jsonl(selection_path)
    # Cache enough replacements for HTTP/decode failures and duplicate images.
    requested = max(count * 4, count + 1000)
    if len(cached) >= requested or (root / f"selection-{seed}-complete.json").exists():
        return cached
    shards = _metadata(root)
    sizes = [pq.ParquetFile(path).metadata.num_rows for path in shards]
    total = sum(sizes)
    indices = candidate_indices(total, min(requested, total), seed)
    rank_by_index = {index: rank for rank, index in enumerate(indices)}
    selected = []
    offset = 0
    for path, size in zip(shards, sizes, strict=True):
        local = [index - offset for index in indices if offset <= index < offset + size]
        if local:
            table = pq.read_table(path).take(local)
            for index, row in zip(local, table.to_pylist(), strict=True):
                row["metadata_row_index"] = offset + index
                row["selection_rank"] = rank_by_index[offset + index]
                row["metadata_shard"] = path.name
                selected.append(row)
        offset += size
    selected.sort(key=lambda row: row["selection_rank"])
    write_jsonl(selection_path, selected)
    atomic_json(
        root / f"selection-{seed}-policy.json",
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "seed": seed,
            "source_rows": total,
            "sampled_candidates": len(selected),
            "sampling": "numpy PCG64 permutation across every metadata row; no aesthetic filtering",
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "photographer_groups": "Unavailable in this source; group only by image family",
            "content_review": "Source is described as photos; photo eligibility remains pending review",
        },
    )
    if len(selected) == total:
        atomic_json(root / f"selection-{seed}-complete.json", {"rows": total})
    return selected


def _fetch_image(root: Path, row: dict) -> dict:
    source_id = row["id"]
    # Source IDs are hashes, but do not trust them as filesystem paths.
    if not source_id.isalnum():
        raise ValueError("Unexpected source ID")
    cached_path = root / "records" / f"{source_id}.json"
    expected_digest = None
    if cached_path.exists():
        record = json.loads(cached_path.read_text())
        expected_digest = record["sha256"]
        image_path = Path(record["image_path"])
        if image_path.is_file():
            hashes = _image_hashes(image_path)
            if image_path.stat().st_size != record["image_bytes"] or hashes["sha256"] != expected_digest:
                raise ValueError("Cached original bytes no longer match their acquisition checksum")
            updated = {**record, **_source_observation(row, hashes)}
            if not record.get("pixel_decode_verified"):
                with Image.open(image_path) as image:
                    image.load()
                updated["pixel_decode_verified"] = True
            if updated != record:
                atomic_json(cached_path, updated)
            # An original can be reused in another seed's candidate order.
            return {**updated, "selection_rank": row["selection_rank"]}
    image_path = root / "images" / f"{source_id}.image"
    download(row["url"], image_path, max_bytes=MAX_IMAGE_BYTES)
    hashes = _image_hashes(image_path)
    image_digest = hashes["sha256"]
    if expected_digest and image_digest != expected_digest:
        raise ValueError("Downloaded original no longer matches its earlier acquisition checksum")
    with Image.open(image_path) as image:
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Decoded image exceeds the documented pixel limit")
        orientation = image.getexif().get(274, 1)
        image_format = image.format
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        image.load()
    record = {
        "id": f"megalith-{source_id}",
        "source": REPOSITORY,
        "source_id": source_id,
        "source_url": SOURCE_URL,
        "download_url": row["url"],
        "source_revision": REVISION,
        "source_split": "train",
        "source_group": f"megalith-{source_id}",
        "license": {
            "id": "CC0-1.0",
            "url": row["license"],
            "attribution": "Original creator absent from source metadata; CC0 requires no attribution",
            "evidence_path": str(root / "metadata" / "README.md"),
            "row_evidence": {"shard": row["metadata_shard"], "global_row_index": row["metadata_row_index"]},
        },
        "image_path": str(image_path),
        "image_bytes": image_path.stat().st_size,
        "sha256": image_digest,
        "original_width": width,
        "original_height": height,
        "original_exif_orientation": orientation,
        "image_format": image_format,
        "pixel_decode_verified": True,
        "source_labels": {
            "aesthetic_score": {
                "value": row.get("aesthetic_score"),
                "kind": "source_model_score",
                "human_ground_truth": False,
                "calibration": "Not established by this builder",
                "source_revision": REVISION,
            }
        },
        "source_caption": row.get("caption"),
        "content_kind": "photo",
        "source_origin": "camera",
        "content_kind_evidence": "Source-pool description only; individual image not reviewed",
        "photo_domain_eligible": None,
        "selection_rank": row["selection_rank"],
        "metadata_row_index": row["metadata_row_index"],
        **_source_observation(row, hashes),
    }
    atomic_json(cached_path, record)
    return record


def _acquire_source(root: Path, count: int, seed: int) -> list[dict]:
    """Download original image bytes; replace failures in the pinned candidate order."""
    if count < 1:
        raise ValueError("Count must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = _selection(root, count, seed)
    result, failures, seen_ids, seen_hashes = [], [], set(), set()
    permanent_failures = {
        item["source_id"]: item["reason"]
        for item in read_jsonl(root / f"failures-{seed}.jsonl")
        if item["reason"].startswith("HTTPError: 404 ")
    }

    def fetch(row: dict) -> tuple[dict, dict | None, str | None]:
        try:
            if not eligible(row):
                return row, None, "source terms, URL, MIME, or documented dimension limit"
            if row["id"] in permanent_failures:
                return row, None, permanent_failures[row["id"]]
            return row, _fetch_image(root, row), None
        except (requests.RequestException, OSError, ValueError, SyntaxError) as error:
            return row, None, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=32) as executor:
        for offset in range(0, len(candidates), 128):
            # executor.map preserves candidate order even when HTTP completion order differs.
            for row, record, error in executor.map(fetch, candidates[offset : offset + 128]):
                if record and record["source_id"] not in seen_ids and record["sha256"] not in seen_hashes:
                    seen_ids.add(record["source_id"])
                    seen_hashes.add(record["sha256"])
                    result.append(record)
                else:
                    failures.append(
                        {
                            "source_id": row["id"],
                            "selection_rank": row["selection_rank"],
                            "reason": error or "duplicate source ID or original file checksum",
                        }
                    )
            write_jsonl(root / f"acquired-{seed}.jsonl", result[:count])
            write_jsonl(root / f"failures-{seed}.jsonl", failures)
            print(
                f"Megalith: {min(len(result), count)}/{count} unique images; {len(failures)} skipped",
                flush=True,
            )
            if len(result) >= count:
                return result[:count]
    raise RuntimeError(f"Only acquired {len(result)}/{count} images from pinned replacement candidates")


def acquire(root: Path, count: int, seed: int) -> list[dict]:
    """Acquire photo candidates under a shared data root, as required by the builder."""
    return _acquire_source(Path(root) / "sources" / "megalith", count, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=20260927)
    arguments = parser.parse_args()
    acquire(arguments.root, arguments.count, arguments.seed)
