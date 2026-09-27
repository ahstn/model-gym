"""Validate image bytes and keep source families together across splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from visual_quality.storage import file_sha256, read_jsonl, write_jsonl


def inspect_image(record: dict, root: Path) -> dict:
    """Hash the source bytes and the orientation-corrected RGB pixels separately."""
    item = dict(record)
    path = Path(item["image_path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image is outside the dataset root: {path}")
    if not item.get("license", {}).get("id"):
        raise ValueError(f"Missing license identity: {item['id']}")
    with Image.open(path) as image:
        original_width, original_height = image.size
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        width, height = image.size
        pixels = hashlib.sha256(f"{width}x{height}:RGB:".encode() + image.tobytes()).hexdigest()
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (gray[:, 1:] > gray[:, :-1]).ravel()
        dhash = sum(int(bit) << index for index, bit in enumerate(bits))
    item.update(
        image_path=path.relative_to(root.resolve()).as_posix(),
        sha256=file_sha256(path),
        decoded_sha256=pixels,
        dhash=f"{dhash:016x}",
        original_width=original_width,
        original_height=original_height,
        width=width,
        height=height,
        bytes=path.stat().st_size,
    )
    return item


def inspect_records(records: list[dict], root: Path) -> tuple[list[dict], list[dict]]:
    cache_path = root / "acquisition" / "validated.jsonl"
    cached = {row["id"]: row for row in read_jsonl(cache_path)}
    valid, failures = [], []

    def check(record: dict) -> tuple[dict | None, dict | None]:
        try:
            old = cached.get(record["id"])
            path = Path(record["image_path"])
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Image is outside the dataset root: {path}")
            if not record.get("license", {}).get("id"):
                raise ValueError(f"Missing license identity: {record['id']}")
            stat = path.stat()
            if (
                old
                and (root / old["image_path"]).resolve() == path
                and old.get("cache_mtime_ns") == stat.st_mtime_ns
                and old["bytes"] == stat.st_size
            ):
                # Keep updated source metadata even when the same bytes were inspected earlier.
                fields = {
                    key: old[key]
                    for key in (
                        "image_path",
                        "sha256",
                        "decoded_sha256",
                        "dhash",
                        "original_width",
                        "original_height",
                        "width",
                        "height",
                        "bytes",
                        "cache_mtime_ns",
                    )
                }
                return dict(record) | fields, None
            item = inspect_image(record, root)
            item["cache_mtime_ns"] = stat.st_mtime_ns
            return item, None
        except (OSError, ValueError, KeyError, Image.DecompressionBombError) as error:
            return None, {"id": record.get("id"), "error": str(error)}

    with ThreadPoolExecutor(max_workers=4) as workers:
        for item, failure in workers.map(check, records):
            if failure:
                failures.append(failure)
            else:
                valid.append(item)
    cached.update({row["id"]: row for row in valid})
    write_jsonl(cache_path, (cached[key] for key in sorted(cached)))
    return valid, failures


def deduplicate(records: list[dict], exclusion_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove byte/pixel duplicates and exact known evaluation matches."""
    blocked = {
        str(row[key])
        for row in exclusion_records
        for key in ("sha256", "decoded_sha256", "id")
        if row.get(key)
    }
    blocked_sources = {
        (row["source"], str(row["source_id"]))
        for row in exclusion_records
        if row.get("source") and row.get("source_id") is not None
    }
    seen, kept, dropped = set(), [], []
    for record in records:
        keys = {record["sha256"], record["decoded_sha256"]}
        if (
            blocked.intersection(keys | {record["id"]})
            or (record["source"], str(record["source_id"])) in blocked_sources
        ):
            dropped.append({"id": record["id"], "reason": "evaluation_exclusion"})
        elif seen.intersection(keys):
            dropped.append({"id": record["id"], "reason": "duplicate_pixels_or_bytes"})
        else:
            seen.update(keys)
            kept.append(record)
    return kept, dropped


class _HashIndex:
    """A BK-tree for conservative perceptual candidate matching in 64-bit hashes."""

    def __init__(self):
        self.tree = None

    def nearby(self, value: int, radius: int) -> list[str]:
        matches, stack = [], [self.tree] if self.tree else []
        while stack:
            key, ids, children = stack.pop()
            distance = (key ^ value).bit_count()
            if distance <= radius:
                matches.extend(ids)
            stack.extend(child for edge, child in children.items() if abs(edge - distance) <= radius)
        return matches

    def add(self, value: int, identity: str) -> None:
        if self.tree is None:
            self.tree = (value, [identity], {})
            return
        node = self.tree
        while True:
            key, ids, children = node
            distance = (key ^ value).bit_count()
            if distance == 0:
                ids.append(identity)
                return
            if distance not in children:
                children[distance] = (value, [identity], {})
                return
            node = children[distance]


def family_groups(records: list[dict], radius: int) -> tuple[dict[str, str], list[dict]]:
    """Join known families and near-duplicate photos; retain low-detail fault images."""
    parents = {row["id"]: row["id"] for row in records}

    def find(identity):
        while identity != parents[identity]:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    def join(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    families, photo_index, matches = {}, _HashIndex(), []
    for record in records:
        identity = record["id"]
        family = str(record["source_group"])
        if family in families:
            join(identity, families[family])
        families[family] = identity
        if record["content_kind"] != "photo":
            continue
        value = int(record["dhash"], 16)
        for match in photo_index.nearby(value, radius):
            join(identity, match)
            matches.append({"left": match, "right": identity, "method": f"dhash64<={radius}"})
        photo_index.add(value, identity)
    return {identity: find(identity) for identity in parents}, matches


def verify_manifest(records: list[dict], root: Path, *, check_bytes: bool = True) -> dict:
    ids, source_splits, groups, hashes = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    errors = []
    for row in records:
        identity = row["id"]
        if identity in ids:
            errors.append(f"Duplicate ID: {identity}")
        ids.add(identity)
        source_splits[row["source"]].add(row["source_split"])
        groups[row["split_group"]].add(row["split"])
        hashes[row["decoded_sha256"]].add(row["split"])
        if row["source_split"] not in {"train", "generated"}:
            errors.append(f"Non-training source split: {identity}")
        if row.get("technical_quality") is not None or row.get("aesthetics") is not None:
            errors.append(f"Unexpected inferred rating in assembled candidate: {identity}")
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"Image outside dataset root: {identity}")
        elif not path.exists() or (check_bytes and file_sha256(path) != row["sha256"]):
            errors.append(f"Missing or changed image: {identity}")
    errors.extend(f"Family leaks: {key}" for key, values in groups.items() if len(values) > 1)
    errors.extend(f"Pixels leak: {key}" for key, values in hashes.items() if len(values) > 1)
    report = {
        "rows": len(records),
        "counts_by_split": dict(Counter(row["split"] for row in records)),
        "counts_by_source": dict(Counter(row["source"] for row in records)),
        "source_splits": {key: sorted(value) for key, value in source_splits.items()},
        "source_families": len(groups),
        "errors": errors,
    }
    return report


def verify_dataset(root: Path, config: dict) -> dict:
    """Verify a complete published build, including quotas and manifest identity."""
    destination = root / config["name"]
    required = ["config.json", "dataset.json", "software.json", "report.json"] + [
        f"{s}.jsonl" for s in config["splits"]
    ]
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        return {"errors": [f"Missing build file: {name}" for name in missing]}
    saved_config = json.loads((destination / "config.json").read_text())
    saved_report = json.loads((destination / "report.json").read_text())
    records = [row for split in config["splits"] for row in read_jsonl(destination / f"{split}.jsonl")]
    report = verify_manifest(records, root)
    if saved_config != config:
        report["errors"].append("Frozen configuration differs from requested configuration")
    if saved_report.get("complete") is not True:
        report["errors"].append("Build is not marked complete")
    for name in ("config.json", "dataset.json", "software.json"):
        if file_sha256(destination / name) != saved_report.get("metadata_sha256", {}).get(name):
            report["errors"].append(f"Metadata hash differs: {name}")
    for split, quota in config["splits"].items():
        path = destination / f"{split}.jsonl"
        if file_sha256(path) != saved_report.get("manifest_sha256", {}).get(split):
            report["errors"].append(f"Manifest hash differs: {split}")
        split_records = read_jsonl(path)
        if any(row["split"] != split for row in split_records):
            report["errors"].append(f"Records carry wrong split in: {split}")
        actual = Counter(row["component"] for row in split_records)
        expected = Counter(quota)
        if split == "train":
            expected["distortions"] = config["distortion_parents"] * config["variants_per_parent"]
        if actual != expected:
            report["errors"].append(f"Wrong component quotas in {split}: {dict(actual)} != {dict(expected)}")
    return report
