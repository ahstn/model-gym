"""Verify a downloaded R2 snapshot and write a portable image-label export."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def snapshot_path(root, key):
    rel = Path(key)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unexpected snapshot path: {key}")
    path = root / rel
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Snapshot path escapes restore directory: {key}")
    return path


def restore(root):
    root = root.resolve()
    snapshot = json.loads((root / "snapshot.json").read_text())
    for entry in snapshot["objects"]:
        path = snapshot_path(root, entry["key"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {path}")
        verify(path, entry["sha256"])
    archive = root / "metadata.tar.zst"
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            tar.extractall(root, filter="data")
    finally:
        process.stdout.close()
    if process.wait():
        raise RuntimeError("Archive decompression failed")
    restored = {entry["key"] for entry in snapshot["objects"]}
    for entry in json.loads((root / "metadata-inventory.json").read_text()):
        verify(snapshot_path(root, entry["path"]), entry["sha256"])
        restored.add(entry["path"])
    aliases = snapshot.get("legacy_path_aliases", {})
    if not set(aliases.values()) <= restored:
        raise ValueError("Historical path alias has no verified snapshot file")
    old_root = snapshot["source_data_root"].rstrip("/") + "/"

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            if value in aliases:
                return str(root / aliases[value])
            if value.startswith(old_root):
                key = "data/" + value[len(old_root) :]
                if key in restored:
                    return str(root / key)
        return value

    source = snapshot_path(
        root,
        snapshot.get("canonical_labels_key", "data/labeling-20000-2026-11-16/exports/labelled-images.jsonl"),
    )
    if "canonical_labels_sha256" in snapshot:
        verify(source, snapshot["canonical_labels_sha256"])
    image_hashes = {
        entry["key"]: entry["sha256"] for entry in snapshot["objects"] if entry["key"].startswith("data/")
    }
    restored_images = set()
    output = root / "labelled-images.portable.jsonl"
    count = 0
    with source.open() as stream, output.open("w") as destination:
        for line in stream:
            record = portable(json.loads(line))
            image = Path(record["image_path"]).resolve()
            if not image.is_relative_to(root):
                raise ValueError("Label does not point to a restored snapshot image")
            key = image.relative_to(root).as_posix()
            if image_hashes.get(key) != record["view_sha256"] or key in restored_images:
                raise ValueError("Label references a duplicate or unverified snapshot image")
            verify(image, record["view_sha256"])
            restored_images.add(key)
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count != snapshot["labels"] or restored_images != image_hashes.keys():
        raise ValueError(f"Unexpected label count: {count}")
    print(f"Verified {count} images and labels; portable export: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_directory", type=Path)
    args = parser.parse_args()
    restore(args.snapshot_directory.resolve())
