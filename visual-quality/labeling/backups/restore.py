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


def restore(root):
    snapshot = json.loads((root / "snapshot.json").read_text())
    for entry in snapshot["objects"]:
        path = root / entry["key"]
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
    for entry in json.loads((root / "metadata-inventory.json").read_text()):
        verify(root / entry["path"], entry["sha256"])
    old_root = snapshot["source_data_root"].rstrip("/") + "/"

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(old_root):
            candidate = root / "data" / value[len(old_root) :]
            if candidate.is_file():
                return str(candidate)
        return value

    source = root / "data/labeling-20000-2026-11-16/exports/labelled-images.jsonl"
    output = root / "labelled-images.portable.jsonl"
    count = 0
    with source.open() as stream, output.open("w") as destination:
        for line in stream:
            record = portable(json.loads(line))
            verify(Path(record["image_path"]), record["view_sha256"])
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count != snapshot["labels"]:
        raise ValueError(f"Unexpected label count: {count}")
    print(f"Verified {count} images and labels; portable export: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_directory", type=Path)
    args = parser.parse_args()
    restore(args.snapshot_directory.resolve())
