"""Create and restore local v3 metadata evidence; image bytes remain outside the bundle."""

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evidence_archive import (
    EvidenceError,
    _check_file,
    _path,
    _safe_destination,
    _sha,
    _size,
    _validate_archive,
)

ROOT = Path(__file__).resolve().parents[2]
TRACK = "tracks/metadata_v3"
DATA = "data/metadata-v3"
EVIDENCE_FILES = {
    f"{TRACK}/dataset.json",
    f"{TRACK}/controls.json",
    f"{TRACK}/results/summary.json",
}
TEXT = {
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".html",
    ".csv",
    ".py",
    ".rs",
    ".js",
    ".toml",
    ".log",
    ".lock",
    ".xml",
}
IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp", ".svg", ".avif", ".heic", ".dng"}
NOTICE = "Local unpublished evidence archive. Image bytes were excluded and remain local; this is not a full image backup."
REPRODUCTION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "ARCHIVE.md",
    ".gitignore",
    ".gitattributes",
    "evidence_archive.py",
    "cost_reservations.py",
    "prompts/final/ai.json",
    "tracks/settings.py",
)


def _digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def _descriptor(root, path):
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _digest(path)}


def _complete(root):
    """Use the existing offline report validation, then require every expected attempt."""
    from tracks.metadata_v3.features import ARMS
    from tracks.metadata_v3.report import build_report

    runs = root / DATA / "runs"
    pending = runs / ".inflight.json"
    if pending.exists() and json.loads(pending.read_text()):
        raise EvidenceError("Finish or reconcile all in-flight requests before creating a bundle")
    result = build_report(
        root / TRACK / "dataset.json",
        root / TRACK / "protocol.json",
        runs,
        controls_path=root / TRACK / "controls.json",
        project_root=root,
        bootstrap_replicates=0,
    )
    if set(result["selection"]) != set(ARMS) or set(result["native"]) != set(ARMS):
        raise EvidenceError("All seven selections and native evaluations must finish first")
    panels = [panel for variants in result["native"].values() for panel in variants.values()]
    panels += [panel for arms in result["controls"].values() for panel in arms.values()]
    if not panels or any(panel["unattempted"] or panel["expected"] < 1 for panel in panels):
        raise EvidenceError("Complete every native and control attempt before creating a bundle")
    return {
        "protocol_sha256": result["protocol_sha256"],
        "selection_count": len(ARMS),
        "attempts_complete": True,
    }


def _inventory(root, output, receipt):
    files = set()
    for relative in (DATA, TRACK, "research/metadata-2026-10-12"):
        directory = _safe_destination(root, relative)
        if not directory.is_dir():
            raise EvidenceError("Missing evidence directory: " + relative)
        files.update(directory.rglob("*"))
    protocol = json.loads((root / TRACK / "protocol.json").read_text())
    files.update(_safe_destination(root, _path(name)) for name in protocol.get("code_sha256", {}))
    files.update(_safe_destination(root, name) for name in REPRODUCTION_FILES)
    files.update((root / "tests").glob("test_metadata_v3*.py"))
    included, excluded = [], []
    for path in sorted(files):
        relative = str(path.relative_to(root))
        if path in {receipt, root / TRACK / "evidence-bundle.json"}:
            continue
        if path.is_relative_to(output) or path.is_relative_to(root / DATA / "evidence-bundles"):
            continue
        if "__pycache__" in path.parts or path.name in {".coordinator.lock", ".budget.lock"}:
            continue
        _safe_destination(root, relative)
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError("Evidence must be a regular file: " + relative)
        record = _descriptor(root, path)
        if path.suffix.lower() in IMAGES:
            excluded.append(record)
            continue
        if (
            path.suffix.lower() not in TEXT
            and path.name not in {".gitignore", ".gitattributes"}
            and "license" not in path.name.lower()
            and "licence" not in path.name.lower()
        ):
            raise EvidenceError("Unknown evidence file type; review before bundling: " + relative)
        content = path.read_bytes()
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceError("Non-text evidence requires explicit review: " + relative) from error
        if b"\0" in content or _has_private_key(content):
            raise EvidenceError("Binary or private-key content is excluded: " + relative)
        included.append(record)
    return included, excluded


def _has_private_key(content):
    for marker in re.finditer(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content):
        end = content.find(b"-----END ", marker.end())
        body = content[marker.end() : end].replace(b"\\n", b"").strip()
        # The fetched C2PA settings guide contains this literal documentation placeholder.
        if end < 0 or body != b"MIIExample...":
            return True
    return False


def _write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create(root=ROOT, output=None, receipt=None):
    root = _safe_destination(root)
    output = _safe_destination(output or root / DATA / "evidence-bundles/final")
    receipt = _safe_destination(receipt or root / TRACK / "evidence-bundle.json")
    if not output.is_relative_to(root / DATA) or output == root / DATA:
        raise EvidenceError("Bundle output must be a directory within ignored data/metadata-v3")
    if not receipt.is_relative_to(root) or receipt.is_relative_to(output):
        raise EvidenceError("Use a compact receipt within the project and outside the bundle directory")
    paths = [output / "evidence.tar.gz", output / "index.json", receipt]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise EvidenceError("Bundle outputs are immutable; use fresh output and receipt paths")
    completion = _complete(root)
    members, images = _inventory(root, output, receipt)
    if not members:
        raise EvidenceError("No metadata evidence found")
    output.mkdir(parents=True, exist_ok=True)
    archive_path, index_path, _ = paths
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for member in members:
            source = _safe_destination(root, member["path"])
            _check_file(source, member)
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != member["sha256"]:
                raise EvidenceError("Evidence changed while bundling: " + member["path"])
            entry = tarfile.TarInfo(member["path"])
            entry.size, entry.mode = len(content), 0o644
            archive.addfile(entry, io.BytesIO(content))
    # Validate the written archive before recording a successful receipt.
    index = {
        "version": 1,
        "members": members,
        "excluded_images": images,
        "notice": NOTICE,
        "archive": _descriptor(root, archive_path),
        "completion": completion,
    }
    with _validate_archive(archive_path, index):
        pass
    _write_json(index_path, index)
    result = {
        "version": 1,
        "notice": NOTICE,
        "archive": index["archive"],
        "index": _descriptor(root, index_path),
        "member_count": len(members),
        "excluded_image_count": len(images),
        "excluded_image_bytes": sum(row["size_bytes"] for row in images),
        "completion": completion,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json(receipt, result)
    return result


def _read_bundle(receipt, root, archive_path=None):
    value = json.loads(Path(receipt).read_bytes())
    if value.get("version") != 1:
        raise EvidenceError("Unknown bundle receipt version")
    for descriptor in (value["archive"], value["index"]):
        _path(descriptor["path"])
        _size(descriptor["size_bytes"])
        _sha(descriptor["sha256"])
    index_path = _safe_destination(root, value["index"]["path"])
    _check_file(index_path, value["index"])
    index = json.loads(index_path.read_bytes())
    if index.get("version") != 1 or index.get("archive") != value["archive"]:
        raise EvidenceError("Bundle receipt and index differ")
    names = set()
    for member in index["members"] + index["excluded_images"]:
        if set(member) != {"path", "size_bytes", "sha256"}:
            raise EvidenceError("Invalid bundle member fields")
        name = _path(member["path"])
        _size(member["size_bytes"])
        _sha(member["sha256"])
        if name in names:
            raise EvidenceError("Duplicate bundle or excluded-image path")
        names.add(name)
    for name in names:
        if any(str(parent) in names for parent in Path(name).parents if str(parent) != "."):
            raise EvidenceError("Bundle file and directory paths overlap")
    if (
        len(index["members"]) != value["member_count"]
        or len(index["excluded_images"]) != value["excluded_image_count"]
    ):
        raise EvidenceError("Bundle inventory counts differ from receipt")
    return index, Path(archive_path) if archive_path else _safe_destination(root, value["archive"]["path"])


def verify(receipt, root=ROOT, archive_path=None):
    index, path = _read_bundle(receipt, root, archive_path)
    with _validate_archive(path, index):
        pass
    return {
        "verified_members": len(index["members"]),
        "excluded_images": len(index["excluded_images"]),
        "notice": NOTICE,
    }


def restore(receipt, root=ROOT, archive_path=None, destination=None, *, evidence_only=False):
    index, path = _read_bundle(receipt, root, archive_path)
    destination = _safe_destination(destination or root)
    with tempfile.TemporaryDirectory(prefix="metadata-v3-evidence-") as temporary:
        staged = Path(temporary) / "archive"
        with path.open("rb") as source, staged.open("xb") as output:
            remaining = index["archive"]["size_bytes"]
            while chunk := source.read(min(1024 * 1024, remaining + 1)):
                if len(chunk) > remaining:
                    raise EvidenceError("Archive exceeds indexed size")
                output.write(chunk)
                remaining -= len(chunk)
            if remaining:
                raise EvidenceError("Archive is incomplete")
        with _validate_archive(staged, index) as archive:
            members = [
                member
                for member in index["members"]
                if not evidence_only
                or member["path"] in EVIDENCE_FILES
                or (
                    member["path"].startswith(DATA + "/")
                    and not member["path"].startswith(DATA + "/evidence-bundles/")
                )
            ]
            pending = []
            for member in members:
                target = _safe_destination(destination, member["path"])
                if target.exists():
                    _check_file(target, member)
                else:
                    pending.append((member, target))
            for member, target in pending:
                _safe_destination(destination, member["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member["path"]) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return {
        "restored_members": len(pending),
        "identical_skipped": len(members) - len(pending),
        "excluded_images": len(index["excluded_images"]),
        "notice": NOTICE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify", "restore"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Restore ignored manifests, full summary and raw data without touching tracked reports or code",
    )
    args = parser.parse_args()
    if args.evidence_only and args.command != "restore":
        parser.error("--evidence-only requires restore")
    try:
        if args.command == "create":
            result = create(args.root, args.output, args.receipt)
        else:
            receipt = args.receipt or args.root / TRACK / "evidence-bundle.json"
            result = (
                verify(receipt, args.root, args.archive)
                if args.command == "verify"
                else restore(
                    receipt,
                    args.root,
                    args.archive,
                    args.destination,
                    evidence_only=args.evidence_only,
                )
            )
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f"Evidence bundle failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
