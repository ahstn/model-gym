"""Restore and verify checksummed evidence from the public experiment archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener

ROOT = Path(__file__).resolve().parent
CHUNK_BYTES = 1024 * 1024
REPOSITORY = "https://github.com/ahstn/model-gym"
RESTORE_COMMAND = "python evidence_archive.py restore"


class EvidenceError(ValueError):
    """Archive or local evidence cannot be trusted."""


def _path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise EvidenceError("Archive member requires a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in value.split("/")):
        raise EvidenceError("Archive member path traversal is forbidden")
    if str(path) != value or ":" in path.parts[0]:
        raise EvidenceError("Archive member path is not canonical")
    return value


def _size(value):
    if type(value) is not int or value < 0:
        raise EvidenceError("Archive size must be a nonnegative integer")
    return value


def _sha(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise EvidenceError("Archive index requires a lowercase hexadecimal checksum")
    return value


def read_index(root=ROOT):
    """Validate index shape and every member before using paths or limits."""
    try:
        index_path = _safe_destination(root, "archive-index.json")
        index = json.loads(index_path.read_bytes())
    except (OSError, ValueError) as error:
        raise EvidenceError("Cannot read the evidence archive index") from error
    if (
        not isinstance(index, dict)
        or type(index.get("version")) is not int
        or index.get("version") != 1
        or not isinstance(index.get("archive_id"), str)
        or not index["archive_id"]
        or index.get("repository") != REPOSITORY
        or index.get("image_bytes_included") is not False
    ):
        raise EvidenceError("Unsupported evidence archive index")
    _sha(index.get("source_commit"), 40)
    archive = index.get("archive")
    if not isinstance(archive, dict):
        raise EvidenceError("Archive index has no archive descriptor")
    _sha(archive.get("sha256"))
    _size(archive.get("size_bytes"))
    url = archive.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/ahstn/model-gym/releases/download/")
    ):
        raise EvidenceError("Archive URL must be a public HTTPS repository release asset")
    members = index.get("members")
    if not isinstance(members, list) or not members:
        raise EvidenceError("Archive index requires members")
    names = set()
    for member in members:
        if not isinstance(member, dict) or set(member) != {"path", "size_bytes", "sha256", "restore"}:
            raise EvidenceError("Invalid archive member descriptor")
        name = _path(member["path"])
        if name in names:
            raise EvidenceError("Archive index contains duplicate members")
        names.add(name)
        _size(member["size_bytes"])
        _sha(member["sha256"])
        if type(member["restore"]) is not bool:
            raise EvidenceError("Member restore flag must be boolean")
    # A file cannot also be the parent directory of another indexed file.
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise EvidenceError("Archive member file and directory paths overlap")
    return index


def _hash_stream(handle, expected_size):
    checksum = hashlib.sha256()
    size = 0
    while chunk := handle.read(min(CHUNK_BYTES, expected_size - size + 1)):
        size += len(chunk)
        if size > expected_size:
            raise EvidenceError("Evidence exceeds indexed size")
        checksum.update(chunk)
    if size != expected_size:
        raise EvidenceError("Evidence size differs from the archive index")
    return checksum.hexdigest()


def _check_file(path, member):
    if not path.is_file() or path.stat().st_size != member["size_bytes"]:
        raise EvidenceError(f"Evidence is missing or has a different size: {member['path']}")
    with path.open("rb") as handle:
        if _hash_stream(handle, member["size_bytes"]) != member["sha256"]:
            raise EvidenceError(f"Evidence checksum differs: {member['path']}")


def _safe_destination(destination, name=None):
    """Reject symlinks before resolution, including any existing ancestor."""
    path = Path(os.path.abspath(destination))
    if name is not None:
        path = path.joinpath(*PurePosixPath(name).parts)
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise EvidenceError(f"Symlink destination is forbidden: {ancestor}")
        if ancestor != path and ancestor.exists() and not ancestor.is_dir():
            raise EvidenceError(f"Destination ancestor is not a directory: {ancestor}")
    return path


def _selected(index, snapshot=False, scope=None):
    if scope is not None:
        scope = _path(scope.rstrip("/"))
    return [
        member
        for member in index["members"]
        if (snapshot or member["restore"])
        and (scope is None or member["path"] == scope or member["path"].startswith(scope + "/"))
    ]


def verify(destination=ROOT, *, snapshot=False, root=ROOT):
    """Check every selected local member against the archive index."""
    index = read_index(root)
    destination = _safe_destination(destination)
    members = _selected(index, snapshot)
    for member in members:
        path = _safe_destination(destination, member["path"])
        _check_file(path, member)
    return {"archive_id": index["archive_id"], "verified": len(members), "snapshot": snapshot}


def require_restored(root: Path, scope: str | None = None):
    """Reject incomplete archived evidence before callers calculate or spend costs."""
    root = Path(root)
    index_path = root / "archive-index.json"
    if not index_path.exists() and not index_path.is_symlink():
        return
    try:
        index = read_index(root)
        destination = _safe_destination(root)
        for member in _selected(index, scope=scope):
            _check_file(_safe_destination(destination, member["path"]), member)
    except (EvidenceError, OSError) as error:
        raise EvidenceError(
            f"Archived evidence is missing or corrupt. Run `{RESTORE_COMMAND}` from {root}. {error}"
        ) from error


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlparse(new_url).scheme != "https":
            raise EvidenceError("Archive download redirected away from HTTPS")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _download(index, path):
    descriptor = index["archive"]
    # urllib follows GitHub's public asset redirects. Require HTTPS at the final endpoint.
    with (
        build_opener(_HTTPSRedirect()).open(descriptor["url"], timeout=60) as response,
        path.open("xb") as output,
    ):
        if urlparse(response.geturl()).scheme != "https":
            raise EvidenceError("Archive download redirected away from HTTPS")
        remaining = descriptor["size_bytes"]
        while chunk := response.read(min(CHUNK_BYTES, remaining + 1)):
            if len(chunk) > remaining:
                raise EvidenceError("Downloaded archive exceeds indexed size")
            output.write(chunk)
            remaining -= len(chunk)
        if remaining:
            raise EvidenceError("Downloaded archive is incomplete")


@contextmanager
def _validate_archive(path, index):
    descriptor = index["archive"]
    with Path(path).open("rb") as handle:
        if _hash_stream(handle, descriptor["size_bytes"]) != descriptor["sha256"]:
            raise EvidenceError("Archive checksum differs from the index")
    expected = {member["path"]: member for member in index["members"]}
    seen = set()
    with tarfile.open(path, mode="r:*") as archive:
        for entry in archive:
            name = _path(entry.name)
            if entry.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                raise EvidenceError(f"Archive member must be a regular file: {name}")
            if name in seen or name not in expected:
                raise EvidenceError(f"Archive has a duplicate or unexpected member: {name}")
            seen.add(name)
            member = expected[name]
            if entry.size != member["size_bytes"]:
                raise EvidenceError(f"Archive member size differs: {name}")
            with archive.extractfile(entry) as handle:
                if _hash_stream(handle, member["size_bytes"]) != member["sha256"]:
                    raise EvidenceError(f"Archive member checksum differs: {name}")
        if seen != set(expected):
            raise EvidenceError("Archive is missing indexed members")
        yield archive


def restore(archive_path=None, destination=ROOT, *, snapshot=False, root=ROOT):
    """Validate the full archive and all conflicts before copying selected files."""
    index = read_index(root)
    destination = _safe_destination(destination)
    members = _selected(index, snapshot)
    with tempfile.TemporaryDirectory(prefix="model-gym-evidence-") as temporary:
        # Copy local input to private staging so validation and extraction use identical bytes.
        staged = Path(temporary) / "archive"
        if archive_path is None:
            _download(index, staged)
        else:
            with Path(archive_path).open("rb") as source, staged.open("xb") as output:
                remaining = index["archive"]["size_bytes"]
                while chunk := source.read(min(CHUNK_BYTES, remaining + 1)):
                    if len(chunk) > remaining:
                        raise EvidenceError("Archive exceeds indexed size")
                    output.write(chunk)
                    remaining -= len(chunk)
                if remaining:
                    raise EvidenceError("Archive is incomplete")
        with _validate_archive(staged, index) as archive:
            pending = []
            for member in members:
                path = _safe_destination(destination, member["path"])
                if path.exists():
                    _check_file(path, member)
                else:
                    pending.append((member, path))
            # Explicit copies avoid tar extraction path and link semantics.
            for member, path in pending:
                _safe_destination(destination, member["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member["path"]) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_BYTES)
    return {
        "archive_id": index["archive_id"],
        "restored": len(pending),
        "identical_skipped": len(members) - len(pending),
        "snapshot": snapshot,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("restore", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--destination", type=Path, default=ROOT)
        child.add_argument("--snapshot", action="store_true")
        if command == "restore":
            child.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        result = (
            restore(args.archive, args.destination, snapshot=args.snapshot)
            if args.command == "restore"
            else verify(args.destination, snapshot=args.snapshot)
        )
    except (EvidenceError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"Evidence verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
