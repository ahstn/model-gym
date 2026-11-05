"""Offline checks for archive integrity, safe restoration and evidence guards."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_archive as archive

FILES = {
    "tracks/quality_v2/final/baseline.jsonl": b"original evidence\n",
    "tracks/ai_detection_v2/protocol.json": b"original protocol\n",
    "tracks/run.py": b"original source\n",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as output:
        for name, content, kind in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                entry.linkname = "../escape"
            output.addfile(entry, io.BytesIO(content) if entry.isfile() else None)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "index"
    root.mkdir()
    path = tmp_path / "archive.tar.gz"
    make_archive(path, [(name, content, tarfile.REGTYPE) for name, content in FILES.items()])
    index = {
        "version": 1,
        "archive_id": "original-v1",
        "source_commit": "b" * 40,
        "repository": archive.REPOSITORY,
        "image_bytes_included": False,
        "archive": {
            "url": "https://github.com/ahstn/model-gym/releases/download/evidence-v1/evidence.tar.gz",
            "size_bytes": path.stat().st_size,
            "sha256": sha(path.read_bytes()),
        },
        "members": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": sha(content),
                "restore": name != "tracks/run.py",
            }
            for name, content in FILES.items()
        ],
    }
    index_path = root / "archive-index.json"
    index_path.write_text(json.dumps(index))
    return root, path, index, tmp_path / "restored"


def rebind(bundle):
    root, path, index, _ = bundle
    index["archive"].update(size_bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    (root / "archive-index.json").write_text(json.dumps(index))


def test_default_restore_snapshot_and_idempotence(bundle):
    root, path, _, destination = bundle
    result = archive.restore(path, destination, root=root)
    assert result["restored"] == 2
    assert not (destination / "tracks/run.py").exists()
    assert archive.verify(destination, root=root)["verified"] == 2
    assert archive.restore(path, destination, root=root)["identical_skipped"] == 2
    result = archive.restore(path, destination, snapshot=True, root=root)
    assert result["restored"] == 1
    assert archive.verify(destination, snapshot=True, root=root)["verified"] == 3
    for name, content in FILES.items():
        assert (destination / name).read_bytes() == content


@pytest.mark.parametrize(
    "failure",
    [
        "archive_sha",
        "archive_size",
        "member_sha",
        "member_size",
        "missing",
        "extra",
        "duplicate",
        "traversal",
        "absolute",
        "symlink",
        "hardlink",
        "directory",
    ],
)
def test_invalid_archive_writes_nothing(bundle, failure):
    root, path, index, destination = bundle
    entries = [(name, content, tarfile.REGTYPE) for name, content in FILES.items()]
    if failure == "archive_sha":
        index["archive"]["sha256"] = "0" * 64
    elif failure == "archive_size":
        index["archive"]["size_bytes"] += 1
    elif failure == "member_sha":
        index["members"][-1]["sha256"] = "0" * 64  # Even non-restored source must validate.
    elif failure == "member_size":
        index["members"][-1]["size_bytes"] += 1
    else:
        if failure == "missing":
            entries.pop()
        elif failure == "duplicate":
            entries.append(entries[0])
        elif failure in {"symlink", "hardlink", "directory"}:
            kind = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "directory": tarfile.DIRTYPE}[
                failure
            ]
            entries[-1] = (entries[-1][0], b"", kind)
        else:
            name = {"extra": "unexpected.txt", "traversal": "../escape", "absolute": "/escape"}[failure]
            entries.append((name, b"extra", tarfile.REGTYPE))
        make_archive(path, entries)
        rebind(bundle)
    (root / "archive-index.json").write_text(json.dumps(index))
    with pytest.raises(archive.EvidenceError):
        archive.restore(path, destination, root=root)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name", ["../escape", "absolute/../escape", "/absolute", "a//b", "a/./b", "a\\b", "a/", "C:/escape"]
)
def test_invalid_index_paths_rejected(bundle, name):
    root, path, index, destination = bundle
    index["members"][0]["path"] = name
    (root / "archive-index.json").write_text(json.dumps(index))
    with pytest.raises(archive.EvidenceError):
        archive.restore(path, destination, root=root)
    assert not destination.exists()


def test_conflict_on_last_file_causes_no_partial_restore(bundle):
    root, path, _, destination = bundle
    conflict = destination / list(FILES)[1]
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"conflicting evidence")
    with pytest.raises(archive.EvidenceError):
        archive.restore(path, destination, root=root)
    assert not (destination / next(iter(FILES))).exists()
    assert conflict.read_bytes() == b"conflicting evidence"


@pytest.mark.parametrize("location", ["destination", "ancestor", "file"])
def test_symlink_destinations_rejected(bundle, location):
    root, path, _, destination = bundle
    target = destination.parent / "target"
    target.mkdir()
    if location == "destination":
        destination.symlink_to(target, target_is_directory=True)
    elif location == "ancestor":
        destination.mkdir()
        (destination / "tracks").symlink_to(target, target_is_directory=True)
    else:
        leaf = destination / next(iter(FILES))
        leaf.parent.mkdir(parents=True)
        leaf.symlink_to(target / "missing")
    with pytest.raises(archive.EvidenceError, match="Symlink"):
        archive.restore(path, destination, root=root)
    assert not list(target.iterdir())


def test_require_restored_checks_scope_and_checksums(bundle):
    root, path, _, destination = bundle
    archive.restore(path, destination, root=root)
    (destination / "archive-index.json").write_bytes((root / "archive-index.json").read_bytes())
    archive.require_restored(destination)
    corrupt = destination / list(FILES)[1]
    corrupt.write_bytes(b"original protocoX\n")
    archive.require_restored(destination, "tracks/quality_v2/")
    with pytest.raises(archive.EvidenceError, match="python evidence_archive.py restore"):
        archive.require_restored(destination)
    corrupt.unlink()
    with pytest.raises(archive.EvidenceError, match="missing or corrupt"):
        archive.require_restored(destination, "tracks/ai_detection_v2/")
    archive.require_restored(destination, "tracks/fresh_track/")


def test_require_restored_without_index_is_noop(tmp_path):
    archive.require_restored(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_exact_ledger_scope_checks_real_index_before_loading_history(bundle, monkeypatch, failure):
    import bench_runner

    root, path, _, destination = bundle
    archive.restore(path, destination, root=root)
    (destination / "archive-index.json").write_bytes((root / "archive-index.json").read_bytes())
    relative = next(iter(FILES))
    ledger = destination / relative
    archive.require_restored(destination, scope=relative)
    if failure == "missing":
        ledger.unlink()
    else:
        ledger.write_bytes(b"X" + ledger.read_bytes()[1:])
    monkeypatch.setattr(bench_runner, "ROOT", destination)
    with pytest.raises(archive.EvidenceError, match="python evidence_archive.py restore"):
        bench_runner.load_ledger(ledger)


def test_duplicate_index_and_parent_file_overlap_rejected(bundle):
    root, _, index, _ = bundle
    index["members"].append(dict(index["members"][0]))
    (root / "archive-index.json").write_text(json.dumps(index))
    with pytest.raises(archive.EvidenceError, match="duplicate"):
        archive.read_index(root)
    index["members"][-1]["path"] = "tracks"
    (root / "archive-index.json").write_text(json.dumps(index))
    with pytest.raises(archive.EvidenceError, match="overlap"):
        archive.read_index(root)


def test_corrupt_member_bytes_rejected_even_when_archive_checksum_matches(bundle):
    root, path, _, destination = bundle
    entries = [(name, content, tarfile.REGTYPE) for name, content in FILES.items()]
    name, content, kind = entries[-1]
    entries[-1] = (name, b"X" + content[1:], kind)
    make_archive(path, entries)
    rebind(bundle)
    with pytest.raises(archive.EvidenceError, match="member checksum"):
        archive.restore(path, destination, root=root)
    assert not destination.exists()


def test_dangling_index_symlink_cannot_disable_required_evidence(tmp_path):
    (tmp_path / "archive-index.json").symlink_to(tmp_path / "missing-index.json")
    with pytest.raises(archive.EvidenceError, match="python evidence_archive.py restore"):
        archive.require_restored(tmp_path)


def test_download_redirect_cannot_downgrade_to_http():
    handler = archive._HTTPSRedirect()
    with pytest.raises(archive.EvidenceError, match="HTTPS"):
        handler.redirect_request(None, None, 302, "redirect", {}, "http://example.com/archive")
