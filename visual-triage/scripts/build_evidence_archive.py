"""Build a deterministic, indexed evidence snapshot from a pinned Git commit."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/ahstn/model-gym"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def build(source_commit, restore_paths, archive_path, index_path, release_tag):
    commit = git("rev-parse", "--verify", source_commit + "^{commit}").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", "-z", commit, "--", ".").decode().split("\0")
    paths = sorted(path for path in paths if path)
    if not restore_paths <= set(paths):
        raise ValueError("Restore list contains paths absent from the pinned snapshot")
    if archive_path.exists() or index_path.exists():
        raise ValueError("Archive and index are immutable; use fresh output paths")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path in paths:
            content = git("show", f"{commit}:visual-triage/{path}")
            member = tarfile.TarInfo(path)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
            members.append(
                {
                    "path": path,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "restore": path in restore_paths,
                }
            )
    result = {
        "version": 1,
        "archive_id": release_tag,
        "source_commit": commit,
        "repository": REPOSITORY,
        "archive": {
            "url": f"{REPOSITORY}/releases/download/{release_tag}/{archive_path.name}",
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "size_bytes": archive_path.stat().st_size,
        },
        "members": members,
        "image_bytes_included": False,
    }
    with index_path.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--restore-paths", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    result = build(
        args.source_commit,
        set(json.loads(args.restore_paths.read_text())),
        args.archive,
        args.index,
        args.release_tag,
    )
    print(json.dumps({**result["archive"], "members": len(result["members"])}))


if __name__ == "__main__":
    main()
