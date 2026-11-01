"""Archive final stopped-run receipts deterministically, without image views or secrets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01")
TOKEN_FILE = Path("/tmp/qrealign-lite-token")
CREDENTIAL_NAMES = (
    "OPENROUTER_API_KEY",
    "RUN_INFRA_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "RUNPOD_API_KEY",
)
PATTERNS = {
    "sk-or-v1": rb"(?<![A-Za-z0-9_-])sk-or-v1-[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])",
    "hf": rb"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{12,}(?![A-Za-z0-9_-])",
    "rpa": rb"(?<![A-Za-z0-9_-])rpa_[A-Za-z0-9_-]{12,}(?![A-Za-z0-9_-])",
}
MAX_FILE_BYTES = 100_000_000
MAX_ARCHIVE_BYTES = 95 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def credentials() -> dict[str, bytes]:
    values = {key: os.environ[key].encode() for key in CREDENTIAL_NAMES if os.environ.get(key)}
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_bytes().strip()
        if token:
            values["qrealign-lite-token"] = token
    return values


def scan(path: str, content: bytes, secrets: dict[str, bytes]) -> None:
    # Exact known values are checked everywhere. Pattern-only checks require a
    # token boundary so random substrings inside encrypted metadata are not keys.
    for label, value in secrets.items():
        if value and value in content:
            raise ValueError(f"Credential detected in {path}: {label}")
    for label, pattern in PATTERNS.items():
        if re.search(pattern, content):
            raise ValueError(f"Credential detected in {path}: pattern:{label}")


def transient(relative: Path) -> bool:
    return any(
        part in {"tmp", "temp", "__pycache__"}
        or part.endswith((".lock", ".pid", ".tmp", ".temp", "~"))
        or part.startswith((".tmp-", ".tmp."))
        for part in relative.parts
    )


def snapshot(root: Path, secrets: dict[str, bytes]) -> dict:
    if root.is_symlink() or root.absolute() != root.resolve() or not root.is_dir():
        raise ValueError(f"Source must be an existing directory without symlink components: {root}")
    files, excluded = [], []
    for directory, directories, names in os.walk(root, followlinks=False):
        directories.sort()
        for name in sorted([*directories, *names]):
            path = Path(directory) / name
            relative = path.relative_to(root)
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Symlink or path outside source root: {relative}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"Non-regular source file: {relative}")
            if transient(relative):
                excluded.append(relative.as_posix())
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"Unexpected file exceeds 100 MB; review before packaging: {relative}")
            content = path.read_bytes()
            if len(content) > MAX_FILE_BYTES:
                raise ValueError(f"File grew beyond 100 MB: {relative}")
            scan(relative.as_posix(), content, secrets)
            files.append({"path": relative.as_posix(), "bytes": len(content), "sha256": sha(content)})
    return {
        "source_path": str(root),
        "files": sorted(files, key=lambda row: row["path"]),
        "excluded_transients": sorted(excluded),
    }


def archive(source: dict, destination: Path, secrets: dict[str, bytes]) -> None:
    root = Path(source["source_path"])
    with (
        destination.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for row in source["files"]:
            path = root / row["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Symlink or path outside source root: {row['path']}")
            content = path.read_bytes()
            scan(row["path"], content, secrets)
            if len(content) != row["bytes"] or sha(content) != row["sha256"]:
                raise ValueError(f"Source changed during archive: {row['path']}")
            info = tarfile.TarInfo(row["path"])
            info.size, info.mode, info.mtime, info.uid, info.gid = len(content), 0o644, 0, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(content))
    if destination.stat().st_size >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"Archive reaches 95 MiB; split by phase before packaging: {destination.name}")


def validate_existing(output: Path, expected: dict) -> None:
    if output.is_symlink() or (output / "index.json").is_symlink() or not (output / "index.json").is_file():
        raise ValueError("Refusing to overwrite an existing destination without its valid index")
    actual = json.loads((output / "index.json").read_text())
    if actual != expected or {p.name for p in output.iterdir()} != {
        "index.json",
        *(a["archive"] for a in expected["archives"]),
    }:
        raise ValueError(
            "Existing archive index differs from final source hashes or contents; refusing overwrite"
        )
    for item in actual["archives"]:
        path = output / item["archive"]
        if (
            path.is_symlink()
            or path.stat().st_size != item["archive_bytes"]
            or sha(path.read_bytes()) != item["archive_sha256"]
        ):
            raise ValueError(f"Existing archive does not match its index: {item['archive']}")


def package(
    sources: dict[str, Path], output: Path, *, all_runs_stopped: bool, secrets: dict[str, bytes]
) -> dict:
    if not all_runs_stopped:
        raise ValueError("Package only after all API/Lite drivers and services have stopped")
    if not sources or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) for name in sources):
        raise ValueError("Archive names must be safe simple names")
    sources = {name: path.absolute() for name, path in sorted(sources.items())}
    if any(output.resolve().is_relative_to(root.resolve()) for root in sources.values()):
        raise ValueError("Archive destination must be outside every source root")
    before = {name: snapshot(root, secrets) for name, root in sources.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".execution-records-", dir=output.parent) as temporary:
        stage = Path(temporary) / "complete"
        stage.mkdir()
        records = []
        for name, source in before.items():
            path = stage / f"{name}.tar.gz"
            archive(source, path, secrets)
            records.append(
                {
                    "name": name,
                    **source,
                    "archive": path.name,
                    "archive_bytes": path.stat().st_size,
                    "archive_sha256": sha(path.read_bytes()),
                }
            )
        if before != {name: snapshot(root, secrets) for name, root in sources.items()}:
            raise ValueError("Source snapshot changed during packaging; no final archives published")
        index = {
            "schema_version": 1,
            "status": "final_artifact_archive-not-inference",
            "all_runs_stopped_asserted": True,
            "determinism": "Sorted relative file paths; tar/gzip mtime=0, uid/gid=0, empty owner names, regular mode=0644; source file bytes and embedded JSON timestamps unchanged",
            "credentials_scan_status": {
                "status": "passed",
                "exact_value_labels_checked": sorted(key for key, value in secrets.items() if value),
                "pattern_labels_checked": sorted(PATTERNS),
                "files_scanned": sum(len(s["files"]) for s in before.values()),
            },
            "archives": records,
        }
        if output.exists():
            validate_existing(output, index)
        else:
            (stage / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
            stage.rename(output)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=HERE / "execution-records")
    parser.add_argument(
        "--all-runs-stopped",
        action="store_true",
        help="Assert every API/Lite driver and service has stopped; source snapshots are also checked",
    )
    args = parser.parse_args()
    sources = {name: args.data / "results/api" / name for name in ("deepseek", "luna", "jev", "liquid")}
    sources.update(
        lite=args.data / "results/lite",
        **{
            "lite-operations": args.data / "lite-operations",
            "api-postpass": args.data / "results/api/postpass",
        },
    )
    index = package(sources, args.output, all_runs_stopped=args.all_runs_stopped, secrets=credentials())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": index["status"],
                "archives": len(index["archives"]),
                "total_archive_bytes": sum(a["archive_bytes"] for a in index["archives"]),
            }
        )
    )


if __name__ == "__main__":
    main()
