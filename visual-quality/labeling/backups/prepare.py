"""Prepare a portable 20K snapshot without copying or re-encoding scored views."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(data, repo, output):
    output.mkdir(parents=True, exist_ok=True)
    current = data / "labeling-20000-2026-11-16"
    initial = data / "labeling-10000-2026-11-01"
    cases = json.loads((current / "combined-manifest.json").read_text())["cases"]
    assert len(cases) == 20000 and len({c["case_id"] for c in cases}) == 20000

    def image_entry(case):
        path = Path(case["image_path"])
        sha = digest(path)
        if sha != case["view_sha256"]:
            raise ValueError(f"Changed view: {case['case_id']}")
        relative = "data/" + path.relative_to(data).as_posix()
        return {
            "case_id": case["case_id"],
            "local_path": str(path),
            "key": relative,
            "sha256": sha,
            "bytes": path.stat().st_size,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(image_entry, cases))
    assert len({x["key"] for x in images}) == 20000
    write_json(output / "image-index.json", images)

    # Explicit data allowlist: do not recursively archive operations, credentials,
    # downloaded candidates, caches, obsolete exports or unscored originals.
    selected = set()
    for root in (current, initial):
        for name in (
            "manifest.json",
            "combined-manifest.json",
            "stats.jsonl",
            "stats.provenance.json",
            "corpus-embeddings.npz",
            "related-images.json",
            "possible-duplicates.json",
            "content-coverage.json",
            "corpus-audit.json",
            "audit.json",
            "report.md",
            "api-provider-snapshot.json",
            "initial-evidence-pins.json",
            "references-all.jsonl",
            "references-freeze.json",
            "final-independent-audit.json",
            "corpus-integrity-audit.json",
        ):
            if (root / name).is_file():
                selected.add(root / name)
        for name in ("results", "references", "review", "executed-sources", "execution-sources"):
            selected.update(
                p
                for p in (root / name).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            )
    for name in ("labelled-images.jsonl", "scores-long.jsonl", "scores-long.csv", "summary.json"):
        selected.add(current / "exports" / name)

    # Include raw evidence referenced outside the two runs (e.g. reused Lite).
    missing_raw = []

    def evidence(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "raw_response_path" and item:
                    path = Path(item)
                    if path.is_file():
                        path.relative_to(data)  # Fail on unexpected external paths.
                        selected.add(path)
                    else:
                        missing_raw.append(item)
                evidence(item)
        elif isinstance(value, list):
            for item in value:
                evidence(item)

    label_count = 0
    with (current / "exports/labelled-images.jsonl").open() as stream:
        for line in stream:
            evidence(json.loads(line))
            label_count += 1
    assert label_count == 20000
    if missing_raw:
        raise ValueError(f"{len(missing_raw)} referenced raw responses missing")

    files = {("data/" + p.relative_to(data).as_posix()): p for p in selected}
    source_roots = (
        "visual-quality/src",
        "visual-quality/prompts",
        "visual-quality/labeling/prompts",
        "visual-quality/labeling/runtime",
        "visual-quality/labeling/runs/expansion-20000-2026-11-16",
        "visual-quality/labeling/backups",
        "image-stats/src",
    )
    for name in source_roots:
        files.update(
            {
                ("code/" + p.relative_to(repo).as_posix()): p
                for p in (repo / name).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
        )
    for name in (
        "pyproject.toml",
        "uv.lock",
        "visual-quality/pyproject.toml",
        "visual-quality/AGENTS.md",
        "image-stats/pyproject.toml",
        "visual-quality/labeling/routes.json",
        "visual-quality/labeling/ensemble.json",
        "visual-quality/labeling/profile-provenance.json",
    ):
        files["code/" + name] = repo / name
    forbidden = ("PRIVATE KEY-----", '"authorization": "Bearer ', '"api_key": "sk-')
    inventory = []
    for name, path in sorted(files.items()):
        if path.is_symlink() or path.name.startswith(".env") or path.suffix in (".pem", ".key"):
            raise ValueError(f"Unexpected credential or symlink: {name}")
        if path.suffix in (".json", ".jsonl", ".py", ".md", ".toml", ""):
            with path.open(errors="replace") as stream:
                for line in stream:
                    # This script contains the patterns, not credentials.
                    if (
                        any(marker in line for marker in forbidden)
                        and path.resolve() != Path(__file__).resolve()
                    ):
                        raise ValueError(f"Possible credential in {name}")
        inventory.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(output / "metadata-inventory.json", inventory)
    files["image-index.json"] = output / "image-index.json"
    files["metadata-inventory.json"] = output / "metadata-inventory.json"
    archive = output / "metadata.tar.zst"
    with archive.open("wb") as destination:
        process = subprocess.Popen(["zstd", "-T4", "-6", "-q"], stdin=subprocess.PIPE, stdout=destination)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as tar:
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("zstd compression failed")
    objects = images + [
        {
            "local_path": str(archive),
            "key": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
        }
    ]
    for name in ("image-index.json", "metadata-inventory.json"):
        path = output / name
        objects.append(
            {"local_path": str(path), "key": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    write_json(output / "upload-plan.json", objects)
    summary = {
        "image_count": len(images),
        "image_bytes": sum(x["bytes"] for x in images),
        "labels": label_count,
        "metadata_file_count": len(inventory),
        "metadata_uncompressed_bytes": sum(x["bytes"] for x in inventory),
        "archive_bytes": archive.stat().st_size,
        "total_object_bytes": sum(x["bytes"] for x in objects),
        "object_count": len(objects),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_data_root": str(data),
        "excluded": "Unscored originals/candidates, caches, credentials, temporary operations, redundant exports",
        "complete_ensemble_pairs": 19997,
    }
    write_json(output / "snapshot-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.repo.resolve(), args.output.resolve())
