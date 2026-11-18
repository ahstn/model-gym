"""Prepare a portable labeled-dataset snapshot without copying or re-encoding scored views."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def finish_snapshot(output, objects, summary, labels):
    """Produce every root object required by restore, then seal the upload plan."""
    compressed = output / "labelled-images.jsonl.zst"
    with compressed.open("wb") as destination:
        subprocess.run(["zstd", "-T4", "-6", "-q", "-c", str(labels)], stdout=destination, check=True)
    shutil.copyfile(Path(__file__).with_name("restore.py"), output / "restore.py")
    write_json(
        output / "snapshot-summary.json",
        {k: v for k, v in summary.items() if k not in ("object_count", "total_object_bytes")},
    )
    payload = list(objects)
    for name in ("labelled-images.jsonl.zst", "restore.py", "snapshot-summary.json"):
        path = output / name
        payload.append(
            {"local_path": str(path), "key": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    snapshot = {
        **summary,
        "schema_version": "r2-snapshot-v2",
        "objects": [{k: row[k] for k in ("key", "bytes", "sha256")} for row in payload],
    }
    # Counts include the payload, excluding the self-referential manifest and
    # the completion receipt that can only be written after remote verification.
    snapshot["object_count"] = len(payload)
    snapshot["total_object_bytes"] = sum(row["bytes"] for row in payload)
    path = output / "snapshot.json"
    write_json(path, snapshot)
    payload.append(
        {"local_path": str(path), "key": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
    )
    write_json(output / "upload-plan.json", payload)
    return snapshot


INITIAL = "labeling-10000-2026-11-01"
PREVIOUS = "labeling-20000-2026-11-16"
LATEST = "labeling-40000-2026-11-17"
RUN_SIZES = {PREVIOUS: 20_000, LATEST: 40_000}


def index_images(data, current, output):
    expected = RUN_SIZES[current.name]
    cases = json.loads((current / "combined-manifest.json").read_text())["cases"]
    if len(cases) != expected or len({c["case_id"] for c in cases}) != expected:
        raise ValueError("Unexpected combined manifest count or duplicate case IDs")

    def image_entry(case):
        path = Path(case["image_path"])
        relative = path.resolve(strict=True).relative_to(data)
        if path != data / relative or path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected image path: {path}")
        sha = digest(path)
        size = path.stat().st_size
        if sha != case["view_sha256"] or size != case["delivered_bytes"]:
            raise ValueError(f"Changed view: {case['case_id']}")
        return {
            "case_id": case["case_id"],
            "local_path": str(path),
            "key": "data/" + relative.as_posix(),
            "sha256": sha,
            "bytes": size,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(image_entry, cases))
    if len({x["key"] for x in images}) != expected:
        raise ValueError("Duplicate image object keys")
    # Write atomically so a concurrently prepared image-only transfer sees a full index.
    staging = output / "image-index.json.tmp"
    write_json(staging, images)
    staging.replace(output / "image-index.json")
    return images


def prepare(data, repo, output, run=PREVIOUS):
    output.mkdir(parents=True, exist_ok=True)
    current = data / run
    initial = data / INITIAL
    images = index_images(data, current, output)

    # Explicit data allowlist: do not recursively archive operations, credentials,
    # downloaded candidates, caches, obsolete exports or unscored originals.
    selected = set()
    generations = [initial, data / PREVIOUS] + ([current] if run == LATEST else [])
    for root in generations:
        for name in (
            "manifest.json",
            "combined-manifest.json",
            "stats.jsonl",
            "stats.provenance.json",
            "corpus-embeddings.npz",
            "cluster-prototypes.npz",
            "REPORT.md",
            "benchmark-manifest.json",
            "benchmark-confirmation-manifest.json",
            "benchmark-examples-manifest.json",
            "embedding-operations/provenance.json",
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
        for name in (
            "results",
            "references",
            "review",
            "executed-sources",
            "execution-sources",
            "compatibility-probe",
        ):
            selected.update(
                p
                for p in (root / name).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            )
    # Keep every published export generation, including the byte-identical prefixes.
    for root in generations:
        for name in (
            "labelled-images.jsonl",
            "scores-long.jsonl",
            "scores-long.csv",
            "summary.json",
            "new-labelled-images.jsonl",
            "new-scores-long.jsonl",
        ):
            path = root / "exports" / name
            if path.is_file():
                selected.add(path)
    if run == LATEST:
        # Operational directories also contain machine credentials. Only these named
        # finalized evidence files belong in the scored-data backup.
        selected.update(
            data / name
            for name in (
                "sources/megalith/metadata/README.md",
                "sources/vizwiz/evidence/provenance.json",
                "sources/vizwiz/evidence/caption-media-license.html",
            )
        )
        selected.update(
            current / "acquisition" / name
            for name in ("megalith-source/metadata/README.md", "megalith-candidates.provenance.json")
        )
        selected.update(
            current / "operations" / name
            for name in (
                "completion.json",
                "final-code-checks.json",
                "finalization-and-publication-checks.json",
                "selected-corpus-validation.json",
                "related-validation.json",
                "stats-audit-final.json",
                "final-recovery-verification.json",
                "jev-dimension-comparison.json",
                "deepseek-empty-recovery-root-review.json",
                "deepseek-empty-recovery-source-archive.json",
                "deepseek-empty-recovery-plan-review.json",
                "deepseek-empty-recovery-freeze.json",
                "deepseek-empty-recovery-check.json",
                "recovery-freeze-preconditions.json",
            )
        )
        selected.update(
            current / "embedding-operations" / name
            for name in (
                "final-candidate-verification.json",
                "cleanup-status.json",
                "billing-receipt.json",
                "deletion-verification.json",
                "no-network-volumes.json",
                "cpu-overlap.json",
                "cpu-gpu-overlap.json",
            )
        )
    pinned = {}
    legacy_paths = {}
    rehearsal = "rehearsal-4000-2026-11-01"
    legacy_dir = data / rehearsal / "published"

    def pin(path, expected):
        if expected and pinned.setdefault(path, expected) != expected:
            raise ValueError(f"Conflicting historical evidence pins for {path}")

    # Follow request and response paths in the final labels, including reused Lite evidence.
    def evidence(value):
        if isinstance(value, dict):
            # Some frozen rows name a historical checkout instead of the data root.
            # Only the two source exports and the referenced timing receipts moved.
            for key in ("source", "review_timing_path"):
                original = value.get(key)
                if isinstance(original, str) and original.startswith("/tmp/"):
                    marker = f"/visual-quality/experiments/{rehearsal}/"
                    if marker in original:
                        suffix = Path(original.split(marker, 1)[1])
                        if suffix in (Path("references.jsonl"), Path("scores-long.jsonl")) or (
                            key == "review_timing_path"
                            and suffix.suffix == ".json"
                            and suffix.parts[:2] in (("references", "timings"), ("references", "revisions"))
                        ):
                            path = legacy_dir / suffix
                            if not path.resolve().is_relative_to(legacy_dir):
                                raise ValueError(f"Unexpected historical evidence path: {original}")
                            selected.add(path)
                            legacy_paths[original] = "data/" + path.relative_to(data).as_posix()
                            if key == "review_timing_path":
                                pin(path, value["review_timing_sha256"])
            for key, hash_field in (
                ("raw_response_path", "raw_response_sha256"),
                ("request_path", "request_record_sha256"),
            ):
                if item := value.get(key):
                    path = Path(item)
                    path.relative_to(data)
                    selected.add(path)
                    pin(path, value.get(hash_field))
            for item in value.values():
                evidence(item)
        elif isinstance(value, list):
            for item in value:
                evidence(item)

    summary = json.loads((current / "exports/summary.json").read_text())
    audit = json.loads((current / "audit.json").read_text())
    labels = current / "exports/labelled-images.jsonl"
    if not audit["passed"] or summary["exports"][labels.name] != audit["exports_sha256"]:
        raise ValueError("Missing successful audit of canonical labels")
    for name, sha in summary["exports"].items():
        path = current / "exports" / name
        if path not in selected:
            raise ValueError(f"Unexpected export in summary: {name}")
        pin(path, sha)
    verified_images = {entry["case_id"]: entry for entry in images}
    ensemble_pairs = 0
    label_count = 0
    with (current / "exports/labelled-images.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            image = verified_images.pop(record["case_id"])
            if record["image_path"] != image["local_path"] or record["view_sha256"] != image["sha256"]:
                raise ValueError(f"Manifest and labels disagree for {record['case_id']}")
            evidence(record["assessments"])
            ensemble = record["assessments"]["photo_ensemble"]
            ensemble_pairs += all(ensemble[k] is not None for k in ("technical_score", "aesthetic_score"))
            label_count += 1
    if (
        verified_images
        or label_count != RUN_SIZES[run]
        or ensemble_pairs != summary["coverage"]["photo_ensemble"]["numeric_pairs"]
    ):
        raise ValueError("Image/label counts or numerical ensemble coverage do not match the audited corpus")
    if run == LATEST:
        for key, sha in json.loads((current / "initial-evidence-pins.json").read_text()).items():
            path = Path(key)
            if path.is_relative_to(data):
                if path not in selected:
                    raise ValueError(f"Unselected historical dataset pin: {path}")
            elif "visual-quality/labeling/records/" in path.as_posix():
                path = repo / "visual-quality/labeling/records" / path.name
            else:
                raise ValueError(f"Unexpected publication pin: {path}")
            pin(path, sha)
    files = {("data/" + p.relative_to(data).as_posix()): p for p in selected}
    source_roots = (
        "visual-quality/src",
        "visual-quality/prompts",
        "visual-quality/labeling/prompts",
        "visual-quality/labeling/runtime",
        "visual-quality/labeling/runs/expansion-20000-2026-11-16",
        "visual-quality/labeling/backups",
        "visual-quality/labeling/records",
        *(("visual-quality/labeling/runs/expansion-40000-2026-11-17",) if run == LATEST else ()),
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
        sha = digest(path)
        if path in pinned and pinned.pop(path) != sha:
            raise ValueError(f"Historical or raw evidence checksum mismatch: {name}")
        inventory.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
    if pinned:
        raise ValueError(f"Missing {len(pinned)} pinned evidence files")
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
        "complete_ensemble_pairs": ensemble_pairs,
        "legacy_path_aliases": legacy_paths,
        "canonical_labels_key": "data/" + labels.relative_to(data).as_posix(),
        "canonical_labels_sha256": audit["exports_sha256"],
    }
    snapshot = finish_snapshot(output, objects, summary, current / "exports/labelled-images.jsonl")
    print(
        json.dumps(
            {k: v for k, v in snapshot.items() if k not in ("objects", "legacy_path_aliases")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", choices=RUN_SIZES, default=PREVIOUS)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.repo.resolve(), args.output.resolve(), args.run)
