"""Deterministic source-family allocation and complete pilot manifests."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl
from visual_quality.validation import deduplicate, family_groups, inspect_records, verify_manifest

SOURCES = ("megalith", "vizwiz", "plotqa", "clevr", "owned_controls")


def stable_key(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def source_counts(config: dict) -> dict[str, int]:
    return {
        source: sum(counts[source] for counts in config["splits"].values()) + config["spares"].get(source, 0)
        for source in SOURCES
    }


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config["variants_per_parent"] != 3:
        raise ValueError("This generator version implements exactly three variants per parent")
    if config["distortion_parents"] > config["splits"]["train"]["megalith"]:
        raise ValueError("Distortion parents must be within the broad-photo training quota")
    if set(config["splits"]) != {"train", "development", "test"}:
        raise ValueError("Expected separate train, development, and test quotas")
    if not config["name"] or Path(config["name"]).name != config["name"]:
        raise ValueError("The pilot name must be one directory name")
    for counts in config["splits"].values():
        if set(counts) != set(SOURCES) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("Each source needs an explicit nonnegative integer quota")
    return config


def acquire(root: Path, config: dict, sources: list[str] | None = None) -> dict[str, int]:
    from visual_quality import controls, generate, megalith, vizwiz

    adapters = {
        "megalith": megalith.acquire,
        "vizwiz": vizwiz.acquire,
        "plotqa": controls.acquire_plotqa,
        "clevr": controls.acquire_clevr,
        "owned_controls": generate.make_owned_controls,
    }
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    acquisition = root / "acquisition"
    identity = {"seed": config["seed"], "counts": source_counts(config)}
    lock = acquisition / "request.json"
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise ValueError("Acquisition root belongs to a different seed or quota plan; use a new root")
    atomic_json(lock, identity)
    counts = {}
    for source in sources or list(SOURCES):
        count = identity["counts"][source]
        print(f"Acquiring {source}: {count} source images", flush=True)
        records = adapters[source](root, count, config["seed"])
        if len(records) < count:
            raise ValueError(f"{source} returned {len(records)} of {count} required candidates")
        for record in records:
            record["component"] = source
        write_jsonl(acquisition / f"{source}.jsonl", records)
        counts[source] = len(records)
        print(f"Acquired {source}: {len(records)} cached source images", flush=True)
    return counts


def assign_splits(records: list[dict], groups: dict[str, str], config: dict) -> list[dict]:
    families = defaultdict(list)
    for record in records:
        families[groups[record["id"]]].append(record)
    # Larger groups first; singletons can then fill exact quotas without splitting families.
    ordered = sorted(families, key=lambda key: (-len(families[key]), stable_key(config["seed"], key)))
    used, selected = set(), []
    owned_kinds = sorted({row["content_kind"] for row in records if row["component"] == "owned_controls"})
    for split in ("test", "development", "train"):
        remaining = Counter(config["splits"][split])
        owned_quota = remaining["owned_controls"]
        if owned_kinds and owned_quota % len(owned_kinds):
            raise ValueError("Owned-control quota must divide evenly across its content kinds")
        owned_remaining = Counter({kind: owned_quota // len(owned_kinds) for kind in owned_kinds})
        for identity in ordered:
            if identity in used:
                continue
            rows = families[identity]
            contribution = Counter(row["component"] for row in rows)
            owned_contribution = Counter(
                row["content_kind"] for row in rows if row["component"] == "owned_controls"
            )
            if any(count > remaining[source] for source, count in contribution.items()):
                continue
            if any(count > owned_remaining[kind] for kind, count in owned_contribution.items()):
                continue
            for record in rows:
                selected.append(dict(record, split=split, split_group=identity))
            used.add(identity)
            remaining.subtract(contribution)
            owned_remaining.subtract(owned_contribution)
        shortfalls = {key: value for key, value in remaining.items() if value}
        if shortfalls:
            raise ValueError(f"Cannot fill {split} without breaking source families: {shortfalls}")
    return selected


def _final_record(record: dict, root: Path) -> dict:
    item = dict(record)
    item.pop("cache_mtime_ns", None)
    item["license"] = dict(item["license"])
    evidence = item["license"].get("evidence_path")
    if evidence and Path(evidence).is_absolute() and Path(evidence).is_relative_to(root):
        item["license"]["evidence_path"] = Path(evidence).relative_to(root).as_posix()
    item.update(
        technical_quality=None,
        aesthetics=None,
        rating_status="not_requested",
        task_masks={
            "technical_quality": False,
            "aesthetics": False,
            "photo_domain": item.get("photo_domain_eligible") is not None,
        },
    )
    return item


def assemble(root: Path, config: dict, exclusion_paths: list[Path]) -> dict:
    from visual_quality.generate import make_variants

    root = root.resolve()
    destination = root / config["name"]
    destination.mkdir(parents=True, exist_ok=True)
    for path in exclusion_paths:
        if not path.is_file():
            raise ValueError(f"Evaluation exclusion manifest not found: {path}")
    config_path = destination / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("This output directory already has a different frozen configuration")
    atomic_json(config_path, config)
    atomic_json(destination / "report.json", {"complete": False, "status": "assembling"})
    records = []
    for source in SOURCES:
        path = root / "acquisition" / f"{source}.jsonl"
        if not path.exists():
            raise ValueError(f"Source not acquired: {source}; run acquire first")
        records.extend(read_jsonl(path))
    print(f"Validating {len(records)} acquired images", flush=True)
    records, failures = inspect_records(records, root)
    write_jsonl(destination / "invalid-images.jsonl", failures)
    exclusions = [row for path in exclusion_paths for row in read_jsonl(path)]
    records, removed = deduplicate(records, exclusions)
    write_jsonl(destination / "excluded-images.jsonl", removed)
    groups, near_matches = family_groups(records, config["near_duplicate_hamming_distance"])
    write_jsonl(destination / "near-duplicate-candidates.jsonl", near_matches)
    records = assign_splits(records, groups, config)
    parents = sorted(
        (row for row in records if row["split"] == "train" and row["component"] == "megalith"),
        key=lambda row: stable_key(config["seed"], "distortion:" + row["id"]),
    )[: config["distortion_parents"]]
    # Generator reads absolute file paths, while public manifests use paths relative to data root.
    parent_inputs = [dict(row, image_path=str(root / row["image_path"])) for row in parents]
    print(f"Generating three variants for {len(parents)} training parents", flush=True)
    variants = make_variants(root, parent_inputs, config["seed"])
    parent_by_id = {row["id"]: row for row in parents}
    for row in variants:
        parent = parent_by_id[row["parent_id"]]
        row.update(component="distortions", split="train", split_group=parent["split_group"])
    variants, variant_failures = inspect_records(variants, root)
    if variant_failures or len(variants) != 3 * len(parents):
        atomic_json(destination / "variant-failures.json", variant_failures)
        raise ValueError("Generated variants failed image validation")
    records = [_final_record(row, root) for row in records + variants]
    records.sort(key=lambda row: (row["split"], row["component"], row["id"]))
    report = verify_manifest(records, root, check_bytes=False)
    report.update(
        complete=not report["errors"],
        quality_labels_generated=False,
        training_ready=False,
        counts_by_component=dict(Counter(row["component"] for row in records)),
        decoded_unique_images=len({row["decoded_sha256"] for row in records}),
        duplicate_variant_pixels=sum(
            row["decoded_sha256"] in {p["decoded_sha256"] for p in parents} for row in variants
        ),
        invalid_acquired_images=len(failures),
        excluded_exact_matches=len(removed),
        near_duplicate_pairs=len(near_matches),
        external_exclusions=[{"path": str(path), "sha256": file_sha256(path)} for path in exclusion_paths],
        pending=[
            "Independent photo eligibility and quality-band audit",
            "Human development audit and teacher scoring/calibration",
            "Full external benchmark overlap audit beyond supplied exclusions and official train boundaries",
        ],
    )
    if report["errors"]:
        atomic_json(destination / "report.json", report)
        raise ValueError(f"Manifest validation failed: {report['errors'][:3]}")
    manifest_hashes = {}
    for split in config["splits"]:
        path = destination / f"{split}.jsonl"
        write_jsonl(path, (row for row in records if row["split"] == split))
        manifest_hashes[split] = file_sha256(path)
    report["manifest_sha256"] = manifest_hashes
    package = Path(__file__).parent
    project = package.parent.parent
    atomic_json(
        destination / "software.json",
        {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("pillow", "numpy", "requests", "pyarrow", "remotezip")
            },
            "module_sha256": {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))},
            "uv_lock_sha256": file_sha256(project / "uv.lock"),
        },
    )
    atomic_json(
        destination / "dataset.json",
        {
            "schema_version": 1,
            "name": config["name"],
            "image_root": "..",
            "seed": config["seed"],
            "manifests": {split: f"{split}.jsonl" for split in config["splits"]},
            "path_policy": "image_path is relative to image_root resolved from this file's directory",
            "software": "software.json",
            "label_policy": "Human source labels are preserved separately; no teacher or inferred MOS labels",
            "source_revisions": {
                source: sorted({str(row["source_revision"]) for row in records if row["component"] == source})
                for source in SOURCES
            },
        },
    )
    report["metadata_sha256"] = {
        name: file_sha256(destination / name) for name in ("config.json", "dataset.json", "software.json")
    }
    atomic_json(destination / "report.json", report)
    return report
