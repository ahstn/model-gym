"""Validate bounded, structured labeling exports before they enter version control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ALLOWED_FILES = frozenset(
    {
        "manifest.json",
        "references.jsonl",
        "scores.csv",
        "scores.jsonl",
        "labelled-images.jsonl",
        "publication-case-ids.json",
        "provenance.json",
    }
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check_nested_ids(value, allowed: set[str], context: str) -> None:
    if isinstance(value, dict):
        if "case_id" in value and value["case_id"] not in allowed:
            raise ValueError(f"Non-public image in {context}: {value['case_id']}")
        for item in value.values():
            _check_nested_ids(item, allowed, context)
    elif isinstance(value, list):
        for item in value:
            _check_nested_ids(item, allowed, context)


def verify_publication(directory: Path, *, check_images: bool = False) -> dict:
    """Check all export paths, nested records, hashes, joins and numeric outcomes.

    Only the listed structured formats are permitted. Extra HTML, images, logs,
    archives or symlinks are rejected rather than allowed to hide other labels.
    This checks a publication directory, not historical archives elsewhere.
    """
    directory = Path(directory)
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Publication export cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != ALLOWED_FILES:
        raise ValueError(f"Unexpected or missing publication files: {sorted(actual ^ ALLOWED_FILES)}")
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["records"].items():
        if name not in ALLOWED_FILES or name == "provenance.json" or sha(directory / name) != expected:
            raise ValueError(f"Publication record hash mismatch: {name}")
    if set(provenance["records"]) != ALLOWED_FILES - {"provenance.json"}:
        raise ValueError("Publication provenance must hash every data file")
    ids = json.loads((directory / "publication-case-ids.json").read_text())
    if not isinstance(ids, list) or not 1 <= len(ids) <= 1000 or len(ids) != len(set(ids)):
        raise ValueError("Publication requires 1–1,000 distinct image IDs")
    allowed = set(ids)
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or set(cases) != allowed:
        raise ValueError("Publication manifest and allowlist differ")
    ordered = sorted(cases.values(), key=lambda case: case["benchmark_order"])
    if [c["benchmark_order"] for c in ordered] != list(range(1, len(ids) + 1)):
        raise ValueError("Published images must be the first explicit benchmark orders")
    if [c["case_id"] for c in ordered] != ids:
        raise ValueError("Publication allowlist is not in benchmark order")
    if check_images:
        for case in cases.values():
            if sha(Path(case["image_path"])) != case["view_sha256"]:
                raise ValueError("Delivered image changed")
    for name in sorted(ALLOWED_FILES):
        if name.endswith(".json"):
            _check_nested_ids(json.loads((directory / name).read_text()), allowed, name)
        elif name.endswith(".jsonl"):
            _check_nested_ids(jsonlines(directory / name), allowed, name)
        else:
            with (directory / name).open(newline="") as stream:
                for row in csv.DictReader(stream):
                    _check_nested_ids(row, allowed, name)
                    for value in row.values():
                        if value and value.startswith(("{", "[")):
                            _check_nested_ids(json.loads(value), allowed, name)
    rows = jsonlines(directory / "scores.jsonl")
    seen, counts = set(), Counter()
    for row in rows:
        key = row["case_id"], row["model"], row["arm"]
        if key in seen or row["view_sha256"] != cases[row["case_id"]]["view_sha256"]:
            raise ValueError("Duplicate observation or mismatched delivered view")
        seen.add(key)
        counts[row["model"]] += 1
        for task in ("technical_score", "aesthetic_score"):
            value = row[task]
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 5
            ):
                raise ValueError("Invalid published score")
        if row.get("training_ready") is not False:
            raise ValueError("Published observations must remain pending training review")
    expected_models = {"luna", "deepseek", "liquid", "jev", "qrealign_lite", "photo_ensemble", "reference"}
    arms = {model: "reference" if model == "reference" else "baseline" for model in expected_models}
    expected_keys = {(identity, model, arms[model]) for identity in ids for model in expected_models}
    if seen != expected_keys:
        raise ValueError("Publication requires every model/reference slot for every image")
    lookup = {(r["case_id"], r["model"], r["arm"]): r for r in rows}
    for name in ("references.jsonl", "labelled-images.jsonl"):
        records = jsonlines(directory / name)
        if len(records) != len(ids) or {r["case_id"] for r in records} != allowed:
            raise ValueError(f"Publication coverage differs: {name}")
        if any(r["view_sha256"] != cases[r["case_id"]]["view_sha256"] for r in records):
            raise ValueError(f"Publication view mismatch: {name}")
        for record in records:
            identity = record["case_id"]
            if name == "labelled-images.jsonl":
                expected = {model: lookup[identity, model, arms[model]] for model in expected_models}
                if record.get("assessments") != expected:
                    raise ValueError("Nested assessments and score table differ")
                masks = record.get("task_masks")
                if (
                    record.get("training_ready") is not False
                    or not isinstance(masks, dict)
                    or set(masks) != {"technical_quality", "aesthetics", "photo_domain"}
                    or any(value is not False for value in masks.values())
                ):
                    raise ValueError("Published image training masks must remain disabled")
            else:
                reference = lookup[identity, "reference", "reference"]
                for task, field in (
                    ("technical_quality", "technical_score"),
                    ("aesthetics", "aesthetic_score"),
                ):
                    if record[task]["score"] != reference[field]:
                        raise ValueError("Reference assessments and score table differ")
    with (directory / "scores.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(rows):
        raise ValueError("CSV and JSONL score counts differ")
    csv_keys = {(r["case_id"], r["model"], r["arm"]) for r in csv_rows}
    if len(csv_keys) != len(csv_rows) or csv_keys != seen:
        raise ValueError("CSV and JSONL observation identities differ")
    for csv_row in csv_rows:
        original = lookup[csv_row["case_id"], csv_row["model"], csv_row["arm"]]
        for field in ("technical_score", "aesthetic_score", "latency_seconds", "price_usd"):
            value = csv_row[field]
            if (value == "") != (original[field] is None) or (
                value and float(value) != float(original[field])
            ):
                raise ValueError("CSV and JSONL outcomes differ")
    return {"views": len(ids), "rows": len(rows), "models": dict(counts), "image_bytes_checked": check_images}
