"""Publish portable copies of the two completed 72-image trials.

Frozen experiment files and ignored inference records remain untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATA = EXPERIMENTS.parent / "data"
TRIALS = {
    "photo-scope": "photo-scope-trial-2026-10-31",
    "deepseek-high": "deepseek-high-trial-2026-10-31",
}
SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{24,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_.-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/]{32,}"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def scan(path: Path) -> None:
    text = path.read_text()
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Sensitive or embedded image data detected in {path}")


def portable_value(value: object, image_map: dict[str, str]) -> object:
    if isinstance(value, str):
        return image_map.get(value, value)
    if isinstance(value, list):
        return [portable_value(item, image_map) for item in value]
    if isinstance(value, dict):
        return {key: portable_value(item, image_map) for key, item in value.items()}
    return value


def main() -> None:
    manifest = json.loads((EXPERIMENTS / TRIALS["photo-scope"] / "manifest.json").read_text())
    cases = manifest["cases"]
    assert len(cases) == 72
    image_map = {}
    image_records = []
    images = HERE / "images"
    images.mkdir(exist_ok=True)
    for case in cases:
        source = Path(case["image_path"])
        assert digest(source) == case["view_sha256"]
        destination = images / source.name
        shutil.copyfile(source, destination)
        assert digest(destination) == case["view_sha256"]
        image_map[str(source)] = f"../images/{source.name}"
        image_records.append({
            "case_id": case["case_id"],
            "path": str(destination.relative_to(HERE)),
            "sha256": digest(destination),
            "bytes": destination.stat().st_size,
            "source_path": str(source),
        })

    runs = []
    gallery_links = []
    for short_name, experiment in TRIALS.items():
        gallery = HERE / short_name
        gallery.mkdir(exist_ok=True)
        source_gallery = EXPERIMENTS / experiment / "collage"
        for name in ("collage.css",):
            shutil.copyfile(source_gallery / name, gallery / name)
        for source in sorted(source_gallery.glob("*.html")):
            def replace_link(match: re.Match[str]) -> str:
                attr, original = match.groups()
                resolved = str((source.parent / original).resolve())
                replacement = image_map.get(resolved, original)
                return f'{attr}="{replacement}"'
            transformed = re.sub(r'(src|href)="([^"]+)"', replace_link, source.read_text())
            destination = gallery / source.name
            destination.write_text(transformed)
            for link in re.findall(r'(?:src|href)="([^"]+)"', transformed):
                if "://" not in link and not link.startswith("#"):
                    gallery_links.append(gallery / link)
        with (source_gallery / "scores.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        for row in rows:
            row["image_path"] = image_map[row["image_path"]]
        with (gallery / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        selection = json.loads((source_gallery / "selection.json").read_text())
        write_json(gallery / "selection.json", portable_value(selection, image_map))

        for source_run in sorted((DATA / experiment).glob("*/images")):
            route = source_run.parent.name
            destination = HERE / "results" / short_name / route
            destination.mkdir(parents=True, exist_ok=True)
            report = json.loads((source_run / "report.json").read_text())
            assert report["status"] == "complete" and report["planned"] == 72
            for name in ("run.json", "report.json", "observations.jsonl"):
                source = source_run / name
                scan(source)
                shutil.copyfile(source, destination / name)
            records = []
            for attempt in sorted((source_run / "attempts").iterdir()):
                request = attempt / "request.json"
                result = attempt / "result.json"
                scan(request)
                scan(result)
                records.append({
                    "attempt_id": attempt.name,
                    "request": json.loads(request.read_text()),
                    "result": json.loads(result.read_text()),
                    "source_request_sha256": digest(request),
                    "source_result_sha256": digest(result),
                })
            assert len(records) == 72
            (destination / "attempts.jsonl").write_text("".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ))
            runs.append({
                "experiment": experiment, "route": route, "attempts": len(records),
                "run_id": report["run_id"], "path": str(destination.relative_to(HERE)),
                "known_cost_usd": report["known_cost_usd"],
            })

    assert all(path.is_file() for path in gallery_links)
    for path in HERE.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".html", ".csv"}:
            scan(path)
    write_json(HERE / "publication-manifest.json", {
        "schema_version": "published-trials-v1",
        "scope": "Two completed 72-image trials, 360 original model responses, shared original images",
        "image_count": len(image_records), "image_bytes": sum(row["bytes"] for row in image_records),
        "images": image_records, "runs": runs,
        "response_export": "Exact parsed request metadata and result records; raw body strings unchanged. Source file hashes identify original serialization.",
        "portability": "Gallery HTML, CSV image paths and selection image paths use shared relative assets. Frozen experiment records retain original local provenance paths.",
        "validation": {"image_hashes_match": True, "gallery_links_resolve": True,
                       "credential_and_embedded_image_pattern_scan": "passed"},
    })
    print(json.dumps({"images": len(image_records), "image_bytes": sum(row["bytes"] for row in image_records),
                      "runs": len(runs), "attempts": sum(row["attempts"] for row in runs)}))


if __name__ == "__main__":
    main()
