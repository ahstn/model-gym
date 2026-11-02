"""Build a local 10,000-image gallery and a frozen reference-only 72-image collage.

All outputs remain under ignored data. Existing observations, references, the
publication allowlist and the running 4,000-image gallery service are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "rehearsal-4000-2026-11-01"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import gallery
import gallery_png
import validate_gallery

from visual_quality.ensemble import MODEL, configuration

DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")
METADATA_FIELDS = (
    "technical_score_display",
    "aesthetic_score_display",
    "view_sha256",
    "latency_kind",
    "price_basis",
    "scope_floor_applied",
    "reference_eligibility",
    "excluded_probability",
    "limitations",
)
PARTITIONS = {
    "development": "Original development references",
    "representative": "Original representative references",
    "challenge": "Original challenge references",
    "expansion_examples": "New publication examples",
    "expansion_confirmation": "New confirmation references",
    "rehearsal": "Other corpus images",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publication_ids(cases: list[dict]) -> list[str]:
    """Require the explicit, already established first 1,000 publication orders."""
    allowed = sorted(
        [
            case
            for case in cases
            if type(case.get("publication_order")) is int and 1 <= case["publication_order"] <= 1000
        ],
        key=lambda case: case["publication_order"],
    )
    if len(allowed) != 1000 or [c["publication_order"] for c in allowed] != list(range(1, 1001)):
        raise ValueError("Expected the frozen first 1,000 publication orders")
    if any(c["benchmark_order"] != c["publication_order"] for c in allowed):
        raise ValueError("Publication order differs from the frozen benchmark")
    return [case["case_id"] for case in allowed]


def freeze_selection(
    cases: list[dict], references: dict, references_hash: str, manifest_hash: str, path: Path
) -> dict:
    allowed = publication_ids(cases)
    data = [
        {
            "id": identity,
            "scores": {
                "reference:reference": {"technical_score": references[identity]["technical_quality"]["score"]}
            },
        }
        for identity in allowed
    ]
    selected = gallery.choose_pages(data)
    record = {
        "method": "72 equally spaced ranks among the first 1,000 publication references, ordered by reference technical score then case ID; no model scores",
        "case_ids": [case["id"] for case in selected],
        "references_sha256": references_hash,
        "manifest_sha256": manifest_hash,
        "publication_ids_sha256": hashlib.sha256(json.dumps(allowed).encode()).hexdigest(),
    }
    if path.exists():
        if load_json(path) != record:
            raise ValueError("Frozen collage selection or its reference provenance changed")
    else:
        write_json(path, record)
    return {**record, "frozen_selection_sha256": gallery.digest(path)}


def load_inputs(data_root: Path, exports: Path) -> tuple:
    paths = {
        "manifest": data_root / "manifest.json",
        "references": data_root / "references-all.jsonl",
        "reference_freeze": data_root / "references-freeze.json",
        "neighbours": data_root / "related-images.json",
        "labels": HERE / "corpus-cluster-labels.json",
        "scores": exports / "scores-long.csv",
        "images": exports / "labelled-images.jsonl",
        "summary": exports / "summary.json",
        "adapter": Path(__file__).resolve(),
        "html_renderer": SHARED / "gallery.py",
        "png_renderer": SHARED / "gallery_png.py",
        "validator": SHARED / "validate_gallery.py",
    }
    hashes = {name: gallery.digest(path) for name, path in paths.items()}
    cases = load_json(paths["manifest"])["cases"]
    by_id = {case["case_id"]: case for case in cases}
    if len(cases) != 10000 or len(by_id) != 10000:
        raise ValueError("Expected 10,000 distinct manifest images")
    freeze = load_json(paths["reference_freeze"])
    if (
        not freeze["complete"]
        or freeze["references_sha256"] != hashes["references"]
        or freeze["manifest_sha256"] != hashes["manifest"]
    ):
        raise ValueError("Reference freeze does not match the gallery inputs")
    references = {}
    for line in paths["references"].open():
        reference = json.loads(line)
        identity = reference["case_id"]
        if identity in references or reference["view_sha256"] != by_id[identity]["view_sha256"]:
            raise ValueError("Reference identity/view mismatch")
        references[identity] = reference
    if len(references) != 1500 or set(references) != set(freeze["case_ids"]):
        raise ValueError("Expected the exact 1,500 frozen references")
    rows = gallery.read_rows(paths["scores"], recompute_ensemble=False, metadata_fields=METADATA_FIELDS)
    validate_rows(by_id, rows, references)
    metadata = load_metadata(paths["images"], by_id, references)
    related = load_json(paths["neighbours"])
    if related["manifest_sha256"] != hashes["manifest"]:
        raise ValueError("Related-image index refers to a different manifest")
    families = defaultdict(set)
    for case in cases:
        families[case["family_id"]].add(case["case_id"])
    for case in cases:
        identity = case["case_id"]
        recorded = related["family_members"].get(case["family_id"], [identity])
        if set(recorded) != families[case["family_id"]]:
            raise ValueError("Related-image family identity mismatch")
        if any(v["case_id"] not in by_id for v in related["neighbours"][identity]):
            raise ValueError("Unknown related-image identity")
    return paths, hashes, cases, rows, references, metadata, related, load_json(paths["summary"])


def validate_rows(cases: dict, rows: dict, references: dict) -> None:
    expected = {model + ":baseline" for model, _ in gallery.MODELS if model != "reference"}
    if set(rows) != set(cases):
        raise ValueError("Score export and manifest images differ")
    for identity, records in rows.items():
        keys = expected | ({"reference:reference"} if identity in references else set())
        if set(records) != keys:
            raise ValueError("Missing, duplicated or retired model slot: " + identity)
        for row in records.values():
            if row["view_sha256"] != cases[identity]["view_sha256"]:
                raise ValueError("Score delivered-view mismatch: " + identity)
        if identity in references:
            row = records["reference:reference"]
            for dimension, task in (("technical", "technical_quality"), ("aesthetic", "aesthetics")):
                if row[dimension + "_score"] != references[identity][task]["score"]:
                    raise ValueError("Export reference differs from its frozen record")
            if row["latency_seconds"] is not None or row["price_usd"] != 0:
                raise ValueError("Reference latency/price contract changed")
        if (
            records[MODEL + ":baseline"]["latency_seconds"] is not None
            or records[MODEL + ":baseline"]["price_usd"] is not None
        ):
            raise ValueError("Derived ensemble has an independent latency or charge")


def load_metadata(path: Path, cases: dict, references: dict) -> dict:
    metadata = {}
    for line in path.open():
        record = json.loads(line)
        identity = record["case_id"]
        if identity in metadata or record["view_sha256"] != cases[identity]["view_sha256"]:
            raise ValueError("Image statistics identity/view mismatch")
        stats = record["image_stats"]
        if stats.get("used_in_inference") is not False or stats.get("training_features_enabled") is not False:
            raise ValueError("Statistics must remain separate review metadata")
        if stats.get("histogram_entropy") != stats.get("luma_entropy"):
            raise ValueError("Histogram entropy must remain the recorded luma-entropy alias")
        metadata[identity] = {"image_stats": stats}
        if identity in references:
            reference = references[identity]
            metadata[identity]["reference_review"] = {
                key: value
                for key, value in reference.items()
                if key
                in {
                    "review_elapsed_seconds",
                    "review_started_at",
                    "review_completed_at",
                    "review_timing_basis",
                    "eligibility_evidence",
                }
            }
    if set(metadata) != set(cases):
        raise ValueError("Missing image metadata")
    return metadata


def validate_output(
    output: Path, cases: list[dict], rows: dict, metadata: dict, related: dict, scores: Path
) -> dict:
    text = (output / "data.js").read_text()
    if not text.startswith("const CASES=") or not text.endswith(";\n"):
        raise ValueError("Unexpected generated gallery data format")
    data = json.loads(text[len("const CASES=") : -2])
    by_id = {case["case_id"]: case for case in cases}
    if len(data) != len(cases) or {case["id"] for case in data} != set(by_id):
        raise ValueError("Generated gallery image coverage differs")
    previews = load_json(output / "preview-index.json")
    for case in data:
        identity = case["id"]
        original = by_id[identity]
        if case["scores"] != rows[identity] or case["metadata"] != metadata[identity]:
            raise ValueError("Generated scores or separate metadata changed")
        if (
            case["view_sha256"] != original["view_sha256"]
            or case["original"] != Path(original["image_path"]).as_uri()
        ):
            raise ValueError("Generated original image link changed")
        if case["partition"] != original["benchmark_partition"] or case["cluster"] != str(
            original["content_cluster"]
        ):
            raise ValueError("Generated partition or content cluster changed")
        if case["neighbours"] != related["neighbours"][identity]:
            raise ValueError("Generated related images changed")
        siblings = set(related["family_members"].get(original["family_id"], [])) - {identity}
        if set(case["siblings"]) != siblings:
            raise ValueError("Generated family links changed")
        if gallery.digest(output / case["preview"]) != previews[identity]["preview_sha256"]:
            raise ValueError("Generated preview hash differs")
    pages = validate_gallery.validate_pages(argparse.Namespace(gallery=output, scores=scores), data)
    return {
        "status": "passed",
        "images": len(data),
        "score_rows": sum(len(r) for r in rows.values()),
        "preview_hashes_checked": len(data),
        "static_pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--browser-check",
        action="store_true",
        help="also verify filters, navigation, metadata and all previews in local Chromium",
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    exports = (args.exports or root / "exports-preview").resolve()
    output = (args.output or root / "gallery-preview").resolve()
    selection_path = (args.selection or root / "gallery-collage-selection.json").resolve()
    if not output.is_relative_to(root) or not selection_path.is_relative_to(root) or "data" not in root.parts:
        parser.error("Gallery and selection outputs must remain under the ignored data root")
    paths, hashes, cases, rows, references, metadata, related, summary = load_inputs(root, exports)
    selection = freeze_selection(cases, references, hashes["references"], hashes["manifest"], selection_path)
    labels = load_json(paths["labels"])["clusters"]
    # A failed rebuild must not leave a successful proof for replaced artifacts.
    (output / "generation-proof.json").unlink(missing_ok=True)
    data = gallery.make_data(
        cases,
        rows,
        output=output,
        neighbours=related["neighbours"],
        labels=labels,
        cluster_key="content_cluster",
        metadata=metadata,
    )
    indexed = {case["id"]: case for case in data}
    selected = [indexed[identity] for identity in selection["case_ids"]]
    write_json(output / "selection.json", selection)
    coverage = summary["coverage"]
    notice = (
        "Saved export snapshot. "
        + "; ".join(
            f"{label}: {coverage[model]['numeric_pairs']:,}/{coverage[model]['expected']:,} paired scores"
            for model, label in gallery.MODELS
            if model != "reference"
        )
        + ". Missing outcomes remain dashes. Review windows include preparation/tool overhead."
    )
    links = tuple(
        (label, os.path.relpath(path, output))
        for label, path in (
            ("Full score export", paths["scores"]),
            ("Coverage and costs", paths["summary"]),
            ("Page 1", output / "comparison-page-1.png"),
            ("Page 2", output / "comparison-page-2.png"),
            ("Page 3", output / "comparison-page-3.png"),
            ("Page 4", output / "comparison-page-4.png"),
        )
    )
    gallery.write_gallery(
        data,
        selected,
        output=output,
        title="10,000-image scoring comparison",
        reference_count=len(references),
        partition_labels=PARTITIONS,
        treatments=False,
        links=links,
        notice=notice,
    )
    gallery_png.render(
        gallery=output,
        cases={c["case_id"]: c for c in cases},
        rows=rows,
        references_path=paths["references"],
        scores_path=paths["scores"],
        title_text="10,000-image scoring comparison",
        footer_text="Exact view, evidence and image statistics in the gallery",
    )
    if any(gallery.digest(paths[name]) != value for name, value in hashes.items()):
        raise ValueError("An export changed during rendering; rebuild from a stable snapshot")
    validation = validate_output(output, cases, rows, metadata, related, paths["scores"])
    if args.browser_check:
        extra = """
assert(document.body.classList.contains('adaptive-gallery'),'responsive gallery body');
assert(document.querySelector('meta[name=viewport]')!==null,'viewport metadata');
assert(CASES.every(c=>c.metadata?.image_stats&&Object.values(c.scores).every(r=>!('image_stats' in r))),'statistics remain separate from scores');
assert(document.getElementById('grid').textContent.includes('Image statistics (review metadata)'),'statistics metadata visible');
assert(document.getElementById('grid').textContent.includes('Reference review record'),'reference review windows visible');
"""
        script = validate_gallery.BROWSER_SCRIPT.replace(
            "result.status='passed';", extra + "result.status='passed';"
        )
        validation["browser"] = validate_gallery.browser_check(
            argparse.Namespace(gallery=output, chromium=validate_gallery.CHROMIUM), script=script
        )
    proof = {
        "status": "passed",
        "images": len(data),
        "references": len(references),
        "collage_images": len(selected),
        "collage_within_first_1000": True,
        "partitions": dict(Counter(c["partition"] for c in data)),
        "ensemble": configuration(),
        "source_hashes": hashes,
        "source_paths": {k: str(v) for k, v in paths.items()},
        "scores_preserved": True,
        "statistics_in_assessments": False,
        "training_ready": False,
        "service_changed": False,
        "data_sha256": gallery.digest(output / "data.js"),
        "index_sha256": gallery.digest(output / "index.html"),
        "style_sha256": gallery.digest(output / "style.css"),
        "preview_index_sha256": gallery.digest(output / "preview-index.json"),
        "validation": validation,
    }
    write_json(output / "generation-proof.json", proof)
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(data),
                "references": len(references),
                "collage_images": len(selected),
            }
        )
    )


if __name__ == "__main__":
    main()
