"""Publish four validated final collage PNGs within the first 1,000-image boundary.

This reads saved artifacts only. It does not render, call models, or publish the
full gallery. The destination must be absent or empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

import gallery_run as gallery

from visual_quality.publication import ALLOWED_FILES, jsonlines, sha, verify_publication

MODELS = {"luna", "deepseek", "liquid", "jev", "qrealign_lite", "photo_ensemble", "reference"}
NUMERIC_FIELDS = (
    "technical_score",
    "aesthetic_score",
    "technical_score_native",
    "aesthetic_score_native",
    "latency_seconds",
    "price_usd",
    "review_elapsed_seconds",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_final_summary(summary, provenance):
    """Terminal failures need an explicit decision; unfinished work never qualifies."""
    for field, expected in (
        ("images", 10000),
        ("benchmark_images", 1500),
        ("statistics_images", 10000),
        ("reference_records", 1500),
    ):
        require(summary.get(field) == expected, "Final full-run coverage is incomplete: " + field)
    saved = provenance["full_run_summary"]
    require(
        {k: v for k, v in summary.items() if k != "publication_validation"} == saved,
        "Gallery and published records use different full-run summaries",
    )
    require(set(summary["coverage"]) == MODELS, "Unexpected final model slots")
    gaps = []
    for model, coverage in summary["coverage"].items():
        expected = 1500 if model == "reference" else 10000
        statuses = coverage["status_counts"]
        terminal = {"valid", "failed", "invalid", "partial", "http_error", "transport_or_parse_error"}
        if model == "photo_ensemble":
            terminal = {"valid", "partial", "missing_or_invalid_components"}
        elif model == "reference":
            terminal = {"valid"}
        require(
            coverage["expected"] == expected
            and all(type(n) is int and n >= 0 for n in statuses.values())
            and sum(statuses.values()) == expected,
            "Incomplete final observation coverage: " + model,
        )
        require(not (set(statuses) - terminal), "Unfinished observation statuses: " + model)
        pairs = coverage["numeric_pairs"]
        require(type(pairs) is int and 0 <= pairs <= expected, "Invalid numeric coverage: " + model)
        if pairs != expected:
            gaps.append(model)
    require("reference" not in gaps, "Final references must be complete")
    reason = provenance.get("incomplete_models_reason")
    require(
        not gaps or isinstance(reason, str) and bool(reason.strip()),
        "Numeric gaps require a documented decision",
    )


def check_selected_scores(source, records, selected, cases):
    published = {
        (r["case_id"], r["model"], r["arm"]): r
        for r in jsonlines(records / "scores.jsonl")
        if r["case_id"] in selected
    }
    seen = set()
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row["case_id"]
            if identity not in selected:
                continue
            key = identity, row["model"], row["arm"]
            require(key in published and key not in seen, "Unexpected or duplicate selected score slot")
            seen.add(key)
            original = published[key]
            require(row["view_sha256"] == cases[identity]["view_sha256"], "Selected score view mismatch")
            require(row["status"] == original["status"], "Selected status differs from published records")
            for field in NUMERIC_FIELDS:
                actual, expected = row.get(field), original.get(field)
                actual = None if actual in (None, "", "null") else Decimal(actual)
                expected = None if expected is None else Decimal(str(expected))
                require(actual == expected, "Selected score/latency/price differs: " + field)
            for field in ("technical_score_display", "aesthetic_score_display"):
                require((row.get(field) or None) == original.get(field), "Selected display score differs")
    require(seen == set(published) and len(seen) == 72 * 7, "Selected score slots are incomplete")


def check_gallery(directory, records):
    pins = {records / name: sha(records / name) for name in ALLOWED_FILES}
    require(
        verify_publication(records)["views"] == 1000, "Exactly the first 1,000 published images are required"
    )
    for name in ("generation-proof.json", "selection.json", "png-validation.json"):
        pins[directory / name] = sha(directory / name)
    proof = gallery.load_json(directory / "generation-proof.json")
    validation = proof["validation"]
    require(
        proof.get("status") == validation.get("status") == "passed"
        and proof.get("images") == 10000
        and proof.get("references") == 1500
        and proof.get("collage_images") == 72
        and proof.get("collage_within_first_1000") is True
        and proof.get("scores_preserved") is True
        and proof.get("statistics_in_assessments") is False
        and proof.get("training_ready") is False,
        "A successful final gallery generation proof is required",
    )
    paths = {name: Path(value) for name, value in proof["source_paths"].items()}
    require(set(paths) == set(proof["source_hashes"]), "Incomplete gallery input provenance")
    require(
        {"manifest", "references", "scores", "summary", "adapter", "png_renderer", "validator"}
        <= paths.keys(),
        "Missing gallery input provenance",
    )
    pins.update({paths[name]: value for name, value in proof["source_hashes"].items()})
    for name, field in (
        ("data.js", "data_sha256"),
        ("index.html", "index_sha256"),
        ("style.css", "style_sha256"),
        ("preview-index.json", "preview_index_sha256"),
    ):
        pins[directory / name] = proof[field]
    check_hashes(pins)
    provenance = gallery.load_json(records / "provenance.json")
    check_final_summary(gallery.load_json(paths["summary"]), provenance)
    cases = gallery.load_json(paths["manifest"])["cases"]
    require(len(cases) == len({c["case_id"] for c in cases}) == 10000, "Full manifest coverage differs")
    allowed = gallery.publication_ids(cases)
    require(
        allowed == gallery.load_json(records / "publication-case-ids.json"), "Publication allowlist changed"
    )
    published_cases = {c["case_id"]: c for c in gallery.load_json(records / "manifest.json")["cases"]}
    frozen = paths["manifest"].parent / "gallery-collage-selection.json"
    require(frozen.is_file(), "The original frozen selection must exist")
    pins[frozen] = sha(frozen)
    selection = gallery.freeze_selection(
        cases,
        {r["case_id"]: r for r in jsonlines(paths["references"])},
        proof["source_hashes"]["references"],
        proof["source_hashes"]["manifest"],
        frozen,
    )
    require(
        selection == gallery.load_json(directory / "selection.json"),
        "Gallery selection differs from its freeze",
    )
    selected = selection["case_ids"]
    require(
        len(selected) == len(set(selected)) == 72 and set(selected) <= set(allowed),
        "Non-public collage image",
    )
    by_id = {c["case_id"]: c for c in cases}
    for identity in selected:
        case, published = by_id[identity], published_cases[identity]
        for field in ("view_sha256", "family_id", "benchmark_order", "publication_order"):
            require(case[field] == published[field], "Selected image identity differs: " + field)
        pins[Path(case["image_path"])] = case["view_sha256"]
    check_selected_scores(paths["scores"], records, set(selected), published_cases)
    source = (directory / "data.js").read_text()
    require(source.startswith("const CASES=") and source.endswith(";\n"), "Unexpected gallery data format")
    data = json.loads(source[len("const CASES=") : -2])
    require(len(data) == 10000 and {c["id"] for c in data} == set(by_id), "Full gallery coverage differs")
    pages = gallery.validate_gallery.validate_pages(
        argparse.Namespace(gallery=directory, scores=paths["scores"]), data
    )
    require(pages == validation["static_pages"], "Validated PNG pages differ from the generation proof")
    for index, page in enumerate(pages, 1):
        pins[directory / f"comparison-page-{index}.png"] = page["png_sha256"]
        pins[directory / page["html"]] = page["html_sha256"]
    check_hashes(pins)
    return selection, pages, pins, provenance


def check_hashes(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, "Artifact hash mismatch: " + str(path))


def publish_collage(directory, records, destination):
    directory, records, destination = map(Path, (directory, records, destination))
    require(not destination.is_symlink(), "Collage destination cannot be a symlink")
    require(
        not destination.exists() or destination.is_dir() and not any(destination.iterdir()),
        "Collage destination must be empty",
    )
    selection, pages, pins, records_provenance = check_gallery(directory, records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collage-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "collage"
        staging.mkdir()
        for index, page in enumerate(pages, 1):
            name = f"comparison-page-{index}.png"
            shutil.copyfile(directory / name, staging / name)
            require(sha(staging / name) == page["png_sha256"], "Copied PNG hash mismatch")
        # Construct bounded records instead of copying arbitrary gallery metadata.
        gallery.write_json(
            staging / "selection.json",
            {
                k: selection[k]
                for k in (
                    "method",
                    "case_ids",
                    "references_sha256",
                    "manifest_sha256",
                    "publication_ids_sha256",
                    "frozen_selection_sha256",
                )
            },
        )
        gallery.write_json(
            staging / "provenance.json",
            {
                "schema_version": "labeling-collage-publication-v1",
                "publication_limit": 1000,
                "collage_images": 72,
                "generation_proof_sha256": pins[directory / "generation-proof.json"],
                "png_validation_sha256": pins[directory / "png-validation.json"],
                "publication_provenance_sha256": pins[records / "provenance.json"],
                "publication_records": records_provenance["records"],
                "selection_sha256": sha(staging / "selection.json"),
                "pngs": {f"comparison-page-{i}.png": p["png_sha256"] for i, p in enumerate(pages, 1)},
                "training_ready": False,
            },
        )
        (staging / "README.md").write_text(
            "# Reference-range collage\n\n"
            "- [Page 1](comparison-page-1.png)\n"
            "- [Page 2](comparison-page-2.png)\n"
            "- [Page 3](comparison-page-3.png)\n"
            "- [Page 4](comparison-page-4.png)\n\n"
            "Four final comparison pages show 72 frozen images selected only from the first 1,000 "
            "explicit publication orders. Selection spans assistant reference technical scores; "
            "it is not a population-weighted sample or a human opinion benchmark.\n\n"
            "Technical and aesthetic scores remain separate. Missing assessments or measurements "
            "appear as dashes. Terminal missing outcomes are documented in the bounded records' "
            "publication provenance. Lite and Reference prices use the assigned $0.00 accounting "
            "rule; infrastructure is separate. The ensemble has no independent HTTP latency or charge.\n\n"
            "The publisher checked the saved generation proof, frozen selection, actual PNG and "
            "source hashes, delivered image identities, completed full-run coverage, and selected "
            "scores, latency and prices against the bounded publication records. The full gallery "
            "remains under ignored run data. These labels are experimental; training is disabled.\n"
        )
        check_hashes(pins)
        require(
            not destination.exists() or not any(destination.iterdir()), "Collage destination became nonempty"
        )
        staging.replace(destination)
    return {"destination": str(destination), "pages": 4, "images": 72, "publication_limit": 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = publish_collage(args.gallery.resolve(), args.records.resolve(), args.destination.absolute())
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
