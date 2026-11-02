"""Separate image-stats metadata for label debugging, never inference input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

METRIC_FIELDS = ("sharpness_peak", "luma_entropy", "focus_area", "channel_clip", "colourfulness")
DESCRIPTIONS = {
    "sharpness_peak": {"unit": "luma^2", "meaning": "Laplacian variance of the sharpest tile"},
    "luma_entropy": {"unit": "bits", "meaning": "Shannon entropy of the 256-bin luma histogram"},
    "histogram_entropy": {"unit": "bits", "alias_of": "luma_entropy"},
    "focus_area": {
        "unit": "fraction",
        "meaning": "Share of tiles with at least one quarter of this image's peak Laplacian variance",
        "caution": "Relative measure; uniformly blurred images can have high values",
    },
    "channel_clip": {"unit": "fraction", "meaning": "Pixels with any RGB channel at or above 254"},
    "colourfulness": {"unit": "0-255", "meaning": "Hasler-Suesstrunk colourfulness"},
}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configuration() -> dict:
    import cv2
    import image_stats
    import numpy
    import PIL

    return {
        "algorithm_version": image_stats.ALGORITHM_VERSION,
        "max_edge": 1536,
        "source_code_sha256": {
            p.name: file_sha(p) for p in sorted(Path(image_stats.__file__).parent.glob("*.py"))
        },
        "software": {"numpy": numpy.__version__, "Pillow": PIL.__version__, "opencv": cv2.__version__},
    }


def extract_metadata(case: dict, record: dict, config: dict) -> dict:
    """Join only statistics computed from this exact delivered view."""
    if record["sha256"] != case["view_sha256"]:
        raise ValueError("Statistics and delivered-image hashes differ")
    analysis = record["analysis"]
    metrics = {name: analysis["metrics"][name] for name in METRIC_FIELDS}
    metrics["histogram_entropy"] = metrics["luma_entropy"]
    return {
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "view_sha256": case["view_sha256"],
        "image_stats": {
            **metrics,
            "configuration": config,
            "analysed_dimensions": analysis["analysed_dimensions"],
            "native_dimensions": analysis["native_dimensions"],
            "definitions": DESCRIPTIONS,
            "role": "debugging_and_review_metadata",
            "used_in_inference": False,
            "training_features_enabled": False,
        },
        "image_stats_full": record,
    }


def analyse_case(case: dict, cache_dir: Path, config: dict) -> dict:
    from image_stats import analyse_file

    path = Path(case["image_path"])
    if file_sha(path) != case["view_sha256"]:
        raise ValueError(f"Delivered image changed: {case['case_id']}")
    source = Path(case["source_path"]) if case.get("source_path") else None
    source_sha = file_sha(source) if source else None
    if source and source_sha != case["source_sha256"]:
        raise ValueError(f"Source image changed: {case['case_id']}")
    identity = {"view_sha256": case["view_sha256"], "source_sha256": source_sha, "configuration": config}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["cache_identity"] != identity:
            raise ValueError("Cached statistics identity mismatch")
        record = cached["record"]
    else:
        record = analyse_file(path, max_edge=config["max_edge"], source_path=source)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"cache_identity": identity, "record": record}, sort_keys=True) + "\n"
        )
        temporary.replace(cache_path)
    return extract_metadata(case, record, config)
