"""Verify the current labeling profile and retained record integrity offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from visual_quality.labeling_profile import load_profile

HERE = Path(__file__).resolve().parent


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(check_images: bool = False) -> dict:
    profile = load_profile(HERE)
    records = HERE / "records"
    provenance = json.loads((records / "provenance.json").read_text())
    for name, expected in provenance["records"].items():
        if file_sha(records / name) != expected:
            raise ValueError(f"Record hash mismatch: {name}")
    manifest = json.loads((records / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(cases) != len(manifest["cases"]) or len(cases) != 4000:
        raise ValueError("Manifest must contain 4,000 unique views")
    if check_images:
        for case in cases.values():
            if file_sha(Path(case["image_path"])) != case["view_sha256"]:
                raise ValueError(f"Delivered image hash mismatch: {case['case_id']}")
    seen = set()
    counts = Counter()
    numeric_pairs = Counter()
    with (records / "scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["case_id"], row["model"], row["arm"])
            if key in seen or row["case_id"] not in cases:
                raise ValueError(f"Duplicate or unrecognized score row: {key}")
            seen.add(key)
            if row["view_sha256"] != cases[row["case_id"]]["view_sha256"]:
                raise ValueError(f"Score row has a different view: {key}")
            for task in ("technical_score", "aesthetic_score"):
                if row[task]:
                    value = float(row[task])
                    if not math.isfinite(value) or not 1 <= value <= 5:
                        raise ValueError(f"Invalid score for {key}: {task}")
            if row["training_ready"].lower() != "false":
                raise ValueError(f"Unexpected training-ready row: {key}")
            counts[row["model"]] += 1
            if row["technical_score"] and row["aesthetic_score"]:
                numeric_pairs[row["model"]] += 1
    if dict(counts) != provenance["model_rows"]:
        raise ValueError("Score counts differ from the recorded manifest")
    return {
        "views": len(cases),
        "rows": len(seen),
        "models": dict(sorted(counts.items())),
        "numeric_score_pairs": dict(sorted(numeric_pairs.items())),
        "image_bytes_checked": check_images,
        "profile_sha256": profile["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="store_true", help="also hash every local delivered image")
    args = parser.parse_args()
    print(json.dumps(verify(args.images), indent=2))


if __name__ == "__main__":
    main()
