"""Append 22 independently selected, previously unexposed source images to the frozen 50."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from visual_quality.storage import file_sha256, read_jsonl

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "prompt-ablation-2026-10-30"
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
SOURCE = DATA / "pilot-10k-v1"
OUTPUT = DATA / HERE.name
SEED = 20261031
QUOTAS = {"BLR": 2, "BRT": 1, "DRK": 1, "FRM": 2, "OBS": 1, "ROT": 1}

spec = importlib.util.spec_from_file_location("prior_cohort_preparation", PRIOR / "prepare.py")
prior = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prior)
# Reuse unchanged source provenance, overlap, rendering, and immutable-write utilities.
# Point the audit at this experiment so every earlier experiment (including the 50) is excluded.
prior.HERE = HERE
prior.OUTPUT = OUTPUT
prior.SEED = SEED


def select(rows: list[dict], historical: list[dict]) -> list[dict]:
    if len({row["id"] for row in rows}) != len(rows) or any(row["split"] != "train" for row in rows):
        raise ValueError("Expected unique training records")
    previous, selected = list(historical), []
    distortion_parents = {row["parent_id"] for row in rows if row["component"] == "distortions"}

    def accept(row: dict, role: str) -> bool:
        if prior.overlaps([row], previous) or prior.metadata(row)["frames"] != 1:
            return False
        selected.append({
            "row": row,
            "role": role,
            "operator": None,
            "severity": None,
            "vizwiz_stratum": prior.vizwiz_stratum(row) if row["component"] == "vizwiz" else None,
            "stratum": role,
            "issue_tags": prior.issue_tags(row) or [role],
        })
        previous.append(row)
        return True

    for defect, quota in QUOTAS.items():
        pool = [r for r in rows if r["component"] == "vizwiz" and prior.vizwiz_stratum(r) == defect]
        count = 0
        for row in sorted(pool, key=lambda r: prior.order(defect, r["id"])):
            if accept(row, "vizwiz"):
                count += 1
                if count == quota:
                    break
        if count != quota:
            raise ValueError(f"Insufficient unused images for {defect}")
    pool = [r for r in rows if r["component"] == "megalith" and r["id"] not in distortion_parents]
    count = 0
    for row in sorted(pool, key=lambda r: prior.order("standalone", r["id"])):
        if accept(row, "standalone_megalith"):
            count += 1
            if count == 14:
                break
    if len(selected) != 22 or count != 14:
        raise ValueError("Incomplete extension")
    return selected


def verify() -> dict:
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["manifest_id"] != prior.digest({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("Manifest content hash differs")
    for filename, expected in manifest["inputs"].items():
        if file_sha256(Path(filename)) != expected:
            raise ValueError(f"Preparation input changed: {filename}")
    cases = manifest["cases"]
    original = json.loads((PRIOR / "manifest.json").read_text())["cases"]
    if cases[:50] != original:
        raise ValueError("Existing 50 metadata changed")
    if [c["case_id"] for c in cases] != [f"P{i:03d}" for i in range(1, 73)]:
        raise ValueError("Expected ordered IDs P001 through P072")
    if len({c["image_id"] for c in cases}) != 72 or len({c["view_sha256"] for c in cases}) != 72:
        raise ValueError("Duplicate image identity or delivered bytes")
    if len({c["family_id"] for c in cases}) != 54:
        raise ValueError("Expected 54 families")
    for c in cases:
        for path, key in ((c["image_path"], "view_sha256"), (c["source_path"], "source_sha256")):
            if file_sha256(Path(path)) != c[key]:
                raise ValueError(f"Image bytes changed: {c['case_id']}")
        if Path(c["image_path"]).stat().st_size != c["delivered_bytes"] or c["delivered_bytes"] > prior.MAX_PNG_BYTES:
            raise ValueError("Image exceeds byte policy")
        with Image.open(c["image_path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (c["width"], c["height"]):
                raise ValueError("View metadata differs")
        if max(c["width"], c["height"]) > c["target_edge"] or c["target_edge"] > prior.MAX_EDGE:
            raise ValueError("View exceeds edge policy")
        if any(c["task_masks"].values()):
            raise ValueError("Experimental labels must not be enabled for training")
    new = cases[50:]
    if Counter(c["component"] for c in new) != {"vizwiz": 8, "megalith": 14}:
        raise ValueError("Extension component mix differs")
    if Counter(c["vizwiz_stratum"] for c in new if c["component"] == "vizwiz") != QUOTAS:
        raise ValueError("Extension flaw mix differs")
    for path, expected in manifest["packet_files"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("Anonymous image packet differs")
    return {"verified": True, "manifest_id": manifest["manifest_id"], "images": 72,
            "preserved_images": 50, "new_images": 22, "families": 54,
            "new_components": dict(Counter(c["component"] for c in new)),
            "max_delivered_bytes": max(c["delivered_bytes"] for c in cases)}


def prepare() -> dict:
    if (HERE / "manifest.json").exists():
        return verify()
    rows = read_jsonl(SOURCE / "train.jsonl")
    historical, audit = prior.historical_exclusions(rows)
    selections = select(rows, historical)
    policy = prior.preprocessing_policy()
    new = prior.build_views(selections, policy)
    for index, case in enumerate(new, 51):
        case["case_id"] = f"P{index:03d}"
    old_manifest = json.loads((PRIOR / "manifest.json").read_text())
    old = old_manifest["cases"]
    cases = [*old, *new]
    # Review packets contain only image IDs, pixel paths/hashes and dimensions; no source labels.
    packets = {}
    for name, subset in (("new-images", new), ("all-images", cases)):
        path = OUTPUT / "review-packets" / f"{name}.json"
        packet = [{"case_id": c["case_id"], "path": c["image_path"], "view_sha256": c["view_sha256"],
                   "width": c["width"], "height": c["height"]} for c in subset]
        prior.freeze(path, prior.encoded(packet))
        packets[str(path)] = file_sha256(path)
    pins = {str(p): file_sha256(p) for p in [Path(__file__).resolve(), PRIOR / "prepare.py",
            PRIOR / "manifest.json", SOURCE / "dataset.json", SOURCE / "train.jsonl", Path(prior.views.__file__)]}
    pins.update(audit["ledger_files"])
    manifest = {
        "schema_version": "photo-scope-extension-v1", "seed": SEED, "inputs": pins,
        "cases": cases, "excluded_prior": audit, "preprocessing_policy": policy,
        "prior_corpus": {"path": str(PRIOR / "manifest.json"), "manifest_id": old_manifest["manifest_id"],
                         "case_ids": [c["case_id"] for c in old], "metadata_preserved_exactly": True,
                         "references": "Historical quality-only rubric; no reference values copied into this manifest"},
        "selection_policy": {"new_count": 22, "new_case_ids": [c["case_id"] for c in new],
            "split": "train", "method": "SHA256 canonical JSON [seed, category, image_id], smallest first",
            "vizwiz_quotas": QUOTAS, "standalone_megalith": 14, "new_distortions": 0,
            "source_requirements": "Single frame; Megalith source IDs with no synthetic distortion children",
            "exclusion_keys": list(prior.IDENTITY_KEYS), "near_duplicate_hamming_threshold": 2,
            "near_duplicate_scope": "All historical exposure sources/delivered views and new selected families",
            "pixel_appearance_used_for_selection": False, "provider_scores_used_for_selection": False,
            "photo_domain_filter_applied": False,
            "reference_status": "New judgments pending; packets omit source labels and model scores"},
        "counts": {"images": 72, "preserved_images": 50, "new_images": 22,
                   "families": len({c["family_id"] for c in cases}),
                   "new_components": dict(Counter(c["component"] for c in new)),
                   "strata": dict(Counter(c["stratum"] for c in cases))},
        "packet_files": packets, "training_status": "pending_human_review; every label task mask remains false",
    }
    manifest["manifest_id"] = prior.digest(manifest)
    prior.freeze(HERE / "manifest.json", prior.encoded(manifest))
    prior.freeze(HERE / "extension-selection.json", prior.encoded({"seed": SEED,
        "manifest_id": manifest["manifest_id"], "case_ids": [c["case_id"] for c in new],
        "cases": new, "selected_before_provider_inference": True, "image_appearance_not_inspected_for_selection": True}))
    return verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else verify()
    if args.command == "prepare":
        prior.freeze(HERE / "preparation-validation.json", prior.encoded(result))
    print(json.dumps(result, indent=2))
