"""Freeze two disjoint 50-case cohorts from existing pilot training records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from visual_quality.storage import file_sha256, read_jsonl
from visual_quality.views import load_views, prepare_views

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = EXPERIMENT.parents[1]
REPO = PROJECT.parent
DATA = PROJECT / "data"
SOURCE = DATA / "pilot-10k-v1"
PRIOR = DATA / "label-pilot-12/views"
SEED = 20261018
OPERATORS = ("gaussian_blur", "gaussian_noise", "jpeg_compression")
VIZWIZ_QUOTAS = {"BLR": 3, "BRT": 2, "DRK": 2, "FRM": 3, "OBS": 2, "ROT": 2}
DEFECTS = tuple(sorted((*VIZWIZ_QUOTAS, "OTH")))
IDENTITY_KEYS = ("split_group", "sha256", "decoded_sha256")


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def freeze(path: Path, payload: bytes) -> None:
    """Reuse identical bytes and refuse to replace an existing frozen artifact."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def order(stage: str, category: str, image_id: str) -> str:
    return hashlib.sha256(encoded([SEED, stage, category, image_id])).hexdigest()


def identities(rows: list[dict]) -> set[tuple[str, str]]:
    return {(key, row[key]) for row in rows for key in IDENTITY_KEYS}


def near_other_stage(rows: list[dict], reference: list[dict]) -> bool:
    return any(
        (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= 2
        for row in rows
        for other in reference
    )


def vizwiz_stratum(row: dict) -> str:
    votes = row["source_labels"]["human_flaw_votes"]
    return min(DEFECTS, key=lambda defect: (-votes[defect], defect))


def image_metadata(row: dict) -> dict:
    # Read image headers only. Pixel appearance never determines selection.
    with Image.open(DATA / row["image_path"]) as image:
        return {
            "native_mode": image.mode,
            "frames": getattr(image, "n_frames", 1),
            "icc_profile_present": image.info.get("icc_profile") is not None,
            "alpha_or_transparency_present": "A" in image.getbands() or "transparency" in image.info,
        }


def select_stage(stage: str, rows: list[dict], reference: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in rows}
    families = defaultdict(list)
    for row in rows:
        if row["component"] == "distortions":
            families[row["parent_id"]].append(row)
    occupied = identities(reference)
    selected = []

    def accept(batch: list[dict], role: str, operator: str | None = None) -> bool:
        if occupied & identities(batch) or near_other_stage(batch, reference):
            return False
        if any(image_metadata(row)["frames"] != 1 for row in batch):
            return False
        for row in batch:
            selected.append(
                {
                    "row": row,
                    "role": "variant" if row["component"] == "distortions" else role,
                    "operator": operator,
                    "severity": row.get("transform", {}).get("severity_index"),
                    "vizwiz_stratum": vizwiz_stratum(row) if row["component"] == "vizwiz" else None,
                }
            )
        occupied.update(identities(batch))
        return True

    for operator in OPERATORS:
        count = 0
        for parent_id in sorted(families, key=lambda key: order(stage, operator, key)):
            variants = sorted(families[parent_id], key=lambda row: row["transform"]["severity_index"])
            if {row["transform"]["operator"] for row in variants} != {operator}:
                continue
            if [row["transform"]["severity_index"] for row in variants] != [1, 2, 3]:
                raise ValueError(f"Incomplete severity family: {parent_id}")
            parent = by_id[parent_id]
            if parent["component"] != "megalith" or any(
                row["split_group"] != parent["split_group"]
                or row["parent_source"]["sha256"] != parent["sha256"]
                for row in variants
            ):
                raise ValueError(f"Source parent differs from its variant provenance: {parent_id}")
            batch = [parent, *variants]
            if occupied & identities(batch) or near_other_stage(batch, reference):
                continue
            metadata = image_metadata(parent)
            if metadata["icc_profile_present"] or metadata["alpha_or_transparency_present"]:
                continue
            if accept(batch, "source_parent", operator):
                selected[-4]["parent_metadata"] = metadata
                count += 1
                if count == 2:
                    break
        if count != 2:
            raise ValueError(f"Insufficient eligible source parents for {stage}/{operator}")

    for defect, quota in VIZWIZ_QUOTAS.items():
        pool = [row for row in rows if row["component"] == "vizwiz" and vizwiz_stratum(row) == defect]
        count = 0
        for row in sorted(pool, key=lambda row: order(stage, defect, row["id"])):
            if accept([row], "vizwiz"):
                count += 1
                if count == quota:
                    break
        if count != quota:
            raise ValueError(f"Insufficient VizWiz rows for {stage}/{defect}")

    pool = [row for row in rows if row["component"] == "megalith" and row["id"] not in families]
    count = 0
    for row in sorted(pool, key=lambda row: order(stage, "standalone", row["id"])):
        if accept([row], "standalone_megalith"):
            count += 1
            if count == 12:
                break
    if count != 12 or len(selected) != 50:
        raise ValueError(f"Incomplete stage: {stage}")
    return selected


def stage_record(stage: str, selections: list[dict], inputs: dict) -> dict:
    rows = [case["row"] for case in selections]
    dataset_dir = DATA / f"prompt-calibration-{stage}-50-v1"
    views_dir = DATA / "prompt-calibration-2026-10-18" / stage / "views"
    dataset = {
        "schema_version": 1,
        "name": dataset_dir.name,
        "image_root": "..",
        "manifests": {"train": "train.jsonl"},
        "seed": SEED,
        "derived_from": {
            "dataset": relative(SOURCE / "dataset.json"),
            "dataset_sha256": inputs[relative(SOURCE / "dataset.json")],
            "manifest": relative(SOURCE / "train.jsonl"),
            "manifest_sha256": inputs[relative(SOURCE / "train.jsonl")],
        },
        "label_policy": "Source records unchanged; no quality scores or eligibility labels inferred",
        "purpose": "Frozen prompt calibration cohort; source parents are not verified clean references",
    }
    freeze(dataset_dir / "dataset.json", encoded(dataset))
    train = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()
    freeze(dataset_dir / "train.jsonl", train)
    preparation = prepare_views(dataset_dir, views_dir, limit=50, seed=SEED, max_edge=1536)
    views = {view["image_id"]: view for view in load_views(views_dir)}
    if set(views) != {row["id"] for row in rows}:
        raise ValueError(f"Fixed views do not contain every selected case in stage {stage}")
    cases = []
    for index, selection in enumerate(selections, 1):
        row = selection["row"]
        view = views[row["id"]]
        case = {key: value for key, value in selection.items() if key != "row"}
        case.update(
            case_id=f"{stage.upper()}{index:02d}",
            image_id=row["id"],
            image_path=row["image_path"],
            family_id=row["split_group"],
            **{key: row[key] for key in (*IDENTITY_KEYS, "dhash")},
            view_id=view["view_id"],
            view_sha256=view["view_sha256"],
        )
        if row["component"] == "vizwiz":
            case["human_flaw_votes"] = row["source_labels"]["human_flaw_votes"]
        cases.append(case)
    return {
        "dataset_dir": relative(dataset_dir),
        "views_dir": relative(views_dir),
        "dataset_sha256": file_sha256(dataset_dir / "dataset.json"),
        "train_sha256": file_sha256(dataset_dir / "train.jsonl"),
        "views_sha256": preparation["views_sha256"],
        "counts": {
            "total": len(cases),
            "roles": dict(Counter(case["role"] for case in cases)),
            "source_parents_by_operator": dict(
                Counter(case["operator"] for case in cases if case["role"] == "source_parent")
            ),
            "vizwiz_strata": dict(
                Counter(case["vizwiz_stratum"] for case in cases if case["role"] == "vizwiz")
            ),
        },
        "cases": cases,
    }


def main() -> None:
    input_paths = (
        SOURCE / "dataset.json",
        SOURCE / "train.jsonl",
        PRIOR / "preparation.json",
        PRIOR / "views.jsonl",
        PROJECT / "src/visual_quality/views.py",
        Path(__file__).resolve(),
    )
    inputs = {relative(path): file_sha256(path) for path in input_paths}
    manifest_path = EXPERIMENT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["inputs"] != inputs:
        raise ValueError("Frozen source, prior views, or preparation code changed")
    rows = read_jsonl(SOURCE / "train.jsonl")
    if len({row["id"] for row in rows}) != len(rows) or any(row["split"] != "train" for row in rows):
        raise ValueError("Expected unique source IDs in the training split")
    prior_views = load_views(PRIOR)
    if len(prior_views) != 12:
        raise ValueError("Expected exactly 12 prior views")
    prior_groups = {view["split_group"] for view in prior_views}
    reference = [row for row in rows if row["split_group"] in prior_groups]
    prior_count = len(reference)
    stages = {}
    for stage in ("a", "b"):
        selections = select_stage(stage, rows, reference)
        stages[stage] = stage_record(stage, selections, inputs)
        reference.extend(case["row"] for case in selections)
    if {relative(path): file_sha256(path) for path in input_paths} != inputs:
        raise ValueError("Inputs changed during preparation")
    manifest = {
        "schema_version": "prompt-calibration-cohorts-v1",
        "seed": SEED,
        "inputs": inputs,
        "selection_policy": {
            "split": "train",
            "method": "SHA-256 of JSON [seed, stage, category, image_id]; smallest first",
            "source_parent": "Existing Megalith train row; no ICC profile or alpha/transparency metadata",
            "families_per_operator_per_stage": 2,
            "operators": list(OPERATORS),
            "variant_severity_indices": [1, 2, 3],
            "vizwiz_quotas_per_stage": VIZWIZ_QUOTAS,
            "vizwiz_stratum": "Largest human named-flaw vote count; alphabetical code breaks ties; no score inference",
            "standalone_megalith": "12 rows per stage; exclude all parents of existing distortion families",
            "exact_exclusion_keys": list(IDENTITY_KEYS),
            "prior_exclusion": "All source train rows in any family used by the prior 12 views",
            "near_duplicate_exclusion": "Reject dHash Hamming distance <= 2 against prior-family rows and earlier stages",
            "within_stage": "Family siblings are allowed; distinct selections cannot share any exact exclusion key",
            "pixel_appearance_used_for_selection": False,
            "source_frame_count_required": 1,
            "max_edge": 1536,
        },
        "excluded_prior": {
            "view_ids": [view["view_id"] for view in prior_views],
            "image_ids": [view["image_id"] for view in prior_views],
            "split_groups": sorted(prior_groups),
            "expanded_source_row_count": prior_count,
        },
        "limitations": [
            "Source parents have no human quality labels and are not verified clean references.",
            "Existing variants were generated before fixed-view preparation; source-parent and variant paths differ.",
            "dHash is a conservative duplicate heuristic, not proof that all visual or semantic overlap is absent.",
            "No training row has OTH as its dominant human flaw; the six observed strata are used.",
        ],
        "stages": stages,
    }
    freeze(manifest_path, encoded(manifest))
    print(
        json.dumps(
            {
                "manifest": relative(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "stages": {
                    stage: {key: value for key, value in record.items() if key != "cases"}
                    for stage, record in stages.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
