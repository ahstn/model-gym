"""Freeze 200 fresh image views and blinded review packets for the RunInfra trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from visual_quality import views
from visual_quality.storage import file_sha256, read_jsonl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
OUTPUT = DATA / HERE.name
MANIFEST = HERE / "manifest.json"
SOURCE = DATA / "pilot-10k-v1"
PRIOR = DATA / "label-pilot-12/views/views.jsonl"
OLD = HERE.parent / "prompt-calibration-2026-10-18/manifest.json"
SEED = 20261026
MAX_EDGE = 1536
MAX_PNG_BYTES = 2_650_000
OPERATORS = ("gaussian_blur", "gaussian_noise", "jpeg_compression")
VIZWIZ_QUOTAS = {"BLR": 12, "BRT": 8, "DRK": 8, "FRM": 12, "OBS": 8, "ROT": 8}
IDENTITY_KEYS = ("split_group", "sha256", "decoded_sha256")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def freeze(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)


def order(category: str, identity: str) -> str:
    return digest([SEED, category, identity])


def identities(rows: list[dict]) -> set[tuple[str, str]]:
    return {(key, row[key]) for row in rows for key in IDENTITY_KEYS}


def overlaps(batch: list[dict], previous: list[dict]) -> bool:
    if identities(batch) & identities(previous):
        return True
    return any(
        (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= 2
        for row in batch
        for other in previous
    )


def vizwiz_stratum(row: dict) -> str:
    votes = row["source_labels"]["human_flaw_votes"]
    return min((*VIZWIZ_QUOTAS, "OTH"), key=lambda defect: (-votes[defect], defect))


def metadata(row: dict) -> dict:
    with Image.open(DATA / row["image_path"]) as image:
        return {
            "frames": getattr(image, "n_frames", 1),
            "icc": image.info.get("icc_profile") is not None,
            "alpha": "A" in image.getbands() or "transparency" in image.info,
        }


def historical_exclusions(rows: list[dict]) -> tuple[list[dict], dict]:
    prior = read_jsonl(PRIOR)
    old = json.loads(OLD.read_text())
    groups = {row["split_group"] for row in prior}
    groups.update(case["family_id"] for stage in old["stages"].values() for case in stage["cases"])
    excluded = [row for row in rows if row["split_group"] in groups]
    return excluded, {
        "source": "Original 12-image pilot and both 50-image prompt-calibration cohorts",
        "family_ids": sorted(groups),
        "expanded_source_row_count": len(excluded),
        "selected_prior_exposure": False,
        "holdout_policy": "Training rows only; development and final-test splits remain untouched",
    }


def select(rows: list[dict], historical: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or any(row["split"] != "train" for row in rows):
        raise ValueError("Expected unique training records")
    families = defaultdict(list)
    for row in rows:
        if row["component"] == "distortions":
            families[row["parent_id"]].append(row)
    previous = list(historical)
    selected = []

    def accept(batch: list[dict], role: str, operator: str | None = None) -> bool:
        if overlaps(batch, previous) or any(metadata(row)["frames"] != 1 for row in batch):
            return False
        for row in batch:
            selected.append(
                {
                    "row": row,
                    "role": "variant" if row["component"] == "distortions" else role,
                    "operator": operator,
                    "severity": row.get("transform", {}).get("severity_index"),
                    "vizwiz_stratum": vizwiz_stratum(row) if row["component"] == "vizwiz" else None,
                    "stratum": "controlled" if operator else role,
                }
            )
        previous.extend(batch)
        return True

    for operator in OPERATORS:
        count = 0
        for parent_id in sorted(families, key=lambda key: order(operator, key)):
            variants = sorted(families[parent_id], key=lambda row: row["transform"]["severity_index"])
            if {row["transform"]["operator"] for row in variants} != {operator}:
                continue
            parent = by_id[parent_id]
            if [row["transform"]["severity_index"] for row in variants] != [1, 2, 3]:
                raise ValueError("Incomplete source severity family")
            if parent["component"] != "megalith" or any(
                row["split_group"] != parent["split_group"]
                or row["parent_source"]["sha256"] != parent["sha256"]
                for row in variants
            ):
                raise ValueError("Variant parent provenance differs")
            parent_metadata = metadata(parent)
            if parent_metadata["icc"] or parent_metadata["alpha"]:
                continue
            if accept([parent, *variants], "source_parent", operator):
                count += 1
                if count == 8:
                    break
        if count != 8:
            raise ValueError(f"Insufficient source families: {operator}")

    for defect, quota in VIZWIZ_QUOTAS.items():
        pool = [row for row in rows if row["component"] == "vizwiz" and vizwiz_stratum(row) == defect]
        count = 0
        for row in sorted(pool, key=lambda row: order(defect, row["id"])):
            if accept([row], "vizwiz"):
                count += 1
                if count == quota:
                    break
        if count != quota:
            raise ValueError(f"Insufficient VizWiz stratum: {defect}")

    pool = [row for row in rows if row["component"] == "megalith" and row["id"] not in families]
    count = 0
    for row in sorted(pool, key=lambda row: order("standalone", row["id"])):
        if accept([row], "standalone_megalith"):
            count += 1
            if count == 48:
                break
    if count != 48 or len(selected) != 200:
        raise ValueError("Incomplete 200-image cohort")
    return selected


def render_family(payloads: list[bytes], *, max_edge: int = MAX_EDGE, max_bytes: int = MAX_PNG_BYTES):
    """Use one size cap for every family member; rerender from original source bytes."""
    if not payloads or max_edge < 64 or max_bytes <= 0:
        raise ValueError("Invalid family rendering request")
    edge = max_edge
    while True:
        rendered = [views._render(payload, edge) for payload in payloads]
        if all(len(data) <= max_bytes for data, _ in rendered):
            return edge, rendered
        if edge == 64:
            raise ValueError("Even minimum-size family exceeds the PNG byte budget")
        edge = max(64, edge * 9 // 10)


def preprocessing_policy() -> dict:
    policy = views._policy(MAX_EDGE)
    policy["version"] = "runinfra-family-bounded-full-image-srgb-png-v1"
    policy.pop("selection")
    policy["byte_budget"] = {
        "max_png_bytes": MAX_PNG_BYTES,
        "max_request_bytes": 3_670_016,
        "method": "Start edge1536; floor(edge*0.9) until every family PNG fits; minimum64",
        "controlled_family": "Use identical edge cap for parent and all three variants",
        "singletons": "Apply the same rule independently",
        "render_source": "Always original source bytes, never a previously resized view",
        "request_verification": "Runner must verify exact serialized request bytes before any inference",
    }
    return policy


def build_views(selections: list[dict], policy: dict) -> list[dict]:
    families = defaultdict(list)
    for index, selection in enumerate(selections, 1):
        families[selection["row"]["split_group"]].append((index, selection))
    cases = []
    policy_hash = digest(policy)
    for family_id, members in families.items():
        payloads = [(DATA / selection["row"]["image_path"]).read_bytes() for _, selection in members]
        for payload, (_, selection) in zip(payloads, members, strict=True):
            if hashlib.sha256(payload).hexdigest() != selection["row"]["sha256"]:
                raise ValueError("Source image hash differs")
        edge, rendered = render_family(payloads)
        for (index, selection), (payload, preprocessing) in zip(members, rendered, strict=True):
            row = selection["row"]
            view_hash = hashlib.sha256(payload).hexdigest()
            path = OUTPUT / "views" / f"{view_hash}.png"
            freeze(path, payload)
            case = {key: value for key, value in selection.items() if key != "row"}
            case.update(
                case_id=f"T{index:03d}",
                image_id=row["id"],
                family_id=family_id,
                source_path=str(DATA / row["image_path"]),
                source_sha256=row["sha256"],
                decoded_sha256=row["decoded_sha256"],
                dhash=row["dhash"],
                split="train",
                split_group=row["split_group"],
                view_id=digest([row["id"], row["sha256"], policy_hash, edge, view_hash]),
                view_sha256=view_hash,
                image_path=str(path),
                mime_type="image/png",
                width=preprocessing["output_dimensions"][0],
                height=preprocessing["output_dimensions"][1],
                delivered_bytes=len(payload),
                target_edge=edge,
                preprocessing=preprocessing,
                photo_domain_eligible=row.get("photo_domain_eligible"),
                component=row["component"],
            )
            cases.append(case)
    return sorted(cases, key=lambda case: case["case_id"])


def packets(cases: list[dict]) -> dict:
    shuffled = sorted(cases, key=lambda case: order("blind-review-packet", case["case_id"]))
    result = {}
    for index, start in enumerate(range(0, len(shuffled), 67), 1):
        rows = [
            {
                "case_id": c["case_id"],
                "path": c["image_path"],
                "view_sha256": c["view_sha256"],
                "width": c["width"],
                "height": c["height"],
            }
            for c in shuffled[start : start + 67]
        ]
        path = OUTPUT / "review-packets" / f"packet-{index}.json"
        freeze(path, encoded(rows))
        result[str(path)] = file_sha256(path)
    return result


def input_pins() -> dict:
    paths = [
        SOURCE / "dataset.json",
        SOURCE / "train.jsonl",
        PRIOR,
        OLD,
        Path(__file__),
        Path(views.__file__),
    ]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if (
        digest({key: value for key, value in manifest.items() if key != "manifest_id"})
        != manifest["manifest_id"]
    ):
        raise ValueError("Frozen manifest identity differs")
    if manifest["inputs"] != input_pins():
        raise ValueError("Frozen preparation inputs differ")
    for path, expected in manifest["packet_files"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("Blinded review packet differs")
    cases = manifest["cases"]
    if len(cases) != 200 or len({case["image_id"] for case in cases}) != 200:
        raise ValueError("Expected exactly 200 unique images")
    if (
        len({case["source_sha256"] for case in cases}) != 200
        or len({case["view_sha256"] for case in cases}) != 200
    ):
        raise ValueError("Duplicate source or delivered image bytes")
    family_edges = defaultdict(set)
    policy_hash = digest(manifest["preprocessing_policy"])
    for case in cases:
        family_edges[case["family_id"]].add(case["target_edge"])
        source, view = Path(case["source_path"]), Path(case["image_path"])
        if file_sha256(source) != case["source_sha256"] or file_sha256(view) != case["view_sha256"]:
            raise ValueError("Source or delivered image hash differs")
        if view.stat().st_size != case["delivered_bytes"] or case["delivered_bytes"] > MAX_PNG_BYTES:
            raise ValueError("Delivered image exceeds frozen byte budget")
        if (
            digest(
                [
                    case["image_id"],
                    case["source_sha256"],
                    policy_hash,
                    case["target_edge"],
                    case["view_sha256"],
                ]
            )
            != case["view_id"]
        ):
            raise ValueError("View identity differs")
        with Image.open(view) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or list(image.size) != [case["width"], case["height"]]
            ):
                raise ValueError("Delivered image metadata differs")
        if max(case["width"], case["height"]) > case["target_edge"]:
            raise ValueError("Delivered image exceeds its family edge cap")
    if len(family_edges) != 128 or any(len(edges) != 1 for edges in family_edges.values()):
        raise ValueError("Family count or common edge policy differs")
    return {"manifest_id": manifest["manifest_id"], "verified": True, "images": 200, "families": 128}


def prepare() -> dict:
    if MANIFEST.exists():
        return verify()
    pins = input_pins()
    rows = read_jsonl(SOURCE / "train.jsonl")
    historical, exclusion_record = historical_exclusions(rows)
    selections = select(rows, historical)
    policy = preprocessing_policy()
    cases = build_views(selections, policy)
    manifest = {
        "schema_version": "runinfra-string-trial-cohort-v1",
        "seed": SEED,
        "inputs": pins,
        "cases": cases,
        "excluded_prior": exclusion_record,
        "selection_policy": {
            "split": "train",
            "method": "SHA256 canonical JSON [seed, category, image_id], smallest first",
            "families_per_operator": 8,
            "operators": list(OPERATORS),
            "severity_indices": [1, 2, 3],
            "vizwiz_quotas": VIZWIZ_QUOTAS,
            "standalone_megalith": 48,
            "exclusion_keys": list(IDENTITY_KEYS),
            "near_duplicate_hamming_threshold": 2,
            "near_duplicate_scope": "All historical exclusions and all already selected new families; siblings allowed",
            "source_parent": "Megalith, single frame, no ICC or alpha; not presumed pristine",
            "pixel_appearance_used_for_selection": False,
        },
        "preprocessing_policy": policy,
        "counts": {
            "images": len(cases),
            "families": len({c["family_id"] for c in cases}),
            "strata": dict(Counter(c["stratum"] for c in cases)),
            "roles": dict(Counter(c["role"] for c in cases)),
            "target_edges": dict(Counter(c["target_edge"] for c in cases)),
        },
        "packet_files": packets(cases),
        "references": "Fresh blinded assistant judgments on these exact delivered views; not human ground truth",
        "training_status": "pending_human_review; every label task mask must remain false",
    }
    manifest["manifest_id"] = digest(manifest)
    if input_pins() != pins:
        raise ValueError("Preparation inputs changed during freeze")
    freeze(MANIFEST, encoded(manifest))
    return verify()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
