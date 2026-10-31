"""Freeze 1,999 unexposed photo-pool images, one known poster, and a 72-image audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from visual_quality.storage import file_sha256, read_jsonl

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "prompt-ablation-2026-10-30"
PHOTO_SCOPE = HERE.parent / "photo-scope-trial-2026-10-31"
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
SOURCE = DATA / "pilot-10k-v1"
OUTPUT = DATA / HERE.name
SEED = 202610312000
FAMILIES_PER_OPERATOR = 40
VIZWIZ_QUOTAS = {"BLR": 40, "BRT": 40, "DRK": 40, "FRM": 40, "OBS": 40, "ROT": 40}
STANDALONE_COUNT = 1279

spec = importlib.util.spec_from_file_location("frozen_photo_preparation", PRIOR / "prepare.py")
prior = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prior)
prior.HERE, prior.OUTPUT, prior.SEED = HERE, OUTPUT, SEED


class ExposureIndex:
    """The existing identity/dHash overlap rule, indexed without repeated set rebuilding."""

    def __init__(self, rows: list[dict]):
        self.keys = set()
        self.hashes = set()
        self.extend(rows)

    def extend(self, rows: list[dict]) -> None:
        self.keys.update(prior.identities(rows))
        self.hashes.update(int(row["dhash"], 16) for row in rows if row.get("dhash"))

    def overlaps(self, rows: list[dict]) -> bool:
        return bool(prior.identities(rows) & self.keys) or any(
            (int(row["dhash"], 16) ^ old).bit_count() <= 2
            for row in rows
            if row.get("dhash")
            for old in self.hashes
        )


def select(rows: list[dict], historical: list[dict]) -> tuple[list[dict], dict]:
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or any(row["split"] != "train" for row in rows):
        raise ValueError("Expected unique training source records")
    families = defaultdict(list)
    for row in rows:
        if row["component"] == "distortions":
            families[row["parent_id"]].append(row)
    index = ExposureIndex(historical)
    historical_index = ExposureIndex(historical)
    fresh = [row for row in rows if not historical_index.overlaps([row])]
    capacity = {
        "source_rows": len(rows),
        "source_components": dict(Counter(row["component"] for row in rows)),
        "unexposed_components_before_interselection_deduplication": dict(
            Counter(row["component"] for row in fresh)
        ),
        "unexposed_standalone_megalith": sum(
            row["component"] == "megalith" and row["id"] not in families for row in fresh
        ),
        "unexposed_vizwiz_strata": dict(
            Counter(prior.vizwiz_stratum(row) for row in fresh if row["component"] == "vizwiz")
        ),
    }
    selections, rejections = [], Counter()

    def accept(batch: list[dict], role: str, operator: str | None = None) -> bool:
        if index.overlaps(batch):
            rejections["historical_or_selected_identity_or_near_duplicate"] += 1
            return False
        if len({row["sha256"] for row in batch}) != len(batch) or len(
            {row["decoded_sha256"] for row in batch}
        ) != len(batch):
            rejections["identical_family_members"] += 1
            return False
        if any(prior.metadata(row)["frames"] != 1 for row in batch):
            rejections["multiple_frames"] += 1
            return False
        for row in batch:
            selections.append(
                {
                    "row": row,
                    "role": "variant" if row["component"] == "distortions" else role,
                    "operator": operator,
                    "severity": row.get("transform", {}).get("severity_index"),
                    "vizwiz_stratum": prior.vizwiz_stratum(row) if row["component"] == "vizwiz" else None,
                    "stratum": "controlled" if operator else role,
                    "issue_tags": prior.issue_tags(row) or [role],
                    "known_anchor": False,
                }
            )
        index.extend(batch)
        return True

    for operator in prior.OPERATORS:
        count = 0
        for parent_id in sorted(families, key=lambda value: prior.order(operator, value)):
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
            metadata = prior.metadata(parent)
            if metadata["icc"] or metadata["alpha"]:
                rejections["controlled_parent_icc_or_alpha"] += 1
                continue
            if accept([parent, *variants], "source_parent", operator):
                count += 1
                if count == FAMILIES_PER_OPERATOR:
                    break
        if count != FAMILIES_PER_OPERATOR:
            raise ValueError(f"Insufficient fresh controlled families: {operator}")
    for defect, quota in VIZWIZ_QUOTAS.items():
        pool = [row for row in rows if row["component"] == "vizwiz" and prior.vizwiz_stratum(row) == defect]
        count = 0
        for row in sorted(pool, key=lambda value: prior.order(defect, value["id"])):
            if accept([row], "vizwiz"):
                count += 1
                if count == quota:
                    break
        if count != quota:
            raise ValueError(f"Insufficient fresh VizWiz stratum: {defect}")
    pool = [row for row in rows if row["component"] == "megalith" and row["id"] not in families]
    count = 0
    for row in sorted(pool, key=lambda value: prior.order("standalone", value["id"])):
        if accept([row], "standalone_megalith"):
            count += 1
            if count == STANDALONE_COUNT:
                break
    if len(selections) != 1999 or count != STANDALONE_COUNT:
        raise ValueError("Incomplete fresh corpus")
    for number, selection in enumerate(selections, 1):
        selection["case_id"] = f"T{number:04d}"
    return selections, {"capacity": capacity, "selection_rejections": dict(rejections)}


def choose_audit(selections: list[dict]) -> list[str]:
    selected, families = [], set()

    def take(pool: list[dict], category: str, count: int) -> None:
        accepted = 0
        for item in sorted(pool, key=lambda value: prior.order(category, value["row"]["id"])):
            family = item["row"]["split_group"]
            if family in families:
                continue
            selected.append(item["case_id"])
            families.add(family)
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError(f"Incomplete independent-family audit stratum: {category}")

    take([item for item in selections if item["role"] == "standalone_megalith"], "audit-standalone", 41)
    for defect in VIZWIZ_QUOTAS:
        take([item for item in selections if item["vizwiz_stratum"] == defect], f"audit-vizwiz-{defect}", 3)
    for operator in prior.OPERATORS:
        for severity in (None, 1, 2, 3):
            take(
                [
                    item
                    for item in selections
                    if item["operator"] == operator and item["severity"] == severity
                ],
                f"audit-controlled-{operator}-{severity}",
                1,
            )
    selected.append("T2000")
    if len(selected) != 72 or len(set(selected)) != 72:
        raise ValueError("Audit requires exactly 72 unique cases")
    return sorted(selected)


def resolve_source_formats(
    selections: list[dict], historical: list[dict], rows: list[dict], audit_ids: list[str]
) -> list[dict]:
    """Replace unrenderable non-audit sources without relaxing the colour policy."""

    def rejection(item: dict) -> str | None:
        row = item["row"]
        if prior.metadata(row)["icc"]:
            try:
                prior.views._render((DATA / row["image_path"]).read_bytes(), 64)
            except ValueError as error:
                return str(error)
        return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        rejected = [
            (item, reason)
            for item, reason in zip(selections, executor.map(rejection, selections), strict=True)
            if reason
        ]
    rejected_ids = {item["row"]["id"] for item, _ in rejected}
    index = ExposureIndex(
        [*historical, *(item["row"] for item in selections if item["row"]["id"] not in rejected_ids)]
    )
    parents = {row["parent_id"] for row in rows if row["component"] == "distortions"}
    pool = sorted(
        (row for row in rows if row["component"] == "megalith" and row["id"] not in parents),
        key=lambda row: prior.order("standalone", row["id"]),
    )
    replacements, resolved = [], {item["case_id"]: item for item in selections}
    for item, reason in rejected:
        if item["case_id"] in audit_ids or item["role"] != "standalone_megalith":
            raise ValueError("A frozen audit case or non-standalone source cannot be replaced")
        for candidate in pool:
            if (
                candidate["id"] in rejected_ids
                or index.overlaps([candidate])
                or prior.metadata(candidate)["frames"] != 1
            ):
                continue
            if rejection({"row": candidate}):
                continue
            replacement = {**item, "row": candidate, "replacement_for_image_id": item["row"]["id"]}
            resolved[item["case_id"]] = replacement
            index.extend([candidate])
            replacements.append(
                {
                    "case_id": item["case_id"],
                    "rejected_image_id": item["row"]["id"],
                    "rejected_source_sha256": item["row"]["sha256"],
                    "reason": reason,
                    "replacement_image_id": candidate["id"],
                    "replacement_source_sha256": candidate["sha256"],
                }
            )
            break
        else:
            raise ValueError("No valid deterministic source-format replacement")
    prior.freeze(
        HERE / "source-format-resolution.json",
        prior.encoded(
            {
                "initial_selection_sha256": file_sha256(HERE / "selection.json"),
                "selected_before_provider_inference": True,
                "frozen_72_audit_cases_unchanged": True,
                "method": "Reject inputs incompatible with unchanged embedded-ICC-to-sRGB conversion; next unused eligible standalone in original seeded hash order",
                "appearance_or_scores_used": False,
                "replacements": replacements,
            }
        ),
    )
    return [resolved[item["case_id"]] for item in selections]


def render_group(members: list[dict], policy: dict) -> list[dict]:
    cases = prior.build_views(members, policy)
    by_image = {item["row"]["id"]: item for item in members}
    for case in cases:
        item = by_image[case["image_id"]]
        row = item["row"]
        case["case_id"] = item["case_id"]
        case["source_provenance"] = {
            key: row.get(key)
            for key in (
                "source",
                "source_id",
                "source_revision",
                "source_url",
                "source_origin",
                "license",
                "parent_id",
                "transform",
            )
        }
    return cases


def anchor_case() -> dict:
    old = json.loads((PHOTO_SCOPE / "manifest.json").read_text())
    case = next(case for case in old["cases"] if case["case_id"] == "P048").copy()
    case.update(
        case_id="T2000",
        known_anchor=True,
        prior_case_id="P048",
        stratum="known_poster_anchor",
        role="known_poster_anchor",
    )
    if file_sha256(Path(case["image_path"])) != case["view_sha256"]:
        raise ValueError("Known poster anchor changed")
    return case


def write_packets(cases: list[dict], audit_ids: list[str]) -> dict:
    by_id = {case["case_id"]: case for case in cases}
    subset = [by_id[case_id] for case_id in audit_ids]
    if len({case["family_id"] for case in subset}) != 72:
        raise ValueError("Audit images must have 72 distinct families")
    shuffled = sorted(subset, key=lambda case: prior.order("blind-review-packet", case["case_id"]))
    files = {}
    for index, start in enumerate((0, 36), 1):
        packet = [
            {
                "case_id": case["case_id"],
                "path": case["image_path"],
                "view_sha256": case["view_sha256"],
                "width": case["width"],
                "height": case["height"],
            }
            for case in shuffled[start : start + 36]
        ]
        path = OUTPUT / "review-packets" / f"packet-{index}.json"
        prior.freeze(path, prior.encoded(packet))
        files[str(path)] = file_sha256(path)
    selection = {
        "seed": SEED,
        "case_ids": audit_ids,
        "images": 72,
        "families": 72,
        "selected_before_provider_inference": True,
        "provider_scores_used_for_selection": False,
        "pixel_appearance_used_for_selection": False,
        "method": "SHA256(seed, audit stratum, source image ID), smallest first; distinct families",
        "quotas": {"standalone_megalith": 41, "vizwiz": 18, "controlled": 12, "known_poster_anchor": 1},
        "controlled_quota": "One parent and each of three severities per operator, drawn from four distinct families",
        "vizwiz_quota": "Three cases per dominant source-metadata flaw",
        "known_anchor": "T2000 is prior P048, intentionally retained as a non-photographic policy regression check",
        "cases": [
            {
                key: case.get(key)
                for key in (
                    "case_id",
                    "image_id",
                    "family_id",
                    "image_path",
                    "view_sha256",
                    "width",
                    "height",
                    "stratum",
                    "operator",
                    "severity",
                    "vizwiz_stratum",
                    "known_anchor",
                    "prior_case_id",
                )
            }
            for case in subset
        ],
        "packet_files": files,
    }
    prior.freeze(HERE / "collage-selection.json", prior.encoded(selection))
    return files


def render(selections: list[dict], policy: dict, audit_ids: list[str]) -> tuple[list[dict], dict]:
    families = defaultdict(list)
    for item in selections:
        families[item["row"]["split_group"]].append(item)
    audit_set = set(audit_ids)
    audit_groups, remaining = [], []
    for members in families.values():
        (audit_groups if any(item["case_id"] in audit_set for item in members) else remaining).append(members)
    cases = [anchor_case()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(render_group, group, policy) for group in audit_groups]):
            cases.extend(future.result())
        packets = write_packets(cases, audit_ids)
        print(json.dumps({"status": "72 anonymous review cases frozen", "packet_files": packets}), flush=True)
        previous_progress = len(cases) // 100
        for future in as_completed([executor.submit(render_group, group, policy) for group in remaining]):
            cases.extend(future.result())
            if len(cases) // 100 != previous_progress:
                previous_progress = len(cases) // 100
                print(json.dumps({"rendered_images": len(cases), "total": 2000}), flush=True)
    return sorted(cases, key=lambda case: case["case_id"]), packets


def verify() -> dict:
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["manifest_id"] != prior.digest(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    ):
        raise ValueError("Frozen manifest identity differs")
    for filename, expected in {**manifest["inputs"], **manifest["packet_files"]}.items():
        if file_sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    cases = manifest["cases"]
    if [case["case_id"] for case in cases] != [f"T{index:04d}" for index in range(1, 2001)]:
        raise ValueError("Expected ordered IDs T0001 through T2000")
    for key in ("image_id", "source_sha256", "view_sha256"):
        if len({case[key] for case in cases}) != 2000:
            raise ValueError(f"Duplicate {key}")
    edges = defaultdict(set)
    for case in cases:
        for path, key in ((case["source_path"], "source_sha256"), (case["image_path"], "view_sha256")):
            if file_sha256(Path(path)) != case[key]:
                raise ValueError(f"Image bytes changed: {case['case_id']}")
        if (
            Path(case["image_path"]).stat().st_size != case["delivered_bytes"]
            or case["delivered_bytes"] > prior.MAX_PNG_BYTES
        ):
            raise ValueError("Image exceeds byte policy")
        with Image.open(case["image_path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (case["width"], case["height"]):
                raise ValueError("Delivered image metadata differs")
        if max(case["width"], case["height"]) > case["target_edge"] or case["target_edge"] > prior.MAX_EDGE:
            raise ValueError("Image exceeds edge policy")
        if any(case["task_masks"].values()) or case["split"] != "train":
            raise ValueError("Experimental labels must remain disabled for training")
        edges[case["family_id"]].add(case["target_edge"])
    if any(len(values) != 1 for values in edges.values()) or len(edges) != 1640:
        raise ValueError("Family edge policy or count differs")
    if Counter(case["stratum"] for case in cases) != {
        "controlled": 480,
        "vizwiz": 240,
        "standalone_megalith": 1279,
        "known_poster_anchor": 1,
    }:
        raise ValueError("Corpus composition differs")
    if cases[-1] != anchor_case():
        raise ValueError("Known anchor provenance or bytes differ")
    selection = json.loads((HERE / "collage-selection.json").read_text())
    subset = [case for case in cases if case["case_id"] in selection["case_ids"]]
    if len(subset) != 72 or len({case["family_id"] for case in subset}) != 72:
        raise ValueError("Audit sample is incomplete or has related families")
    return {
        "verified": True,
        "manifest_id": manifest["manifest_id"],
        "images": 2000,
        "fresh_images": 1999,
        "known_anchors": 1,
        "families": 1640,
        "audit_images": 72,
        "audit_families": 72,
        "max_delivered_bytes": max(case["delivered_bytes"] for case in cases),
        "strata": dict(Counter(case["stratum"] for case in cases)),
        "training_masks_enabled": 0,
    }


def prepare() -> dict:
    if (HERE / "manifest.json").exists():
        return verify()
    rows = read_jsonl(SOURCE / "train.jsonl")
    historical, audit = prior.historical_exclusions(rows)
    selections, selection_audit = select(rows, historical)
    audit_ids = choose_audit(selections)
    prior.freeze(
        HERE / "selection.json",
        prior.encoded(
            {
                "seed": SEED,
                "selected_before_provider_inference": True,
                **selection_audit,
                "fresh_cases": [
                    {
                        key: item[key]
                        for key in ("case_id", "role", "operator", "severity", "vizwiz_stratum", "stratum")
                    }
                    | {
                        "image_id": item["row"]["id"],
                        "source_sha256": item["row"]["sha256"],
                        "family_id": item["row"]["split_group"],
                    }
                    for item in selections
                ],
                "known_anchor": {
                    "case_id": "T2000",
                    "prior_case_id": "P048",
                    "prior_manifest": str(PHOTO_SCOPE / "manifest.json"),
                },
                "audit_case_ids": audit_ids,
            }
        ),
    )
    selections = resolve_source_formats(selections, historical, rows, audit_ids)
    policy = prior.preprocessing_policy()
    cases, packets = render(selections, policy, audit_ids)
    pins = {
        str(path): file_sha256(path)
        for path in (
            Path(__file__).resolve(),
            PRIOR / "prepare.py",
            PHOTO_SCOPE / "manifest.json",
            SOURCE / "dataset.json",
            SOURCE / "train.jsonl",
            Path(prior.views.__file__),
            HERE / "selection.json",
            HERE / "source-format-resolution.json",
            HERE / "collage-selection.json",
        )
    }
    pins.update(audit["ledger_files"])
    manifest = {
        "schema_version": "dual-high-2000-photo-cohort-v1",
        "seed": SEED,
        "inputs": pins,
        "cases": cases,
        "excluded_prior": {
            **audit,
            "selected_prior_exposure": True,
            "exception": "Only T2000, prior P048 poster, intentionally retained; all other 1,999 cases exclude known exposure identities/families/near duplicates",
        },
        "selection_policy": {
            "split": "train",
            "method": "SHA256 canonical JSON [seed, category, image_id], smallest first",
            "families_per_operator": FAMILIES_PER_OPERATOR,
            "operators": list(prior.OPERATORS),
            "severity_indices": [1, 2, 3],
            "vizwiz_quotas": VIZWIZ_QUOTAS,
            "standalone_megalith": STANDALONE_COUNT,
            "known_poster_anchors": 1,
            "exclusion_keys": list(prior.IDENTITY_KEYS),
            "near_duplicate_hamming_threshold": 2,
            "near_duplicate_scope": "Historical exposure sources and delivered views, and selected new families; controlled siblings allowed",
            "source_parent": "Megalith, single frame, no ICC or alpha; not presumed pristine",
            "pixel_appearance_used_for_selection": False,
            "provider_scores_used_for_selection": False,
            "photo_domain_filter_applied": False,
            "source_model_aesthetic_scores_used": False,
            "issue_tags": "Frozen source metadata only, never shown to reviewers",
            "known_anchor": "One deliberate policy regression case; exclude it from fresh-only estimates",
            "source_format_replacements": "source-format-resolution.json records any pre-inference replacements of unrenderable non-audit sources; frozen 72 cases remain unchanged",
        },
        "preprocessing_policy": policy,
        "counts": {
            "images": 2000,
            "fresh_images": 1999,
            "known_anchors": 1,
            "families": len({case["family_id"] for case in cases}),
            "strata": dict(Counter(case["stratum"] for case in cases)),
            "roles": dict(Counter(case["role"] for case in cases)),
            "target_edges": dict(Counter(case["target_edge"] for case in cases)),
        },
        "packet_files": packets,
        "references": "Fresh blinded assistant judgments requested only for the frozen 72-image sample; not human ground truth",
        "training_status": "pending_human_review; every experimental label task mask remains false",
    }
    manifest["manifest_id"] = prior.digest(manifest)
    prior.freeze(HERE / "manifest.json", prior.encoded(manifest))
    result = verify()
    prior.freeze(HERE / "preparation-validation.json", prior.encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"), nargs="?", default="prepare")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else verify(), indent=2))
